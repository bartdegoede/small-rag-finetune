# Fine-tuning small LMs for ask-my-blog: dataset and runs (design)

Date: 2026-10-15. Status: approved design, before the pilot.

## Why

The blog's /ask/ page eval (blog repo,
`docs/superpowers/specs/2026-10-12-ask-my-blog-results.md`) found that browser-sized
models can't do grounded 2-sentence answers: faithful answers out of 30 were
0-3 for the ≤350MB models, 15 for Qwen3-0.6B, 28 for Gemma 4 E2B (3GB).
The question for the next post: can fine-tuning close that gap on a Mac?

## What the smoke test established

One epoch of LoRA on 900 examples reshaped from neural-bridge/rag-dataset-12000
(free data, original targets), judged on the blog's 45 questions with the same
Opus judge:

| model | faithful /30 | refused or non-answer /30 | invented /30 | negatives invented /15 |
|---|---|---|---|---|
| SmolLM2-360M base (official export) | 3 | 3 | 24 | 15 |
| SmolLM2-360M base (our export) | 2 | 4 | 24 | 14 |
| SmolLM2-360M smoke | 5 | 14 | 11 | 8 |
| Qwen3-0.6B base (official export) | 15 | 0 | 15 | 5 |
| Qwen3-0.6B smoke | 7 | 17 | 6 | 6 |

- The pipeline works end to end: MLX LoRA → fuse → ONNX q4f16 → transformers.js.
  Our export of the base model scores like the official one.
- Training time per 900 examples: 62 min on the M1 (16GB), 18 min (SmolLM2-360M)
  and 27 min (Qwen3-0.6B) on the M4 (24GB). Qwen3 at batch 4 without gradient
  checkpointing runs out of GPU memory; batch 2 × accumulation 2 with
  checkpointing peaks at about 2GB.
- Free web-text data mostly teaches refusal: invented answers turn into
  "The blog does not seem to cover this." rather than correct ones, and Qwen3
  loses half its correct answers. The passages (short web snippets, one obvious
  answer sentence) look nothing like the blog's long technical chunks.

## Corpus

Openly licensed technical writing that reads like the blog:

| source | licence | form |
|---|---|---|
| Go blog (`golang/website`, `_content/blog`) | CC-BY 4.0 | markdown |
| Rust blog (`rust-lang/blog.rust-lang.org`) | MIT / Apache-2.0 | markdown |
| Rust RFCs (`rust-lang/rfcs`) | MIT / Apache-2.0 | markdown |
| Python PEPs (`python/peps`) | public domain / CC0-1.0 | reStructuredText |

Sources are pinned to a commit. Each document is converted to prose with
section headings (markdown through `sss_eval`; PEPs through a small RST
section splitter). The blog itself is not used, so its eval stays clean.

Documents are split per source into train / valid / test (90 / 5 / 5) before
anything else, so no document appears in two splits.

## Passages and retrieval

Exactly what the browser does, per source (one source = one site):

- chunk with `sss_eval.chunk_post` (the blog's chunker, same size and overlap),
- embed with `minishlab/potion-retrieval-32M`,
- retrieve the top 4 by cosine,
- render with `prompt.build_messages` (byte-identical to `prompt.js`, guarded by
  the parity test), passage label `[i] Title — Heading`.

## Examples (about 5,000)

**Answerable (~80%).** Claude reads one gold chunk and writes questions it
answers, in a mix of styles: fact lookup, how/why, "which X did they use", and
phrasings like the blog eval's questions. Each question goes through retrieval.
If the gold chunk isn't in the top 4 it replaces the 4th passage and the list
is re-sorted by score, so the answer is not always passage [1].

**Refusals (~20%)**, half of each:

- *removed*: an answerable question with every chunk of its document excluded
  from retrieval, so the 4 passages are same-site look-alikes;
- *near-miss*: questions Claude writes about something the gold chunk touches
  on but does not answer, retrieved normally.

**Targets (teacher distillation).** The teacher model gets the exact
system + user messages the small model will see and writes the answer: at
most 2 sentences, plain declarative prose, no "according to passage [2]", and
exactly `The blog does not seem to cover this.` when the passages don't answer.
The label for each example is the teacher's answer, so a refusal comes from
the teacher reading the passages, not from how the example was built.
Construction kind is kept as metadata. If the teacher refuses an "answerable"
example or answers a "removed" one, the example is kept with the teacher's
answer and counted, which tells us how often our construction is wrong.

**Models.** Questions: Sonnet 5.5. Teacher answers: Opus 5.5, or Sonnet 5.5 if
the pilot shows it is equally faithful. Both through the Message Batches API
(50% off). Python `anthropic` SDK.

**Output** (`data/<name>/`): `train.jsonl`, `valid.jsonl`, `test.jsonl` as
`{"messages": [system, user, assistant]}` for mlx-lm, plus a metadata file per
split with source, document, gold chunk, kind, question style and teacher model.

## Pilot (before any batch spend)

20 documents through the full pipeline with the real models and prompts.
Report measured input/output tokens and cost per example, the projected cost
for 5,000 examples (Sonnet vs Opus teacher), and about 10 examples for the user
to review. The batch run starts only after the user approves the numbers.

## Training

LoRA on all layers with mlx-lm on the M4, `--mask-prompt`, batch 2 ×
gradient accumulation 2, gradient checkpointing, 1-2 epochs.

- SmolLM2-360M-Instruct: learning rate 1e-4.
- Qwen3-0.6B: learning rate 2e-5 to 5e-5, to keep what it already does well.
- Later, if time allows: Gemma 3 270M (gated, needs HF auth), LFM2.5-350M.

Checkpoints are chosen on the synthetic valid split. The blog eval is only run
on the chosen checkpoint, never used for selection.

## Evaluation

- **Main:** the blog's 45 questions (`scripts/ask_eval/results/prompts.jsonl`),
  generated with transformers.js q4f16 on CPU with the blog's decode settings
  (`js/generate.mjs`), judged by the blog's `judge.py` (Opus 5.5) with
  `--runs-dir`. About $0.40 per model.
- **Secondary:** the synthetic test split, same judge, for a held-out check
  that isn't the blog.
- **Success:** Qwen3-0.6B above 15/30 faithful with at most 5/15 negatives
  invented; SmolLM2-360M at 10/30 or better.

## Export fix (before publishing)

Our q4f16 files are larger than the official ones (SmolLM2-360M 300MB vs 273MB,
Qwen3-0.6B 649MB vs 570MB) because the lm_head is quantized separately instead
of tied to the embedding, and the graph isn't onnxslim'd. Fix in
`scripts/export_onnx.py` and check the eval score doesn't change.

## Publishing

- Dataset on Hugging Face, CC-BY 4.0, with per-source attribution and licences.
- Models on Hugging Face in the transformers.js layout (`onnx/model_q4f16.onnx`)
  plus merged safetensors and the LoRA adapter.
- Code on GitHub. Repo names to be confirmed with the user.
- Before releasing the dataset: check Anthropic's current terms on using Claude
  output to train models (user decision).

## Out of scope

Full fine-tuning (LoRA only), models over 1B, changing the blog's prompt or
retrieval, WebGPU benchmarking.
