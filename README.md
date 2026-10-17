# small-rag-finetune

Fine-tune a small language model to answer a question from retrieved passages in
at most two sentences, or to say the passages don't cover it. The target is the
in-browser "ask my blog" page on [bart.degoe.de](https://bart.degoe.de), where
the model has to fit a download budget of about 350 MB and run on WebGPU or WASM
through transformers.js.

The repo builds the training data with Claude, trains LoRA adapters with mlx-lm
on a Mac, picks a checkpoint on judged answers, exports it to the ONNX q4f16
layout transformers.js loads, and scores it on the blog's 45 evaluation
questions.

- **Model:** [bartdegoede/qwen3-0.6b-small-rag](https://huggingface.co/bartdegoede/qwen3-0.6b-small-rag)
  (v3), ready for transformers.js.
- **Dataset:** [bartdegoede/small-rag-answers](https://huggingface.co/datasets/bartdegoede/small-rag-answers).
- **Use it on your own site:** [`examples/browser/`](examples/browser/) is a
  static "ask the site" page, and "Evaluate it on your own site" below checks how
  well it does on your posts.
- **Results:** [`docs/results.md`](docs/results.md).

The training data comes from Go, Rust and Python writing, not from the blog it
was built for, so nothing in the model is specific to that blog.

## Results

Qwen3-0.6B on the blog's 45 questions (30 the blog answers, 15 it doesn't),
judged by Claude Opus 5.5, repetition penalty off:

| | faithful (of 30) | wrong answers | invented (of 15) |
|---|---|---|---|
| base model | 17 | 9 | 2 |
| v1: dense two-sentence answers | 8 | 18 | 0 |
| v2: v1 without "the blog doesn't say..." hedges | 9 | 17 | 0 |
| **v3: one short extractive sentence** | **16** | **3** | **0** |
| v4: v3 plus broad questions | 13 | 7 | 0 |

v3 is as often right as the base model, with a third of its wrong answers, and
never answers a question the blog doesn't cover. What didn't work, and why, is
in [`docs/results.md`](docs/results.md): a 0.6B model copies the density of its
targets without reading as carefully, and validation loss kept falling while the
answers got worse.

## How the dataset is made

1. **Corpus.** Openly licensed technical writing, each source pinned to a commit
   (`src/small_rag_finetune/sources.py`): the Go blog (CC-BY-4.0), the Rust blog
   and Rust RFCs (MIT OR Apache-2.0), and the Python PEPs (public domain / CC0).
   That gives 2,193 documents, split 90/5/5 into train/valid/test **by document**,
   so no document's passages appear in two splits.
2. **Retrieval, the same as the blog.** Documents are chunked (600 characters,
   120 overlap) and embedded with `minishlab/potion-base-8M` at 128 dimensions:
   the blog's own `static-site-search-eval` setup. Each example gets the top 4
   passages, so the model trains on the same kind of noisy context it sees in
   production.
3. **Questions.** Claude Sonnet 5.5 writes questions about a chunk. There are
   three kinds: *answerable* (the gold chunk is among the passages, at a random
   position), *removed* (the document is taken out of retrieval, so the right
   answer is usually a refusal), and *near-miss* (a related question the
   passages probably don't answer).
4. **Answers.** Claude Opus 5.5 answers each example from the passages under the
   blog's system prompt. The teacher decides whether to answer or refuse, so a
   near-miss question that a neighbouring chunk does answer gets a real answer.
   Answers over two sentences, or that mention "passages", are dropped.

The versions differ in step 4 (v3: `generate --style extractive`), in
post-processing (v2: `strip-hedges`), or add examples (v4: `generate-broad`,
then `combine`). Design: [`docs/design.md`](docs/design.md); pilot:
[`docs/pilot-report.md`](docs/pilot-report.md); v1 make-up:
[`docs/dataset-v1.md`](docs/dataset-v1.md).

Costs through the Message Batches API: v1 $16.45, v3 answers $13.71, v4 broad
questions and answers about $6. Grading a checkpoint on the dev split costs about
$1.20.

## Setup

Apple Silicon (training uses MLX), Python 3.12+, [uv](https://docs.astral.sh/uv/),
Node and pnpm.

```bash
uv sync
pnpm install           # transformers.js, for js/generate.mjs
uv run pytest          # tests use fakes: no network, no API key
```

Steps that call Claude need `ANTHROPIC_API_KEY` in the environment.
`tests/test_prompt_parity.py` checks that `prompt.py` matches the blog's
`prompt.js` byte for byte; it needs a checkout of
[the blog](https://github.com/bartdegoede/blog) (`BLOG_DIR`, default
`~/projects/blog`) and is skipped without one.

## Building the dataset

```bash
uv run small-rag-finetune fetch-sources   # sparse checkouts at the pinned commits -> data/sources/
uv run small-rag-finetune build-corpus    # normalize to markdown posts -> data/corpus/
uv run small-rag-finetune pilot           # 5 docs per source, both teachers, cost report -> data/pilot/
uv run small-rag-finetune generate        # v1 via Message Batches -> data/v1/
uv run small-rag-finetune generate --style extractive --questions-from data/v1 --out data/v3
uv run small-rag-finetune generate-broad --out data/v4-broad
uv run small-rag-finetune combine data/v3 data/v4-broad --out data/v4
```

Every Claude step is resumable: each batch phase keeps its results and the
in-flight batch id under `<out>/batches/`, so a restart polls the running batch
instead of paying for it again. `generate` and `generate-broad` take `--sample N`
for a pilot run first.

Each line of `train.jsonl` holds the chat messages (system, user with the 4
numbered passages and the question, assistant), the format `mlx_lm.lora` reads.
Next to it, `train.meta.jsonl` records each example's source, document, kind and
teacher.

## Training

```bash
nohup scripts/train_v3.sh &
```

LoRA on all layers (rank 8, prompt masked), one epoch at batch 2 with gradient
accumulation 2 and gradient checkpointing, saving a checkpoint every quarter
epoch. Logs go to `logs/`, adapters to `runs/`. Qwen3-0.6B takes about 1h20m on a
24GB M4 Mac mini (v4, with more data, 1h40m). Without gradient checkpointing it
runs out of memory at batch 4. `scripts/train_v{1,2,4}.sh` are the other runs.

Plot the loss curves as a theme-aware SVG for the blog (`{{< svg >}}` shortcode,
styled by the blog's `assets/css/extended/charts.css`):

```bash
uv run small-rag-finetune loss-chart logs/v3-qwen3-0.6b.log runs/charts/loss-v3.svg --name Qwen3-0.6B
```

## Export to the browser

Merge a checkpoint's adapter, then build `onnx/model_q4f16.onnx` with the
onnxruntime-genai model builder (4-bit weights, fp16 activations, the tool behind
the official onnx-community files):

```bash
mkdir -p runs/v3-it2264
cp runs/v3-qwen3-0.6b/0002264_adapters.safetensors runs/v3-it2264/adapters.safetensors
cp runs/v3-qwen3-0.6b/adapter_config.json runs/v3-it2264/
uv run mlx_lm.fuse --model Qwen/Qwen3-0.6B \
  --adapter-path runs/v3-it2264 --save-path runs/fused/v3-qwen3-0.6b-it2264
uv run scripts/export_onnx.py runs/fused/v3-qwen3-0.6b-it2264 runs/onnx/v3-qwen3-0.6b-it2264 \
  --tokenizer Qwen/Qwen3-0.6B
```

`export_onnx.py` is a standalone script with its own dependencies (PEP 723). The
output loads in transformers.js 4.3.1 with `dtype: "q4f16"`, on both WebGPU and
WASM (tested in headless Chromium), and is smaller than the official files:
Qwen3-0.6B 346 MB (official 570 MB), SmolLM2-360M 195 MB (official 272 MB). It
applies three fixes, each explained in its docstring: the KV-cache dimension,
the inlined chat template, and the 4-bit embedding rewritten in standard ops,
because onnxruntime-web's WASM build lacks the `GatherBlockQuantized` kernel.

`hf-stage-model` lays out the Hub repo: the merged weights for transformers and
MLX, the ONNX file under `onnx/`, greedy generation defaults and the card in
`src/small_rag_finetune/model_card.md`.

## Picking a checkpoint

Not on validation loss: in v2 it fell at every checkpoint while the answers got
worse. Instead every checkpoint answers the dataset's test split (documents no
checkpoint trained on), and Opus grades each answer against its passages with
the blog judge's rubric, through the Batch API:

```bash
uv run small-rag-finetune dev-prompts data/v3 runs/dev/prompts.jsonl
ARM=dev-v3-it2264 node js/generate.mjs runs/onnx/v3-qwen3-0.6b-it2264 runs/dev/prompts.jsonl \
  > runs/dev/dev-v3-it2264.jsonl
uv run small-rag-finetune dev-judge runs/dev/dev-v3-it2264.jsonl
```

Pick the checkpoint with the most faithful answers. The blog's 45 questions are
kept for one final check of the picked checkpoint.

## Evaluate it on your own site

Write 30 to 50 questions about your posts in a JSONL file: most that your posts
answer, and some they don't, marked `"covered": false` (the right answer to
those is the refusal). `eval/blog-questions.jsonl` is the example, the 45
questions for bart.degoe.de:

```json
{"id": "q10", "question": "Which Qwen3 model does the local coding agent run?"}
{"id": "n07", "question": "Which GPU should I buy to run Llama 70B locally?", "covered": false}
```

Then build the prompts from your posts (same chunking and retrieval as the
browser page), answer them with the model through transformers.js, and have
Claude Opus 5.5 grade each answer against its passages (Batch API, about $0.20
for 45 questions; needs `ANTHROPIC_API_KEY`):

```bash
uv run small-rag-finetune eval-prompts runs/site/prompts.jsonl \
  --corpus path/to/your/posts --questions my-questions.jsonl
uv run hf download bartdegoede/qwen3-0.6b-small-rag --include "*.json" --include "*.txt" \
  --include "onnx/*" --local-dir runs/onnx/qwen3-0.6b-small-rag   # the 330 MB browser build only
ARM=small-rag node js/generate.mjs runs/onnx/qwen3-0.6b-small-rag runs/site/prompts.jsonl \
  > runs/site/small-rag.jsonl
uv run small-rag-finetune dev-judge runs/site/small-rag.jsonl \
  --prompts runs/site/prompts.jsonl --out runs/site/judge
```

The summary counts *faithful*, *unsupported* (a claim the passages don't
support) and *non-answer* (a wrong refusal, or rambling), separately for
`answered` (questions your posts cover) and `refused` (questions they don't).
Run the base model the same way to see whether the fine-tune helps on your site:
download `onnx-community/Qwen3-0.6B-ONNX` with the same `--include` flags (but
`--include "onnx/model_q4f16.onnx"`, as that repo has every precision). On bart.degoe.de's
posts as they are now, the fine-tune gets 18 of 30 right and refuses 13 of 15
correctly.

The published results in [`docs/results.md`](docs/results.md) use
`eval/blog-prompts.jsonl` instead: the same questions with the passages frozen
when the eval was made, graded by the blog repo's `scripts/ask_eval/judge.py`
with the same rubric. Since then a new post changed the top 4 for about half the
questions, so rebuilding the prompts gives different numbers.

`js/generate.mjs` uses the browser page's decoding settings: greedy, at most 120
new tokens, repetition penalty 1.0 (`REP_PENALTY` overrides it).

## Publishing

```bash
uv run small-rag-finetune hf-export dist/hf   # dataset: v3, broad and v1 configs, card, licences
uv run small-rag-finetune hf-stage-model runs/fused/v3-qwen3-0.6b-it2264 \
  runs/onnx/v3-qwen3-0.6b-it2264 dist/model/qwen3-0.6b-small-rag   # model
```

The cards are `src/small_rag_finetune/dataset_card.md` and `model_card.md`.

## Development

```bash
uv run black src tests && uv run ruff check src tests
uv run pytest
```

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The dataset is CC-BY-4.0, with the sources'
own licences for the passages; see its card.
