# small-rag-finetune

Fine-tune small language models (SmolLM2-360M, Qwen3-0.6B) to answer a question
from retrieved passages in at most two sentences, or to say the passages don't
cover it. The target is the in-browser "ask my blog" page on
[bart.degoe.de](https://bart.degoe.de), where the model has to fit a
download budget of about 350 MB and run on WebGPU or WASM through transformers.js.

The repo builds the training data, trains LoRA adapters with mlx-lm, picks a
checkpoint, exports it to the ONNX q4f16 layout transformers.js loads, and scores
it on the blog's 45 evaluation questions.

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
   position), *removed* (the gold chunk is taken out, so the right answer is a
   refusal), and *near-miss* (a related question the passages probably don't
   answer).
4. **Answers.** Claude Opus 5.5 answers each example from the passages under the
   blog's system prompt. The teacher decides whether to answer or refuse, so a
   near-miss question that a neighbouring chunk does answer gets a real answer.
   Answers over two sentences, or that mention "passages", are dropped.

Dataset v1 (`data/v1/`) cost $16.45 through the Message Batches API:

| split | examples |
|---|---|
| train | 4,529 |
| valid | 256 |
| test | 259 |

About 13% of answers are refusals ("The blog does not seem to cover this."). Each
line of `train.jsonl` holds the chat messages (system, user with the 4 numbered
passages and the question, assistant), the format `mlx_lm.lora` reads. Next to it,
`train.meta.jsonl` records each example's source, document, kind, gold passage
rank and teacher.

Design and decisions: `docs/superpowers/specs/2026-10-15-dataset-and-finetune-design.md`;
pilot findings: `docs/superpowers/specs/2026-10-15-pilot-report.md`.

## Setup

Apple Silicon (training uses MLX), Python 3.12+, [uv](https://docs.astral.sh/uv/),
Node and pnpm.

```bash
uv sync
pnpm install           # transformers.js, for js/generate.mjs
uv run pytest          # tests use fakes: no network, no API key
```

Steps that call Claude need `ANTHROPIC_API_KEY` in the environment.

## Building the dataset

```bash
uv run small-rag-finetune fetch-sources   # sparse checkouts at the pinned commits -> data/sources/
uv run small-rag-finetune build-corpus    # normalize to markdown posts -> data/corpus/
uv run small-rag-finetune pilot           # 5 docs per source, both teachers, cost report -> data/pilot/
uv run small-rag-finetune generate        # full dataset via Message Batches -> data/v1/
```

`generate` is resumable: each batch phase keeps its results and the in-flight
batch id under `data/v1/batches/`, so a restart polls the running batch instead
of paying for it again. Run the pilot first; its report (`data/pilot/report.md`)
projects the cost of the full run.

## Training

```bash
nohup scripts/train_v1.sh &
```

This trains a LoRA on all layers (rank 8, prompt masked) for 2 epochs at batch 2
with gradient accumulation 2 and gradient checkpointing. It validates on the whole
valid split and saves a checkpoint every quarter epoch. Logs go to `logs/`,
adapters to `runs/`. On a 24GB M4 Mac mini, SmolLM2-360M takes about 1h50m.
Without gradient checkpointing, Qwen3-0.6B runs out of memory at batch 4.

Pick the checkpoint with the lowest validation loss, never by the blog eval:

```bash
uv run small-rag-finetune best-checkpoint logs/v1-qwen3-0.6b.log runs/v1-qwen3-0.6b
# -> runs/v1-qwen3-0.6b-best/ (adapters.safetensors + adapter_config.json)
```

For both models the lowest validation loss came at the end of the first epoch.
The second epoch lowers training loss sharply and raises validation loss.

Plot the curves as a theme-aware SVG for the blog (`{{< svg >}}` shortcode,
styled by the blog's `assets/css/extended/charts.css`):

```bash
uv run small-rag-finetune loss-chart logs/v1-qwen3-0.6b.log runs/charts/loss-qwen3-0.6b.svg --name Qwen3-0.6B
```

## Export to the browser

Merge the adapter, then build `onnx/model_q4f16.onnx` with the onnxruntime-genai
model builder (4-bit weights, fp16 activations, the tool behind the official
onnx-community files):

```bash
uv run mlx_lm.fuse --model Qwen/Qwen3-0.6B \
  --adapter-path runs/v1-qwen3-0.6b-best --save-path runs/fused/v1-qwen3-0.6b
uv run scripts/export_onnx.py runs/fused/v1-qwen3-0.6b runs/onnx/v1-qwen3-0.6b \
  --tokenizer Qwen/Qwen3-0.6B
```

`export_onnx.py` is a standalone script with its own dependencies (PEP 723). The
output loads in transformers.js 4.3.1 with `dtype: "q4f16"`, on both WebGPU and
WASM, and is smaller than the official files: SmolLM2-360M 195 MB (official
272 MB), Qwen3-0.6B 346 MB (official 570 MB). The docstring explains the two
fixes it applies: the KV-cache dimension and the inlined chat template.

## Evaluation

Answer the blog's 45 questions (30 answerable, 15 the blog doesn't cover) with
transformers.js, using the same decoding settings as the blog page:

```bash
node js/generate.mjs runs/onnx/v1-qwen3-0.6b eval/blog-prompts.jsonl > runs/eval/v1-qwen3-0.6b.jsonl
```

The answers are graded by the blog repo's judge (Claude Opus 5.5; about $0.44
per model). It labels each answer *faithful*, *unsupported* or *non-answer*.
Run it from a checkout of the blog:

```bash
uv run python scripts/ask_eval/judge.py --runs-dir <this repo>/runs/eval \
  --out <this repo>/runs/eval/judgements.jsonl --arm v1-qwen3-0.6b
```

Compare against the base model exported the same way (`genai-base-*` arms), so
the fine-tune is the only difference. Results go in `docs/superpowers/specs/`.

## Development

```bash
uv run black src tests && uv run ruff check src tests
uv run pytest
```
