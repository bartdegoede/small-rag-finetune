---
license: cc-by-4.0
language:
- en
task_categories:
- question-answering
- text-generation
tags:
- rag
- retrieval-augmented-generation
- synthetic
- small-language-models
pretty_name: Short RAG answers for small language models
size_categories:
- 1K<n<10K
configs:
- config_name: v3
  default: true
  data_files:
  - split: train
    path: v3/train.jsonl
  - split: validation
    path: v3/validation.jsonl
  - split: test
    path: v3/test.jsonl
- config_name: broad
  data_files:
  - split: train
    path: broad/train.jsonl
  - split: validation
    path: broad/validation.jsonl
  - split: test
    path: broad/test.jsonl
- config_name: v1
  data_files:
  - split: train
    path: v1/train.jsonl
  - split: validation
    path: v1/validation.jsonl
  - split: test
    path: v1/test.jsonl
---

# Short RAG answers for small language models

Questions about technical writing, each with the top 4 passages a small
embedding model retrieved for it, and a short answer written from those passages
only, or a refusal when the passages don't answer the question. It was made to
fine-tune a 0.6B-parameter model for an in-browser "ask my blog" page on
[bart.degoe.de](https://bart.degoe.de), where the model must answer in at most
two sentences from retrieved passages, and say so when they don't cover the
question.

Code, results and the full write-up of what each version did to the model:
[github.com/bartdegoede/small-rag-finetune](https://github.com/bartdegoede/small-rag-finetune).

## Made with Claude

This dataset is synthetic. **The questions were written by Claude Sonnet 5.5 and
the answers by Claude Opus 5.5** (Anthropic), through the Message Batches API.
Nobody reviewed the examples by hand. Answers were filtered automatically (over
two sentences, or mentioning "passages", was dropped) and checked by sampling.

## Configurations

| config | examples (train / validation / test) | refusals | answers |
|---|---|---|---|
| `v3` (default) | 4,530 / 255 / 260 | 16% | one short sentence with the fact asked for, in the passage's own words |
| `broad` | 1,199 / 77 / 76 | 11% | overview and "how do I" questions about a whole document, answered in one or two sentences |
| `v1` | 4,529 / 256 / 259 | 13% | up to two dense sentences combining several facts |

`v3` and `v1` share the questions and passages; only the answers differ.

What they did to Qwen3-0.6B (LoRA, one epoch, judged by Claude Opus 5.5 on the
blog's 45 questions: 30 the blog answers, 15 it doesn't):

| | faithful (of 30) | wrong answers | invented (of 15) |
|---|---|---|---|
| base model | 17 | 9 | 2 |
| `v1` | 8 | 18 | 0 |
| `v3` | 16 | 3 | 0 |
| `v3` + `broad` | 13 | 7 | 0 |

`v3` is the one that worked: as often right as the base model, with a third of
its wrong answers. A 0.6B model imitates the density of `v1`'s answers without
reading the passages as carefully, and the `broad` answers, which summarise
several passages, made it garble overview questions. Both are kept for
comparison.

## Fields

Each row:

- `messages`: system, user (numbered passages and the question) and assistant
  turns, ready for chat fine-tuning (e.g. `mlx_lm.lora`, TRL's `SFTTrainer`).
- `answer`, `refusal`: the assistant turn, and whether it is the refusal "The
  blog does not seem to cover this."
- `id`, `source`, `doc`: where the question came from (`doc` is the document's
  slug in its source).
- `kind`: `answerable` (the passage the question was written from is among the
  4), `removed` (that document was taken out of retrieval, so the expected answer
  is usually a refusal), `near-miss` (a related question the passages probably
  don't answer), and in `broad`, `broad` / `broad-removed`.
- `question`; `gold_rank` and `gold_retrieved` (v1, v3) or `doc_retrieved`
  (broad): whether retrieval found the source passage or document by itself. If
  it didn't, the source passage was put in at a random position.
- `teacher`: the model that wrote the answer.

The teacher decided whether to answer, so a `near-miss` or `removed` question
that a neighbouring passage does answer has a real answer.

## How it was made

1. **Corpus.** 2,193 documents from four openly licensed sources (below), split
   90/5/5 into train/validation/test **by document**, so no document is in two
   splits.
2. **Retrieval.** Documents are chunked (600 characters, 120 overlap) and
   embedded with [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M)
   at 128 dimensions; each question gets the top 4 chunks. The passages are as
   noisy as they are in production.
3. **Questions** by Claude Sonnet 5.5 from one chunk (v1/v3), or from a
   document's title, headings and opening (broad).
4. **Answers** by Claude Opus 5.5 under the system prompt in `messages`, with
   extra style guidance per version.

## Sources and attribution

The passages and questions are derived from these sources, each pinned to a
commit. Text from the documents is quoted in the passages, and the answers
reuse its wording.

| source | repository (commit) | licence |
|---|---|---|
| `go-blog` | [golang/website](https://github.com/golang/website/tree/bacaf18c1c9234fabbd08847eb33073c50d9254b/_content/blog), `_content/blog` | CC-BY-4.0, © The Go Authors |
| `rust-blog` | [rust-lang/blog.rust-lang.org](https://github.com/rust-lang/blog.rust-lang.org/tree/c76891ecae17a22ee76a29096c532709be853b87/content), `content` | MIT OR Apache-2.0, the Rust project |
| `rust-rfcs` | [rust-lang/rfcs](https://github.com/rust-lang/rfcs/tree/d2705ad64ba47e812a2b6b19d6f613fb4c14c5e0/text), `text` | MIT OR Apache-2.0, the Rust project |
| `peps` | [python/peps](https://github.com/python/peps/tree/b80c7bf8d5a89fc47b732504a9e1219b79677531/peps), `peps` | public domain / CC0-1.0 |

Changes: documents were converted to plain markdown, split into chunks and
quoted as passages; nothing else was edited. The Rust passages remain available
under MIT OR Apache-2.0; both licence texts are in `licenses/`. The dataset as a whole (the selection, questions and
answers) is released under CC-BY-4.0.

## Limitations

- English technical writing about Go, Rust and Python only. The system prompt
  says "blog" for every source.
- Synthetic and not reviewed by hand: some answers will be wrong or refuse when
  they shouldn't.
- The refusal is one fixed sentence; a model trained on this will use exactly
  that wording.

## Citation

```bibtex
@misc{degoede2026smallrag,
  author = {Bart de Goede},
  title = {Short RAG answers for small language models},
  year = {2026},
  howpublished = {\url{https://huggingface.co/datasets/bartdegoede/small-rag-answers}}
}
```
