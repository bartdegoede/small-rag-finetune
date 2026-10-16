# Plan 2: pilot fixes and the full dataset

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three pilot findings, then generate about 5,000 examples with the Message Batches API (Sonnet 5.5 questions, Opus 5.5 teacher) and write train/valid/test files mlx-lm can train on.

**Architecture:** Same pipeline as the pilot (`index.py`, `examples.py`, `claude.py`), driven by a resumable `generate.py`. A small `batch.py` submits a dict of `custom_id → params`, records the in-flight batch id on disk, polls, appends results to a JSONL file and resubmits failures (up to 3 rounds), so an interrupted run resumes instead of paying twice. Gold chunks are drawn per source and per split (document-level 90/5/5).

**Tech Stack:** Python 3.12, `anthropic` SDK (Message Batches), sss_eval, model2vec, click, pytest.

Spec: `docs/superpowers/specs/2026-10-15-dataset-and-finetune-design.md`. Pilot findings: `docs/superpowers/specs/2026-10-15-pilot-report.md`. Decision (user, 2026-10-15): Opus 5.5 is the teacher (~$15 projected for 5,000 examples with batch).

**Conventions:** as in Plan 1. Work in `~/projects/small-rag-finetune` on branch `plan1-corpus-pilot`; `unset VIRTUAL_ENV;` and `uv run --no-sync`; API calls through `zsh -ic` (key in the interactive profile); black + ruff before each commit; tests use fakes only.

## File structure

| file | change |
|---|---|
| `src/small_rag_finetune/normalize.py` | strip `{#anchor}` heading attributes (Go and Rust blogs) |
| `src/small_rag_finetune/claude.py` | teacher guidance: no "passages", partial answers allowed |
| `src/small_rag_finetune/examples.py` | random position for a swapped-in gold chunk; `pick_gold` moves here and takes a split |
| `src/small_rag_finetune/pilot.py` | use the moved `pick_gold`, pass `rng` |
| `src/small_rag_finetune/batch.py` | new: resumable Message Batches runner |
| `src/small_rag_finetune/generate.py` | new: full generation, target checks, dataset files |
| `src/small_rag_finetune/__init__.py` | `generate` command |
| `tests/fakes.py` | `FakeBatches` on `FakeClient` |

---

### Task 1: Strip heading anchors

**Files:** Modify `src/small_rag_finetune/normalize.py`; test `tests/test_normalize.py`

- [ ] **Step 1: Failing test** (append)

```python
def test_heading_anchor_attributes_are_stripped(tmp_path):
    go = write(tmp_path, "g.md", "---\ntitle: T\n---\n\n## Builds for Go {#go}\n\nText.\n")
    rb = write(tmp_path, "content/r.md", '+++\ntitle = "T"\n+++\n\n## Why {#why}\n\nText.\n')
    assert "## Builds for Go\n" in go_blog(go).body
    assert "## Why\n" in rust_blog(rb).body
```

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_normalize.py -q` → 1 failed.

- [ ] **Step 3: Implement.** Add next to the other patterns:

```python
_HEADING_ATTR = re.compile(r"[ \t]*\{#[^}\n]*\}")
```

In `go_blog`: `body = _HEADING_ATTR.sub("", _GO_ACTION.sub("", raw[m.end() :]))`.
In `rust_blog`: `body = _HEADING_ATTR.sub("", _LINK_DEF.sub("", raw[m.end() :]))`.

- [ ] **Step 4:** tests pass. Rebuild: `rm -rf data/corpus && uv run --no-sync small-rag-finetune build-corpus`; then `rg -c "\{#" data/corpus | wc -l` → 0 (or only code samples).

- [ ] **Step 5: Commit** `Strip {#anchor} heading attributes from Go and Rust blog posts`

---

### Task 2: Teacher guidance

**Files:** Modify `src/small_rag_finetune/claude.py`; test `tests/test_claude.py`

- [ ] **Step 1: Failing test** (append)

```python
from small_rag_finetune.claude import TEACHER_GUIDANCE


def test_teacher_guidance_rules():
    g = TEACHER_GUIDANCE
    assert "Never mention the passages" in g
    assert "only part of the question" in g
    assert g.rstrip().endswith("The blog does not seem to cover this.")
```

- [ ] **Step 2:** run → fails on the first assert.

- [ ] **Step 3: Implement.** Replace `TEACHER_GUIDANCE` with:

```python
TEACHER_GUIDANCE = """

How to answer:
- Answer the question directly in plain declarative sentences. Never mention the passages, passage numbers like [2], or phrases such as "according to".
- Use only facts stated in the passages. Prefer one sentence; use a second only when the question needs it.
- If they answer only part of the question, give that part; you may add a short clause that the blog does not go into more detail.
- If the passages do not answer the question at all, reply with exactly: The blog does not seem to cover this."""
```

- [ ] **Step 4:** tests pass. **Step 5: Commit** `Teacher guidance: never mention passages, allow partial answers`

---

### Task 3: Random position for a swapped-in gold chunk; `pick_gold` per split

**Files:** Modify `src/small_rag_finetune/examples.py`, `src/small_rag_finetune/pilot.py`; tests `tests/test_examples.py`

- [ ] **Step 1: Failing tests.** In `tests/test_examples.py`, replace `test_with_gold_keeps_hits_when_gold_is_retrieved` and `test_with_gold_replaces_the_fourth_and_resorts` with the tests below, add `import random`, add `rng=random.Random(0)` to every `examples_for(...)` call, and append the `pick_gold` tests:

```python
def test_with_gold_keeps_hits_when_gold_is_retrieved():
    hits = [Hit(i, 1.0 - i / 10, None) for i in range(4)]
    out, found = with_gold(hits, hits[2], random.Random(0))
    assert found and out == hits


def test_with_gold_drops_the_fourth_and_inserts_at_a_random_position():
    hits = [Hit(i, 0.9 - i / 10, None) for i in range(4)]
    gold = Hit(9, 0.1, None)
    positions = set()
    for seed in range(40):
        out, found = with_gold(hits, gold, random.Random(seed))
        assert not found
        assert [h.index for h in out if h.index != 9] == [0, 1, 2]
        positions.add([h.index for h in out].index(9))
    assert positions == {0, 1, 2, 3}


def test_pick_gold_spreads_over_documents_then_repeats(tmp_path):
    idx = index(tmp_path)
    picks = pick_gold(idx, "src", 100, random.Random(0), split=None)
    assert len(picks) == len(set(picks))
    first = [idx.chunks[i].post for i in picks[: len({c.post for c in idx.chunks})]]
    assert len(first) == len(set(first))


def test_pick_gold_respects_the_split(tmp_path):
    idx = index(tmp_path)
    for split in ("train", "valid", "test"):
        for i in pick_gold(idx, "src", 10, random.Random(0), split=split):
            assert split_of("src", idx.chunks[i].post) == split
```

Imports to add: `from small_rag_finetune.examples import pick_gold` and `from small_rag_finetune.corpus import split_of`. `small_corpus` snippets are ~400 characters, above `MIN_SNIPPET`.

- [ ] **Step 2:** run → fails (`with_gold() takes 2 positional arguments`, `cannot import name 'pick_gold'`).

- [ ] **Step 3: Implement** in `examples.py`:

```python
import random
from collections import defaultdict

MIN_SNIPPET = 300  # gold chunks shorter than this rarely support 3 questions


def with_gold(
    hits: list[Hit], gold: Hit, rng: random.Random
) -> tuple[list[Hit], bool]:
    """If retrieval missed the gold chunk, it replaces the 4th hit at a random
    position: keeping its (low) score order would put the answer last almost
    every time (pilot: 37/54 in passage [4])."""
    if any(h.index == gold.index for h in hits):
        return hits, True
    kept = list(hits[: TOP_K - 1])
    kept.insert(rng.randrange(len(kept) + 1), gold)
    return kept, False


def pick_gold(
    index: SourceIndex, source: str, n: int, rng: random.Random, split: str | None = "train"
) -> list[int]:
    """n distinct gold chunks from one split's documents: one per document
    first (documents in random order), then second chunks, and so on."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(index.chunks):
        if len(c.snippet) >= MIN_SNIPPET and (split is None or split_of(source, c.post) == split):
            by_doc[c.post].append(i)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for d in docs:
        rng.shuffle(by_doc[d])
    out: list[int] = []
    for depth in range(max(map(len, by_doc.values()), default=0)):
        for d in docs:
            if depth < len(by_doc[d]):
                out.append(by_doc[d][depth])
                if len(out) == n:
                    return out
    return out
```

`examples_for` gets a keyword `rng: random.Random` and calls `with_gold(index.top(q), gold, rng)`. Update the module docstring's first bullet to "…it replaces the 4th at a random position…".

In `pilot.py`: delete `MIN_SNIPPET` and `pick_gold`, import `pick_gold` from `.examples`, drop the now-unused `split_of`/`defaultdict` imports only if unused, and pass `rng=rng` to `examples_for`.

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.

- [ ] **Step 5: Commit** `Swapped-in gold chunk at a random position; pick_gold per split`

---

### Task 4: Resumable batch runner

**Files:** Create `src/small_rag_finetune/batch.py`; modify `tests/fakes.py`; test `tests/test_batch.py`

- [ ] **Step 1: Fakes.** In `tests/fakes.py` factor the reply into a function and add batches:

```python
def reply_for(params: dict) -> str:
    return FakeClient.ANSWER if "system" in params else FakeClient.QUESTIONS


class FakeBatches:
    """client.messages.batches: every batch ends at once; ids in `fail` error once."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.batches: dict[str, list[dict]] = {}
        self.created: list[int] = []

    def create(self, requests):
        bid = f"batch_{len(self.batches)}"
        self.batches[bid] = list(requests)
        self.created.append(len(requests))
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended")

    def results(self, bid):
        for r in self.batches[bid]:
            cid = r["custom_id"]
            if cid in self.fail:
                self.fail.discard(cid)
                yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="errored"))
                continue
            msg = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=reply_for(r["params"]))],
                usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
                model=r["params"]["model"],
            )
            yield SimpleNamespace(
                custom_id=cid, result=SimpleNamespace(type="succeeded", message=msg)
            )
```

In `FakeClient.__init__` add `self.batches = FakeBatches()`, and in `create` use `text = reply_for(params)`.

- [ ] **Step 2: Failing tests**

```python
# tests/test_batch.py
import json

from fakes import FakeClient

from small_rag_finetune.batch import run_batch

REQS = {
    "q00000": {"model": "claude-sonnet-5-5", "messages": []},
    "t00000": {"model": "claude-opus-5-5", "system": "s", "messages": []},
}


def test_collects_results_with_text_and_tokens(tmp_path):
    client = FakeClient()
    out = run_batch(client, REQS, tmp_path, poll=0)
    assert set(out) == set(REQS)
    assert out["q00000"]["text"].startswith("Q1:")
    assert out["t00000"]["model"] == "claude-opus-5-5"
    assert out["t00000"]["output_tokens"] == 200
    assert not (tmp_path / "inflight.json").exists()


def test_failures_are_resubmitted(tmp_path):
    client = FakeClient()
    client.batches.fail = {"t00000"}
    out = run_batch(client, REQS, tmp_path, poll=0)
    assert set(out) == set(REQS)
    assert client.batches.created == [2, 1]


def test_a_second_run_reuses_saved_results(tmp_path):
    run_batch(FakeClient(), REQS, tmp_path, poll=0)
    client = FakeClient()
    assert set(run_batch(client, REQS, tmp_path, poll=0)) == set(REQS)
    assert client.batches.created == []


def test_an_inflight_batch_is_polled_not_resubmitted(tmp_path):
    client = FakeClient()
    bid = client.batches.create(
        [{"custom_id": k, "params": v} for k, v in REQS.items()]
    ).id
    (tmp_path / "inflight.json").write_text(json.dumps({"id": bid}))
    out = run_batch(client, REQS, tmp_path, poll=0)
    assert set(out) == set(REQS)
    assert client.batches.created == [2]  # only the one made above
```

- [ ] **Step 3:** run → `ModuleNotFoundError: No module named 'small_rag_finetune.batch'`.

- [ ] **Step 4: Implement**

```python
# src/small_rag_finetune/batch.py
"""Message Batches (50% off): submit, poll, collect, resubmit failures.

Resumable through files in one directory per phase:
- results.jsonl: one line per succeeded request (custom_id, model, text, tokens)
- inflight.json: the batch being processed, so a restart polls it instead of
  paying for it twice
custom_id must match ^[a-zA-Z0-9_-]{1,64}$, so callers use ids like q00042.
"""

import json
import time
from pathlib import Path

MAX_ROUNDS = 3


def _load(d: Path) -> dict[str, dict]:
    path = d / "results.jsonl"
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line)
    return {r["custom_id"]: r for r in rows}


def run_batch(
    client, requests: dict[str, dict], d: Path, *, poll: float = 60
) -> dict[str, dict]:
    d.mkdir(parents=True, exist_ok=True)
    results = _load(d)
    inflight = d / "inflight.json"
    for _ in range(MAX_ROUNDS):
        if inflight.exists():
            batch_id = json.loads(inflight.read_text())["id"]
        else:
            todo = [cid for cid in requests if cid not in results]
            if not todo:
                break
            batch = client.messages.batches.create(
                requests=[{"custom_id": cid, "params": requests[cid]} for cid in todo]
            )
            batch_id = batch.id
            inflight.write_text(json.dumps({"id": batch_id, "n": len(todo)}))
            print(f"submitted {batch_id}: {len(todo)} requests", flush=True)
        while client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(poll)
        with (d / "results.jsonl").open("a") as f:
            for r in client.messages.batches.results(batch_id):
                if r.result.type != "succeeded":
                    continue
                msg = r.result.message
                row = {
                    "custom_id": r.custom_id,
                    "model": msg.model,
                    "text": next((b.text for b in msg.content if b.type == "text"), ""),
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                }
                results[r.custom_id] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        inflight.unlink()
    return {cid: results[cid] for cid in requests if cid in results}
```

- [ ] **Step 5:** `uv run --no-sync pytest -q` → all pass. **Step 6: Commit** `Resumable Message Batches runner`

---

### Task 5: Full generation and dataset files

**Files:** Create `src/small_rag_finetune/generate.py`; modify `src/small_rag_finetune/__init__.py`; test `tests/test_generate.py`

Quotas: the pilot gave 3.5 examples per gold chunk, so ~1,430 gold chunks for 5,000 examples: per source 322 train, 18 valid, 18 test (≈4,500 / 250 / 250 examples).

- [ ] **Step 1: Failing tests**

```python
# tests/test_generate.py
import json

from fakes import FakeClient, FakeModel, write_post

from small_rag_finetune.generate import run_generate, target_problem
from small_rag_finetune.prompt import REFUSAL

TOPICS = ["generics", "async", "walrus", "lifetimes", "modules", "go", "rust", "python"]


def corpus(root):
    for i, t in enumerate(TOPICS * 3):
        write_post(root / "src", f"{t}-{i}", f"About {t}", f"All about {t} in practice. " * 20)


def test_target_problem():
    assert target_problem(None) == "missing"
    assert target_problem("") == "missing"
    assert target_problem("Passage [2] says so.") == "mentions-passages"
    assert target_problem("The passages say X.") == "mentions-passages"
    assert target_problem("One. Two. Three.") == "over-2-sentences"
    assert target_problem(REFUSAL) is None
    assert target_problem("Go 1.26 adds three packages.") is None


def test_generate_writes_mlx_lm_files_meta_and_summary(tmp_path):
    corpus(tmp_path / "corpus")
    out = tmp_path / "v1"
    summary = run_generate(
        tmp_path / "corpus", out, ["src"], {"train": 4, "valid": 2, "test": 2},
        client=FakeClient(), model=FakeModel(), poll=0,
    )
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    meta = [json.loads(line) for line in (out / "train.meta.jsonl").read_text().splitlines()]
    assert train and len(train) == len(meta)
    roles = [m["role"] for m in train[0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert train[0]["messages"][2]["content"] == FakeClient.ANSWER
    assert {"id", "source", "doc", "kind", "question", "gold_rank", "teacher"} <= set(meta[0])
    assert summary["examples"]["train"] == len(train)
    assert summary["usd_batch"] > 0
    assert (out / "summary.json").exists()


def test_generate_resumes_without_new_batches(tmp_path):
    corpus(tmp_path / "corpus")
    args = (tmp_path / "corpus", tmp_path / "v1", ["src"], {"train": 3, "valid": 0, "test": 0})
    run_generate(*args, client=FakeClient(), model=FakeModel(), poll=0)
    client = FakeClient()
    run_generate(*args, client=client, model=FakeModel(), poll=0)
    assert client.batches.created == []
```

- [ ] **Step 2:** run → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/small_rag_finetune/generate.py
"""The full dataset: questions (Sonnet) and teacher answers (Opus) through the
Message Batches API, then train/valid/test files for mlx-lm.

Re-running resumes: both phases keep their results on disk (batch.py), and
example construction is deterministic given those results and the seed.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from sss_eval.chunk import split_sentences

from .batch import run_batch
from .claude import (
    cost,
    parse_questions,
    question_params,
    teacher_params,
)
from .examples import pick_gold, examples_for
from .index import SourceIndex
from .pilot import NEAR_MISS_RATE, REMOVED_RATE, write_jsonl

TEACHER = "claude-opus-5-5"
QUOTAS = {"train": 322, "valid": 18, "test": 18}  # gold chunks per source
_PASSAGE_TALK = re.compile(r"\bpassages?\b|\[\d\]", re.IGNORECASE)


def target_problem(answer: str | None) -> str | None:
    if not answer:
        return "missing"
    if _PASSAGE_TALK.search(answer):
        return "mentions-passages"
    if len(split_sentences(answer)) > 2:
        return "over-2-sentences"
    return None


def _usd(results: dict[str, dict]) -> float:
    return sum(
        cost(r["model"], r["input_tokens"], r["output_tokens"], batch=True)
        for r in results.values()
    )


def run_generate(
    corpus_root: Path,
    out_dir: Path,
    sources: list[str],
    quotas: dict[str, int],
    *,
    seed: int = 0,
    client=None,
    model=None,
    poll: float = 60,
) -> dict:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    rng = random.Random(seed)
    indexes = {s: SourceIndex.build(corpus_root / s, model=model) for s in sources}
    golds = [
        (s, gi)
        for s in sources
        for split, n in quotas.items()
        for gi in pick_gold(indexes[s], s, n, rng, split=split)
    ]

    q_reqs = {
        f"q{k:05d}": question_params(indexes[s].chunks[gi])
        for k, (s, gi) in enumerate(golds)
    }
    q_res = run_batch(client, q_reqs, out_dir / "batches" / "questions", poll=poll)

    examples = []
    for k, (s, gi) in enumerate(golds):
        r = q_res.get(f"q{k:05d}")
        if r is None:
            continue
        qs = parse_questions(r["text"])
        removed = {j for j in range(len(qs.answerable)) if rng.random() < REMOVED_RATE}
        near = rng.random() < NEAR_MISS_RATE
        examples += examples_for(
            indexes[s], s, gi, qs, use_near_miss=near, removed=removed, rng=rng
        )

    t_reqs = {
        f"t{k:05d}": teacher_params(TEACHER, e.messages()) for k, e in enumerate(examples)
    }
    t_res = run_batch(client, t_reqs, out_dir / "batches" / "teacher", poll=poll)

    splits: dict[str, list[tuple]] = {"train": [], "valid": [], "test": []}
    dropped: Counter = Counter()
    for k, e in enumerate(examples):
        r = t_res.get(f"t{k:05d}")
        answer = " ".join(r["text"].split()) if r else None
        problem = target_problem(answer)
        if problem:
            dropped[problem] += 1
            continue
        splits[e.split].append((e, answer))

    for split, rows in splits.items():
        write_jsonl(
            out_dir / f"{split}.jsonl",
            [{"messages": [*e.messages(), {"role": "assistant", "content": a}]} for e, a in rows],
        )
        write_jsonl(
            out_dir / f"{split}.meta.jsonl",
            [
                {
                    "id": e.id, "source": e.source, "doc": e.doc, "kind": e.kind,
                    "question": e.question, "gold_rank": e.gold_rank,
                    "gold_retrieved": e.gold_retrieved, "teacher": TEACHER,
                }
                for e, _ in rows
            ],
        )
    summary = {
        "examples": {s: len(rows) for s, rows in splits.items()},
        "kinds": dict(Counter(e.kind for rows in splits.values() for e, _ in rows)),
        "refusals": sum(a == "The blog does not seem to cover this." for rows in splits.values() for _, a in rows),
        "dropped": dict(dropped),
        "usd_batch": round(_usd(q_res) + _usd(t_res), 2),
        "seed": seed,
        "quotas": quotas,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
```

Use `REFUSAL` from `.prompt` for the refusal count instead of the literal.

CLI (append to `__init__.py`):

```python
@main.command()
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("data/corpus"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/v1"))
@click.option("--seed", default=0, show_default=True)
def generate(corpus: Path, out: Path, seed: int) -> None:
    """Full dataset via the Message Batches API (resumable)."""
    from .generate import QUOTAS, run_generate
    from .sources import SOURCES

    summary = run_generate(corpus, out, list(SOURCES), QUOTAS, seed=seed)
    click.echo(json.dumps(summary, indent=2))
```

(add `import json` at the top of `__init__.py`).

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass; black, ruff clean.

- [ ] **Step 5: Commit** `Full dataset generation via Message Batches, resumable`

---

### Task 6: Generate the dataset (spends ~$15, approved)

- [ ] **Step 1: Run in the background** (batches usually finish within an hour each, at most 24h):

`cd ~/projects/small-rag-finetune && mkdir -p logs && zsh -ic 'unset VIRTUAL_ENV; uv run --no-sync small-rag-finetune generate' > logs/generate-v1.log 2>&1`

Expected log: `submitted msgbatch_…: ~1430 requests`, later `submitted msgbatch_…: ~5000 requests`, then the summary JSON. If interrupted, re-run the same command: it resumes.

- [ ] **Step 2: Check the summary:** about 4,500 / 250 / 250 examples; refusals 15-20%; dropped under 3%; `usd_batch` near $15. Spot-read 10 random train lines and 5 refusals with `jq`.

- [ ] **Step 3: Record** the summary in `docs/superpowers/specs/2026-10-15-dataset-v1.md` (counts, cost, dropped reasons, three sample examples) and commit `Dataset v1: generation summary`.
