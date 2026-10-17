"""v4's broad slice: questions about a whole document, answered from plain
retrieval.

v3 learned specific-fact questions and refused most broad ones on the blog
("What has the blog written about hashing and bit arrays?"). Per document,
Sonnet writes a TOPIC question ("What has the blog written about X?") and a
GOAL question ("How can I ...?", without naming the tool). The student sees the
top 4 passages for the question. If none is from the document, its best
chunk replaces the 4th at a random position (as in examples.with_gold): in a
corpus of ~600 documents per source retrieval often misses the document, and
the pilot's teacher then refused, which would teach "broad questions get a
refusal". For REMOVED_RATE of the questions the document is excluded, so
refusals stay in the mix. Opus answers
with BROAD_GUIDANCE (or refuses); the teacher decides.

Writes <out>/{train,valid,test}.jsonl + .meta.jsonl like generate, so
`combine` can add them to v3.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .batch import run_batch
from .claude import (
    BROAD_GUIDANCE,
    broad_question_params,
    cost,
    parse_broad,
    teacher_params,
)
from .corpus import split_of
from .examples import Example, passage, with_gold
from .generate import SPLITS, TEACHER, target_problem
from .index import Hit, SourceIndex
from .pilot import write_jsonl
from .prompt import REFUSAL

DOCS = {"train": 150, "valid": 10, "test": 10}  # per source
REMOVED_RATE = 0.2
OPENING_CHARS = 1200
MAX_HEADINGS = 12


def documents(index: SourceIndex) -> dict[str, list[int]]:
    """Chunk indexes per document, in order."""
    out: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(index.chunks):
        out[c.post].append(i)
    return out


def describe(index: SourceIndex, chunk_ids: list[int]) -> tuple[str, list[str], str]:
    """Title, distinct section headings, and the opening text of a document."""
    chunks = [index.chunks[i] for i in chunk_ids]
    headings = list(dict.fromkeys(c.heading for c in chunks if c.heading))
    opening = ""
    for c in chunks:
        if len(opening) >= OPENING_CHARS:
            break
        opening += c.snippet + " "
    return chunks[0].title, headings[:MAX_HEADINGS], opening[:OPENING_CHARS].strip()


def pick_docs(
    docs: dict[str, list[int]], source: str, n: int, split: str, rng: random.Random
) -> list[str]:
    names = sorted(d for d in docs if split_of(source, d) == split)
    rng.shuffle(names)
    return names[:n]


def _meta(e: Example) -> dict:
    return {
        "id": e.id,
        "source": e.source,
        "doc": e.doc,
        "kind": e.kind,
        "question": e.question,
        "doc_retrieved": e.gold_retrieved,
        "teacher": TEACHER,
    }


def run_broad(
    corpus_root: Path,
    out_dir: Path,
    sources: list[str],
    docs_per_split: dict[str, int] = DOCS,
    *,
    seed: int = 0,
    client=None,
    model=None,
    poll: float = 60,
    sample: int | None = None,
) -> dict:
    """`sample` answers only that many random examples (a pilot)."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    rng = random.Random(seed)
    indexes = {s: SourceIndex.build(corpus_root / s, model=model) for s in sources}
    picked = []  # (source, doc, chunk ids)
    for s in sources:
        docs = documents(indexes[s])
        for split, n in docs_per_split.items():
            picked += [(s, d, docs[d]) for d in pick_docs(docs, s, n, split, rng)]

    q_reqs = {
        f"b{k:05d}": broad_question_params(*describe(indexes[s], ids))
        for k, (s, _, ids) in enumerate(picked)
    }
    q_res = run_batch(client, q_reqs, out_dir / "batches" / "questions", poll=poll)

    examples: list[Example] = []
    for k, (s, doc, ids) in enumerate(picked):
        r = q_res.get(f"b{k:05d}")
        if r is None:
            continue
        for j, q in enumerate(parse_broad(r["text"])):
            removed = rng.random() < REMOVED_RATE
            index = indexes[s]
            if removed:
                hits, found = index.top(q, exclude_post=doc), None
            else:
                hits = index.top(q)
                found = any(h.chunk.post == doc for h in hits)
                if not found:
                    scores = index.scores(q)
                    best = max(ids, key=lambda i: scores[i])
                    gold = Hit(best, float(scores[best]), index.chunks[best])
                    hits, _ = with_gold(hits, gold, rng)
            kind = "broad-removed" if removed else "broad"
            examples.append(
                Example(
                    f"{s}/{doc}/broad{j}",
                    s,
                    doc,
                    split_of(s, doc),
                    kind,
                    q,
                    [passage(h.chunk) for h in hits],
                    None,
                    found,
                )
            )

    keep = range(len(examples))
    if sample is not None:
        keep = sorted(random.Random(seed + 1).sample(keep, sample))
    examples = [examples[k] for k in keep]
    t_reqs = {
        f"t{k:05d}": teacher_params(TEACHER, e.messages(), BROAD_GUIDANCE)
        for k, e in zip(keep, examples)
    }
    t_res = run_batch(client, t_reqs, out_dir / "batches" / "teacher", poll=poll)

    splits: dict[str, list[tuple[Example, str]]] = {s: [] for s in SPLITS}
    dropped: Counter = Counter()
    for k, e in zip(keep, examples):
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
            [
                {"messages": [*e.messages(), {"role": "assistant", "content": a}]}
                for e, a in rows
            ],
        )
        write_jsonl(out_dir / f"{split}.meta.jsonl", [_meta(e) for e, _ in rows])

    def usd(res):
        return round(
            sum(
                cost(x["model"], x["input_tokens"], x["output_tokens"], batch=True)
                for x in res.values()
            ),
            2,
        )

    kept = [(e, a) for rows in splits.values() for e, a in rows]
    summary = {
        "examples": {s: len(rows) for s, rows in splits.items()},
        "kinds": dict(Counter(e.kind for e, _ in kept)),
        "refusals": dict(Counter(e.kind for e, a in kept if a == REFUSAL)),
        "doc_retrieved": sum(bool(e.gold_retrieved) for e, _ in kept),
        "dropped": dict(dropped),
        "usd_batch": {"questions": usd(q_res), "teacher": usd(t_res)},
        "teacher": TEACHER,
        "seed": seed,
        "docs_per_split": docs_per_split,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def combine(sources: list[Path], out: Path) -> dict[str, int]:
    """Concatenate datasets split by split (examples and meta), shuffled with a
    fixed seed so the broad examples don't all come last in training."""
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        rows = []
        for src in sources:
            ex = (src / f"{split}.jsonl").read_text().splitlines()
            meta = (src / f"{split}.meta.jsonl").read_text().splitlines()
            rows += list(zip(ex, meta, strict=True))
        random.Random(0).shuffle(rows)
        (out / f"{split}.jsonl").write_text("".join(e + "\n" for e, _ in rows))
        (out / f"{split}.meta.jsonl").write_text("".join(m + "\n" for _, m in rows))
        counts[split] = len(rows)
    return counts
