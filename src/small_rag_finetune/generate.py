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
    TEACHER_GUIDANCE,
    cost,
    parse_questions,
    question_params,
    teacher_params,
)
from .examples import Example, examples_for, pick_gold
from .index import SourceIndex
from .pilot import NEAR_MISS_RATE, REMOVED_RATE, write_jsonl
from .prompt import REFUSAL

TEACHER = "claude-opus-5-5"
QUOTAS = {"train": 322, "valid": 18, "test": 18}  # gold chunks per source
SPLITS = ("train", "valid", "test")
# "passage [2]", not code like x[0]
_PASSAGE_TALK = re.compile(r"\bpassages?\b|(?<![\w\]])\[\d\]", re.IGNORECASE)


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


def _meta(e: Example) -> dict:
    return {
        "id": e.id,
        "source": e.source,
        "doc": e.doc,
        "kind": e.kind,
        "question": e.question,
        "gold_rank": e.gold_rank,
        "gold_retrieved": e.gold_retrieved,
        "teacher": TEACHER,
    }


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
    guidance: str = TEACHER_GUIDANCE,
    sample: int | None = None,
) -> dict:
    """`guidance` is the teacher-only answer style. `sample` answers only a
    random subset of that many examples (a pilot of a new style)."""
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

    examples: list[Example] = []
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

    keep = range(len(examples))
    if sample is not None:
        keep = sorted(random.Random(seed + 1).sample(keep, sample))
    examples = [examples[k] for k in keep]
    t_reqs = {
        f"t{k:05d}": teacher_params(TEACHER, e.messages(), guidance)
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

    kept = [(e, a) for rows in splits.values() for e, a in rows]
    summary = {
        "examples": {s: len(rows) for s, rows in splits.items()},
        "kinds": dict(Counter(e.kind for e, _ in kept)),
        "refusals": sum(a == REFUSAL for _, a in kept),
        "dropped": dict(dropped),
        "usd_batch": {
            "questions": round(_usd(q_res), 2),
            "teacher": round(_usd(t_res), 2),
        },
        "teacher": TEACHER,
        "seed": seed,
        "quotas": quotas,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
