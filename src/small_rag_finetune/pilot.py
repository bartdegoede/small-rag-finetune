"""The pilot: a few documents per source through the whole pipeline, both
teachers on every example, and a report with measured cost per example."""

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from sss_eval.chunk import split_sentences

from .claude import (
    BATCH_DISCOUNT,
    QUESTION_MODEL,
    TEACHERS,
    Call,
    complete,
    cost,
    normalize_answer,
    parse_questions,
    question_params,
    target_key,
    teacher_params,
)
from .examples import examples_for, pick_gold
from .index import SourceIndex
from .prompt import REFUSAL

REMOVED_RATE = 0.1  # share of answerable questions turned into "removed"
NEAR_MISS_RATE = 0.5  # share of gold chunks that also get their near-miss question
TARGET_EXAMPLES = 5000


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def run_pilot(
    corpus_root: Path,
    out_dir: Path,
    sources: list[str],
    per_source: int,
    *,
    seed: int = 0,
    client=None,
    model=None,
    workers: int = 8,
) -> tuple[list[dict], list[Call]]:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    rng = random.Random(seed)
    calls: list[Call] = []
    examples = []
    for name in sources:
        index = SourceIndex.build(corpus_root / name, model=model)
        for gi in pick_gold(index, name, per_source, rng):
            text, usage = complete(client, question_params(index.chunks[gi]))
            calls.append(
                Call(
                    "questions", QUESTION_MODEL, usage.input_tokens, usage.output_tokens
                )
            )
            qs = parse_questions(text)
            removed = {
                j for j in range(len(qs.answerable)) if rng.random() < REMOVED_RATE
            }
            near = rng.random() < NEAR_MISS_RATE
            examples += examples_for(
                index, name, gi, qs, use_near_miss=near, removed=removed, rng=rng
            )

    def teach(job):
        ex, m = job
        text, usage = complete(client, teacher_params(m, ex.messages()))
        call = Call("teacher", m, usage.input_tokens, usage.output_tokens)
        return ex.id, m, normalize_answer(text), call

    targets: dict[str, dict] = defaultdict(dict)
    jobs = [(e, m) for e in examples for m in TEACHERS]
    with ThreadPoolExecutor(workers) as pool:
        for ex_id, m, answer, call in pool.map(teach, jobs):
            targets[ex_id][target_key(m)] = answer
            calls.append(call)

    rows = [{**e.to_dict(), **targets[e.id]} for e in examples]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "examples.jsonl", rows)
    write_jsonl(out_dir / "calls.jsonl", [asdict(c) for c in calls])
    (out_dir / "report.md").write_text(pilot_report(rows, calls))
    return rows, calls


def _refused(row: dict, model: str) -> bool:
    return row[target_key(model)] == REFUSAL


def _frac(rows: list[dict], pred) -> str:
    return f"{sum(1 for r in rows if pred(r))}/{len(rows)}"


def _teacher_line(rows: list[dict], model: str) -> str:
    def of(kind):
        return [r for r in rows if r["kind"] == kind]

    def answered(r):
        return not _refused(r, model)

    def refused(r):
        return _refused(r, model)

    answers = [r[target_key(model)] for r in rows if answered(r)]
    over = sum(len(split_sentences(a)) > 2 for a in answers)
    mean = sum(map(len, answers)) / max(len(answers), 1)
    return (
        f"| {model} | {_frac(of('answerable'), refused)} "
        f"| {_frac(of('removed'), answered)} | {_frac(of('near-miss'), answered)} "
        f"| {over}/{len(answers)} | {mean:.0f} |"
    )


def _review_samples(rows: list[dict]) -> list[dict]:
    picks = {"answerable": 6, "removed": 2, "near-miss": 2}
    return [
        r
        for kind, n in picks.items()
        for r in [x for x in rows if x["kind"] == kind][:n]
    ]


def pilot_report(rows: list[dict], calls: list[Call]) -> str:
    kinds = Counter(r["kind"] for r in rows)
    per_kind = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    out = ["# Pilot report", "", f"Examples: {len(rows)} ({per_kind})"]
    ans = [r for r in rows if r["kind"] == "answerable"]
    if ans:
        out.append(
            f"Gold chunk retrieved unaided: {_frac(ans, lambda r: r['gold_retrieved'])}; "
            f"gold at rank 1: {_frac(ans, lambda r: r['gold_rank'] == 1)}"
        )

    out += [
        "",
        "## Teachers",
        "",
        (
            "| teacher | answerable refused | removed answered | near-miss answered "
            "| answers over 2 sentences | mean answer chars |"
        ),
        "|---|---|---|---|---|---|",
    ]
    out += [_teacher_line(rows, m) for m in TEACHERS]
    agree = sum(_refused(r, TEACHERS[0]) == _refused(r, TEACHERS[1]) for r in rows)
    out += ["", f"Teachers agree on answer vs refuse: {agree}/{len(rows)}"]

    out += [
        "",
        "## Cost (pilot, standard rates)",
        "",
        "| role | model | calls | input tokens | output tokens | USD |",
        "|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str], list[Call]] = defaultdict(list)
    for c in calls:
        groups[(c.role, c.model)].append(c)
    for (role, model), cs in sorted(groups.items()):
        i = sum(c.input_tokens for c in cs)
        o = sum(c.output_tokens for c in cs)
        out.append(
            f"| {role} | {model} | {len(cs)} | {i:,} | {o:,} | {cost(model, i, o):.3f} |"
        )

    def spent(role, model=None):
        return sum(
            cost(c.model, c.input_tokens, c.output_tokens)
            for c in calls
            if c.role == role and (model is None or c.model == model)
        )

    q_cost = spent("questions")
    out += [
        "",
        f"Projected for {TARGET_EXAMPLES:,} examples with the Batch API (50% off):",
        "",
    ]
    for model in TEACHERS:
        per = (q_cost + spent("teacher", model)) / max(len(rows), 1)
        total = per * TARGET_EXAMPLES * BATCH_DISCOUNT
        out.append(
            f"- questions + teacher {model}: ${per:.4f} per example "
            f"at standard rates -> ${total:.2f}"
        )

    out += ["", "## Examples for review", ""]
    for r in _review_samples(rows):
        out += [f"### {r['kind']}: {r['question']}", ""]
        for i, p in enumerate(r["passages"], 1):
            mark = " (gold)" if r["gold_rank"] == i else ""
            heading = f" — {p['heading']}" if p["heading"] else ""
            out.append(f"{i}. {p['title']}{heading}{mark}")
        if r["gold_rank"]:
            out += ["", "> " + r["passages"][r["gold_rank"] - 1]["snippet"]]
        out.append("")
        out += [f"- **{m}**: {r[target_key(m)]}" for m in TEACHERS]
        out.append("")
    return "\n".join(out) + "\n"
