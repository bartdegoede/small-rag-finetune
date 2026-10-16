"""Pick checkpoints on judged answers, not validation loss.

v2 showed validation loss falling while the blog eval got worse: the loss
measures imitation of the teacher, not faithfulness. So each checkpoint answers
the dataset's held-out test split (documents no checkpoint trained on, so the
blog's 45 questions stay a final, untouched check), and Opus grades every answer
against its passages with the blog judge's rubric, through the Batch API.

    small-rag-finetune dev-prompts data/v3 runs/dev/prompts.jsonl
    node js/generate.mjs runs/onnx/<arm> runs/dev/prompts.jsonl > runs/dev/<arm>.jsonl
    small-rag-finetune dev-judge runs/dev/<arm>.jsonl
"""

import json
from collections import Counter
from pathlib import Path

from .batch import run_batch
from .claude import cost
from .prompt import REFUSAL

JUDGE = "claude-opus-5-5"
# The blog's scripts/ask_eval/judge.py INSTRUCTIONS (n=2), verbatim.
RUBRIC = """You are grading a short answer written by a small language model.
The model was given numbered passages from a blog and a question, and told to answer
in at most 2 sentences using only the passages.

Label the answer:
- "faithful": every factual claim in the answer is supported by the passages, and it
  addresses the question (saying the blog does not cover it counts as faithful ONLY if
  the passages really do not answer the question).
- "unsupported": at least one claim is not supported by the passages (invented,
  contradicted, or from outside knowledge), even if it happens to be true.
- "non-answer": no unsupported claims, but it does not answer the question although the
  passages do (declines wrongly, rambles, repeats the question, empty).

An answer that is cut off mid-sentence is judged on what it says, not penalised for the
cut-off itself. Judge only against the passages, not your own knowledge of the topic."""
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["faithful", "unsupported", "non-answer"]},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["label", "unsupported_claims", "reason"],
    "additionalProperties": False,
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def dev_prompts(data_dir: Path, split: str = "test") -> list[dict]:
    """One row per held-out example: the student's system + user messages, and
    whether the teacher refused (the expected behaviour)."""
    examples = _rows(data_dir / f"{split}.jsonl")
    metas = _rows(data_dir / f"{split}.meta.jsonl")
    return [
        {
            "id": m["id"],
            "messages": ex["messages"][:2],
            "teacher_refused": ex["messages"][2]["content"] == REFUSAL,
        }
        for ex, m in zip(examples, metas, strict=True)
    ]


def judge_params(prompt: dict, summary: str) -> dict:
    user = prompt["messages"][1]["content"]
    content = f"{RUBRIC}\n\n<prompt>\n{user}\n</prompt>\n\n<answer>{summary}</answer>"
    return {
        "model": JUDGE,
        "max_tokens": 4000,
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
        },
        "messages": [{"role": "user", "content": content}],
    }


def tally(prompts: list[dict], verdicts: dict[str, dict]) -> dict:
    """Faithful share on examples the teacher answered, and on those it refused."""
    by_id = {p["id"]: p for p in prompts}
    out: dict[str, Counter] = {"answered": Counter(), "refused": Counter()}
    for pid, v in verdicts.items():
        group = "refused" if by_id[pid]["teacher_refused"] else "answered"
        out[group][v["label"]] += 1
    return {
        g: {
            "n": sum(c.values()),
            **dict(c),
            "faithful_rate": c["faithful"] / max(1, sum(c.values())),
        }
        for g, c in out.items()
    }


def run_dev_judge(
    client, prompts: list[dict], answers: Path, out_dir: Path, *, poll: float = 60
) -> dict:
    """Judge one arm's answers (generate.mjs output); resumable per arm."""
    rows = _rows(answers)
    arm = rows[0]["arm"]
    by_id = {p["id"]: p for p in prompts}
    # custom_id must match ^[a-zA-Z0-9_-]{1,64}$; example ids contain "/".
    keyed = {f"j{k:04d}": r for k, r in enumerate(rows)}
    reqs = {cid: judge_params(by_id[r["id"]], r["summary"]) for cid, r in keyed.items()}
    res = run_batch(client, reqs, out_dir / arm, poll=poll)
    verdicts = {keyed[cid]["id"]: json.loads(r["text"]) for cid, r in res.items()}
    with (out_dir / arm / "verdicts.jsonl").open("w") as f:
        for r in keyed.values():
            if r["id"] in verdicts:
                f.write(
                    json.dumps({"arm": arm, "id": r["id"], **verdicts[r["id"]]}) + "\n"
                )
    usd = sum(
        cost(r["model"], r["input_tokens"], r["output_tokens"], batch=True)
        for r in res.values()
    )
    return {
        "arm": arm,
        "judged": len(verdicts),
        "usd_batch": round(usd, 2),
        **tally(prompts, verdicts),
    }
