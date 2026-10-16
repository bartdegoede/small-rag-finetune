"""Remove partial-answer hedges ("The blog does not go into ...") from teacher
answers. Opus adds one when something really is missing from the passages; a
0.6B model trained on them learns to append the clause anyway, often wrongly
(v1 results). Full refusals stay: they are the behaviour we want."""

import json
import re
from pathlib import Path

REFUSAL = "The blog does not seem to cover this."
_NEG = r"(?:does not|doesn['’]t|did not|didn['’]t)"
_VERB = (
    r"(?:explain|go into|say|show|give|describe|detail|specify|mention|state|list|name|"
    r"spell out|discuss|"
    r"cover (?:the|how|why|what|which|whether|any))"
)
# "The blog does not <anything>" is always a hedge in a teacher answer (the
# refusal is handled first); "it does not" only with the verbs above, since
# "it doesn't install the toolchain" is an answer.
_HEDGE_AT = (
    rf"(?:the blog|the post|the passages?) (?:also )?{_NEG} \w"
    rf"|the (?:blog|post) (?:gives|offers|provides) (?:no|only)\b"
    rf"|it (?:also )?{_NEG} (?:\w+ )?{_VERB}"
)
_HEDGE = re.compile(rf"\b(?:{_HEDGE_AT})", re.IGNORECASE)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(*])")
_CLAUSE = re.compile(
    rf"\s*[,;:]?\s*(?:(?:but |and |though |although |while )?(?:{_HEDGE_AT})"
    rf"|but {_NEG} {_VERB}).*$",
    re.IGNORECASE,
)


def strip_hedge(answer: str) -> str | None:
    """The answer without hedges; None when the hedge is the answer's lead."""
    if answer.strip() == REFUSAL:
        return answer
    sents = _SENT.split(answer.strip())
    if _HEDGE.match(sents[0]):
        return None
    kept = []
    for s in sents:
        if _HEDGE.match(s):
            continue
        cut = _CLAUSE.sub("", s)
        if cut != s:
            s = cut.rstrip(" ,;:") + "."
        if _HEDGE.search(s):  # a hedge we can't cut cleanly
            return None
        kept.append(s)
    return " ".join(kept) if kept else None


def strip_split(src: Path, out: Path, split: str) -> dict[str, int]:
    """Rewrite <src>/<split>.jsonl and its .meta.jsonl into <out>."""
    out.mkdir(parents=True, exist_ok=True)
    rows = (src / f"{split}.jsonl").read_text().splitlines()
    metas = (src / f"{split}.meta.jsonl").read_text().splitlines()
    counts = {"kept": 0, "stripped": 0, "dropped": 0}
    with (
        (out / f"{split}.jsonl").open("w") as f,
        (out / f"{split}.meta.jsonl").open("w") as m,
    ):
        for row, meta in zip(rows, metas, strict=True):
            ex = json.loads(row)
            answer = ex["messages"][-1]["content"]
            new = strip_hedge(answer)
            if new is None:
                counts["dropped"] += 1
                continue
            counts["kept" if new == answer else "stripped"] += 1
            ex["messages"][-1]["content"] = new
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            m.write(meta + "\n")
    return counts
