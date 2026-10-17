"""Evaluate a model on your own site: your posts, your questions.

Write a few dozen questions about your site in a JSONL file, one per line:

    {"id": "q01", "question": "How did I add search to the blog?"}
    {"id": "n01", "question": "Which GPU should I buy?", "covered": false}

`covered: false` marks a question the site doesn't answer (the right answer is
the refusal); it defaults to true. `eval-prompts` retrieves the top 4 passages
for each question from a folder of markdown posts, with the same chunking and
embedding model the training data used, and writes prompts that
`js/generate.mjs` answers and `dev-judge` grades.
"""

import json
from pathlib import Path

from .examples import passage
from .index import SourceIndex
from .prompt import build_messages


def read_questions(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: question ids must be unique")
    return rows


def site_prompts(index: SourceIndex, questions: list[dict]) -> list[dict]:
    out = []
    for q in questions:
        hits = index.top(q["question"])
        covered = q.get("covered", True)
        out.append(
            {
                "id": q["id"],
                "question": q["question"],
                "covered": covered,
                # dev-judge's name for "the right answer is a refusal".
                "teacher_refused": not covered,
                "best_score": round(hits[0].score, 4) if hits else None,
                "posts": [h.chunk.post for h in hits],
                "messages": build_messages(
                    q["question"], [passage(h.chunk) for h in hits]
                ),
            }
        )
    return out
