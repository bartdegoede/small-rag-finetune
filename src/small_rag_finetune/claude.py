"""Everything that talks to Claude: prompts, request params, parsing, cost.

Requests are plain dicts so the same params work for messages.create (pilot)
and for Message Batches (plan 2). Replies are plain text with a fixed line
format; no structured-output features needed.
"""

import re
from dataclasses import dataclass

from .prompt import REFUSAL

QUESTION_MODEL = "claude-sonnet-5-5"
TEACHERS = ("claude-opus-5-5", "claude-sonnet-5-5")
# USD per 1M tokens (input, output), standard rates.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
BATCH_DISCOUNT = 0.5

QUESTION_PROMPT = """You are writing questions for a site-search assistant that answers from retrieved passages.

<title>{title}</title>
<section>{heading}</section>
<passage>
{passage}
</passage>

Write 3 questions that a reader of this site might type into its search box and that THIS passage answers on its own.
- Each question must make sense without seeing the passage: name the subject (the project, feature, release, PEP or RFC) instead of writing "this" or "the passage".
- The answer must be stated in the passage, not inferred from general knowledge.
- Mix styles: a fact lookup, a how or why question, and one that needs a detail from the middle or end of the passage.
- Under 20 words each, plain wording; do not copy more than three words in a row from the passage.
- If the passage cannot support a question (for example it is only a list of names or links), write NONE on that line.

Then write one NEAR-MISS question: on the same subject, something a reader could plausibly ask, that this passage does NOT answer, not even partly. Write NONE if you cannot.

Reply with exactly these four lines and nothing else:
Q1: ...
Q2: ...
Q3: ...
NEAR: ..."""

TEACHER_GUIDANCE = """

How to answer:
- Answer the question directly in plain declarative sentences. Never mention the passages, passage numbers like [2], or phrases such as "according to".
- Use only facts stated in the passages. Prefer one sentence; use a second only when the question needs it.
- If they answer only part of the question, give that part; you may add a short clause that the blog does not go into more detail.
- If the passages do not answer the question at all, reply with exactly: The blog does not seem to cover this."""

_Q = re.compile(r"^Q\d+:\s*(.+?)\s*$", re.MULTILINE)
_NEAR = re.compile(r"^NEAR:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Questions:
    answerable: list[str]
    near_miss: str | None


@dataclass(frozen=True)
class Call:
    role: str  # "questions" | "teacher"
    model: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens, which are billed as output


def question_params(chunk) -> dict:
    content = QUESTION_PROMPT.format(
        title=chunk.title, heading=chunk.heading or "(none)", passage=chunk.snippet
    )
    return {
        "model": QUESTION_MODEL,
        "max_tokens": 4000,
        "output_config": {"effort": "medium"},
        "messages": [{"role": "user", "content": content}],
    }


def teacher_params(model: str, messages: list[dict]) -> dict:
    """The student's exact system + user messages; the guidance is teacher-only."""
    system, user = messages
    return {
        "model": model,
        "max_tokens": 4000,
        "output_config": {"effort": "low"},
        "system": system["content"] + TEACHER_GUIDANCE,
        "messages": [{"role": "user", "content": user["content"]}],
    }


def complete(client, params: dict):
    msg = client.messages.create(**params)
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return text, msg.usage


def parse_questions(text: str) -> Questions:
    answerable = [q for q in _Q.findall(text) if q.upper() != "NONE"]
    m = _NEAR.search(text)
    near = m.group(1) if m and m.group(1).upper() != "NONE" else None
    return Questions(answerable, near)


def normalize_answer(text: str) -> str:
    t = " ".join(text.split())
    if t.lower().startswith("the blog does not seem to cover"):
        return REFUSAL
    return t


def cost(
    model: str, input_tokens: int, output_tokens: int, batch: bool = False
) -> float:
    pin, pout = PRICES[model]
    usd = input_tokens / 1e6 * pin + output_tokens / 1e6 * pout
    return usd * BATCH_DISCOUNT if batch else usd


def target_key(model: str) -> str:
    """claude-opus-5-5 -> target_opus"""
    return "target_" + model.removeprefix("claude-").split("-")[0]
