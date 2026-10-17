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

# v3: answers a 0.6B model can imitate. v1/v2 targets combined several facts in
# dense sentences; the student learned the density, not the reading, and got
# the details wrong (docs/superpowers/specs/2026-10-16-v1-results.md).
EXTRACTIVE_GUIDANCE = """

How to answer:
- Answer in one short sentence that states the fact the question asks for. Use a second short sentence only if the question asks for two things.
- Reuse the wording of the passage that answers it; do not paraphrase, merge, or add facts from other passages.
- Never mention the passages, passage numbers like [2], or phrases such as "according to".
- If the passages answer only part of the question, answer that part and stop; do not say what is missing.
- If the passages do not answer the question at all, reply with exactly: The blog does not seem to cover this."""

# v4: broad questions about a whole document. The blog eval asks "What has the
# blog written about X?" and goal questions ("How can I make an audio version of
# a blog post?"); v1-v3 only had specific facts from one chunk, and v3 refused
# most broad questions.
BROAD_PROMPT = """You are writing questions for a site-search assistant that answers from retrieved passages.

<title>{title}</title>
<sections>
{headings}
</sections>
<opening>
{opening}
</opening>

Write 2 broad questions a reader might type into the search box of the site this document is on, that this document as a whole answers.
- TOPIC: a question about what the site has written on the document's subject, e.g. "What has the blog written about X?" or "What does the blog say about X?". Name the subject in a reader's words.
- GOAL: a "How can I ..." or "How do I ..." question describing the reader's problem in plain words, without naming the project, tool, PEP or RFC that solves it.
- Under 16 words each.

Reply with exactly these two lines and nothing else:
TOPIC: ...
GOAL: ..."""

BROAD_GUIDANCE = """

How to answer:
- The question is broad. Say in one or two short sentences what the passages cover on it, reusing their wording: for "what has the blog written about X", name what the posts explain; for "how can I ...", give the approach the passages describe.
- Use only facts stated in the passages; do not merge unrelated passages or add outside knowledge.
- Never mention the passages, passage numbers like [2], or phrases such as "according to".
- Do not say what is missing.
- If none of the passages is about the question's subject, reply with exactly: The blog does not seem to cover this."""

_TOPIC = re.compile(r"^TOPIC:\s*(.+?)\s*$", re.MULTILINE)
_GOAL = re.compile(r"^GOAL:\s*(.+?)\s*$", re.MULTILINE)


def broad_question_params(title: str, headings: list[str], opening: str) -> dict:
    content = BROAD_PROMPT.format(
        title=title,
        headings="\n".join(headings) or "(none)",
        opening=opening,
    )
    return {
        "model": QUESTION_MODEL,
        "max_tokens": 2000,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": content}],
    }


def parse_broad(text: str) -> list[str]:
    """[topic, goal], skipping missing or NONE lines."""
    out = []
    for rx in (_TOPIC, _GOAL):
        m = rx.search(text)
        if m and m.group(1).upper() != "NONE":
            out.append(m.group(1))
    return out


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


def teacher_params(
    model: str, messages: list[dict], guidance: str = TEACHER_GUIDANCE
) -> dict:
    """The student's exact system + user messages; the guidance is teacher-only."""
    system, user = messages
    return {
        "model": model,
        "max_tokens": 4000,
        "output_config": {"effort": "low"},
        "system": system["content"] + guidance,
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
