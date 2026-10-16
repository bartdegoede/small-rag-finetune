"""Python port of the blog's assets/js/ask/prompt.js.

Training has to see exactly the prompt the browser sends, so this mirrors
systemPrompt() and buildMessages() character for character.
"""

TOP_K = 4
MAX_SENTENCES = 2
MAX_PASSAGE_CHARS = 4800
MIN_PASSAGE_CHARS = 100

REFUSAL = "The blog does not seem to cover this."


def system_prompt(n: int = MAX_SENTENCES) -> str:
    return (
        "You answer questions about a personal technical blog. "
        "Use only the numbered passages provided. "
        f"Answer in at most {n} sentences. "
        "If the passages do not answer the question, "
        "say that the blog does not seem to cover it."
    )


def _label(p: dict, i: int) -> str:
    heading = f" — {p['heading']}" if p.get("heading") else ""
    return f"[{i + 1}] {p['title']}{heading}"


def build_messages(question: str, passages: list[dict]) -> list[dict]:
    """passages: [{title, heading, snippet}], best first."""
    parts: list[str] = []
    used = 0
    for i, p in enumerate(passages):
        sep = 2 if parts else 0
        head = _label(p, i)
        room = MAX_PASSAGE_CHARS - used - sep - len(head) - 1
        if room < MIN_PASSAGE_CHARS:
            break
        body = p["snippet"][:room]
        parts.append(f"{head}\n{body}")
        used += sep + len(head) + 1 + len(body)
    user = "Passages:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user},
    ]
