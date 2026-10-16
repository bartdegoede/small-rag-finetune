"""prompt.py must produce byte-identical messages to the blog's prompt.js."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from small_rag_finetune.prompt import build_messages

PROMPT_JS = Path.home() / "projects/blog/assets/js/ask/prompt.js"

CASES = [
    (
        "How did the blog add search?",
        [
            {"title": "Post A", "heading": "Setup", "snippet": "x" * 1500},
            {"title": "Post B", "heading": None, "snippet": "Short one."},
            {"title": "Post C", "heading": "", "snippet": "y" * 3000},
            {"title": "Post D", "heading": "Tail", "snippet": "z" * 900},
        ],
    ),
    ("q?", [{"title": "T", "heading": "H", "snippet": "café — naïve"}]),
]


@pytest.mark.skipif(
    not PROMPT_JS.exists() or not shutil.which("node"),
    reason="needs the blog checkout and node",
)
@pytest.mark.parametrize("question,passages", CASES)
def test_matches_prompt_js(question, passages):
    js = (
        f"import {{buildMessages}} from {json.dumps(str(PROMPT_JS))};"
        f"console.log(JSON.stringify(buildMessages({json.dumps(question)},"
        f"{json.dumps(passages)})))"
    )
    out = subprocess.run(
        ["node", "--input-type=module", "-e", js],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert build_messages(question, passages) == json.loads(out)
