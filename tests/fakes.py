"""Stand-ins for the embedding model and the Anthropic client."""

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class FakeModel:
    """Bag-of-words over a tiny vocabulary: enough like model2vec for ranking tests."""

    VOCAB = (
        "go",
        "rust",
        "python",
        "generics",
        "async",
        "walrus",
        "lifetimes",
        "modules",
    )

    def encode(self, texts):
        return np.array(
            [[t.lower().count(w) + 0.01 for w in self.VOCAB] for t in texts],
            dtype=np.float32,
        )


def write_post(root: Path, slug: str, title: str, body: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{slug}.md").write_text(
        f"---\ntitle: {title}\nslug: {slug}\n---\n\n{body}\n"
    )


def small_corpus(root: Path) -> None:
    write_post(
        root, "generics", "Go generics", "Go generics add type parameters. " * 12
    )
    write_post(
        root, "async", "Async Rust", "Async rust uses futures and executors. " * 12
    )
    write_post(
        root, "walrus", "The walrus", "The python walrus operator assigns. " * 12
    )
    write_post(root, "lifetimes", "Lifetimes", "Rust lifetimes track borrows. " * 12)


VERDICT = '{"label": "faithful", "unsupported_claims": [], "reason": "ok"}'


def reply_for(params: dict) -> str:
    if "format" in params.get("output_config", {}):
        return VERDICT
    return FakeClient.ANSWER if "system" in params else FakeClient.QUESTIONS


class FakeBatches:
    """client.messages.batches: every batch ends at once; ids in `fail` error once."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.batches: dict[str, list[dict]] = {}
        self.created: list[int] = []

    def create(self, requests):
        bid = f"batch_{len(self.batches)}"
        self.batches[bid] = list(requests)
        self.created.append(len(requests))
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended")

    def results(self, bid):
        for r in self.batches[bid]:
            cid = r["custom_id"]
            if cid in self.fail:
                self.fail.discard(cid)
                result = SimpleNamespace(type="errored")
                yield SimpleNamespace(custom_id=cid, result=result)
                continue
            msg = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=reply_for(r["params"]))],
                usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
                model=r["params"]["model"],
            )
            result = SimpleNamespace(type="succeeded", message=msg)
            yield SimpleNamespace(custom_id=cid, result=result)


class FakeClient:
    """client.messages.create(**params) and client.messages.batches, with canned
    replies and fixed usage."""

    QUESTIONS = (
        "Q1: What do go generics add?\nQ2: How are go generics written?\n"
        "Q3: NONE\nNEAR: When will rust get generics?"
    )
    ANSWER = "Go generics add type parameters to functions and types."

    def __init__(self):
        self.messages = self
        self.calls = []
        self.batches = FakeBatches()
        self._lock = threading.Lock()

    def create(self, **params):
        with self._lock:
            self.calls.append(params)
        text = reply_for(params)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=text),
            ],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
            stop_reason="end_turn",
        )
