"""Reshape neural-bridge's open RAG datasets into the ask-my-blog task.

Each example is the blog's exact prompt (4 numbered passages + a question) and
a target that either answers in at most two sentences or refuses.

- answerable: the passage that holds the answer plus the 3 chunks from *other*
  documents that look most like the question (distractors), in retrieval order.
- refusal: the question's own document is removed, so the 4 passages are only
  look-alikes. neural-bridge's hallucination set adds questions whose context
  never answered them.
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .prompt import REFUSAL, TOP_K, build_messages

SOURCES = {
    "answerable": (
        "neural-bridge/rag-dataset-12000",
        "data/train-00000-of-00001-9df3a936e1f63191.parquet",
    ),
    "unanswerable": (
        "neural-bridge/rag-hallucination-dataset-1000",
        "data/train-00000-of-00001-44cee39b8c9485bf.parquet",
    ),
}
# The blog's search model (sss_eval); same retriever, same kind of distractors.
EMBED_MODEL = "minishlab/potion-retrieval-32M"
CHUNK_CHARS = 1100
_SENT_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+")


@dataclass
class Chunk:
    doc: int
    title: str
    text: str


def load(kind: str) -> list[dict]:
    repo, path = SOURCES[kind]
    rows = pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()
    return [r for r in rows if r["context"] and r["question"] and r["answer"]]


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Greedy paragraph packing; a paragraph longer than `size` is split on sentences."""
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        pieces.extend(_SENT_END.split(para) if len(para) > size else [para])
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) > size:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def title_of(context: str) -> str:
    first = context.strip().split("\n", 1)[0]
    return first if len(first) <= 70 else first[:67].rsplit(" ", 1)[0] + "…"


def clamp_sentences(text: str, n: int = 2) -> str:
    return " ".join(_SENT_END.split(text.strip())[:n])


def _overlap(a: str, b: str) -> float:
    wa = set(_WORD.findall(a.lower()))
    return len(wa & set(_WORD.findall(b.lower()))) / (len(wa) or 1)


def _passage(c: Chunk) -> dict:
    return {"title": c.title, "heading": None, "snippet": c.text}


def build(n_answerable: int, n_refusal: int, seed: int = 0) -> list[dict]:
    from model2vec import StaticModel

    rng = random.Random(seed)
    answerable = load("answerable")
    unanswerable = load("unanswerable")

    chunks: list[Chunk] = []
    for doc, row in enumerate(answerable + unanswerable):
        t = title_of(row["context"])
        chunks.extend(Chunk(doc, t, c) for c in chunk_text(row["context"]))
    doc_of = np.array([c.doc for c in chunks])

    model = StaticModel.from_pretrained(EMBED_MODEL)
    emb = model.encode([c.text for c in chunks], normalize=True)

    def scores_for(question: str, exclude_doc: int | None = None) -> np.ndarray:
        s = emb @ model.encode([question], normalize=True)[0]
        if exclude_doc is not None:
            s = np.where(doc_of == exclude_doc, -np.inf, s)
        return s

    def by_score(ids, s) -> list[Chunk]:
        return [chunks[i] for i in sorted(ids, key=lambda i: -s[i])]

    n_removed = n_refusal - n_refusal // 2
    n_unanswerable = n_refusal // 2
    ans_ids = rng.sample(range(len(answerable)), n_answerable + n_removed)
    out: list[dict] = []

    # Answerable: gold = own chunk sharing most words with the answer, plus the
    # 3 best chunks from other documents; all 4 in retrieval order.
    for doc in ans_ids[:n_answerable]:
        row = answerable[doc]
        gold = max(
            np.flatnonzero(doc_of == doc),
            key=lambda i: _overlap(row["answer"], chunks[i].text),
        )
        s_all, s_other = scores_for(row["question"]), scores_for(row["question"], doc)
        others = np.argsort(-s_other)[: TOP_K - 1]
        out.append(
            _example(
                row["question"],
                by_score([gold, *others], s_all),
                clamp_sentences(row["answer"]),
                "answerable",
            )
        )

    # Refusal by removal: same kind of question, its own document excluded.
    for doc in ans_ids[n_answerable:]:
        row = answerable[doc]
        s = scores_for(row["question"], doc)
        out.append(
            _example(
                row["question"],
                by_score(np.argsort(-s)[:TOP_K], s),
                REFUSAL,
                "refusal-removed",
            )
        )

    # Refusal from the hallucination set: its own context never answers it.
    for j in rng.sample(range(len(unanswerable)), n_unanswerable):
        row = unanswerable[j]
        s = scores_for(row["question"])
        out.append(
            _example(
                row["question"],
                by_score(np.argsort(-s)[:TOP_K], s),
                REFUSAL,
                "refusal-unanswerable",
            )
        )

    rng.shuffle(out)
    return out


def _example(question: str, picked: list[Chunk], target: str, kind: str) -> dict:
    msgs = build_messages(question, [_passage(c) for c in picked])
    return {"messages": [*msgs, {"role": "assistant", "content": target}], "kind": kind}


def write_splits(examples: list[dict], out_dir: Path, n_valid: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"valid": examples[:n_valid], "train": examples[n_valid:]}
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(
                    json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n"
                )
        with open(out_dir / f"{name}.kinds.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"kind": r["kind"]}) + "\n")
