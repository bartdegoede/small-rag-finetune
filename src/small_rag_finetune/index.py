"""The blog's retrieval, one index per source.

Same chunker and settings as the blog's scripts/build_search_index.py
(sss_eval chunk_post, 600 chars, 120 overlap, title prefix) and the same
model (potion-base-8M truncated to 128 dims). The browser ships an int8 copy
of these vectors; float32 here ranks the same up to near-ties.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sss_eval.chunk import Chunk, chunk_post
from sss_eval.corpus import load_corpus

from .prompt import TOP_K

MODEL_ID = "minishlab/potion-base-8M"
DIMS = 128
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class Hit:
    index: int
    score: float
    chunk: Chunk


class SourceIndex:
    def __init__(self, chunks: list[Chunk], model) -> None:
        self.chunks = chunks
        self.model = model
        self.post = np.array([c.post for c in chunks])
        self.emb = self._encode([c.text for c in chunks])

    @classmethod
    def build(cls, corpus_dir: Path, model=None) -> "SourceIndex":
        if model is None:
            from model2vec import StaticModel

            model = StaticModel.from_pretrained(MODEL_ID, dimensionality=DIMS)
        chunks = [
            c
            for post in load_corpus(corpus_dir)
            for c in chunk_post(
                post, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, title_prefix=True
            )
        ]
        return cls(chunks, model)

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self.model.encode(texts), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def scores(self, query: str, exclude_post: str | None = None) -> np.ndarray:
        s = self.emb @ self._encode([query])[0]
        if exclude_post is not None:
            s = np.where(self.post == exclude_post, -np.inf, s)
        return s

    def top(
        self, query: str, k: int = TOP_K, exclude_post: str | None = None
    ) -> list[Hit]:
        """Best first; ties broken by chunk order, like retrieve.js topChunks."""
        s = self.scores(query, exclude_post)
        order = np.lexsort((np.arange(len(s)), -s))[:k]
        return [
            Hit(int(i), float(s[i]), self.chunks[i]) for i in order if np.isfinite(s[i])
        ]
