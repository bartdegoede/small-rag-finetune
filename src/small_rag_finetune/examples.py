"""From a gold chunk and Claude's questions to the exact prompts the student sees.

- answerable: retrieval for the question; if the gold chunk isn't in the top 4
  it replaces the 4th at a random position, so the answer can be in any passage.
- removed: same question, every chunk of the gold document excluded.
- near-miss: Claude's related question the gold chunk doesn't answer.
"""

import random
from collections import defaultdict
from dataclasses import asdict, dataclass

from .claude import Questions
from .corpus import split_of
from .index import Hit, SourceIndex
from .prompt import TOP_K, build_messages

MIN_SNIPPET = 300  # gold chunks shorter than this rarely support 3 questions


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    doc: str
    split: str
    kind: str  # answerable | removed | near-miss
    question: str
    passages: list[dict]
    gold_rank: int | None  # 1-based position of the gold chunk, if present
    gold_retrieved: bool | None  # answerable only: retrieval found it unaided

    def messages(self) -> list[dict]:
        return build_messages(self.question, self.passages)

    def to_dict(self) -> dict:
        return {**asdict(self), "messages": self.messages()}


def passage(chunk) -> dict:
    return {"title": chunk.title, "heading": chunk.heading, "snippet": chunk.snippet}


def with_gold(hits: list[Hit], gold: Hit, rng: random.Random) -> tuple[list[Hit], bool]:
    """If retrieval missed the gold chunk, it replaces the 4th hit at a random
    position: keeping its (low) score order would put the answer last almost
    every time (pilot: 37/54 in passage [4])."""
    if any(h.index == gold.index for h in hits):
        return hits, True
    kept = list(hits[: TOP_K - 1])
    kept.insert(rng.randrange(len(kept) + 1), gold)
    return kept, False


def pick_gold(
    index: SourceIndex,
    source: str,
    n: int,
    rng: random.Random,
    split: str | None = "train",
) -> list[int]:
    """n distinct gold chunks from one split's documents: one per document
    first (documents in random order), then second chunks, and so on."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(index.chunks):
        in_split = split is None or split_of(source, c.post) == split
        if len(c.snippet) >= MIN_SNIPPET and in_split:
            by_doc[c.post].append(i)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for d in docs:
        rng.shuffle(by_doc[d])
    out: list[int] = []
    for depth in range(max(map(len, by_doc.values()), default=0)):
        for d in docs:
            if depth < len(by_doc[d]):
                out.append(by_doc[d][depth])
                if len(out) == n:
                    return out
    return out


def examples_for(
    index: SourceIndex,
    source: str,
    gold_index: int,
    qs: Questions,
    *,
    use_near_miss: bool,
    removed: set[int],
    rng: random.Random,
) -> list[Example]:
    gold_chunk = index.chunks[gold_index]
    doc = gold_chunk.post
    split = split_of(source, doc)
    base = f"{source}/{doc}/{gold_index}"

    def make(tag, kind, question, hits, rank, found) -> Example:
        passages = [passage(h.chunk) for h in hits]
        return Example(
            f"{base}/{tag}", source, doc, split, kind, question, passages, rank, found
        )

    out = []
    for j, q in enumerate(qs.answerable):
        if j in removed:
            hits = index.top(q, exclude_post=doc)
            out.append(make(j, "removed", q, hits, None, None))
            continue
        gold = Hit(gold_index, float(index.scores(q)[gold_index]), gold_chunk)
        hits, found = with_gold(index.top(q), gold, rng)
        rank = next(i + 1 for i, h in enumerate(hits) if h.index == gold_index)
        out.append(make(j, "answerable", q, hits, rank, found))
    if use_near_miss and qs.near_miss:
        hits = index.top(qs.near_miss)
        out.append(make("near", "near-miss", qs.near_miss, hits, None, None))
    return out
