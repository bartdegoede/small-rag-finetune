"""From a gold chunk and Claude's questions to the exact prompts the student sees.

- answerable: retrieval for the question; if the gold chunk isn't in the top 4
  it replaces the 4th and the list is re-sorted by score, so the answer is not
  always passage [1].
- removed: same question, every chunk of the gold document excluded.
- near-miss: Claude's related question the gold chunk doesn't answer.
"""

from dataclasses import asdict, dataclass

from .claude import Questions
from .corpus import split_of
from .index import Hit, SourceIndex
from .prompt import TOP_K, build_messages


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


def with_gold(hits: list[Hit], gold: Hit) -> tuple[list[Hit], bool]:
    if any(h.index == gold.index for h in hits):
        return hits, True
    kept = [*hits[: TOP_K - 1], gold]
    return sorted(kept, key=lambda h: (-h.score, h.index)), False


def examples_for(
    index: SourceIndex,
    source: str,
    gold_index: int,
    qs: Questions,
    *,
    use_near_miss: bool,
    removed: set[int],
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
        hits, found = with_gold(index.top(q), gold)
        rank = next(i + 1 for i, h in enumerate(hits) if h.index == gold_index)
        out.append(make(j, "answerable", q, hits, rank, found))
    if use_near_miss and qs.near_miss:
        hits = index.top(qs.near_miss)
        out.append(make("near", "near-miss", qs.near_miss, hits, None, None))
    return out
