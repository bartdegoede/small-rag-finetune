from fakes import FakeModel, small_corpus

from small_rag_finetune.index import CHUNK_OVERLAP, CHUNK_SIZE, SourceIndex


def build(tmp_path):
    small_corpus(tmp_path)
    return SourceIndex.build(tmp_path, model=FakeModel())


def test_matches_blog_index_settings():
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (600, 120)


def test_chunks_carry_title_heading_and_snippet(tmp_path):
    idx = build(tmp_path)
    c = next(c for c in idx.chunks if c.post == "generics")
    assert c.title == "Go generics"
    assert c.text.startswith("Go generics\n\n")
    assert len(c.snippet) <= CHUNK_SIZE


def test_top_ranks_by_cosine(tmp_path):
    idx = build(tmp_path)
    hits = idx.top("go generics", k=2)
    assert hits[0].chunk.post == "generics"
    assert hits[0].score >= hits[1].score


def test_exclude_post_removes_the_document(tmp_path):
    idx = build(tmp_path)
    hits = idx.top("go generics", k=4, exclude_post="generics")
    assert hits and all(h.chunk.post != "generics" for h in hits)


def test_scores_has_one_value_per_chunk(tmp_path):
    idx = build(tmp_path)
    assert len(idx.scores("rust")) == len(idx.chunks)
