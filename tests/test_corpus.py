from collections import Counter

from sss_eval.post import parse_post

from small_rag_finetune.corpus import MIN_BODY_CHARS, split_of, write_corpus
from small_rag_finetune.normalize import Doc
from small_rag_finetune.sources import SOURCES, Source


def test_split_is_deterministic_and_roughly_90_5_5():
    assert split_of("go-blog", "go1.21") == split_of("go-blog", "go1.21")
    counts = Counter(split_of("s", f"doc-{i}") for i in range(4000))
    assert 0.87 < counts["train"] / 4000 < 0.93
    assert 0.03 < counts["valid"] / 4000 < 0.07
    assert 0.03 < counts["test"] / 4000 < 0.07


def test_split_depends_on_source():
    pairs = [(split_of("a", f"d{i}"), split_of("b", f"d{i}")) for i in range(200)]
    assert any(x != y for x, y in pairs)


def test_write_corpus_writes_files_sss_eval_can_parse(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("long", "short", "none"):
        (checkout / f"{name}.txt").write_text(name)

    def fake(path):
        if path.stem == "none":
            return None
        n = MIN_BODY_CHARS if path.stem == "long" else 10
        return Doc(path.stem, f"Title: {path.stem}", "## Part\n\n" + "x" * n)

    src = Source(
        "fake", "https://example.invalid/repo", "abc", ".", "*.txt", "MIT", fake
    )
    assert write_corpus(src, checkout, tmp_path / "corpus") == 1
    post = parse_post(tmp_path / "corpus" / "fake" / "long.md")
    assert post.slug == "long"
    assert post.title == "Title: long"
    assert post.sections[0].heading == "Part"


def test_four_sources_are_pinned():
    assert set(SOURCES) == {"go-blog", "rust-blog", "rust-rfcs", "peps"}
    assert all(len(s.commit) == 40 for s in SOURCES.values())
