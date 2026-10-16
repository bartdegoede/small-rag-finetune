from fakes import FakeModel, small_corpus

from small_rag_finetune.claude import Questions
from small_rag_finetune.examples import examples_for, with_gold
from small_rag_finetune.index import Hit, SourceIndex


def index(tmp_path):
    small_corpus(tmp_path)
    return SourceIndex.build(tmp_path, model=FakeModel())


def gold_of(idx, post):
    return next(i for i, c in enumerate(idx.chunks) if c.post == post)


def test_with_gold_keeps_hits_when_gold_is_retrieved():
    hits = [Hit(i, 1.0 - i / 10, None) for i in range(4)]
    out, found = with_gold(hits, hits[2])
    assert found and out == hits


def test_with_gold_replaces_the_fourth_and_resorts():
    hits = [Hit(i, 0.9 - i / 10, None) for i in range(4)]  # 0.9 .. 0.6
    out, found = with_gold(hits, Hit(9, 0.65, None))
    assert not found
    assert [h.index for h in out] == [0, 1, 2, 9]
    assert [h.index for h in with_gold(hits, Hit(9, 0.95, None))[0]] == [9, 0, 1, 2]


def test_answerable_examples_contain_the_gold_passage(tmp_path):
    idx = index(tmp_path)
    qs = Questions(["What does the python walrus operator do?"], None)
    [ex] = examples_for(
        idx, "src", gold_of(idx, "walrus"), qs, use_near_miss=False, removed=set()
    )
    assert ex.kind == "answerable"
    assert ex.passages[ex.gold_rank - 1]["title"] == "The walrus"
    assert ex.gold_retrieved is True
    assert ex.messages()[1]["content"].endswith(
        "Question: What does the python walrus operator do?"
    )


def test_removed_examples_exclude_the_whole_document(tmp_path):
    idx = index(tmp_path)
    qs = Questions(["What does the python walrus operator do?"], None)
    [ex] = examples_for(
        idx, "src", gold_of(idx, "walrus"), qs, use_near_miss=False, removed={0}
    )
    assert ex.kind == "removed"
    assert ex.gold_rank is None
    assert all(p["title"] != "The walrus" for p in ex.passages)


def test_near_miss_is_added_only_when_asked(tmp_path):
    idx = index(tmp_path)
    gi = gold_of(idx, "generics")
    qs = Questions(["What do go generics add?"], "When will rust get generics?")
    with_near = examples_for(idx, "src", gi, qs, use_near_miss=True, removed=set())
    assert [e.kind for e in with_near] == ["answerable", "near-miss"]
    without = examples_for(idx, "src", gi, qs, use_near_miss=False, removed=set())
    assert len(without) == 1


def test_example_ids_are_unique_and_to_dict_has_messages(tmp_path):
    idx = index(tmp_path)
    gi = gold_of(idx, "generics")
    qs = Questions(
        ["What do go generics add?", "How are go generics written?"], "Rust?"
    )
    exs = examples_for(idx, "src", gi, qs, use_near_miss=True, removed={1})
    assert len({e.id for e in exs}) == 3
    d = exs[0].to_dict()
    assert d["messages"][0]["role"] == "system"
    assert d["split"] in {"train", "valid", "test"}
