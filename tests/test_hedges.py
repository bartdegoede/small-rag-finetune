import json

from small_rag_finetune.hedges import REFUSAL, strip_hedge, strip_split


def test_refusal_is_kept():
    assert strip_hedge(REFUSAL) == REFUSAL


def test_plain_answer_is_kept():
    a = "Go 1.26 added three packages. They are crypto/hpke and two test helpers."
    assert strip_hedge(a) == a


def test_trailing_hedge_sentence_is_dropped():
    a = "PEP 570 adds the / syntax. The blog does not go into further detail."
    assert strip_hedge(a) == "PEP 570 adds the / syntax."


def test_trailing_it_doesnt_sentence_is_dropped():
    a = "aiter() waited for Python 3.7. It does not say when aiter() was added."
    assert strip_hedge(a) == "aiter() waited for Python 3.7."


def test_trailing_clause_is_cut():
    a = "Collectives recognise members, but the blog does not go into how that works."
    assert strip_hedge(a) == "Collectives recognise members."
    b = "Lowering starts with raw types; the blog does not go into the second step."
    assert strip_hedge(b) == "Lowering starts with raw types."
    c = "It writes an AST, but it doesn't say what happens on a syntax error."
    assert strip_hedge(c) == "It writes an AST."


def test_leading_hedge_drops_the_example():
    a = "The blog doesn't explain how to make it permanent. It does say 1.13 enables it."
    assert strip_hedge(a) is None


def test_strip_split_rewrites_answers_and_meta(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    rows = [
        "A is B. The blog does not explain why.",
        "The blog does not explain C.",
        REFUSAL,
    ]
    with (
        (src / "train.jsonl").open("w") as f,
        (src / "train.meta.jsonl").open("w") as m,
    ):
        for i, a in enumerate(rows):
            msgs = [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": a},
            ]
            f.write(json.dumps({"messages": msgs}) + "\n")
            m.write(json.dumps({"id": str(i)}) + "\n")
    counts = strip_split(src, out, "train")
    assert counts == {"kept": 1, "stripped": 1, "dropped": 1}
    got = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert [g["messages"][-1]["content"] for g in got] == ["A is B.", REFUSAL]
    meta = [
        json.loads(x)["id"] for x in (out / "train.meta.jsonl").read_text().splitlines()
    ]
    assert meta == ["0", "2"]


def test_other_verbs_after_the_blog_are_hedges():
    a = "Cassandra leads the group, but the blog does not list its other members."
    assert strip_hedge(a) == "Cassandra leads the group."
    b = "S is a slice of E. The blog does not spell out the full signature."
    assert strip_hedge(b) == "S is a slice of E."


def test_it_doesnt_with_an_ordinary_verb_is_an_answer():
    a = "No, it doesn't install the toolchain in your PATH. It runs it from the cache."
    assert strip_hedge(a) == a


def test_gives_no_and_subjectless_clause_are_hedges():
    assert strip_hedge("The blog gives no date. It notes #12425.") is None
    a = "The flaw was fixed in 0.4.5 but does not say whether it was exploited."
    assert strip_hedge(a) == "The flaw was fixed in 0.4.5."
