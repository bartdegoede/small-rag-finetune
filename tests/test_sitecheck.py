import json

import pytest
from fakes import FakeModel, small_corpus

from small_rag_finetune.index import SourceIndex
from small_rag_finetune.prompt import TOP_K
from small_rag_finetune.sitecheck import read_questions, site_prompts


def test_site_prompts_retrieve_from_the_site(tmp_path):
    small_corpus(tmp_path)
    index = SourceIndex.build(tmp_path, model=FakeModel())
    rows = site_prompts(
        index,
        [
            {"id": "q1", "question": "go generics"},
            {"id": "n1", "question": "rust lifetimes", "covered": False},
        ],
    )
    q1, n1 = rows
    assert q1["posts"][0] == "generics" and len(q1["posts"]) <= TOP_K
    assert (q1["covered"], q1["teacher_refused"]) == (True, False)
    assert (n1["covered"], n1["teacher_refused"]) == (False, True)
    assert [m["role"] for m in q1["messages"]] == ["system", "user"]
    assert q1["messages"][1]["content"].endswith("Question: go generics")


def test_read_questions_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(
        json.dumps({"id": "a", "question": "x"}) * 1
        + "\n"
        + json.dumps({"id": "a", "question": "y"})
        + "\n"
    )
    with pytest.raises(ValueError):
        read_questions(p)
