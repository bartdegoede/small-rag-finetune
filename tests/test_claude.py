from types import SimpleNamespace

from fakes import FakeClient

from small_rag_finetune.claude import (
    QUESTION_MODEL,
    TEACHERS,
    complete,
    cost,
    normalize_answer,
    parse_questions,
    question_params,
    target_key,
    teacher_params,
)
from small_rag_finetune.prompt import REFUSAL, build_messages


def test_parse_questions_skips_none():
    qs = parse_questions("Q1: What is X?\nQ2: NONE\nQ3: Why Y?\nNEAR: When Z?\n")
    assert qs.answerable == ["What is X?", "Why Y?"]
    assert qs.near_miss == "When Z?"


def test_parse_questions_without_near_miss():
    assert parse_questions("Q1: A?\nNEAR: NONE").near_miss is None


def test_normalize_answer_collapses_whitespace_and_refusals():
    assert normalize_answer("  Go 1.21\n adds PGO.  ") == "Go 1.21 adds PGO."
    assert normalize_answer("The blog does not seem to cover this topic.") == REFUSAL


def test_question_params_carry_the_passage():
    chunk = SimpleNamespace(title="Go 1.21", heading="Tools", snippet="PGO is now GA.")
    p = question_params(chunk)
    assert p["model"] == QUESTION_MODEL
    content = p["messages"][0]["content"]
    assert "Go 1.21" in content and "Tools" in content and "PGO is now GA." in content


def test_teacher_sees_the_student_prompt_plus_guidance():
    msgs = build_messages(
        "What is PGO?", [{"title": "T", "heading": None, "snippet": "s" * 200}]
    )
    p = teacher_params(TEACHERS[0], msgs)
    assert p["system"].startswith(msgs[0]["content"])
    assert len(p["system"]) > len(msgs[0]["content"])
    assert p["messages"] == [{"role": "user", "content": msgs[1]["content"]}]


def test_complete_returns_the_text_block():
    text, usage = complete(FakeClient(), {"model": "m", "messages": []})
    assert text.startswith("Q1:")
    assert usage.input_tokens == 1000


def test_cost_and_batch_discount():
    assert cost("claude-opus-5-5", 1_000_000, 100_000) == 4.0 + 2.0
    assert cost("claude-sonnet-5-5", 1_000_000, 0, batch=True) == 1.0


def test_target_keys():
    assert [target_key(m) for m in TEACHERS] == ["target_opus", "target_sonnet"]


def test_teacher_guidance_rules():
    from small_rag_finetune.claude import TEACHER_GUIDANCE as g

    assert "Never mention the passages" in g
    assert "only part of the question" in g
    assert g.rstrip().endswith("The blog does not seem to cover this.")
