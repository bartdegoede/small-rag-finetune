import json

from fakes import FakeClient

from small_rag_finetune.devjudge import dev_prompts, judge_params, run_dev_judge
from small_rag_finetune.prompt import REFUSAL


def write_split(d, answers):
    d.mkdir()
    with (d / "test.jsonl").open("w") as f, (d / "test.meta.jsonl").open("w") as m:
        for i, a in enumerate(answers):
            msgs = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"Passages:\n\n[1] T\nbody {i}\n\nQ{i}?"},
                {"role": "assistant", "content": a},
            ]
            f.write(json.dumps({"messages": msgs}) + "\n")
            m.write(json.dumps({"id": f"src/doc/{i}/0"}) + "\n")


def test_dev_prompts_drop_the_answer_and_mark_refusals(tmp_path):
    write_split(tmp_path / "v3", ["An answer.", REFUSAL])
    rows = dev_prompts(tmp_path / "v3")
    assert [r["id"] for r in rows] == ["src/doc/0/0", "src/doc/1/0"]
    assert [len(r["messages"]) for r in rows] == [2, 2]
    assert [r["teacher_refused"] for r in rows] == [False, True]


def test_judge_params_carry_rubric_prompt_answer_and_schema():
    p = {"messages": [{}, {"content": "Passages: [1] X\n\nQ?"}]}
    params = judge_params(p, "It is X.")
    text = params["messages"][0]["content"]
    assert (
        "faithful" in text and "[1] X" in text and "<answer>It is X.</answer>" in text
    )
    assert params["output_config"]["format"]["type"] == "json_schema"


def test_run_dev_judge_tallies_by_teacher_behaviour(tmp_path):
    write_split(tmp_path / "v3", ["An answer.", REFUSAL, "Another."])
    prompts = dev_prompts(tmp_path / "v3")
    answers = tmp_path / "arm.jsonl"
    answers.write_text(
        "".join(
            json.dumps({"arm": "arm", "id": p["id"], "summary": "x"}) + "\n"
            for p in prompts
        )
    )
    s = run_dev_judge(FakeClient(), prompts, answers, tmp_path / "judge", poll=0)
    assert s["judged"] == 3
    assert s["answered"]["n"] == 2 and s["answered"]["faithful_rate"] == 1.0
    assert s["refused"]["n"] == 1
    assert (tmp_path / "judge" / "arm" / "verdicts.jsonl").exists()
