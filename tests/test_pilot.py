import json

from fakes import FakeClient, FakeModel, write_post

from small_rag_finetune.claude import TEACHERS
from small_rag_finetune.pilot import run_pilot

TOPICS = ["generics", "async", "walrus", "lifetimes", "modules", "go", "rust", "python"]


def corpus(root):
    for i, t in enumerate(TOPICS):
        write_post(
            root / "src", f"{t}-{i}", f"About {t}", f"All about {t} in practice. " * 20
        )


def pilot(tmp_path, out):
    return run_pilot(
        tmp_path / "corpus",
        tmp_path / out,
        ["src"],
        3,
        client=FakeClient(),
        model=FakeModel(),
        workers=2,
    )


def test_pilot_writes_examples_calls_and_report(tmp_path):
    corpus(tmp_path / "corpus")
    rows, calls = pilot(tmp_path, "out")
    assert rows
    assert all(r["target_opus"] and r["target_sonnet"] for r in rows)
    assert all(r["split"] == "train" for r in rows)
    assert sum(c.role == "questions" for c in calls) == 3
    assert sum(c.role == "teacher" for c in calls) == len(rows) * len(TEACHERS)
    lines = (tmp_path / "out" / "examples.jsonl").read_text().splitlines()
    assert len(lines) == len(rows) and "messages" in json.loads(lines[0])
    report = (tmp_path / "out" / "report.md").read_text()
    assert "Projected for 5,000 examples" in report
    assert "claude-opus-5-5" in report and "claude-sonnet-5-5" in report


def test_pilot_is_deterministic(tmp_path):
    corpus(tmp_path / "corpus")
    a, _ = pilot(tmp_path, "a")
    b, _ = pilot(tmp_path, "b")
    assert [r["id"] for r in a] == [r["id"] for r in b]
