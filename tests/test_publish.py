import json

from small_rag_finetune import publish
from small_rag_finetune.prompt import REFUSAL


def write(d, split, answers):
    d.mkdir(parents=True, exist_ok=True)
    ex = [
        {"messages": [{"role": "system"}, {"role": "user"}, {"content": a}]}
        for a in answers
    ]
    meta = [{"id": f"{d.name}/{split}/{k}", "source": "peps"} for k in range(len(ex))]
    (d / f"{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ex))
    (d / f"{split}.meta.jsonl").write_text("".join(json.dumps(x) + "\n" for x in meta))


def test_export_merges_meta_and_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "CONFIGS", {"v3": "data/v3", "broad": "data/v4-broad"})
    for data in publish.CONFIGS.values():
        for split in publish.SPLITS:
            write(tmp_path / data, split, ["Yes.", REFUSAL])
    for src in publish.LICENCES.values():
        (tmp_path / src).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / src).write_text("licence")
    counts = publish.export(tmp_path, tmp_path / "out")
    assert counts["v3"] == {"train": 2, "validation": 2, "test": 2}
    rows = [
        json.loads(x)
        for x in (tmp_path / "out/broad/validation.jsonl").read_text().splitlines()
    ]
    assert rows[0]["id"] == "v4-broad/valid/0"
    assert [r["answer"] for r in rows] == ["Yes.", REFUSAL]
    assert [r["refusal"] for r in rows] == [False, True]
    assert rows[0]["messages"][2] == {"content": "Yes."}
    assert (tmp_path / "out/README.md").read_text().startswith("---")
    assert (tmp_path / "out/licenses/rust-LICENSE-MIT").exists()
