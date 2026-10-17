import json

from fakes import FakeClient, FakeModel, write_post

from small_rag_finetune.broad import combine, run_broad
from small_rag_finetune.claude import parse_broad

TOPICS = ["generics", "async", "walrus", "lifetimes", "modules", "go", "rust", "python"]


def corpus(root):
    for i, t in enumerate(TOPICS * 3):
        body = f"## Intro\n\nAll about {t} in practice. " * 20
        write_post(root / "src", f"{t}-{i}", f"About {t}", body)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_parse_broad():
    assert parse_broad("TOPIC: What about X?\nGOAL: How do I Y?") == [
        "What about X?",
        "How do I Y?",
    ]
    assert parse_broad("TOPIC: NONE\nGOAL: How do I Y?") == ["How do I Y?"]


def test_run_broad_writes_examples_from_plain_retrieval(tmp_path):
    corpus(tmp_path / "corpus")
    client = FakeClient()
    summary = run_broad(
        tmp_path / "corpus",
        tmp_path / "broad",
        ["src"],
        {"train": 3, "valid": 1, "test": 1},
        client=client,
        model=FakeModel(),
        poll=0,
    )
    train = read(tmp_path / "broad" / "train.jsonl")
    meta = read(tmp_path / "broad" / "train.meta.jsonl")
    assert len(train) == len(meta) > 0
    assert {m["kind"] for m in meta} <= {"broad", "broad-removed"}
    assert meta[0]["question"] in FakeClient.BROAD_QUESTIONS
    reqs = [r for b in client.batches.batches.values() for r in b]
    teacher = [r for r in reqs if "system" in r["params"]]
    assert all("broad" in r["params"]["system"] for r in teacher)
    assert summary["usd_batch"]["teacher"] > 0


def test_combine_concatenates_and_keeps_meta_aligned(tmp_path):
    for name, n in (("a", 3), ("b", 2)):
        d = tmp_path / name
        d.mkdir()
        for split in ("train", "valid", "test"):
            (d / f"{split}.jsonl").write_text(
                "".join(json.dumps({"x": f"{name}{i}"}) + "\n" for i in range(n))
            )
            (d / f"{split}.meta.jsonl").write_text(
                "".join(json.dumps({"id": f"{name}{i}"}) + "\n" for i in range(n))
            )
    counts = combine([tmp_path / "a", tmp_path / "b"], tmp_path / "ab")
    assert counts == {"train": 5, "valid": 5, "test": 5}
    ex = read(tmp_path / "ab" / "train.jsonl")
    meta = read(tmp_path / "ab" / "train.meta.jsonl")
    assert [e["x"] for e in ex] == [m["id"] for m in meta]
