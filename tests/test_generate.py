import json

from fakes import FakeClient, FakeModel, write_post

from small_rag_finetune.generate import run_generate, target_problem
from small_rag_finetune.prompt import REFUSAL

TOPICS = ["generics", "async", "walrus", "lifetimes", "modules", "go", "rust", "python"]


def corpus(root):
    for i, t in enumerate(TOPICS * 3):
        write_post(
            root / "src", f"{t}-{i}", f"About {t}", f"All about {t} in practice. " * 20
        )


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_target_problem():
    assert target_problem(None) == "missing"
    assert target_problem("") == "missing"
    assert target_problem("Passage [2] says so.") == "mentions-passages"
    assert target_problem("The passages say X.") == "mentions-passages"
    assert target_problem("One. Two. Three.") == "over-2-sentences"
    assert target_problem(REFUSAL) is None
    assert target_problem("It returns x[0] and m[1].") is None
    assert target_problem("Go 1.26 adds three packages.") is None


def test_generate_writes_mlx_lm_files_meta_and_summary(tmp_path):
    corpus(tmp_path / "corpus")
    out = tmp_path / "v1"
    summary = run_generate(
        tmp_path / "corpus",
        out,
        ["src"],
        {"train": 4, "valid": 2, "test": 2},
        client=FakeClient(),
        model=FakeModel(),
        poll=0,
    )
    train, meta = read(out / "train.jsonl"), read(out / "train.meta.jsonl")
    assert train and len(train) == len(meta)
    assert [m["role"] for m in train[0]["messages"]] == ["system", "user", "assistant"]
    assert train[0]["messages"][2]["content"] == FakeClient.ANSWER
    keys = {"id", "source", "doc", "kind", "question", "gold_rank", "teacher"}
    assert keys <= set(meta[0])
    assert summary["examples"]["train"] == len(train)
    assert summary["usd_batch"] > 0
    assert (out / "summary.json").exists()


def test_generate_resumes_without_new_batches(tmp_path):
    corpus(tmp_path / "corpus")
    quotas = {"train": 3, "valid": 0, "test": 0}
    args = (tmp_path / "corpus", tmp_path / "v1", ["src"], quotas)
    first = run_generate(*args, client=FakeClient(), model=FakeModel(), poll=0)
    client = FakeClient()
    second = run_generate(*args, client=client, model=FakeModel(), poll=0)
    assert client.batches.created == []
    assert first["examples"] == second["examples"]
