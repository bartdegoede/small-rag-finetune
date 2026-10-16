import json

from fakes import FakeClient

from small_rag_finetune.batch import run_batch

REQS = {
    "q00000": {"model": "claude-sonnet-5-5", "messages": []},
    "t00000": {"model": "claude-opus-5-5", "system": "s", "messages": []},
}


def test_collects_results_with_text_and_tokens(tmp_path):
    out = run_batch(FakeClient(), REQS, tmp_path, poll=0)
    assert set(out) == set(REQS)
    assert out["q00000"]["text"].startswith("Q1:")
    assert out["t00000"]["model"] == "claude-opus-5-5"
    assert out["t00000"]["output_tokens"] == 200
    assert not (tmp_path / "inflight.json").exists()


def test_failures_are_resubmitted(tmp_path):
    client = FakeClient()
    client.batches.fail = {"t00000"}
    out = run_batch(client, REQS, tmp_path, poll=0)
    assert set(out) == set(REQS)
    assert client.batches.created == [2, 1]


def test_a_second_run_reuses_saved_results(tmp_path):
    run_batch(FakeClient(), REQS, tmp_path, poll=0)
    client = FakeClient()
    assert set(run_batch(client, REQS, tmp_path, poll=0)) == set(REQS)
    assert client.batches.created == []


def test_an_inflight_batch_is_polled_not_resubmitted(tmp_path):
    client = FakeClient()
    requests = [{"custom_id": k, "params": v} for k, v in REQS.items()]
    bid = client.batches.create(requests).id
    (tmp_path / "inflight.json").write_text(json.dumps({"id": bid}))
    out = run_batch(client, REQS, tmp_path, poll=0)
    assert set(out) == set(REQS)
    assert client.batches.created == [2]  # only the one made above
