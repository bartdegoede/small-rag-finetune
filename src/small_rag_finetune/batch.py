"""Message Batches (50% off): submit, poll, collect, resubmit failures.

Resumable through files in one directory per phase:
- results.jsonl: one line per succeeded request (custom_id, model, text, tokens)
- inflight.json: the batch being processed, so a restart polls it instead of
  paying for it twice
custom_id must match ^[a-zA-Z0-9_-]{1,64}$, so callers use ids like q00042.
"""

import json
import time
from pathlib import Path

MAX_ROUNDS = 3


def _load(d: Path) -> dict[str, dict]:
    path = d / "results.jsonl"
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line)
    return {r["custom_id"]: r for r in rows}


def _row(r) -> dict:
    msg = r.result.message
    return {
        "custom_id": r.custom_id,
        "model": msg.model,
        "text": next((b.text for b in msg.content if b.type == "text"), ""),
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }


def run_batch(
    client, requests: dict[str, dict], d: Path, *, poll: float = 60
) -> dict[str, dict]:
    d.mkdir(parents=True, exist_ok=True)
    results = _load(d)
    inflight = d / "inflight.json"
    for _ in range(MAX_ROUNDS):
        if inflight.exists():
            batch_id = json.loads(inflight.read_text())["id"]
        else:
            todo = [cid for cid in requests if cid not in results]
            if not todo:
                break
            batch = client.messages.batches.create(
                requests=[{"custom_id": cid, "params": requests[cid]} for cid in todo]
            )
            batch_id = batch.id
            inflight.write_text(json.dumps({"id": batch_id, "n": len(todo)}))
            print(f"submitted {batch_id}: {len(todo)} requests", flush=True)
        while client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(poll)
        with (d / "results.jsonl").open("a") as f:
            for r in client.messages.batches.results(batch_id):
                if r.result.type != "succeeded":
                    continue
                row = _row(r)
                results[r.custom_id] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        inflight.unlink()
    return {cid: results[cid] for cid in requests if cid in results}
