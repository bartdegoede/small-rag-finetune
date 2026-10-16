# Plan 3: training, export and eval on dataset v1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune SmolLM2-360M and Qwen3-0.6B on dataset v1, pick each run's checkpoint on the validation loss, export to transformers.js q4f16 and score them on the blog's 45 questions with the same Opus judge.

**Architecture:** Training on the M4 (`scripts/train_v1.sh`, 2 epochs, checkpoint and validation every quarter epoch). A `best-checkpoint` command reads the mlx-lm log, picks the lowest validation loss and prepares an adapter directory; `mlx_lm.fuse` merges it; `scripts/export_onnx.py` (onnxruntime-genai builder) writes `onnx/model_q4f16.onnx`; `js/generate.mjs` answers the 45 prompts; the blog's `judge.py --runs-dir` grades them.

**Tech Stack:** mlx-lm 0.32, onnxruntime-genai 0.18, transformers.js 4.3.1 (onnxruntime-node), the blog's `scripts/ask_eval/judge.py` (Opus 5.5).

## Done before this plan was written (2026-10-16)

- Training started on the M4: `scripts/train_v1.sh` queues SmolLM2-360M (lr 1e-4) then Qwen3-0.6B (lr 5e-5), 4,530 iterations each (2 epochs at batch 2), `--steps-per-eval 566 --save-every 566 --val-batches -1`, gradient accumulation 2, gradient checkpointing. Logs: `logs/v1-*.log` on the M4.
- Export switched to the onnxruntime-genai builder (commit "Export with the onnxruntime-genai builder…"): SmolLM2-360M q4f16 195MB (was 300MB, official 272MB). The KV-cache dim is pinned to `head_dim` because transformers.js otherwise passes 0. Parity check: base SmolLM2-360M through the new export scores 2/30 faithful, 15/15 negatives invented (official file: 3/30, 15/15; old export: 2/30, 14/15).

---

### Task 1: `best-checkpoint` command

**Files:** Create `src/small_rag_finetune/checkpoints.py`; modify `src/small_rag_finetune/__init__.py`; test `tests/test_checkpoints.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_checkpoints.py
import json

from small_rag_finetune.checkpoints import best_checkpoint, val_losses

LOG = (
    "\x1b[1m  iter   train_loss\x1b[0m\n"
    "     1    \x1b[1;35mval\x1b[0m \x1b[1m1.712\x1b[0m    54.59s\n"
    "    50    1.077 ▼       72      4.5k\n"
    "   566    val 0.901    55.10s\n"
    "  save  0000566_adapters.safetensors\n"
    "  1132    val 0.874    55.00s\n"
    "  1698    val 0.880    55.20s\n"
)


def test_val_losses_strip_ansi_and_skip_iteration_1():
    assert val_losses(LOG) == {566: 0.901, 1132: 0.874, 1698: 0.880}


def test_best_checkpoint_prepares_an_adapter_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_config.json").write_text(json.dumps({"rank": 8}))
    for it in (566, 1132, 1698):
        (run / f"{it:07d}_adapters.safetensors").write_text(str(it))
    log = tmp_path / "train.log"
    log.write_text(LOG)
    it, out = best_checkpoint(log, run)
    assert it == 1132
    assert (out / "adapters.safetensors").read_text() == "1132"
    assert json.loads((out / "adapter_config.json").read_text()) == {"rank": 8}
```

- [ ] **Step 2:** run → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/small_rag_finetune/checkpoints.py
"""Pick a run's checkpoint on validation loss (never on the blog eval)."""

import re
import shutil
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_VAL = re.compile(r"^\s*(\d+)\s+val\s+([0-9.]+)", re.MULTILINE)


def val_losses(log_text: str) -> dict[int, float]:
    """Validation loss per iteration; iteration 1 is the untrained baseline."""
    text = _ANSI.sub("", log_text)
    return {int(i): float(v) for i, v in _VAL.findall(text) if int(i) > 1}


def best_checkpoint(log: Path, run_dir: Path) -> tuple[int, Path]:
    """Copy the best checkpoint to <run_dir>-best/ as an mlx-lm adapter dir."""
    losses = val_losses(log.read_text())
    it = min(losses, key=losses.get)
    out = run_dir.with_name(run_dir.name + "-best")
    out.mkdir(exist_ok=True)
    shutil.copy2(run_dir / f"{it:07d}_adapters.safetensors", out / "adapters.safetensors")
    shutil.copy2(run_dir / "adapter_config.json", out / "adapter_config.json")
    return it, out
```

CLI:

```python
@main.command()
@click.argument("log", type=click.Path(exists=True, path_type=Path))
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
def best_checkpoint(log: Path, run_dir: Path) -> None:
    """Pick the lowest-validation-loss checkpoint into <run_dir>-best/."""
    from .checkpoints import best_checkpoint as pick, val_losses

    it, out = pick(log, run_dir)
    click.echo(f"val losses: {val_losses(log.read_text())}")
    click.echo(f"best: iteration {it} -> {out}")
```

- [ ] **Step 4:** tests pass, black/ruff clean. **Step 5: Commit** `best-checkpoint: pick a run's adapter on validation loss`

---

### Task 2: Fuse, export and answer the 45 questions (per finished run)

Run on the M4 after `git pull`-style sync of the repo (rsync without `data/`, `runs/`). For `<run>` = `v1-smollm2-360m` (base `HuggingFaceTB/SmolLM2-360M-Instruct`) and `v1-qwen3-0.6b` (base `Qwen/Qwen3-0.6B`):

- [ ] **Step 1:** `uv run --no-sync small-rag-finetune best-checkpoint logs/<run>.log runs/<run>` → prints the losses and the chosen iteration.
- [ ] **Step 2:** `uv run --no-sync mlx_lm.fuse --model <base> --adapter-path runs/<run>-best --save-path runs/fused/<run>`
- [ ] **Step 3:** `uv run scripts/export_onnx.py runs/fused/<run> runs/onnx/<run> --tokenizer <base>` → `runs/onnx/<run>/onnx/model_q4f16.onnx` (SmolLM2 ≈ 195MB).
- [ ] **Step 4:** `node js/generate.mjs runs/onnx/<run> eval/blog-prompts.jsonl > runs/eval/<run>.jsonl` → 45 lines. Do this while the other model is not training (CPU contention), or on the M1 after copying `runs/onnx/<run>`.
- [ ] **Step 5:** Also export base Qwen3-0.6B the same way once (`runs/onnx/genai-base-qwen3-0.6b`) and generate its 45 answers, the like-for-like baseline (the SmolLM2 one exists: `genai-base-smollm2-360m`).

### Task 3: Judge and compare

- [ ] **Step 1:** Copy `runs/eval/*.jsonl` to the M1. From `~/projects/blog`:
  `zsh -ic 'unset VIRTUAL_ENV; uv run python scripts/ask_eval/judge.py --runs-dir /Users/bart/projects/small-rag-finetune/runs/eval --out /Users/bart/projects/small-rag-finetune/runs/eval/judgements.jsonl --workers 4 --arm <run>'` (about $0.44 per arm).
- [ ] **Step 2:** Tally per arm: faithful / unsupported / non-answer on q01–q30, invented on n01–n15, median ms. Read every answer of the fine-tuned arms next to its base arm; note overused partial-answer clauses ("the blog does not go into…"), refusals of answerable questions, and format problems.
- [ ] **Step 3:** Write `docs/superpowers/specs/2026-10-16-v1-results.md`: table (base vs fine-tuned, both models, plus the published numbers for Gemma 4 E2B 28/30 and Haiku 27/30), the checkpoint chosen and its validation curve, download sizes, latency, and 4-6 answer pairs. Commit `v1 results`.
- [ ] **Step 4:** Report to the user with a recommendation: publish, or iterate (e.g. strip partial-answer clauses, change the learning rate, epochs).

Publishing (Hugging Face model and dataset repos, GitHub repo, blog post) is a separate plan once the results are in and the user has picked names.
