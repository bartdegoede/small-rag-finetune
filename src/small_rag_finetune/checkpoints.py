"""Pick a run's checkpoint on validation loss (never on the blog eval)."""

import re
import shutil
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_VAL = re.compile(r"^\s*(\d+)\s+val\s+([0-9.]+)", re.MULTILINE)
_TRAIN = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+[▲▼]", re.MULTILINE)


def _clean(log_text: str) -> str:
    return _ANSI.sub("", log_text).replace("\r", "\n")


def val_losses(log_text: str, baseline: bool = False) -> dict[int, float]:
    """Validation loss per iteration; iteration 1 is the untrained baseline,
    left out unless asked for (it has no checkpoint)."""
    found = _VAL.findall(_clean(log_text))
    return {int(i): float(v) for i, v in found if baseline or int(i) > 1}


def train_losses(log_text: str) -> dict[int, float]:
    """Training loss per report (every 50 iterations by default)."""
    return {int(i): float(v) for i, v in _TRAIN.findall(_clean(log_text))}


def best_checkpoint(log: Path, run_dir: Path) -> tuple[int, Path]:
    """Copy the best checkpoint to <run_dir>-best/ as an mlx-lm adapter dir."""
    losses = val_losses(log.read_text())
    it = min(losses, key=losses.get)
    out = run_dir.with_name(run_dir.name + "-best")
    out.mkdir(exist_ok=True)
    src = run_dir / f"{it:07d}_adapters.safetensors"
    shutil.copy2(src, out / "adapters.safetensors")
    shutil.copy2(run_dir / "adapter_config.json", out / "adapter_config.json")
    return it, out
