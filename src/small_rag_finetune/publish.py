"""The Hugging Face dataset: one config per dataset version, one row per
example with its chat messages and metadata together.

    small-rag-finetune hf-export dist/hf
"""

import json
import shutil
from pathlib import Path

from .prompt import REFUSAL

# config name -> data directory; the first is the default.
CONFIGS = {"v3": "data/v3", "broad": "data/v4-broad", "v1": "data/v1"}
SPLITS = {"train": "train", "valid": "validation", "test": "test"}
CARD = Path(__file__).parent / "dataset_card.md"
# Licence texts that must travel with the Rust passages (MIT OR Apache-2.0).
LICENCES = {
    "rust-LICENSE-MIT": "data/sources/rust-blog/LICENSE-MIT",
    "rust-LICENSE-APACHE": "data/sources/rust-blog/LICENSE-APACHE",
}


def rows(data_dir: Path, split: str) -> list[dict]:
    examples = (data_dir / f"{split}.jsonl").read_text().splitlines()
    metas = (data_dir / f"{split}.meta.jsonl").read_text().splitlines()
    out = []
    for e, m in zip(examples, metas, strict=True):
        messages = json.loads(e)["messages"]
        out.append(
            {
                **json.loads(m),
                "answer": messages[2]["content"],
                "refusal": messages[2]["content"] == REFUSAL,
                "messages": messages,
            }
        )
    return out


def export(root: Path, out: Path) -> dict[str, dict[str, int]]:
    """Write <out>/<config>/<split>.jsonl, the card as <out>/README.md and
    the Rust licence texts to <out>/licenses/."""
    counts = {}
    for config, data in CONFIGS.items():
        (out / config).mkdir(parents=True, exist_ok=True)
        counts[config] = {}
        for split, name in SPLITS.items():
            rs = rows(root / data, split)
            (out / config / f"{name}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
            )
            counts[config][name] = len(rs)
    shutil.copy(CARD, out / "README.md")
    (out / "licenses").mkdir(exist_ok=True)
    for name, src in LICENCES.items():
        shutil.copy(root / src, out / "licenses" / name)
    return counts


MODEL_CARD = Path(__file__).parent / "model_card.md"
# Greedy, as evaluated: transformers.js and transformers both read these
# defaults, and Qwen3's own (sampling at temperature 0.6) would change answers.
GREEDY = {
    "bos_token_id": 151643,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "do_sample": False,
    "repetition_penalty": 1.0,
}


def stage_model(fused: Path, onnx: Path, out: Path) -> list[str]:
    """One Hub repo for both runtimes: the merged safetensors for transformers
    and MLX at the root, the q4f16 ONNX file under onnx/ for transformers.js,
    sharing config and the tokenizer files the ONNX export was tested with."""
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    for name in ["model.safetensors", "model.safetensors.index.json"]:
        shutil.copy(fused / name, out / name)
    for p in onnx.iterdir():
        if p.is_file() and p.name != "generation_config.json":
            shutil.copy(p, out / p.name)
    shutil.copy(onnx / "onnx/model_q4f16.onnx", out / "onnx/model_q4f16.onnx")
    (out / "generation_config.json").write_text(json.dumps(GREEDY, indent=2) + "\n")
    shutil.copy(MODEL_CARD, out / "README.md")
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
