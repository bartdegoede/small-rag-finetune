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
