# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "optimum-onnx[onnxruntime]==0.1.0",
#     "transformers>=4.51,<4.58",
#     "torch",
#     "onnx",
#     "onnxslim",
#     "onnx-ir",
#     "tqdm",
# ]
# ///
"""Export a fused (merged-LoRA) Hugging Face model to ONNX and quantize to q4f16,
the layout transformers.js loads (`onnx/model_q4f16.onnx`).

Runs in its own environment because optimum-onnx pins transformers < 4.58 and
mlx-lm needs 5.x:

    uv run scripts/export_onnx.py runs/fused/smollm2-360m runs/onnx/smollm2-360m \
        --tokenizer HuggingFaceTB/SmolLM2-360M-Instruct

--tokenizer: LoRA never changes the tokenizer, and mlx-lm saves it in the
transformers 5 format, which optimum (4.5x) silently skips. So the tokenizer
files are taken from the base model's Hub repo instead.

q4f16 = ONNX Runtime's MatMul4BitsQuantizer (block size 32, symmetric) on the
fp32 export, then the remaining weights to fp16: the same recipe as
transformers.js 3.7.0's scripts/quantize.py (vendored, Apache-2.0).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "src" / "small_rag_finetune" / "vendor"


def inline_chat_template(model_dir: Path) -> None:
    """transformers >= 4.5x writes chat_template.jinja; transformers.js 4.3.1 only
    reads tokenizer_config.json["chat_template"], so copy it in."""
    jinja = model_dir / "chat_template.jinja"
    if not jinja.exists():
        return
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["chat_template"] = jinja.read_text()
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "vocab.json", "merges.txt", "tokenizer.model",
]


def copy_tokenizer(repo: str, model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for name in TOKENIZER_FILES:
        try:
            shutil.copy2(hf_hub_download(repo, name), model_dir / name)
        except EntryNotFoundError:
            pass


def main(src: str, out: str, tokenizer: str | None = None) -> None:
    src_p, out_p = Path(src), Path(out)
    fp32 = out_p / "_fp32"
    subprocess.run(
        ["optimum-cli", "export", "onnx", "--model", str(src_p),
         "--task", "text-generation-with-past", str(fp32)],
        check=True,
    )
    onnx_dir = out_p / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    # Config, tokenizer and generation files go next to onnx/, like the Hub repos.
    for f in fp32.iterdir():
        if f.suffix != ".onnx" and not f.name.endswith(".onnx_data"):
            shutil.copy2(f, out_p / f.name)
    if tokenizer:
        copy_tokenizer(tokenizer, out_p)
    inline_chat_template(out_p)
    sys.path.insert(0, str(VENDOR.parent))
    from vendor.quantize import QuantizationArguments, QuantMode, quantize

    quantize(str(fp32), str(onnx_dir), QuantizationArguments(modes=[QuantMode.Q4F16]))
    shutil.rmtree(fp32)
    print(f"wrote {onnx_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", help="Hub repo to take tokenizer files from")
    a = ap.parse_args()
    main(a.src, a.out, a.tokenizer)
