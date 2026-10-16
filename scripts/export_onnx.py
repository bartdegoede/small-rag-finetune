# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "onnxruntime-genai==0.18.0",
#     "transformers>=5",
#     "torch",
#     "onnx",
#     "onnx-ir",
#     "huggingface_hub",
# ]
# ///
"""Export a fused (merged-LoRA) Hugging Face model to the layout transformers.js
loads: `onnx/model_q4f16.onnx` next to config and tokenizer files.

    uv run scripts/export_onnx.py runs/fused/smollm2-360m runs/onnx/smollm2-360m \
        --tokenizer HuggingFaceTB/SmolLM2-360M-Instruct

Uses the onnxruntime-genai model builder (`-p int4 -e webgpu`: 4-bit weights,
fp16 activations), the tool behind the official onnx-community files. It emits
fused GroupQueryAttention / RotaryEmbedding ops, and for models with tied
embeddings (SmolLM2, Qwen3-0.6B) one shared 4-bit matrix serves both the
embedding and the LM head. The first version of this script used optimum +
transformers.js's quantize.py: 8k-node graphs and a separate LM head copy,
about 10-15% bigger.

--tokenizer: LoRA never changes the tokenizer, so the tokenizer files come
from the base model's Hub repo, in the format transformers.js expects.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
]
CONFIG_FILES = ["config.json", "generation_config.json"]


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


def copy_tokenizer(repo: str, model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for name in TOKENIZER_FILES:
        try:
            shutil.copy2(hf_hub_download(repo, name), model_dir / name)
        except EntryNotFoundError:
            pass


def fix_kv_dims(model, config: dict) -> None:
    """onnxruntime-genai 0.18 declares the KV cache's last dim as a symbol
    ('kv_cache_dim'); transformers.js sizes its empty cache from the declared
    shape and passes 0, which GroupQueryAttention rejects. Pin it to head_dim,
    as the official onnx-community files do."""
    head_dim = config.get("head_dim") or (
        config["hidden_size"] // config["num_attention_heads"]
    )
    for v in [*model.graph.input, *model.graph.output]:
        for d in v.type.tensor_type.shape.dim:
            if d.dim_param == "kv_cache_dim":
                d.ClearField("dim_param")
                d.dim_value = head_dim


def main(src: str, out: str, tokenizer: str) -> None:
    import onnx

    src_p, out_p = Path(src), Path(out)
    built = out_p / "_genai"
    subprocess.run(
        [sys.executable, "-m", "onnxruntime_genai.models.builder",
         "-i", str(src_p), "-o", str(built), "-p", "int4", "-e", "webgpu",
         "-c", str(out_p / "_cache")],
        check=True,
    )
    onnx_dir = out_p / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    # One self-contained file (< 2GB), so transformers.js needs no external data.
    model = onnx.load(str(built / "model.onnx"), load_external_data=True)
    fix_kv_dims(model, json.loads((src_p / "config.json").read_text()))
    onnx.save(model, str(onnx_dir / "model_q4f16.onnx"), save_as_external_data=False)
    for name in CONFIG_FILES:
        if (src_p / name).exists():
            shutil.copy2(src_p / name, out_p / name)
    copy_tokenizer(tokenizer, out_p)
    inline_chat_template(out_p)
    shutil.rmtree(built)
    shutil.rmtree(out_p / "_cache", ignore_errors=True)
    print(f"wrote {onnx_dir / 'model_q4f16.onnx'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", required=True, help="Hub repo for tokenizer files")
    a = ap.parse_args()
    main(a.src, a.out, a.tokenizer)
