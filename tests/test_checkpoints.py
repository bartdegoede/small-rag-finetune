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
