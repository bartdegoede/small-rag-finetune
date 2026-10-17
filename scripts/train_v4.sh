#!/bin/bash
# v4: Qwen3-0.6B on dataset v4 (v3 plus broad topic and goal questions per
# document), one epoch (5,729 examples at batch 2). Checkpoints are
# picked with dev-judge (judged answers), not validation loss.
#   nohup scripts/train_v4.sh &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs
/usr/bin/time -l uv run --no-sync mlx_lm.lora --model Qwen/Qwen3-0.6B \
  --train --data data/v4 --fine-tune-type lora --num-layers -1 --mask-prompt \
  --batch-size 2 --grad-accumulation-steps 2 --grad-checkpoint --iters 2865 \
  --max-seq-length 2048 --steps-per-report 50 --steps-per-eval 716 --save-every 716 \
  --val-batches -1 --seed 0 --learning-rate 5e-5 \
  --adapter-path runs/v4-qwen3-0.6b > logs/v4-qwen3-0.6b.log 2>&1
