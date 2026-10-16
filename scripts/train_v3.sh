#!/bin/bash
# v3: Qwen3-0.6B on dataset v3 (extractive teacher answers: one fact, the
# passage's wording), one epoch (4,530 examples at batch 2). Checkpoints are
# picked with dev-judge (judged answers), not validation loss.
#   nohup scripts/train_v3.sh &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs
/usr/bin/time -l uv run --no-sync mlx_lm.lora --model Qwen/Qwen3-0.6B \
  --train --data data/v3 --fine-tune-type lora --num-layers -1 --mask-prompt \
  --batch-size 2 --grad-accumulation-steps 2 --grad-checkpoint --iters 2265 \
  --max-seq-length 2048 --steps-per-report 50 --steps-per-eval 566 --save-every 566 \
  --val-batches -1 --seed 0 --learning-rate 5e-5 \
  --adapter-path runs/v3-qwen3-0.6b > logs/v3-qwen3-0.6b.log 2>&1
