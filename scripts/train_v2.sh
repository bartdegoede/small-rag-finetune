#!/bin/bash
# v2: Qwen3-0.6B only, on dataset v2 (v1 without partial-answer hedges), for one
# epoch (4,477 examples at batch 2): v1 was best at the end of epoch 1 for both
# models. Validation and a checkpoint every quarter epoch.
#   nohup scripts/train_v2.sh &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs
/usr/bin/time -l uv run --no-sync mlx_lm.lora --model Qwen/Qwen3-0.6B \
  --train --data data/v2 --fine-tune-type lora --num-layers -1 --mask-prompt \
  --batch-size 2 --grad-accumulation-steps 2 --grad-checkpoint --iters 2240 \
  --max-seq-length 2048 --steps-per-report 50 --steps-per-eval 560 --save-every 560 \
  --val-batches -1 --seed 0 --learning-rate 5e-5 \
  --adapter-path runs/v2-qwen3-0.6b > logs/v2-qwen3-0.6b.log 2>&1
