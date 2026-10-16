#!/bin/bash
# LoRA fine-tune of SmolLM2-360M, then Qwen3-0.6B, on dataset v1: 2 epochs each
# (4,529 examples at batch 2), validation and a checkpoint every quarter epoch.
# Fits a 24GB Apple Silicon Mac thanks to gradient checkpointing.
#   nohup scripts/train_v1.sh &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs
common="--train --data data/v1 --fine-tune-type lora --num-layers -1 --mask-prompt \
  --batch-size 2 --grad-accumulation-steps 2 --grad-checkpoint --iters 4530 \
  --max-seq-length 2048 --steps-per-report 50 --steps-per-eval 566 --save-every 566 \
  --val-batches -1 --seed 0"
/usr/bin/time -l uv run --no-sync mlx_lm.lora --model HuggingFaceTB/SmolLM2-360M-Instruct \
  $common --learning-rate 1e-4 --adapter-path runs/v1-smollm2-360m > logs/v1-smollm2-360m.log 2>&1
/usr/bin/time -l uv run --no-sync mlx_lm.lora --model Qwen/Qwen3-0.6B \
  $common --learning-rate 5e-5 --adapter-path runs/v1-qwen3-0.6b > logs/v1-qwen3-0.6b.log 2>&1
