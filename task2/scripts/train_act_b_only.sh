#!/usr/bin/env bash
set -euo pipefail

python src/train_act_calvin.py \
  --mode b_only \
  --batch-size 64 \
  --steps 12000 \
  --num-workers 8 \
  --log-freq 50 \
  --val-freq 1000 \
  --max-val-batches 128
