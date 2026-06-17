#!/usr/bin/env bash
set -euo pipefail

python src/evaluate_zero_shot_d.py \
  --root /root/autodl-tmp/cv_hw3_task2/data/calvin-lerobot/splitD \
  --batch-size 64 \
  --num-workers 8 \
  --stride 1 \
  --out-dir /root/autodl-tmp/cv_hw3_task2/outputs/zero_shot_d
