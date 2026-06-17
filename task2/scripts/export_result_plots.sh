#!/usr/bin/env bash
set -euo pipefail

python src/plot_results.py \
  --b-run /root/autodl-tmp/cv_hw3_task2/outputs/act_b_only_bs64_valcurve_20260616_174028 \
  --joint-run /root/autodl-tmp/cv_hw3_task2/outputs/act_joint_abc_bs64_valcurve_20260616_180359 \
  --zero-shot-dir /root/autodl-tmp/cv_hw3_task2/outputs/zero_shot_d \
  --output-dir /root/autodl-tmp/cv_hw3_task2/outputs/validation_curve_comparison
