#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/root/autodl-tmp/cv_hw3_task2/data/calvin-lerobot}"
mkdir -p "$DATA_DIR"

hf download xiaoma26/calvin-lerobot \
  --repo-type dataset \
  --local-dir "$DATA_DIR"

echo "Dataset downloaded to: $DATA_DIR"
find "$DATA_DIR" -maxdepth 1 -type d -name 'split*' -print | sort
