#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-25}"
SEED="${SEED:-42}"
LOSS="${LOSS:-mse_cc}"

python src/train.py \
  --run-name "simple_final_seed${SEED}" \
  --model simple \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --loss "$LOSS" \
  --lambda-cc 0.1 \
  --seed "$SEED" \
  --val-manifest data/splits/val_seed42.txt \
  --device "$DEVICE" \
  --amp
