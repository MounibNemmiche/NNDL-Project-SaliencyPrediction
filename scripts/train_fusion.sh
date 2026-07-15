#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-25}"
SEED="${SEED:-42}"
LOSS="${LOSS:-mse_cc}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [[ "$SEED" == "42" ]]; then
  RUN_NAME="${RUN_NAME:-fusion_final}"
else
  RUN_NAME="${RUN_NAME:-fusion_final_seed${SEED}}"
fi

python src/train.py \
  --run-name "$RUN_NAME" \
  --model fusion \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr 1e-4 \
  --loss "$LOSS" \
  --lambda-cc 0.1 \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --val-manifest data/splits/val_seed42.txt \
  --device "$DEVICE" \
  --no-amp
