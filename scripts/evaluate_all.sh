#!/usr/bin/env bash
set -euo pipefail

: "${SIMPLE_CHECKPOINT:?Set SIMPLE_CHECKPOINT to the SimpleCNN best checkpoint}"
: "${FUSION_CHECKPOINT:?Set FUSION_CHECKPOINT to the FusionCNN best checkpoint}"

DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evaluation/final}"
MANIFEST="${MANIFEST:-data/splits/test_seed42.txt}"
IMAGE_DIR="${IMAGE_DIR:-dataset/images/images/val}"
MAP_DIR="${MAP_DIR:-dataset/maps/val}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

COMMON_ARGS=(
  --manifest "$MANIFEST"
  --image-dir "$IMAGE_DIR"
  --map-dir "$MAP_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

python src/evaluate.py --model center "${COMMON_ARGS[@]}"
python src/evaluate.py --checkpoint "$SIMPLE_CHECKPOINT" "${COMMON_ARGS[@]}"
python src/evaluate.py --checkpoint "$FUSION_CHECKPOINT" "${COMMON_ARGS[@]}"
