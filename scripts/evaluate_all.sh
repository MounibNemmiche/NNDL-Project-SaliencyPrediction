#!/usr/bin/env bash
set -euo pipefail

: "${SIMPLE_CHECKPOINT:?Set SIMPLE_CHECKPOINT to the SimpleCNN best checkpoint}"
: "${FUSION_CHECKPOINT:?Set FUSION_CHECKPOINT to the FusionCNN best checkpoint}"

DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evaluation/final}"

python src/evaluate.py --model center --output-dir "$OUTPUT_DIR" --device "$DEVICE"
python src/evaluate.py --checkpoint "$SIMPLE_CHECKPOINT" --output-dir "$OUTPUT_DIR" --device "$DEVICE"
python src/evaluate.py --checkpoint "$FUSION_CHECKPOINT" --output-dir "$OUTPUT_DIR" --device "$DEVICE"
