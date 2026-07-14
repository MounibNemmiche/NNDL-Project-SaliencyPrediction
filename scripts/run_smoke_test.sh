#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests
python -m pytest -q
python src/smoke_test.py --device "${DEVICE:-cpu}"
