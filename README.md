# Lightweight Multi-Scale Saliency Prediction

PyTorch implementation of vision-based saliency prediction on SALICON. The final study compares a fixed center-bias prior, a lightweight SimpleCNN encoder-decoder, and a MultiScaleFusionCNN that adds side outputs and learned scale fusion to the same backbone.

## Current Integration Status

The dataset, three models, checkpoint/path utilities, and executable train/evaluate/predict pipeline are integrated and covered by CPU/CUDA tests. Final training results remain pending; uploaded synthetic pipeline checks are not final SALICON evidence.

## Native Setup

Python 3.10 is the submission target. Docker is optional and CPU-only.

### venv

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Conda

```bash
conda env create -f environment.yml
conda activate saliency
```

For NVIDIA systems, select the correct CUDA build from the [official PyTorch installer](https://pytorch.org/get-started/locally/) if the default wheel does not match the machine. Apple Silicon uses native PyTorch with MPS; Docker cannot provide Metal acceleration.

## Dataset

The team archive is stored locally and ignored by Git:

```text
dataset/images/images/train    10,000 images
dataset/maps/train             10,000 saliency maps
dataset/images/images/val       5,000 images
dataset/maps/val                5,000 saliency maps
```

See [data/README.md](data/README.md) for sources, alternative paths, pairing rules, and the reproducible read check.

Generate the fixed split once:

```bash
python -m src.datasets.create_splits \
  --image-dir dataset/images/images/val \
  --map-dir dataset/maps/val \
  --val-size 500 \
  --seed 42 \
  --val-output data/splits/val_seed42.txt \
  --test-output data/splits/test_seed42.txt
```

The tracked split contains 500 validation filenames and 4,500 disjoint final-test filenames. The final test is not used for tuning or checkpoint selection.

## Verification

These commands require no SALICON download:

```bash
python -m compileall -q src
python -m pytest -q
python src/smoke_test.py --device cpu
python src/smoke_test.py --device auto
```

Student B must additionally run the same mandatory fusion smoke test on the RTX 4090:

```bash
python src/smoke_test.py --device cuda
```

An explicit unavailable `cuda` or `mps` request fails instead of silently running on CPU. `auto` selects CUDA, then MPS, then CPU.

## Real-Data Read Check

```bash
python -m src.datasets.salicon_dataset \
  --image_dir dataset/images/images/train \
  --map_dir dataset/maps/train \
  --max_samples 4
```

## Experiment Pipeline

Screen a model with a fixed validation split:

```bash
python src/train.py \
  --run-name fusion_msecc_screen_seed42 \
  --model fusion \
  --epochs 10 \
  --batch-size 32 \
  --loss mse_cc \
  --lambda-cc 0.1 \
  --train-samples 2000 \
  --val-manifest data/splits/val_seed42.txt \
  --device cuda \
  --amp
```

Resume the same run without changing its model or loss contract:

```bash
python src/train.py \
  --resume outputs/runs/fusion_msecc_screen_seed42/checkpoints/last.pth \
  --epochs 20 \
  --device cuda \
  --amp
```

Evaluate the fixed center baseline or a learned checkpoint on the disjoint labeled final-test manifest:

```bash
python src/evaluate.py \
  --model center \
  --manifest data/splits/test_seed42.txt \
  --output-dir outputs/evaluation/final_seed42 \
  --device cuda

python src/evaluate.py \
  --checkpoint outputs/runs/fusion_final_seed42/checkpoints/best.pth \
  --manifest data/splits/test_seed42.txt \
  --output-dir outputs/evaluation/final_seed42 \
  --visualize outputs/evaluation/final_seed42/fusion_grid.png \
  --device cuda
```

Generate raw, heatmap, and overlay predictions. CUDA-trained checkpoints can be loaded on CPU or MPS:

```bash
python src/predict.py \
  --checkpoint outputs/runs/fusion_final_seed42/checkpoints/best.pth \
  --input dataset/images/images/val/COCO_val2014_000000000164.jpg \
  --output-dir outputs/predictions/fusion_final_seed42 \
  --save-raw --save-heatmap --save-overlay \
  --device cpu
```

The final run layout is:

```text
outputs/runs/<run_name>/
  checkpoints/best.pth
  checkpoints/last.pth
  training_log.csv
  config.json
  loss_curve.png
  cc_curve.png
```

Checkpoints, full run folders, and the local `report_assets/` workspace are ignored by Git.
Evaluation writes `evaluation_results.csv`, `per_image_metrics.csv`, a JSON summary, an optional qualitative grid, and `fusion_weights.csv` for FusionCNN. Training logs include epoch duration and peak CUDA memory.

## Metrics

| Metric | Direction | Definition |
|---|---|---|
| MSE | Lower | Mean squared pixel error |
| CC | Higher | Per-image Pearson correlation, averaged across the batch |
| SIM | Higher | Histogram intersection after each map is normalized to unit mass |

The combined loss is `MSE + lambda_cc * (1 - CC)`.

## Optional Docker CPU Check

```bash
docker compose build
docker compose run --rm saliency
```

Docker is not required for development, CUDA training, or Apple Silicon MPS.

## Troubleshooting

| Problem | Resolution |
|---|---|
| Dataset path not found | Pass the archive-native paths shown above or your own explicit directories |
| Missing saliency maps | Confirm every JPEG stem has a matching PNG stem |
| CUDA explicitly unavailable | Install a compatible NVIDIA driver and PyTorch CUDA wheel, or choose `--device cpu` |
| MPS explicitly unavailable | Use native macOS PyTorch on supported Apple Silicon, or choose `--device cpu` |
| Import failure | Run commands from the repository root |
| Fusion import fails | Pull the latest tested `main` or `dev` and install all requirements |

## Branch Policy

- No direct commits to `main`.
- Feature branches merge into `dev` through review.
- Compile, pytest, and CPU smoke must pass before merging into `dev`.
- `dev` merges into `main` only after tiny real-data train/resume/evaluate/predict checks and CPU/CUDA fusion smoke tests pass.
