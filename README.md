# Saliency Prediction — SALICON

Lightweight Multi-Scale CNN for Vision-Based Saliency Prediction on the SALICON dataset.

---

## Requirements

- Python 3.10
- PyTorch ≥ 2.0 with the appropriate backend (CUDA / MPS / CPU)

---

## Setup

### Option A — venv (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — venv (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option C — Conda
```bash
conda env create -f environment.yml
conda activate saliency
```

### Verify installation
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
# Apple Silicon
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## Dataset

See [`data/README.md`](data/README.md) for download and folder structure.

---

## Smoke Test

The smoke test verifies the full Part A pipeline using **synthetic data** — no real dataset required.

```bash
python src/smoke_test.py --device auto    # pick best available device
python src/smoke_test.py --device cpu     # force CPU
python src/smoke_test.py --device cuda    # force CUDA
python src/smoke_test.py --device mps     # Apple Silicon
```

Expected output: `Smoke test passed.`

---

## Training

```bash
# SimpleCNN — 20 epochs on full SALICON
python src/train.py \
  --train_image_dir data/SALICON/train_images \
  --train_map_dir   data/SALICON/train_maps \
  --val_image_dir   data/SALICON/val_images \
  --val_map_dir     data/SALICON/val_maps \
  --model simple \
  --epochs 20 \
  --batch_size 8 \
  --image_size 224 \
  --loss mse \
  --device auto

# MultiScaleFusionCNN (Student B)
python src/train.py ... --model fusion ...

# Limit samples for a quick dry run
python src/train.py ... --train_samples 2000 --val_samples 500 ...

# Resume interrupted training
python src/train.py ... --resume outputs/checkpoints/simple_last.pth ...
```

---

## Evaluation

```bash
# Center-bias baseline (no checkpoint needed)
python src/evaluate.py \
  --test_image_dir data/SALICON/val_images \
  --test_map_dir   data/SALICON/val_maps \
  --model center \
  --device auto

# SimpleCNN
python src/evaluate.py \
  --test_image_dir data/SALICON/val_images \
  --test_map_dir   data/SALICON/val_maps \
  --model simple \
  --checkpoint outputs/checkpoints/simple_best.pth \
  --device auto

# FusionCNN (Student B)
python src/evaluate.py ... --model fusion --checkpoint outputs/checkpoints/fusion_best.pth ...
```

---

## Prediction (single image)

```bash
python src/predict.py \
  --image path/to/image.jpg \
  --model fusion \
  --checkpoint outputs/checkpoints/fusion_best.pth \
  --device auto
```

---

## Device Support

| Platform | Command |
|---|---|
| NVIDIA GPU | `--device cuda` |
| Apple Silicon | `--device mps` |
| CPU | `--device cpu` |
| Auto-select | `--device auto` (default) |

> **Note for Apple Silicon users**: MPS is preferred over CPU. Docker is CPU-only.

---

## Optional Docker (CPU smoke test only)

```bash
docker compose run --rm saliency python src/smoke_test.py --device cpu
```

Docker is **not required** for normal development.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA not available` | Install the CUDA-compatible PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) |
| `MPS not available` | Requires macOS 12.3+ and PyTorch ≥ 1.12 |
| `FileNotFoundError` on dataset | Check `data/README.md` for correct folder structure |
| `ModuleNotFoundError` | Run from the project root, not from inside `src/` |

---

## Project Structure

```
src/
  datasets/salicon_dataset.py   # SALICON loader (Student A)
  models/
    center_bias.py              # Gaussian prior baseline (Student A)
    simple_cnn.py               # Encoder-decoder baseline (Student A)
    multiscale_fusion_cnn.py    # Proposed model (Student B)
  losses/saliency_losses.py     # MSE + CC losses (Student A)
  metrics/saliency_metrics.py   # MSE, CC, SIM metrics (Student A)
  utils/
    device.py                   # Cross-platform device selection (Student A)
    seed.py                     # Reproducibility helper (Student A)
    visualization.py            # Grid + overlay visualization (Student A)
    checkpoint.py               # Checkpoint save/load (Student B)
    paths.py                    # Path utilities (Student B)
  smoke_test.py                 # Part A acceptance test (Student A)
  train.py                      # Training loop (Student B)
  evaluate.py                   # Evaluation script (Student B)
  predict.py                    # Single-image prediction (Student B)
data/
  README.md
  SALICON/
    train_images/  val_images/  test_images/
    train_maps/    val_maps/    test_maps/
outputs/
  checkpoints/  logs/  plots/  visualizations/
report_assets/
scripts/
```
