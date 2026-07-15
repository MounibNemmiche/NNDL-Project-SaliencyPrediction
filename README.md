# Lightweight Multi-Scale Saliency Prediction

PyTorch implementation of vision-based saliency prediction on SALICON. The final study compares a fixed center-bias prior, a lightweight SimpleCNN encoder-decoder, and a MultiScaleFusionCNN that adds side outputs and learned scale fusion to the same backbone.

## Current Integration Status

The dataset, three models, checkpoint/path utilities, and executable train/evaluate/predict pipeline are fully integrated and tested. Six final learned-model runs (two architectures and seeds 42, 7, and 123) were trained on an NVIDIA GeForce RTX 4060 Ti, as recorded in the saved run metadata. Their checkpoints were hash-verified and independently re-evaluated on an RTX 4060 Laptop GPU.

The complete training checkpoints, metrics CSVs, curves, qualitative prediction grids, and fusion weights can be downloaded from the Google Drive folder:
**[Google Drive Benchmark Results](https://drive.google.com/drive/folders/1lc0stHBMINM8bdDuhqfzOeHfQdkbwt0u?usp=sharing)**

All models were evaluated on the same fixed 4,500-image test manifest (`test_seed42.txt`). Checkpoint selection used validation CC only.

| Model | MSE (lower) | CC (higher) | SIM (higher) | Parameters |
|---|---:|---:|---:|---:|
| Center bias | 0.110151 | 0.537825 | 0.526015 | 0 |
| SimpleCNN | 0.015306 +/- 0.000982 | 0.788157 +/- 0.002919 | 0.624782 +/- 0.012700 | 979,713 |
| MultiScaleFusionCNN | **0.014977 +/- 0.000371** | **0.789525 +/- 0.001797** | **0.649212 +/- 0.009487** | 979,945 |

Learned-model values are mean +/- sample standard deviation over the three seeds. Compared with SimpleCNN, fusion lowers mean MSE by 2.15%, raises CC by 0.001368, and raises SIM by 0.024431 (3.91% relative) while adding only 232 parameters. Fusion improves SIM in every seed; the MSE and CC gains are more initialization-dependent.

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

On an NVIDIA system, also run the mandatory CUDA smoke test:

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

The final experiment contract is 25 epochs, batch size 64, Adam with learning rate `1e-4`, `MSE + 0.1 * (1 - CC)`, no mixed precision, and seeds 42, 7, and 123. The scripts reproduce these settings and use the historical seed-42 run names (`simple_final` and `fusion_final`) so their paths match the submitted evidence:

```bash
for seed in 42 7 123; do
  SEED="$seed" bash scripts/train_simple.sh
  SEED="$seed" bash scripts/train_fusion.sh
done
```

Evaluate all three predictors on the fixed 4,500-image manifest after setting the two checkpoint paths:

```bash
SIMPLE_CHECKPOINT=outputs/runs/simple_final/checkpoints/best.pth \
FUSION_CHECKPOINT=outputs/runs/fusion_final/checkpoints/best.pth \
bash scripts/evaluate_all.sh
```

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
  --checkpoint outputs/runs/fusion_final/checkpoints/best.pth \
  --manifest data/splits/test_seed42.txt \
  --output-dir outputs/evaluation/final_seed42 \
  --visualize outputs/evaluation/final_seed42/fusion_grid.png \
  --device cuda
```

Generate raw, heatmap, and overlay predictions. CUDA-trained checkpoints can be loaded on CPU or MPS:

```bash
python src/predict.py \
  --checkpoint outputs/runs/fusion_final/checkpoints/best.pth \
  --input dataset/images/images/val/COCO_val2014_000000000164.jpg \
  --output-dir outputs/predictions/fusion_final \
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
