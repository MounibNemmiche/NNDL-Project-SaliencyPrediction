# SALICON Dataset

The dataset is local-only and must never be committed. This project uses the 10,000 labeled SALICON training samples and the 5,000 labeled validation samples. The unlabeled test images and raw `.mat` fixation files are not needed for model training or the final local comparison.

Dataset references:

- Official project: <https://salicon.net/>
- Public mirror identified in the project plan: <https://www.kaggle.com/datasets/roshan401/salicon>

The final report must identify the exact archive or mirror actually shared by the team.

## Team Layout

The downloaded archives currently use this layout:

```text
dataset/
  images/images/
    train/             10,000 COCO JPEG images
    val/                5,000 COCO JPEG images
    test/               5,000 unlabeled JPEG images
  maps/
    train/             10,000 saliency PNG maps
    val/                5,000 saliency PNG maps
  fixations/           raw MAT files; unused by this project
```

Image and map stems must match, for example:

```text
COCO_train2014_000000000009.jpg
COCO_train2014_000000000009.png
```

Do not move or duplicate the dataset. Pass these directories to the CLI, or use the defaults in `config.yaml`.

## Deterministic Validation and Test Split

The official SALICON validation data is divided into a 500-sample validation set for checkpoint selection and a disjoint 4,500-sample final test set. Generate the tracked filename manifests with:

```bash
python -m src.datasets.create_splits \
  --image-dir dataset/images/images/val \
  --map-dir dataset/maps/val \
  --val-size 500 \
  --seed 42 \
  --val-output data/splits/val_seed42.txt \
  --test-output data/splits/test_seed42.txt
```

Never tune architecture, loss, or hyperparameters on `test_seed42.txt`.

## Reproducible Read Check

```bash
python -m src.datasets.salicon_dataset \
  --image_dir dataset/images/images/train \
  --map_dir dataset/maps/train \
  --max_samples 4
```

## Alternative Layouts

The loader does not hard-code the team layout. A professor may instead use:

```text
data/SALICON/train_images
data/SALICON/train_maps
data/SALICON/val_images
data/SALICON/val_maps
```

In that case, pass those paths explicitly. Dataset archives, images, maps, credentials, and private download links remain ignored by Git.
