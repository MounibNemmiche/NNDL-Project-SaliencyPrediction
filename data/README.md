# Dataset — SALICON

The SALICON dataset is **not committed to Git**. Download and place files manually.

---

## Expected Folder Structure

```
data/
  SALICON/
    train_images/       ← COCO train 2014 images used by SALICON (~10k JPEGs)
    train_maps/         ← corresponding saliency maps (grayscale PNGs)
    val_images/         ← COCO val 2014 images used by SALICON (~5k JPEGs)
    val_maps/           ← corresponding saliency maps (grayscale PNGs)
    test_images/        ← COCO test images (no maps — for blind prediction)
    test_maps/          ← leave empty or omit
```

Filenames must match: `COCO_train2014_000000XXXXXX.jpg` ↔ `COCO_train2014_000000XXXXXX.png`

---

## Download

### Option 1 — Official SALICON website
1. Register at http://salicon.net/
2. Download:
   - `train_images.zip` (images)
   - `val_images.zip`
   - `test_images.zip`
   - `train_maps.zip` (saliency maps)
   - `val_maps.zip`
3. Extract each archive into the corresponding folder above.

### Option 2 — Kaggle (unofficial mirror)
Search for "SALICON saliency dataset" on Kaggle.

---

## Alternative Paths

If your data lives elsewhere, pass paths explicitly to every script:

```bash
python src/train.py \
  --train_image_dir /path/to/your/train_images \
  --train_map_dir   /path/to/your/train_maps \
  --val_image_dir   /path/to/your/val_images \
  --val_map_dir     /path/to/your/val_maps \
  ...
```

---

## Smoke Test (no real data required)

The smoke test uses **synthetic random tensors** and does not need real images:

```bash
python src/smoke_test.py --device cpu
```

---

## Gitignore Notes

The following are automatically excluded from Git (see `.gitignore`):
- `data/SALICON/**` — all images and maps
- `*.zip`, `*.tar`, `*.tar.gz` — raw archives
- `outputs/checkpoints/*.pth` — model checkpoints
