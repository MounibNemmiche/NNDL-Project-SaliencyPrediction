from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    _CMAP = matplotlib.colormaps["jet"]
except AttributeError:
    _CMAP = cm.get_cmap("jet")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _STD + _MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def make_heatmap(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    arr = tensor.cpu().numpy().astype(np.float32)
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8)
    rgba = _CMAP(arr)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def save_prediction_grid(
    images:    torch.Tensor,
    preds:     torch.Tensor,
    targets:   Optional[torch.Tensor],
    filenames: List[str],
    save_path: str | Path,
    max_items: int = 8,
) -> None:
    n     = min(len(filenames), max_items)
    n_col = 3 if targets is not None else 2
    col_labels = ["Image", "Ground Truth", "Prediction"] if targets is not None \
                 else ["Image", "Prediction"]

    fig, axes = plt.subplots(n, n_col, figsize=(4 * n_col, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row in range(n):
        img_np  = denormalize_image(images[row])
        pred_np = make_heatmap(preds[row])

        axes[row, 0].imshow(img_np)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title(col_labels[0], fontsize=10)

        if targets is not None:
            gt_np = make_heatmap(targets[row])
            axes[row, 1].imshow(gt_np)
            axes[row, 1].axis("off")
            if row == 0:
                axes[row, 1].set_title(col_labels[1], fontsize=10)
            axes[row, 2].imshow(pred_np)
            axes[row, 2].axis("off")
            if row == 0:
                axes[row, 2].set_title(col_labels[2], fontsize=10)
        else:
            axes[row, 1].imshow(pred_np)
            axes[row, 1].axis("off")
            if row == 0:
                axes[row, 1].set_title(col_labels[1], fontsize=10)

        axes[row, 0].set_ylabel(filenames[row][:20], fontsize=7)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_overlay(
    image:     torch.Tensor,
    pred:      torch.Tensor,
    save_path: str | Path,
    alpha:     float = 0.5,
) -> None:
    img_np  = denormalize_image(image).astype(np.float32)
    heat_np = make_heatmap(pred).astype(np.float32)
    overlay = np.clip((1 - alpha) * img_np + alpha * heat_np, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import tempfile

    img  = torch.randn(3, 64, 64)
    pred = torch.rand(1, 64, 64)
    gt   = torch.rand(1, 64, 64)

    with tempfile.TemporaryDirectory() as tmp:
        grid_path    = os.path.join(tmp, "grid.png")
        overlay_path = os.path.join(tmp, "overlay.png")

        save_prediction_grid(
            images=img.unsqueeze(0), preds=pred.unsqueeze(0),
            targets=gt.unsqueeze(0), filenames=["test.jpg"],
            save_path=grid_path,
        )
        save_overlay(img, pred, save_path=overlay_path)

        assert os.path.exists(grid_path)
        assert os.path.exists(overlay_path)

    arr = denormalize_image(img)
    assert arr.shape == (64, 64, 3)
    print("visualization.py OK")
