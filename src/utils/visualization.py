from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


_CMAP = matplotlib.colormaps["magma"]
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] image tensor, got {tensor.shape}")
    image = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    image = np.clip((image * _STD + _MEAN) * 255.0, 0, 255)
    return image.astype(np.uint8)


def saliency_to_array(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().squeeze().numpy()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D saliency map, got shape {array.shape}")
    return np.clip(array, 0.0, 1.0)


def make_heatmap(tensor: torch.Tensor) -> np.ndarray:
    rgba = _CMAP(saliency_to_array(tensor))
    return (rgba[:, :, :3] * 255.0).astype(np.uint8)


def save_raw_saliency_map(tensor: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grayscale = (saliency_to_array(tensor) * 255.0).round().astype(np.uint8)
    Image.fromarray(grayscale, mode="L").save(output_path)


def save_prediction_grid(
    images: torch.Tensor,
    preds: torch.Tensor,
    targets: Optional[torch.Tensor],
    filenames: Sequence[str],
    save_path: str | Path,
    max_items: int = 8,
) -> None:
    n_items = min(len(filenames), max_items)
    if n_items <= 0:
        raise ValueError("At least one item is required")

    columns = 3 if targets is not None else 2
    labels = ["Image", "Ground truth", "Prediction"] if targets is not None else ["Image", "Prediction"]
    figure, axes = plt.subplots(n_items, columns, figsize=(3.2 * columns, 2.8 * n_items))
    axes = np.asarray(axes).reshape(n_items, columns)

    for row in range(n_items):
        axes[row, 0].imshow(denormalize_image(images[row]))
        if targets is not None:
            axes[row, 1].imshow(make_heatmap(targets[row]))
            axes[row, 2].imshow(make_heatmap(preds[row]))
        else:
            axes[row, 1].imshow(make_heatmap(preds[row]))
        for column in range(columns):
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(labels[column], fontsize=10)
        axes[row, 0].set_ylabel(Path(filenames[row]).stem[-12:], fontsize=7)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_model_comparison(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    filenames: Sequence[str],
    output_path: str | Path,
    max_items: int = 4,
) -> None:
    if not predictions:
        raise ValueError("At least one model prediction is required")
    n_items = min(len(filenames), max_items)
    labels = ["Image", "Ground truth", *predictions.keys()]
    columns = len(labels)
    figure, axes = plt.subplots(n_items, columns, figsize=(2.7 * columns, 2.7 * n_items))
    axes = np.asarray(axes).reshape(n_items, columns)

    for row in range(n_items):
        axes[row, 0].imshow(denormalize_image(images[row]))
        axes[row, 1].imshow(make_heatmap(targets[row]))
        for offset, prediction in enumerate(predictions.values(), start=2):
            axes[row, offset].imshow(make_heatmap(prediction[row]))
        for column, label in enumerate(labels):
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(label, fontsize=9)
        axes[row, 0].set_ylabel(Path(filenames[row]).stem[-12:], fontsize=7)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_overlay(
    image: torch.Tensor,
    pred: torch.Tensor,
    save_path: str | Path,
    alpha: float = 0.5,
) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    image_array = denormalize_image(image).astype(np.float32)
    heatmap = make_heatmap(pred).astype(np.float32)
    overlay = np.clip((1.0 - alpha) * image_array + alpha * heatmap, 0, 255).astype(np.uint8)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(overlay)
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
