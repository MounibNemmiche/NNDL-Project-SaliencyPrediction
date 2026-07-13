from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.salicon_dataset import SaliconDataset
from src.losses.saliency_losses import combined_mse_cc_loss
from src.metrics.saliency_metrics import compute_cc, compute_mse, compute_sim
from src.models.center_bias import CenterBiasBaseline
from src.models.simple_cnn import SimpleCNN
from src.utils.device import DEVICE_CHOICES, get_device
from src.utils.seed import set_seed
from src.utils.visualization import save_overlay, save_prediction_grid, save_raw_saliency_map


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_temporary_dataset(root: Path, count: int = 8, size: int = 72) -> tuple[Path, Path]:
    image_dir = root / "images"
    map_dir = root / "maps"
    image_dir.mkdir()
    map_dir.mkdir()

    yy, xx = np.mgrid[:size, :size]
    for index in range(count):
        rng = np.random.default_rng(1000 + index)
        image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        center_x = size * (0.35 + 0.04 * index)
        center_y = size * (0.45 + 0.02 * index)
        gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * (size * 0.16) ** 2))
        saliency = np.round(gaussian / gaussian.max() * 255.0).astype(np.uint8)
        stem = f"smoke_{index:02d}"
        Image.fromarray(image, mode="RGB").save(image_dir / f"{stem}.jpg")
        Image.fromarray(saliency, mode="L").save(map_dir / f"{stem}.png")
    return image_dir, map_dir


def _load_fusion_class(require_fusion: bool):
    try:
        from src.models.multiscale_fusion_cnn import MultiScaleFusionCNN
    except ImportError as error:
        if require_fusion:
            raise RuntimeError(
                "Fusion integration is required, but MultiScaleFusionCNN is unavailable"
            ) from error
        return None
    return MultiScaleFusionCNN


def _validate_output(output: torch.Tensor, batch_size: int, image_size: int, name: str) -> None:
    _check(output.shape == (batch_size, 1, image_size, image_size), f"{name} output shape")
    _check(torch.isfinite(output).all().item(), f"{name} output is finite")
    _check(output.min().item() >= 0.0 and output.max().item() <= 1.0, f"{name} range")


def run_smoke_test(device: torch.device, require_fusion: bool = False) -> None:
    set_seed(42)
    image_size = 64
    with tempfile.TemporaryDirectory(prefix="saliency_smoke_") as temp_dir:
        root = Path(temp_dir)
        image_dir, map_dir = _write_temporary_dataset(root)
        dataset = SaliconDataset(
            image_dir=image_dir,
            map_dir=map_dir,
            image_size=image_size,
            augment=False,
            require_maps=True,
        )
        loader = SaliconDataset.make_loader(dataset, batch_size=2, shuffle=False, seed=42)
        batch = next(iter(loader))
        images = batch["image"].to(device)
        targets = batch["saliency"].to(device)
        filenames = batch["filename"]
        _check(len(dataset) == 8, "temporary dataset size")
        _check(images.dtype == torch.float32, "image dtype")
        _check(targets.dtype == torch.float32, "saliency dtype")

        center = CenterBiasBaseline(image_size=image_size).to(device).eval()
        simple = SimpleCNN(image_size=image_size).to(device)
        with torch.no_grad():
            center_output = center(images)
            simple_output = simple.eval()(images)
        _validate_output(center_output, 2, image_size, "CenterBiasBaseline")
        _validate_output(simple_output, 2, image_size, "SimpleCNN")

        optimizer = torch.optim.Adam(simple.parameters(), lr=1e-4)
        simple.train()
        optimizer.zero_grad(set_to_none=True)
        train_output = simple(images)
        train_loss = combined_mse_cc_loss(train_output, targets, lambda_cc=0.1)
        _check(torch.isfinite(train_loss).item(), "training loss is finite")
        train_loss.backward()
        gradients = [parameter.grad for parameter in simple.parameters() if parameter.requires_grad]
        _check(all(gradient is not None for gradient in gradients), "all SimpleCNN gradients exist")
        _check(all(torch.isfinite(gradient).all().item() for gradient in gradients), "gradients are finite")
        optimizer.step()

        predictions: dict[str, torch.Tensor] = {
            "center": center_output,
            "simple": simple_output,
        }
        fusion_class = _load_fusion_class(require_fusion)
        if fusion_class is None:
            print("Fusion integration pending: Part A smoke test only.")
        else:
            fusion = fusion_class(image_size=image_size).to(device)
            fusion.train()
            fusion_output = fusion(images)
            _validate_output(fusion_output, 2, image_size, "MultiScaleFusionCNN")
            side_outputs = fusion(images, return_side_outputs=True)
            expected_keys = {"final", "side1", "side2", "side3", "main"}
            _check(set(side_outputs) == expected_keys, "fusion side-output keys")
            for name, output in side_outputs.items():
                _validate_output(output, 2, image_size, f"fusion {name}")
            fusion_loss = combined_mse_cc_loss(fusion_output, targets, lambda_cc=0.1)
            fusion_loss.backward()
            _check(
                all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                    for parameter in fusion.parameters()
                    if parameter.requires_grad
                ),
                "all fusion gradients are finite",
            )
            predictions["fusion"] = fusion_output.detach()

        for name, prediction in predictions.items():
            metrics = (
                compute_mse(prediction, targets),
                compute_cc(prediction, targets),
                compute_sim(prediction, targets),
            )
            _check(all(torch.isfinite(metric).item() for metric in metrics), f"{name} metrics")
            _check(0.0 <= metrics[2].item() <= 1.0, f"{name} SIM range")

        grid_path = root / "prediction_grid.png"
        overlay_path = root / "overlay.png"
        raw_path = root / "raw_prediction.png"
        save_prediction_grid(
            images=images.detach().cpu(),
            preds=simple_output.detach().cpu(),
            targets=targets.detach().cpu(),
            filenames=filenames,
            save_path=grid_path,
        )
        save_overlay(images[0].detach().cpu(), simple_output[0].detach().cpu(), overlay_path)
        save_raw_saliency_map(simple_output[0].detach().cpu(), raw_path)
        _check(all(path.is_file() and path.stat().st_size > 0 for path in (grid_path, overlay_path, raw_path)), "visualizations exist")

    print("Smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end saliency pipeline smoke test")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument(
        "--require-fusion",
        action="store_true",
        help="Fail unless Student B's fusion model is installed and passes all checks",
    )
    args = parser.parse_args()
    run_smoke_test(get_device(args.device), require_fusion=args.require_fusion)


if __name__ == "__main__":
    main()
