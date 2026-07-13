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
from src.models import get_model
from src.models.center_bias import CenterBiasBaseline
from src.models.multiscale_fusion_cnn import MultiScaleFusionCNN
from src.models.simple_cnn import SimpleCNN
from src.utils.checkpoint import load_checkpoint, make_state, save_checkpoint
from src.utils.device import DEVICE_CHOICES, get_device
from src.utils.paths import resolve_dirs, validate_dirs
from src.utils.seed import set_seed
from src.utils.visualization import (
    save_model_comparison,
    save_overlay,
    save_prediction_grid,
    save_raw_saliency_map,
)


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
        sigma = size * 0.16
        gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2))
        saliency = np.round(gaussian / gaussian.max() * 255.0).astype(np.uint8)
        stem = f"smoke_{index:02d}"
        Image.fromarray(image, mode="RGB").save(image_dir / f"{stem}.jpg")
        Image.fromarray(saliency, mode="L").save(map_dir / f"{stem}.png")
    return image_dir, map_dir


def _validate_output(output: torch.Tensor, batch_size: int, image_size: int, name: str) -> None:
    _check(output.shape == (batch_size, 1, image_size, image_size), f"{name} output shape")
    _check(torch.isfinite(output).all().item(), f"{name} output is finite")
    _check(output.min().item() >= 0.0 and output.max().item() <= 1.0, f"{name} range")


def _check_gradients(model: torch.nn.Module, name: str) -> None:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    _check(gradients and all(gradient is not None for gradient in gradients), f"{name} gradients exist")
    _check(
        all(torch.isfinite(gradient).all().item() for gradient in gradients if gradient is not None),
        f"{name} gradients are finite",
    )


def run_smoke_test(device: torch.device) -> None:
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
        _check(len(dataset) == 8 and dataset.has_maps, "temporary dataset contract")
        _check(images.dtype == torch.float32, "image dtype")
        _check(targets.dtype == torch.float32, "saliency dtype")

        center = CenterBiasBaseline(image_size=image_size).to(device).eval()
        simple = SimpleCNN(image_size=image_size).to(device)
        fusion = MultiScaleFusionCNN(image_size=image_size).to(device)
        with torch.no_grad():
            center_output = center(images)
            simple_output = simple.eval()(images)
            fusion_output = fusion.eval()(images)
            side_outputs = fusion(images, return_side_outputs=True)
        _validate_output(center_output, 2, image_size, "CenterBiasBaseline")
        _validate_output(simple_output, 2, image_size, "SimpleCNN")
        _validate_output(fusion_output, 2, image_size, "MultiScaleFusionCNN")
        expected_keys = {"final", "side1", "side2", "side3", "main"}
        _check(set(side_outputs) == expected_keys, "fusion side-output keys")
        for name, output in side_outputs.items():
            _validate_output(output, 2, image_size, f"fusion {name}")

        simple_parameters = sum(parameter.numel() for parameter in simple.parameters())
        fusion_parameters = sum(parameter.numel() for parameter in fusion.parameters())
        _check(fusion_parameters > simple_parameters, "fusion adds parameters")
        _check(
            fusion_parameters - simple_parameters < simple_parameters * 0.01,
            "fusion overhead remains below one percent",
        )
        _check(isinstance(get_model("center"), CenterBiasBaseline), "center registry")
        _check(isinstance(get_model("simple"), SimpleCNN), "simple registry")
        _check(isinstance(get_model("fusion"), MultiScaleFusionCNN), "fusion registry")

        simple_optimizer = torch.optim.Adam(simple.parameters(), lr=1e-4)
        simple.train()
        simple_optimizer.zero_grad(set_to_none=True)
        simple_train_output = simple(images)
        simple_loss = combined_mse_cc_loss(simple_train_output, targets, lambda_cc=0.1)
        _check(torch.isfinite(simple_loss).item(), "SimpleCNN loss is finite")
        simple_loss.backward()
        _check_gradients(simple, "SimpleCNN")
        simple_optimizer.step()

        fusion_optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-4)
        fusion.train()
        fusion_optimizer.zero_grad(set_to_none=True)
        fusion_train_output = fusion(images)
        fusion_loss = combined_mse_cc_loss(fusion_train_output, targets, lambda_cc=0.1)
        _check(torch.isfinite(fusion_loss).item(), "FusionCNN loss is finite")
        fusion_loss.backward()
        _check_gradients(fusion, "FusionCNN")
        fusion_optimizer.step()

        predictions = {"center": center_output, "simple": simple_output, "fusion": fusion_output}
        for name, prediction in predictions.items():
            metrics = (
                compute_mse(prediction, targets),
                compute_cc(prediction, targets),
                compute_sim(prediction, targets),
            )
            _check(all(torch.isfinite(metric).item() for metric in metrics), f"{name} metrics")
            _check(0.0 <= metrics[2].item() <= 1.0, f"{name} SIM range")

        checkpoint_state = make_state(
            model=simple,
            optimizer=simple_optimizer,
            epoch=1,
            best_metric=0.5,
            model_name="simple",
            config={"seed": 42, "split_hash": dataset.manifest_hash},
        )
        checkpoint_path = save_checkpoint(checkpoint_state, root / "checkpoints", "smoke.pth")
        restored = SimpleCNN(image_size=image_size).to(device)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
        metadata = load_checkpoint(
            checkpoint_path, restored, device, optimizer=restored_optimizer
        )
        _check(metadata["epoch"] == 1 and metadata["model_name"] == "simple", "checkpoint metadata")
        for expected, actual in zip(simple.parameters(), restored.parameters()):
            _check(torch.equal(expected, actual), "checkpoint model parameters")

        canonical_root = root / "canonical_data"
        for directory in ("train_images", "train_maps", "val_images", "val_maps"):
            (canonical_root / directory).mkdir(parents=True)
        resolved = resolve_dirs(data_root=str(canonical_root))
        validate_dirs(
            resolved,
            required=["train_image_dir", "train_map_dir", "val_image_dir", "val_map_dir"],
        )

        grid_path = root / "prediction_grid.png"
        overlay_path = root / "overlay.png"
        raw_path = root / "raw_prediction.png"
        comparison_path = root / "model_comparison.png"
        cpu_images = images.detach().cpu()
        cpu_targets = targets.detach().cpu()
        save_prediction_grid(
            images=cpu_images,
            preds=simple_output.detach().cpu(),
            targets=cpu_targets,
            filenames=filenames,
            save_path=grid_path,
        )
        save_overlay(cpu_images[0], simple_output[0].detach().cpu(), overlay_path)
        save_raw_saliency_map(simple_output[0].detach().cpu(), raw_path)
        save_model_comparison(
            images=cpu_images,
            targets=cpu_targets,
            predictions={name: output.detach().cpu() for name, output in predictions.items()},
            filenames=filenames,
            output_path=comparison_path,
        )
        artifacts = (grid_path, overlay_path, raw_path, comparison_path, checkpoint_path)
        _check(all(path.is_file() and path.stat().st_size > 0 for path in artifacts), "artifacts exist")

    print("Smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end saliency pipeline smoke test")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args()
    run_smoke_test(get_device(args.device))


if __name__ == "__main__":
    main()
