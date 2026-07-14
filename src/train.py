from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.salicon_dataset import SaliconDataset
from src.losses.saliency_losses import combined_mse_cc_loss, mse_loss
from src.metrics.saliency_metrics import compute_cc, compute_mse, compute_sim
from src.models import get_model
from src.utils.checkpoint import load_checkpoint, make_state, save_checkpoint
from src.utils.device import DEVICE_CHOICES, get_device
from src.utils.paths import print_dirs, resolve_dirs, validate_dirs
from src.utils.seed import set_seed


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a saliency prediction model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--model", choices=["simple", "fusion"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--loss", choices=["mse", "mse_cc"])
    parser.add_argument("--lambda-cc", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=DEVICE_CHOICES)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--train-image-dir")
    parser.add_argument("--train-map-dir")
    parser.add_argument("--val-image-dir")
    parser.add_argument("--val-map-dir")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--resume")
    return parser.parse_args()


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return config


def merge_config(config: dict, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(config)
    training_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "loss": args.loss,
        "lambda_cc": args.lambda_cc,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "amp": args.amp,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
    }
    for key, value in training_overrides.items():
        if value is not None:
            config["training"][key] = value

    if args.model is not None:
        config["model"]["name"] = args.model
    if args.image_size is not None:
        config["preprocessing"]["image_size"] = args.image_size
    if args.output_dir is not None:
        config["outputs"]["root_dir"] = args.output_dir

    data_overrides = {
        "train_image_dir": args.train_image_dir,
        "train_map_dir": args.train_map_dir,
        "val_image_dir": args.val_image_dir,
        "val_map_dir": args.val_map_dir,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
    }
    for key, value in data_overrides.items():
        if value is not None:
            config["data"][key] = value
    return config


def validate_config(config: dict) -> None:
    training = config["training"]
    positive = {
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "image_size": config["preprocessing"]["image_size"],
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if training["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    if training["loss"] not in {"mse", "mse_cc"}:
        raise ValueError("training.loss must be 'mse' or 'mse_cc'")
    if training["lambda_cc"] < 0:
        raise ValueError("lambda_cc cannot be negative")


def build_loss(name: str, lambda_cc: float) -> LossFunction:
    if name == "mse":
        return mse_loss
    if name == "mse_cc":
        return lambda prediction, target: combined_mse_cc_loss(
            prediction, target, lambda_cc=lambda_cc
        )
    raise ValueError(f"Unsupported loss: {name}")


def plot_curves(log_path: Path, output_dir: Path) -> None:
    with log_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [float(row["train_loss"]) for row in rows], label="Train loss")
    axis.plot(epochs, [float(row["val_loss"]) for row in rows], label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axis.grid(True)
    axis.legend()
    figure.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [float(row["val_cc"]) for row in rows], label="Validation CC")
    axis.set(xlabel="Epoch", ylabel="CC", title="Validation correlation coefficient")
    axis.grid(True)
    axis.legend()
    figure.savefig(output_dir / "cc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _capture_random_state(loader: torch.utils.data.DataLoader) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if loader.generator is not None:
        state["loader_generator"] = loader.generator.get_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state: dict, loader: torch.utils.data.DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if loader.generator is not None and "loader_generator" in state:
        loader.generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _resolve_run_dir(config: dict, args: argparse.Namespace, model_name: str) -> tuple[Path, Path | None]:
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        if resume_path.parent.name != "checkpoints":
            raise ValueError("Resume checkpoint must be inside a run's checkpoints directory")
        run_dir = resume_path.parent.parent
        if args.run_name is not None and run_dir.name != args.run_name:
            raise ValueError("--run-name must match the resumed checkpoint's run directory")
        return run_dir, resume_path

    run_name = args.run_name or f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(config["outputs"]["root_dir"]) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and will not be overwritten: {run_dir}"
        )
    return run_dir, None


def _assert_resume_compatible(config: dict, checkpoint: dict) -> None:
    old_config = checkpoint.get("config", {})
    checks = (
        ("model", config["model"]["name"], checkpoint.get("model_name")),
        (
            "image_size",
            config["preprocessing"]["image_size"],
            old_config.get("preprocessing", {}).get("image_size"),
        ),
        ("loss", config["training"]["loss"], old_config.get("training", {}).get("loss")),
        (
            "lambda_cc",
            config["training"]["lambda_cc"],
            old_config.get("training", {}).get("lambda_cc"),
        ),
    )
    for name, current, previous in checks:
        if previous is not None and current != previous:
            raise ValueError(
                f"Cannot resume with changed {name}: checkpoint={previous!r}, current={current!r}"
            )


def train() -> None:
    args = parse_args()
    base_config = load_yaml_config(args.config)
    if args.resume:
        resume_preview = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(resume_preview.get("config"), dict):
            base_config = resume_preview["config"]
    config = merge_config(base_config, args)
    validate_config(config)
    set_seed(config["training"]["seed"])

    model_name = config["model"]["name"].lower().strip()
    if model_name not in {"simple", "fusion"}:
        raise ValueError("Only the simple and fusion models can be trained")

    run_dir, resume_path = _resolve_run_dir(config, args, model_name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dirs = resolve_dirs(
        train_image_dir=config["data"].get("train_image_dir"),
        train_map_dir=config["data"].get("train_map_dir"),
        val_image_dir=config["data"].get("val_image_dir"),
        val_map_dir=config["data"].get("val_map_dir"),
    )
    print_dirs(dirs)
    validate_dirs(dirs, required=["train_image_dir", "train_map_dir", "val_image_dir", "val_map_dir"])

    image_size = config["preprocessing"]["image_size"]
    train_dataset = SaliconDataset(
        image_dir=dirs["train_image_dir"],
        map_dir=dirs["train_map_dir"],
        manifest=config["data"].get("train_manifest"),
        image_size=image_size,
        augment=config["preprocessing"]["augmentation"]["horizontal_flip"],
        max_samples=config["training"]["train_samples"],
        require_maps=True,
    )
    val_dataset = SaliconDataset(
        image_dir=dirs["val_image_dir"],
        map_dir=dirs["val_map_dir"],
        manifest=config["data"].get("val_manifest"),
        image_size=image_size,
        augment=False,
        max_samples=config["training"]["val_samples"],
        require_maps=True,
    )

    device = get_device(config["training"]["device"])
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")
    loader_options = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
        "seed": config["training"]["seed"],
        "pin_memory": use_cuda,
    }
    train_loader = SaliconDataset.make_loader(train_dataset, shuffle=True, **loader_options)
    val_loader = SaliconDataset.make_loader(val_dataset, shuffle=False, **loader_options)

    config["resolved"] = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_manifest_hash": train_dataset.manifest_hash,
        "val_manifest_hash": val_dataset.manifest_hash,
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if use_cuda else None,
    }

    model = get_model(model_name, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    use_amp = bool(config["training"]["amp"] and use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = build_loss(config["training"]["loss"], config["training"]["lambda_cc"])

    start_epoch = 1
    best_cc = -math.inf
    if resume_path is not None:
        checkpoint_preview = torch.load(resume_path, map_location="cpu", weights_only=False)
        _assert_resume_compatible(config, checkpoint_preview)
        checkpoint = load_checkpoint(resume_path, model, device, optimizer=optimizer, scaler=scaler)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_cc = float(checkpoint.get("best_metric", -math.inf))
        if "random_state" in checkpoint_preview:
            _restore_random_state(checkpoint_preview["random_state"], train_loader)
        print(f"Resumed from epoch {start_epoch - 1} with best validation CC {best_cc:.6f}")

    epochs = int(config["training"]["epochs"])
    if start_epoch > epochs:
        raise ValueError(f"Resume checkpoint is already at epoch {start_epoch - 1}; epochs={epochs}")

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = run_dir / "training_log.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_mse",
        "val_cc",
        "val_sim",
        "learning_rate",
        "epoch_seconds",
        "peak_gpu_memory_mb",
    ]
    if resume_path is None:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
    elif not log_path.is_file():
        raise FileNotFoundError(f"Resume log not found: {log_path}")
    else:
        with log_path.open("r", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if not existing or int(existing[-1]["epoch"]) != start_epoch - 1:
            raise ValueError("Training log does not match the resume checkpoint epoch")

    non_blocking = use_cuda
    print(
        f"Training {model_name} with {config['training']['loss']} for epochs "
        f"{start_epoch}-{epochs} ({len(train_dataset)} train, {len(val_dataset)} validation samples)"
    )
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        train_loss_total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images = batch["image"].to(device, non_blocking=non_blocking)
            targets = batch["saliency"].to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                predictions = model(images)
                loss = loss_function(predictions, targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_total += loss.item() * images.size(0)

        model.eval()
        totals = {"loss": 0.0, "mse": 0.0, "cc": 0.0, "sim": 0.0}
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images = batch["image"].to(device, non_blocking=non_blocking)
                targets = batch["saliency"].to(device, non_blocking=non_blocking)
                predictions = model(images)
                batch_size = images.size(0)
                values = {
                    "loss": loss_function(predictions, targets),
                    "mse": compute_mse(predictions, targets),
                    "cc": compute_cc(predictions, targets),
                    "sim": compute_sim(predictions, targets),
                }
                if not all(torch.isfinite(value).item() for value in values.values()):
                    raise FloatingPointError(f"Non-finite validation metric at epoch {epoch}")
                for name, value in values.items():
                    totals[name] += value.item() * batch_size

        row = {
            "epoch": epoch,
            "train_loss": train_loss_total / len(train_dataset),
            "val_loss": totals["loss"] / len(val_dataset),
            "val_mse": totals["mse"] / len(val_dataset),
            "val_cc": totals["cc"] / len(val_dataset),
            "val_sim": totals["sim"] / len(val_dataset),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if use_cuda else 0.0
            ),
        }
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        plot_curves(log_path, run_dir)

        improved = row["val_cc"] > best_cc
        if improved:
            best_cc = row["val_cc"]
        state = make_state(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_cc,
            model_name=model_name,
            config=config,
            scaler=scaler,
        )
        state["random_state"] = _capture_random_state(train_loader)
        save_checkpoint(state, checkpoint_dir, "last.pth")
        if improved:
            save_checkpoint(state, checkpoint_dir, "best.pth")

        print(
            f"Epoch {epoch}/{epochs}: train={row['train_loss']:.6f}, "
            f"val_mse={row['val_mse']:.6f}, val_cc={row['val_cc']:.6f}, "
            f"val_sim={row['val_sim']:.6f}, seconds={row['epoch_seconds']:.1f}, "
            f"peak_mb={row['peak_gpu_memory_mb']:.1f}"
        )

    print(f"Training completed successfully: {run_dir}")


if __name__ == "__main__":
    train()
