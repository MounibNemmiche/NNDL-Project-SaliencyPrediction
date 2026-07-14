from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.salicon_dataset import SaliconDataset
from src.losses.saliency_losses import combined_mse_cc_loss
from src.metrics.saliency_metrics import compute_mse, compute_cc, compute_sim
from src.models import get_model
from src.utils.checkpoint import make_state, save_checkpoint, load_checkpoint
from src.utils.device import get_device, DEVICE_CHOICES
from src.utils.paths import resolve_dirs, validate_dirs, print_dirs
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train simple or fusion saliency prediction models.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run folder.")
    parser.add_argument("--model", type=str, choices=["simple", "fusion", "center"], default=None, help="Override model name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", type=str, choices=DEVICE_CHOICES, default=None, help="Override device preference.")
    parser.add_argument("--lambda-cc", type=float, default=None, help="Override lambda_cc parameter.")
    parser.add_argument("--amp", action="store_true", default=None, help="Override AMP usage (enable).")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Override AMP usage (disable).")
    parser.add_argument("--train-samples", type=int, default=None, help="Limit number of training samples.")
    parser.add_argument("--val-samples", type=int, default=None, help="Limit number of validation samples.")
    
    # Path overrides
    parser.add_argument("--train-image-dir", type=str, default=None)
    parser.add_argument("--train-map-dir", type=str, default=None)
    parser.add_argument("--val-image-dir", type=str, default=None)
    parser.add_argument("--val-map-dir", type=str, default=None)
    parser.add_argument("--val-manifest", type=str, default=None)

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(config: dict, args: argparse.Namespace) -> dict:
    # Merge CLI arguments into configuration dict
    if args.model is not None:
        config["model"]["name"] = args.model
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.lambda_cc is not None:
        config["training"]["lambda_cc"] = args.lambda_cc
    if args.amp is not None:
        config["training"]["amp"] = args.amp
    if args.train_samples is not None:
        config["training"]["train_samples"] = args.train_samples
    if args.val_samples is not None:
        config["training"]["val_samples"] = args.val_samples
        
    # Paths
    if args.train_image_dir is not None:
        config["data"]["train_image_dir"] = args.train_image_dir
    if args.train_map_dir is not None:
        config["data"]["train_map_dir"] = args.train_map_dir
    if args.val_image_dir is not None:
        config["data"]["val_image_dir"] = args.val_image_dir
    if args.val_map_dir is not None:
        config["data"]["val_map_dir"] = args.val_map_dir
    if args.val_manifest is not None:
        config["data"]["val_manifest"] = args.val_manifest
        
    return config


def plot_curves(log_path: Path, output_dir: Path) -> None:
    epochs, train_losses, val_losses, val_ccs = [], [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            val_ccs.append(float(row["val_cc"]))
            
    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", color="red", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Combined Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot CC curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_ccs, label="Val CC", color="orange", marker="^")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson Correlation Coefficient (CC)")
    plt.title("Validation CC Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "cc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def train() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config = merge_config(config, args)

    # Set random seed for reproducibility
    set_seed(config["training"]["seed"])

    # Model checks
    model_name = config["model"]["name"].lower().strip()
    if model_name == "center":
        raise ValueError(
            "The 'center' model is a fixed Gaussian baseline and cannot be trained. "
            "Please use evaluate.py to benchmark it, or select 'simple' or 'fusion' to train."
        )

    # Setup directories
    if not args.run_name:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{model_name}_{timestamp}"
    else:
        run_name = args.run_name

    outputs_root = Path(config["outputs"]["root_dir"])
    run_dir = outputs_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save finalized configuration for reproducibility
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # Resolve and validate dataset directories
    dirs = resolve_dirs(
        train_image_dir=config["data"].get("train_image_dir"),
        train_map_dir=config["data"].get("train_map_dir"),
        val_image_dir=config["data"].get("val_image_dir"),
        val_map_dir=config["data"].get("val_map_dir"),
    )
    print_dirs(dirs)
    validate_dirs(dirs, required=["train_image_dir", "train_map_dir", "val_image_dir", "val_map_dir"])

    # Load datasets
    image_size = config["preprocessing"]["image_size"]
    hflip = config["preprocessing"]["augmentation"]["horizontal_flip"]
    
    train_dataset = SaliconDataset(
        image_dir=dirs["train_image_dir"],
        map_dir=dirs["train_map_dir"],
        image_size=image_size,
        augment=hflip,
        max_samples=config["training"]["train_samples"],
    )
    val_dataset = SaliconDataset(
        image_dir=dirs["val_image_dir"],
        map_dir=dirs["val_map_dir"],
        image_size=image_size,
        augment=False,
        max_samples=config["training"]["val_samples"],
        manifest=config["data"].get("val_manifest"),
    )

    train_loader = SaliconDataset.make_loader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        seed=config["training"]["seed"],
    )
    val_loader = SaliconDataset.make_loader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        seed=config["training"]["seed"],
    )

    # Instantiate model and send to device
    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")
    model = get_model(model_name, image_size=image_size).to(device)

    # Setup optimizer and AMP scaler
    lr = config["training"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # We can also add a StepLR scheduler if desired, or let it run without one.
    # To keep code clean and portable, we instantiate a dummy StepLR scheduler that does nothing (e.g. step_size=1000).
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
    
    use_amp = config["training"]["amp"] and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_cc = -1.0 # pearson correlation coefficient ranges from -1 to 1, higher is better

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint_info = load_checkpoint(
            args.resume, model, device, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        start_epoch = checkpoint_info["epoch"] + 1
        best_cc = checkpoint_info.get("best_metric", -1.0)
        # Handle if the resumed config differs or needs to be loaded
        print(f"Resumed from epoch {checkpoint_info['epoch']} with previous best CC of {best_cc:.4f}")

    # Set up CSV logging
    log_path = run_dir / "training_log.csv"
    log_file_exists = log_path.exists()
    
    with open(log_path, "a" if log_file_exists and start_epoch > 1 else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not (log_file_exists and start_epoch > 1):
            writer.writerow(["epoch", "train_loss", "val_loss", "val_mse", "val_cc", "val_sim", "lr"])

    lambda_cc = config["training"]["lambda_cc"]
    epochs = config["training"]["epochs"]

    print(f"Starting training on {model_name} for {epochs - start_epoch + 1} epochs...")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss_total = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            images = batch["image"].to(device)
            targets = batch["saliency"].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(images)
                loss = combined_mse_cc_loss(preds, targets, lambda_cc=lambda_cc)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_total += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss = train_loss_total / len(train_dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_total = 0.0
        val_mse_total = 0.0
        val_cc_total = 0.0
        val_sim_total = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images = batch["image"].to(device)
                targets = batch["saliency"].to(device)
                
                preds = model(images)
                loss = combined_mse_cc_loss(preds, targets, lambda_cc=lambda_cc)
                
                val_loss_total += loss.item() * images.size(0)
                val_mse_total += compute_mse(preds, targets).item() * images.size(0)
                val_cc_total += compute_cc(preds, targets).item() * images.size(0)
                val_sim_total += compute_sim(preds, targets).item() * images.size(0)
                
        val_loss = val_loss_total / len(val_dataset)
        val_mse = val_mse_total / len(val_dataset)
        val_cc = val_cc_total / len(val_dataset)
        val_sim = val_sim_total / len(val_dataset)
        
        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log to file
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_mse:.6f}", f"{val_cc:.6f}", f"{val_sim:.6f}", f"{current_lr:.6e}"])

        # Plot curves
        plot_curves(log_path, run_dir)

        epoch_duration = time.time() - epoch_start
        print(f"Epoch {epoch}/{epochs} Finished in {epoch_duration:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val MSE: {val_mse:.4f} | Val CC: {val_cc:.4f} | Val SIM: {val_sim:.4f}")

        # Checkpoints saving
        state = make_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=max(best_cc, val_cc),
            model_name=model_name,
            config=config,
        )
        
        save_checkpoint(state, checkpoint_dir, "last.pth")
        
        if val_cc > best_cc:
            best_cc = val_cc
            save_checkpoint(state, checkpoint_dir, "best.pth")
            print(f" ==> Saved new best model checkpoint to best.pth (CC={best_cc:.4f})")
            
    print(f"Training completed successfully. Run folder is saved at: {run_dir}")


if __name__ == "__main__":
    train()
