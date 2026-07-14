from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.salicon_dataset import SaliconDataset
from src.metrics.saliency_metrics import compute_mse, compute_cc, compute_sim
from src.models import get_model
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device, DEVICE_CHOICES
from src.utils.paths import resolve_dirs, validate_dirs, print_dirs
from src.utils.visualization import save_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saliency prediction model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file, or 'center' for the baseline.")
    parser.add_argument("--model", type=str, choices=["simple", "fusion", "center"], default=None, help="Force model type if loading fails.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to default config file.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to splits manifest file (e.g. data/splits/test_seed42.txt).")
    parser.add_argument("--image-dir", type=str, default=None, help="Override image directory.")
    parser.add_argument("--map-dir", type=str, default=None, help="Override ground truth map directory.")
    parser.add_argument("--device", type=str, choices=DEVICE_CHOICES, default="auto", help="Device preference.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation metrics as JSON.")
    parser.add_argument("--visualize", type=str, default=None, help="Path to save prediction grid visualization.")
    parser.add_argument("--test-samples", type=int, default=None, help="Limit number of samples evaluated.")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    # Determine batch size
    batch_size = args.batch_size if args.batch_size is not None else config["evaluation"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    seed = config["training"]["seed"]

    # Select device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load model and configurations
    checkpoint_is_center = args.checkpoint.lower().strip() == "center" or args.model == "center"
    
    if checkpoint_is_center:
        model_name = "center"
        image_size = config["preprocessing"]["image_size"]
        model = get_model(model_name, image_size=image_size).to(device)
        print("Loaded baseline CenterBiasBaseline model.")
    else:
        # Load checkpoint metadata
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        print(f"Loading checkpoint metadata from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_name = args.model if args.model is not None else state.get("model_name", "simple")
        
        ckpt_config = state.get("config", {})
        image_size = ckpt_config.get("preprocessing", {}).get("image_size", config["preprocessing"]["image_size"])
        
        model = get_model(model_name, image_size=image_size).to(device)
        load_checkpoint(checkpoint_path, model, device)
        print(f"Loaded '{model_name}' model from checkpoint epoch {state.get('epoch', 'N/A')}.")

    # Set model to evaluation mode
    model.eval()

    # Resolve directories
    # We choose default dirs depending on if we are running validation or test manifest.
    # We check the manifest filename to guess. Or we default to validation if not specified.
    manifest_path_str = args.manifest
    if manifest_path_str is None:
        manifest_path_str = config["data"].get("val_manifest")
        print(f"No manifest specified. Defaulting to: {manifest_path_str}")

    manifest_path = Path(manifest_path_str)
    
    # Check if this is the test split or validation split to infer defaults
    is_test_split = "test" in manifest_path.name.lower()
    
    default_img_dir = config["data"]["test_image_dir"] if is_test_split else config["data"]["val_image_dir"]
    default_map_dir = config["data"].get("test_map_dir") or config["data"]["val_map_dir"] # default maps
    if is_test_split and config["data"].get("test_map_dir"):
        default_map_dir = config["data"]["test_map_dir"]
    
    img_dir = args.image_dir if args.image_dir is not None else default_img_dir
    map_dir = args.map_dir if args.map_dir is not None else default_map_dir

    dirs = resolve_dirs(
        val_image_dir=img_dir,
        val_map_dir=map_dir,
    )
    # Map resolved val dirs to required keys
    dirs_eval = {
        "image_dir": dirs["val_image_dir"],
        "map_dir": dirs["val_map_dir"]
    }
    
    print_dirs(dirs_eval)
    validate_dirs(dirs_eval, required=["image_dir", "map_dir"])

    # Load dataset
    max_samples = args.test_samples if args.test_samples is not None else config["evaluation"].get("test_samples")
    dataset = SaliconDataset(
        image_dir=dirs_eval["image_dir"],
        map_dir=dirs_eval["map_dir"],
        image_size=image_size,
        augment=False,
        max_samples=max_samples,
        manifest=manifest_path,
    )
    
    loader = SaliconDataset.make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    print(f"Evaluating {len(dataset)} samples...")
    total_mse = 0.0
    total_cc = 0.0
    total_sim = 0.0

    # Collect some samples for visualization if requested
    vis_images = []
    vis_preds = []
    vis_targets = []
    vis_filenames = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            images = batch["image"].to(device)
            targets = batch["saliency"].to(device)
            filenames = batch["filename"]
            
            preds = model(images)
            
            # Metrics
            mse_val = compute_mse(preds, targets).item()
            cc_val = compute_cc(preds, targets).item()
            sim_val = compute_sim(preds, targets).item()
            
            total_mse += mse_val * images.size(0)
            total_cc += cc_val * images.size(0)
            total_sim += sim_val * images.size(0)

            # Store samples for visualization
            if args.visualize and len(vis_images) < 8:
                needed = 8 - len(vis_images)
                count = min(needed, images.size(0))
                vis_images.append(images[:count].cpu())
                vis_preds.append(preds[:count].cpu())
                vis_targets.append(targets[:count].cpu())
                vis_filenames.extend(filenames[:count])

    avg_mse = total_mse / len(dataset)
    avg_cc = total_cc / len(dataset)
    avg_sim = total_sim / len(dataset)

    results = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "manifest": str(manifest_path),
        "split_hash": dataset.manifest_hash,
        "samples_evaluated": len(dataset),
        "MSE": avg_mse,
        "CC": avg_cc,
        "SIM": avg_sim,
    }

    print("\n" + "="*40)
    print(f"  Evaluation Results: {model_name}")
    print(f"  Split manifest:     {manifest_path.name}")
    print("="*40)
    print(f"  MSE: {avg_mse:.6f} (lower is better)")
    print(f"  CC:  {avg_cc:.6f} (higher is better)")
    print(f"  SIM: {avg_sim:.6f} (higher is better)")
    print("="*40 + "\n")

    # Save to JSON output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to: {out_path}")

    # Generate visualization grid
    if args.visualize and vis_images:
        grid_img = torch.cat(vis_images, dim=0)
        grid_pred = torch.cat(vis_preds, dim=0)
        grid_tgt = torch.cat(vis_targets, dim=0)
        
        vis_path = Path(args.visualize)
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_prediction_grid(
            images=grid_img,
            preds=grid_pred,
            targets=grid_tgt,
            filenames=vis_filenames,
            save_path=vis_path,
            max_items=8,
        )
        print(f"Qualitative prediction grid saved to: {vis_path}")


if __name__ == "__main__":
    main()
