from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.salicon_dataset import SaliconDataset
from src.metrics.saliency_metrics import compute_cc, compute_mse, compute_sim
from src.models import get_model
from src.utils.checkpoint import load_checkpoint
from src.utils.device import DEVICE_CHOICES, get_device
from src.utils.paths import print_dirs, resolve_dirs, validate_dirs
from src.utils.visualization import save_prediction_grid


SUMMARY_FIELDS = [
    "model",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "manifest",
    "split_hash",
    "samples",
    "mse",
    "cc",
    "sim",
    "parameters",
    "trainable_parameters",
    "seconds",
    "inference_seconds",
    "milliseconds_per_image",
    "device",
]
PER_IMAGE_FIELDS = [
    "model",
    "checkpoint_sha256",
    "split_hash",
    "filename",
    "mse",
    "cc",
    "sim",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saliency prediction model.")
    parser.add_argument("--checkpoint", help="Checkpoint path; not required for center bias")
    parser.add_argument("--model", choices=["simple", "fusion", "center"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--image-dir")
    parser.add_argument("--map-dir")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--output", help="Optional legacy path for the summary JSON")
    parser.add_argument("--visualize")
    parser.add_argument("--test-samples", type=int)
    return parser.parse_args()


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    device = get_device(args.device)

    checkpoint_arg = args.checkpoint
    is_center = args.model == "center" or checkpoint_arg == "center"
    checkpoint_path: Path | None = None
    checkpoint_hash = "not_applicable"
    checkpoint_bytes = 0
    if is_center:
        model_name = "center"
        image_size = config["preprocessing"]["image_size"]
        model = get_model(model_name, image_size=image_size).to(device)
    else:
        if checkpoint_arg is None:
            raise ValueError("--checkpoint is required for simple and fusion models")
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_name = args.model or state.get("model_name")
        if model_name not in {"simple", "fusion"}:
            raise ValueError(f"Checkpoint has an invalid model name: {model_name!r}")
        checkpoint_config = state.get("config", {})
        image_size = checkpoint_config.get("preprocessing", {}).get(
            "image_size", config["preprocessing"]["image_size"]
        )
        model = get_model(model_name, image_size=image_size).to(device)
        load_checkpoint(checkpoint_path, model, device)
        checkpoint_hash = file_sha256(checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
    model.eval()

    manifest = Path(args.manifest or config["data"]["test_manifest"])
    image_dir = args.image_dir or config["data"].get("test_image_dir") or config["data"]["val_image_dir"]
    map_dir = args.map_dir or config["data"].get("test_map_dir") or config["data"]["val_map_dir"]
    dirs = resolve_dirs(val_image_dir=image_dir, val_map_dir=map_dir)
    eval_dirs = {"image_dir": dirs["val_image_dir"], "map_dir": dirs["val_map_dir"]}
    print_dirs(eval_dirs)
    validate_dirs(eval_dirs, required=["image_dir", "map_dir"])

    max_samples = args.test_samples
    if max_samples is None:
        max_samples = config["evaluation"].get("test_samples")
    dataset = SaliconDataset(
        image_dir=eval_dirs["image_dir"],
        map_dir=eval_dirs["map_dir"],
        manifest=manifest,
        image_size=image_size,
        augment=False,
        max_samples=max_samples,
        require_maps=True,
    )
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = config["training"]["num_workers"]
    loader = SaliconDataset.make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=config["training"]["seed"],
        pin_memory=device.type == "cuda",
    )

    totals = {"mse": 0.0, "cc": 0.0, "sim": 0.0}
    per_image_rows: list[dict] = []
    vis_images: list[torch.Tensor] = []
    vis_predictions: list[torch.Tensor] = []
    vis_targets: list[torch.Tensor] = []
    vis_filenames: list[str] = []
    non_blocking = device.type == "cuda"
    inference_seconds = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {model_name}"):
            images = batch["image"].to(device, non_blocking=non_blocking)
            targets = batch["saliency"].to(device, non_blocking=non_blocking)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            predictions = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_start
            batch_count = images.size(0)
            batch_metrics = {
                "mse": compute_mse(predictions, targets),
                "cc": compute_cc(predictions, targets),
                "sim": compute_sim(predictions, targets),
            }
            for name, value in batch_metrics.items():
                if not torch.isfinite(value).item():
                    raise FloatingPointError(f"Non-finite {name} during evaluation")
                totals[name] += value.item() * batch_count

            for index, filename in enumerate(batch["filename"]):
                sample_prediction = predictions[index : index + 1]
                sample_target = targets[index : index + 1]
                per_image_rows.append(
                    {
                        "model": model_name,
                        "checkpoint_sha256": checkpoint_hash,
                        "split_hash": dataset.manifest_hash,
                        "filename": filename,
                        "mse": compute_mse(sample_prediction, sample_target).item(),
                        "cc": compute_cc(sample_prediction, sample_target).item(),
                        "sim": compute_sim(sample_prediction, sample_target).item(),
                    }
                )

            if args.visualize and len(vis_filenames) < 8:
                count = min(8 - len(vis_filenames), batch_count)
                vis_images.append(images[:count].cpu())
                vis_predictions.append(predictions[:count].cpu())
                vis_targets.append(targets[:count].cpu())
                vis_filenames.extend(batch["filename"][:count])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    summary = {
        "model": model_name,
        "checkpoint": str(checkpoint_path) if checkpoint_path else "none",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint_bytes,
        "manifest": str(manifest),
        "split_hash": dataset.manifest_hash,
        "samples": len(dataset),
        "mse": totals["mse"] / len(dataset),
        "cc": totals["cc"] / len(dataset),
        "sim": totals["sim"] / len(dataset),
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "seconds": elapsed,
        "inference_seconds": inference_seconds,
        "milliseconds_per_image": inference_seconds * 1000.0 / len(dataset),
        "device": str(device),
        "image_dir": str(eval_dirs["image_dir"]),
        "map_dir": str(eval_dirs["map_dir"]),
        "image_size": image_size,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.output) if args.output else output_dir / f"{model_name}_evaluation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_csv(output_dir / "evaluation_results.csv", SUMMARY_FIELDS, [{key: summary[key] for key in SUMMARY_FIELDS}])
    append_csv(output_dir / "per_image_metrics.csv", PER_IMAGE_FIELDS, per_image_rows)
    if model_name == "fusion":
        weights = model.fusion.weight.detach().cpu().reshape(-1).tolist()
        append_csv(
            output_dir / "fusion_weights.csv",
            ["checkpoint_sha256", "side1", "side2", "side3", "main", "bias"],
            [
                {
                    "checkpoint_sha256": checkpoint_hash,
                    "side1": weights[0],
                    "side2": weights[1],
                    "side3": weights[2],
                    "main": weights[3],
                    "bias": model.fusion.bias.detach().cpu().item(),
                }
            ],
        )

    if args.visualize and vis_filenames:
        save_prediction_grid(
            images=torch.cat(vis_images),
            preds=torch.cat(vis_predictions),
            targets=torch.cat(vis_targets),
            filenames=vis_filenames,
            save_path=args.visualize,
            max_items=8,
        )

    print(
        f"{model_name}: samples={len(dataset)}, MSE={summary['mse']:.6f}, "
        f"CC={summary['cc']:.6f}, SIM={summary['sim']:.6f}, "
        f"ms/image={summary['milliseconds_per_image']:.3f}"
    )
    print(f"Evaluation evidence saved to: {output_dir}")


if __name__ == "__main__":
    main()
