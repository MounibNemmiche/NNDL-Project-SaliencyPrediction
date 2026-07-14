from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import functional as transform_functional
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import get_model
from src.utils.checkpoint import load_checkpoint
from src.utils.device import DEVICE_CHOICES, get_device
from src.utils.visualization import make_heatmap, save_overlay, save_raw_saliency_map


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate saliency predictions.")
    parser.add_argument("--checkpoint", help="Checkpoint path; not required for center bias")
    parser.add_argument("--model", choices=["simple", "fusion", "center"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-dir", help="Base directory when --input is a manifest")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--save-heatmap", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def collect_images(input_path: Path, image_dir: str | None) -> list[Path]:
    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        images = [input_path]
    elif input_path.is_file():
        base_dir = Path(image_dir) if image_dir else input_path.parent
        names = [
            line.strip()
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        images = [base_dir / name for name in names]
        missing = [path for path in images if not path.is_file()]
        if missing:
            examples = ", ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(f"{len(missing)} manifest images were not found: {examples}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not images:
        raise RuntimeError(f"No supported images found for input: {input_path}")
    return images


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    config = load_config(args.config)
    device = get_device(args.device)

    is_center = args.model == "center" or args.checkpoint == "center"
    if is_center:
        model_name = "center"
        image_size = config["preprocessing"]["image_size"]
        model = get_model(model_name, image_size=image_size).to(device)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for simple and fusion models")
        checkpoint_path = Path(args.checkpoint)
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
    model.eval()

    image_paths = collect_images(Path(args.input), args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_raw = args.save_raw
    save_overlay_output = args.save_overlay
    save_heatmap = args.save_heatmap
    if not (save_raw or save_overlay_output or save_heatmap):
        save_raw = True

    failures: list[str] = []
    completed = 0
    with torch.inference_mode():
        for image_path in tqdm(image_paths, desc=f"Predicting with {model_name}"):
            try:
                with Image.open(image_path) as source:
                    rgb_image = source.convert("RGB")
                    original_tensor = transform_functional.to_tensor(rgb_image)
                    resized = rgb_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
                    model_tensor = transform_functional.to_tensor(resized)

                normalized_model = transform_functional.normalize(model_tensor, MEAN, STD)
                prediction = model(normalized_model.unsqueeze(0).to(device)).cpu()
                prediction_original = F.interpolate(
                    prediction,
                    size=original_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                normalized_original = transform_functional.normalize(original_tensor, MEAN, STD)
                stem = image_path.stem

                if save_raw:
                    save_raw_saliency_map(prediction_original, output_dir / f"{stem}.png")
                if save_overlay_output:
                    save_overlay(
                        normalized_original,
                        prediction_original,
                        output_dir / f"{stem}_overlay.png",
                        alpha=args.alpha,
                    )
                if save_heatmap:
                    Image.fromarray(make_heatmap(prediction_original)).save(
                        output_dir / f"{stem}_heatmap.png"
                    )
                completed += 1
            except Exception as error:
                failures.append(f"{image_path}: {error}")

    if failures:
        details = "\n".join(failures[:10])
        raise RuntimeError(
            f"Prediction failed for {len(failures)} of {len(image_paths)} images:\n{details}"
        )
    print(f"Generated predictions for {completed} images in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
