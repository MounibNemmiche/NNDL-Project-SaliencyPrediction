from __future__ import annotations

import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from src.models import get_model
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device, DEVICE_CHOICES
from src.utils.visualization import save_raw_saliency_map, save_overlay, make_heatmap


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict saliency maps for images using a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file, or 'center' for baseline.")
    parser.add_argument("--model", type=str, choices=["simple", "fusion", "center"], default=None, help="Force model type if loading fails.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to default config file.")
    parser.add_argument("--input", type=str, required=True, help="Path to an image file, a directory of images, or a manifest text file.")
    parser.add_argument("--image-dir", type=str, default=None, help="Folder containing images (only needed if input is a manifest file).")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Folder to save prediction results.")
    parser.add_argument("--device", type=str, choices=DEVICE_CHOICES, default="auto", help="Device preference.")
    parser.add_argument("--save-raw", action="store_true", help="Save raw 8-bit grayscale PNG saliency maps.")
    parser.add_argument("--save-overlay", action="store_true", help="Save overlays of saliency maps on original images.")
    parser.add_argument("--save-heatmap", action="store_true", help="Save colorized heatmaps.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Opacity of overlay (default: 0.5).")
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

    # Collect input files
    input_path = Path(args.input)
    image_paths: list[Path] = []
    
    if input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if not image_paths:
            raise RuntimeError(f"No supported images found in directory: {input_path}")
    elif input_path.is_file():
        if input_path.suffix.lower() in _IMG_EXTS:
            image_paths = [input_path]
        else:
            # Treat as manifest file
            print(f"Reading input manifest: {input_path}")
            with open(input_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.read().splitlines() if line.strip() and not line.lstrip().startswith("#")]
            
            base_dir = Path(args.image_dir) if args.image_dir else input_path.parent
            for name in lines:
                path = base_dir / name
                if path.is_file():
                    image_paths.append(path)
                else:
                    raise FileNotFoundError(f"Image from manifest not found: {path}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Found {len(image_paths)} images to process.")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup image preprocessing
    normalize = transforms.Normalize(mean=_MEAN, std=_STD)

    # Ensure at least one output option is active
    save_raw = args.save_raw
    save_overlay_opt = args.save_overlay
    save_heatmap = args.save_heatmap
    
    if not (save_raw or save_overlay_opt or save_heatmap):
        print("Warning: No output options (--save-raw, --save-overlay, --save-heatmap) were specified.")
        print("Defaulting to --save-raw.")
        save_raw = True

    # Process images
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Generating Saliency Predictions"):
            try:
                # Load image
                with Image.open(img_path) as source:
                    img_rgb = source.convert("RGB")
                    orig_w, orig_h = img_rgb.size
                    
                    # Resize to model input size
                    resized_img = img_rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
                    
                    # Normalise and convert to tensor
                    image_tensor = transforms.functional.to_tensor(resized_img)
                    normalized_tensor = normalize(image_tensor).to(device).unsqueeze(0)
                
                # Predict saliency map
                pred_tensor = model(normalized_tensor).squeeze(0)  # Shape: [1, image_size, image_size]
                
                # Setup output filenames
                stem = img_path.stem
                
                # 1. Save raw grayscale saliency map (interpolated back to original image size)
                if save_raw:
                    pred_resized = F.interpolate(
                        pred_tensor.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False
                    ).squeeze(0)
                    raw_out = output_dir / f"{stem}.png"
                    save_raw_saliency_map(pred_resized, raw_out)
                    
                # 2. Save overlay image (uses model resolution image and prediction)
                if save_overlay_opt:
                    overlay_out = output_dir / f"{stem}_overlay.png"
                    save_overlay(image_tensor, pred_tensor.cpu(), overlay_out, alpha=args.alpha)
                    
                # 3. Save color heatmap
                if save_heatmap:
                    pred_resized = F.interpolate(
                        pred_tensor.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False
                    ).squeeze(0)
                    heatmap_arr = make_heatmap(pred_resized)
                    heatmap_out = output_dir / f"{stem}_heatmap.png"
                    Image.fromarray(heatmap_arr).save(heatmap_out)
                    
            except Exception as e:
                print(f"Error processing image {img_path.name}: {e}")
                continue

    print(f"Predictions successfully generated and saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
