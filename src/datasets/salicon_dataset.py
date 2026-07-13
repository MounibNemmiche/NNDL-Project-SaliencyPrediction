from __future__ import annotations

import hashlib
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_RESAMPLING = getattr(Image, "Resampling", Image)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError(f"Manifest is empty: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest contains duplicate entries: {path}")
    if any(Path(name).name != name for name in names):
        raise ValueError("Manifest entries must be filenames, not paths")
    return names


class SaliconDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        map_dir: Optional[str | Path] = None,
        image_size: int = 224,
        augment: bool = False,
        max_samples: Optional[int] = None,
        manifest: Optional[str | Path] = None,
        require_maps: Optional[bool] = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")

        self.image_size = image_size
        self.augment = augment
        self.image_dir = Path(image_dir)
        self.map_dir = Path(map_dir) if map_dir is not None else None
        self.require_maps = map_dir is not None if require_maps is None else require_maps

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        all_images = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMG_EXTS
        )
        if not all_images:
            raise RuntimeError(f"No supported images found in {self.image_dir}")

        if manifest is not None:
            requested = _read_manifest(Path(manifest))
            by_name = {path.name: path for path in all_images}
            by_stem = {path.stem: path for path in all_images}
            missing_images: list[str] = []
            image_paths: list[Path] = []
            for name in requested:
                path = by_name.get(name)
                if path is None and not Path(name).suffix:
                    path = by_stem.get(name)
                if path is None:
                    missing_images.append(name)
                else:
                    image_paths.append(path)
            if missing_images:
                preview = ", ".join(missing_images[:5])
                raise FileNotFoundError(
                    f"{len(missing_images)} manifest images were not found in "
                    f"{self.image_dir}: {preview}"
                )
        else:
            image_paths = all_images

        self.has_maps = False
        if self.map_dir is None:
            if self.require_maps:
                raise ValueError("map_dir is required when require_maps=True")
        else:
            if not self.map_dir.is_dir():
                if self.require_maps:
                    raise FileNotFoundError(f"Map directory not found: {self.map_dir}")
            else:
                map_paths = sorted(
                    path
                    for path in self.map_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".png"
                )
                if not map_paths and self.require_maps:
                    raise RuntimeError(f"No PNG saliency maps found in {self.map_dir}")

                map_stems = {path.stem for path in map_paths}
                missing_maps = [path.name for path in image_paths if path.stem not in map_stems]
                if missing_maps:
                    warnings.warn(
                        f"Ignoring {len(missing_maps)} images without matching PNG maps "
                        f"in {self.map_dir}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                image_paths = [path for path in image_paths if path.stem in map_stems]
                self.has_maps = bool(image_paths)

        if self.require_maps and not image_paths:
            raise RuntimeError("No valid image/saliency-map pairs were found")
        if not image_paths:
            raise RuntimeError("No usable images were found")

        if max_samples is not None:
            image_paths = image_paths[:max_samples]

        self.img_paths = image_paths
        self.filenames = [path.name for path in image_paths]
        manifest_text = "\n".join(self.filenames) + "\n"
        self.manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()[:12]
        self._normalize = transforms.Normalize(mean=_MEAN, std=_STD)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path = self.img_paths[idx]
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size), _RESAMPLING.BILINEAR
            )

        saliency = None
        if self.has_maps and self.map_dir is not None:
            map_path = self.map_dir / f"{image_path.stem}.png"
            with Image.open(map_path) as source:
                saliency = source.convert("L").resize(
                    (self.image_size, self.image_size), _RESAMPLING.BILINEAR
                )

        if self.augment and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            if saliency is not None:
                saliency = transforms.functional.hflip(saliency)

        image_tensor = transforms.functional.to_tensor(image)
        sample: dict[str, torch.Tensor | str] = {
            "image": self._normalize(image_tensor),
            "filename": image_path.name,
        }

        if saliency is not None:
            saliency_array = np.asarray(saliency, dtype=np.float32) / 255.0
            sample["saliency"] = torch.from_numpy(saliency_array.copy()).unsqueeze(0)
        return sample

    @staticmethod
    def make_loader(
        dataset: "SaliconDataset",
        batch_size: int = 8,
        shuffle: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        pin_memory: Optional[bool] = None,
    ) -> DataLoader:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Read and validate a SALICON subset")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--map_dir")
    parser.add_argument("--manifest")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=4)
    args = parser.parse_args()

    dataset = SaliconDataset(
        image_dir=args.image_dir,
        map_dir=args.map_dir,
        manifest=args.manifest,
        image_size=args.image_size,
        augment=False,
        max_samples=args.max_samples,
    )
    sample = dataset[0]
    print(
        f"Dataset: {len(dataset)} samples, has_maps={dataset.has_maps}, "
        f"manifest_hash={dataset.manifest_hash}"
    )
    print(f"image: {sample['image'].shape}, dtype={sample['image'].dtype}")
    if "saliency" in sample:
        saliency = sample["saliency"]
        print(
            f"saliency: {saliency.shape}, dtype={saliency.dtype}, "
            f"range=[{saliency.min():.4f}, {saliency.max():.4f}]"
        )
    print(f"filename: {sample['filename']}")


if __name__ == "__main__":
    main()
