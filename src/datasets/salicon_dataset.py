from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class SaliconDataset(Dataset):
    def __init__(
        self,
        image_dir:   str,
        map_dir:     Optional[str] = None,
        image_size:  int = 224,
        augment:     bool = False,
        max_samples: Optional[int] = None,
    ):
        self.image_size = image_size
        self.augment    = augment

        img_dir = Path(image_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {img_dir}")

        img_paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if len(img_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        self.map_dir  = None
        self.has_maps = False

        if map_dir is not None:
            mp = Path(map_dir)
            if mp.exists():
                n_maps = sum(1 for f in mp.iterdir() if f.suffix.lower() == ".png")
                if n_maps > 0:
                    self.map_dir  = mp
                    self.has_maps = True
                    img_paths = [
                        p for p in img_paths
                        if (self.map_dir / (p.stem + ".png")).exists()
                    ]

        if max_samples is not None and max_samples < len(img_paths):
            img_paths = img_paths[:max_samples]

        self.img_paths  = img_paths
        self._normalize = transforms.Normalize(mean=_MEAN, std=_STD)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.img_paths[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        sal_pil = None
        if self.has_maps:
            map_path = self.map_dir / (img_path.stem + ".png")
            sal_pil  = Image.open(map_path).convert("L")
            sal_pil  = sal_pil.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.augment and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            if sal_pil is not None:
                sal_pil = transforms.functional.hflip(sal_pil)

        image_t = transforms.functional.to_tensor(image)
        image_t = self._normalize(image_t)

        sample = {
            "image":    image_t,
            "filename": img_path.name,
        }

        if sal_pil is not None:
            sal_arr = np.array(sal_pil, dtype=np.float32)
            if sal_arr.max() > 1.0:
                sal_arr /= 255.0
            sample["saliency"] = torch.from_numpy(sal_arr).unsqueeze(0)

        return sample

    @staticmethod
    def make_loader(
        dataset:     "SaliconDataset",
        batch_size:  int = 8,
        shuffle:     bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",   required=True)
    parser.add_argument("--map_dir",     default=None)
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=4)
    args = parser.parse_args()

    ds = SaliconDataset(
        image_dir=args.image_dir,
        map_dir=args.map_dir,
        image_size=args.image_size,
        augment=True,
        max_samples=args.max_samples,
    )
    print(f"Dataset: {len(ds)} samples  has_maps={ds.has_maps}")

    s = ds[0]
    print(f"  image    : {s['image'].shape}  dtype={s['image'].dtype}")
    if "saliency" in s:
        sal = s["saliency"]
        print(f"  saliency : {sal.shape}  min={sal.min():.4f}  max={sal.max():.4f}")
    print(f"  filename : {s['filename']}")
    print("salicon_dataset.py OK")
