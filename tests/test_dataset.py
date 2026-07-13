from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.create_splits import create_splits
from src.datasets.salicon_dataset import SaliconDataset


def _write_pair(image_dir: Path, map_dir: Path, stem: str, size: int = 12) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    base = np.arange(size * size, dtype=np.uint8).reshape(size, size)
    image = np.stack((base, np.flipud(base), np.fliplr(base)), axis=-1)
    Image.fromarray(image, mode="RGB").save(image_dir / f"{stem}.png")
    Image.fromarray(base, mode="L").save(map_dir / f"{stem}.png")


def test_manifest_order_shapes_and_ranges(tmp_path: Path) -> None:
    image_dir, map_dir = tmp_path / "images", tmp_path / "maps"
    for stem in ("one", "two", "three"):
        _write_pair(image_dir, map_dir, stem)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("three.png\none.png\n", encoding="utf-8")

    dataset = SaliconDataset(image_dir, map_dir, image_size=16, manifest=manifest)
    assert dataset.filenames == ["three.png", "one.png"]
    assert len(dataset.manifest_hash) == 12
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["image"].dtype == torch.float32
    assert sample["saliency"].shape == (1, 16, 16)
    assert sample["saliency"].dtype == torch.float32
    assert 0.0 <= sample["saliency"].min() <= sample["saliency"].max() <= 1.0


def test_missing_maps_are_reported_and_zero_pairs_fail(tmp_path: Path) -> None:
    image_dir, map_dir = tmp_path / "images", tmp_path / "maps"
    _write_pair(image_dir, map_dir, "matched")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "missing.png")
    with pytest.warns(RuntimeWarning, match="1 images"):
        dataset = SaliconDataset(image_dir, map_dir, require_maps=True)
    assert dataset.filenames == ["matched.png"]

    empty_maps = tmp_path / "empty_maps"
    empty_maps.mkdir()
    with pytest.raises(RuntimeError, match="No PNG"):
        SaliconDataset(image_dir, empty_maps, require_maps=True)
    with pytest.raises(FileNotFoundError, match="Map directory"):
        SaliconDataset(image_dir, tmp_path / "absent", require_maps=True)


def test_horizontal_flip_is_synchronized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir, map_dir = tmp_path / "images", tmp_path / "maps"
    _write_pair(image_dir, map_dir, "sample", size=10)
    plain = SaliconDataset(image_dir, map_dir, image_size=10, augment=False)[0]
    monkeypatch.setattr(random, "random", lambda: 0.0)
    flipped = SaliconDataset(image_dir, map_dir, image_size=10, augment=True)[0]
    assert torch.allclose(flipped["image"], torch.flip(plain["image"], dims=(-1,)))
    assert torch.allclose(flipped["saliency"], torch.flip(plain["saliency"], dims=(-1,)))


def test_deterministic_splits_are_complete_and_disjoint(tmp_path: Path) -> None:
    image_dir, map_dir = tmp_path / "images", tmp_path / "maps"
    for index in range(10):
        _write_pair(image_dir, map_dir, f"sample_{index:02d}")
    first = create_splits(image_dir, map_dir, val_size=3, seed=42)
    second = create_splits(image_dir, map_dir, val_size=3, seed=42)
    assert first == second
    validation, test = first
    assert len(validation) == 3
    assert len(test) == 7
    assert set(validation).isdisjoint(test)
    assert len(set(validation + test)) == 10


def test_seeded_loader_shuffle_is_reproducible(tmp_path: Path) -> None:
    image_dir, map_dir = tmp_path / "images", tmp_path / "maps"
    for index in range(6):
        _write_pair(image_dir, map_dir, f"sample_{index:02d}")
    dataset = SaliconDataset(image_dir, map_dir)
    first = [name for batch in SaliconDataset.make_loader(dataset, 2, True, seed=7) for name in batch["filename"]]
    second = [name for batch in SaliconDataset.make_loader(dataset, 2, True, seed=7) for name in batch["filename"]]
    assert first == second
