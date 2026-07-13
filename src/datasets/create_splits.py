from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from src.datasets.salicon_dataset import SaliconDataset


def _manifest_hash(names: list[str]) -> str:
    content = "\n".join(names) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def _write_manifest(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")


def create_splits(
    image_dir: str | Path,
    map_dir: str | Path,
    val_size: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    dataset = SaliconDataset(image_dir=image_dir, map_dir=map_dir, require_maps=True)
    names = dataset.filenames.copy()
    if val_size <= 0 or val_size >= len(names):
        raise ValueError(f"val_size must be between 1 and {len(names) - 1}")

    rng = random.Random(seed)
    rng.shuffle(names)
    val_names = sorted(names[:val_size])
    test_names = sorted(names[val_size:])

    if set(val_names) & set(test_names):
        raise RuntimeError("Generated validation and test manifests overlap")
    if len(val_names) + len(test_names) != len(dataset):
        raise RuntimeError("Generated manifests do not cover every valid pair")
    return val_names, test_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic, disjoint SALICON validation/test manifests"
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-output", default="data/splits/val_seed42.txt")
    parser.add_argument("--test-output", default="data/splits/test_seed42.txt")
    args = parser.parse_args()

    val_names, test_names = create_splits(
        image_dir=args.image_dir,
        map_dir=args.map_dir,
        val_size=args.val_size,
        seed=args.seed,
    )
    val_output = Path(args.val_output)
    test_output = Path(args.test_output)
    if val_output.resolve() == test_output.resolve():
        raise ValueError("Validation and test outputs must be different files")

    _write_manifest(val_output, val_names)
    _write_manifest(test_output, test_names)
    print(
        f"Validation: {len(val_names)} files, sha256={_manifest_hash(val_names)}, "
        f"path={val_output}"
    )
    print(
        f"Final test: {len(test_names)} files, sha256={_manifest_hash(test_names)}, "
        f"path={test_output}"
    )


if __name__ == "__main__":
    main()
