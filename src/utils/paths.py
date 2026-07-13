from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_dirs(
    train_image_dir: Optional[str] = None,
    train_map_dir:   Optional[str] = None,
    val_image_dir:   Optional[str] = None,
    val_map_dir:     Optional[str] = None,
    test_image_dir:  Optional[str] = None,
    test_map_dir:    Optional[str] = None,
    data_root:       Optional[str] = None,
) -> dict[str, Optional[Path]]:
    root = Path(data_root) if data_root else None

    def _infer(explicit: Optional[str], *rel_parts: str) -> Optional[Path]:
        if explicit:
            return Path(explicit)
        if root:
            return root / Path(*rel_parts)
        return None

    dirs = {
        "train_image_dir": _infer(train_image_dir, "train_images"),
        "train_map_dir":   _infer(train_map_dir,   "train_maps"),
        "val_image_dir":   _infer(val_image_dir,   "val_images"),
        "val_map_dir":     _infer(val_map_dir,     "val_maps"),
        "test_image_dir":  _infer(test_image_dir,  "test_images"),
        "test_map_dir":    _infer(test_map_dir,    "test_maps"),
    }
    return dirs


def validate_dirs(dirs: dict[str, Optional[Path]], required: list[str]) -> None:
    for key in required:
        path = dirs.get(key)
        if path is None:
            raise ValueError(
                f"Required path '{key}' was not provided. "
                f"Pass --{key} explicitly or use --data_root."
            )
        if not path.exists():
            raise FileNotFoundError(
                f"Required directory does not exist: {path}  (key={key})"
            )


def print_dirs(dirs: dict[str, Optional[Path]]) -> None:
    print("Resolved data paths:")
    for key, path in dirs.items():
        status = str(path) if path else "not set"
        print(f"  {key:<20} {status}")


if __name__ == "__main__":
    print("path utilities import successfully")
