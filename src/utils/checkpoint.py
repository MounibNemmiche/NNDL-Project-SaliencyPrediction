from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: str | Path,
    filename: str,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    return state


def make_state(
    model: nn.Module,
    optimizer: Any,
    epoch: int,
    best_metric: float,
    model_name: str,
    config: dict,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
        "best_metric":          best_metric,
        "model_name":           model_name,
        "config":               config,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    return state


if __name__ == "__main__":
    import tempfile
    import torch.optim as optim
    from src.models.simple_cnn import SimpleCNN

    device = torch.device("cpu")
    model  = SimpleCNN().to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3)

    state = make_state(
        model=model, optimizer=opt, epoch=1,
        best_metric=0.42, model_name="simple",
        config={"lr": 1e-3},
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = save_checkpoint(state, tmp, "test.pth")
        assert path.exists()

        model2 = SimpleCNN().to(device)
        opt2   = optim.Adam(model2.parameters(), lr=1e-3)
        info   = load_checkpoint(path, model2, device, optimizer=opt2)

        assert info["epoch"] == 1
        assert abs(info["best_metric"] - 0.42) < 1e-6
        assert info["model_name"] == "simple"

    print("checkpoint.py OK")
