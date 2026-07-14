from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.simple_cnn import SimpleCNN
from src.utils.checkpoint import load_checkpoint, make_state, save_checkpoint


def test_checkpoint_round_trip_on_cpu(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = SimpleCNN(image_size=32, base_ch=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    output = model(torch.rand(2, 3, 32, 32))
    output.mean().backward()
    optimizer.step()
    scheduler.step()
    state = make_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        best_metric=0.42,
        model_name="simple",
        config={"seed": 42, "split_hash": "abc123"},
    )
    checkpoint = save_checkpoint(state, tmp_path, "portable.pth")

    restored = SimpleCNN(image_size=32, base_ch=8)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    metadata = load_checkpoint(
        checkpoint,
        restored,
        torch.device("cpu"),
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert metadata["epoch"] == 3
    assert metadata["best_metric"] == pytest.approx(0.42)
    assert metadata["config"]["split_hash"] == "abc123"
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_missing_checkpoint_has_clear_error(tmp_path: Path) -> None:
    model = SimpleCNN(image_size=32, base_ch=8)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "missing.pth", model, torch.device("cpu"))
