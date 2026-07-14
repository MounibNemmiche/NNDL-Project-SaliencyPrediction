from __future__ import annotations

import pytest
import torch

from src.models.center_bias import CenterBiasBaseline
from src.models.multiscale_fusion_cnn import MultiScaleFusionCNN
from src.models.simple_cnn import SimpleCNN


def test_center_bias_contract() -> None:
    model = CenterBiasBaseline(image_size=32, sigma_ratio=0.25)
    output = model(torch.zeros(3, 3, 32, 32))
    assert output.shape == (3, 1, 32, 32)
    assert output.min().item() == pytest.approx(0.0)
    assert output.max().item() == pytest.approx(1.0)
    assert torch.equal(output[0], output[2])
    assert "center_bias" in dict(model.named_buffers())


def test_simple_cnn_shape_range_and_gradients() -> None:
    model = SimpleCNN(image_size=32, base_ch=8)
    inputs = torch.rand(2, 3, 33, 41)
    output = model(inputs)
    assert output.shape == (2, 1, 33, 41)
    assert torch.isfinite(output).all()
    assert 0.0 <= output.min().item() <= output.max().item() <= 1.0
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_fusion_contract() -> None:
    model = MultiScaleFusionCNN(image_size=32, base_ch=8)
    inputs = torch.rand(2, 3, 32, 32)
    outputs = model(inputs, return_side_outputs=True)
    assert set(outputs) == {"final", "side1", "side2", "side3", "main"}
    assert all(output.shape == (2, 1, 32, 32) for output in outputs.values())
    outputs["final"].mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
