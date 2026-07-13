from __future__ import annotations

import pytest
import torch

from src.losses.saliency_losses import cc_loss, cc_score, combined_mse_cc_loss, mse_loss
from src.metrics.saliency_metrics import compute_cc, compute_mse, compute_sim


def test_identical_nonconstant_maps_are_perfect() -> None:
    target = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    assert mse_loss(target, target).item() == pytest.approx(0.0)
    assert cc_score(target, target).item() == pytest.approx(1.0, abs=1e-6)
    assert cc_loss(target, target).item() == pytest.approx(0.0, abs=1e-6)
    assert compute_mse(target, target).item() == pytest.approx(0.0)
    assert compute_cc(target, target).item() == pytest.approx(1.0, abs=1e-6)
    assert compute_sim(target, target).item() == pytest.approx(1.0, abs=1e-6)


def test_inverse_maps_have_negative_cc() -> None:
    target = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
    inverse = 1.0 - target
    assert cc_score(inverse, target).item() == pytest.approx(-1.0, abs=1e-6)
    assert compute_cc(inverse, target).item() == pytest.approx(-1.0, abs=1e-6)


def test_constant_maps_remain_finite() -> None:
    zeros = torch.zeros(2, 1, 8, 8)
    ones = torch.ones_like(zeros)
    values = [
        cc_score(zeros, zeros),
        cc_loss(zeros, ones),
        compute_cc(zeros, ones),
        compute_sim(zeros, ones),
    ]
    assert all(torch.isfinite(value) for value in values)


def test_combined_loss_formula_and_backward() -> None:
    torch.manual_seed(42)
    prediction = torch.rand(3, 1, 8, 8, requires_grad=True)
    target = torch.rand_like(prediction)
    expected = mse_loss(prediction, target) + 0.2 * (1.0 - cc_score(prediction, target))
    actual = combined_mse_cc_loss(prediction, target, lambda_cc=0.2)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert 0.0 <= compute_sim(prediction.detach(), target).item() <= 1.0


def test_mismatched_shapes_fail() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        compute_mse(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 5, 5))
