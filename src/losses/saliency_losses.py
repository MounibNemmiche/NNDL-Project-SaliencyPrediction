from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-8


def _validate_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {target.shape}")
    if pred.ndim < 2:
        raise ValueError("Prediction and target must include batch and feature dimensions")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_shapes(pred, target)
    return F.mse_loss(pred, target)


def cc_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_shapes(pred, target)
    pred_flat = pred.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    covariance = (pred_centered * target_centered).mean(dim=1)
    pred_std = pred_flat.std(dim=1, correction=0)
    target_std = target_flat.std(dim=1, correction=0)
    denominator = (pred_std * target_std).clamp_min(EPS)
    correlation = (covariance / denominator).clamp(-1.0, 1.0)
    return correlation.mean()


def cc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - cc_score(pred, target)


def combined_mse_cc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_cc: float = 0.1,
) -> torch.Tensor:
    if lambda_cc < 0:
        raise ValueError("lambda_cc must be non-negative")
    return mse_loss(pred, target) + lambda_cc * cc_loss(pred, target)
