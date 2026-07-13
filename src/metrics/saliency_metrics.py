from __future__ import annotations

import torch


EPS = 1e-8


def _validate_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {target.shape}")
    if pred.ndim < 2:
        raise ValueError("Prediction and target must include batch and feature dimensions")


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_shapes(pred, target)
    return ((pred.float() - target.float()) ** 2).mean()


def compute_cc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_shapes(pred, target)
    pred_flat = pred.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    covariance = (pred_centered * target_centered).mean(dim=1)
    pred_std = pred_flat.std(dim=1, correction=0)
    target_std = target_flat.std(dim=1, correction=0)
    correlation = covariance / (pred_std * target_std).clamp_min(EPS)
    return correlation.clamp(-1.0, 1.0).mean()


def compute_sim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_shapes(pred, target)
    pred_flat = pred.float().flatten(start_dim=1).clamp_min(0)
    target_flat = target.float().flatten(start_dim=1).clamp_min(0)
    pred_distribution = pred_flat / pred_flat.sum(dim=1, keepdim=True).clamp_min(EPS)
    target_distribution = target_flat / target_flat.sum(dim=1, keepdim=True).clamp_min(EPS)
    return torch.minimum(pred_distribution, target_distribution).sum(dim=1).mean()
