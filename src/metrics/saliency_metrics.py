from __future__ import annotations

import torch

EPS = 1e-8


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def compute_cc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    p = pred.view(B, -1).float()
    g = target.view(B, -1).float()

    p_mu  = p.mean(dim=1, keepdim=True)
    g_mu  = g.mean(dim=1, keepdim=True)
    p_std = p.std(dim=1, keepdim=True).clamp(min=EPS)
    g_std = g.std(dim=1, keepdim=True).clamp(min=EPS)

    cov  = ((p - p_mu) * (g - g_mu)).mean(dim=1)
    corr = cov / (p_std.squeeze(1) * g_std.squeeze(1)).clamp(min=EPS)
    return corr.mean()


def compute_sim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    p = pred.view(B, -1).clamp(min=0).float()
    g = target.view(B, -1).clamp(min=0).float()

    p = p / p.sum(dim=1, keepdim=True).clamp(min=EPS)
    g = g / g.sum(dim=1, keepdim=True).clamp(min=EPS)

    similarity = torch.minimum(p, g).sum(dim=1)
    return similarity.mean()


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    B, H, W = 4, 32, 32

    pred   = torch.rand(B, 1, H, W)
    target = torch.rand(B, 1, H, W)

    print("=== Perfect prediction ===")
    print(f"  MSE : {compute_mse(pred, pred).item():.6f}  (expect 0)")
    print(f"  CC  : {compute_cc(pred, pred).item():.6f}   (expect 1)")
    print(f"  SIM : {compute_sim(pred, pred).item():.6f}  (expect 1)")

    print("\n=== Random prediction ===")
    print(f"  MSE : {compute_mse(pred, target).item():.6f}")
    print(f"  CC  : {compute_cc(pred, target).item():.6f}")
    print(f"  SIM : {compute_sim(pred, target).item():.6f}")

    print("saliency_metrics.py OK")
