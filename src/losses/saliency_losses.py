from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def cc_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    p = pred.view(B, -1)
    g = target.view(B, -1)

    p_mu  = p.mean(dim=1, keepdim=True)
    g_mu  = g.mean(dim=1, keepdim=True)
    p_std = p.std(dim=1, keepdim=True).clamp(min=EPS)
    g_std = g.std(dim=1, keepdim=True).clamp(min=EPS)

    cov  = ((p - p_mu) * (g - g_mu)).mean(dim=1)
    corr = cov / (p_std.squeeze(1) * g_std.squeeze(1)).clamp(min=EPS)
    return corr.mean()


def cc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -cc_score(pred, target)


def combined_mse_cc_loss(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    lambda_cc: float = 0.1,
) -> torch.Tensor:
    return mse_loss(pred, target) + lambda_cc * cc_loss(pred, target)


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    B, H, W = 4, 32, 32

    pred   = torch.rand(B, 1, H, W, requires_grad=True)
    target = torch.rand(B, 1, H, W)

    l_mse  = mse_loss(pred, target)
    l_cc   = cc_loss(pred, target)
    l_comb = combined_mse_cc_loss(pred, target, lambda_cc=0.1)
    cc_val = cc_score(pred, target)

    print(f"  mse_loss              : {l_mse.item():.6f}")
    print(f"  cc_score              : {cc_val.item():.6f}")
    print(f"  cc_loss               : {l_cc.item():.6f}")
    print(f"  combined_mse_cc_loss  : {l_comb.item():.6f}")

    l_comb.backward()
    print(f"  grad norm on pred     : {pred.grad.norm():.6f}")
    print("saliency_losses.py OK")
