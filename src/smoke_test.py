from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.visualization import save_prediction_grid, save_overlay, denormalize_image
from src.utils.checkpoint import make_state, save_checkpoint, load_checkpoint
from src.utils.paths import resolve_dirs, validate_dirs, print_dirs
from src.models.center_bias import CenterBiasBaseline
from src.models.simple_cnn import SimpleCNN
from src.models.multiscale_fusion_cnn import MultiScaleFusionCNN
from src.models import get_model
from src.losses.saliency_losses import mse_loss, cc_score, cc_loss, combined_mse_cc_loss
from src.metrics.saliency_metrics import compute_mse, compute_cc, compute_sim


def _check(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}", flush=True)
        sys.exit(1)


def run_smoke_test(device: torch.device) -> None:
    print(f"\n{'='*55}")
    print(f"  Smoke test  |  device={device}")
    print(f"{'='*55}")

    set_seed(42)

    B, C, H, W = 8, 3, 224, 224

    print("\n[1] Synthetic data ...")
    images  = torch.rand(B, C, H, W).to(device)
    targets = torch.rand(B, 1, H, W).to(device)
    _check(images.shape == (B, C, H, W), "image shape")
    _check(targets.shape == (B, 1, H, W), "target shape")
    print("    OK")

    print("\n[2] CenterBiasBaseline forward ...")
    cb = CenterBiasBaseline(image_size=H).to(device)
    cb.eval()
    with torch.no_grad():
        y_cb = cb(images)
    _check(y_cb.shape == (B, 1, H, W), f"center_bias output shape {y_cb.shape}")
    _check(y_cb.min() >= 0.0 and y_cb.max() <= 1.0, "center_bias values in [0,1]")
    print(f"    output={y_cb.shape}  range=[{y_cb.min():.3f}, {y_cb.max():.3f}]  OK")

    print("\n[3] SimpleCNN forward ...")
    net = SimpleCNN(image_size=H).to(device)
    net.eval()
    with torch.no_grad():
        y_net = net(images)
    _check(y_net.shape == (B, 1, H, W), f"simple_cnn output shape {y_net.shape}")
    _check(y_net.min() >= 0.0 and y_net.max() <= 1.0, "simple_cnn values in [0,1]")
    n_simple = sum(p.numel() for p in net.parameters())
    print(f"    output={y_net.shape}  params={n_simple:,}  OK")

    print("\n[4] MultiScaleFusionCNN forward + side outputs + backward ...")
    fusion = MultiScaleFusionCNN(image_size=H).to(device)
    fusion.eval()
    with torch.no_grad():
        y_f = fusion(images)
        d   = fusion(images, return_side_outputs=True)
    _check(y_f.shape == (B, 1, H, W), f"fusion output shape {y_f.shape}")
    _check(y_f.min() >= 0.0 and y_f.max() <= 1.0, "fusion values in [0,1]")
    _check(torch.isfinite(y_f).all(), "fusion output is finite")
    for k, v in d.items():
        _check(v.shape == (B, 1, H, W), f"side output '{k}' shape {v.shape}")
    n_fusion = sum(p.numel() for p in fusion.parameters())
    overhead = n_fusion - n_simple
    fusion.train()
    y_f_train = fusion(images[:2])
    y_f_train.sum().backward()
    _check(fusion.fusion.weight.grad is not None, "fusion conv receives gradients")
    _check(fusion.side1.weight.grad is not None, "side1 head receives gradients")
    print(f"    output={y_f.shape}  params={n_fusion:,}  overhead=+{overhead:,}  OK")

    print("\n[5] Model registry ...")
    _check(type(get_model("center")).__name__ == "CenterBiasBaseline", "registry center")
    _check(type(get_model("simple")).__name__ == "SimpleCNN",          "registry simple")
    _check(type(get_model("fusion")).__name__ == "MultiScaleFusionCNN", "registry fusion")
    print("    OK")

    print("\n[6] Loss functions ...")
    net.eval()
    with torch.no_grad():
        y_net = net(images)
    pred_req = y_net.detach().requires_grad_(True)
    l_mse  = mse_loss(pred_req, targets)
    l_cc   = cc_loss(pred_req, targets)
    l_comb = combined_mse_cc_loss(pred_req, targets, lambda_cc=0.1)
    cc_val = cc_score(pred_req, targets)
    _check(l_mse.shape == (), "mse_loss is scalar")
    _check(l_cc.shape == (),  "cc_loss is scalar")
    _check(-1.0 <= cc_val.item() <= 1.0, f"cc_score in [-1,1]: {cc_val.item():.4f}")
    l_comb.backward()
    _check(pred_req.grad is not None, "gradients flow through combined loss")
    print(f"    mse={l_mse.item():.4f}  cc_score={cc_val.item():.4f}  "
          f"combined={l_comb.item():.4f}  grad_norm={pred_req.grad.norm():.4f}  OK")

    print("\n[7] Evaluation metrics ...")
    with torch.no_grad():
        m_mse = compute_mse(y_net, targets)
        m_cc  = compute_cc(y_net, targets)
        m_sim = compute_sim(y_net, targets)
    _check(m_mse.shape == (), "compute_mse is scalar")
    _check(m_cc.shape  == (), "compute_cc is scalar")
    _check(m_sim.shape == (), "compute_sim is scalar")
    _check(0.0 <= m_sim.item() <= 1.0, f"SIM in [0,1]: {m_sim.item():.4f}")
    print(f"    MSE={m_mse.item():.4f}  CC={m_cc.item():.4f}  SIM={m_sim.item():.4f}  OK")

    print("\n[8] One training step on SimpleCNN ...")
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer.zero_grad()
    pred_train = net(images[:2])
    loss = combined_mse_cc_loss(pred_train, targets[:2], lambda_cc=0.1)
    loss.backward()
    optimizer.step()
    _check(loss.item() > 0, "training loss > 0")
    print(f"    loss={loss.item():.4f}  OK")

    print("\n[9] Checkpoint round-trip ...")
    with tempfile.TemporaryDirectory() as tmp:
        state = make_state(
            model=net, optimizer=optimizer, epoch=1,
            best_metric=0.5, model_name="simple",
            config={"lr": 1e-3},
        )
        ckpt_path = save_checkpoint(state, tmp, "test.pth")
        _check(ckpt_path.exists(), "checkpoint file saved")
        net2 = SimpleCNN(image_size=H).to(device)
        opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
        info = load_checkpoint(ckpt_path, net2, device, optimizer=opt2)
        _check(info["epoch"] == 1, "epoch restored")
        _check(info["model_name"] == "simple", "model_name restored")
    print("    OK")

    print("\n[10] Path utilities ...")
    from src.utils.paths import resolve_dirs, validate_dirs, print_dirs
    dirs = resolve_dirs(data_root="data/SALICON")
    _check("train_image_dir" in dirs, "resolve_dirs returns train_image_dir key")
    print("    OK")

    print("\n[11] Visualization utilities ...")
    with tempfile.TemporaryDirectory() as tmp:
        grid_path    = os.path.join(tmp, "grid.png")
        overlay_path = os.path.join(tmp, "overlay.png")
        save_prediction_grid(
            images=images[:2].cpu(), preds=y_net[:2].cpu(),
            targets=targets[:2].cpu(),
            filenames=["smoke_0.jpg", "smoke_1.jpg"],
            save_path=grid_path,
        )
        save_overlay(images[0].cpu(), y_net[0].cpu(), save_path=overlay_path)
        _check(os.path.exists(grid_path),    "grid PNG saved")
        _check(os.path.exists(overlay_path), "overlay PNG saved")
    arr = denormalize_image(images[0].cpu())
    _check(arr.shape == (H, W, 3), f"denormalize shape {arr.shape}")
    print("    OK")

    print("\n[12] Dataset class importable ...")
    from src.datasets.salicon_dataset import SaliconDataset
    _check(callable(SaliconDataset), "SaliconDataset is callable")
    print("    OK")

    print(f"\n{'='*55}")
    print("  Smoke test passed.")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full smoke test")
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    args   = parser.parse_args()
    device = get_device(args.device)
    run_smoke_test(device)


if __name__ == "__main__":
    main()
