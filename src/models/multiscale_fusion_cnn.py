from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiScaleFusionCNN(nn.Module):
    def __init__(self, image_size: int = 224, base_ch: int = 32):
        super().__init__()
        c = base_ch

        self.enc1 = nn.Sequential(
            _conv_bn_relu(3,     c),
            _conv_bn_relu(c,     c),
            nn.MaxPool2d(2, 2),
        )
        self.enc2 = nn.Sequential(
            _conv_bn_relu(c,     c * 2),
            _conv_bn_relu(c * 2, c * 2),
            nn.MaxPool2d(2, 2),
        )
        self.enc3 = nn.Sequential(
            _conv_bn_relu(c * 2, c * 4),
            _conv_bn_relu(c * 4, c * 4),
            nn.MaxPool2d(2, 2),
        )

        self.bottleneck = nn.Sequential(
            _conv_bn_relu(c * 4, c * 8),
            _conv_bn_relu(c * 8, c * 4),
        )

        self.dec3 = _conv_bn_relu(c * 4, c * 2)
        self.dec2 = _conv_bn_relu(c * 2, c)
        self.dec1 = _conv_bn_relu(c,     c)

        self.main_head = nn.Conv2d(c, 1, kernel_size=1)

        self.side1 = nn.Conv2d(c,     1, kernel_size=1)
        self.side2 = nn.Conv2d(c * 2, 1, kernel_size=1)
        self.side3 = nn.Conv2d(c * 4, 1, kernel_size=1)

        self.fusion = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_side_outputs: bool = False):
        H, W = x.shape[2], x.shape[3]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d = F.interpolate(b,  scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec3(d)

        d = F.interpolate(d,  scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec2(d)

        d = F.interpolate(d,  scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec1(d)

        logit_main  = self.main_head(d)
        logit_side1 = self.side1(e1)
        logit_side2 = self.side2(e2)
        logit_side3 = self.side3(e3)

        logit_main  = F.interpolate(logit_main,  size=(H, W), mode="bilinear", align_corners=False)
        logit_side1 = F.interpolate(logit_side1, size=(H, W), mode="bilinear", align_corners=False)
        logit_side2 = F.interpolate(logit_side2, size=(H, W), mode="bilinear", align_corners=False)
        logit_side3 = F.interpolate(logit_side3, size=(H, W), mode="bilinear", align_corners=False)

        fused = self.fusion(torch.cat([logit_side1, logit_side2, logit_side3, logit_main], dim=1))
        final = torch.sigmoid(fused)

        if return_side_outputs:
            return {
                "final": final,
                "side1": torch.sigmoid(logit_side1),
                "side2": torch.sigmoid(logit_side2),
                "side3": torch.sigmoid(logit_side3),
                "main":  torch.sigmoid(logit_main),
            }

        return final


if __name__ == "__main__":
    m = MultiScaleFusionCNN(image_size=224)
    x = torch.rand(2, 3, 224, 224)
    y = m(x)
    d = m(x, return_side_outputs=True)
    n = sum(p.numel() for p in m.parameters())
    print(f"output shape     : {y.shape}")
    print(f"side output keys : {sorted(d.keys())}")
    print(f"total params     : {n:,}")
    assert y.shape == (2, 1, 224, 224)
    assert 0.0 <= y.min() and y.max() <= 1.0
    for k, v in d.items():
        assert v.shape == (2, 1, 224, 224), f"{k} shape mismatch: {v.shape}"
        assert torch.isfinite(v).all(), f"{k} has non-finite values"
    y.sum().backward()
    assert m.fusion.weight.grad is not None
    assert m.side1.weight.grad is not None
    print("multiscale_fusion_cnn.py OK")
