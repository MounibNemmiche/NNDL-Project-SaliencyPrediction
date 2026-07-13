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


class SimpleCNN(nn.Module):
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

        self.head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return self.head(d)


if __name__ == "__main__":
    m = SimpleCNN(image_size=224)
    x = torch.zeros(2, 3, 224, 224)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"  output shape  : {y.shape}")
    print(f"  values range  : [{y.min():.4f}, {y.max():.4f}]")
    print(f"  total params  : {n:,}")
    assert y.shape == (2, 1, 224, 224)
    assert 0.0 <= y.min() and y.max() <= 1.0
    print("simple_cnn.py OK")
