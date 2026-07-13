from __future__ import annotations
import torch
import torch.nn as nn


class CenterBiasBaseline(nn.Module):
    def __init__(self, image_size: int = 224, sigma: float = 0.25):
        super().__init__()
        cb = self._make_gaussian(image_size, sigma)
        self.register_buffer("center_bias", cb)

    @staticmethod
    def _make_gaussian(size: int, sigma: float) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, size)
        x = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        return g.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.center_bias.unsqueeze(0).expand(B, -1, -1, -1)


if __name__ == "__main__":
    m = CenterBiasBaseline(image_size=224)
    x = torch.zeros(2, 3, 224, 224)
    y = m(x)
    print(f"  output shape : {y.shape}")
    print(f"  values range : [{y.min():.4f}, {y.max():.4f}]")
    assert y.shape == (2, 1, 224, 224)
    assert 0.0 <= y.min() and y.max() <= 1.0
    print("center_bias.py OK")
