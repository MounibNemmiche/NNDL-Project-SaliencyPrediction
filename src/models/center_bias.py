from __future__ import annotations

import torch
import torch.nn as nn


class CenterBiasBaseline(nn.Module):
    def __init__(self, image_size: int = 224, sigma_ratio: float = 0.25) -> None:
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if sigma_ratio <= 0:
            raise ValueError("sigma_ratio must be positive")
        self.image_size = image_size
        self.sigma_ratio = sigma_ratio
        self.register_buffer(
            "center_bias", self._make_gaussian(image_size, sigma_ratio), persistent=True
        )

    @staticmethod
    def _make_gaussian(size: int, sigma_ratio: float) -> torch.Tensor:
        coordinate = torch.arange(size, dtype=torch.float32)
        center = (size - 1) / 2.0
        sigma = sigma_ratio * size
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        gaussian = torch.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma**2))
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min()).clamp_min(1e-8)
        return gaussian.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {x.shape}")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Center bias was built for {self.image_size}x{self.image_size}, "
                f"got {tuple(x.shape[-2:])}"
            )
        return self.center_bias.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
