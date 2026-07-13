from __future__ import annotations

import torch.nn as nn
from src.models.center_bias import CenterBiasBaseline
from src.models.simple_cnn import SimpleCNN
from src.models.multiscale_fusion_cnn import MultiScaleFusionCNN

_REGISTRY: dict[str, type] = {
    "center": CenterBiasBaseline,
    "simple": SimpleCNN,
    "fusion": MultiScaleFusionCNN,
}


def get_model(name: str, **kwargs) -> nn.Module:
    name = name.lower().strip()
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


__all__ = ["get_model", "CenterBiasBaseline", "SimpleCNN", "MultiScaleFusionCNN"]
