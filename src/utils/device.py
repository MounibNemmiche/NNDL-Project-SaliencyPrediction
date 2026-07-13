from __future__ import annotations

import torch


DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def get_device(preference: str = "auto") -> torch.device:
    preference = preference.lower().strip()
    if preference not in DEVICE_CHOICES:
        choices = ", ".join(DEVICE_CHOICES)
        raise ValueError(f"Unknown device preference '{preference}'. Choose from: {choices}")

    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if preference == "auto":
        if cuda_available:
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested explicitly, but PyTorch cannot access CUDA")
    if preference == "mps" and not mps_available:
        raise RuntimeError("MPS was requested explicitly, but PyTorch cannot access MPS")
    return torch.device(preference)
