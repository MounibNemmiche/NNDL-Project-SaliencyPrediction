from __future__ import annotations

import pytest
import torch

from src.utils.device import get_device


def test_cpu_and_invalid_device() -> None:
    assert get_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="Unknown device"):
        get_device("tpu")


def test_explicit_unavailable_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested explicitly"):
        get_device("cuda")


def test_explicit_unavailable_mps_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(torch.backends, "mps"):
        pytest.skip("This PyTorch build has no MPS backend object")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested explicitly"):
        get_device("mps")
