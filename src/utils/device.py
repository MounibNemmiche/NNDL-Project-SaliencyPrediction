import torch


def get_device(preference: str = "auto") -> torch.device:
    preference = preference.lower().strip()

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[device] WARNING: CUDA requested but not available — falling back to cpu.")
        return torch.device("cpu")

    if preference == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device] WARNING: MPS requested but not available — falling back to cpu.")
        return torch.device("cpu")

    if preference == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device preference '{preference}'. "
        "Choose from: auto, cuda, mps, cpu."
    )


if __name__ == "__main__":
    for pref in ("auto", "cuda", "mps", "cpu"):
        d = get_device(pref)
        print(f"  get_device({pref!r:6s}) -> {d}")
    print("device.py OK")
