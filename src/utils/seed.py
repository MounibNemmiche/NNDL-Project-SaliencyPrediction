import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    print(f"  Seed set to 42")
    print(f"  torch.rand(3) = {torch.rand(3).tolist()}")
    set_seed(42)
    print(f"  torch.rand(3) = {torch.rand(3).tolist()}  (must match above)")
    print("seed.py OK")
