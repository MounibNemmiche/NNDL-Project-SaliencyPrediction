# Frozen Part A Facts

## Dataset and Preprocessing

- SALICON labeled training set: 10,000 matched JPEG/PNG pairs.
- SALICON labeled validation source: 5,000 matched JPEG/PNG pairs.
- Checkpoint-selection validation split: 500 filenames, seed 42, hash prefix `28634c48abb6`.
- Untouched final-test split: 4,500 filenames, seed 42, hash prefix `7ad21b6a799f`.
- Input: RGB resized to `224 x 224`, converted to float32, ImageNet mean/std normalization.
- Target: grayscale saliency map resized to `224 x 224`, float32 in `[0, 1]`.
- Training augmentation: synchronized horizontal image/map flip only.

## Baselines

- CenterBiasBaseline: zero trainable parameters and one `224 x 224` Gaussian buffer.
- Center-bias standard deviation: `0.25 * image_size` pixels.
- SimpleCNN: 979,713 total/trainable parameters with `base_ch=32`.
- SimpleCNN output: one sigmoid saliency map at the exact input spatial size.

## Objective and Metrics

```text
MSE = mean((prediction - target)^2)
CC loss = 1 - Pearson CC
Combined loss = MSE + lambda_cc * (1 - CC)
SIM = sum(min(normalized prediction, normalized target))
```

CC uses population standard deviation (`correction=0`) per image and averages over the batch. Constant maps use an epsilon denominator and remain finite.

## Verification Evidence

- Native environment tested: Python 3.12.10, PyTorch 2.11.0+cu128, CPU execution.
- Submission environment tested in Docker: Python 3.10, PyTorch 2.5.1+cpu.
- Native result: 15 tests passed, one fusion-dependent test skipped.
- Docker result: 15 tests passed, one fusion-dependent test skipped.
- Temporary-file CPU smoke test passed natively and in Docker.
- Real SALICON training and manifest-based validation reads passed.

CUDA is unavailable in Student A's current Python environment. MPS hardware is unavailable. Student B must run the final mandatory-fusion smoke test on the RTX 4090.
