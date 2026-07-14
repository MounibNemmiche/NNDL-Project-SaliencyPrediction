from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.train import build_loss


def _write_pair(image_path: Path, map_path: Path | None) -> None:
    rng = np.random.default_rng(sum(image_path.name.encode("utf-8")))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), mode="RGB").save(
        image_path
    )
    if map_path is not None:
        map_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8), mode="L").save(
            map_path
        )


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments], capture_output=True, text=True, check=True
    )


def test_loss_selection_is_not_ignored() -> None:
    prediction = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    mse = build_loss("mse", lambda_cc=0.1)(prediction, target)
    combined = build_loss("mse_cc", lambda_cc=0.1)(prediction, target)
    assert combined > mse


def test_pipeline_cli_integration(tmp_path: Path) -> None:
    train_images = tmp_path / "images" / "train"
    train_maps = tmp_path / "maps" / "train"
    val_images = tmp_path / "images" / "val"
    val_maps = tmp_path / "maps" / "val"
    predict_images = tmp_path / "images" / "predict"
    for index in range(4):
        _write_pair(train_images / f"train_{index}.jpg", train_maps / f"train_{index}.png")
    for index in range(2):
        _write_pair(val_images / f"val_{index}.jpg", val_maps / f"val_{index}.png")
        _write_pair(predict_images / f"predict_{index}.jpg", None)

    val_manifest = tmp_path / "val_manifest.txt"
    test_manifest = tmp_path / "test_manifest.txt"
    val_manifest.write_text("val_0.jpg\nval_1.jpg\n", encoding="utf-8")
    test_manifest.write_text("val_0.jpg\nval_1.jpg\n", encoding="utf-8")
    runs_dir = tmp_path / "outputs" / "runs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  train_image_dir: "{train_images.as_posix()}"
  train_map_dir: "{train_maps.as_posix()}"
  val_image_dir: "{val_images.as_posix()}"
  val_map_dir: "{val_maps.as_posix()}"
  test_image_dir: "{val_images.as_posix()}"
  test_map_dir: "{val_maps.as_posix()}"
  train_manifest: null
  val_manifest: "{val_manifest.as_posix()}"
  test_manifest: "{test_manifest.as_posix()}"
preprocessing:
  image_size: 32
  augmentation:
    horizontal_flip: true
model:
  name: simple
training:
  epochs: 2
  batch_size: 2
  learning_rate: 0.001
  seed: 42
  loss: mse
  lambda_cc: 0.1
  train_samples: null
  val_samples: null
  num_workers: 0
  device: cpu
  amp: false
evaluation:
  test_samples: null
  batch_size: 2
outputs:
  root_dir: "{runs_dir.as_posix()}"
  visualization_dir: "{(tmp_path / 'outputs' / 'visualizations').as_posix()}"
""",
        encoding="utf-8",
    )

    _run("src/train.py", "--config", str(config_path), "--run-name", "test_run")
    run_dir = runs_dir / "test_run"
    for relative in (
        "checkpoints/best.pth",
        "checkpoints/last.pth",
        "training_log.csv",
        "config.json",
        "loss_curve.png",
        "cc_curve.png",
    ):
        assert (run_dir / relative).is_file()

    duplicate = subprocess.run(
        [sys.executable, "src/train.py", "--config", str(config_path), "--run-name", "test_run"],
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode != 0
    assert "will not be overwritten" in duplicate.stderr

    _run(
        "src/train.py",
        "--config",
        str(config_path),
        "--resume",
        str(run_dir / "checkpoints" / "last.pth"),
        "--epochs",
        "3",
    )
    rows = list(csv.DictReader((run_dir / "training_log.csv").open(encoding="utf-8")))
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    assert all(float(row["epoch_seconds"]) > 0 for row in rows)
    assert all(float(row["peak_gpu_memory_mb"]) == 0 for row in rows)

    evaluation_dir = tmp_path / "outputs" / "evaluation"
    summary_json = evaluation_dir / "summary.json"
    grid = evaluation_dir / "grid.png"
    _run(
        "src/evaluate.py",
        "--checkpoint",
        str(run_dir / "checkpoints" / "best.pth"),
        "--config",
        str(config_path),
        "--output-dir",
        str(evaluation_dir),
        "--output",
        str(summary_json),
        "--visualize",
        str(grid),
        "--device",
        "cpu",
    )
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["samples"] == 2
    assert summary["image_dir"] == str(val_images)
    assert summary["checkpoint_sha256"] != "not_applicable"
    assert summary["checkpoint_bytes"] > 0
    assert 0 < summary["inference_seconds"] <= summary["seconds"]
    assert grid.is_file()
    assert len(list(csv.DictReader((evaluation_dir / "evaluation_results.csv").open()))) == 1
    assert len(list(csv.DictReader((evaluation_dir / "per_image_metrics.csv").open()))) == 2

    _run(
        "src/evaluate.py",
        "--model",
        "center",
        "--config",
        str(config_path),
        "--output-dir",
        str(evaluation_dir),
        "--device",
        "cpu",
    )
    assert len(list(csv.DictReader((evaluation_dir / "evaluation_results.csv").open()))) == 2

    prediction_dir = tmp_path / "outputs" / "predictions"
    result = _run(
        "src/predict.py",
        "--checkpoint",
        str(run_dir / "checkpoints" / "best.pth"),
        "--config",
        str(config_path),
        "--input",
        str(predict_images),
        "--output-dir",
        str(prediction_dir),
        "--save-raw",
        "--save-overlay",
        "--save-heatmap",
        "--device",
        "cpu",
    )
    assert "Generated predictions for 2 images" in result.stdout
    for index in range(2):
        for suffix in (".png", "_overlay.png", "_heatmap.png"):
            assert (prediction_dir / f"predict_{index}{suffix}").is_file()
