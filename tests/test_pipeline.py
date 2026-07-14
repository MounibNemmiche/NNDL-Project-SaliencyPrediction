from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest


def _write_dummy_pair(image_path: Path, map_path: Path | None) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(image_path)
    
    if map_path is not None:
        map_path.parent.mkdir(parents=True, exist_ok=True)
        saliency = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        Image.fromarray(saliency, mode="L").save(map_path)


def test_pipeline_cli_integration(tmp_path: Path) -> None:
    # Setup directories
    train_img_dir = tmp_path / "images" / "train"
    train_map_dir = tmp_path / "maps" / "train"
    val_img_dir = tmp_path / "images" / "val"
    val_map_dir = tmp_path / "maps" / "val"
    test_img_dir = tmp_path / "images" / "test"
    
    # Generate 4 training pairs and 2 validation pairs
    for i in range(4):
        _write_dummy_pair(train_img_dir / f"train_{i}.jpg", train_map_dir / f"train_{i}.png")
    for i in range(2):
        _write_dummy_pair(val_img_dir / f"val_{i}.jpg", val_map_dir / f"val_{i}.png")
    # Generate 2 test images
    for i in range(2):
        _write_dummy_pair(test_img_dir / f"test_{i}.jpg", None)
        
    # Write custom config file
    config_path = tmp_path / "custom_config.yaml"
    config_content = f"""
data:
  train_image_dir: "{train_img_dir.as_posix()}"
  train_map_dir: "{train_map_dir.as_posix()}"
  val_image_dir: "{val_img_dir.as_posix()}"
  val_map_dir: "{val_map_dir.as_posix()}"
  test_image_dir: "{test_img_dir.as_posix()}"
  val_manifest: "{tmp_path.as_posix()}/val_manifest.txt"
  test_manifest: "{tmp_path.as_posix()}/test_manifest.txt"

preprocessing:
  image_size: 32
  augmentation:
    horizontal_flip: true

model:
  name: "simple"

training:
  epochs: 2
  batch_size: 2
  learning_rate: 1.0e-3
  seed: 42
  loss: "mse"
  lambda_cc: 0.1
  train_samples: null
  val_samples: null
  num_workers: 0
  device: "cpu"
  amp: false

evaluation:
  test_samples: null
  batch_size: 2

outputs:
  root_dir: "{(tmp_path / 'outputs' / 'runs').as_posix()}"
  visualization_dir: "{(tmp_path / 'outputs' / 'visualizations').as_posix()}"
"""
    config_path.write_text(config_content, encoding="utf-8")
    
    # Generate validation/test manifests
    val_manifest = tmp_path / "val_manifest.txt"
    val_manifest.write_text("val_0.jpg\nval_1.jpg\n", encoding="utf-8")
    
    # --- 1. Test train.py ---
    cmd_train = [
        sys.executable,
        "train.py",
        "--config", str(config_path),
        "--run-name", "test_run",
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True, check=True)
    assert "Training completed successfully" in res_train.stdout
    
    run_dir = tmp_path / "outputs" / "runs" / "test_run"
    assert run_dir.exists()
    assert (run_dir / "checkpoints" / "best.pth").is_file()
    assert (run_dir / "checkpoints" / "last.pth").is_file()
    assert (run_dir / "training_log.csv").is_file()
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "loss_curve.png").is_file()
    assert (run_dir / "cc_curve.png").is_file()

    # --- 2. Test train.py resume ---
    cmd_resume = [
        sys.executable,
        "train.py",
        "--config", str(config_path),
        "--resume", str(run_dir / "checkpoints" / "last.pth"),
        "--epochs", "3",
        "--run-name", "test_run",
    ]
    res_resume = subprocess.run(cmd_resume, capture_output=True, text=True, check=True)
    assert "Resuming training from checkpoint" in res_resume.stdout
    assert "Epoch 3/3" in res_resume.stdout

    # --- 3. Test evaluate.py ---
    eval_json = tmp_path / "outputs" / "evaluation.json"
    eval_vis = tmp_path / "outputs" / "prediction_grid.png"
    cmd_eval = [
        sys.executable,
        "evaluate.py",
        "--checkpoint", str(run_dir / "checkpoints" / "best.pth"),
        "--config", str(config_path),
        "--manifest", str(val_manifest),
        "--image-dir", str(val_img_dir),
        "--map-dir", str(val_map_dir),
        "--output", str(eval_json),
        "--visualize", str(eval_vis),
        "--device", "cpu",
    ]
    res_eval = subprocess.run(cmd_eval, capture_output=True, text=True, check=True)
    assert "MSE:" in res_eval.stdout
    assert eval_json.is_file()
    assert eval_vis.is_file()
    
    with open(eval_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        assert "MSE" in metrics
        assert "CC" in metrics
        assert "SIM" in metrics
        assert metrics["samples_evaluated"] == 2

    # --- 4. Test evaluate.py with 'center' baseline ---
    cmd_eval_center = [
        sys.executable,
        "evaluate.py",
        "--checkpoint", "center",
        "--config", str(config_path),
        "--manifest", str(val_manifest),
        "--image-dir", str(val_img_dir),
        "--map-dir", str(val_map_dir),
        "--device", "cpu",
    ]
    res_eval_center = subprocess.run(cmd_eval_center, capture_output=True, text=True, check=True)
    assert "Loaded baseline CenterBiasBaseline model." in res_eval_center.stdout

    # --- 5. Test predict.py ---
    pred_dir = tmp_path / "outputs" / "predictions"
    cmd_predict = [
        sys.executable,
        "predict.py",
        "--checkpoint", str(run_dir / "checkpoints" / "best.pth"),
        "--config", str(config_path),
        "--input", str(test_img_dir),
        "--output-dir", str(pred_dir),
        "--save-raw",
        "--save-overlay",
        "--save-heatmap",
        "--device", "cpu",
    ]
    res_pred = subprocess.run(cmd_predict, capture_output=True, text=True, check=True)
    assert "Predictions successfully generated" in res_pred.stdout
    assert (pred_dir / "test_0.png").is_file()
    assert (pred_dir / "test_0_overlay.png").is_file()
    assert (pred_dir / "test_0_heatmap.png").is_file()
    assert (pred_dir / "test_1.png").is_file()
    assert (pred_dir / "test_1_overlay.png").is_file()
    assert (pred_dir / "test_1_heatmap.png").is_file()
