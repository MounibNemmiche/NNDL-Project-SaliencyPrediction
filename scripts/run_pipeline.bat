@echo off
setlocal enabledelayedexpansion
set RUN_NAME=dummy_run_%RANDOM%
echo =====================================================================
echo   Vision Saliency Prediction - End-to-End Pipeline Verification
echo =====================================================================

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    exit /b 1
)

echo.
echo [1] Setting up dummy dataset for pipeline verification...
python -c "import os, numpy as np; from PIL import Image; [os.makedirs(f'data/dummy/images/{s}', exist_ok=True) for s in ['train', 'val', 'test']]; [os.makedirs(f'data/dummy/maps/{s}', exist_ok=True) for s in ['train', 'val']]; [[Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)).save(f'data/dummy/images/{s}/COCO_{s}2014_{i:012d}.jpg') for i in range(10 if s=='train' else 5)] for s in ['train', 'val', 'test']]; [[Image.fromarray(np.random.randint(0, 256, (224, 224), dtype=np.uint8)).save(f'data/dummy/maps/{s}/COCO_{s}2014_{i:012d}.png') for i in range(10 if s=='train' else 5)] for s in ['train', 'val']]"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create dummy datasets.
    exit /b 1
)
echo Dummy data generated in data/dummy/

echo.
echo [2] Generating deterministic splits manifest...
python -m src.datasets.create_splits --image-dir data/dummy/images/val --map-dir data/dummy/maps/val --val-size 2 --seed 42 --val-output data/dummy/val_manifest.txt --test-output data/dummy/test_manifest.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to generate splits.
    exit /b 1
)

echo.
echo [3] Running dummy training (2 epochs, CPU, Fusion model)...
python src/train.py --model fusion --epochs 2 --batch-size 2 --train-image-dir data/dummy/images/train --train-map-dir data/dummy/maps/train --val-image-dir data/dummy/images/val --val-map-dir data/dummy/maps/val --val-manifest data/dummy/val_manifest.txt --run-name %RUN_NAME% --device cpu --train-samples 4 --val-samples 2
if %errorlevel% neq 0 (
    echo [ERROR] Training failed.
    exit /b 1
)

echo.
echo [4] Running evaluation on validation manifest...
python src/evaluate.py --checkpoint outputs/runs/%RUN_NAME%/checkpoints/best.pth --manifest data/dummy/val_manifest.txt --image-dir data/dummy/images/val --map-dir data/dummy/maps/val --output-dir outputs/runs/%RUN_NAME%/evaluation --visualize outputs/runs/%RUN_NAME%/prediction_grid.png --device cpu
if %errorlevel% neq 0 (
    echo [ERROR] Evaluation failed.
    exit /b 1
)

echo.
echo [5] Generating predictions on test images...
python src/predict.py --checkpoint outputs/runs/%RUN_NAME%/checkpoints/best.pth --input data/dummy/images/test --output-dir outputs/runs/%RUN_NAME%/predictions --save-raw --save-overlay --save-heatmap --device cpu
if %errorlevel% neq 0 (
    echo [ERROR] Prediction generation failed.
    exit /b 1
)

echo.
echo =====================================================================
echo   Pipeline verified successfully!
echo   Outputs are saved in: outputs/runs/%RUN_NAME%/
echo =====================================================================
pause
