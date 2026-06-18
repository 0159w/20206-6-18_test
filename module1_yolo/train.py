"""
YOLOv8 Training Script — Fire Detection with Transfer Learning.

This script:
  1. Loads a pre-trained YOLOv8 model (yolov8n.pt)
  2. Fine-tunes on the fire dataset using transfer learning
  3. Saves the best weights to runs/train/weights/best.pt

Usage:
    python module1_yolo/train.py

Requirements:
    pip install ultralytics pyyaml
    Dataset in YOLO format under ./data/fire_dataset/ (see data_converter.py)
"""

import yaml
from pathlib import Path
from ultralytics import YOLO

from module1_yolo.config import (
    DATASET_ROOT,
    MODEL_NAME,
    IMG_SIZE,
    EPOCHS,
    BATCH_SIZE,
    DEVICE,
    WORKERS,
    LR,
    PATIENCE,
    CLASS_NAMES,
)


def create_dataset_yaml():
    """
    Create data.yaml configuration for YOLOv8 training.
    YOLOv8 expects a data.yaml pointing to train/val image directories.
    """
    data_yaml = {
        "train": str(Path(DATASET_ROOT) / "images/train"),
        "val": str(Path(DATASET_ROOT) / "images/val"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = Path(DATASET_ROOT) / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"✓ Dataset YAML created at {yaml_path}")
    return str(yaml_path)


def train():
    """
    Train YOLOv8 on fire dataset using transfer learning.

    Steps:
        1. Load pre-trained YOLOv8n model
        2. Freeze backbone layers (optional, for faster fine-tuning)
        3. Train on fire dataset
        4. Validate and save best weights
    """
    print("🚀 Starting YOLOv8 Fire Detection Training")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Dataset: {DATASET_ROOT}")
    print(f"   Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Device: {DEVICE}")
    print(f"   Image Size: {IMG_SIZE}, Learning Rate: {LR}")

    # Create dataset config
    data_yaml = create_dataset_yaml()

    # ── 1. Load pre-trained model (transfer learning) ──
    model = YOLO(MODEL_NAME)

    # ── 2. Fine-tune ──
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        lr0=LR,
        patience=PATIENCE,
        project="module1_yolo/runs",
        name="train",
        exist_ok=True,
        pretrained=True,  # Transfer learning: use pre-trained weights
        optimizer="AdamW",
        amp=True,         # Mixed precision for faster training
    )

    print("\n✅ Training complete!")
    print(f"   Best weights: module1_yolo/runs/train/weights/best.pt")
    return results


if __name__ == "__main__":
    train()
