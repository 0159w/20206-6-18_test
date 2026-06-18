"""
Configuration loader for YOLOv8 module.
Loads settings from config.yaml.
"""

import os
import yaml

_CONFIG = None


def load_config():
    global _CONFIG
    if _CONFIG is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


# Dataset
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "fire_dataset"))

# Classes
CLASS_NAMES = load_config().get("class_names", ["fire"])

# Training
MODEL_NAME = load_config().get("model_name", "yolov8n.pt")
IMG_SIZE = load_config().get("img_size", 640)
EPOCHS = load_config().get("epochs", 50)
BATCH_SIZE = load_config().get("batch_size", 16)
DEVICE = load_config().get("device", "cuda")
WORKERS = load_config().get("workers", 4)
LR = load_config().get("lr", 0.001)
PATIENCE = load_config().get("patience", 10)

# Inference
CONF_THRESHOLD = load_config().get("conf_threshold", 0.5)
IOU_THRESHOLD = load_config().get("iou_threshold", 0.45)

# Ensure dataset root is absolute
dataset_root_abs = os.environ.get("FIRE_DATASET_ROOT")
if dataset_root_abs:
    DATASET_ROOT = dataset_root_abs
