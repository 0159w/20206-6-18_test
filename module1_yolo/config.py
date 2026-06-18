"""
Configuration loader for YOLOv8 module.
Loads settings from config.yaml. Every value can be overridden by an environment variable.
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


def _env(name: str, default):
    """Return os.environ[name] if set, otherwise the yaml / fallback value."""
    return os.environ.get(name, default)


# Dataset
DATASET_ROOT = os.environ.get(
    "FIRE_DATASET_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "fire_dataset")),
)

# Classes
CLASS_NAMES = load_config().get("class_names", ["fire"])

# Training
MODEL_NAME = _env("YOLO_MODEL_NAME", load_config().get("model_name", "yolov8n.pt"))
IMG_SIZE = int(_env("YOLO_IMG_SIZE", load_config().get("img_size", 640)))
EPOCHS = int(_env("YOLO_EPOCHS", load_config().get("epochs", 50)))
BATCH_SIZE = int(_env("YOLO_BATCH_SIZE", load_config().get("batch_size", 16)))
DEVICE = _env("YOLO_DEVICE", load_config().get("device", "cuda"))
WORKERS = int(_env("YOLO_WORKERS", load_config().get("workers", 4)))
LR = float(_env("YOLO_LR", load_config().get("lr", 0.001)))
PATIENCE = int(_env("YOLO_PATIENCE", load_config().get("patience", 10)))

# Inference
CONF_THRESHOLD = float(_env("YOLO_CONF_THRESHOLD", load_config().get("conf_threshold", 0.5)))
IOU_THRESHOLD = float(_env("YOLO_IOU_THRESHOLD", load_config().get("iou_threshold", 0.45)))
