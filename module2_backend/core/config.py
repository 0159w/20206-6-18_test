"""Application configuration — overridable via environment variables."""

import os
from pathlib import Path

# Database
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./mine_inspection.db")

# Upload directory for photos (directory is created in lifespan)
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))

# YOLO model weights path
YOLO_WEIGHTS_PATH = os.environ.get(
    "YOLO_WEIGHTS_PATH",
    "module1_yolo/runs/train/weights/best.pt",
)

# Pagination defaults
DEFAULT_PAGE = int(os.environ.get("DEFAULT_PAGE", "1"))
DEFAULT_PAGE_SIZE = int(os.environ.get("DEFAULT_PAGE_SIZE", "20"))
