"""Application configuration."""

from pathlib import Path

# Database
DATABASE_URL = "sqlite:///./mine_inspection.db"

# Upload directory for photos (directory is created in lifespan)
UPLOAD_DIR = Path("./uploads")

# YOLO model weights path
YOLO_WEIGHTS_PATH = "module1_yolo/runs/train/weights/best.pt"

# Pagination defaults
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
