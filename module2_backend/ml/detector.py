"""YOLO model inference wrapper.

In production, this loads the trained YOLOv8 model.
For the coding test, we simulate the inference result.
"""

import time
from pathlib import Path
from typing import Tuple


def predict_safety(image_path: str) -> Tuple[bool, float, float]:
    """
    Run YOLOv8 inference on the input image and determine if the scene is safe.

    In production:
        - Loads the YOLOv8 model from `YOLO_WEIGHTS_PATH`
        - Runs inference on the image
        - Checks if any fire/smoke detections exceed the confidence threshold
        - Returns (is_safe, confidence, inference_time_ms)

    For the coding test, this simulates a model call and always
    returns a placeholder result.

    Args:
        image_path: Path to the input image file.

    Returns:
        Tuple of (is_safe: bool, confidence: float, inference_time_ms: float)
        - is_safe: True if no violations detected, False otherwise
        - confidence: Detection confidence score (0.0 ~ 1.0)
        - inference_time_ms: Time spent on inference in milliseconds
    """
    start = time.perf_counter()

    # ── Production code (commented out for test) ──
    # from ultralytics import YOLO
    # model = YOLO(YOLO_WEIGHTS_PATH)
    # results = model(image_path)
    # detections = results[0].boxes
    # fire_detected = any(
    #     int(cls) == 0 and conf > 0.5
    #     for cls, conf in zip(detections.cls, detections.conf)
    # )
    # is_safe = not fire_detected
    # confidence = float(detections.conf.max()) if len(detections) > 0 else 0.0

    # ── Placeholder for coding test ──
    is_safe = True       # Simulate: no fire detected
    confidence = 0.98    # High confidence

    elapsed_ms = (time.perf_counter() - start) * 1000
    return is_safe, confidence, elapsed_ms
