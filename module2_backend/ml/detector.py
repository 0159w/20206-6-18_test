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
    # if detections is not None and len(detections.cls) > 0:
    #     cls_tensor = detections.cls.int().tolist()
    #     conf_tensor = detections.conf.tolist()
    #     fire_detected = any(
    #         cls_id == 0 and conf_val > 0.5
    #         for cls_id, conf_val in zip(cls_tensor, conf_tensor)
    #     )
    #     is_safe = not fire_detected
    #     confidence = max(conf_tensor) if conf_tensor else 0.0
    # else:
    #     is_safe = True
    #     confidence = 0.0

    # ── Placeholder for coding test ──
    is_safe = True       # Simulate: no fire detected
    confidence = 0.98    # High confidence

    elapsed_ms = (time.perf_counter() - start) * 1000
    return is_safe, confidence, elapsed_ms
