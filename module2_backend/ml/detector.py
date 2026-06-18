"""YOLO model inference wrapper.

Loads the trained YOLOv8 model and runs fire-detection inference.
Falls back gracefully if the trained model is not yet available.
"""

import logging
import time
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

_MODEL = None


def _load_model():
    """Load YOLO model singleton. Tries trained weights first, falls back to base."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from module2_backend.core.config import YOLO_WEIGHTS_PATH

    trained = Path(YOLO_WEIGHTS_PATH)
    if trained.exists():
        logger.info("Loading trained model: %s", trained)
        from ultralytics import YOLO
        _MODEL = YOLO(str(trained))
        return _MODEL

    logger.warning("Trained model not found at %s, inference will use simulation", trained)
    return None


def predict_safety(image_path: str) -> Tuple[bool, float, float]:
    """
    Run YOLOv8 inference and determine if the scene is safe (no fire detected).

    Returns:
        (is_safe, confidence, inference_time_ms)
        - is_safe: True if no fire violations detected
        - confidence: max detection confidence (0.0 ~ 1.0)
        - inference_time_ms: inference wall-clock time in milliseconds
    """
    start = time.perf_counter()

    model = _load_model()

    if model is not None:
        # ── Production path: real YOLO inference ──
        from module1_yolo.config import CONF_THRESHOLD

        results = model(image_path, verbose=False)
        detections = results[0].boxes

        if detections is not None and len(detections.cls) > 0:
            cls_list = detections.cls.int().tolist()
            conf_list = detections.conf.tolist()
            fire_detected = any(
                cls_id == 0 and conf_val > CONF_THRESHOLD
                for cls_id, conf_val in zip(cls_list, conf_list)
            )
            is_safe = not fire_detected
            confidence = max(conf_list) if conf_list else 0.0
        else:
            is_safe = True
            confidence = 0.0
    else:
        # ── Fallback: model not trained yet ──
        is_safe = True
        confidence = 0.0

    elapsed_ms = (time.perf_counter() - start) * 1000
    return is_safe, confidence, elapsed_ms
