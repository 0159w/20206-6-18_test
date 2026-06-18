"""
YOLOv8 Inference Script — Fire Detection.

Loads the trained model and runs inference on images/videos.

Usage:
    # Single image
    python module1_yolo/predict.py --source path/to/image.jpg

    # Video / webcam
    python module1_yolo/predict.py --source path/to/video.mp4
    python module1_yolo/predict.py --source 0  # webcam

    # Directory
    python module1_yolo/predict.py --source ./test_images/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from module1_yolo.config import CONF_THRESHOLD, IOU_THRESHOLD


def predict(source: str, weights: str = "module1_yolo/runs/train/weights/best.pt"):
    """
    Run YOLOv8 inference on images/videos.

    Args:
        source: Path to image, video, directory, or 0 for webcam.
        weights: Path to trained model weights.
    """
    # Load trained model
    model = YOLO(weights)

    print(f"🔍 Running inference with {weights}")
    print(f"   Source: {source}")
    print(f"   Confidence threshold: {CONF_THRESHOLD}")
    print(f"   IOU threshold: {IOU_THRESHOLD}")

    # Run inference
    results = model(
        source=source,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,          # Save annotated images
        save_txt=True,      # Save YOLO format labels
        project="module1_yolo/runs",
        name="predict",
        exist_ok=True,
    )

    print(f"\n✅ Inference complete!")
    print(f"   Results saved to: module1_yolo/runs/predict/")

    # Print detection summary
    total_fire = 0
    for r in results:
        fire_count = sum(1 for c in r.boxes.cls if int(c) == 0) if r.boxes else 0
        total_fire += fire_count

    print(f"   Total fire detections: {total_fire}")
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Fire Detection Inference")
    parser.add_argument("--source", default="./test_images", help="Image, video, directory, or webcam (0)")
    parser.add_argument("--weights", default="module1_yolo/runs/train/weights/best.pt", help="Model weights")
    args = parser.parse_args()

    predict(source=args.source, weights=args.weights)


if __name__ == "__main__":
    main()
