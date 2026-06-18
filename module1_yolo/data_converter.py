"""
Data Converter — Kaggle Fire Dataset → YOLO format.

This script converts the Kaggle fire detection dataset (usually in
VOC XML or COCO JSON format) to standard YOLO detection format:

    data/fire_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml       ← Dataset config for YOLOv8 training

Usage:
    python module1_yolo/data_converter.py --input /path/to/kaggle_fire --output ./data/fire_dataset

Note: The Kaggle fire dataset (https://www.kaggle.com/datasets/atulyakumar98/fire-dataset)
      provides annotations in COCO JSON format (annotations.json).
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def convert_coco_to_yolo(
    coco_json_path: str,
    image_dir: str,
    output_dir: str,
    split: str = "train",
):
    """
    Convert COCO JSON annotations to YOLO format.

    YOLO format: one .txt file per image, each line: class_id x_center y_center width height
    All values normalized to [0, 1].
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Build image_id → filename mapping
    img_map = {img["id"]: img["file_name"] for img in coco["images"]}

    # Build category_id → class index mapping (0-based)
    cat_map = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}

    # Group annotations by image_id
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Output paths
    out_img_dir = Path(output_dir) / "images" / split
    out_label_dir = Path(output_dir) / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    img_src_dir = Path(image_dir)

    for img_id, anns in anns_by_img.items():
        filename = img_map[img_id]
        img_width = coco["images"][0].get("width", 640)
        img_height = coco["images"][0].get("height", 640)

        # Copy image
        src = img_src_dir / filename
        if src.exists():
            shutil.copy2(src, out_img_dir / filename)

        # Write YOLO label file
        label_path = out_label_dir / Path(filename).with_suffix(".txt")
        with open(label_path, "w") as f:
            for ann in anns:
                cat_id = cat_map[ann["category_id"]]
                bbox = ann["bbox"]  # COCO: [x, y, width, height]
                x, y, w, h = bbox
                # Convert to YOLO normalized format
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"✓ Converted {len(anns_by_img)} images to YOLO format ({split} split)")


def create_data_yaml(output_dir: str, class_names: list):
    """Create data.yaml for YOLOv8 training."""
    data_yaml = {
        "train": str(Path(output_dir) / "images/train"),
        "val": str(Path(output_dir) / "images/val"),
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, "w") as f:
        import yaml
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"✓ Created {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Kaggle Fire Dataset to YOLO format")
    parser.add_argument("--input", required=True, help="Path to Kaggle dataset root")
    parser.add_argument("--output", default="./data/fire_dataset", help="Output directory")
    parser.add_argument("--classes", nargs="+", default=["fire"], help="Class names")
    args = parser.parse_args()

    # Convert train split
    convert_coco_to_yolo(
        coco_json_path=os.path.join(args.input, "annotations", "instances_train.json"),
        image_dir=os.path.join(args.input, "train"),
        output_dir=args.output,
        split="train",
    )

    # Convert val split
    convert_coco_to_yolo(
        coco_json_path=os.path.join(args.input, "annotations", "instances_val.json"),
        image_dir=os.path.join(args.input, "val"),
        output_dir=args.output,
        split="val",
    )

    create_data_yaml(args.output, args.classes)
    print("✅ Dataset conversion complete!")
