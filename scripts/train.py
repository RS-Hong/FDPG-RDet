"""Train FDPG-RDet on a YOLO OBB dataset."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / ".ultralytics"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))

from ultralytics import YOLO  # noqa: E402

DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/v8/yolov8s-fdpg-rdet-obb.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs/train"))
    parser.add_argument("--name", default="fdpg-rdet-s")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model, task="obb")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        patience=60,
        seed=42,
        deterministic=True,
        amp=True,
        box=10.0,
        cls=0.25,
        dfl=2.0,
        degrees=90,
        translate=0.1,
        scale=0.35,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.7,
        close_mosaic=25,
    )


if __name__ == "__main__":
    main()
