"""Run FDPG-RDet oriented-box prediction on images or a directory."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument("--source", required=True, help="Image, directory, video, or supported stream")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(ROOT / "runs/predict"))
    parser.add_argument("--name", default="fdpg-rdet")
    parser.add_argument("--save-txt", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights, task="obb")
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
    )


if __name__ == "__main__":
    main()
