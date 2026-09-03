"""Build FDPG-RDet and run a small forward/backward smoke test."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / ".ultralytics"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))

from ultralytics import YOLO  # noqa: E402


def main():
    model = YOLO(ROOT / "ultralytics/cfg/models/v8/yolov8s-fdpg-rdet-obb.yaml", task="obb").model
    model.train()
    outputs = model(torch.randn(2, 3, 128, 128))
    tensors = []
    for output in outputs if isinstance(outputs, (list, tuple)) else [outputs]:
        tensors.extend(output if isinstance(output, (list, tuple)) else [output])
    loss = sum(x.float().mean() for x in tensors if torch.is_tensor(x))
    loss.backward()
    print(f"FDPG-RDet smoke test passed (parameters={sum(p.numel() for p in model.parameters()):,}).")


if __name__ == "__main__":
    main()
