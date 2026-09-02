# FDPG-RDet

Official research implementation of **Frequency-Dynamic and Position-Guided Rotated Detection for Multitemporal Object-Level Mapping of Plastic Greenhouses**.

FDPG-RDet is an oriented-object detector designed for the elongated geometry, dense arrangement, repetitive roof texture, and arbitrary orientation of plastic greenhouses in high-resolution remote-sensing imagery. It augments a YOLOv8s-OBB detector with two task-specific components:

- **FDR-Adapter**: a Frequency-Dynamic Residual Adapter that strengthens direction-sensitive and periodic greenhouse features.
- **EPGI**: an Edge-aware Positional Guidance Injection module that restores shallow boundary and positional cues for more accurate oriented-box localization.

> **Release status:** training, validation, and prediction code is available. Model weights and the dataset are not stored in this repository.

## Architecture

![FDPG-RDet architecture](assets/architecture.png)

The FDR-Adapter and EPGI modules are inserted into the P3 and P4 detection branches, while the standard three-scale OBB head operates on P3, P4, and P5 features.

## Main results

Evaluation on the plastic-greenhouse OBB dataset used in the manuscript produced the following results:

| Precision (%) | Recall (%) | F1-score (%) | mAP<sub>50</sub> (%) | mAP<sub>50-95</sub> (%) |
| ---: | ---: | ---: | ---: | ---: |
| 86.06 | 84.37 | 85.20 | 91.41 | 71.53 |

### Qualitative comparison

![Qualitative comparison](assets/qualitative_comparison.png)

### Ablation study

![Ablation comparison](assets/ablation_results.png)

## Repository layout

```text
FDPG-RDet/
├── assets/                         # Selected manuscript figures
├── configs/
│   ├── yolov8s-fdpg-rdet-obb.yaml # Final model architecture
│   └── greenhouse-obb.example.yaml
├── fdpg/                           # Public Ultralytics integration
├── fdpg_runtime/                   # Protected runnable FDPG modules
├── pyarmor_runtime_000000/         # Windows/Linux runtime libraries
├── scripts/
│   ├── train.py
│   ├── val.py
│   ├── predict.py
│   └── smoke_test.py
└── requirements.txt
```

Paper-figure generation, geographic analysis, manual-validation utilities, datasets, experiment runs, and model weights are intentionally excluded.

## Installation

The protected research runtime currently supports **Python 3.11** on **Windows x86_64** and **Linux x86_64**.

```bash
git clone https://github.com/RS-Hong/FDPG-RDet.git
cd FDPG-RDet

conda create -n fdpg-rdet python=3.11 -y
conda activate fdpg-rdet
pip install -r requirements.txt
```

Verify the installation:

```bash
python scripts/smoke_test.py
```

The repository is pinned to `ultralytics==8.3.221` because the model-registration interface is version-sensitive.

## Dataset

The dataset follows the Ultralytics YOLO oriented-bounding-box format:

```text
FDPG-Greenhouse-OBB/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label row contains a class index followed by four normalized corner coordinates:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Copy `configs/greenhouse-obb.example.yaml` to `configs/greenhouse-obb.yaml` and replace its `path` with the absolute dataset directory.

### Dataset download

> **Dataset link:** `[To be added]`
>
> **Access code:** `[To be added, if required]`

## Training

The default options reproduce the main manuscript setup: 512 x 512 input, AdamW, 300 maximum epochs, early-stopping patience of 60, and random seed 42.

```bash
python scripts/train.py \
  --data configs/greenhouse-obb.yaml \
  --device 0 \
  --batch 32 \
  --epochs 300
```

Training outputs are written to `runs/train/` and are ignored by Git.

## Validation

Weights are not included. Supply a locally trained or separately downloaded checkpoint:

```bash
python scripts/val.py \
  --weights /path/to/best.pt \
  --data configs/greenhouse-obb.yaml \
  --split test \
  --device 0
```

## Prediction

```bash
python scripts/predict.py \
  --weights /path/to/best.pt \
  --source /path/to/images \
  --conf 0.40 \
  --iou 0.60 \
  --device 0 \
  --save-txt
```

Predicted oriented boxes and optional text labels are written to `runs/predict/`.

## Loading FDPG-RDet in Python

Always use `FDPGModel` instead of constructing `ultralytics.YOLO` directly. The wrapper registers the protected FDPG layers before the model YAML or checkpoint is loaded.

```python
from fdpg import FDPGModel

model = FDPGModel("configs/yolov8s-fdpg-rdet-obb.yaml")
model.train(data="configs/greenhouse-obb.yaml", imgsz=512, epochs=300)

trained = FDPGModel("/path/to/best.pt")
results = trained.predict("/path/to/images", imgsz=512, conf=0.40)
```

## Protected-module notice

The executable implementations of FDR-Adapter and EPGI are distributed in obfuscated form to protect the unpublished method while preserving model construction, training, validation, and prediction. Obfuscation raises the cost of casual source inspection but should not be interpreted as cryptographic secrecy. The readable implementation can be released after the associated paper's publication.

The current protected build was generated with the PyArmor 9.2.7 non-profit trial runtime and is intended for non-commercial academic evaluation. Replace it with a properly licensed build before any commercial distribution or use.

Do not remove or rename `fdpg_runtime/` or `pyarmor_runtime_000000/`.

## Citation

If this repository contributes to your research, please cite the associated paper after its bibliographic information becomes available. A complete BibTeX entry will be added after publication.

```bibtex
@article{hong_fdpg_rdet,
  title   = {Frequency-Dynamic and Position-Guided Rotated Detection for Multitemporal Object-Level Mapping of Plastic Greenhouses},
  author  = {Hong, Ruikai and others},
  note    = {Manuscript under review}
}
```

## Acknowledgements

This project uses [Ultralytics](https://github.com/ultralytics/ultralytics) as its detection framework. Please also comply with the licenses and citation requirements of all upstream dependencies.

## Contact

For questions, please open a GitHub issue in this repository.
