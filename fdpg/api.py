"""Select both a trained checkpoint and its correct train/val/predict route."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("modes.yaml")
MODES = ("nms", "nms-free")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


class FDPG:
    """Switch between separate conventional/IPC checkpoints, loaded on demand.

    Examples:
        model = FDPG(mode="nms")
        conventional = model.predict("image.png", device="0")
        model.switch_mode("nms-free")
        ipc = model.predict("image.png", device="0")

    A mode switch discards the old predictor instead of reusing its cached
    postprocessing route. It never rewrites weights or changes a checkpoint's
    training method. One instance is intended for sequential use.
    """

    def __init__(self, mode=None, weights=None, config=None):
        self.config_path = Path(config).resolve() if config else DEFAULT_CONFIG
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(self.config, dict) or not isinstance(self.config.get("modes"), dict):
            raise ValueError("FDPG configuration must contain a modes mapping")
        self._weight_overrides = {}
        self._yolo = None
        self._loaded_checkpoint = None
        self.switch_mode(mode or self.config["default_mode"], weights=weights)

    @property
    def mode(self):
        return self._mode

    @property
    def checkpoint(self):
        return self._weight_overrides.get(
            self.mode, repository_path(self.config["modes"][self.mode]["weights"])
        )

    def switch_mode(self, mode, weights=None):
        if mode not in MODES or mode not in self.config["modes"]:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if weights is not None:
            self._weight_overrides[mode] = repository_path(weights)
        self._mode = mode
        self._yolo = None
        self._loaded_checkpoint = None
        return self

    def describe(self):
        return {
            "mode": self.mode,
            "weights": str(self.checkpoint),
            "weights_exist": self.checkpoint.is_file(),
            "postprocess": "rotated NMS" if self.mode == "nms" else "confidence filter + score Top-K; no NMS",
            "training_loss": "native one-to-many OBB" if self.mode == "nms" else "one-to-one OBB + IPC (0.2)",
            "config": str(self.config_path),
        }

    @staticmethod
    def _wrapper(mode):
        # Keep `--help` and `info` usable without loading torch or any weights.
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics_config"))
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from .training import ConventionalYOLO, IPCYOLO

        return ConventionalYOLO if mode == "nms" else IPCYOLO

    def load(self):
        if self._yolo is not None:
            return self._yolo
        path = self.checkpoint
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise FileNotFoundError(f"A local trained .pt checkpoint is required: {path}")
        spec = self.config["modes"][self.mode]
        registered_path = repository_path(spec["weights"])
        if path == registered_path and spec.get("sha256") and file_hash(path) != spec["sha256"]:
            raise ValueError(f"Registered {self.mode} checkpoint SHA256 mismatch: {path}")
        candidate = self._wrapper(self.mode)(str(path), task="obb")
        from .training import check_model_mode

        check_model_mode(candidate.model, self.mode)
        self._yolo = candidate
        self._loaded_checkpoint = path
        return candidate

    def _arguments(self, task, kwargs):
        # Disallow bypassing the chosen implementation via a raw Ultralytics
        # predictor/trainer/validator or an export-only `nms` flag.
        forbidden = {"model", "task", "mode", "nms", "predictor", "validator", "trainer", "cfg"} & kwargs.keys()
        if forbidden:
            raise ValueError(f"Use FDPG's mode/config switch instead of overriding {sorted(forbidden)}")
        if self.mode == "nms-free" and "iou" in kwargs:
            raise ValueError("NMS-free mode has no NMS IoU threshold")
        args = dict(self.config.get("inference", {}))
        if self.mode == "nms-free":
            args.pop("iou", None)
        args.update(conf=0.5 if task == "predict" else 0.01, workers=0, plots=False, verbose=False)
        args.update(project=str(ROOT / "runs/fdpg_switch" / self.mode), name=task, exist_ok=False)
        if task == "val":
            args.update(data=str(repository_path(self.config["data"])), split="val")
        args.update(kwargs)
        return args

    def predict(self, source, **kwargs):
        if source is None:
            raise ValueError("An image, directory, array or other explicit prediction source is required")
        args = self._arguments("predict", kwargs)
        # AutoBackend mutates its model when it fuses Conv-BN or casts to half.
        # Reload so predict/val order and a previous half=True call cannot
        # change a subsequent FP32 run. Weight loading is outside inference.
        self._yolo = None
        model = self.load()
        return model.predict(source=source, **args)

    def val(self, **kwargs):
        self._yolo = None
        return self.load().val(**self._arguments("val", kwargs))

    def train(self, *, resume=None, **kwargs):
        """Start a NEW random-initialized run, or explicitly resume its last.pt.

        The registered paper checkpoints are never used as training inputs.
        Completed/stripped checkpoints cannot restore optimizer state.
        """
        forbidden = {"model", "task", "mode", "nms", "predictor", "validator", "trainer", "cfg", "pretrained"} & kwargs.keys()
        if forbidden:
            raise ValueError(f"Training owns the model and criterion; do not override {sorted(forbidden)}")
        wrapper = self._wrapper(self.mode)
        if resume:
            unsupported = kwargs.keys() - {"data", "device", "workers"}
            if unsupported:
                raise ValueError(f"Resume preserves the original training configuration; cannot override {sorted(unsupported)}")
            args = {"data": str(repository_path(self.config["data"])), **kwargs}
            path = repository_path(resume)
            # Use the same strict mode check as inference before starting a
            # trainer, and check resumability before creating a run directory.
            probe = FDPG(self.mode, weights=path, config=self.config_path)
            model = probe.load()
            ckpt = model.ckpt
            if ckpt.get("optimizer") is None or ckpt.get("epoch", -1) < 0:
                raise ValueError("This checkpoint has no resumable optimizer/epoch; use an unfinished run's last.pt")
            args["resume"] = str(path)
        else:
            args = dict(self.config.get("training", {}))
            args.update(data=str(repository_path(self.config["data"])),
                        project=str(ROOT / "runs/fdpg_switch" / self.mode), name="train", exist_ok=False,
                        pretrained=False, resume=False)
            args.update(kwargs)
            model = wrapper(str(repository_path(self.config["model"])), task="obb")

        def record_mode(trainer):
            evidence = {"mode": self.mode, "initialization": str(resume) if resume else "random",
                        "mode_switch_during_run": False, "args": vars(trainer.args)}
            (trainer.save_dir / "fdpg_mode.json").write_text(
                json.dumps(evidence, ensure_ascii=True, indent=2, default=str), encoding="utf-8"
            )

        model.add_callback("on_pretrain_routine_end", record_mode)
        result = model.train(**args)
        self._yolo = model
        self._loaded_checkpoint = (model.trainer.best if model.trainer.best.exists() else model.trainer.last).resolve()
        model.ckpt_path = str(self._loaded_checkpoint)
        self._weight_overrides[self.mode] = self._loaded_checkpoint
        return result
