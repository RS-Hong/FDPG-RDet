"""Conventional and IPC training routes for the single-class FDPG OBB model."""

from copy import copy

from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBPredictor, OBBTrainer
from ultralytics.nn.modules.head import OBB
from ultralytics.nn.tasks import OBBModel

from .checkpoint import load_local_checkpoint
from .runtime import ConventionalValidator, IPCModel, IPCPredictor, IPCValidator


def check_model_mode(model, mode):
    if not isinstance(model, OBBModel):
        raise ValueError("Expected a native OBB model checkpoint")
    actual = "nms-free" if isinstance(model, IPCModel) else "nms"
    if actual == "nms" and type(model) is not OBBModel:
        raise ValueError("Unsupported OBB model subclass")
    if type(model.model[-1]) is not OBB or model.model[-1].nc != 1:
        raise ValueError("Expected the original single-class OBB head")
    if actual != mode or getattr(model, "fdpg_mode", actual) != actual:
        raise ValueError(f"Checkpoint mode is {actual!r}, requested {mode!r}; use matching trained weights")


class ConventionalTrainer(OBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None:
            check_model_mode(weights, "nms")
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.fdpg_mode = "nms"
        return model

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return ConventionalValidator(self.test_loader, save_dir=self.save_dir,
                                     args=copy(self.args), _callbacks=self.callbacks)


class IPCTrainer(OBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data["nc"] != 1:
            raise ValueError("FDPG-IPC supports one class")
        if weights is not None:
            check_model_mode(weights, "nms-free")
        model = IPCModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        model.fdpg_mode = "nms-free"
        if weights is not None:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "center_loss", "competition_loss")
        return IPCValidator(self.test_loader, save_dir=self.save_dir,
                            args=copy(self.args), _callbacks=self.callbacks)


class _LocalCheckpointYOLO(YOLO):
    def _load(self, weights, task=None):
        self.model, self.ckpt = load_local_checkpoint(weights)
        self.task = "obb"
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
        self.ckpt_path = self.model.pt_path
        self.overrides.update(model=self.ckpt_path, task=self.task)
        self.model_name = self.ckpt_path


class ConventionalYOLO(_LocalCheckpointYOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping["obb"] = dict(model=OBBModel, trainer=ConventionalTrainer,
                              validator=ConventionalValidator, predictor=OBBPredictor)
        return mapping


class IPCYOLO(_LocalCheckpointYOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping["obb"] = dict(model=IPCModel, trainer=IPCTrainer,
                              validator=IPCValidator, predictor=IPCPredictor)
        return mapping
