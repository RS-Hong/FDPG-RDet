"""Native OBB inference with confidence filtering and score Top-K."""

from contextlib import contextmanager
from functools import partial
from threading import RLock

import torch
import ultralytics.engine.validator as validator_module
from ultralytics.models.yolo.obb import OBBPredictor, OBBValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import attempt_compile, select_device

_VALIDATION_LOCK = RLock()


def select_without_nms(predictions, conf, max_det, classes=None):
    tensor = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    if tensor.ndim != 3 or tensor.shape[1] != 6:
        raise ValueError("Expected single-class OBB predictions [B, 6, N]")
    selected = []
    for rows in tensor.transpose(1, 2):
        rows = rows[rows[:, 4] > conf]
        if classes is not None and 0 not in classes:
            rows = rows[:0]
        rows = rows[rows[:, 4].argsort(descending=True)[:max_det]]
        selected.append(torch.cat([rows[:, :5], torch.zeros_like(rows[:, :1]), rows[:, 5:]], 1))
    return selected


class ForwardOnlyBackend(AutoBackend):
    def warmup(self, imgsz=(1, 3, 640, 640)):
        # The framework's default warmup also invokes NMS on dummy boxes.
        if self.device.type != "cpu":
            self.forward(torch.zeros(*imgsz, device=self.device, dtype=torch.half if self.fp16 else torch.float))


@contextmanager
def validation_backend():
    with _VALIDATION_LOCK:
        previous = validator_module.AutoBackend
        try:
            validator_module.AutoBackend = partial(ForwardOnlyBackend, fuse=False)
            yield
        finally:
            validator_module.AutoBackend = previous


class FP32ValidationTrainer:
    amp = False

    def __init__(self, trainer):
        self._trainer = trainer

    def __getattr__(self, name):
        return getattr(self._trainer, name)


class ConventionalValidator(OBBValidator):
    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            return super().__call__(trainer=FP32ValidationTrainer(trainer), model=model)
        with validation_backend():
            return super().__call__(model=model)


class IPCValidator(ConventionalValidator):
    def postprocess(self, predictions):
        return [{"bboxes": torch.cat([rows[:, :4], rows[:, 6:7]], 1).float(),
                 "conf": rows[:, 4].float(), "cls": rows[:, 5].float()}
                for rows in select_without_nms(predictions, self.args.conf, self.args.max_det, self.args.classes)]


class IPCPredictor(OBBPredictor):
    def setup_model(self, model, verbose=True):
        self.model = ForwardOnlyBackend(
            model=model or self.args.model, device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn, data=self.args.data, fp16=self.args.half, fuse=True, verbose=verbose,
        )
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

    def postprocess(self, predictions, img, orig_imgs, **kwargs):
        selected = select_without_nms(predictions, self.args.conf, self.args.max_det, self.args.classes)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        return self.construct_results(selected, img, orig_imgs)


class IPCModel(OBBModel):
    def init_criterion(self):
        from .loss import IPCLossAdapter

        return IPCLossAdapter(self)
