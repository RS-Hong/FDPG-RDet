"""Load current checkpoints and the retained single-head IPC checkpoints."""

import pickle
import types
from pathlib import Path

import torch
from ultralytics.utils import DEFAULT_CFG_DICT

from .runtime import IPCModel


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        legacy_models = {
            ("experiments.nms_free_obb.native_restore_20261103.model", "NMSFreeOBBModel"),
            ("experiments.nms_free_obb.native_recovery_20261104.model", "NativeSizeFilteredOBBModel"),
            ("experiments.nms_free_obb.local_nmsfree_scratch260_20261104.adapter", "LocalNMSFreeModel"),
            ("experiments.nms_free_obb.discussion_evidence_20261105.paired_loss", "OnModel"),
        }
        if (module, name) in legacy_models:
            return IPCModel
        if module.startswith("experiments."):
            raise ValueError(f"Unsupported experimental checkpoint class: {module}.{name}")
        return super().find_class(module, name)


def load_local_checkpoint(path):
    path = Path(path).resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise FileNotFoundError(path)
    loader = types.ModuleType("fdpg_checkpoint_pickle")
    loader.Unpickler = _CheckpointUnpickler
    loader.load = lambda stream, **kwargs: _CheckpointUnpickler(stream, **kwargs).load()
    with path.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False, pickle_module=loader)
    model = checkpoint.get("ema") or checkpoint["model"]
    model = model.float().eval()
    model.args = {**DEFAULT_CFG_DICT, **checkpoint.get("train_args", {})}
    model.pt_path = str(path)
    model.task = "obb"
    for layer in model.modules():
        if hasattr(layer, "inplace"):
            layer.inplace = True
        elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, "recompute_scale_factor"):
            layer.recompute_scale_factor = None
    return model, checkpoint
