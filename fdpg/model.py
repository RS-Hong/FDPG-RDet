"""FDPG-RDet integration with Ultralytics 8.3.221."""

import inspect
import sys
from pathlib import Path
from typing import Any


def register_fdpg_modules() -> None:
    """Register the protected FDPG layers before Ultralytics parses a model YAML or checkpoint."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.tasks as tasks
    import fdpg_runtime.fdconv as fdconv
    from fdpg_runtime.layers import FDCConv, FDCResidual, ShallowPositionInject

    # Preserve the original import paths stored in Ultralytics checkpoints.
    sys.modules["ultralytics.nn.modules.fdconv"] = fdconv

    for namespace in (modules, block, tasks):
        setattr(namespace, "FDCConv", FDCConv)
        setattr(namespace, "FDCResidual", FDCResidual)
        setattr(namespace, "ShallowPositionInject", ShallowPositionInject)

    if getattr(tasks.parse_model, "_fdpg_patched", False):
        return

    source = inspect.getsource(tasks.parse_model)
    base_anchor = "            C2f,\n"
    epgi_anchor = "        elif m is torch.nn.BatchNorm2d:\n            args = [ch[f]]\n"
    if base_anchor not in source or epgi_anchor not in source:
        raise RuntimeError("Unsupported Ultralytics parser. Install ultralytics==8.3.221.")

    source = source.replace(base_anchor, base_anchor + "            FDCResidual,\n", 1)
    source = source.replace(
        epgi_anchor,
        epgi_anchor
        + "        elif m is ShallowPositionInject:\n"
        + "            c2 = ch[f[0]]\n"
        + "            args = [ch[f[0]], ch[f[1]], *args]\n",
        1,
    )
    exec(compile(source, tasks.__file__, "exec"), tasks.__dict__)
    tasks.parse_model._fdpg_patched = True


class FDPGModel:
    """Small facade that guarantees FDPG layers are registered before model loading."""

    def __new__(cls, model: str | Path = "configs/yolov8s-fdpg-rdet-obb.yaml", task: str = "obb", **kwargs: Any):
        register_fdpg_modules()
        from ultralytics import YOLO

        return YOLO(str(model), task=task, **kwargs)
