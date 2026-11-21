"""Run `python -m fdpg --help` from the repository root."""

import argparse
import json
from pathlib import Path

from .api import FDPG, MODES, repository_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="FDPG OBB: switch trained conventional and IPC NMS-free models")
    parser.add_argument("task", choices=("info", "predict", "val", "train"))
    parser.add_argument("--mode", choices=MODES, help="override default_mode in fdpg/modes.yaml")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--weights", type=Path, help="local checkpoint; must match the selected mode")
    parser.add_argument("--source", help="prediction image or directory")
    parser.add_argument("--data", help="dataset YAML for validation/training")
    parser.add_argument("--device", default="0", help="GPU index, or cpu")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float, help="predict default 0.5; val default 0.01")
    parser.add_argument("--iou", type=float, help="rotated NMS threshold (nms mode only)")
    parser.add_argument("--max-det", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path, help="unfinished run's last.pt; mode must match")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--save-txt", action="store_true", default=None)
    parser.add_argument("--save-conf", action="store_true", default=None)
    parser.add_argument("--no-save", action="store_true", help="do not save prediction images")
    parser.add_argument("--amp", action="store_true", help="enable AMP for a NEW training run")
    opts = parser.parse_args(argv)
    if opts.task == "predict" and not opts.source:
        parser.error("predict requires --source")
    if opts.task != "train" and (opts.epochs is not None or opts.resume or opts.amp):
        parser.error("--epochs, --resume and --amp are training options")
    if opts.task == "train" and opts.weights:
        parser.error("train starts from random initialization; use --resume for a resumable last.pt")
    if opts.task != "predict" and (opts.source or opts.no_save or opts.save_txt or opts.save_conf):
        parser.error("--source, --no-save, --save-txt and --save-conf are prediction options")
    model = FDPG(opts.mode, weights=opts.weights, config=opts.config)
    if opts.iou is not None and model.mode == "nms-free":
        parser.error("NMS-free mode has no NMS IoU threshold; omit --iou")
    description = model.describe()
    if opts.task == "train":
        description.pop("weights")
        description.pop("weights_exist")
        description.update(initialization=str(opts.resume) if opts.resume else "random",
                           architecture=str(repository_path(model.config["model"])))
    print(json.dumps(description, ensure_ascii=True, indent=2))
    if opts.task == "info":
        return
    args = {key: getattr(opts, key) for key in
            ("device", "imgsz", "conf", "iou", "max_det", "batch", "workers", "epochs", "project", "name", "data",
             "save_txt", "save_conf") if getattr(opts, key) is not None}
    if opts.task == "predict":
        model.predict(opts.source, save=not opts.no_save, **args)
    elif opts.task == "val":
        metrics = model.val(**args)
        print(json.dumps(metrics.results_dict, ensure_ascii=True, indent=2))
    else:
        if opts.amp:
            args["amp"] = True
        model.train(resume=opts.resume, **args)


if __name__ == "__main__":
    main()
