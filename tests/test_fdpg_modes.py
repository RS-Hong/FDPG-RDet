"""Run with: python -m unittest discover -s tests -p test_fdpg_modes.py -v"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics_config"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch
from torch import nn

from fdpg import FDPG
from fdpg.loss import instance_competition_loss
from fdpg.runtime import IPCModel, select_without_nms
from fdpg.training import check_model_mode
from ultralytics.nn.modules.head import OBB
from ultralytics.nn.tasks import OBBModel


def model_shell(model_class, nc=1):
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.model = nn.ModuleList([OBB(nc=nc, ne=1, ch=(8, 16, 32))])
    return model


class ModeTests(unittest.TestCase):
    def test_switch_clears_cached_model_and_separates_output(self):
        model = FDPG("nms")
        normal_checkpoint = model.checkpoint
        normal_output = model._arguments("predict", {})["project"]
        model._yolo = object()
        self.assertIs(model.switch_mode("nms-free"), model)
        self.assertIsNone(model._yolo)
        self.assertNotEqual(model.checkpoint, normal_checkpoint)
        self.assertNotEqual(model._arguments("predict", {})["project"], normal_output)
        self.assertEqual(model.switch_mode("nms").checkpoint, normal_checkpoint)

    def test_custom_weights_are_remembered_per_mode(self):
        with TemporaryDirectory() as folder:
            normal = Path(folder) / "normal.pt"
            ipc = Path(folder) / "ipc.pt"
            model = FDPG("nms", weights=normal)
            model.switch_mode("nms-free", weights=ipc)
            self.assertEqual(model.checkpoint, ipc.resolve())
            model.switch_mode("nms")
            self.assertEqual(model.checkpoint, normal.resolve())

    def test_mode_rejects_unknown_values_and_route_overrides(self):
        model = FDPG("nms-free")
        with self.assertRaises(ValueError):
            model.switch_mode("automatic")
        for option in ("nms", "model", "predictor", "validator", "trainer", "cfg", "iou"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                model._arguments("predict", {option: True})

    def test_checkpoint_mode_must_match_model(self):
        normal = model_shell(OBBModel)
        ipc = model_shell(IPCModel)
        check_model_mode(normal, "nms")
        check_model_mode(ipc, "nms-free")
        for model, mode in ((normal, "nms-free"), (ipc, "nms")):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                check_model_mode(model, mode)
        ipc.fdpg_mode = "nms"
        with self.assertRaises(ValueError):
            check_model_mode(ipc, "nms-free")

    def test_unknown_architecture_and_multiclass_head_are_rejected(self):
        class OtherOBBModel(OBBModel):
            pass

        for model in (model_shell(OtherOBBModel), model_shell(OBBModel, nc=2)):
            with self.subTest(model=type(model).__name__), self.assertRaises(ValueError):
                check_model_mode(model, "nms")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        # Identical geometry must not cause suppression in the IPC route.
        self.raw = torch.tensor([
            [[30., 30., 30., 30.], [40., 40., 40., 40.],
             [20., 20., 20., 20.], [10., 10., 10., 10.],
             [.8, .9, .5, .1], [.2, .2, .2, .2]]
        ])

    def test_overlapping_boxes_are_retained_and_sorted(self):
        rows = select_without_nms(self.raw, conf=.5, max_det=1000)[0]
        self.assertEqual(rows.shape, (2, 7))
        torch.testing.assert_close(rows[:, 4], torch.tensor([.9, .8]))
        torch.testing.assert_close(rows[:, 5], torch.zeros(2))

    def test_top_k_threshold_and_class_filter(self):
        rows = select_without_nms((self.raw, None), conf=.5, max_det=1)[0]
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0, 4]), .9, places=6)
        self.assertEqual(len(select_without_nms(self.raw, .5, 100, classes=[1])[0]), 0)
        self.assertEqual(len(select_without_nms(self.raw, .95, 100)[0]), 0)

    def test_empty_predictions_and_invalid_shape(self):
        self.assertEqual(select_without_nms(torch.empty(1, 6, 0), .5, 100)[0].shape, (0, 7))
        with self.assertRaises(ValueError):
            select_without_nms(torch.empty(1, 5, 3), .5, 100)


class CompetitionTests(unittest.TestCase):
    @staticmethod
    def evaluate(boxes, gt, matches):
        logits = torch.zeros(1, len(boxes), 1, requires_grad=True)
        outputs = {"boxes": torch.tensor([boxes], dtype=torch.float32), "logits": logits}
        targets = [{"boxes": torch.tensor(gt, dtype=torch.float32), "labels": torch.zeros(len(gt), dtype=torch.long)}]
        loss, metrics = instance_competition_loss(outputs, targets, [matches])
        loss.backward()
        return loss, metrics, logits.grad[0, :, 0]

    def test_neighbor_representative_is_protected_even_if_it_overlaps_another_gt(self):
        a, b = [30., 30., 20., 10., 0.], [34., 30., 20., 10., 0.]
        _, metrics, grad = self.evaluate(
            [a, a, a], [a, b], (torch.tensor([0, 1]), torch.tensor([0, 1]))
        )
        self.assertEqual(int(metrics["competition_groups"]), 1)
        self.assertEqual(int(metrics["competition_negatives"]), 1)
        self.assertLess(float(grad[0]), 0.)
        self.assertEqual(float(grad[1]), 0.)
        self.assertGreater(float(grad[2]), 0.)

    def test_unmatched_candidate_belongs_to_its_best_overlap_gt(self):
        a, b = [30., 30., 20., 10., 0.], [60., 30., 20., 10., 0.]
        _, metrics, grad = self.evaluate(
            [a, b, b], [a, b], (torch.tensor([0, 1]), torch.tensor([0, 1]))
        )
        self.assertEqual(int(metrics["competition_groups"]), 1)
        self.assertEqual(float(grad[0]), 0.)
        self.assertLess(float(grad[1]), 0.)
        self.assertGreater(float(grad[2]), 0.)

    def test_low_quality_representative_does_not_activate_competition(self):
        gt = [30., 30., 20., 10., 0.]
        far = [200., 200., 20., 10., 0.]
        loss, metrics, grad = self.evaluate(
            [far, gt], [gt], (torch.tensor([0]), torch.tensor([0]))
        )
        self.assertEqual(float(loss), 0.)
        self.assertEqual(int(metrics["competition_groups"]), 0)
        torch.testing.assert_close(grad, torch.zeros(2))

    def test_low_overlap_candidate_is_not_in_competition_group(self):
        gt = [30., 30., 20., 10., 0.]
        far = [200., 200., 20., 10., 0.]
        loss, metrics, grad = self.evaluate(
            [gt, far], [gt], (torch.tensor([0]), torch.tensor([0]))
        )
        self.assertEqual(float(loss), 0.)
        self.assertEqual(int(metrics["competition_negatives"]), 0)
        torch.testing.assert_close(grad, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
