"""One-to-one rotated-box loss with instance-protected competition."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from ultralytics.utils.metrics import batch_probiou, probiou
from ultralytics.utils.tal import dist2rbox, make_anchors


class HungarianOBBLoss(nn.Module):
    def __init__(self, box_weight=5.0, center_weight=2.0, cls_weight=1.0,
                 gamma=2.0, alpha=0.25):
        super().__init__()
        self.box_weight = box_weight
        self.center_weight = center_weight
        self.cls_weight = cls_weight
        self.gamma, self.alpha = gamma, alpha
        self.last_matches = []

    @torch.no_grad()
    def match(self, boxes, logits, target, image_size):
        gt = target["boxes"].to(device=boxes.device, dtype=torch.float32)
        labels = target["labels"].to(device=boxes.device, dtype=torch.long)
        if not gt.numel() or not boxes.numel():
            empty = torch.empty(0, device=boxes.device, dtype=torch.long)
            return empty, empty
        height, width = image_size
        scale = boxes.new_tensor((width, height))
        probability = logits.float().sigmoid().clamp(1e-6, 1 - 1e-6)
        positive = self.alpha * (1 - probability).pow(self.gamma) * (-probability.log())
        negative = (1 - self.alpha) * probability.pow(self.gamma) * (-(1 - probability).log())
        classification = (positive - negative)[:, labels]
        quality = batch_probiou(boxes.float(), gt).clamp(0, 1)
        center = torch.cdist(boxes[:, :2].float() / scale, gt[:, :2] / scale, p=1)
        cost = self.cls_weight * classification + self.box_weight * (1 - quality) + self.center_weight * center
        if not torch.isfinite(cost).all():
            raise FloatingPointError("Nonfinite Hungarian cost")
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        return (torch.as_tensor(rows, dtype=torch.long, device=boxes.device),
                torch.as_tensor(cols, dtype=torch.long, device=boxes.device))

    def forward(self, outputs, targets, image_size):
        boxes, logits = outputs["boxes"].float(), outputs["logits"].float()
        if len(targets) != boxes.shape[0]:
            raise ValueError("One target dictionary is required per image")
        if not torch.isfinite(boxes).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite model outputs")
        height, width = image_size
        scale = boxes.new_tensor((width, height))
        labels_one_hot = torch.zeros_like(logits)
        quality_targets = torch.zeros_like(logits)
        box_loss = boxes.sum() * 0.0
        center_loss = boxes.sum() * 0.0
        matched_quality = boxes.new_zeros(())
        matched_count, target_count = 0, 0
        self.last_matches = []
        for batch_idx, target in enumerate(targets):
            target_count += len(target["boxes"])
            rows, cols = self.match(boxes[batch_idx], logits[batch_idx], target, image_size)
            self.last_matches.append((rows, cols))
            if len(rows) == 0:
                continue
            gt = target["boxes"].to(device=boxes.device, dtype=torch.float32)[cols]
            labels = target["labels"].to(device=boxes.device, dtype=torch.long)[cols]
            quality = probiou(boxes[batch_idx, rows], gt).reshape(-1).clamp(0, 1)
            labels_one_hot[batch_idx, rows, labels] = 1.0
            quality_targets[batch_idx, rows, labels] = quality.detach().pow(self.gamma)
            box_loss = box_loss + (1 - quality).sum()
            center_loss = center_loss + F.l1_loss(
                boxes[batch_idx, rows, :2] / scale, gt[:, :2] / scale, reduction="sum"
            )
            matched_quality = matched_quality + quality.detach().sum()
            matched_count += len(rows)
        normalizer = max(target_count, 1)
        probability = logits.sigmoid()
        weight = labels_one_hot + (1 - labels_one_hot) * probability.detach().pow(self.gamma)
        classification_loss = (F.binary_cross_entropy_with_logits(
            logits, quality_targets, reduction="none"
        ) * weight).sum()
        classification_loss = classification_loss / normalizer
        box_loss, center_loss = box_loss / normalizer, center_loss / normalizer
        total = self.cls_weight * classification_loss + self.box_weight * box_loss + self.center_weight * center_loss
        metrics = {
            "loss": total.detach(), "cls": classification_loss.detach(), "box": box_loss.detach(),
            "center": center_loss.detach(), "matched_quality": matched_quality / max(matched_count, 1),
            "matched": boxes.new_tensor(matched_count), "targets": boxes.new_tensor(target_count),
        }
        return total, metrics


def instance_competition_loss(outputs, targets, matches, quality_threshold=0.5,
                              temperature=1.0, max_negatives=32):
    """Compete against same-instance unmatched boxes, protecting all representatives."""
    if temperature <= 0 or max_negatives < 1:
        raise ValueError("temperature and max_negatives must be positive")
    logits, boxes = outputs["logits"], outputs["boxes"]
    total = logits.sum() * 0.0
    group_count = negative_count = 0
    for batch_idx, (target, (rows, cols)) in enumerate(zip(targets, matches)):
        gt = target["boxes"].to(device=boxes.device, dtype=torch.float32)
        labels = target["labels"].to(device=boxes.device, dtype=torch.long)
        if len(gt) == 0 or len(rows) == 0:
            continue
        with torch.no_grad():
            quality = batch_probiou(boxes[batch_idx].detach().float(), gt)
            best_quality, owner = quality.max(dim=1)
            available = best_quality >= quality_threshold
            available[rows] = False
        for row, col in zip(rows, cols):
            if quality[row, col] < quality_threshold:
                continue
            candidates = torch.where(available & (owner == col))[0]
            if len(candidates) == 0:
                continue
            label = labels[col]
            if len(candidates) > max_negatives:
                candidates = candidates[
                    logits[batch_idx, candidates, label].detach().topk(max_negatives).indices
                ]
            representative = logits[batch_idx, row, label].reshape(1)
            scores = torch.cat((representative, logits[batch_idx, candidates, label])) / temperature
            total = total + torch.logsumexp(scores, dim=0) - scores[0]
            group_count += 1
            negative_count += len(candidates)
    loss = total / max(group_count, 1)
    return loss, {
        "competition": loss.detach(), "competition_groups": logits.new_tensor(group_count),
        "competition_negatives": logits.new_tensor(negative_count),
    }


class IPCObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = HungarianOBBLoss(gamma=2.0)
        self.last_matches = []

    def forward(self, outputs, targets, image_size=(544, 544)):
        total, metrics = self.main(outputs, targets, image_size)
        self.last_matches = self.main.last_matches
        competition, competition_metrics = instance_competition_loss(outputs, targets, self.last_matches)
        total = total + 0.2 * competition
        # Preserve the original objective's operation order while replacing
        # the representative's q**2 target by q. Unmatched targets remain zero.
        extra = outputs["logits"].sum() * 0.0
        for batch_idx, (target, (rows, cols)) in enumerate(zip(targets, self.last_matches)):
            if not len(rows):
                continue
            gt = target["boxes"].to(outputs["boxes"].device).float()[cols]
            labels = target["labels"].to(outputs["logits"].device).long()[cols]
            selected = outputs["logits"][batch_idx, rows, labels].float()
            with torch.no_grad():
                quality = probiou(outputs["boxes"][batch_idx, rows].detach().float(), gt).reshape(-1).clamp(0, 1)
            extra = extra + (
                F.binary_cross_entropy_with_logits(selected, quality, reduction="sum")
                - F.binary_cross_entropy_with_logits(selected, quality.square(), reduction="sum")
            )
        extra = extra / max(sum(len(target["boxes"]) for target in targets), 1)
        total = total + self.main.cls_weight * extra
        metrics.update(competition_metrics)
        metrics.update(positive_target_delta=extra.detach(), detection_loss=total.detach(),
                       dfl_weight=total.new_tensor(0.0), loss=total.detach())
        return total, metrics


def decode_native_raw(raw, head):
    maps, angles = raw
    if not isinstance(maps, list) or len(maps) != len(head.stride):
        raise ValueError("Expected original OBB raw maps and angles")
    batch = maps[0].shape[0]
    joined = torch.cat([value.float().reshape(batch, head.no, -1) for value in maps], dim=2)
    distribution, scores = joined.split((4 * head.reg_max, head.nc), dim=1)
    distribution = distribution.transpose(1, 2)
    scores = scores.transpose(1, 2)
    angles = angles.float().transpose(1, 2)
    anchors, strides = make_anchors(maps, head.stride, 0.5)
    bins = torch.arange(head.reg_max, device=distribution.device, dtype=torch.float32)
    distances = distribution.reshape(batch, -1, 4, head.reg_max).softmax(-1).matmul(bins)
    geometry = dist2rbox(distances, angles, anchors.float()[None]) * strides.float()[None]
    boxes = torch.cat((geometry, angles), dim=-1)
    return {"boxes": boxes, "logits": scores, "distribution_logits": distribution,
            "anchor_points": anchors.float(), "stride_tensor": strides.float(), "reg_max": head.reg_max}


def filter_network_targets(targets, device):
    result = []
    for target in targets:
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        labels = target["labels"].to(device=device, dtype=torch.long)
        if boxes.shape != (len(labels), 5) or not bool(torch.isfinite(boxes).all()):
            raise ValueError("Expected finite network-pixel xywhr and one label per box")
        if bool((labels != 0).any()) or bool((boxes[:, 2:4] <= 0).any()):
            raise ValueError("Single-class positive-size ground truth required")
        keep = (boxes[:, 2:4] >= 2).all(-1)
        result.append({"boxes": boxes[keep], "labels": labels[keep]})
    return result


class IPCLossAdapter:
    """Adapt native OBB training batches to the single-head IPC objective."""

    def __init__(self, model):
        head = model.model[-1]
        if head.nc != 1 or head.reg_max != 16:
            raise ValueError("FDPG-IPC requires the single-class DFL16 OBB head")
        self.head = SimpleNamespace(no=head.no, nc=head.nc, reg_max=head.reg_max,
                                    stride=head.stride.detach().clone())
        self.objective = IPCObjective()

    def prepare(self, predictions, batch):
        boxes = batch["bboxes"]
        if boxes.ndim != 2 or boxes.shape[1] != 5 or not torch.isfinite(boxes).all():
            raise ValueError("Expected finite native xywhr targets")
        if (batch["cls"] != 0).any():
            raise ValueError("Expected single-class targets")
        height, width = batch["img"].shape[-2:]
        # Keep the native OBB prefilter and the final pixel-size check.
        keep = (boxes[:, 2] * height >= 2) & (boxes[:, 3] * width >= 2)
        filtered = dict(batch)
        for key in ("bboxes", "cls", "batch_idx"):
            filtered[key] = batch[key][keep.to(batch[key].device)]
        self.filtered_small_count = getattr(self, "filtered_small_count", 0) + int((~keep).sum())
        raw = predictions if isinstance(predictions[0], list) else predictions[1]
        outputs = decode_native_raw(raw, self.head)
        boxes = filtered["bboxes"].to(outputs["boxes"].device).float()
        scale = boxes.new_tensor([width, height, width, height, 1.0])
        batch_idx = filtered["batch_idx"].to(boxes.device).reshape(-1)
        labels = filtered["cls"].to(boxes.device).long().reshape(-1)
        targets = [
            {"boxes": boxes[batch_idx == index] * scale, "labels": labels[batch_idx == index]}
            for index in range(batch["img"].shape[0])
        ]
        targets = filter_network_targets(targets, boxes.device)
        return outputs, targets, (height, width)

    def __call__(self, predictions, batch):
        with torch.autocast(device_type=batch["img"].device.type, enabled=False):
            outputs, targets, size = self.prepare(predictions, batch)
            scalar, details = self.objective(outputs, targets, image_size=size)
            items = torch.stack([
                5 * details["box"], details["cls"] + details["positive_target_delta"],
                2 * details["center"], 0.2 * details["competition"],
            ]).detach()
            return scalar * batch["img"].shape[0], items
