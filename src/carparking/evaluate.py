"""Slot-level evaluation against ground-truth bays (PKLot XML), for any slot source.

pred: [{"poly": 4+ corners, "state": 0 FREE | 1 BUSY}]   (occupancy() / YoloDetector output)
gt:   [{"poly": corners, "state": 0 | 1}]                 (gt_from_pklot)
Matching: one-to-one Hungarian assignment on IoU (maximise total IoU); pairs with IoU < iou_thr are discarded.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from carparking.dynamic_iou import iou
from carparking.pklot import parse_spaces


def gt_from_pklot(xml_path) -> list[dict] | None:
    """PKLot XML -> ground-truth bays; None if any bay is unlabelled (skip that image)."""
    spaces = parse_spaces(xml_path)
    return None if spaces is None else [{"poly": pts, "state": int(cls == 0)} for cls, pts in spaces]  # 0 = busy_slot


def match_slots(pred_polys, gt_polys, iou_thr: float = 0.5) -> list[tuple[int, int, float]]:
    """-> [(pred index, gt index, IoU)] one-to-one, each with IoU >= iou_thr."""
    if not len(pred_polys) or not len(gt_polys):
        return []
    m = np.array([[iou(p, g) for g in gt_polys] for p in pred_polys])
    rows, cols = linear_sum_assignment(m, maximize=True)
    return [(int(r), int(c), float(m[r, c])) for r, c in zip(rows, cols) if m[r, c] >= iou_thr]


def score_image(pred: list[dict], gt: list[dict], iou_thr: float = 0.5) -> dict:
    """Counts + metrics for one image. Rates are NaN when undefined (e.g. no predictions)."""
    pairs = match_slots([p["poly"] for p in pred], [g["poly"] for g in gt], iou_thr)
    correct = sum(pred[p]["state"] == gt[g]["state"] for p, g, _ in pairs)
    gt_busy, pred_busy = sum(g["state"] for g in gt), sum(p["state"] for p in pred)
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"gt_slots": len(gt), "pred_slots": len(pred), "matched": len(pairs), "occ_correct": correct,
            "precision": div(len(pairs), len(pred)), "recall": div(len(pairs), len(gt)),
            "occ_accuracy": div(correct, len(pairs)),
            "mean_match_iou": float(np.mean([v for *_, v in pairs])) if pairs else float("nan"),
            "gt_busy": gt_busy, "pred_busy": pred_busy,
            "capacity_err": abs(len(pred) - len(gt)), "busy_err": abs(pred_busy - gt_busy),
            "free_err": abs((len(pred) - pred_busy) - (len(gt) - gt_busy))}


def summarize(rows: list[dict]) -> dict:
    """Aggregate over images: micro-averaged rates (pooled counts, so big lots weigh more) + mean count errors."""
    tot = {k: sum(r[k] for r in rows) for k in ("gt_slots", "pred_slots", "matched", "occ_correct")}
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    mean = lambda k: float(np.mean([r[k] for r in rows])) if rows else float("nan")  # noqa: E731
    return {"images": len(rows), **tot,
            "precision": div(tot["matched"], tot["pred_slots"]), "recall": div(tot["matched"], tot["gt_slots"]),
            "occ_accuracy": div(tot["occ_correct"], tot["matched"]),
            "capacity_MAE": mean("capacity_err"), "busy_MAE": mean("busy_err"), "free_MAE": mean("free_err")}
