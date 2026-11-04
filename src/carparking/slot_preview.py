"""Slot preview for the fixed slot editor ("Test slots"): which drawn slots would be FREE or BUSY right now.

Read-only and stateless: the slots as drawn in the editor (saved or not) are checked against the vehicles in the
current frame (the worker's latest detections while it monitors; else one frame grabbed on demand and run through the
camera's own detector here, detect_on_demand), with Fixed mode's own matching (fixed_slots.match_score under the
camera's match rule, one vehicle per slot, best pairs first, exactly as FixedSlotTracker._match). Nothing is stored, no tracker, worker,
database or saved map is touched. Unlike the worker it applies no hold / exit buffer: it is the instant answer for
the current frame, so a car just arriving or leaving shows its new state at once.
"""
from __future__ import annotations

import threading

import numpy as np

from carparking.fixed_slots import MATCH_RULES, match_score, validate_slots
from carparking.history import DISPLAY_TUNING

# on-demand detection (Test slots while monitoring is off): the models stay loaded in this process after the first test
# ({(weights, imgsz): YoloDetector}, live.build_detector's cache); one inference at a time
_MODELS: dict = {}
_LOCK = threading.Lock()


def detect_on_demand(run: dict, frame_bgr, root) -> list[dict]:
    """Vehicles in one frame, found the way the worker finds them (ParkingMonitor.process with the camera's settings):
    its detector (run["detector"], live.build_detector), its confidence and target classes, the parking area, Fixed
    mode's keep rule (box centre inside the area). -> detections in the snapshot's format ({"label", "conf", "src",
    "kind": "vehicle", "box" [x1, y1, x2, y2] fractions}). Blocking (seconds on a CPU; the first call loads the model):
    run it in a thread. Nothing is stored. The worker's zoomed re-check of hidden slots is not run: whole frame only."""
    from carparking.live import build_detector
    from carparking.monitor import ParkingArea, inside

    m5 = run.get("m5") or {}
    targets = None if m5.get("classes") is None else list(m5["classes"]) + list(m5.get("extra") or [])
    h, w = frame_bgr.shape[:2]
    with _LOCK:
        det, sliced, _, _ = build_detector(run["detector"], frame_bgr, root, _MODELS)
        if run.get("conf") is not None:
            det.conf = run["conf"]
        if targets is not None and hasattr(det, "set_targets"):
            det.set_targets(targets)
        found = det(frame_bgr, sliced=sliced)
    mask = ParkingArea.from_settings(m5).mask(frame_bgr.shape)
    out = []
    for d in found:
        if d["state"] is None or (targets is not None and d["label"] not in targets) or not inside(mask, d["poly"]):
            continue
        x1, y1, x2, y2 = d["box"]
        out.append({"label": d["label"], "conf": round(float(d["conf"]), 3), "src": d.get("src") or "", "kind": "vehicle",
                    "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]})
    return out


def _rect(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.float32([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def preview_slots(slots: list[dict], detections: list[dict], rule: str = "ground", min_iou: float = 0.15,
                  only: list[int] | None = None, bay_share: float = DISPLAY_TUNING["bay_share"][0]) -> list[dict]:
    """slots: [{"id", "name", "poly" fractions}] (validated here; ValueError names the slot). detections: the worker's
    snapshot detections ({"label", "conf", "kind", "box" [x1, y1, x2, y2] fractions}); only kind "vehicle" counts.
    only: test just these slot ids (the others still compete for the vehicles, so a car standing across two slots is
    given to the same one as for "test all"). -> [{"id", "name", "state" FREE / BUSY, "score", "vehicle": the
    detection that occupies it or None}] for the tested slots, in the slots' order."""
    if rule not in MATCH_RULES:
        raise ValueError(f"unknown match rule {rule!r}; use one of {list(MATCH_RULES)}")
    slots = validate_slots(slots)
    vehicles = [d for d in detections if d.get("kind", "vehicle") == "vehicle" and len(d.get("box") or ()) == 4]
    pairs = sorted(((v, i, j) for i, d in enumerate(vehicles) for j, s in enumerate(slots)
                    if (v := match_score(_rect(d["box"]), s["poly"], rule, min_iou, share=bay_share)) > 0), reverse=True)
    taken, used = {}, set()
    for v, i, j in pairs:  # one-to-one, best pairs first (FixedSlotTracker._match)
        if i not in used and j not in taken:
            taken[j] = (i, v)
            used.add(i)
    wanted = None if only is None else set(only)
    out = []
    for j, s in enumerate(slots):
        if wanted is not None and s["id"] not in wanted:
            continue
        hit = taken.get(j)
        out.append({"id": s["id"], "name": s["name"], "state": "BUSY" if hit else "FREE",
                    "score": round(hit[1], 3) if hit else 0.0, "vehicle": vehicles[hit[0]] if hit else None})
    return out
