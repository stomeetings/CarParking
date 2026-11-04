"""Static sample-image test: one picture through the camera's whole detection stack, to check the system against a known scene.

The picture is treated as one camera frame. The camera's own detector finds the vehicles (and, with "Blocked by object" on, the objects), then a fresh
ParkingMonitor with the camera's settings and slots processes it the way the worker does, with the optional layers: spillage protection, the
non-vehicle obstacle filter, blocked by object. A monitor confirms a state only after its hold time and a few frames, so the same frame is replayed
(the detector runs ONCE, its answer is replayed) at one-second steps until everything has settled; the result is what the camera would show if that
picture stayed in front of it.

Nothing live is touched: a throw-away tracker and an in-memory history database, no worker, no live database, no saved slots. The notices and entries
the run produced are returned (they are NOT written to the real event log). numpy / cv2 only; the detector is built by carparking.live.build_detector
(slot_preview's model cache) and is passed in as raw detections so the pipeline is testable without a model.
"""
from __future__ import annotations

import dataclasses
import math
import time

from carparking.blocked import LABELS as BLOCK_LABELS

MAX_STEP = 30.0  # seconds between replayed frames, at most (a long hold time is crossed in a few big steps, not hundreds of frames)


class Replay:
    """A detector that answers with the detections it was given, every time (the real one ran once on the picture)."""
    conf = None
    task = "detect"

    def __init__(self, dets):
        self.dets, self.counts = dets, {}

    def set_targets(self, targets):
        pass

    def __call__(self, frame, sliced: bool = False):
        return [dict(d) for d in self.dets]


def detector_spec(run: dict, weights: str | None = None, bay: bool = False) -> dict:
    """The detector spec to run: the camera's own (weights None), else these weights (a path relative to the project root) as a whole-frame vehicle
    detector at the camera's whole-frame size (1920 px when it has none), or as the PKLot bay model (`bay`: it finds busy / free BAYS, not cars)."""
    if weights is None:
        return run["detector"]
    if bay:
        return {"kind": "bay", "weights": [weights], "bay": {"baseline": weights}, "choice": "baseline", "whole": None}
    return {"kind": "coco", "weights": [weights], "whole": run["detector"].get("whole") or 1920}


def detect_frame(run: dict, frame, root, objects: bool = False, tune: dict | None = None) -> list[dict]:
    """The raw detections of one frame from the camera's detector (pixels: poly, box, label, conf, state), the vehicle classes and, with
    `objects`, the object classes of the blocked-by-object layer too. Blocking (seconds on a CPU, the model loads the first time): run it in a thread.
    tune: {"weights": other weights than the camera's (relative path), "bay": they are the PKLot bay model, "conf", "iou"} overrides for this call; the
    cached detector is put back as it was afterwards (it is shared with the live preview)."""
    from carparking import slot_preview
    from carparking.live import build_detector

    tune = tune or {}
    m5 = run.get("m5") or {}
    own = tune.get("weights") is None
    targets = None if m5.get("classes") is None else list(m5["classes"]) + list(m5.get("extra") or [])
    with slot_preview._LOCK:
        det, sliced, _, _ = build_detector(detector_spec(run, tune.get("weights"), bool(tune.get("bay"))), frame, root, slot_preview._MODELS)
        saved = {a: getattr(det, a, None) for a in ("conf", "iou", "whole_iou", "obb_iou", "classes", "extra")}
        try:
            conf = tune.get("conf", run.get("conf"))
            if conf is not None:
                det.conf = conf
            if tune.get("iou") is not None and hasattr(det, "iou"):
                det.iou = det.whole_iou = tune["iou"]  # the model's own NMS and the duplicate merge after it
                if hasattr(det, "obb_iou"):
                    det.obb_iou = tune["iou"]  # (oriented-box weights use this one)
            if not own and not tune.get("bay") and hasattr(det, "vehicle_names"):
                targets = det.vehicle_names()  # another model's own class names (a fine-tune's classes are not the COCO ones)
            if objects and targets is not None:
                targets = targets + [label for label in BLOCK_LABELS if label not in targets]
            if targets is not None and hasattr(det, "set_targets") and not tune.get("bay"):
                det.set_targets(targets)
            return [dict(d) for d in det(frame, sliced=sliced)]
        finally:
            for a, v in saved.items():
                if v is not None and hasattr(det, a):
                    setattr(det, a, v)


def schedule(m5: dict) -> tuple[int, float]:
    """(frames, seconds between them) that settle every state of a picture that never changes. A bay turns BUSY after the hold time, and spillage
    then needs the hold time again (it starts when its car is confirmed): two hold times, plus a few frames for the confirmations that also count
    frames. The step is a fifth of the hold time (at least a second), so a long hold is crossed in about 16 frames."""
    hold = max(float(m5.get("hold") or 0), 0.0)
    dt = min(MAX_STEP, max(1.0, hold / 5.0))
    return int(math.ceil(2 * hold / dt)) + 6, dt


def run_sample(frame, run: dict, slots: list[dict], raw: list[dict], layers: dict, tz=None, ground_cam=None, tune: dict | None = None,
               auto_surface=None, split: bool = False) -> dict:
    """The picture `frame` with the detections `raw` through a monitor built from the camera settings `run` on `slots` (fixed-slot dicts, fractions).
    layers: {"spillage", "obstacle_filter", "blocked_objects"} booleans. ground_cam: a calibrated geometry.CameraModel for spillage's ground metric.
    tune: {"footprint": the share of an object's footprint in a bay that blocks it, "encroach": the share of a bay one vehicle covers that makes it
    encroached, "conf": the confidence the detections were taken at (an object under the Blocked layer's own minimum is not lowered below it)}.
    auto_surface (fractions polygon) / split: the live auto grid's vehicle-driven rules (Worker._auto_run): a vehicle no bay claims by its foot point gets the bay its wheel
    footprint overlaps, one standing on the surface off the grid gets a bay of its own (D1, D2 ...), a second vehicle in a bay gets a sub-bay. Both off (the default): only the given slots.
    -> {"shape", "counts", "slots", "detections", "area", "events", "layers", "frames", ...}: the snapshot's fields (what the dashboard cards and the overlay draw)
    plus the notices / entries the run logged (to a throw-away database)."""
    from carparking.history import HistoryDB
    from carparking.live import _box_key, _rejected_boxes, build_monitor

    m5 = dict(run["m5"])
    m5.update(learn=False, learn_hold=0)
    if not run.get("fixed_mode"):
        m5["match"] = "foot"  # the auto grid's rule (Worker._auto_run)
    sandbox = {**run, "fixed_mode": True, "fixed": slots, "auto": {"surface": [list(map(float, p)) for p in auto_surface]} if auto_surface is not None else None,
               "split": bool(split), "m5": m5}
    db = HistoryDB(":memory:", run["camera"], tz, detector=run.get("db_detector") or "sample")
    try:
        mon = build_monitor(sandbox, Replay(raw), False, db)
        mon.start(frame.shape)
        tr = mon.tracker
        if hasattr(tr, "cover_fallback"):
            tr.cover_fallback = True  # a small far car whose foot point misses its thin bay still holds it
        want = {k: bool(layers.get(k)) for k in ("spillage", "obstacle_filter", "blocked_objects")}
        if hasattr(tr, "set_spillage"):
            tr.set_spillage(want["spillage"])
            tr.set_obstacle_filter(want["obstacle_filter"])
            tr.set_blocked(want["blocked_objects"])
            mon.set_blocked(want["blocked_objects"] and bool(getattr(tr, "blocked", 0)), tr.blocked_cfg.labels)
            tr.ground_cam = ground_cam if want["spillage"] else None
            tune = tune or {}
            if tune.get("footprint") is not None:
                f = float(tune["footprint"])
                tr.blocked_cfg = dataclasses.replace(tr.blocked_cfg, share=f, leave=round(f / 2, 3), min_conf=min(tr.blocked_cfg.min_conf, float(tune.get("conf", 1.0))))
            if tune.get("encroach") is not None:
                e = float(tune["encroach"])
                tr.spill_cfg = dataclasses.replace(tr.spill_cfg, share=e, total=max(tr.spill_cfg.total, e), leave=min(tr.spill_cfg.leave, round(e * 0.66, 3)))
        n, dt = schedule(m5)
        t0 = time.time() - n * dt
        res = None
        for k in range(n):
            res = mon.process(frame, t=t0 + k * dt)
        st = mon.state()
        h, w = frame.shape[:2]
        dropped = _rejected_boxes(tr)
        box = lambda d: [round(d["box"][0] / w, 4), round(d["box"][1] / h, 4), round(d["box"][2] / w, 4), round(d["box"][3] / h, 4)]  # noqa: E731
        dets = [{"label": d["label"], "conf": round(float(d["conf"]), 3), "src": d.get("src") or "", "kind": kind, "box": box(d),
                 **({"rejected": dropped[_box_key(d)]} if kind == "vehicle" and _box_key(d) in dropped else {})}
                for kind, group in (("vehicle", res.vehicles), ("other", res.others)) for d in group]
        out = {"shape": [w, h], "counts": st["counts"], "slots": st["slots"], "detections": dets,
               "area": {k: m5.get(k, 0.0) for k in ("top", "bottom", "left", "right")} | {"roi": m5.get("roi")},
               "mode": "fixed", "vehicles": sum(_box_key(d) not in dropped for d in res.vehicles), "others": len(res.others),
               "events": list(reversed(db.events(limit=500, kinds=None))), "layers": {**want, "ground": ground_cam is not None and want["spillage"]},
               "frames": n, "step_s": dt, "raw": len(raw), "unplaced": list(st.get("unplaced", ()))}
        if "rejected" in st:
            reasons: dict[str, int] = {}
            for r in st["rejected"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            out["obstacle"] = {"rejected": len(st["rejected"]), "reasons": reasons}
        return out
    finally:
        db.close()


def classify_drawn(slots: list[dict], dets: list[dict], w: int, h: int, min_iou: float = 0.25, min_cover: float = 0.5) -> list[dict]:
    """BUSY / FREE / UNKNOWN for hand-drawn slots from a bay model's boxes (no calibration, no tracker, no layers).
    slots: fixed-slot dicts (id, name, poly as fractions). dets: detect_frame() output of a bay model (px poly, state 1 = busy, 0 = free, conf).
    A slot takes the state of the model bay it overlaps most (IoU >= min_iou, or the bay covers >= min_cover of the slot); no such bay = UNKNOWN.
    -> [{"id", "name", "state", "conf", "poly", "hidden", "phantom", "duplicate_of"}] in the input order."""
    import cv2
    import numpy as np

    def mask(poly_px, x0, y0, mw, mh):
        m = np.zeros((mh, mw), np.uint8)
        cv2.fillPoly(m, [np.round(np.array(poly_px, np.float32) - [x0, y0]).astype(np.int32)], 1)
        return m

    bays = [(np.array(d["poly"], np.float32), d) for d in dets if d.get("state") is not None]
    out = []
    for s in slots:
        sp = np.array([[x * w, y * h] for x, y in s["poly"]], np.float32)
        best, best_iou = None, 0.0
        for bp, d in bays:
            x0, y0 = np.floor(np.minimum(sp.min(0), bp.min(0))).astype(int)
            x1, y1 = np.ceil(np.maximum(sp.max(0), bp.max(0))).astype(int)
            mw, mh = max(int(x1 - x0), 1) + 1, max(int(y1 - y0), 1) + 1
            a, b = mask(sp, x0, y0, mw, mh), mask(bp, x0, y0, mw, mh)
            inter = float((a & b).sum())
            if inter <= 0:
                continue
            iou = inter / float(a.sum() + b.sum() - inter)
            cover = inter / max(float(a.sum()), 1.0)
            if (iou >= min_iou or cover >= min_cover) and iou > best_iou:
                best, best_iou = d, iou
        state = "UNKNOWN" if best is None else ("BUSY" if best["state"] == 1 else "FREE")
        out.append({"id": s["id"], "name": s.get("name") or f"S{s['id']}", "state": state, "hidden": False, "phantom": False, "duplicate_of": None,
                    "conf": round(float(best["conf"]), 3) if best is not None else 0.0, "poly": s["poly"]})
    return out


def classify_crops(frame, slots: list[dict], detect, pad: float = 0.3, size: int = 640) -> dict[int, tuple[str, float]]:
    """A bay model's verdict on each hand-drawn slot looked at ALONE: the slot's rectangle (plus `pad` of context) is cropped from the picture, enlarged to
    `size` px and handed to `detect(crop) -> detections` (a bay model: px poly, state 1 busy / 0 free, conf). The most confident bay whose centre lies in
    the slot decides. -> {slot id: (state, conf)} for the slots the model could decide; the others are left out (the caller falls back)."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    out: dict[int, tuple[str, float]] = {}
    for s in slots:
        sp = np.array([[x * w, y * h] for x, y in s["poly"]], np.float32)
        (x0, y0), (x1, y1) = sp.min(0), sp.max(0)
        px, py = pad * (x1 - x0), pad * (y1 - y0)
        a, b, c, d = int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(w, x1 + px)) + 1, int(min(h, y1 + py)) + 1
        crop = frame[b:d, a:c]
        if crop.size == 0:
            continue
        k = size / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        inside = ((sp - [a, b]) * k).astype(np.float32).reshape(-1, 1, 2)
        best = None
        for det in detect(crop):
            if det.get("state") is None:
                continue
            cx, cy = np.array(det["poly"], np.float32).mean(0)
            if cv2.pointPolygonTest(inside, (float(cx), float(cy)), False) >= 0 and (best is None or det["conf"] > best["conf"]):
                best = det
        if best is not None:
            out[s["id"]] = ("BUSY" if best["state"] == 1 else "FREE", round(float(best["conf"]), 3))
    return out


def classify_by_vehicles(slots: list[dict], dets: list[dict], w: int, h: int, min_share: float = 0.25, ground: float = 0.4) -> dict[int, tuple[str, float]]:
    """BUSY / FREE for hand-drawn slots from a VEHICLE detector's boxes (px box, conf). A vehicle stands on the ground, and on an angled view its box is mostly
    body above the ground it stands on, so only the lower `ground` share of its box (the wheel footprint) is compared with the slots, and each vehicle belongs
    to ONE slot, the one holding the most of it: a car in slot 10 does not make slot 11 busy because its tall box leans over it. A vehicle goes to a slot when
    >= min_share of its footprint is in the slot, or when it covers >= min_share of the slot. A slot no vehicle is given is FREE.
    -> {slot id: (state, conf of the strongest vehicle given to it, else 0.0)}."""
    import cv2
    import numpy as np

    polys = {s["id"]: np.array([[x * w, y * h] for x, y in s["poly"]], np.float32) for s in slots}
    out: dict[int, tuple[str, float]] = {s["id"]: ("FREE", 0.0) for s in slots}

    def mask(poly, x0, y0, mw, mh):
        m = np.zeros((mh, mw), np.uint8)
        cv2.fillPoly(m, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
        return m

    for d in dets:
        bx0, by0, bx1, by1 = (float(v) for v in d["box"])
        foot = np.array([[bx0, by1 - ground * (by1 - by0)], [bx1, by1 - ground * (by1 - by0)], [bx1, by1], [bx0, by1]], np.float32)
        best, best_score = None, 0.0
        for sid, sp in polys.items():
            x0, y0 = np.floor(np.minimum(sp.min(0), foot.min(0))).astype(int)
            x1, y1 = np.ceil(np.maximum(sp.max(0), foot.max(0))).astype(int)
            mw, mh = max(int(x1 - x0), 1) + 1, max(int(y1 - y0), 1) + 1
            a, f = mask(sp, x0, y0, mw, mh), mask(foot, x0, y0, mw, mh)
            inter = float((a & f).sum())
            if inter <= 0:
                continue
            score = max(inter / max(float(f.sum()), 1.0), inter / max(float(a.sum()), 1.0))
            if score >= min_share and score > best_score:
                best, best_score = sid, score
        if best is not None and float(d["conf"]) > out[best][1]:
            out[best] = ("BUSY", round(float(d["conf"]), 3))
    return out
