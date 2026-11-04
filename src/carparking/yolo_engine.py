"""Dynamic detector: Ultralytics YOLO boxes -> BUSY/FREE detections, no slots.json needed.

Weights that work (the task is read from the weights file):
  * our fine-tuned model (models/parking_yolo.pt, classes busy_slot / free_slot): BUSY and FREE boxes
  * our OBB model (models/parking_yolo_obb_holdout-PUCPR.pt, same classes): rotated boxes, tight on diagonal bays
  * a stock COCO model (yolov8n.pt / yolo11n.pt): vehicles only, reported as BUSY; it cannot see empty bays
Licence: Ultralytics is AGPL-3.0 (docs/decisions.md D11).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from carparking.dynamic_iou import intersection_area

__all__ = ["VEHICLES", "HybridDetector", "YoloDetector", "ios", "merge_dets", "rotated_nms", "state_of",
           "tile_starts"]

# COCO names plus common names from custom labelling (camera fine-tunes); any other class is ignored
VEHICLES = {"car", "bus", "truck", "motorcycle", "vehicle", "pickup", "van", "suv", "motorbike", "bike", "scooter"}


def state_of(label: str) -> int | None:
    """Class name -> 0 FREE / 1 BUSY / None (ignored class, e.g. COCO 'person')."""
    label = label.lower()
    if "free" in label or "empty" in label:
        return 0
    if "busy" in label or "occupied" in label or label in VEHICLES:
        return 1
    return None


def _det(poly, label: str, conf: float, keep=()) -> dict | None:
    """4 corners -> detection dict; "box" is the axis-aligned hull (scene_filter, tables), "poly" what to draw.
    A class that is neither a vehicle nor a bay (e.g. COCO 'person') is dropped, unless it is in keep: then it comes
    back with state None (shown, never a parking slot)."""
    state = state_of(label)
    if state is None and label not in keep:
        return None
    poly = [tuple(map(int, p)) for p in poly]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return {"box": (min(xs), min(ys), max(xs), max(ys)), "poly": poly, "state": state, "conf": float(conf),
            "label": label}


def ios(a, b) -> float:
    """Intersection over the smaller of two convex polygons (rotated boxes)."""
    a, b = np.float32(a), np.float32(b)
    small = min(cv2.contourArea(a), cv2.contourArea(b))
    return intersection_area(a, b) / small if small > 0 else 0.0


def rotated_nms(dets: list[dict], threshold: float) -> list[dict]:
    """Greedy class-agnostic NMS on "poly" with IOS: a box cut at a tile edge lies inside the whole box (IOS ~1) and
    is dropped; rotated neighbours barely overlap and are kept.
    ponytail: O(n^2) in Python, fine for a few hundred boxes; vectorise if frames carry thousands."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(ios(d["poly"], k["poly"]) <= threshold for k in keep):
            keep.append(d)
    return keep


def tile_starts(n: int, size: int, overlap: float) -> list[int]:
    """Start offsets of overlapping tiles along one axis; the last tile is flush with the edge."""
    if n <= size:
        return [0]
    step = max(1, int(size * (1 - overlap)))
    return list(range(0, n - size, step)) + [n - size]


class YoloDetector:
    """frame (BGR) -> [{"box": (x1, y1, x2, y2), "poly": [4 x (x, y)], "state": 0|1, "conf": float, "label": str}]

    OBB weights (task "obb"): one whole-frame pass at obb_imgsz (1280: tiny far-away bays keep more pixels), rotated
    NMS at obb_iou. With sliced=True (dense high-angle views) the frame is also cut into slice_size tiles (overlap as
    for SAHI), each run at its native size, and everything is merged with rotated_nms (IOS obb_merge). Own tiling,
    because SAHI's merge only understands axis-aligned boxes.

    Detect weights, SAHI sliced inference: the frame is cut into overlapping 640x640 tiles, each tile goes through
    YOLO at its training size, and the boxes are merged back, so small, distant bays in a large CCTV frame stay detectable.
    Dense rows: lenient YOLO NMS (iou 0.7, class-agnostic) and a SAHI merge that keeps adjacent bays (NMS + IOS 0.8).
    Top band (top_band of the height, far from the camera): a second pass with smaller, more-overlapping tiles
    (each tile is upscaled to 640 = zoom) and a more lenient iou for steep diagonals; it replaces the main pass there.

    whole_imgsz (detect weights only): skip SAHI and run one whole-frame pass at this size. For vehicles on a close
    CCTV view: a car spans several 640 px tiles there, SAHI returns fragments (half a pickup, a wheel) that jump
    between frames, and the whole frame is also ~18x faster on CPU.

    near_imgsz (with whole_imgsz, smaller than it): a second whole-frame pass at this size for the vehicles near the
    camera. A high resolution keeps far, small cars apart, but a big car close to the lens is found weakly there
    (Front Parking 02, 2026-10-22: the maroon hatchback in front at 0.23-0.24 with a wrong box at 1920 px, 0.93 with
    the right box at 640 px). Only its boxes whose bottom edge lies in the lowest near_band of the frame are used, and
    they win a duplicate against the main pass's box of the same car; the far rows stay the main pass's (a low
    resolution merges the cars there). "src" "near" on its boxes.
    """

    def __init__(self, weights, conf: float = 0.25, device=None, slice_size: int = 640, overlap: float = 0.3,
                 iou: float = 0.7, merge_metric: str = "IOS", merge_threshold: float = 0.8,
                 top_band: float = 0.15, top_slice: int = 320, top_overlap: float = 0.5, top_iou: float = 0.85,
                 obb_imgsz: int = 1280, obb_iou: float = 0.45, obb_merge: float = 0.6, whole_imgsz: int | None = None,
                 near_imgsz: int | None = None, near_band: float = 0.3):
        # lazy imports: the static ResNet path and the tests don't need ultralytics / sahi
        from ultralytics import YOLO
        self.conf, self.device = conf, device
        self.slice_size, self.overlap = slice_size, overlap
        self.model = YOLO(str(weights))
        self.task = self.model.task  # "detect" or "obb"
        self.whole_imgsz = whole_imgsz if self.task == "detect" else None
        self.near_imgsz = near_imgsz if self.whole_imgsz and near_imgsz and near_imgsz < self.whole_imgsz else None
        self.near_band = near_band
        if self.task == "detect" and not self.whole_imgsz:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            self._sliced = get_sliced_prediction
            self.sahi = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=str(weights),
                                                           confidence_threshold=conf, device=device or "cpu")
            self.model = self.sahi.model  # the underlying Ultralytics YOLO
        elif self.task not in ("detect", "obb"):
            raise ValueError(f"{weights}: task '{self.task}' is not supported (need detect or obb weights)")
        # SAHI calls Ultralytics without iou/agnostic_nms/max_det; Model.predict merges .overrides into every call.
        # Higher iou = fewer overlapping neighbours suppressed; agnostic = one box per bay, never busy+free together.
        self.model.overrides.update(agnostic_nms=True, max_det=1000)  # iou is set per pass in _pass
        self.iou, self.merge_metric, self.merge_threshold = iou, merge_metric, merge_threshold
        self.top_band, self.top_slice, self.top_overlap, self.top_iou = top_band, top_slice, top_overlap, top_iou
        self.obb_imgsz, self.obb_iou, self.obb_merge = obb_imgsz, obb_iou, obb_merge
        self.names = self.model.names
        self.can_see_free = any(state_of(n) == 0 for n in self.names.values())  # False for COCO weights
        # class ids worth keeping: COCO -> [2, 3, 5, 7] (car, motorcycle, bus, truck). Filtered inside the model, so a
        # 'potted plant' box can't win the class-agnostic NMS against the motorbike behind it
        self.classes = [i for i, n in self.names.items() if state_of(n) is not None]
        self.extra: set[str] = set()  # non-vehicle classes to return with state None (set_targets)
        self.weights = Path(weights)
        self.whole_iou, self.whole_ios = 0.45, 0.8  # duplicate merge on the whole-frame path (merge_dets)

    def vehicle_names(self) -> list[str]:
        """Classes that count for parking (vehicles, or FREE/BUSY bays), in model order."""
        return [n for n in self.names.values() if state_of(n) is not None]

    def other_names(self) -> list[str]:
        """Classes that never occupy a slot (COCO person, dog, ...): can be shown via set_targets."""
        return [n for n in self.names.values() if state_of(n) is None]

    def set_targets(self, names):
        """Classes to detect (in-model filter on the whole-frame path): vehicles/bays plus any shown-only classes."""
        names = set(names)
        self.classes = [i for i, n in self.names.items() if n in names]
        self.extra = {n for n in names if state_of(n) is None}

    def describe(self, sliced: bool = False) -> str:
        """Short inference-mode label for UI badges."""
        if self.task == "obb":
            return f"{self.obb_imgsz}px" + (f" + {self.slice_size}px tiles" if sliced else " whole frame")
        if self.whole_imgsz:
            return f"{self.whole_imgsz}px whole frame" + (f" + {self.near_imgsz}px near pass" if self.near_imgsz else "")
        return f"SAHI {self.slice_size}px tiles"

    def __call__(self, frame_bgr, sliced: bool = False) -> list[dict]:
        """sliced only affects OBB weights; the detect path always uses SAHI tiles."""
        if self.task == "obb":
            return self._obb(frame_bgr, sliced)
        if self.whole_imgsz:
            dets = self._whole(frame_bgr, self.whole_imgsz)
            if self.near_imgsz:
                low = (1 - self.near_band) * frame_bgr.shape[0]
                near = [{**d, "src": "near"} for d in self._whole(frame_bgr, self.near_imgsz) if d["box"][3] >= low]
                dets = near + dets  # near first: its box of a car close to the lens beats the main pass's
            # the in-model NMS is lenient (iou 0.7, kept for dense PKLot bays) and blind to fragments: a fine-tune at
            # low conf returns several boxes on one covered bike, which would become duplicate slots
            return merge_dets(dets, self.whole_iou, self.whole_ios)
        self.sahi.confidence_threshold = self.conf  # the app changes .conf after construction (sidebar slider)
        return self._sahi(frame_bgr)

    def _whole(self, frame_bgr, imgsz: int) -> list[dict]:
        """One whole-frame pass at imgsz -> detections, best first."""
        r = self.model.predict(frame_bgr, imgsz=imgsz, conf=self.conf, iou=self.iou, agnostic_nms=True,
                               classes=self.classes, max_det=1000, device=self.device or "cpu", verbose=False)[0].boxes
        dets = [_det([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], self.names[int(c)], p, self.extra)  # agnostic NMS:
                # no car+truck pair for one vehicle
                for (x1, y1, x2, y2), c, p in zip(r.xyxy.tolist(), r.cls.tolist(), r.conf.tolist())]
        return sorted((d for d in dets if d), key=lambda d: -d["conf"])

    def detect_region(self, frame_bgr, box, conf: float = 0.30, imgsz: int | None = None, pad: float = 0.4,
                      border: float = 0.02, behind: bool = False) -> list[dict]:
        """A second, zoomed look at one region (box = x1, y1, x2, y2 in pixels, padded by pad of its size per side)
        at a lower confidence -> vehicle detections in frame pixels, "src" "zoom". A car mostly hidden behind the
        car in front (Front Parking 02, 2026-10-20: only the roof and tail lights of the car behind the white sedan
        show) gets no box from the whole frame even at conf 0.05; on a crop around its slot at the crop's own
        resolution it does (0.60; imgsz None = the crop's size, 640-1280: shrunk to 640 it is found on some crops
        only).
        Boxes cut by the crop edge (within border of it, where that edge is not the frame's) are dropped: they are
        pieces of the cars around the region.
        behind=True: the crop ends at the region's bottom (its ground line, + border) instead of below it, and boxes
        reaching that bottom edge are kept. For a car whose lower part is hidden by the car in front: with the front
        car in the crop the two white cars are one box (Front Parking 02, 2026-10-22: the white hatchback behind the
        sedan, never a box of its own in the whole frame even at conf 0.05 / NMS 0.9); cut at its ground line, it
        is found alone (0.82). Its box then ends where the front car hides it, the crop's bottom edge."""
        if self.task != "detect":
            return []
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        px, py = pad * (x2 - x1), pad * (y2 - y1)
        bottom = y2 + (border * (y2 - y1) if behind else py)
        cx1, cy1, cx2, cy2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, bottom))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return []
        imgsz = imgsz or min(1280, max(640, -(-max(cx2 - cx1, cy2 - cy1) // 32) * 32))
        r = self.model.predict(frame_bgr[cy1:cy2, cx1:cx2], imgsz=imgsz, conf=conf, iou=self.iou, agnostic_nms=True,
                               classes=self.classes, max_det=100, device=self.device or "cpu", verbose=False)[0].boxes
        bx, by = border * (cx2 - cx1), border * (cy2 - cy1)
        out = []
        for (a, b, c, d), cls, p in zip(r.xyxy.tolist(), r.cls.tolist(), r.conf.tolist()):
            cut = ((a <= bx and cx1 > 0) or (b <= by and cy1 > 0) or (c >= cx2 - cx1 - bx and cx2 < w) or
                   (d >= cy2 - cy1 - by and cy2 < h and not behind))
            det = None if cut else _det([(a + cx1, b + cy1), (c + cx1, b + cy1), (c + cx1, d + cy1), (a + cx1, d + cy1)],
                                        self.names[int(cls)], p)
            if det:
                out.append({**det, "src": "zoom"})
        return out

    def _sahi(self, frame_bgr) -> list[dict]:
        dets = self._pass(frame_bgr, self.slice_size, self.overlap, self.iou, self.merge_threshold)
        band_h = int(frame_bgr.shape[0] * self.top_band)
        if band_h == 0:  # top_band=0 switches the second pass off
            return dets
        # crop 2x the band so cars straddling its edge are seen whole; each pass owns the boxes whose centre is on its
        # side of band_h. ponytail: a car whose two boxes land on opposite sides is counted twice; cross-pass NMS if seen
        top = self._pass(frame_bgr[:2 * band_h], self.top_slice, self.top_overlap, self.top_iou,
                         max(self.merge_threshold, self.top_iou))  # SAHI merge at 0.8 IOS would re-delete the diagonals
        in_band = lambda d: (d["box"][1] + d["box"][3]) / 2 < band_h  # noqa: E731
        return [d for d in dets if not in_band(d)] + [d for d in top if in_band(d)]

    def _pass(self, img_bgr, slice_size, overlap, iou, merge_threshold) -> list[dict]:
        # ponytail: mutates the shared model; fine for the local single-user app, one detector per thread if served
        self.model.overrides["iou"] = iou
        # SAHI expects RGB numpy and flips to BGR itself before calling Ultralytics
        r = self._sliced(img_bgr[:, :, ::-1], self.sahi, slice_height=slice_size, slice_width=slice_size,
                         overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0,
                         # NMS keeps the best box (GREEDYNMM unions boxes into fat 2-bay boxes). IOS removes tile-seam
                         # duplicates (a truncated box inside the full one has IOS ~1) yet keeps adjacent bays.
                         postprocess_type="NMS", postprocess_match_metric=self.merge_metric,
                         postprocess_match_threshold=merge_threshold, postprocess_class_agnostic=True)
        dets = []
        for p in r.object_prediction_list:  # SAHI has no class filter, so _det drops unknown classes
            x1, y1, x2, y2 = p.bbox.to_xyxy()
            dets.append(_det([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], p.category.name, p.score.value))
        return [d for d in dets if d]

    def _obb(self, frame_bgr, sliced: bool = False) -> list[dict]:
        h, w = frame_bgr.shape[:2]
        jobs = [(0, 0, frame_bgr, self.obb_imgsz)]  # (x offset, y offset, image, imgsz)
        if sliced:
            s = self.slice_size
            jobs += [(x, y, frame_bgr[y:y + s, x:x + s], s)
                     for y in tile_starts(h, s, self.overlap) for x in tile_starts(w, s, self.overlap)]
        dets = []
        for x, y, img, imgsz in jobs:
            # Ultralytics treats numpy frames as BGR (OpenCV order), so no conversion here
            r = self.model.predict(img, imgsz=imgsz, conf=self.conf, iou=self.obb_iou, agnostic_nms=True,
                                   max_det=1000, device=self.device or "cpu", verbose=False)[0].obb
            dets += [_det([(px + x, py + y) for px, py in poly], self.names[int(c)], p)
                     for poly, c, p in zip(r.xyxyxyxy.tolist(), r.cls.tolist(), r.conf.tolist())]
        dets = [d for d in dets if d]
        return rotated_nms(dets, self.obb_merge) if sliced else dets


def _iou_ios(a, b) -> tuple[float, float]:
    """IoU and intersection over the smaller area of two convex polygons (axis-aligned or rotated boxes)."""
    a, b = np.float32(a), np.float32(b)
    inter = intersection_area(a, b)  # exact: cv2.intersectConvexConvex fails on nearly shared edges
    area_a, area_b = cv2.contourArea(a), cv2.contourArea(b)
    union, small = area_a + area_b - inter, min(area_a, area_b)
    return (inter / union if union > 0 else 0.0), (inter / small if small > 0 else 0.0)


def _same_depth(a, b, tol: float = 0.15) -> bool:
    """Bottom edges level (within tol of the taller box's height): a fragment of the same vehicle. In a low-angle view a
    car parked further back stands higher in the frame, its box inside the box of the car in front."""
    ya, yb = np.float32(a)[:, 1], np.float32(b)[:, 1]
    return abs(ya.max() - yb.max()) <= tol * max(ya.max() - ya.min(), yb.max() - yb.min())


def _same_column(a, b, tol: float = 0.85) -> bool:
    """The narrower box spans >= tol of the wider box's width: a horizontal slice of the same vehicle (its roof and
    windows, its lower half). A car parked behind another is narrower in a low-angle view (perspective) and offset
    sideways: Front Parking 02's dark car spans 39 % of the SUV's width in front of it, test_engine's car behind 65 %."""
    xa, xb = np.float32(a)[:, 0], np.float32(b)[:, 0]
    inter = min(xa.max(), xb.max()) - max(xa.min(), xb.min())
    return inter >= tol * max(np.ptp(xa), np.ptp(xb))


def merge_dets(dets: list[dict], iou: float = 0.45, ios: float = 0.8) -> list[dict]:
    """NMS over detections in priority order (best first), within one model or across models. Vehicles merge across
    vehicle classes (COCO 'truck' + camera model 'car' = one pickup); any other class only with itself, so a person
    beside a car survives. Duplicate = IoU >= iou (any vehicle class), or >= ios of the smaller box inside the other
    and it is a fragment of the same vehicle:
    * level bottom edges, same class (a vertical slice: half a pickup, a piece of the covered bike), or
    * the same width, any vehicle class (a horizontal slice: the upper half of the car in front, its roof).
    Same class only for the first, so a real motorbike parked in front of a car, whose box lies inside the car's box,
    is kept; a car parked behind another (its box inside the front car's box, bottom higher) is kept unless it spans
    the front car's full width. iou 0.45: the most overlapping real pair on Front Parking 02 (SUV / red car behind
    the sedan) is at 0.37. The kept box lists every source that saw the object in "src" ("cam+coco")."""
    group = lambda d: "vehicle" if d["state"] is not None else d["label"]  # noqa: E731

    def duplicate(k, d):
        if group(k) != group(d):
            return False
        v_iou, v_ios = _iou_ios(k["poly"], d["poly"])
        if v_iou >= iou:
            return True
        if v_ios < ios:
            return False
        same_label = k["label"] == d["label"]
        return (same_label and _same_depth(k["poly"], d["poly"])) or \
            (group(k) == "vehicle" and _same_column(k["poly"], d["poly"]))

    keep = []
    for d in dets:
        for k in keep:
            if duplicate(k, d):
                if d.get("src") and d["src"] not in k["src"].split("+"):
                    k["src"] += "+" + d["src"]
                break
        else:
            keep.append({**d, "src": d.get("src", "")})
    return keep


class HybridDetector:
    """Several detectors on the same frame, merged into one list (merge_dets) -> same interface as YoloDetector.

    Members in priority order: the camera fine-tune first (it knows this scene: the covered bike, a pickup in shade),
    stock COCO second (people, animals, vehicle types the fine-tune never saw). On a duplicate the higher-priority
    box is kept, whatever the confidences (the two models' scores are not on one scale). Each detection carries
    "src" (e.g. "cam", "coco", "cam+coco"); .counts holds the raw per-model counts of the last frame."""

    task = "hybrid"
    can_see_free = False

    def __init__(self, members, tags, iou: float = 0.45, ios: float = 0.8):
        self.members, self.tags, self.iou, self.ios = list(members), list(tags), iou, ios
        self.counts: dict[str, int] = {}

    @property
    def conf(self) -> float:
        return self.members[0].conf

    @conf.setter
    def conf(self, value: float):
        for m in self.members:
            m.conf = value

    def vehicle_names(self) -> list[str]:
        return list(dict.fromkeys(n for m in self.members for n in m.vehicle_names()))

    def other_names(self) -> list[str]:
        return list(dict.fromkeys(n for m in self.members for n in m.other_names()))

    def set_targets(self, names):
        for m in self.members:
            m.set_targets(names)

    def describe(self, sliced: bool = False) -> str:
        return " + ".join(f"{t}: {m.weights.name} {m.describe(sliced)}" for m, t in zip(self.members, self.tags))

    def __call__(self, frame_bgr, sliced: bool = False) -> list[dict]:
        dets = []
        for rank, (m, tag) in enumerate(zip(self.members, self.tags)):
            out = m(frame_bgr, sliced=sliced)
            self.counts[tag] = len(out)
            dets += [{**d, "src": tag, "rank": rank} for d in out]
        dets.sort(key=lambda d: (d["rank"], -d["conf"]))
        return merge_dets(dets, self.iou, self.ios)

    def detect_region(self, frame_bgr, box, **kw) -> list[dict]:
        """YoloDetector.detect_region by every member, merged in priority order."""
        dets = [d for m in self.members for d in m.detect_region(frame_bgr, box, **kw)]
        return merge_dets(dets, self.iou, self.ios)
