"""FROZEN BACKUP (2026-10-12) of yolo_engine.py for the axis-aligned weights (models/backup_axis_aligned/).
Do not edit: the live engine is yolo_engine.py. Same API: from carparking.legacy_engine import YoloDetector.

Dynamic detector: Ultralytics YOLO boxes -> BUSY/FREE detections, no slots.json needed.

Two kinds of weights work:
  * our fine-tuned model (models/parking_yolo.pt, classes busy_slot / free_slot): BUSY and FREE boxes
  * a stock COCO model (yolov8n.pt / yolo11n.pt): vehicles only, reported as BUSY; it cannot see empty bays
Licence: Ultralytics is AGPL-3.0 (docs/decisions.md D11).
"""
from __future__ import annotations

VEHICLES = {"car", "bus", "truck", "motorcycle"}


def state_of(label: str) -> int | None:
    """Class name -> 0 FREE / 1 BUSY / None (ignored class, e.g. COCO 'person')."""
    label = label.lower()
    if "free" in label or "empty" in label:
        return 0
    if "busy" in label or "occupied" in label or label in VEHICLES:
        return 1
    return None


class YoloDetector:
    """frame (BGR) -> [{"box": (x1, y1, x2, y2), "state": 0|1, "conf": float, "label": str}]

    Runs SAHI sliced inference: the frame is cut into overlapping 640x640 tiles, each tile goes through YOLO at
    its training size, and the boxes are merged back, so small, distant bays in a large CCTV frame stay detectable.
    Dense rows: lenient YOLO NMS (iou 0.7, class-agnostic) and a SAHI merge that keeps adjacent bays (NMS + IOS 0.8).
    Top band (top_band of the height, far from the camera): a second pass with smaller, more-overlapping tiles
    (each tile is upscaled to 640 = zoom) and a more lenient iou for steep diagonals; it replaces the main pass there.
    """

    def __init__(self, weights, conf: float = 0.25, device=None, slice_size: int = 640, overlap: float = 0.3,
                 iou: float = 0.7, merge_metric: str = "IOS", merge_threshold: float = 0.8,
                 top_band: float = 0.15, top_slice: int = 320, top_overlap: float = 0.5, top_iou: float = 0.85):
        # lazy imports: the static ResNet path and the tests don't need ultralytics / sahi
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        self._sliced = get_sliced_prediction
        self.conf, self.device = conf, device
        self.slice_size, self.overlap = slice_size, overlap
        self.sahi = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=str(weights),
                                                       confidence_threshold=conf, device=device or "cpu")
        self.model = self.sahi.model  # the underlying Ultralytics YOLO
        # SAHI calls Ultralytics without iou/agnostic_nms/max_det; Model.predict merges .overrides into every call.
        # Higher iou = fewer overlapping neighbours suppressed; agnostic = one box per bay, never busy+free together.
        self.model.overrides.update(agnostic_nms=True, max_det=1000)  # iou is set per pass in _pass
        self.iou, self.merge_metric, self.merge_threshold = iou, merge_metric, merge_threshold
        self.top_band, self.top_slice, self.top_overlap, self.top_iou = top_band, top_slice, top_overlap, top_iou
        self.names = self.model.names
        self.can_see_free = any(state_of(n) == 0 for n in self.names.values())  # False for COCO weights

    def __call__(self, frame_bgr) -> list[dict]:
        self.sahi.confidence_threshold = self.conf  # the app changes .conf after construction (sidebar slider)
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
        out = []
        for p in r.object_prediction_list:
            state = state_of(p.category.name)
            if state is None:  # e.g. COCO 'person': SAHI has no class filter, so drop it here
                continue
            out.append({"box": tuple(map(int, p.bbox.to_xyxy())), "state": state, "conf": float(p.score.value),
                        "label": p.category.name})
        return out
