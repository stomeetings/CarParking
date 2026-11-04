"""Class-name mapping of carparking/yolo_engine.py (COCO and camera fine-tunes)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.yolo_engine import HybridDetector, YoloDetector, _det, merge_dets, state_of  # noqa: E402


def box(label, x1, y1, x2, y2, conf=0.5, src=None):
    d = _det([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], label, conf, keep={"person", "dog"})
    return {**d, "src": src} if src else d


class FakeModel:
    """Stands in for a YoloDetector: fixed detections, records the targets it was given."""

    def __init__(self, name, names, dets):
        self.weights, self.names, self.dets, self.conf, self.targets = Path(name), names, dets, 0.25, None

    def vehicle_names(self):
        return [n for n in self.names if state_of(n) is not None]

    def other_names(self):
        return [n for n in self.names if state_of(n) is None]

    def set_targets(self, names):
        self.targets = set(names)

    def describe(self, sliced=False):
        return "1280px whole frame"

    def __call__(self, frame, sliced=False):
        return [dict(d) for d in self.dets]


def test_keep_returns_shown_only_classes_with_state_none():
    assert _det([(0, 0), (1, 0), (1, 1), (0, 1)], "person", 0.9) is None  # default: dropped as before
    assert _det([(0, 0), (1, 0), (1, 1), (0, 1)], "person", 0.9, keep={"person"})["state"] is None


def test_merge_vehicles_across_classes_and_models():
    pickup_cam = box("car", 100, 100, 400, 250, 0.6, "cam")
    pickup_coco = box("truck", 105, 98, 405, 252, 0.9, "coco")  # same pickup, other class, higher conf
    half = box("car", 250, 100, 400, 250, 0.8, "coco")  # SAHI-style fragment: IoU 0.5, fully inside
    bike = box("motorcycle", 420, 200, 520, 260, 0.4, "cam")
    out = merge_dets([pickup_cam, bike, pickup_coco, half])
    assert [(d["label"], d["src"]) for d in out] == [("car", "cam+coco"), ("motorcycle", "cam")]


def test_bike_in_front_of_a_car_survives_but_bike_fragments_merge():
    car = box("car", 100, 100, 500, 300, 0.9)
    bike = box("motorcycle", 120, 220, 220, 300, 0.8)  # 100% inside the car box, IoU 0.1: a real bike in front
    piece = box("motorcycle", 125, 250, 215, 300, 0.4)  # a fragment of the same bike
    assert [d["label"] for d in merge_dets([car, bike, piece])] == ["car", "motorcycle"]


def test_car_parked_behind_inside_the_front_cars_box_survives():
    # Front Parking 02, 2026-10-19 live frame (shares of 2560x1440): a COCO box over the dark car reaches up over the
    # white hatchback parked behind it; the hatchback's box is ~100% inside but ends higher (further back)
    front = box("car", 794, 302, 1382, 547, 0.46)
    behind = box("car", 998, 317, 1382, 475, 0.23)
    double = box("car", 947, 317, 1382, 562, 0.17)  # the same dark car again: level bottom -> merged
    assert [d["conf"] for d in merge_dets([front, behind, double])] == [0.46, 0.23]


def test_silver_car_split_into_two_boxes_is_one_car():
    # Front Parking 02, 2026-10-19 live frame (2560x1440): the silver hatchback in the foreground
    silver = box("car", 538, 622, 1268, 1224, 0.61)
    upper = box("car", 541, 620, 1266, 880, 0.44)  # its roof + windows only: bottom 344 px higher, IoU 0.43
    lower = box("truck", 545, 900, 1262, 1226, 0.31)  # its lower body, as 'truck'
    sedan = box("car", 594, 457, 1431, 890, 0.76)  # the white sedan parked behind it: box reaches over its roof
    out = merge_dets([sedan, silver, upper, lower])
    assert [d["conf"] for d in out] == [0.76, 0.61]  # one silver car, and the sedan (IoU 0.29 with it) kept
    # the merge keeps every real car of that frame (most overlapping real pair: SUV / red car, IoU 0.37)
    live = [box("car", *(round(v * s) for v, s in zip(b, (2560, 1440, 2560, 1440))), c) for b, c in [
        ((0.208, 0.399, 0.495, 0.847), 0.61), ((0.250, 0.327, 0.562, 0.587), 0.76),
        ((0.289, 0.276, 0.545, 0.383), 0.51), ((0.328, 0.212, 0.534, 0.344), 0.88),
        ((0.447, 0.212, 0.530, 0.314), 0.75), ((0.443, 0.182, 0.568, 0.261), 0.87),
        ((0.467, 0.148, 0.568, 0.201), 0.58)]]
    assert len(merge_dets(sorted(live, key=lambda d: -d["conf"]))) == 7


def test_person_beside_a_car_is_not_merged():
    car = box("car", 100, 100, 400, 250, 0.9, "cam")
    person = box("person", 300, 80, 360, 250, 0.7, "coco")  # 100% inside the car box, but another group
    out = merge_dets([car, person])
    assert [d["label"] for d in out] == ["car", "person"] and out[1]["state"] is None


def test_hybrid_detector_priority_counts_and_targets():
    cam = FakeModel("cam_front_parking_02.pt", ["car", "motorcycle"],
                    [box("car", 100, 100, 400, 250, 0.55), box("motorcycle", 420, 200, 520, 260, 0.6)])
    coco = FakeModel("yolov8n.pt", ["person", "car", "motorcycle", "bus", "truck", "dog"],
                     [box("truck", 102, 99, 401, 251, 0.95), box("person", 600, 100, 650, 250, 0.8),
                      box("bus", 900, 100, 1200, 300, 0.7)])
    hyb = HybridDetector([cam, coco], ["cam", "coco"])
    hyb.conf = 0.15
    assert cam.conf == coco.conf == 0.15
    assert hyb.vehicle_names() == ["car", "motorcycle", "bus", "truck"] and hyb.other_names() == ["person", "dog"]
    hyb.set_targets(["car", "motorcycle", "truck", "person"])
    assert cam.targets == coco.targets == {"car", "motorcycle", "truck", "person"}
    out = hyb(None)
    # the camera model's 'car' wins over COCO's more confident 'truck' for the same pickup
    assert [(d["label"], d["src"]) for d in out] == [("motorcycle", "cam"), ("car", "cam+coco"), ("person", "coco"),
                                                     ("bus", "coco")]
    assert hyb.counts == {"cam": 2, "coco": 3}
    assert "cam: cam_front_parking_02.pt" in hyb.describe() and "coco: yolov8n.pt" in hyb.describe()


def test_vehicle_names_are_busy():
    for n in ("car", "truck", "bus", "motorcycle", "Vehicle", "pickup", "van", "SUV", "motorbike", "bike", "scooter"):
        assert state_of(n) == 1, n


def test_bay_and_other_names():
    assert state_of("free_slot") == 0 and state_of("busy_slot") == 1
    assert state_of("person") is None and state_of("potted plant") is None


class _Out:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return self.v


class _CropModel:
    """model.predict stand-in: fixed boxes in crop pixels; records the crop and the call's conf."""

    def __init__(self, boxes):
        self.boxes, self.calls = boxes, []

    def predict(self, img, **kw):
        self.calls.append((img.shape[:2], kw["conf"], kw["imgsz"]))
        boxes = SimpleNamespace(xyxy=_Out([b[:4] for b in self.boxes]), cls=_Out([2] * len(self.boxes)),
                                conf=_Out([b[4] for b in self.boxes]))
        return [SimpleNamespace(boxes=boxes)]


def test_detect_region_zooms_on_a_padded_crop_and_drops_boxes_cut_by_its_edge():
    det = object.__new__(YoloDetector)
    det.task, det.iou, det.classes, det.device, det.names = "detect", 0.7, [2], None, {2: "car"}
    # slot (800, 300)-(1000, 400), padded by 40 % per side: crop x 720-1080, y 260-440 (360 x 180 px)
    det.model = _CropModel([(50, 40, 250, 150, 0.6),  # the car behind, whole inside the crop
                            (10, 100, 300, 179, 0.8)])  # the car in front, cut by the crop's bottom edge
    out = det.detect_region(np.zeros((1000, 2000, 3), np.uint8), (800, 300, 1000, 400), conf=0.25)
    assert det.model.calls == [((180, 360), 0.25, 640)]
    assert [(d["box"], d["src"], d["label"]) for d in out] == [((770, 300, 970, 410), "zoom", "car")]
    det.model = _CropModel([(0, 0, 100, 100, 0.6)])  # at the frame's own edge: not a cut
    assert len(det.detect_region(np.zeros((1000, 2000, 3), np.uint8), (0, 0, 100, 100))) == 1


def test_detect_region_behind_ends_the_crop_at_the_ground_line_and_keeps_boxes_on_it():
    det = object.__new__(YoloDetector)
    det.task, det.iou, det.classes, det.device, det.names = "detect", 0.7, [2], None, {2: "car"}
    # slot (800, 300)-(1000, 400): crop x 720-1080, y 260-402 (the bottom: the slot's + 2 % of its height)
    det.model = _CropModel([(50, 40, 250, 142, 0.9)])  # the car behind, its lower part hidden: ends on the crop's edge
    out = det.detect_region(np.zeros((1000, 2000, 3), np.uint8), (800, 300, 1000, 400), conf=0.3, behind=True)
    assert det.model.calls == [((142, 360), 0.3, 640)]
    assert [(d["box"], d["src"]) for d in out] == [((770, 300, 970, 402), "zoom")]
    det.model = _CropModel([(50, 40, 250, 179, 0.9)])  # the normal crop (y 260-440): a box on its bottom edge, dropped
    assert det.detect_region(np.zeros((1000, 2000, 3), np.uint8), (800, 300, 1000, 400), conf=0.3) == []


def test_merge_keeps_the_car_behind_the_sedan_apart_from_its_neighbours():
    # Front Parking 02, 2026-10-20 10:19 (shares of the 2560 x 1440 frame): the zoom box of the car behind the
    # white sedan lies wholly inside the sedan's box and touches the hatchback's; it is a vehicle of its own (bottom
    # 0.20 of the frame above the sedan's, 74 % of its width), so the merge keeps all three
    px = lambda b: [round(v * s) for v, s in zip(b, (2560, 1440, 2560, 1440))]  # noqa: E731
    hatch = box("car", *px((0.358, 0.215, 0.529, 0.336)), conf=0.94)
    sedan = box("car", *px((0.254, 0.279, 0.565, 0.590)), conf=0.88)
    behind = box("car", *px((0.328, 0.277, 0.559, 0.389)), conf=0.60)
    assert len(merge_dets([hatch, sedan, behind])) == 3
    roof = box("car", *px((0.262, 0.279, 0.560, 0.400)), conf=0.5)  # the sedan's upper half: a slice, merged
    assert len(merge_dets([hatch, sedan, behind, roof])) == 3


class _SizeModel:
    """model.predict stand-in: fixed boxes per imgsz (frame pixels); records the sizes it ran at."""

    def __init__(self, by_size):
        self.by_size, self.sizes = by_size, []

    def predict(self, img, **kw):
        self.sizes.append(kw["imgsz"])
        found = self.by_size.get(kw["imgsz"], [])
        boxes = SimpleNamespace(xyxy=_Out([b[:4] for b in found]), cls=_Out([2] * len(found)),
                                conf=_Out([b[4] for b in found]))
        return [SimpleNamespace(boxes=boxes)]


def test_near_pass_finds_the_big_car_in_front_and_leaves_the_far_rows_to_the_main_pass():
    # Front Parking 02, 2026-10-22 (1920 x 1080): 1920 px finds the maroon hatchback in front at 0.24 with a wrong box,
    # 640 px at 0.94 with the right one; 640 px boxes the two white cars behind it as one (bottom 0.59: not near)
    px = lambda b: [round(v * s) for v, s in zip(b, (1920, 1080, 1920, 1080))]  # noqa: E731
    main = [(*px((0.25, 0.23, 0.56, 0.60)), 0.82), (*px((0.40, 0.18, 0.51, 0.25)), 0.84),
            (*px((0.17, 0.37, 0.48, 0.83)), 0.24)]
    low = [(*px((0.09, 0.46, 0.47, 0.83)), 0.94), (*px((0.21, 0.20, 0.55, 0.59)), 0.76)]
    det = object.__new__(YoloDetector)
    det.task, det.iou, det.classes, det.device, det.names, det.extra, det.conf = "detect", 0.7, [2], None, {2: "car"}, set(), 0.23
    det.whole_imgsz, det.near_imgsz, det.near_band, det.whole_iou, det.whole_ios = 1920, 640, 0.3, 0.45, 0.8
    det.model = _SizeModel({1920: main, 640: low})
    out = det(np.zeros((1080, 1920, 3), np.uint8))
    assert det.model.sizes == [1920, 640]
    assert [(d["conf"], d["src"]) for d in out] == [(0.94, "near"), (0.84, ""), (0.82, "")]  # the 0.24 box: merged
    det.near_imgsz, det.model = None, _SizeModel({1920: main, 640: low})  # off: the main pass alone, as before
    assert [d["conf"] for d in det(np.zeros((1080, 1920, 3), np.uint8))] == [0.84, 0.82, 0.24]
    assert det.model.sizes == [1920]
