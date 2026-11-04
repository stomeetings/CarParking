"""Sample-image pipeline (carparking/sample_test.py): one picture through a camera's whole stack in a throw-away tracker. The detector is scripted (no model);
the monitor, trackers and layers are the real ones. The HTTP side is tested in test_sample_sessions.py."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from test_blocked_objects import thing  # noqa: E402
from test_live import run_cfg  # noqa: E402
from test_spillage import SHAPE, SLOTS, car_in, det, over  # noqa: E402

from carparking import sample_test  # noqa: E402

FRAME = np.full(SHAPE, 110, np.uint8)
ALL = {"spillage": True, "obstacle_filter": True, "blocked_objects": True}
NONE = {"spillage": False, "obstacle_filter": False, "blocked_objects": False}


def run(**over):
    return run_cfg(fixed_mode=True, **over)


def go(raw, layers, slots=SLOTS, **kw):
    return sample_test.run_sample(FRAME, run(**kw), slots, raw, layers)


def by(out):
    return {s["id"]: s for s in out["slots"]}


# ---- the pipeline ----------------------------------------------------------------------------------------------------------------------------
def test_a_parked_car_makes_its_bay_busy_and_logs_an_entry_to_the_throwaway_history():
    out = go([car_in(0)], NONE)
    assert [by(out)[i]["state"] for i in (1, 2, 3, 4)] == ["BUSY", "FREE", "FREE", "FREE"]
    assert (out["counts"]["busy"], out["counts"]["free"], out["counts"]["slots"]) == (1, 3, 4)
    assert [(e["event"], e["slot_id"]) for e in out["events"]] == [("entry", 1)]
    assert [d["kind"] for d in out["detections"]] == ["vehicle"] and out["detections"][0]["label"] == "car"
    assert out["shape"] == [2000, 1000] and out["frames"] >= 8 and out["layers"] == {**NONE, "ground": False}
    assert "obstacle" not in out and "available" not in out["counts"]  # no layer on: the plain state


def test_a_pot_blocks_its_bay_with_the_layer_on_and_is_invisible_with_it_off():
    out = go([car_in(0), thing(2)], ALL)
    s = by(out)
    assert s[3]["state"] == "FREE" and s[3]["avail"] == "BLOCKED" and s[3]["display"] == "BLOCKED" and s[3]["blocked_by"] == ["potted plant"]
    c = out["counts"]
    assert (c["busy"], c["blocked_objects"], c["free"], c["available"]) == (1, 1, 2, 2)
    assert [(e["event"], e["slot_id"]) for e in out["events"]] == [("entry", 1), ("object_blocked", 3)]
    assert [d["label"] for d in out["detections"] if d["kind"] == "other"] == ["potted plant"] and out["vehicles"] == 1 and out["others"] == 1
    off = go([car_in(0), thing(2)], NONE)
    assert by(off)[3]["state"] == "FREE" and "avail" not in by(off)[3] and off["counts"]["free"] == 3
    assert [e["event"] for e in off["events"]] == ["entry"]


def test_spillage_marks_the_bay_a_neighbours_car_stands_on():
    out = go([over(0, 0.45)], ALL)
    s = by(out)
    assert s[1]["state"] == "BUSY" and s[2]["state"] == "FREE" and s[2]["avail"] == "ENCROACHED" and s[2]["encroached_by"] == ["A1"]
    assert (out["counts"]["encroached"], out["counts"]["available"], out["counts"]["free"]) == (1, 2, 2)
    assert [e["event"] for e in out["events"]] == ["entry", "encroached"]
    assert by(go([over(0, 0.45)], NONE))[2].get("avail") is None  # spillage off: a plain FREE bay


def test_the_obstacle_filter_drops_a_cone_sized_box_and_says_why():
    cone = det(bay_x_cone := 0.62, 0.64, bottom=0.69, top=0.62)  # a box far too small for its bay
    off = go([cone], NONE)
    assert by(off)[3]["state"] == "BUSY"  # without the filter a small box holds a bay
    on = go([cone], ALL)
    assert by(on)[3]["state"] == "FREE" and on["obstacle"] == {"rejected": 1, "reasons": {"too_small": 1}}
    assert [d.get("rejected") for d in on["detections"]] == ["too_small"] and on["vehicles"] == 0
    assert bay_x_cone  # (the position only)


def test_every_layer_at_once_each_bay_has_one_name_and_the_counts_add_up():
    out = go([over(0, 0.45), thing(3)], ALL)
    s = by(out)
    names = {i: s[i].get("avail") or s[i]["state"] for i in s}
    assert names == {1: "BUSY", 2: "ENCROACHED", 3: "FREE", 4: "BLOCKED"}
    c = out["counts"]
    assert (c["busy"], c["encroached"], c["blocked_objects"], c["available"], c["slots"]) == (1, 1, 1, 1, 4) and c["free"] == 1  # three FREE-by-state bays, two of them not usable


def test_an_auto_grid_cameras_run_matches_on_the_foot_point_and_the_run_is_not_changed():
    cfg = run_cfg(fixed_mode=False)
    before = repr(cfg)
    out = sample_test.run_sample(FRAME, cfg, SLOTS, [car_in(1)], NONE)
    assert by(out)[2]["state"] == "BUSY" and repr(cfg) == before  # the caller's settings are copied, never edited


def test_the_replay_schedule_follows_the_hold_time_in_steps_that_stay_small():
    assert sample_test.schedule({"hold": 0}) == (6, 1.0) and sample_test.schedule({"hold": 3}) == (12, 1.0)
    assert sample_test.schedule({"hold": 15}) == (16, 3.0) and sample_test.schedule({"hold": 120}) == (16, 24.0)
    assert sample_test.schedule({"hold": 900}) == (66, 30.0)  # the step is capped, the frame count grows
    for hold in (2, 15, 40):  # a long entry delay still settles, with spillage chained after the car's BUSY
        cfg = run()
        cfg["m5"]["hold"] = hold
        out = sample_test.run_sample(FRAME, cfg, SLOTS, [over(0, 0.45)], ALL)
        assert by(out)[1]["state"] == "BUSY" and by(out)[2]["avail"] == "ENCROACHED", hold


def test_the_replay_detector_answers_the_same_every_time_with_copies():
    d = sample_test.Replay([{"label": "car", "box": (1, 2, 3, 4)}])
    a, b = d(None), d(None)
    assert a == b and a[0] is not b[0]
    a[0]["label"] = "x"
    assert d(None)[0]["label"] == "car"


def test_detect_frame_adds_the_object_classes_only_when_asked(monkeypatch):
    seen = []

    class Det:
        conf = None

        def set_targets(self, targets):
            seen.append(list(targets))

        def __call__(self, frame, sliced=False):
            return []

    monkeypatch.setattr("carparking.live.build_detector", lambda spec, frame, root, cache: (Det(), False, "", True))
    cfg = run()
    cfg["m5"]["classes"] = ["car"]
    sample_test.detect_frame(cfg, FRAME, ROOT, objects=False)
    sample_test.detect_frame(cfg, FRAME, ROOT, objects=True)
    assert seen[0] == ["car"] and seen[1][0] == "car" and "potted plant" in seen[1] and "person" not in seen[1]




def test_detect_frame_applies_the_confidence_and_nms_of_the_call_and_puts_the_detector_back(monkeypatch):
    class Det:
        conf, iou, whole_iou, classes, extra = 0.3, 0.7, 0.45, [2], set()
        seen = None

        def set_targets(self, targets):
            pass

        def __call__(self, frame, sliced=False):
            self.seen = (self.conf, self.iou, self.whole_iou)
            return []

    det = Det()
    monkeypatch.setattr("carparking.live.build_detector", lambda spec, frame, root, cache: (det, False, "", True))
    sample_test.detect_frame(run(), FRAME, ROOT, tune={"conf": 0.12, "iou": 0.3})
    assert det.seen == (0.12, 0.3, 0.3)  # the pass ran with them ...
    assert (det.conf, det.iou, det.whole_iou) == (0.3, 0.7, 0.45)  # ... and the shared detector is as it was


def test_detector_spec_runs_other_weights_whole_frame_or_as_the_bay_model():
    base = {"detector": {"kind": "coco", "weights": ["yolov8s.pt"], "whole": 1280}}
    assert sample_test.detector_spec(base) is base["detector"]
    assert sample_test.detector_spec(base, "yolov8x.pt") == {"kind": "coco", "weights": ["yolov8x.pt"], "whole": 1280}
    assert sample_test.detector_spec({"detector": {"kind": "bay"}}, "yolov8x.pt")["whole"] == 1920
    assert sample_test.detector_spec(base, "models/parking_yolo.pt", bay=True)["kind"] == "bay"


def test_the_footprint_ratio_decides_when_an_object_blocks_a_bay():
    raw = [thing(2, share=0.978)]  # the pot straddles the right edge of bay 3: 60 % of its footprint is inside
    assert by(go(raw, ALL))[3]["avail"] == "BLOCKED"  # the default needs 50 %
    assert by(sample_test.run_sample(FRAME, run(), SLOTS, raw, ALL, tune={"footprint": 0.8}))[3].get("avail") is None
    assert by(sample_test.run_sample(FRAME, run(), SLOTS, raw, ALL, tune={"footprint": 0.3}))[3]["avail"] == "BLOCKED"


def test_the_encroach_ratio_decides_when_a_neighbour_makes_a_free_bay_unusable():
    raw = [over(0, 0.45)]  # a car in bay 1 that also covers 45 % of bay 2
    assert by(go(raw, ALL))[2]["avail"] == "ENCROACHED"  # the default: 30 %
    assert by(sample_test.run_sample(FRAME, run(), SLOTS, raw, ALL, tune={"encroach": 0.6}))[2].get("avail") is None


def test_the_auto_grid_rules_give_a_car_off_the_grid_a_bay_of_its_own_and_are_off_by_default():
    off = car_in(1)  # a car in bay 2 ...
    raw = [off]
    only = [s for s in SLOTS if s["id"] != 2]  # ... on a map that has no bay 2: like a car beyond the packed tiles
    surface = [[0.0, 0.3], [1.0, 0.3], [1.0, 0.95], [0.0, 0.95]]
    plain = sample_test.run_sample(FRAME, run(), only, raw, NONE)
    assert [s["id"] for s in plain["slots"]] == [1, 3, 4] and plain["counts"]["busy"] == 0  # the old behaviour: the car is ignored
    grid = sample_test.run_sample(FRAME, run(), only, raw, NONE, auto_surface=surface, split=True)
    spawned = [s for s in grid["slots"] if s["id"] >= 1000]
    assert len(spawned) == 1 and spawned[0]["name"] == "D1" and spawned[0]["state"] == "BUSY" and grid["counts"]["busy"] == 1
    assert grid["unplaced"] == []
    low = [dict(off, conf=0.3)]  # under SPAWN_CONF: no bay, and the reason is reported
    quiet = sample_test.run_sample(FRAME, run(), only, low, NONE, auto_surface=surface, split=True)
    assert not [s for s in quiet["slots"] if s["id"] >= 1000] and [u["why"] for u in quiet["unplaced"]] == ["confidence under 0.45"]


def test_classify_drawn_takes_the_state_of_the_bay_it_overlaps_most():
    def box(x, y, state, w=100, h=50):
        return {"poly": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], "state": state, "conf": 0.7}

    def slot(i, x, y, w=100, h=50):
        return {"id": i, "name": f"Slot {i}", "poly": [[x / 1000, y / 500], [(x + w) / 1000, y / 500], [(x + w) / 1000, (y + h) / 500], [x / 1000, (y + h) / 500]]}

    dets = [box(100, 100, 1), box(300, 100, 0), box(300, 100, 1, w=20, h=10)]
    out = sample_test.classify_drawn([slot(1, 110, 105), slot(2, 305, 102), slot(3, 700, 400), slot(4, 90, 90, w=300, h=100)], dets, 1000, 500)
    # no bay near slot 3; slot 4 is far bigger than either bay it touches, so neither overlap is enough
    assert [s["state"] for s in out] == ["BUSY", "FREE", "UNKNOWN", "UNKNOWN"]
    assert out[2]["conf"] == 0.0 and out[0]["conf"] == 0.7 and [s["id"] for s in out] == [1, 2, 3, 4]
    assert sample_test.classify_drawn([], dets, 1000, 500) == [] and sample_test.classify_drawn([slot(1, 100, 100)], [], 1000, 500)[0]["state"] == "UNKNOWN"


def test_classify_crops_judges_each_slot_on_its_own_crop_and_leaves_out_what_the_model_cannot_see():
    frame = np.zeros((500, 1000, 3), np.uint8)
    seen = []

    def detect(crop):
        seen.append(crop.shape[:2])
        h, w = crop.shape[:2]
        if len(seen) == 1:  # the first slot: a busy bay in the middle of its crop
            return [{"poly": [(w * 0.3, h * 0.3), (w * 0.7, h * 0.3), (w * 0.7, h * 0.7), (w * 0.3, h * 0.7)], "state": 1, "conf": 0.6}]
        if len(seen) == 2:  # the second: only a bay far outside the slot
            return [{"poly": [(0, 0), (4, 0), (4, 4), (0, 4)], "state": 0, "conf": 0.9}]
        return []

    def slot(i, x, y):
        return {"id": i, "name": f"S{i}", "poly": [[x / 1000, y / 500], [(x + 100) / 1000, y / 500], [(x + 100) / 1000, (y + 50) / 500], [x / 1000, (y + 50) / 500]]}

    out = sample_test.classify_crops(frame, [slot(1, 100, 100), slot(2, 400, 100), slot(3, 800, 400)], detect)
    assert out == {1: ("BUSY", 0.6)} and len(seen) == 3 and all(max(s) == 640 for s in seen)  # each crop enlarged to 640 px


def test_classify_by_vehicles_gives_each_vehicle_to_one_slot_by_its_ground_footprint():
    def car(x0, y0, x1, y1, conf=0.7):
        return {"box": (x0, y0, x1, y1), "conf": conf}

    def slot(i, x, y, w=100, h=50):
        return {"id": i, "name": f"S{i}", "poly": [[x / 1000, y / 500], [(x + w) / 1000, y / 500], [(x + w) / 1000, (y + h) / 500], [x / 1000, (y + h) / 500]]}

    cars = [car(100, 40, 200, 150), car(500, 100, 520, 120, 0.9), car(700, 60, 760, 140)]
    slots = [slot(1, 110, 100), slot(2, 300, 100), slot(3, 480, 80, 200, 100), slot(4, 705, 70, w=30, h=20)]
    out = sample_test.classify_by_vehicles(slots, cars, 1000, 500)
    assert out[1] == ("BUSY", 0.7)  # the car's wheels are in the slot
    assert out[2] == ("FREE", 0.0)  # no vehicle
    assert out[3] == ("BUSY", 0.9)
    assert out[4] == ("FREE", 0.0)  # a tall car box leans over this slot, but it stands on the ground below it: an empty slot stays FREE
    # one car between two slots is one car: it makes the slot holding most of its wheels busy, not both
    pair = sample_test.classify_by_vehicles([slot(10, 100, 100), slot(11, 200, 100)], [car(130, 60, 240, 150)], 1000, 500)
    assert (pair[10][0], pair[11][0]) == ("BUSY", "FREE")
    assert sample_test.classify_by_vehicles([slot(1, 100, 100)], [], 1000, 500) == {1: ("FREE", 0.0)}
