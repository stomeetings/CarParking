import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carparking.fixed_slots import FixedSlotTracker  # noqa: E402
from carparking.slot_preview import preview_slots  # noqa: E402

A = {"id": 1, "name": "A1", "poly": [[0.10, 0.50], [0.30, 0.50], [0.30, 0.80], [0.10, 0.80]]}
B = {"id": 2, "name": "A2", "poly": [[0.30, 0.50], [0.50, 0.50], [0.50, 0.80], [0.30, 0.80]]}
C = {"id": 3, "name": "A3", "poly": [[0.60, 0.50], [0.80, 0.50], [0.80, 0.80], [0.60, 0.80]]}


def det(x1, y1, x2, y2, kind="vehicle"):
    return {"label": "car", "conf": 0.9, "kind": kind, "box": [x1, y1, x2, y2]}


def test_preview_says_free_or_busy_per_drawn_slot_with_the_fixed_rule():
    dets = [det(0.12, 0.40, 0.28, 0.75), det(0.62, 0.30, 0.78, 0.45), det(0.32, 0.55, 0.48, 0.78, kind="other")]
    out = preview_slots([A, B, C], dets)
    assert [(r["id"], r["state"]) for r in out] == [(1, "BUSY"), (2, "FREE"), (3, "FREE")]  # C: the car stands behind it
    assert out[0]["vehicle"]["box"] == [0.12, 0.40, 0.28, 0.75] and out[1]["vehicle"] is None  # a person is no vehicle


def test_one_vehicle_occupies_one_slot_and_selected_only_agrees_with_all():
    across = det(0.20, 0.45, 0.42, 0.78)  # ground point (0.31, 0.747) in B; the box also covers half of A
    all_ = preview_slots([A, B], [across])
    assert [(r["id"], r["state"]) for r in all_] == [(1, "FREE"), (2, "BUSY")]
    assert [(r["id"], r["state"]) for r in preview_slots([A, B], [across], only=[1])] == [(1, "FREE")]
    assert [(r["id"], r["state"]) for r in preview_slots([A, B], [across], only=[2])] == [(2, "BUSY")]


def test_preview_matches_what_the_fixed_tracker_decides():
    rng = np.random.default_rng(5)
    for rule in ("ground", "center", "iou"):
        for _ in range(30):
            dets = []
            for _ in range(rng.integers(0, 4)):
                x, y = rng.uniform(0.05, 0.75), rng.uniform(0.3, 0.6)
                dets.append(det(round(x, 3), round(y, 3), round(x + rng.uniform(0.1, 0.25), 3), round(y + rng.uniform(0.15, 0.3), 3)))
            tr = FixedSlotTracker([A, B, C], rule=rule, hold=0, exit_hold=0, absent_free=True)
            for t in range(3):
                tr.update([{"poly": [(d["box"][0] * 1000, d["box"][1] * 1000), (d["box"][2] * 1000, d["box"][1] * 1000),
                                     (d["box"][2] * 1000, d["box"][3] * 1000), (d["box"][0] * 1000, d["box"][3] * 1000)],
                            "state": 1} for d in dets], (1000, 1000, 3), now=float(t))
            want = {k: {0: "FREE", 1: "BUSY"}[s["state"]] for k, s in tr.slots.items()}
            assert {r["id"]: r["state"] for r in preview_slots([A, B, C], dets, rule)} == want, (rule, dets)


def test_bad_slots_are_refused_like_a_save():
    try:
        preview_slots([{"id": 1, "poly": [[0.1, 0.1], [1.4, 0.1], [0.2, 0.3]]}], [])
    except ValueError as e:
        assert "slot 1" in str(e)
    else:
        raise AssertionError("not refused")


def test_on_demand_detection_filters_like_the_worker(monkeypatch):
    import carparking.live as live
    from carparking.slot_preview import detect_on_demand

    def d(label, state, x1, y1, x2, y2, conf=0.9):
        return {"label": label, "state": state, "conf": conf, "box": (x1, y1, x2, y2),
                "poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]}

    class Fake:
        conf, targets = None, None

        def set_targets(self, names):
            self.targets = names

        def __call__(self, frame, sliced=False):
            return [d("car", 1, 100, 300, 300, 500), d("truck", 1, 500, 300, 700, 500), d("person", None, 320, 300, 360, 400),
                    d("car", 1, 850, 50, 990, 150)]  # the last one: outside the parking area (top 30 % cut)

    fake = Fake()
    monkeypatch.setattr(live, "build_detector", lambda spec, frame, root, cache: (fake, False, "fake", True))
    run = {"detector": {"kind": "coco", "weights": ["x.pt"]}, "conf": 0.42,
           "m5": {"classes": ["car"], "extra": ["person"], "top": 0.3, "bottom": 0.0, "left": 0.0, "right": 0.0, "roi": None}}
    got = detect_on_demand(run, np.zeros((1000, 1000, 3), np.uint8), "/nowhere")
    assert fake.conf == 0.42 and fake.targets == ["car", "person"]  # the camera's confidence and classes
    assert got == [{"label": "car", "conf": 0.9, "src": "", "kind": "vehicle", "box": [0.1, 0.3, 0.3, 0.5]}]
