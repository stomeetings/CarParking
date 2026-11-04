"""Non-vehicle filter (obstacle_filter.py wired into FixedSlotTracker, monitor.state(), live option "obstacle_filter"; post-demo item 2):
cones, bags, people, fences and shadows the detector calls vehicles must not make a bay BUSY, real vehicles must pass, and with the option
off nothing runs. Synthetic row of three bays on a 2000 x 1000 frame, one frame per second; no model, no network, no ./data."""
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.fixed_slots import FixedSlotTracker  # noqa: E402
from carparking.history import HistoryDB  # noqa: E402
from carparking.live import Worker, _dashed_poly, render_overlay  # noqa: E402
from carparking.monitor import ParkingMonitor  # noqa: E402
from carparking.obstacle_filter import CFG, ObstacleCfg, color_breakdown, judge  # noqa: E402

SHAPE = (1000, 2000, 3)
W, H = 2000, 1000
BAY_W, PITCH, TOP, BOTTOM = 0.18, 0.20, 0.50, 0.70  # bays A1..A3: x 0.05-0.23, 0.25-0.43, 0.45-0.63 (360 x 200 px = 72000 px2 each)
SLOTS = [{"id": i + 1, "name": f"A{i + 1}", "poly": [[0.05 + PITCH * i, TOP], [0.05 + PITCH * i + BAY_W, TOP],
                                                    [0.05 + PITCH * i + BAY_W, BOTTOM], [0.05 + PITCH * i, BOTTOM]]} for i in range(3)]
BAY_AREA = BAY_W * W * (BOTTOM - TOP) * H
T0 = 1000.0
ORANGE, RED, GREY, WHITE, DARK = (0, 128, 255), (0, 0, 200), (110, 110, 110), (235, 235, 235), (25, 25, 25)  # BGR


def det(x1, y1, x2, y2, conf=0.8, label="car"):
    return {"poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], "box": (x1, y1, x2, y2), "label": label, "conf": conf, "state": 1}


def bay_px(i):
    return round((0.05 + PITCH * i) * W), round(BOTTOM * H)  # left edge x, bottom y


def car(i, grow=1.0):
    """A car in bay i: 320 x 275 px (1.2 x the bay's area, aspect 1.16)."""
    x0, yb = bay_px(i)
    return det(x0 + 20, yb - 275, x0 + 20 + round(320 * grow), yb - 5)


def box_in(i, w, h, dx=100):
    """A w x h box standing in bay i (its wheels in the bay)."""
    x0, yb = bay_px(i)
    return det(x0 + dx, yb - 10 - h, x0 + dx + w, yb - 10)


def frame_with(*paints):
    """Grey ground with textured paint: (det, bgr colour)."""
    img = np.full(SHAPE, 110, np.uint8)
    rng = np.random.RandomState(0)
    for d, colour in paints:
        x1, y1, x2, y2 = d["box"]
        patch = np.empty((y2 - y1, x2 - x1, 3), np.uint8)
        patch[:] = colour
        noise = rng.randint(-12, 13, patch.shape[:2])[..., None]
        img[y1:y2, x1:x2] = np.clip(patch.astype(int) + noise, 0, 255).astype(np.uint8)
        cv2.line(img, (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2), tuple(int(c * 0.6) for c in colour), 3)  # a body line: edges
    return img


def tracker(on=1, **kw):
    return FixedSlotTracker(SLOTS, "ground", hold=2, exit_hold=3, absent_free=True, obstacle_filter=on, **kw)


def play(tr, plan, frame=None):
    """plan: one list of detections per second -> every event the tracker reports."""
    out = []
    for n, dets in enumerate(plan):
        tr.frame = frame
        out += [(n, e["slot"], e["state"]) for e in tr.update(dets, SHAPE, now=T0 + n) if not (e["initial"] and e["state"] == 0)]
    return out


def reasons(tr):
    return [r["reason"] for r in tr.rejected]


# ---- judge(): the rules ---------------------------------------------------------------------------------------------------------------
def crop_of(colour, w, h):
    return lambda: frame_with((det(0, 0, w, h), colour))[:h, :w]


def test_a_box_too_small_for_its_bay_is_rejected_whatever_its_pixels():
    assert judge((0, 0, 40, 70), BAY_AREA)[0] == "too_small"  # a cone: 4 % of the bay
    assert judge((0, 0, 320, 275), BAY_AREA)[0] is None  # a car: 122 %
    assert judge((0, 0, 0, 0), BAY_AREA)[0] == "too_small"
    assert judge((0, 0, 40, 70), None)[0] is None  # stands in no bay: no size to compare with (aspect still applies)


def test_the_aspect_bounds_reject_people_poles_and_fences_but_not_the_real_vehicles():
    assert judge((0, 0, 90, 250), BAY_AREA)[0] == "aspect"  # 0.36, big enough: a person close to the camera
    assert judge((0, 0, 1300, 200), BAY_AREA)[0] == "aspect"  # 6.5: a fence or kerb run
    for aspect in (0.65, 1.0, 1.77, 2.31, 3.16, 4.0):  # the aspects of the vehicles on camera-1 (0.65 .. 3.16) and a long van
        w = 250
        assert judge((0, 0, w, w / aspect), (w * w / aspect) / 1.2)[0] is None, aspect


def test_size_is_relative_to_a_pins_space_not_the_whole_pin():
    pin = BAY_AREA * 4  # a capacity-4 pin: the reference is one space (the tracker divides by the capacity)
    box = (0, 0, 200, 150)  # 30000 px2
    assert judge(box, pin / 4)[0] is None and judge(box, pin)[0] == "too_small"


def test_pylon_orange_rejects_a_not_whole_vehicle_box_and_never_a_whole_vehicle():
    mid = (0, 0, 150, 200)  # 42 % of the bay: passes the size rule, aspect 0.75
    assert judge(mid, BAY_AREA, crop_of(ORANGE, 150, 200))[0] == "pylon"
    assert judge(mid, BAY_AREA, crop_of(GREY, 150, 200))[0] is None  # same box, no cone colour
    assert judge(mid, BAY_AREA, crop_of(RED, 150, 200))[0] is None  # red is not the cone's hue band
    assert judge((0, 0, 320, 275), BAY_AREA, crop_of(ORANGE, 320, 275))[0] is None  # an orange car (122 %) is a car
    assert judge(mid, BAY_AREA, None)[0] is None  # no pixels: the colour rules are skipped, not guessed


def test_a_flat_dark_crop_is_a_shadow_and_a_black_car_with_edges_is_not():
    flat = lambda: np.full((275, 320, 3), 30, np.uint8)  # noqa: E731
    assert judge((0, 0, 320, 275), BAY_AREA, flat)[0] == "shadow"
    black = np.full((275, 320, 3), 30, np.uint8)
    cv2.rectangle(black, (60, 30), (260, 110), (120, 120, 120), -1)  # windscreen
    for cx in (70, 250):
        cv2.circle(black, (cx, 265), 28, (90, 90, 90), 4)
    assert judge((0, 0, 320, 275), BAY_AREA, lambda: black)[0] is None
    assert judge((0, 0, 320, 275), BAY_AREA, lambda: np.full((275, 320, 3), 200, np.uint8))[0] is None  # bright flat: glare, not a shadow


def test_colour_breakdown_adds_up_and_names_the_cone():
    cb = color_breakdown(crop_of(ORANGE, 150, 200)())
    assert cb["orange"] > 0.9 and cb["orange"] + cb["dark"] + cb["bright"] + cb["other"] == pytest.approx(1.0)
    assert color_breakdown(crop_of(WHITE, 150, 200)())["bright"] > 0.9
    assert color_breakdown(np.zeros((1, 1, 3), np.uint8))["other"] == 1.0  # degenerate crop


def test_cfg_is_validated():
    for bad in (dict(min_aspect=5.0), dict(min_area_share=0.9, color_size=0.5), dict(color_share=0), dict(pylon_hue=(20, 4))):
        with pytest.raises(ValueError):
            ObstacleCfg(**bad)
    assert CFG.min_area_share == 0.25  # the defaults the real-footage check was run with


# ---- FixedSlotTracker -----------------------------------------------------------------------------------------------------------------
def test_without_the_filter_a_cone_sized_box_makes_a_bay_busy_and_with_it_it_does_not():
    cone = box_in(1, 40, 70)
    plan = [[cone]] * 12
    off, on = tracker(0), tracker(1)
    assert [e for e in play(off, plan) if e[2] == 1]  # the false occupancy this removes
    assert play(on, plan) == [] and on.slots[2]["state"] != 1 and set(reasons(on)) == {"too_small"}
    assert on.rejected[0]["slot"] == 2 and on.rejected[0]["box"] == list(cone["box"])


def test_each_kind_of_clutter_is_dropped_before_it_can_occupy_a_bay():
    cases = {"too_small": (box_in(0, 40, 70), None),
             "aspect": (box_in(0, 90, 250, dx=40), None),
             "pylon": (box_in(0, 150, 200), ORANGE),
             "shadow": (car(0), DARK)}
    for reason, (d, colour) in cases.items():
        tr = tracker(1)
        img = frame_with((d, colour or GREY))
        if reason == "shadow":
            x1, y1, x2, y2 = d["box"]
            img[y1:y2, x1:x2] = 30  # flat dark
        assert play(tr, [[d]] * 10, img) == [], reason
        assert reasons(tr) == [reason] and tr.slots[1]["state"] != 1, reason


def test_rapid_transient_shifts_of_clutter_never_make_a_bay_busy():
    rng = random.Random(5)
    tr = tracker(1)
    img = frame_with((box_in(0, 150, 200), ORANGE), (box_in(1, 150, 200), ORANGE), (box_in(2, 150, 200), ORANGE))
    plan = []
    for _ in range(200):  # an orange object that jumps between bays and sizes every second, with a stray small box
        plan.append([box_in(rng.randrange(3), rng.choice((40, 150, 90)), rng.choice((70, 200, 250)), dx=rng.choice((20, 100))) if rng.random() > 0.2 else box_in(0, 30, 50)])
    assert play(tr, plan, img) == [] and all(s["state"] != 1 for s in tr.slots.values())


def test_real_vehicles_pass_and_the_run_is_identical_to_the_filter_off():
    rng = random.Random(11)
    plan = []
    for t in range(300):
        cars = []
        for i in range(3):
            if (t // 40 + i) % 3 != 0 and rng.random() > 0.08:
                w = rng.choice((0.9, 1.0, 1.2)) * 320
                x0, yb = bay_px(i)
                cars.append(det(x0 + 20, yb - round(rng.choice((220, 275))), x0 + 20 + round(w), yb - 5))
        plan.append(cars)
    img = frame_with(*[(c, RED) for c in plan[50]], *[(c, GREY) for c in plan[90]])
    a, b = tracker(0), tracker(1)
    ea, eb = play(a, plan, img), play(b, plan, img)
    assert ea == eb and ea  # the same events, and there were some
    assert {k: (s["state"], s["since"]) for k, s in a.slots.items()} == {k: (s["state"], s["since"]) for k, s in b.slots.items()}
    assert [a.shown(k) for k in a.slots] == [b.shown(k) for k in b.slots]


def test_a_bay_that_is_already_busy_is_never_judged():
    tr = tracker(1)
    img = frame_with((car(0), GREY))
    play(tr, [[car(0)]] * 10, img)
    assert tr.slots[1]["state"] == 1
    play(tr, [[box_in(0, 40, 70)]], img)  # something tiny where the car stands: not judged, not dropped
    assert tr.rejected == []


def test_without_a_frame_only_size_and_aspect_apply_and_the_frame_is_dropped_after_the_update():
    tr = tracker(1)
    assert play(tr, [[box_in(0, 40, 70)]] * 5, None) == [] and reasons(tr) == ["too_small"]
    tr.frame = frame_with((box_in(0, 150, 200), ORANGE))
    tr.update([box_in(0, 150, 200)], SHAPE, now=T0 + 99)
    assert tr.frame is None and reasons(tr) == ["pylon"]
    tr2 = tracker(1)
    assert play(tr2, [[box_in(0, 150, 200)]] * 6, None) != []  # a pylon-sized box without pixels is let through, as documented


def test_switching_it_on_and_off_while_running():
    tr = tracker(0)
    plan = [[box_in(0, 40, 70)]] * 6
    assert [e for e in play(tr, plan) if e[2] == 1]
    tr.set_obstacle_filter(1)
    assert tr.obstacle_filter == 1
    tr.update([box_in(1, 40, 70)], SHAPE, now=T0 + 50)
    assert reasons(tr) == ["too_small"]
    tr.set_obstacle_filter(0)
    assert tr.obstacle_filter == 0 and tr.rejected == []
    tr.set_obstacle_filter(0)  # unchanged


def test_off_is_the_default_and_leaves_no_trace():
    tr = FixedSlotTracker(SLOTS, "ground", hold=2, exit_hold=3, absent_free=True)
    assert tr.obstacle_filter == 0
    play(tr, [[box_in(0, 40, 70)]] * 6)
    assert tr.rejected == []


def test_the_filter_stays_within_the_time_budget_of_a_busy_frame():
    slots = [{"id": i + 1, "name": f"S{i}", "poly": [[0.02 + 0.03 * (i % 30), 0.5], [0.04 + 0.03 * (i % 30), 0.5],
                                                      [0.04 + 0.03 * (i % 30), 0.7], [0.02 + 0.03 * (i % 30), 0.7]]} for i in range(30)]
    tr = FixedSlotTracker(slots, "ground", hold=2, exit_hold=3, obstacle_filter=1)
    dets = [det(round((0.021 + 0.03 * i) * W), 420, round((0.039 + 0.03 * i) * W), 695) for i in range(20)]
    tr.frame = frame_with(*[(d, RED) for d in dets])
    best = min(_timed(tr, dets) for _ in range(7))
    assert best < 5e-3, f"{best * 1000:.1f} ms for 20 vehicles x 30 bays"


def _timed(tr, dets):
    t0 = time.perf_counter()
    for _ in range(10):
        tr._reject(dets, SHAPE)
    return (time.perf_counter() - t0) / 10


# ---- monitor state and the live option -----------------------------------------------------------------------------------------------
class Static:
    def __init__(self, dets):
        self.dets, self.conf, self.targets = dets, None, None

    def __call__(self, frame, sliced=False):
        return [dict(d) for d in self.dets]


def monitor(tmp_path, dets, **kw):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    return ParkingMonitor(Static(dets), db, fixed=SLOTS, match_rule="ground", hold=2, exit_hold=3, absent_free=True, **kw)


def test_the_monitor_hands_over_the_frame_and_state_lists_what_was_dropped(tmp_path):
    frame = frame_with((car(1), RED), (box_in(0, 150, 200), ORANGE))
    on = monitor(tmp_path, [car(1), box_in(0, 150, 200)], obstacle_filter=1)
    for t in range(10):
        on.process(frame, t=T0 + t)
    st = on.state()
    assert [r["reason"] for r in st["rejected"]] == ["pylon"] and st["rejected"][0]["slot"] == 1
    by = {s["id"]: s["state"] for s in st["slots"]}
    assert by == {1: "FREE", 2: "BUSY", 3: "FREE"} and on.tracker.frame is None
    import json
    json.dumps(st)


def test_state_has_no_trace_of_it_with_the_filter_off(tmp_path):
    frame = frame_with((car(1), RED), (box_in(0, 150, 200), ORANGE))
    off = monitor(tmp_path, [car(1), box_in(0, 150, 200)])
    for t in range(10):
        off.process(frame, t=T0 + t)
    st = off.state()
    assert "rejected" not in st and {s["id"]: s["state"] for s in st["slots"]}[1] == "BUSY"  # the orange object counts, as before


def test_the_worker_follows_the_obstacle_filter_option():
    tr = tracker(0)
    opts = {"obstacle_filter": "0"}
    w = SimpleNamespace(mon=SimpleNamespace(tracker=tr), store=SimpleNamespace(option=lambda k, d: opts.get(k, d)))
    Worker._obstacle_filter(w)
    assert tr.obstacle_filter == 0
    opts["obstacle_filter"] = "1"
    Worker._obstacle_filter(w)
    assert tr.obstacle_filter == 1
    opts["obstacle_filter"] = "0"
    Worker._obstacle_filter(w)
    assert tr.obstacle_filter == 0
    Worker._obstacle_filter(SimpleNamespace(mon=None, store=w.store))
    Worker._obstacle_filter(SimpleNamespace(mon=SimpleNamespace(tracker=object()), store=w.store))


# ---- the baked overlay --------------------------------------------------------------------------------------------------------------------
YELLOW = np.array([0, 220, 255])
GREY_LINE = np.array([150, 150, 150])


def edge_band(img, box):
    """The pixels along the left edge of a box (3 px wide band). The overlay image has a banner on top (draw()): the frame sits below it."""
    x1, y1, x2, y2 = box
    top = img.shape[0] - SHAPE[0]
    return img[top + y1 + 2:top + y2 - 2, x1 - 1:x1 + 2].reshape(-1, 3)


def close_to(px, colour, tol):
    return (np.abs(px.astype(int) - colour).max(axis=1) <= tol)


def test_a_dropped_box_is_drawn_grey_and_dashed_with_its_reason_and_a_vehicle_stays_yellow(tmp_path):
    frame = np.full(SHAPE, 110, np.uint8)
    thin = det(1500, 100, 1530, 300)  # aspect 0.15, stands in no bay: dropped for its shape
    on = monitor(tmp_path, [car(1), thin], obstacle_filter=1)
    for t in range(6):
        res = on.process(frame, t=T0 + t)
    assert [r["reason"] for r in on.tracker.rejected] == ["aspect"]
    img = render_overlay(frame, res, on.tracker)
    dropped, kept = edge_band(img, thin["box"]), edge_band(img, car(1)["box"])
    grey, yellow = close_to(dropped, GREY_LINE, 25), close_to(dropped, YELLOW, 60)
    assert 0 < grey.mean() < 0.9 and not yellow.any()  # grey, with gaps (dashed), never the vehicle colour
    assert close_to(kept, YELLOW, 60).mean() > 0.5 and not close_to(kept, GREY_LINE, 10).all()  # the accepted car is a yellow solid line
    top = img.shape[0] - SHAPE[0]
    label = img[top + 60:top + 100, 1495:1800].reshape(-1, 3)  # "ignored: aspect" in the same grey, above the box
    assert close_to(label, GREY_LINE, 12).sum() > 30


def test_with_the_filter_off_the_same_box_is_drawn_as_the_vehicle_it_always_was(tmp_path):
    frame = np.full(SHAPE, 110, np.uint8)
    thin = det(1500, 100, 1530, 300)
    off = monitor(tmp_path, [car(1), thin])
    for t in range(6):
        res = off.process(frame, t=T0 + t)
    img = render_overlay(frame, res, off.tracker)
    band = edge_band(img, thin["box"])
    assert close_to(band, YELLOW, 60).mean() > 0.5 and not close_to(band, GREY_LINE, 10).any()


def test_the_dashed_polygon_helper_leaves_gaps():
    img = np.zeros((100, 200, 3), np.uint8)
    _dashed_poly(img, [(10, 10), (190, 10), (190, 90), (10, 90)], (150, 150, 150))
    top = img[10, 10:190].max(axis=1) > 0
    assert 0.3 < top.mean() < 0.7 and img[50, 100].max() == 0
