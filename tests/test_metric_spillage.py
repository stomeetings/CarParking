"""Spillage on the ground (spillage.GroundMeasure, FixedSlotTracker.ground_cam; post-demo item 1b): with a calibrated camera a bay is
ENCROACHED when the vehicle's ground footprint stands on it, on whichever axis the neighbours lie (side by side, or stacked in depth like
camera-1's bays B1..B7), whatever its 2D box looks like. Synthetic scene: bays on the ground seen by a pinhole camera 6 m up, projected
through the geometry package. No model, no network, no ./data."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.fixed_slots import FixedSlotTracker, footprint_poly  # noqa: E402
from carparking.geometry import CameraModel, to_image  # noqa: E402
from carparking.geometry_live import GeometryFeed  # noqa: E402
from carparking.live import Worker  # noqa: E402
from carparking.spillage import CFG, GroundMeasure, SpillCfg, spill_map  # noqa: E402

W, H = 1920, 1080
CAM = CameraModel(f=1200.0, cx=W / 2, cy=H / 2, pitch=np.radians(30), roll=0.0, h=6.0)
BAY_M, DEPTH_M, Y0 = 2.5, 5.0, 9.0  # bays A1..A4 side by side from x = -5, near edge at y = 9 m
SHAPE = (H, W, 3)
T0 = 1000.0
TIGHT = SpillCfg(share=0.12, leave=0.08, total=0.30)  # a small overhang already counts in pixels


def px(x, y, cam=CAM):
    return to_image(cam, np.array([[x, y]]))[0]


def quad(x0, x1, y0, y1, cam=CAM):
    return [list(px(x, y, cam) / (W, H)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def bay_poly(i, cam=CAM):
    x0 = -5.0 + BAY_M * i
    return quad(x0, x0 + BAY_M, Y0, Y0 + DEPTH_M, cam)


SLOTS = [{"id": i + 1, "name": f"A{i + 1}", "poly": bay_poly(i)} for i in range(4)]
# camera-1's layout: the bays are strips across the lane, one behind the other (the neighbour is BEHIND, not beside)
STACK = [{"id": i + 1, "name": f"B{i + 1}", "poly": quad(-2.4, 2.4, Y0 + 3.0 * i, Y0 + 3.0 * (i + 1))} for i in range(3)]


def boxcar(xa, xb, dy=0.4, cam=CAM):
    """A vehicle by its BOX, from xa to xb metres on the ground, its bottom edge at depth Y0 + dy (its wheels are what the measure derives)."""
    a, b = px(xa, Y0 + dy, cam), px(xb, Y0 + dy, cam)
    ytop = a[1] - 0.45 * (b[0] - a[0])
    return {"poly": [(a[0], ytop), (b[0], ytop), (b[0], a[1]), (a[0], a[1])], "box": (a[0], ytop, b[0], a[1]), "label": "car", "conf": 0.8, "state": 1}


def car(x0, x1, dy=0.4, cam=CAM):
    """A vehicle whose wheels stand from x0 to x1 (metres): the box is wider by the share the measure trims (SpillCfg.wheel_inset)."""
    pad = (x1 - x0) * CFG.wheel_inset / (1 - 2 * CFG.wheel_inset)
    return boxcar(x0 - pad, x1 + pad, dy, cam)


def bay_x(i, share=0.0):
    return -5.0 + BAY_M * i + share * BAY_M


def tracker(ground=True, slots=SLOTS, **kw):
    tr = FixedSlotTracker(slots, "ground", hold=2, exit_hold=3, absent_free=True, spillage=1, **kw)
    tr.ground_cam = CAM if ground else None
    return tr


def settle(tr, dets, n=14):
    for k in range(n):
        tr.update(dets, SHAPE, now=T0 + k)
    return tr


def fps(dets):
    return [footprint_poly(np.float32(d["poly"]) / (W, H)) for d in dets]


def measure(dets, **kw):
    return GroundMeasure(CAM, (W, H), fps(dets), **kw)


def spill(bay_slots, k, dets, owners, cfg=CFG, **kw):
    return spill_map({k: np.array(bay_slots[k - 1]["poly"])}, fps(dets), owners, {k}, cfg=cfg, ground=measure(dets, cfg=cfg, **kw))


def test_the_scene_is_what_the_tests_assume():
    g = measure([car(bay_x(0, 0.1), bay_x(0, 0.9))])
    assert g.width_m(1, SLOTS[0]["poly"]) == pytest.approx(BAY_M, abs=0.02)  # the short side of the bay, in metres


def test_cover_is_the_share_of_the_vehicle_standing_on_the_bay():
    c = car(bay_x(0, 0.1), bay_x(1, 0.40))  # wheels over 3.25 m, 1.0 m of them past the line into A2
    g = measure([c])
    assert g.cover(2, SLOTS[1]["poly"], 0) == pytest.approx(1.0 / 3.25, abs=0.03)
    assert g.cover(3, SLOTS[2]["poly"], 0) == 0.0  # does not reach A3
    assert g.cover(1, SLOTS[0]["poly"], 0) == pytest.approx(2.25 / 3.25, abs=0.03)  # and 2.25 m of it stands in its own bay


def test_a_car_wholly_in_its_bay_encroaches_nobody():
    c = car(bay_x(1, 0.08), bay_x(1, 0.92))
    for k in (1, 3, 4):
        assert spill(SLOTS, k, [c], [2]) == {}


def test_stacked_bays_the_neighbour_behind_is_seen():
    # camera-1's layout: a side-on car in B1 whose far end reaches into B2 (the contact line alone shows nothing: it is a point across the lane)
    c = car(-2.0, 2.0, dy=1.8)
    sp = spill(STACK, 2, [c], [1])
    assert 2 in sp and sp[2].metric and sp[2].cover >= CFG.share
    assert spill(STACK, 3, [c], [1]) == {}  # two bays back: no
    inside = car(-2.0, 2.0, dy=0.3)  # well inside B1
    assert spill(STACK, 2, [inside], [1]) == {}


def test_a_car_with_a_visible_gap_does_not_encroach_but_the_pixel_rule_says_it_does():
    # a van's 2.6 m box leans 14 % into A4, its wheels (the box less 6 % at each end) only about 8 % of itself; a tight share (12 %) shows it
    c = boxcar(bay_x(2, 0.12), bay_x(3, 0.14))
    pix = spill_map({4: np.array(SLOTS[3]["poly"])}, fps([c]), [3], {4}, cfg=TIGHT)
    assert 4 in pix and pix[4].cover >= TIGHT.share and not pix[4].metric
    assert spill(SLOTS, 4, [c], [3], cfg=TIGHT) == {}


def test_union_mode_keeps_the_pixel_reading_when_the_ground_one_is_too_cautious():
    c = boxcar(bay_x(2, 0.12), bay_x(3, 0.14))
    union = spill(SLOTS, 4, [c], [3], cfg=TIGHT, mode="union")
    assert 4 in union and union[4].cover >= TIGHT.share
    assert spill(SLOTS, 4, [c], [3], cfg=TIGHT, mode="auto") == {}
    # and where the ground reading is the higher one, union does not lower it
    on_line = car(bay_x(0, 0.1), bay_x(1, 0.45))
    assert spill(SLOTS, 2, [on_line], [1], mode="union")[2].cover >= spill(SLOTS, 2, [on_line], [1])[2].cover - 1e-9


def test_wheels_over_the_line_encroach_with_the_metric_free_width():
    c = car(bay_x(0, 0.1), bay_x(1, 0.45))
    sp = spill(SLOTS, 2, [c], [1])[2]
    assert sp.metric and sp.by == (1,)
    assert sp.cover == pytest.approx(1.125 / 3.375, abs=0.03)
    assert sp.free_m == pytest.approx(BAY_M * (1 - sp.total), abs=0.02)


def test_the_margin_in_metres_ignores_a_hand_s_width():
    c = car(bay_x(0, 0.1), bay_x(1, 0.04))  # 10 cm into A2
    assert spill(SLOTS, 2, [c], [1], cfg=SpillCfg(share=0.01, leave=0.005, total=0.02)) == {}
    c = car(bay_x(0, 0.1), bay_x(1, 0.20))  # 0.5 m in
    assert measure([c]).cover(2, SLOTS[1]["poly"], 0) > 0.15
    strict = SpillCfg(share=0.30, margin_m=5.0)  # the margin is capped at a quarter of the bay's short side (0.625 m): 0.5 m is still short of it
    assert measure([c], cfg=strict).cover(2, SLOTS[1]["poly"], 0) == 0.0


def test_two_cars_add_up_on_the_ground():
    a, b = car(bay_x(0, 0.1), bay_x(1, 0.36)), car(bay_x(1, 0.64), bay_x(3, 0.5))  # 0.9 m from the left, 0.9 m from the right
    sp = spill(SLOTS, 2, [a, b], [1, 4])[2]
    assert set(sp.by) == {1, 4} and sp.cover < CFG.share and sp.total >= CFG.total  # each under 0.30, together over 0.45


def test_a_car_in_front_of_the_bays_is_not_on_them():
    c = car(bay_x(0, 0.1), bay_x(1, 0.6), dy=-7.0)  # between the lens and the row
    assert measure([c]).cover(2, SLOTS[1]["poly"], 0) in (0.0, None)


def test_a_wrong_camera_height_does_not_change_the_verdict():
    # shares cancel a scale error: the same pixels read with a camera 25 % higher/lower give about the same cover (the guessed depth of a footprint is in metres, so not exactly)
    c = car(bay_x(0, 0.1), bay_x(1, 0.45))
    base = measure([c]).cover(2, SLOTS[1]["poly"], 0)
    for h in (4.5, 7.5):
        cam = CameraModel(CAM.f, CAM.cx, CAM.cy, CAM.pitch, CAM.roll, h)
        got = GroundMeasure(cam, (W, H), fps([c])).cover(2, SLOTS[1]["poly"], 0)
        assert got == pytest.approx(base, abs=0.10), (h, got, base)
        assert got >= CFG.share


def test_a_rolled_camera_still_sees_the_overlap():
    cam = CameraModel(CAM.f, CAM.cx, CAM.cy, CAM.pitch, np.radians(6), CAM.h)
    c = car(bay_x(0, 0.1), bay_x(1, 0.45), cam=cam)
    bay = quad(bay_x(1), bay_x(2), Y0, Y0 + DEPTH_M, cam)
    assert GroundMeasure(cam, (W, H), fps([c])).cover(2, bay, 0) >= CFG.share


def test_unreliable_ground_gives_none_so_the_pixel_rule_answers():
    c = car(bay_x(0, 0.1), bay_x(1, 0.45))
    g = measure([c], max_res=1e-6)  # nothing is measurable that finely
    assert g.cover(2, SLOTS[1]["poly"], 0) is None
    sp = spill_map({2: np.array(SLOTS[1]["poly"])}, fps([c]), [1], {2}, cfg=CFG, ground=g)
    assert 2 not in sp or (not sp[2].metric and sp[2].free_m is None)


def test_tracker_marks_encroached_only_when_the_footprint_crosses():
    wide_gap = [boxcar(bay_x(2, 0.12), bay_x(3, 0.14))]
    on_line = [car(bay_x(2, 0.1), bay_x(3, 0.45))]
    metric, pixel = settle(tracker(True, spill_cfg=TIGHT), wide_gap), settle(tracker(False, spill_cfg=TIGHT), wide_gap)
    assert 4 not in metric.avail and metric.avail_name(4) == "FREE"
    assert 4 in pixel.avail  # no calibration = the pixel threshold, as before
    crossing = settle(tracker(True, spill_cfg=TIGHT), on_line)
    assert 4 in crossing.avail and crossing.avail[4].metric
    assert crossing.shown(4) == 0  # the state stays FREE, only the usable flag changes


def test_tracker_sees_the_stacked_neighbour_and_clears_it_when_the_car_leaves():
    tr = tracker(True, slots=STACK)
    settle(tr, [car(-2.0, 2.0, dy=1.8)])
    assert 2 in tr.avail and tr.avail[2].metric and tr.avail_name(2) == "ENCROACHED"
    for k in range(14, 40):
        tr.update([], SHAPE, now=T0 + k)
    assert tr.avail == {}


def test_tracker_union_mode_is_a_live_switch_with_no_effect_on_states():
    van = [boxcar(bay_x(2, 0.12), bay_x(3, 0.14))]
    auto, union = tracker(True, spill_cfg=TIGHT), tracker(True, spill_cfg=TIGHT)
    union.ground_mode = "union"
    settle(auto, van)
    settle(union, van)
    assert 4 not in auto.avail and 4 in union.avail
    assert [auto.shown(k) for k in (1, 2, 3, 4)] == [union.shown(k) for k in (1, 2, 3, 4)]


def test_no_calibration_is_byte_identical_to_before():
    dets = [car(bay_x(0, 0.1), bay_x(1, 0.45)), car(bay_x(3, 0.1), bay_x(3, 0.9))]
    a, b = tracker(False), FixedSlotTracker(SLOTS, "ground", hold=2, exit_hold=3, absent_free=True, spillage=1)
    for k in range(14):
        ea, eb = a.update(dets, SHAPE, now=T0 + k), b.update(dets, SHAPE, now=T0 + k)
        assert ea == eb and a.avail_events == b.avail_events
    assert {k: (v.cover, v.total, v.by, v.free_m) for k, v in a.avail.items()} == {k: (v.cover, v.total, v.by, v.free_m) for k, v in b.avail.items()}
    assert all(v.free_m is None and not v.metric for v in a.avail.values())


def test_a_broken_camera_falls_back_to_pixels_not_to_a_lost_frame():
    tr = tracker(True)
    tr.ground_cam = object()  # not a camera
    settle(tr, [car(bay_x(0, 0.1), bay_x(1, 0.45))])
    assert 2 in tr.avail and not tr.avail[2].metric


def test_free_width_is_published_in_the_state():
    from carparking.monitor import ParkingMonitor
    src = Path(ParkingMonitor.state.__code__.co_filename).read_text(encoding="utf-8")
    assert 's["free_m"] = sp.free_m' in src  # (the ground metric's extra field, only present when measured)


def test_spillage_config_is_validated():
    with pytest.raises(ValueError):
        SpillCfg(depth_trust=0.1)
    with pytest.raises(ValueError):
        SpillCfg(wheel_inset=0.5)
    with pytest.raises(ValueError):
        SpillCfg(margin_m=-1)


class _Store:
    def __init__(self, ver):
        self.ver = ver

    def active_version(self):
        return self.ver


def test_geometry_feed_camera_is_none_without_a_calibration_and_never_raises():
    feed = GeometryFeed(lambda: _Store(None), "front", log=lambda *a: None)
    assert feed.camera([W, H], 0.0) is None
    broken = GeometryFeed(lambda: (_ for _ in ()).throw(RuntimeError("down")), "front", log=lambda *a: None)
    assert broken.camera([W, H], 0.0) is None


def test_worker_hands_the_camera_to_the_tracker_by_option_and_only_while_spillage_is_on():
    frame = np.zeros((H, W, 3), np.uint8)
    feed = SimpleNamespace(camera=lambda shape, now: CAM if list(shape) == [W, H] else None)
    opts = {}
    tr = tracker(False)
    w = SimpleNamespace(mon=SimpleNamespace(tracker=tr), geometry=feed, store=SimpleNamespace(option=lambda k, d="": opts.get(k, d)))
    Worker._ground(w, frame, 5.0)
    assert tr.ground_cam is CAM and tr.ground_mode == "auto"
    opts["spillage_ground"] = "union"
    Worker._ground(w, frame, 6.0)
    assert tr.ground_cam is CAM and tr.ground_mode == "union"
    opts["spillage_ground"] = "off"  # pixels only
    Worker._ground(w, frame, 7.0)
    assert tr.ground_cam is None
    opts.clear()
    tr.spillage = 0
    Worker._ground(w, frame, 8.0)
    assert tr.ground_cam is None  # layer off: nothing measured
    tr.spillage = 1
    w.geometry = None
    Worker._ground(w, frame, 9.0)
    assert tr.ground_cam is None  # no geometry feed: the pixel rule


def test_state_counts_an_unknown_bay_behind_the_car_as_encroached_without_touching_free(tmp_path):
    from carparking.history import HistoryDB
    from carparking.monitor import ParkingMonitor

    class Static:
        conf = targets = None

        def __call__(self, frame, sliced=False):
            return [dict(d) for d in [car(-2.0, 2.0, dy=1.8)]]

    mon = ParkingMonitor(Static(), HistoryDB(tmp_path / "h.db", "cam", detector="fake"), fixed=STACK, match_rule="ground", hold=2, exit_hold=3,
                         absent_free=True, spillage=1)
    mon.start(SHAPE)
    mon.tracker.ground_cam = CAM
    for t in range(14):
        mon.process(np.zeros(SHAPE, np.uint8), t=T0 + t)
    st = mon.state()
    by_id = {s["id"]: s for s in st["slots"]}
    assert by_id[1]["state"] == "BUSY" and by_id[2]["state"] == "UNKNOWN" and by_id[2]["display"] == "ENCROACHED"
    assert by_id[2]["free_m"] > 0 and by_id[2]["encroached_by"] == ["B1"]
    c = st["counts"]
    assert c["encroached"] == 1 and c["free"] == sum(s["state"] == "FREE" for s in st["slots"])  # only a FREE bay leaves the free count
