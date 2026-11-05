"""A sample picture as its own camera (carparking/sample_session.py): a calibration from four ground pins (or GeoCalib + a typed distance), a bay grid from it,
then the picture through sample_test.run_sample. A synthetic camera and a scripted detector; no models, no network, no ./data."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from test_live import run_cfg  # noqa: E402

from carparking import sample_session as ss  # noqa: E402
from carparking import sample_test  # noqa: E402
from carparking.geometry import CameraModel, to_image  # noqa: E402

W, H = 1920, 1080
CAM = CameraModel(f=1500.0, cx=W / 2, cy=H / 2, pitch=np.radians(32), roll=np.radians(0.5), h=6.0)
FRAME = np.full((H, W, 3), 110, np.uint8)
GROUND = [(-4, 8), (4, 8), (12, 24), (-12, 24)]
SURFACE = (to_image(CAM, GROUND) / (W, H)).tolist()
PINS = (to_image(CAM, [(-1.0, 9.0), (1.5, 9.0), (1.5, 14.0), (-1.0, 14.0)]) / (W, H)).tolist()  # one 2.5 x 5 m bay, near-left first


def car(x, y, half=0.9, height=1.45):
    """A car-sized billboard at ground point (x, y) as the detector's raw detection (pixels)."""
    base = to_image(CAM, [(x - half, y), (x + half, y)])
    top = to_image(CAM.with_height(CAM.h - height), [(x - half, y), (x + half, y)])
    pts = np.vstack([base, top])
    (x1, y1), (x2, y2) = pts.min(axis=0), pts.max(axis=0)
    return {"poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], "box": (x1, y1, x2, y2), "label": "car", "conf": 0.9, "state": 1}


CARS = [car(-3, 11), car(1, 13), car(-1, 17)]


def test_a_picture_is_loaded_upright_and_scaled_down():
    big = np.zeros((1500, 3000, 3), np.uint8)
    frame = ss.load_frame(cv2.imencode(".jpg", big)[1].tobytes())
    assert frame.shape[1] == ss.MAX_WIDTH and frame.shape[0] == 1280  # 3000 x 1500 -> 2560 x 1280
    assert ss.load_frame(cv2.imencode(".png", FRAME)[1].tobytes()).shape == FRAME.shape
    with pytest.raises(ss.SessionError, match="decoded"):
        ss.load_frame(b"not an image")
    with pytest.raises(ss.SessionError, match="too small"):
        ss.load_frame(cv2.imencode(".png", np.zeros((100, 100, 3), np.uint8))[1].tobytes())


def test_four_pins_make_a_high_confidence_calibration_and_a_grid():
    rec = ss.rectangle_record(FRAME, CARS, pins=PINS, sides=(2.5, 5.0), polygon=SURFACE)
    cal = rec["calibration"]
    assert cal["confidence"] == "HIGH" and "manual" in cal["scale_sources"]
    assert cal["h"] == pytest.approx(6.0, rel=0.03) and cal["f"] == pytest.approx(1500.0, rel=0.03)
    assert rec["pins_fit"]["reprojection_px"] < 1.0 and not any("pins do not quite" in w for w in rec["warnings"])
    slots = ss.grid_slots(rec, W, H)
    assert len(slots) >= 6 and [s["id"] for s in slots] == list(range(1, len(slots) + 1))
    for s in slots:
        p = np.asarray(s["poly"])
        assert p.shape == (4, 2) and p.min() >= 0 and p.max() <= 1 and s["name"] == f"B{s['id']}" and s["source"] == "measured"
    # the bays are the size of a bay on the ground (2.5 x 5 m)
    for b in rec["capacity"]["slots"]:
        m = np.asarray(b["poly_m"])
        sides = sorted([np.linalg.norm(m[1] - m[0]), np.linalg.norm(m[2] - m[1])])
        assert sides[0] == pytest.approx(2.5, abs=0.4) and sides[1] == pytest.approx(5.0, abs=0.6)


def test_bad_clicks_say_what_to_fix_and_a_loose_rectangle_is_warned_about():
    with pytest.raises(ss.SessionError, match="in order"):
        ss.rectangle_record(FRAME, CARS, pins=[PINS[0], PINS[2], PINS[1], PINS[3]], sides=(2.5, 5.0), polygon=SURFACE)
    with pytest.raises(ss.SessionError, match="exactly four|at least 4"):
        ss.rectangle_record(FRAME, CARS, pins=PINS[:3], sides=(2.5, 5.0), polygon=SURFACE)
    with pytest.raises(ss.SessionError, match="parking surface"):
        ss.rectangle_record(FRAME, CARS, pins=PINS, sides=(2.5, 5.0), polygon=SURFACE[:2])
    with pytest.raises(ss.SessionError, match="between 0 and 1"):
        ss.rectangle_record(FRAME, CARS, pins=[[2, 2]] + PINS[1:], sides=(2.5, 5.0), polygon=SURFACE)
    with pytest.raises(ss.SessionError, match="positive"):
        ss.rectangle_record(FRAME, CARS, pins=PINS, sides=(0, 5.0), polygon=SURFACE)
    tiny = [(0.30, 0.50), (0.31, 0.50), (0.31, 0.52), (0.30, 0.52)]  # a speck is not a bay-sized rectangle seen by a camera
    with pytest.raises(ss.SessionError):
        ss.rectangle_record(FRAME, CARS, pins=tiny, sides=(2.5, 5.0), polygon=SURFACE)


def test_the_geocalib_route_takes_the_scale_from_a_typed_distance():
    up = (-np.array([np.cos(CAM.pitch) * np.sin(CAM.roll), np.cos(CAM.pitch) * np.cos(CAM.roll), np.sin(CAM.pitch)])).tolist()
    a, b = (to_image(CAM, [(-3, 12), (2, 15)]) / (W, H)).tolist()
    geo = {"f": 1500.0, "up": up, "k1": 0.0, "k2": 0.0, "model": "simple_radial"}
    rec = ss.geocalib_record(FRAME, CARS, geo=geo, depth=None, reference={"a": a, "b": b, "metres": float(np.hypot(5, 3))}, polygon=SURFACE)
    assert rec["calibration"]["confidence"] == "HIGH" and rec["calibration"]["h"] == pytest.approx(6.0, rel=0.03) and rec["calibration"]["lens"] == "simple_radial"
    with pytest.raises(ss.SessionError, match="positive"):
        ss.geocalib_record(FRAME, CARS, geo=geo, depth=None, reference={"a": a, "b": b, "metres": 0}, polygon=SURFACE)
    with pytest.raises(ss.SessionError, match="reference"):
        ss.geocalib_record(FRAME, CARS, geo=geo, depth=None, reference={"a": a}, polygon=SURFACE)


def test_without_any_scale_source_the_error_asks_for_a_reference():
    # no cars, no depth, and (this route) no usable reference distance: build_record refuses and the message reaches the user
    geo = {"f": 1500.0, "up": [0.0, -0.8, -0.5], "k1": 0.0, "k2": 0.0, "model": "pinhole"}
    with pytest.raises(ss.SessionError):
        ss.geocalib_record(FRAME, [], geo=geo, depth=None, reference={"a": [0.4, 0.8], "b": [0.4, 0.8], "metres": 3.0}, polygon=SURFACE)


def test_the_picture_runs_through_the_stack_on_its_own_grid():
    rec = ss.rectangle_record(FRAME, CARS, pins=PINS, sides=(2.5, 5.0), polygon=SURFACE)
    slots = ss.grid_slots(rec, W, H)
    profile = run_cfg()
    profile["m5"].update(top=0.0, right=0.4)  # a live camera's cuts: the session must not inherit them
    before = repr(profile)
    run = ss.session_run(profile)
    assert run["m5"]["right"] == 0.0 and run["fixed_mode"] is False and run["m5"]["roi"] is None and repr(profile) == before
    out = sample_test.run_sample(FRAME, run, slots, CARS, {"spillage": False, "obstacle_filter": False, "blocked_objects": False},
                                 ground_cam=ss.ground_camera(rec, W, H))
    states = [s["state"] for s in out["slots"]]
    assert out["counts"]["slots"] == len(slots) and "BUSY" in states and "FREE" in states
    assert out["counts"]["busy"] >= 1 and out["vehicles"] == 3


# ---- zones (sample-test items 151-152) ----------------------------------------------------------------------------------------------------
ROW = (to_image(CAM, [(-6, 10), (4, 10), (4, 15), (-6, 15)]) / (W, H)).tolist()  # one row along x: 10 m long (4 bays), a bay's depth deep


def zone_grid(raw=()):
    rec = ss.rectangle_record(FRAME, list(raw), pins=PINS, sides=(2.5, 5.0), polygon=ROW, row_outline=ROW)
    return rec, ss.row_grid(rec, ROW, W, H)[0]


def test_a_zone_is_oriented_by_its_own_outline_not_by_the_cars_of_other_rows():
    crooked = [car(-10, 20), car(9, 22), car(12, 21)]  # cars of another block, standing at another angle, outside the zone
    rec, slots = zone_grid(ss.in_zone(CARS + crooked, ROW, W, H))
    assert rec["capacity"]["orientation_source"] == "zone outline" and abs(rec["capacity"]["orientation_deg"] - 90) < 3  # bays across the row
    assert len(ss.in_zone(CARS + crooked, ROW, W, H)) == 2  # only the cars standing in the zone
    bays, over = ss.zone_bays(slots, 1, ROW, W, H)
    assert over == 0 and len(bays) == 4  # 10 m of row = 4 bays of 2.5 m, one row deep: nothing outside the zone, nothing across it
    from carparking.geometry.camera import to_ground
    cam = ss.ground_camera(rec, W, H)
    for b in bays:
        g = to_ground(cam, np.asarray(b["poly"]) * (W, H))
        assert np.ptp(g[:, 0]) == pytest.approx(2.5, abs=0.1) and np.ptp(g[:, 1]) == pytest.approx(5.0, abs=0.1)
        assert -6.1 <= g[:, 0].min() and g[:, 0].max() <= 4.1 and 9.9 <= g[:, 1].min() and g[:, 1].max() <= 15.1
    cx = [np.asarray(b["poly"]).mean(axis=0)[0] for b in bays]
    assert cx == sorted(cx) and [b["name"] for b in bays] == [f"A{n}" for n in range(1, len(bays) + 1)]  # A1, A2 ... along the row


def test_bays_are_numbered_row_by_row_along_the_zone():
    sq = lambda x, y: [[x, y], [x + 0.05, y], [x + 0.05, y + 0.1], [x, y + 0.1]]  # noqa: E731
    slots = [{"id": i, "name": "", "poly": p} for i, p in enumerate([sq(0.3, 0.6), sq(0.1, 0.3), sq(0.1, 0.6), sq(0.2, 0.3), sq(0.2, 0.6)], 1)]
    outline = [[0.05, 0.25], [0.5, 0.25], [0.5, 0.75], [0.05, 0.75]]
    assert [s["id"] for s in ss.order_along_row(slots, outline, W, H)] == [2, 4, 3, 5, 1]  # the upper row left to right, then the lower one


def test_add_next_bays_continues_the_row_on_the_ground():
    from carparking.geometry.camera import to_ground
    rec, slots = zone_grid(ss.in_zone(CARS, ROW, W, H))
    bays, _ = ss.zone_bays(slots, 1, ROW, W, H)
    more, added = ss.extend_zone(rec, bays, ROW, 1, 2, W, H)
    assert added == 2 and len(more) == len(bays) + 2 and [b["name"] for b in more][-2:] == [f"A{len(bays) + 1}", f"A{len(bays) + 2}"]
    cam = ss.ground_camera(rec, W, H)
    c = [to_ground(cam, np.asarray(b["poly"]) * (W, H)).mean(axis=0) for b in more]
    steps = [np.linalg.norm(c[i + 1] - c[i]) for i in range(len(c) - 1)]
    assert max(steps[:-1]) - min(steps[:-1]) < 0.01 and abs(steps[-2] - 2.5) < 0.01  # the same 2.5 m spacing all along, no gap, no jump
    # (the last one reaches past the picture's right edge: it is cut there, so its centre moves in)
    assert len({b["id"] for b in more}) == len(more) and all(b["id"] // 100 == 1 for b in more)
    far, n = ss.extend_zone(rec, more, ROW, 1, 50, W, H)
    assert 0 < n < 50  # stops where the next bay would leave the picture, and says how many


def test_zone_bays_keep_the_bay_width_whatever_the_resolution_and_a_tall_outline_does_not_turn_them():
    from carparking.geometry.camera import to_ground
    for scale in (1.0, 1672 / 1920):  # the same camera and lot at 1920 x 1080 and at 1672 x 941
        w, h = round(W * scale), round(H * scale)
        cam = CameraModel(f=1500.0 * scale, cx=w / 2, cy=h / 2, pitch=np.radians(32), roll=np.radians(0.5), h=6.0)
        frame = np.full((h, w, 3), 110, np.uint8)
        pins = (to_image(cam, [(-1.0, 9.0), (1.5, 9.0), (1.5, 14.0), (-1.0, 14.0)]) / (w, h)).tolist()
        # a row of 4 bays drawn over the cars: its top edge goes 9 m deep on the ground, deeper than the row is long (the old grid turned the bays)
        tall = (to_image(cam, [(-6, 10), (4, 10), (4, 24), (-6, 24)]) / (w, h)).tolist()
        rec = ss.rectangle_record(frame, [], pins=pins, sides=(2.5, 5.0), polygon=tall, row_outline=tall)
        slots, info = ss.row_grid(rec, tall, w, h)
        gc = ss.ground_camera(rec, w, h)
        assert len(slots) == 4 and info["width_m"] == pytest.approx(2.5, abs=0.01) and info["rows"] == 1
        for s in slots:
            g = to_ground(gc, np.asarray(s["poly"]) * (w, h))
            assert np.ptp(g[:, 0]) == pytest.approx(2.5, abs=0.05) and np.ptp(g[:, 1]) == pytest.approx(5.0, abs=0.05)  # 2.5 m along the row, 5 m deep
            assert g[:, 1].min() == pytest.approx(10.0, abs=0.05)  # from the outline's front edge


def test_a_short_row_is_not_squeezed_and_the_zone_cars_set_the_spacing():
    rec = ss.rectangle_record(FRAME, [], pins=PINS, sides=(2.5, 5.0), polygon=ROW, row_outline=ROW)
    short = (to_image(CAM, [(-6, 10), (-2.3, 10), (-2.3, 15), (-6, 15)]) / (W, H)).tolist()  # 3.7 m: one bay, not two of 1.85 m
    slots, info = ss.row_grid(rec, short, W, H)
    assert len(slots) == 1 and info["width_m"] == 2.5
    cars = [car(x, 10.2)["box"] for x in (-4.6, -1.6, 1.4)]  # parked 3.0 m apart (the paint is wider than the typed 2.5 m)
    slots, info = ss.row_grid(rec, ROW, W, H, cars=cars)
    assert "cars" in info["source"] and info["width_m"] == pytest.approx(3.0, abs=0.3)  # laid from the cars in the picture (billboard cars: ~9 % off)
    from carparking.geometry.camera import to_ground
    gc = ss.ground_camera(rec, W, H)
    spans = sorted((float(g[:, 0].min()), float(g[:, 0].max())) for g in (to_ground(gc, np.asarray(s["poly"]) * (W, H)) for s in slots))
    homes = [next(i for i, (a, b) in enumerate(spans) if a <= x <= b) for x in (-4.6, -1.6, 1.4)]
    assert len(set(homes)) == 3  # each car in a bay of its own
    assert all(min(x - spans[i][0], spans[i][1] - x) > 0.5 for x, i in zip((-4.6, -1.6, 1.4), homes))  # and not on a line
    two = ss.row_grid(rec, ROW, W, H, rows=2)[0]
    assert len(two) == 2 * len(ss.row_grid(rec, ROW, W, H)[0])


# ---- zones laid from their cars, on the two sample pictures (item 154) ----------------------------------------------------------------------
# The detections (yolov8s, conf 0.20) of data/sample/image2.png (1672 x 941) and of the same scene's low-resolution copy
# "Authorities barred from collecting parking fee.png" (640 x 480: image2 resized x 0.385, y 0.459), and the zone the user drew
# (data/sample/aftercalibaration.png). With GeoCalib's camera that zone got 16 metric bays of 2.46 m on the high-res picture and 12 on the
# low-res one, for the 7 cars that stand in it: two bays on one car at the row's far end.
HI = (1672, 941, [[357.1, 681.9, 648.0, 858.8], [1277.7, 522.2, 1478.8, 632.9], [919.9, 592.2, 1204.9, 747.4], [1415.5, 483.7, 1585.6, 567.5],
                  [855.8, 643.3, 1121.4, 826.6], [715.3, 828.0, 1061.2, 938.4], [1495.9, 445.4, 1671.8, 535.6], [1067.7, 591.1, 1295.9, 727.8],
                  [1062.0, 521.0, 1373.0, 684.5], [1551.4, 425.9, 1671.7, 492.1], [1061.1, 521.5, 1372.5, 684.4]])
LO = (640, 480, [[137.3, 338.3, 251.1, 414.1], [275.5, 407.2, 414.9, 479.9], [411.1, 259.8, 529.0, 336.0], [493.2, 259.2, 569.2, 310.1],
                 [352.2, 292.6, 465.5, 364.6], [545.0, 240.1, 611.5, 279.1], [577.2, 221.5, 639.9, 263.4], [414.4, 289.8, 501.6, 354.3],
                 [328.4, 317.5, 433.1, 402.7], [599.8, 210.8, 639.9, 235.1], [328.3, 317.5, 433.1, 402.9], [412.1, 292.3, 501.1, 354.7]])
ZONE_HI = [(749, 726), (1480, 468), (1671, 527), (985, 856)]
to_lo = lambda p: (p[0] * 0.385, p[1] * 0.459 + 23)  # noqa: E731


def sample_zone(pic, zone_px):
    w, h, boxes = pic
    outline = (np.asarray(zone_px, float) / (w, h)).tolist()
    cars = [d["box"] for d in ss.in_zone([{"box": b} for b in boxes], outline, w, h)]
    return ss.car_row_grid(outline, w, h, cars), outline, cars


def test_a_zone_on_the_sample_pictures_gets_one_bay_per_car_at_both_resolutions():
    (hi, info_hi), _, cars_hi = sample_zone(HI, ZONE_HI)
    (lo, info_lo), _, _ = sample_zone(LO, [to_lo(p) for p in ZONE_HI])
    assert len(hi) == len(lo) == 7 and info_hi["bays_along"] == 7  # 7 cars in the zone: SUV, silver van, white car, white van, hatchback, red car, silver car
    # the same bays at both resolutions: the low-res ones are the high-res ones resized, within a third of a bay at the far end (the two pictures'
    # detections differ, and the low-res one is resized out of aspect)
    for a, b in zip(hi, lo):
        pa = np.asarray(a["poly"]) * (HI[0], HI[1])
        pb = np.asarray(b["poly"]) * (LO[0], LO[1])
        mapped = np.array([to_lo(p) for p in pa])
        assert np.abs(mapped - pb).max() < 0.35 * np.linalg.norm(pb[1] - pb[0]) + 1.0
    # every car's front (where its box meets the zone's front edge) falls in a bay of its own
    (f0, f1), _ = ss._outline_sides(np.asarray(ZONE_HI, float))
    fronts = sorted({min(range(7), key=lambda i: abs(np.mean(np.asarray(hi[i]["poly"])[:2] * (HI[0], HI[1]), axis=0)[0] - ss.car_position(c)[0]))
                     for c in cars_hi})
    assert len(fronts) >= 6  # (the white car behind the silver van is half hidden: its box is a sliver of it)


def test_bay_edges_from_cars_ignore_how_far_off_each_car_position_is_and_find_empty_bays():
    ts = [0.1, 0.3, 0.5, 0.7]
    e = ss.anchored_edges(ts)
    assert np.allclose(np.diff(e), 0.2) and e[0] == 0.0 and len(e) == 6  # bays from the outline's corner, as wide as the cars are apart
    assert np.allclose(ss.anchored_edges([t + 0.07 for t in ts]), e)  # the same bias on every car cancels
    assert len(ss.anchored_edges([0.1, 0.3, 0.301, 0.5, 0.7])) == 6  # one car seen twice is one car
    gap = ss.anchored_edges([0.1, 0.3, 0.7, 0.9])  # 0.3 -> 0.7: one empty bay between two cars
    assert np.allclose(np.diff(gap), 0.2)
    shrink = ss.anchored_edges([0.1, 0.32, 0.5, 0.66, 0.8])  # perspective: the spacing shrinks along the row, and so do the bays
    assert np.diff(shrink)[0] > np.diff(shrink)[-1]
    assert ss.anchored_edges([0.4]) == []  # one car: the metric grid lays the zone


def test_add_next_continues_bays_laid_from_the_cars_in_the_picture():
    (bays, _), outline, _ = sample_zone(HI, ZONE_HI)
    zb, _ = ss.zone_bays(bays, 1, outline, HI[0], HI[1])
    rec = ss.rectangle_record(np.full((HI[1], HI[0], 3), 110, np.uint8), [], pins=(np.asarray(zb[0]["poly"])).tolist(), sides=(2.5, 5.0), polygon=outline)
    five = zb[:5]
    more, added = ss.extend_zone(rec, five, outline, 1, 1, HI[0], HI[1], in_picture=True)
    assert added == 1 and len(more) == 6
    last, new, sixth = (np.asarray(x["poly"]) * (HI[0], HI[1]) for x in (five[-1], more[-1], zb[5]))
    assert np.allclose(new[0], last[1], atol=1.0)  # it shares the last bay's line
    assert np.abs(new - sixth).max() < 0.3 * np.linalg.norm(sixth[1] - sixth[0])  # and lands where the cars put the sixth bay, within a third of a bay


# ---- the zone's own metric frame (item 155): outline + focal length + typed reference ------------------------------------------------------
GEO = {"hi": (1360.0, [0.09165, -0.96959, -0.22692], -0.0442), "lo": (610.4, [0.07388, -0.95786, -0.27757], -0.2324)}  # GeoCalib on the two pictures
REF_HI = [(883, 746), (1017, 803)]  # the user's reference: the SUV's wheelbase, typed 2.6 m


def sample_frame(which):
    from carparking.geometry.calib import camera_from_geocalib
    pic, zone_px, ref_px = (HI, ZONE_HI, REF_HI) if which == "hi" else (LO, [to_lo(p) for p in ZONE_HI], [to_lo(p) for p in REF_HI])
    w, h, boxes = pic
    zone_px = [(min(x, w - 1.0), y) for x, y in zone_px]
    f, up, k1 = GEO[which]
    cam = camera_from_geocalib(f, w, h, up, k1)
    outline = (np.asarray(zone_px, float) / (w, h)).tolist()
    ref = {"a": [ref_px[0][0] / w, ref_px[0][1] / h], "b": [ref_px[1][0] / w, ref_px[1][1] / h], "metres": 2.6}
    cars = sorted((d["box"] for d in ss.in_zone([{"box": b} for b in boxes], outline, w, h)), key=lambda b: ss.car_position(b)[0])
    return ss.zone_frame(outline, w, h, cam, ref), outline, cars, cam, w, h


def test_the_zone_frame_measures_the_zone_alike_at_both_resolutions():
    hi, lo = sample_frame("hi")[0], sample_frame("lo")[0]
    assert hi is not None and lo is not None
    assert hi.depth_m == pytest.approx(5.1, abs=0.3) and lo.depth_m == pytest.approx(5.2, abs=0.3)  # one bay deep, as drawn
    assert abs(hi.length_m - lo.length_m) / hi.length_m < 0.1  # (the low-res copy is resized out of aspect: its pixels are not square)
    assert hi.bay_t(0.1, 2.5) > hi.bay_t(0.9, 2.5) > 0  # a bay looks shorter farther away


def test_no_zone_gets_bays_far_too_wide_or_narrow_however_many_of_its_cars_are_seen():
    for which in ("hi", "lo"):
        fr, outline, cars, cam, w, h = sample_frame(which)
        widths = {}
        for tag, cs in (("all", cars), ("every other", cars[::2]), ("first and last", [cars[0], cars[-1]]), ("none", [])):
            got = ss.car_row_grid(outline, w, h, cs, 1, cam, fr) or ss.frame_grid(fr, w, h)
            widths[tag] = got[1]["width_m"]
            assert 7 <= got[1]["bays_along"] <= 10, (which, tag, got[1])  # 7 cars stand in the zone; it is ~10 bays of 2.5 m by the frame
        assert ss.car_row_grid(outline, w, h, cars, 1, cam, fr)[1]["bays_along"] == 7  # all its cars seen: one bay per car, none split
        assert all(2.2 <= x <= 3.7 for x in widths.values()), widths  # never two cars in one bay, never one car in two bays (3.6 m: image2's frame is 1.35x long, see the log)


def test_the_frame_lays_a_zone_without_cars():
    fr, outline, _, cam, w, h = sample_frame("hi")
    slots, info = ss.frame_grid(fr, w, h, rows=1)
    assert info["bays_along"] == len(slots) == round(fr.length_m / 2.5) and info["width_m"] == pytest.approx(2.5, abs=0.15)
    slots2, info2 = ss.frame_grid(fr, w, h, rows=2)
    assert len(slots2) == 2 * len(slots) and info2["length_m"] == pytest.approx(fr.depth_m / 2, abs=0.01)


@pytest.mark.parametrize("pitch_deg,width", [(20, 1280), (45, 2560), (80, 640), (45, 640)])
def test_any_angle_and_resolution_gets_true_size_bays_from_the_reference_cars_or_both(pitch_deg, width):
    """A synthetic lot with known truth: a row of 6 bays 2.5 x 5 m (15 x 5 m), seen from 20 deg (oblique) to 80 deg (nearly overhead), 640 to 2560 px
    wide. The reference is a 2.5 m bay edge. Every way of laying the zone must give 6 bays of 2.5 m: no cars, all cars, a car in every other bay,
    only the two end cars."""
    from carparking.geometry.camera import to_ground
    w, h = width, round(width * 9 / 16)
    height = 8.0 if pitch_deg < 60 else 20.0  # (nearly overhead: high enough for the 15 m row to be in the picture)
    cam = CameraModel(f=0.8 * w, cx=w / 2, cy=h / 2, pitch=np.radians(pitch_deg), roll=np.radians(1.0), h=height)
    y0 = height / np.tan(np.radians(pitch_deg)) - 2.5  # the row stands where the camera looks
    ground = [(-7.5, y0), (7.5, y0), (7.5, y0 + 5), (-7.5, y0 + 5)]  # front edge first: the near one
    outline = (to_image(cam, ground) / (w, h)).tolist()
    ref = {"a": (to_image(cam, [(-7.5, y0)])[0] / (w, h)).tolist(), "b": (to_image(cam, [(-5.0, y0)])[0] / (w, h)).tolist(), "metres": 2.5}
    fr = ss.zone_frame(outline, w, h, cam, ref)
    assert fr.length_m == pytest.approx(15.0, rel=0.01) and fr.depth_m == pytest.approx(5.0, rel=0.01)
    def box(i):  # a car standing in bay i: its box in the picture (a 1.8 x 4.5 m footprint, 1.5 m tall)
        x = -7.5 + 2.5 * i + 1.25
        pts = [(x + dx, y0 + 0.25 + dy) for dx in (-0.9, 0.9) for dy in (0.0, 4.5)]
        img = np.vstack([to_image(cam, pts), to_image(cam.with_height(cam.h - 1.5), pts)])
        return (*img.min(axis=0), *img.max(axis=0))
    for cars in ([], [box(i) for i in range(6)], [box(i) for i in (0, 2, 4)], [box(0), box(5)]):
        got = ss.car_row_grid(outline, w, h, cars, 1, cam, fr) or ss.frame_grid(fr, w, h)
        slots, info = got
        assert len(slots) == 6, (pitch_deg, width, len(cars), info)
        for s in slots:
            g = to_ground(cam, np.asarray(s["poly"]) * (w, h))
            assert np.ptp(g[:, 0]) == pytest.approx(2.5, abs=0.3), (pitch_deg, width, len(cars))  # 2.5 m wide
            assert np.ptp(g[:, 1]) == pytest.approx(5.0, abs=0.3)  # 5 m deep


def test_bays_laid_from_the_cars_shrink_steadily_with_distance_like_equal_bays_on_the_ground():
    for which in ("hi", "lo"):
        fr, outline, cars, cam, w, h = sample_frame(which)
        for cs in (cars, cars[::2]):
            slots = ss.car_row_grid(outline, w, h, cs, 1, cam, fr)[0]
            ordered = ss.order_along_row(slots, outline, w, h)
            widths = [float(np.linalg.norm((np.asarray(x["poly"][1]) - x["poly"][0]) * (w, h))) for x in ordered]
            assert all(a > b for a, b in zip(widths, widths[1:])), (which, widths)  # a far bay is narrower than a nearer one, every time
            assert widths[0] / widths[-1] > 1.5  # and by the zone's own perspective, not a token amount
    row = ss.car_row_grid(ROW, W, H, [car(x, 10.2)["box"] for x in (-4.6, -1.6, 1.4)])[0]  # a row at one depth keeps equal bays
    ws = [float(np.linalg.norm((np.asarray(x["poly"][1]) - x["poly"][0]) * (W, H))) for x in row]
    assert max(ws) / min(ws) < 1.25


def test_a_thin_slanted_outline_pairs_its_corners_round_the_outline_not_by_distance():
    # data/sample/camera1.png: a row of bays seen along its length; the near corner of the back edge is farther from the front's first corner than the far one is
    outline = [(700, 830), (1075, 215), (790, 215), (20, 800)]
    (f0, f1), (b0, b1) = ss._outline_sides(np.asarray(outline, float))
    assert f0.tolist() == [700, 830] and f1.tolist() == [1075, 215]
    assert b0.tolist() == [20, 800] and b1.tolist() == [790, 215]  # next to f0 / f1 across the outline: the separators do not cross
    for order in ([outline[0], outline[1], outline[3], outline[2]], [outline[1], outline[0], outline[3], outline[2]]):  # clicked in a bow-tie order
        (g0, g1), (c0, c1) = ss._outline_sides(np.asarray(order, float))
        assert (g0.tolist(), g1.tolist(), c0.tolist(), c1.tolist()) == (f0.tolist(), f1.tolist(), b0.tolist(), b1.tolist())


def test_a_topdown_picture_gets_true_size_bays_from_the_reference_alone():
    w, h = 1448, 1086
    frame = np.full((h, w, 3), 110, np.uint8)
    ref = {"a": [300 / w, 500 / h], "b": [360 / w, 500 / h], "metres": 2.5}  # one painted bay, 60 px wide
    outline = [(300 / w, 500 / h), (900 / w, 500 / h), (900 / w, 620 / h), (300 / w, 620 / h)]
    rec = ss.topdown_record(frame, [], reference=ref, polygon=outline, row_outline=outline)
    assert np.degrees(rec["calibration"]["pitch"]) == pytest.approx(89.5, abs=0.1) and any("top-down" in t for t in rec["warnings"])
    slots, info = ss.row_grid(rec, outline, w, h, reference=ref)
    assert len(slots) == info["bays_along"] == 10 and info["width_m"] == pytest.approx(2.5, abs=0.1)
    from carparking.geometry.camera import to_ground
    cam = ss.ground_camera(rec, w, h)
    for s in slots:
        g = to_ground(cam, np.asarray(s["poly"]) * (w, h))
        assert np.ptp(g[:, 0]) == pytest.approx(2.5, abs=0.2) and np.ptp(g[:, 1]) == pytest.approx(5.0, abs=0.2)  # the same everywhere: a nadir view
    with pytest.raises(ss.SessionError):
        ss.topdown_record(frame, [], reference={**ref, "b": ref["a"]}, polygon=outline)  # two identical points measure nothing


def test_a_zone_two_cars_deep_gets_two_rows_automatically_one_bay_per_car():
    w, h = 1448, 1086
    frame = np.full((h, w, 3), 110, np.uint8)
    ref = {"a": [700 / w, 500 / h], "b": [700 / w, 544 / h], "metres": 2.5}  # a top-down picture: 17.6 px per metre
    deep = [(690 / w, 790 / h), (690 / w, 20 / h), (885 / w, 20 / h), (885 / w, 790 / h)]  # 11 m deep: a column of cars either side of the middle
    rec = ss.topdown_record(frame, [], reference=ref, polygon=deep, row_outline=deep)
    one, info1 = ss.row_grid(rec, deep, w, h, rows=1, reference=ref)
    auto, info0 = ss.row_grid(rec, deep, w, h, rows=0, reference=ref)
    assert info1["rows"] == 1 and info0["rows"] == 2 and len(auto) == 2 * len(one)  # one bay across both cars -> two rows of bays
    from carparking.geometry.camera import to_ground
    cam = ss.ground_camera(rec, w, h)
    depths = [float(np.ptp(to_ground(cam, np.asarray(s["poly"]) * (w, h))[:, 0])) for s in auto]
    assert all(4.5 <= d <= 6.5 for d in depths), depths  # each bay is one car long (it was 11 m)
    narrow = [(690 / w, 790 / h), (690 / w, 20 / h), (780 / w, 20 / h), (780 / w, 790 / h)]  # one car deep (5 m): still one row
    assert ss.row_grid(ss.topdown_record(frame, [], reference=ref, polygon=narrow, row_outline=narrow), narrow, w, h, rows=0, reference=ref)[1]["rows"] == 1
    assert ss.auto_rows(None, 5.0) == 1  # without a reference there is no depth to go by


def _synthetic_row(pitch_deg=35, width=1920, shear=0.0):
    """A 15 x 5 m row seen by a known camera; the outline as DRAWN may be sheared (its back edge shifted sideways by `shear` metres), as a hand-drawn rectangle is."""
    w, h = width, round(width * 9 / 16)
    cam = CameraModel(f=0.8 * w, cx=w / 2, cy=h / 2, pitch=np.radians(pitch_deg), roll=np.radians(1.0), h=8.0)
    y0 = 8.0 / np.tan(np.radians(pitch_deg)) - 2.5
    drawn = (to_image(cam, [(-7.5, y0), (7.5, y0), (7.5 + shear, y0 + 5), (-7.5 + shear, y0 + 5)]) / (w, h)).tolist()
    ref = {"a": (to_image(cam, [(-7.5, y0)])[0] / (w, h)).tolist(), "b": (to_image(cam, [(-5.0, y0)])[0] / (w, h)).tolist(), "metres": 2.5}
    return cam, drawn, ref, w, h


def test_the_outline_follows_the_camera_slope_not_the_drawn_rectangles():
    from carparking.geometry.camera import to_ground
    cam, drawn, ref, w, h = _synthetic_row(shear=2.0)  # the drawn back edge leans 2 m: the bays' sides would lean with it
    fixed = ss.ground_outline(cam, drawn, w, h)
    assert fixed is not None
    g = to_ground(cam, np.asarray(fixed) * (w, h))
    for a, b in ((0, 3), (1, 2)):  # the sides are square to the front edge on the ground
        side = g[b] - g[a]
        assert abs(side[0]) < 0.15 and side[1] == pytest.approx(5.0, abs=0.15), (a, b, side)
    assert np.allclose(np.asarray(fixed)[:2], np.clip(np.asarray(drawn)[:2], 0, 1), atol=1e-3)  # the front edge (on the ground, where the row stands) is kept
    fr = ss.zone_frame(fixed, w, h, cam, ref)
    slots, _ = ss.frame_grid(fr, w, h)
    for s in slots:
        gs = to_ground(cam, np.asarray(s["poly"]) * (w, h))
        assert np.ptp(gs[:, 0]) == pytest.approx(2.5, abs=0.3) and np.ptp(gs[:, 1]) == pytest.approx(5.0, abs=0.3)
    level = CameraModel(f=0.8 * w, cx=w / 2, cy=h / 2, pitch=np.radians(5.0), roll=0.0, h=8.0)  # (its horizon is at 38 % of the picture's height)
    assert ss.ground_outline(level, [[0.1, 0.1], [0.9, 0.1], [0.9, 0.2], [0.1, 0.2]], w, h) is None  # above the horizon: the drawn one is kept


@pytest.mark.parametrize("n", [4, 6, 9, 13])
def test_an_expected_bay_count_is_exact_whatever_the_cars_or_the_zone_length(n):
    cam, outline, ref, w, h = _synthetic_row()
    fr = ss.zone_frame(outline, w, h, cam, ref)
    (f0, f1), (b0, b1) = ss._outline_sides(np.asarray(outline) * (w, h))
    for frame in (fr, None):  # (without a reference: the outline's own perspective)
        edges = ss.count_edges(frame, f0, f1, b0, b1, n)
        slots, info = ss.car_row_grid(outline, w, h, [], 1, cam, frame, edges=edges)
        assert len(slots) == n and info["bays_along"] == n and info["source"] == "the expected bay count"
        widths = [float(np.linalg.norm((np.asarray(s["poly"])[1] - np.asarray(s["poly"])[0]) * (w, h))) for s in slots]
        widths = widths[1:-1]  # (the end bays are cut by the picture's edge)
        steps = np.diff(widths)
        assert (steps <= 0.005 * max(widths)).all() or (steps >= -0.005 * max(widths)).all()  # steadily changing with the perspective, no jumps
