"""Phase 2 setup pipeline pieces (src/carparking/geometry: calib helpers, surface, pipeline) and the job board, on a synthetic
camera with known height / tilt. No models, no live data."""
import asyncio
import json

import cv2
import numpy as np
import pytest

from carparking.geometry import CameraModel, to_ground, to_image
from carparking.geometry.calib import (camera_from_geocalib, height_from_depth, height_from_depth_map,
                                       height_from_reference)
from carparking.geometry.pipeline import SetupError, build_record, usable_vehicle_boxes
from carparking.geometry.surface import mask_to_polygon
from carparking.setup_jobs import JobBoard

W, H_PX = 1920, 1080
CAM = CameraModel(f=1500.0, cx=W / 2, cy=H_PX / 2, pitch=np.radians(32), roll=np.radians(0.5), h=6.0)
SURFACE_M = [(-4, 8), (4, 8), (12, 24), (-12, 24)]  # a trapezoid the camera sees whole, 256 m2


def up_of(cam):
    """The camera-frame UP vector GeoCalib reports for this camera (the inverse of camera_from_geocalib)."""
    sp, cp, sr, cr = np.sin(cam.pitch), np.cos(cam.pitch), np.sin(cam.roll), np.cos(cam.roll)
    return -np.array([cp * sr, cp * cr, sp])


def car_box(cam, x, y, half=0.9, height=1.45):
    base = to_image(cam, [(x - half, y), (x + half, y)])
    top = to_image(cam.with_height(cam.h - height), [(x - half, y), (x + half, y)])
    pts = np.vstack([base, top])
    return (*pts.min(axis=0), *pts.max(axis=0))


def surface_n(cam=CAM, ground=SURFACE_M):
    return (to_image(cam, ground) / (W, H_PX)).tolist()


def test_geocalib_up_vector_gives_the_camera():
    cam = camera_from_geocalib(1500.0, W, H_PX, up_of(CAM))
    assert cam.f == 1500.0 and (cam.cx, cam.cy) == (960, 540)
    assert cam.pitch == pytest.approx(CAM.pitch) and cam.roll == pytest.approx(CAM.roll) and cam.h == 1.0
    level = camera_from_geocalib(1500.0, W, H_PX, (0, -1, 0))  # a level camera: up is -y, no tilt
    assert level.pitch == pytest.approx(0) and level.roll == pytest.approx(0)
    assert camera_from_geocalib(1500.0, W, H_PX, (0, -0.5, -0.866)).pitch > np.radians(59)  # looking down: up has -z


def depth_map(cam, shape=(270, 480)):
    """The metric depth (along the optical axis) a perfect model gives for a flat ground, at a quarter of the resolution."""
    small = CameraModel(cam.f / 4, shape[1] / 2, shape[0] / 2, cam.pitch, cam.roll, cam.h)
    uu, vv = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
    g = to_ground(small.homography(), np.c_[uu.ravel(), vv.ravel()])
    cz = (np.c_[g, np.full(len(g), -cam.h)] @ small.R_cw())[:, 2]  # camera-z of each ground point
    return small, np.where(np.isnan(cz), 0, cz).reshape(shape)


def test_height_from_a_metric_depth_map():
    small, dmap = depth_map(CAM)
    below = (np.arange(dmap.shape[0])[:, None] > small.horizon_v() + 20) & np.ones(dmap.shape, bool)
    got = height_from_depth_map(dmap, below.astype(np.uint8) * 255, small.f, up_of(CAM))
    assert got["h"] == pytest.approx(6.0, rel=0.02) and got["n"] > 1000 and got["spread"] < 0.05
    assert height_from_depth_map(dmap, np.zeros(dmap.shape), small.f, up_of(CAM)) is None  # empty mask
    assert height_from_depth(np.zeros((50, 3)), up_of(CAM)) is None  # too few points


def test_height_from_a_typed_distance():
    unit = CAM.with_height(1.0)
    a, b = to_image(CAM, [(-3, 12), (2, 15)])  # known: 5.83 m apart
    assert height_from_reference(unit, a, b, float(np.hypot(5, 3))) == pytest.approx(6.0, rel=0.01)
    assert height_from_reference(unit, a, a, 5.0) is None and height_from_reference(unit, a, b, 0) is None


def test_mask_to_polygon_fills_cars_and_keeps_the_biggest_piece():
    mask = np.zeros((200, 400), np.uint8)
    mask[100:190, 20:380] = 255  # the lot
    mask[130:170, 100:160] = 0  # a car standing on it: SAM leaves a hole
    mask[10:20, 10:20] = 255  # a stray speck
    car = (95, 120, 165, 172)  # its box: the bottom strip lies on the lot
    out = mask_to_polygon(mask, [car])
    assert out["area_frac"] == pytest.approx(360 * 90 / (400 * 200), abs=0.02)
    xs, ys = np.array(out["polygon"]).T
    assert 0 <= xs.min() and xs.max() <= 1 and 0 <= ys.min() and ys.max() <= 1 and ys.min() > 0.45  # speck dropped
    poly = np.round(np.array(out["polygon"]) * (400, 200)).astype(np.int32)
    filled = np.zeros_like(mask)
    cv2.fillPoly(filled, [poly], 255)
    assert filled[150, 130] == 255  # the car's spot is part of the surface
    elsewhere = mask_to_polygon(mask, [(100, 20, 160, 60)])  # a box in the sky: nothing filled, still a valid polygon
    assert elsewhere is not None and elsewhere["area_frac"] < out["area_frac"] + 0.005
    assert mask_to_polygon(np.zeros((50, 50), np.uint8)) is None
    tiny = np.zeros((200, 400), np.uint8)
    tiny[5:8, 5:8] = 255
    assert mask_to_polygon(tiny) is None


def test_mask_to_polygon_bridges_a_pole_but_not_a_wide_gap():
    mask = np.zeros((200, 400), np.uint8)
    mask[100:190, 20:190] = 255
    mask[100:190, 200:380] = 255  # a 10 px pole between two pieces of pavement (2.5 % of the width, the bridge is 3.5 %)
    joined = mask_to_polygon(mask)
    assert joined["area_frac"] == pytest.approx((360 * 90) / (400 * 200), abs=0.03)  # one surface, the pole filled
    apart = np.zeros((200, 400), np.uint8)
    apart[100:190, 20:150] = 255
    apart[100:190, 260:380] = 255  # 110 px apart: two separate places, the larger piece only
    assert mask_to_polygon(apart)["area_frac"] < 0.2


def test_usable_vehicle_boxes():
    dets = [{"box": (200, 300, 400, 420), "conf": 0.9}, {"box": (0, 300, 200, 420), "conf": 0.9},  # cut by the left edge
            {"box": (500, 300, 700, 420), "conf": 0.2}, {"box": (900, 300, 920, 310), "conf": 0.9},  # weak, tiny
            {"box": (900, 300, 1100, 420), "conf": 0.9, "kind": "other"}]
    assert usable_vehicle_boxes(dets, W, H_PX) == [(200, 300, 400, 420)]


def test_build_record_end_to_end():
    cars = [car_box(CAM, x, y) for x, y in [(-3, 11), (1, 13), (5, 16), (-6, 18), (2, 21)]]
    rec = build_record(width=W, height=H_PX, polygon=surface_n(), f=1500.0, up=up_of(CAM), depth={"h": 6.2}, vehicles=cars,
                       meta={"score": 0.93}, pins={"sam2": {"commit": "x"}})
    cal, bev, cap = rec["calibration"], rec["bev"], rec["capacity"]
    assert cal["h"] == pytest.approx(6.0, rel=0.04) and cal["confidence"] == "MEDIUM" and cal["vehicles_used"] == 5
    assert set(cal["scale_sources"]) == {"vehicles", "depth"} and cal["pins"] == {"sam2": {"commit": "x"}}
    assert np.degrees(cal["pitch"]) == pytest.approx(32, abs=0.01) and len(cal["H"]) == 3
    assert bev["area_m2"] == pytest.approx(256, rel=0.04)
    assert cap["geometric_capacity"] == len(cap["slots"]) >= 6 and cap["bay"] == [2.5, 5.0]  # one double row fits 16 m
    surface = np.float32(bev["surface_m"]).reshape(-1, 1, 2)
    for slot in cap["slots"]:  # every packed bay lies inside the measured surface (a corner may nick a slanted edge by < 0.4 m)
        assert all(cv2.pointPolygonTest(surface, (float(x), float(y)), True) > -0.4 for x, y in slot["poly_m"])
    for slot in cap["slots"]:  # pre-rendered slots: normalised pixel polygons inside the image, and their metres
        px = np.array(slot["poly"])
        assert px.shape == (4, 2) and px.min() >= -0.05 and px.max() <= 1.05 and len(slot["poly_m"]) == 4
    assert rec["surface"]["score"] == 0.93 and rec["image"] == {"width": W, "height": H_PX}
    assert len(rec["warnings"]) == 1 and "reference distance" in rec["warnings"][0]  # the scale is from priors
    json.dumps(rec)  # storable
    # the stored homography maps the polygon's corner pixels back to the ground we started from
    back = to_ground(np.array(cal["H"]), (np.array(surface_n()) * (W, H_PX))[:1])
    assert back[0] == pytest.approx(np.array(SURFACE_M[0]), abs=0.3)


def test_implausible_field_of_view_is_flagged_not_trusted_silently():
    # GeoCalib on a wide-angle lens (live camera-1, 2026-10-23): pinhole f = 5500 px = 20 deg: the draft says so
    narrow = CameraModel(5500.0, W / 2, H_PX / 2, np.radians(35), 0.0, 30.0)  # high up, so a 20 deg view still sees ground
    rec = build_record(width=W, height=H_PX, polygon=(to_image(narrow, [(-5, 36), (5, 36), (6, 48), (-6, 48)]) / (W, H_PX)).tolist(),
                       f=5500.0, up=up_of(narrow), depth={"h": 30.0})
    assert rec["calibration"]["hfov_deg"] == pytest.approx(19.8, abs=0.1)
    assert any("field of view" in w and "20 deg" in w for w in rec["warnings"])
    sane = build_record(width=W, height=H_PX, polygon=surface_n(), f=1500.0, up=up_of(CAM), depth={"h": 6.0})
    assert not any("field of view" in w for w in sane["warnings"])


def test_build_record_scale_sources_and_confidence():
    kw = dict(width=W, height=H_PX, polygon=surface_n(), f=1500.0, up=up_of(CAM))
    depth_only = build_record(**kw, depth={"h": 6.0})
    assert depth_only["calibration"]["confidence"] == "LOW" and depth_only["calibration"]["h"] == 6.0
    assert any("one prior" in w for w in depth_only["warnings"]) and depth_only["calibration"]["scale_uncertainty"] > 0.25
    disagree = build_record(**kw, depth={"h": 9.5}, vehicles=[car_box(CAM, x, y) for x, y in [(-3, 11), (1, 13), (5, 16)]])
    assert disagree["calibration"]["confidence"] == "LOW" and any("disagree" in w for w in disagree["warnings"])
    a, b = to_image(CAM, [(-3, 12), (2, 15)]) / (W, H_PX)
    typed = build_record(**kw, depth={"h": 9.5}, reference={"a": a.tolist(), "b": b.tolist(), "metres": float(np.hypot(5, 3))})
    assert typed["calibration"]["h"] == pytest.approx(6.0, rel=0.01) and typed["calibration"]["confidence"] == "HIGH"


@pytest.mark.parametrize("change,message", [
    ({"up": (0, -1, 0)}, "too level"),
    ({"polygon": [[0.1, 0.1], [0.9, 0.1]]}, "at least 3 points"),
    ({"polygon": [[0.1, 0.1], [1.4, 0.1], [0.9, 0.9]]}, "between 0 and 1"),
    ({"depth": None}, "no scale source"),
    ({"depth": {"h": 150.0}}, "outside"),
    ({"polygon": [[0.45, 0.8], [0.55, 0.8], [0.5, 0.85]], "depth": {"h": 6.0}}, "no measurable ground"),  # a sliver
    ({"polygon": (to_image(CAM, [(-2, 10), (2, 10), (2, 14), (-2, 14)]) / (W, H_PX)).tolist(), "depth": {"h": 6.0}}, "too small"),
])
def test_build_record_refuses_what_cannot_be_measured(change, message):
    kw = dict(width=W, height=H_PX, polygon=surface_n(), f=1500.0, up=up_of(CAM), depth={"h": 6.0}) | change
    with pytest.raises(SetupError, match=message):
        build_record(**kw)


def test_an_outline_drawn_over_the_parked_cars_is_measured_as_drawn_instead_of_refused():
    kw = dict(width=W, height=H_PX, polygon=surface_n(), f=1500.0, up=up_of(CAM), depth={"h": 6.0})
    rec = build_record(**kw, obstacles=[(0, 0, W, H_PX)])  # vehicle boxes that cover the whole outline: cutting them out leaves nothing
    assert rec["bev"]["area_m2"] == pytest.approx(256, rel=0.05) and any("mostly over parked vehicles" in w for w in rec["warnings"])
    # a normal outline with cars is unchanged: no such warning
    cars = [car_box(CAM, x, y) for x, y in [(-3, 11), (1, 13), (5, 16)]]
    assert not any("mostly over parked vehicles" in w for w in build_record(**kw, vehicles=cars, obstacles=cars)["warnings"])


def test_the_no_ground_error_says_where_the_outline_went_wrong():
    # above the horizon: the outline is the sky
    with pytest.raises(SetupError, match=r"no measurable ground.*100% are at or above the horizon"):
        level = CameraModel(f=1500.0, cx=W / 2, cy=H_PX / 2, pitch=np.radians(5), roll=0.0, h=6.0)  # its horizon is at y = 409 px
        build_record(width=W, height=H_PX, polygon=[[0.1, 0.05], [0.9, 0.05], [0.9, 0.2], [0.1, 0.2]], f=1500.0, up=up_of(level), depth={"h": 6.0})
    # a sliver on the ground: measurable, but too thin
    with pytest.raises(SetupError, match=r"measurable\. The measurable part is cut into slivers"):
        build_record(width=W, height=H_PX, polygon=[[0.45, 0.8], [0.55, 0.8], [0.5, 0.85]], f=1500.0, up=up_of(CAM), depth={"h": 6.0})


def test_job_board_lifecycle():
    async def go():
        board, gate = JobBoard(keep=3), asyncio.Event()

        async def slow(report):
            report("working", 0.4)
            await gate.wait()
            return {"ok": 1}

        async def bad(report):
            raise ValueError("readable for the Admin")

        async def crash(report):
            raise KeyError("boom")

        job = board.start("segment", "cam", slow)
        await asyncio.sleep(0.02)
        assert board.running("segment", "cam")["id"] == job["id"] and board.running("calibrate", "cam") is None
        assert (job["state"], job["stage"], job["progress"]) == ("running", "working", 0.4)
        assert "task" not in JobBoard.view(job)
        gate.set()
        await asyncio.sleep(0.02)
        assert (job["state"], job["result"], job["progress"]) == ("done", {"ok": 1}, 1.0) and board.running("segment", "cam") is None
        e1, e2 = board.start("x", "cam", bad), board.start("x", "cam", crash)
        await asyncio.sleep(0.02)
        assert (e1["state"], e1["error"]) == ("error", "readable for the Admin")
        assert e2["state"] == "error" and e2["error"].startswith("KeyError")
        for _ in range(4):  # finished jobs beyond `keep` are forgotten, newest kept
            board.start("y", "cam", bad)
        await asyncio.sleep(0.02)
        assert len(board.jobs) == 3 and board.get(job["id"]) is None

    asyncio.run(go())


def test_the_sample_zone_options_leave_the_live_auto_grid_record_exactly_as_it_was():
    # Camera 1's live auto grid calls build_record without row_outline / min_surface_m2: the record must be the same with their defaults spelled out,
    # its bays oriented by the vehicles / the surface as before, and a small surface still refused; only a sample-test zone passes them.
    cars = [car_box(CAM, x, y) for x, y in [(-3, 11), (1, 13), (5, 16), (-6, 18), (2, 21)]]
    kw = dict(width=W, height=H_PX, polygon=surface_n(), f=1500.0, up=up_of(CAM), depth={"h": 6.2}, vehicles=cars, obstacles=cars)
    plain = build_record(**kw)
    assert build_record(**kw, row_outline=None, min_surface_m2=20.0) == plain
    assert plain["capacity"]["orientation_source"] in ("vehicles", "polygon")
    small = dict(kw, polygon=(to_image(CAM, [(-2, 10), (2, 10), (2, 14), (-2, 14)]) / (W, H_PX)).tolist(), vehicles=(), obstacles=None, depth={"h": 6.0})
    with pytest.raises(SetupError, match="too small"):
        build_record(**small)
    zone = build_record(**small, row_outline=small["polygon"], min_surface_m2=0.0)  # a zone: its own outline orients it, no lot-size rule
    assert zone["capacity"]["orientation_source"] == "zone outline"
