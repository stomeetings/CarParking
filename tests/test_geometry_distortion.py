"""Radial lens distortion in the geometry modules (camera.py, calib.py, pipeline.py): a synthetic camera shaped like the live
camera-1 (GeoCalib simple_radial: f ~ 1977 px, k1 ~ -0.46, 14 deg down) with known height, so the metres are checkable."""
import cv2
import numpy as np
import pytest

from carparking.geometry import CameraModel, to_ground, to_image
from carparking.geometry.bev import area, surface_polygon
from carparking.geometry.calib import (camera_from_geocalib, height_from_depth_map, height_from_reference,
                                       height_from_vehicles)
from carparking.geometry.camera import distort_n, ground_resolution, undistort_n
from carparking.geometry.pipeline import build_record

W, H_PX = 1920, 1080
CAM = CameraModel(f=1977.0, cx=W / 2, cy=H_PX / 2, pitch=np.radians(14), roll=np.radians(0.8), h=4.0, k1=-0.46)
PINHOLE = CameraModel(CAM.f, CAM.cx, CAM.cy, CAM.pitch, CAM.roll, CAM.h)  # the same camera, lens ignored
GROUND = [(-3, 9), (3, 9), (8, 25), (-8, 25)]


def dense(ground, n=60):
    """The polygon's edges sampled finely, as a SAM contour is: a straight ground edge is a curve in a barrel image, so
    four corner pixels joined by straight lines would not be the same polygon."""
    g = np.array(ground, float)
    return np.vstack([a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None] for a, b in zip(g, np.roll(g, -1, axis=0))])


def up_of(cam):
    sp, cp, sr, cr = np.sin(cam.pitch), np.cos(cam.pitch), np.sin(cam.roll), np.cos(cam.roll)
    return -np.array([cp * sr, cp * cr, sp])


def car_box(cam, x, y, half=0.9, height=1.45):
    base = to_image(cam, [(x - half, y), (x + half, y)])
    top = to_image(cam.with_height(cam.h - height), [(x - half, y), (x + half, y)])
    pts = np.vstack([base, top])
    return (*pts.min(axis=0), *pts.max(axis=0))


def test_distortion_inverse_is_exact_and_flags_the_unrepresentable():
    rng = np.random.default_rng(1)
    for k1, k2 in ((-0.46, 0.0), (-0.3, 0.05), (0.2, 0.0), (-0.1, -0.05), (0.0, 0.0)):
        ideal = rng.uniform(-0.45, 0.45, (500, 2))
        back = undistort_n(distort_n(ideal, k1, k2), k1, k2)
        assert np.abs(back - ideal).max() < 1e-9, (k1, k2)
    # k1 = -0.46: the distorted radius can't exceed ~0.567 (the model folds back): beyond it there is no inverse
    inside, outside = undistort_n([[0.5, 0.0]], -0.46)[0], undistort_n([[0.9, 0.0]], -0.46)[0]
    assert not np.isnan(inside).any() and np.isnan(outside).all()
    assert np.array_equal(undistort_n([[0.3, 0.2]], 0.0, 0.0), [[0.3, 0.2]])


def test_camera_undistort_matches_geocalib_forward_model():
    # the forward model is p * (1 + k1 r^2) on normalised coordinates, then * f + c: one hand-computed pixel
    ideal = np.array([[CAM.cx + 600.0, CAM.cy - 300.0]])
    pn = np.array([[600.0, -300.0]]) / CAM.f
    expect = pn * (1 - 0.46 * (pn ** 2).sum()) * CAM.f + (CAM.cx, CAM.cy)
    assert CAM.distort(ideal) == pytest.approx(expect)
    assert CAM.undistort(CAM.distort(ideal)) == pytest.approx(ideal, abs=1e-6)
    assert CAM.with_height(2.0).k1 == -0.46 and not PINHOLE.distorted and CAM.distorted


def test_ground_roundtrip_through_a_distorted_lens():
    g = np.array([(-3.0, 10.0), (4.0, 14.0), (0.0, 20.0), (-6.0, 24.0)])
    px = to_image(CAM, g)
    assert np.allclose(to_ground(CAM, px), g, atol=1e-6)
    # the same pixels read as a pinhole camera land somewhere else: ignoring the lens skews the ground
    assert np.abs(to_ground(PINHOLE, px) - g).max() > 0.25


def test_horizon_and_the_ground_mask_with_a_barrel_lens():
    us = np.linspace(100, W - 100, 5)
    curve = CAM.horizon_px(us)  # the horizon in the distorted image: bent, not a straight row
    assert not np.isnan(curve).any() and np.ptp(curve[:, 1]) > 3
    for (u, v) in curve[[0, 2, 4]]:
        above, below = to_ground(CAM, [(u, v - 25)]), to_ground(CAM, [(u, v + 25)])
        assert np.isnan(above).all() and not np.isnan(below).any()  # above the horizon: no ground; just below: ground
    far = ground_resolution(CAM, [(960, curve[2, 1] + 20)])[0]
    near = ground_resolution(CAM, [(960, 1000)])[0]
    assert far > 10 * near  # a pixel covers far more ground at the horizon


def test_surface_area_is_not_skewed_by_the_lens():
    poly_px = to_image(CAM, dense(GROUND))
    truth = area(np.array(GROUND, float))
    with_lens = area(surface_polygon(CAM, poly_px, max_range=80, max_res=1.0))
    ignoring = area(surface_polygon(PINHOLE, poly_px, max_range=80, max_res=1.0))
    assert with_lens == pytest.approx(truth, rel=0.02)
    assert abs(ignoring - truth) > 3 * abs(with_lens - truth) and abs(ignoring - truth) / truth > 0.03


def test_camera_height_from_cars_survives_distortion():
    cars = [car_box(CAM, x, y) for x, y in [(-3, 10), (1, 13), (4, 16), (-5, 19), (2, 22), (-1, 14)]]
    assert height_from_vehicles(CAM.with_height(1.0), cars)["h"] == pytest.approx(4.0, rel=0.05)
    wrong = height_from_vehicles(PINHOLE.with_height(1.0), cars)  # lens ignored: not trusted to be right
    assert wrong is None or abs(wrong["h"] - 4.0) > abs(height_from_vehicles(CAM.with_height(1.0), cars)["h"] - 4.0)


def test_depth_and_reference_scale_under_distortion():
    shape = (270, 480)  # a quarter-resolution depth map of a flat ground seen through the lens
    small = CameraModel(CAM.f / 4, shape[1] / 2, shape[0] / 2, CAM.pitch, CAM.roll, CAM.h, CAM.k1)
    uu, vv = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
    g = to_ground(small, np.c_[uu.ravel(), vv.ravel()])
    cz = (np.c_[g, np.full(len(g), -small.h)] @ small.R_cw())[:, 2]
    dmap = np.where(np.isnan(cz), 0, cz).reshape(shape)
    mask = np.zeros(shape, np.uint8)
    mask[110:260, 20:460] = 255
    got = height_from_depth_map(dmap, mask, small.f, up_of(CAM), k1=CAM.k1)
    assert got["h"] == pytest.approx(4.0, rel=0.03)
    a, b = to_image(CAM, [(-4.5, 11), (4.5, 11)])  # 9 m apart, out towards the image edges where the lens bends most
    assert height_from_reference(CAM.with_height(1.0), a, b, 9.0) == pytest.approx(4.0, rel=0.01)
    assert abs(height_from_reference(PINHOLE.with_height(1.0), a, b, 9.0) - 4.0) > 0.1


def test_geocalib_conversion_carries_the_lens():
    cam = camera_from_geocalib(1977.0, W, H_PX, up_of(CAM), k1=-0.46, k2=0.0)
    assert (cam.k1, cam.k2) == (-0.46, 0.0) and cam.pitch == pytest.approx(CAM.pitch)


def test_build_record_with_a_wide_angle_lens():
    cars = [car_box(CAM, x, y) for x, y in [(-3, 10), (1, 13), (4, 16), (-5, 19), (2, 22)]]
    poly_n = (to_image(CAM, dense(GROUND)) / (W, H_PX)).tolist()
    kw = dict(width=W, height=H_PX, polygon=poly_n, f=CAM.f, up=up_of(CAM), depth={"h": 4.1}, vehicles=cars)
    rec = build_record(**kw, k1=-0.46, lens="simple_radial")
    cal, truth = rec["calibration"], area(np.array(GROUND, float))
    assert cal["lens"] == "simple_radial" and cal["k1"] == -0.46 and cal["k2"] == 0.0
    assert cal["h"] == pytest.approx(4.0, rel=0.05) and cal["confidence"] == "MEDIUM"  # two agreeing priors
    assert cal["scale_uncertainty"] == 0.25 and len(rec["warnings"]) == 1 and "priors" in rec["warnings"][0]
    # f = 1977 px alone reads as 2 atan(960 / f) = 52 deg; with the barrel distortion undone the ideal image is wider
    assert cal["hfov_deg"] == pytest.approx(59.5, abs=1)
    assert rec["bev"]["area_m2"] == pytest.approx(truth, rel=0.05)
    slots = rec["capacity"]["slots"]
    assert len(slots) >= 4
    truth_poly = np.float32(GROUND).reshape(-1, 1, 2)
    for s_ in slots:  # every packed bay lies inside the true ground polygon (0.3 m tolerance for the polygon's sampling)
        assert all(cv2.pointPolygonTest(truth_poly, (float(x), float(y)), True) > -0.3 for x, y in s_["poly_m"])
    # the stored calibration rebuilds the camera for the worker (its h is the ESTIMATED 4.03 m, not exactly the true 4.0)
    stored = CameraModel(cal["f"], cal["cx"], cal["cy"], cal["pitch"], cal["roll"], cal["h"], cal["k1"], cal["k2"])
    for s in slots:  # pre-rendered bays are in the DISTORTED image: back through the lens they sit on the ground
        px = np.array(s["poly"]) * (W, H_PX)
        assert np.isfinite(px).all() and px[:, 0].min() > -50 and px[:, 0].max() < W + 50
        assert to_ground(stored, px) == pytest.approx(np.array(s["poly_m"]), abs=0.02)
    # without the lens the same inputs give a worse draft: the area is off and the scale sources disagree more
    flat = build_record(**kw)
    assert abs(flat["bev"]["area_m2"] - truth) > 1.3 * abs(rec["bev"]["area_m2"] - truth)
