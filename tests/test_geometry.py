"""Phase 2 geometry core (src/carparking/geometry): synthetic cameras with known height / pitch, no live data."""
import numpy as np
import pytest

from carparking.geometry import (CameraModel, area, fuse_scales, ground_resolution, height_from_box,
                                 height_from_vehicles, nearest_pairs, pack_capacity, poly_distance, surface_polygon,
                                 to_ground, to_image, to_pixel_polys)
from carparking.geometry.bev import footprint
from carparking.geometry.capacity import dominant_angle

CAM = CameraModel(f=1200.0, cx=960.0, cy=540.0, pitch=np.radians(30), roll=np.radians(1.5), h=6.0)


def car_box(cam, x, y, half=0.9, height=1.45):
    """The pixel box of a car-sized upright billboard standing at ground point (x, y)."""
    base = to_image(cam, [(x - half, y), (x + half, y)])
    top = to_image(cam.with_height(cam.h - height), [(x - half, y), (x + half, y)])
    pts = np.vstack([base, top])
    return (*pts.min(axis=0), *pts.max(axis=0))


def test_image_ground_roundtrip():
    px = np.array([[300, 700], [960, 900], [1700, 1000]], float)
    g = to_ground(CAM.homography(), px)
    assert np.allclose(to_image(CAM, g), px, atol=1e-6)


def test_scale_is_linear_in_height():
    px = [(800, 800), (1200, 950)]
    g1 = to_ground(CAM.with_height(1.0).homography(), px)
    assert np.allclose(to_ground(CAM.with_height(6.0).homography(), px), 6.0 * g1)


def test_above_the_horizon_is_not_on_the_ground():
    hz = CAM.horizon_v(960)
    g = to_ground(CAM.homography(), [(960, hz - 30), (960, hz + 80)])
    assert np.isnan(g[0]).all() and not np.isnan(g[1]).any()


def test_a_pixel_covers_more_ground_near_the_horizon():
    hz = CAM.horizon_v(960)
    near, far = ground_resolution(CAM.homography(), [(960, 1050), (960, hz + 40)])
    assert far > 5 * near


def test_height_recovered_from_cars():
    boxes = [car_box(CAM, x, y) for x, y in [(-4, 12), (2, 15), (6, 20), (-8, 18), (0, 25)]]
    got = height_from_vehicles(CAM.with_height(1.0), boxes)
    assert got["n"] == 5 and got["h"] == pytest.approx(6.0, rel=0.03)


def test_height_ignores_outlier_and_needs_enough_cars():
    boxes = [car_box(CAM, x, y) for x, y in [(-4, 12), (2, 15), (6, 20), (-8, 18)]]
    boxes.append((900, 700, 1000, 705))  # a sliver: absurd height
    assert height_from_vehicles(CAM, boxes)["h"] == pytest.approx(6.0, rel=0.03)
    assert height_from_vehicles(CAM, boxes[:2]) is None
    assert height_from_box(CAM, (900, -420, 1000, -300)) is None  # bottom edge above the horizon (row -153)


def test_fuse_scales():
    assert fuse_scales({})["confidence"] == "NONE"
    assert fuse_scales({"vehicles": 6.0})["confidence"] == "LOW"  # one prior alone
    both = fuse_scales({"vehicles": 6.0, "depth": 6.5})
    # two priors that agree: consistent, not proven accurate (measured on the live camera: both 17-21 % low)
    assert both["confidence"] == "MEDIUM" and both["h"] == pytest.approx(6.25) and both["uncertainty"] == 0.25
    assert fuse_scales({"vehicles": 6.0, "depth": 9.0})["confidence"] == "LOW"
    typed = fuse_scales({"vehicles": 6.0, "depth": 9.0, "manual": 5.0})
    assert typed["h"] == 5.0 and typed["confidence"] == "HIGH" and typed["uncertainty"] == 0.05  # only a typed distance


def test_surface_area_from_an_image_polygon():
    ground = [(-10, 8), (10, 8), (10, 28), (-10, 28)]  # 20 x 20 m
    poly_px = to_image(CAM, ground)
    surf = surface_polygon(CAM.homography(), poly_px, max_range=100, max_res=1.0)
    assert area(surf) == pytest.approx(400.0, rel=0.02)


def test_horizon_guard_drops_far_ground():
    poly_px = to_image(CAM, [(-10, 8), (10, 8), (10, 90), (-10, 90)])
    near = surface_polygon(CAM.homography(), poly_px, max_range=40, max_res=1.0)
    assert np.linalg.norm(near, axis=1).max() <= 40 and area(near) < 20 * 82


def test_pack_capacity_rectangle():
    rect = np.array([[0, 0], [25, 0], [25, 10], [0, 10]], float)
    assert len(pack_capacity(rect, angle_deg=90)) == 20  # one double row: 10 bays x 2
    assert len(pack_capacity(rect)) == 20  # default orientation finds it
    tall = np.array([[0, 0], [10, 0], [10, 25], [0, 25]], float)
    assert len(pack_capacity(tall)) == 20  # same lot turned: the same count
    assert len(pack_capacity(rect[:2])) == 0
    assert dominant_angle(rect) in (0.0, 180.0) or abs(dominant_angle(rect)) < 1e-3


def test_pack_capacity_adds_an_aisle_between_double_rows():
    lot = np.array([[0, 0], [25, 0], [25, 26], [0, 26]], float)  # 10 + 6 aisle + 10 -> 2 double rows
    assert len(pack_capacity(lot, angle_deg=90)) == 40


def test_bays_go_back_to_pixels():
    lot = np.array([[-6, 10], [6, 10], [6, 20], [-6, 20]], float)
    bays = pack_capacity(lot, angle_deg=90)
    polys = to_pixel_polys(CAM, bays)
    assert len(polys) == len(bays) > 0 and all(p.shape == (4, 2) for p in polys)


def test_footprint_size_and_distance():
    H = CAM.homography()
    a = footprint(car_box(CAM, -2, 15), H)
    b = footprint(car_box(CAM, 2, 15), H)
    assert area(a) == pytest.approx(4.5 * 1.8, rel=0.05)
    # centres 4 m apart sideways, each 1.8 m wide -> 2.2 m between the cars
    assert poly_distance(a, b) == pytest.approx(2.2, abs=0.25)
    assert poly_distance(a, a) == 0.0


def test_nearest_pairs():
    sq = lambda x, y: np.array([[x, y], [x + 2, y], [x + 2, y + 4], [x, y + 4]], float)  # noqa: E731
    pairs = nearest_pairs([sq(0, 0), sq(3, 0), sq(40, 0)], max_dist=6)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)] and pairs[0][2] == pytest.approx(1.0)
