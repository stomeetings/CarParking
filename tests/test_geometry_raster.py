"""The surface as a ground raster (geometry/bev.py Grid, surface_grid, car_footprint; capacity.pack_grid, bay_angle_from_cars) and
build_record's use of the cars: bodies cut out, footprints added, bays oriented along the cars. Synthetic camera, known ground."""
import cv2
import numpy as np
import pytest

from carparking.geometry import CameraModel, to_image
from carparking.geometry.bev import Grid, add_footprints, car_footprint, surface_grid
from carparking.geometry.capacity import bay_angle_from_cars, pack_grid
from carparking.geometry.pipeline import build_record

W, H_PX = 1920, 1080
CAM = CameraModel(f=1500.0, cx=W / 2, cy=H_PX / 2, pitch=np.radians(32), roll=0.0, h=6.0)
LENS = CameraModel(f=1977.0, cx=W / 2, cy=H_PX / 2, pitch=np.radians(14), roll=0.0, h=4.0, k1=-0.46)


def dense(ground, n=80):
    g = np.array(ground, float)
    return np.vstack([a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None] for a, b in zip(g, np.roll(g, -1, axis=0))])


def mask_of(cam, ground):
    m = np.zeros((H_PX, W), np.uint8)
    cv2.fillPoly(m, [np.round(to_image(cam, dense(ground))).astype(np.int32)], 255)
    return m


def car_box(cam, x, y, half=0.9, height=1.45):
    base = to_image(cam, [(x - half, y), (x + half, y)])
    top = to_image(cam.with_height(cam.h - height), [(x - half, y), (x + half, y)])
    pts = np.vstack([base, top])
    return (*pts.min(axis=0), *pts.max(axis=0))


def up_of(cam):
    sp, cp = np.sin(cam.pitch), np.cos(cam.pitch)
    return -np.array([0.0, cp, sp])


def rect_grid(w, h, cell=0.1, holes=()):
    cells = np.ones((int(h / cell), int(w / cell)), bool)
    for (x, y, hw, hh) in holes:  # holes in metres: x, y, width, height
        cells[int(y / cell):int((y + hh) / cell), int(x / cell):int((x + hw) / cell)] = False
    return Grid(cells, 0.0, 0.0, cell)


def test_surface_grid_measures_the_ground_it_covers():
    ground = [(-6, 9), (6, 9), (10, 22), (-10, 22)]
    truth = 0.5 * (12 + 20) * 13
    for cam in (CAM, LENS):
        gr = surface_grid(cam, mask_of(cam, [(x, y * (0.8 if cam is LENS else 1)) for x, y in ground]))
        t = 0.5 * (12 + 20) * 13 * (0.8 if cam is LENS else 1)
        assert gr.area == pytest.approx(t, rel=0.03), cam.k1
    assert surface_grid(CAM, np.zeros((H_PX, W), np.uint8)) is None
    sky = np.zeros((H_PX, W), np.uint8)
    sky[:30] = 255  # above the horizon of the 14 deg camera (row ~47): no ground
    assert surface_grid(LENS, sky) is None
    assert truth > 0


def test_thin_strips_and_specks_are_not_surface():
    ground = [(-6, 9), (6, 9), (6, 16), (-6, 16)]
    m = mask_of(CAM, ground)
    kept = surface_grid(CAM, m).area
    strip = np.zeros_like(m)
    cv2.fillPoly(strip, [np.round(to_image(CAM, dense([(8, 9), (8.6, 9), (8.6, 16), (8, 16)]))).astype(np.int32)], 255)  # 0.6 m wide
    speck = np.zeros_like(m)
    cv2.fillPoly(speck, [np.round(to_image(CAM, dense([(-9, 12), (-8, 12), (-8, 13), (-9, 13)]))).astype(np.int32)], 255)  # 1 m2
    both = surface_grid(CAM, np.maximum(np.maximum(m, strip), speck))
    assert both.area == pytest.approx(kept, rel=0.02)  # the 0.6 m verge and the 1 m2 speck are gone
    assert surface_grid(CAM, strip) is None
    assert surface_grid(CAM, np.maximum(m, strip), open_m=0).area > kept * 1.03  # without the opening they count


def test_a_car_body_does_not_stretch_the_surface_behind_it():
    ground = [(-8, 9), (8, 9), (8, 14), (-8, 14)]  # a shallow lot: the car's roof ray reaches the ground at ~15.8 m
    car = car_box(CAM, 0.0, 12.0)
    mask = mask_of(CAM, ground)
    x1, y1, x2, y2 = (int(v) for v in car)
    mask[y1:y2, x1:x2] = 255  # SAM-style: the car's whole body counts as lot (the rays through it hit the ground far behind)
    naive = surface_grid(CAM, mask)
    cut = mask.copy()
    cut[y1:y2, x1:x2] = 0
    cleaned = surface_grid(CAM, cut)
    assert naive.polygon()[:, 1].max() > 15.0  # the body's pixels reach ground well beyond the lot
    assert cleaned.polygon()[:, 1].max() < 14.3 and cleaned.area < naive.area


def test_car_footprint_side_on_and_end_on():
    side = car_box(CAM, 0.0, 12.0, half=2.25)  # a 4.5 m wide contact line: seen side-on
    poly, u, ell = car_footprint(CAM, side)
    assert ell == pytest.approx(4.5, rel=0.08) and abs(u[0]) > 0.99  # the box's bottom edge is a little wider than the car
    assert cv2.contourArea(poly.astype(np.float32)) == pytest.approx(ell * 2.2, rel=0.03)  # side-on: 2.2 m deep
    assert poly[2:, 1].min() > poly[:2, 1].max() - 0.05  # reaches away from the camera
    end = car_box(CAM, 0.0, 12.0, half=0.9)  # 1.8 m: end-on
    fp_end = car_footprint(CAM, end)
    assert cv2.contourArea(fp_end[0].astype(np.float32)) == pytest.approx(fp_end[2] * 4.6, rel=0.03) and fp_end[2] < 2.2
    mid = car_footprint(CAM, car_box(CAM, 0.0, 12.0, half=1.4))  # 2.8 m: in between, the depth interpolates
    assert 2.2 < cv2.contourArea(mid[0].astype(np.float32)) / mid[2] < 4.6
    assert car_footprint(CAM, (100, -520, 300, -450)) is None  # above the horizon (row -397 for this camera)


def test_footprints_join_the_open_ground():
    gr = rect_grid(10, 8)
    gr.cells[:, :30] = False  # the left 3 m are not surface
    add_footprints(gr, [np.array([[0.5, 1.0], [2.5, 1.0], [2.5, 5.0], [0.5, 5.0]])])
    assert gr.cells[30, 15] and gr.cells[30, 50] and not gr.cells[70, 5]  # the car's patch, and the open ground; the rest still not


def test_bay_angle_from_cars():
    along = bay_angle_from_cars([((1.0, 0.0), 4.5), ((-1.0, 0.0), 4.2), ((0.99, 0.05), 4.4)])
    assert along[0] == pytest.approx(0, abs=3) or along[0] == pytest.approx(180, abs=3)  # side-on along X: bays long along X
    assert along[1] > 0.99
    end = bay_angle_from_cars([((1.0, 0.0), 1.8), ((1.0, 0.0), 1.9)])
    assert end[0] == pytest.approx(90, abs=1)  # end-on: the contact line is the width, the car runs across it
    mixed = bay_angle_from_cars([((1.0, 0.0), 4.5), ((0.0, 1.0), 4.5)])
    assert mixed[1] == 0.5  # perpendicular rows: half the cars agree, not enough for the pipeline to trust
    # outliers (partly hidden cars at odd angles, a car seen end-on) cannot outvote the cars that agree
    crowd = bay_angle_from_cars([((1, 0), 4.0)] * 3 + [((0.99, 0.1), 3.8), ((0.5, 0.86), 3.5), ((1, 0), 1.2)])
    assert crowd[0] == pytest.approx(2, abs=4) and crowd[1] == pytest.approx(4 / 6)
    assert bay_angle_from_cars([((1, 0), 2.3)]) is None  # an oblique view alone: no vote
    assert bay_angle_from_cars([]) is None


def test_pack_grid_counts():
    lot = rect_grid(25, 10)
    assert len(pack_grid(lot, 90)) == 20  # one double row of 10 bays x 2
    assert len(pack_grid(rect_grid(25, 26), 90)) == 40  # + 6 m aisle + another double row
    bays = pack_grid(lot, 90)
    for b in bays:
        assert cv2.contourArea(b.astype(np.float32)) == pytest.approx(12.5, rel=0.01)
    # the same lot turned 90 degrees, bays pointed along its long side
    assert len(pack_grid(rect_grid(10, 25), 0)) == 20
    assert pack_grid(Grid(np.zeros((50, 50), bool), 0, 0), 90) == []


def test_pack_grid_stops_at_obstacles_and_works_at_any_angle():
    holed = rect_grid(25, 10, holes=[(10.0, 3.0, 1.0, 1.0)])  # a 1 m pole in the lot: the bays on it are gone
    bays = pack_grid(holed, 90)
    assert 14 <= len(bays) < 20
    for b in bays:
        assert not (b[:, 0].min() < 11.0 and b[:, 0].max() > 10.0 and b[:, 1].min() < 4.0 and b[:, 1].max() > 3.0)
    tilted = Grid(np.zeros((300, 300), bool), -15.0, -15.0, 0.1)
    c = np.array([[0.0, -5.0], [9.0, 0.0], [5.0, 9.0], [-4.0, 4.0]])  # a 9 x 9 m square-ish quadrilateral
    cv2.fillPoly(tilted.cells.view(np.uint8), [np.round(tilted.to_cells(c)).astype(np.int32)], 1)
    assert len(pack_grid(tilted, 30)) > 0


def test_build_record_cuts_bodies_adds_footprints_and_orients_along_the_cars():
    # a lot seen by the 14 deg lens camera: three cars side by side along the wall (the camera's depth axis), each pointing
    # across it (side-on to the camera, their long axis along X): the real layout of Front Parking 02
    ground = [(-4, 9), (4, 9), (12, 24), (-12, 24)]
    cars_g = [(0.0, 11.0), (0.0, 13.6), (0.0, 16.2)]
    boxes = [car_box(LENS, x, y, half=2.25) for x, y in cars_g]
    poly = mask_of(LENS, ground)
    for x1, y1, x2, y2 in boxes:  # SAM-style: the cars' bodies are part of the polygon
        poly[int(y1):int(y2), int(x1):int(x2)] = 255
    cs, _ = cv2.findContours(poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = (max(cs, key=cv2.contourArea).reshape(-1, 2) / (W, H_PX)).tolist()
    kw = dict(width=W, height=H_PX, polygon=outline, f=LENS.f, up=up_of(LENS), k1=-0.46, lens="simple_radial",
              depth={"h": 4.0}, vehicles=boxes)
    without = build_record(**kw)
    with_cars = build_record(**kw, obstacles=boxes)
    assert with_cars["bev"]["footprints"] == 3 and without["bev"]["footprints"] == 0
    assert with_cars["bev"]["area_m2"] < without["bev"]["area_m2"]  # the bodies no longer stretch the surface behind them
    cap = with_cars["capacity"]
    assert cap["orientation_source"] == "vehicles" and without["capacity"]["orientation_source"] == "polygon"
    assert min(abs(cap["orientation_deg"]), abs(cap["orientation_deg"] - 180)) < 6  # bays run along the cars (the X axis)
    assert cap["geometric_capacity"] >= 3 and cap["with_cars"] == 3  # one bay per car, the rest free
    assert [s["car"] for s in cap["slots"][:3]] == [True] * 3 and not any(s["car"] for s in cap["slots"][3:])
    # every parked car sits on a packed bay: the bay under the car's centre is within 0.6 m of it
    centres = np.array([[x, y + 1.1] for x, y in cars_g])  # the contact line is the near edge: the car's centre is ~1.1 m behind
    bay_centres = np.array([np.array(s["poly_m"]).mean(axis=0) for s in cap["slots"]])
    for c in centres:
        assert np.linalg.norm(bay_centres - c, axis=1).min() < 0.6


def test_cars_too_far_for_their_position_to_mean_anything_are_ignored_and_reported():
    ground = [(-4, 9), (4, 9), (12, 30), (-12, 30)]
    near, far = car_box(LENS, 0.0, 12.0, half=2.25), car_box(LENS, 0.0, 27.0, half=2.25)  # ~0.04 vs ~0.09 m of ground per pixel
    poly = mask_of(LENS, ground)
    cs, _ = cv2.findContours(poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = (max(cs, key=cv2.contourArea).reshape(-1, 2) / (W, H_PX)).tolist()
    rec = build_record(width=W, height=H_PX, polygon=outline, f=LENS.f, up=up_of(LENS), k1=-0.46, lens="simple_radial",
                       depth={"h": 4.0}, vehicles=[near, far], obstacles=[near, far])
    assert rec["bev"]["footprints"] == 1 and rec["bev"]["cars_too_far"] == 1
    assert any("too far" in w for w in rec["warnings"])
    assert rec["capacity"]["with_cars"] == 1
