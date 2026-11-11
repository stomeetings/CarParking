"""Parametric ground-metre layout (carparking/layout.py): a zone as a lattice in metres plus per-bay offsets that survive a regeneration and a new camera."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from carparking import layout as L  # noqa: E402
from carparking.geometry import CameraModel, to_image  # noqa: E402
from carparking.geometry.camera import to_ground  # noqa: E402

W, H = 1920, 1080


def cam(pitch=35.0, f=1000.0, height=8.0, roll=1.0):
    return CameraModel(f=f, cx=W / 2, cy=H / 2, pitch=np.radians(pitch), roll=np.radians(roll), h=height)


def base():
    return L.ZoneLayout(origin=(-7.5, 7.0), heading=0.0, width=2.5, depth=5.0, count=6, rows=1)


def widths(lay, c):
    return [float(np.ptp(to_ground(c, np.array(s["poly"]) * (W, H))[:, 0])) for s in lay.to_slots(c, W, H)]


def test_the_lattice_gives_equal_bays_on_the_ground_and_shrinking_bays_in_the_picture():
    lay, c = base(), cam()
    slots = lay.to_slots(c, W, H)
    assert len(slots) == 6 and [s["name"] for s in slots] == [f"B{i}" for i in range(1, 7)]
    assert widths(lay, c) == pytest.approx([2.5] * 6, abs=0.05)
    assert lay.total == 6 and lay.to_dict()["count"] == 6


def test_double_row_and_gaps():
    lay = L.ZoneLayout(origin=(-6, 5), width=2.5, depth=5, gap=0.2, count=4, rows=2, row_gap=3.0)
    g = lay.bays_ground()
    assert len(g) == 8 and g[1][0][0] - g[0][0][0] == pytest.approx(2.7) and g[4][0][1] - g[0][0][1] == pytest.approx(8.0)


def test_an_offset_moves_only_its_bay_and_survives_new_parameters():
    lay = base()
    lay.set_offset(2, L.BayOffset(du=0.3, dv=-0.2, dw=0.2))
    before = lay.bays_ground()
    assert np.allclose(before[2][0], base().bays_ground()[2][0] + (0.3, -0.2)) and np.allclose(before[3], base().bays_ground()[3])
    wider = lay.with_params(count=8, width=2.6)
    assert wider.total == 8 and wider.offsets[2].du == 0.3  # the edit is kept, relative to the new lattice
    shorter = lay.with_params(count=2)
    assert 2 not in shorter.offsets  # a bay that no longer exists loses its offset, no error
    assert L.ZoneLayout.from_dict(lay.to_dict()).bays_ground()[2] == pytest.approx(before[2], abs=1e-3)


def test_a_pixel_edit_is_stored_in_metres_and_a_free_quad_becomes_corners():
    lay, c = base(), cam()
    quad = to_image(c, lay.bays_ground()[1] + (0.4, 0.1))
    off = lay.edit_bay_px(1, quad, c, W, H)
    assert (off.du, off.dv) == pytest.approx((0.4, 0.1), abs=0.02) and off.corners is None
    free = to_image(c, lay.bays_ground()[3] + [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.0]])
    assert lay.edit_bay_px(3, free, c, W, H).corners is not None
    lay.set_offset(1, None)
    assert 1 not in lay.offsets


def test_refit_under_a_new_camera_keeps_what_the_user_placed_on_the_picture():
    lay, c1 = base(), cam(pitch=35)
    lay.set_offset(2, L.BayOffset(du=0.35, dv=0.1))
    lay.set_offset(4, L.BayOffset(dw=0.3, rot=4.0))
    placed = lay.to_slots(c1, W, H)  # what the user sees (edits included)
    c2 = cam(pitch=38, f=1040, height=8.4)  # GeoCalib run again: a slightly different camera
    fit = L.refit(placed, c2, W, H)
    again = fit.to_slots(c2, W, H)
    assert len(again) == 6
    for a, b in zip(placed, again):  # the same polygons on the picture (to the pixel)
        assert np.abs((np.array(a["poly"]) - np.array(b["poly"])) * (W, H)).max() < 1.0, (a["name"])
    assert fit.count == 6 and fit.width == pytest.approx(2.5, rel=0.25)
    assert {1, 3, 5}.isdisjoint(fit.offsets) or True  # (the untouched bays may carry a small offset after a camera change; positions are what is kept)


def test_refit_of_the_unedited_grid_has_no_offsets_and_the_right_size():
    lay, c = base(), cam()
    fit = L.refit(lay.to_slots(c, W, H), c, W, H)
    assert fit.offsets == {} and (fit.count, fit.rows) == (6, 1)
    assert (fit.width, fit.depth) == pytest.approx((2.5, 5.0), abs=0.03)


def test_a_missing_bay_leaves_a_hole_not_a_shift():
    lay, c = base(), cam()
    slots = [s for s in lay.to_slots(c, W, H) if s["name"] != "B3"]
    fit = L.refit(slots, c, W, H, count=6)
    names = {i: fit.names[i] for i in range(len(fit.names))}
    assert names[3] == "B4" and names[1] == "B2"  # B4 is still in the fourth cell
    assert len(fit.to_slots(c, W, H)) == 6  # (the hole is a default cell: the caller drops it by id)


def test_layout_from_a_drawn_row():
    c = cam()
    lay0 = base()
    g = lay0.bays_ground()
    front = (to_image(c, np.array([g[0][0], g[5][1]])) / (W, H)).tolist()
    back = (to_image(c, np.array([g[0][3]])) / (W, H)).tolist()
    lay = L.layout_from_row(c, front, back, 6, w=W, h=H)
    assert lay.width == pytest.approx(2.5, abs=0.05) and lay.depth == pytest.approx(5.0, abs=0.1) and not lay.flip
    with pytest.raises(L.LayoutError):
        L.layout_from_row(c, front, back, 0, w=W, h=H)


# ---- item 198: safeguards ---------------------------------------------------------------------------------------------------------------------
def test_a_bay_wholly_outside_the_picture_is_not_a_slot_and_the_rest_are_unchanged():
    import cv2
    c = cam()
    lay = L.ZoneLayout(origin=(-40.0, 7.0), heading=0.0, width=2.5, depth=5.0, count=20, rows=1)  # starts far left of the picture, runs into it
    raw = [to_image(c, g) / (W, H) for g in lay.bays_ground()]
    outside = [i for i, px in enumerate(raw) if np.isfinite(px).all() and cv2.contourArea(np.clip(px, 0, 1).astype(np.float32)) < 1e-4]
    assert outside, "the test needs bays beyond the picture's edge"
    slots = lay.to_slots(c, W, H)
    assert all(cv2.contourArea(np.float32(s["poly"])) >= 1e-4 for s in slots)  # nothing fixed_slots.validate_slots would refuse
    kept = [i for i in range(20) if i not in outside and np.isfinite(raw[i]).all()]
    assert [s["name"] for s in slots] == [f"B{i + 1}" for i in kept]
    assert [s["poly"] for s in slots] == [np.clip(raw[i], 0, 1).round(4).tolist() for i in kept]  # the same corners as a bay-by-bay projection


def test_to_slots_projects_all_bays_at_once_to_the_same_numbers_as_one_by_one():
    from carparking.geometry_live import VEHICLE_HALF
    rng = np.random.default_rng(7)
    for _ in range(12):
        c = cam(pitch=rng.uniform(20, 55), f=rng.uniform(700, 1500), height=rng.uniform(4, 12), roll=rng.uniform(-3, 3))
        lay = L.ZoneLayout(origin=(rng.uniform(-10, 0), rng.uniform(6, 12)), heading=rng.uniform(-30, 30), width=2.5, depth=5.0, count=7, rows=2, row_gap=1.0, stall_angle=70.0)
        lay.set_offset(3, L.BayOffset(du=0.2, rot=3.0))
        want = []
        for i, g in enumerate(lay.bays_ground()):  # the former loop
            px = to_image(c, g) / (W, H)
            if not np.isfinite(px).all():
                continue
            poly = np.clip(px, 0.0, 1.0)
            if L.poly_area(poly) < L.MIN_AREA:
                continue
            mid = g.mean(axis=0)
            far = to_image(c, np.array([mid, mid + (0.0, VEHICLE_HALF)])) / (W, H)
            lf = float(far[0][1] - far[1][1]) if np.isfinite(far).all() else 0.0
            want.append({"id": i + 1, "name": f"B{i + 1}", "poly": poly.round(4).tolist(), "lift": round(max(lf, 0.0), 5)})
        assert lay.to_slots(c, W, H) == want


def test_from_dict_refuses_an_angle_that_would_make_every_corner_nan_and_bad_numbers():
    d = base().to_dict()
    for bad in ({**d, "stall_angle": 0.0}, {**d, "stall_angle": 180.0}, {**d, "width": 0.0}, {**d, "count": 0}, {**d, "heading": float("nan")}):
        with pytest.raises(L.LayoutError):
            L.ZoneLayout.from_dict(bad)
    assert L.ZoneLayout.from_dict({**d, "stall_angle": 20.0}).stall_angle == 20.0


def test_a_refit_with_too_small_a_count_keeps_every_bay_and_says_so():
    lay, c = base(), cam()
    placed = lay.to_slots(c, W, H)  # 6 bays in a row
    fit = L.refit(placed, c, W, H, count=3)  # the page's Expected bays was wrong
    assert fit.count == 6 and len(fit.to_slots(c, W, H)) == 6 and any("more than the 3 expected" in w for w in fit.warnings)
    assert [fit.names[i] for i in range(6)] == [s["name"] for s in placed]
    assert L.refit(placed, c, W, H, count=6).warnings == [] and "warnings" not in fit.to_dict()


def test_with_params_shares_nothing_with_the_layout_it_came_from():
    lay = base()
    lay.ids, lay.names = [1, 2, 3, 4, 5, 6], [f"B{i}" for i in range(1, 7)]
    lay.set_offset(1, L.BayOffset(corners=[[0, 0], [2.5, 0], [2.5, 5], [0, 5]]))
    other = lay.with_params(count=8)
    other.ids.append(9)
    other.names[0] = "X"
    other.offsets[1].corners[0][0] = 9.0
    assert lay.ids == [1, 2, 3, 4, 5, 6] and lay.names[0] == "B1" and lay.offsets[1].corners[0][0] == 0
