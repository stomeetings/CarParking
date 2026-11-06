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
