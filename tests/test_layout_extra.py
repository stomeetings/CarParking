"""Reference dots, master cloning, the metric inward margin, the setup-time line snap and the slot label raster; plus invariants over randomised cameras."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from carparking import layout as L  # noqa: E402
from carparking import layout_extra as X  # noqa: E402
from carparking.geometry import CameraModel, to_image  # noqa: E402
from carparking.geometry.camera import to_ground  # noqa: E402
from carparking.layout_snap import snap_lattice  # noqa: E402
from carparking.slotmap import SlotMap  # noqa: E402

W, H = 1920, 1080


def cam(pitch=35.0, f=1000.0, height=8.0, roll=1.0):
    return CameraModel(f=f, cx=W / 2, cy=H / 2, pitch=np.radians(pitch), roll=np.radians(roll), h=height)


def truth(count=6, **kw):
    return L.ZoneLayout(origin=(-7.5, 7.0), width=2.5, depth=5.0, count=count, rows=1, **kw)


def dot_at(c, lay, k, v=0.0):
    """Pixel (fractions) of line k of the lattice at across-distance v."""
    return tuple((to_image(c, lay.to_ground([[k * (lay.width + lay.gap), v]]))[0] / (W, H)).tolist())


# ---- reference dots ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ks", [(0, 1), (2, 5), (3, 4, 6), (0, 3, 6)])
def test_dots_anywhere_on_the_row_give_the_same_lattice(ks):
    c, t = cam(), truth()
    lay, info = X.layout_from_dots(c, [X.Dot(dot_at(c, t, k), k) for k in ks], 6, W, H, depth=5.0)
    assert lay.width == pytest.approx(2.5, abs=0.04) and info["residual_m"] < 0.03 and info["dropped"] == []
    assert np.allclose(lay.bays_ground()[0], t.bays_ground()[0], atol=0.08) and np.allclose(lay.bays_ground()[5], t.bays_ground()[5], atol=0.1)


def test_one_dot_needs_the_width_and_the_heading_and_may_be_in_the_middle():
    c, t = cam(), truth()
    lay, _ = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 3), 3)], 6, W, H, width=2.5, heading=0.0)
    assert np.allclose(lay.bays_ground()[0], t.bays_ground()[0], atol=0.05)
    with pytest.raises(L.LayoutError):
        X.layout_from_dots(c, [X.Dot(dot_at(c, t, 3), 3)], 6, W, H)


def test_dots_on_the_back_or_middle_of_the_lines_are_placed_by_dot_at():
    c, t = cam(), truth()
    for where, v in (("mid", 2.5), ("back", 5.0)):
        lay, _ = X.layout_from_dots(c, [X.Dot(dot_at(c, t, k, v), k) for k in (0, 2, 4)], 6, W, H, dot_at=where)
        assert np.allclose(lay.bays_ground()[0], t.bays_ground()[0], atol=0.1), where


def test_indices_come_from_the_spacing_and_a_misclick_is_dropped():
    c, t = cam(), truth(7)
    dots = [X.Dot(dot_at(c, t, k)) for k in (0, 1, 2, 3, 4)]
    dots.append(X.Dot(dot_at(c, t, 3.5)))  # a click between two lines
    lay, info = X.layout_from_dots(c, dots, 7, W, H, width=2.5)
    assert len(info["dropped"]) == 1 and lay.width == pytest.approx(2.5, abs=0.05)


# ---- master cloning ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("m", [0, 2, 5])
def test_a_master_bay_is_cloned_along_the_row_in_both_directions(m):
    c, t = cam(), truth()
    master = (to_image(c, t.bays_ground()[m]) / (W, H)).tolist()
    lay, info = X.clone_master(c, master, 6, W, H, master_index=m)
    assert info["width_m"] == pytest.approx(2.5, abs=0.03) and info["depth_m"] == pytest.approx(5.0, abs=0.05) and info["squareness_m"] < 0.05
    for a, b in zip(lay.bays_ground(), t.bays_ground()):
        assert np.allclose(a, b, atol=0.06)


def test_cloning_a_gap_and_a_second_row():
    c, t = cam(), truth()
    lay, _ = X.clone_master(c, (to_image(c, t.bays_ground()[0]) / (W, H)).tolist(), 3, W, H, rows=2, gap=0.2, row_gap=2.0)
    g = lay.bays_ground()
    assert len(g) == 6 and g[1][0][0] - g[0][0][0] == pytest.approx(2.7, abs=0.05) and g[3][0][1] - g[0][0][1] == pytest.approx(7.0, abs=0.08)


# ---- inward margin in metres ----------------------------------------------------------------------------------------------------------------------
def test_the_margin_is_in_metres_on_the_ground_not_pixels():
    c, lay = cam(), truth()
    slots, kept = X.inset_slots(lay, c, W, H, margin=0.15)
    assert kept == [] and len(slots) == 6
    for s, full in zip(slots, lay.bays_ground()):
        g = to_ground(c, np.array(s["poly"]) * (W, H))
        assert np.ptp(g[:, 0]) == pytest.approx(2.5 - 0.3, abs=0.06) and np.ptp(g[:, 1]) == pytest.approx(5.0 - 0.3, abs=0.06)
    assert X.inset_quad(np.array([[0, 0], [0.2, 0], [0.2, 5], [0, 5]]), 0.15) is None  # a 20 cm strip collapses


def test_neighbouring_bays_never_touch_after_the_margin():
    c, lay = cam(), truth()
    slots, _ = X.inset_slots(lay, c, W, H, 0.15)
    sm = SlotMap(slots, W, H, scale=1.0)
    for a, b in zip(slots, slots[1:]):
        ma, mb = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
        cv2.fillPoly(ma, [np.round(np.array(a["poly"]) * (W, H)).astype(np.int32)], 1)
        cv2.fillPoly(mb, [np.round(np.array(b["poly"]) * (W, H)).astype(np.int32)], 1)
        assert int((ma & mb).sum()) == 0
    del sm


# ---- setup-time snap ---------------------------------------------------------------------------------------------------------------------------
def painted_picture(c, t, ppm=50):
    """A synthetic lot: grey ground with white separators every width at the true lattice positions, seen through camera c."""
    lo, hi = np.array([-9.0, 5.0]), np.array([9.0, 14.0])
    size = ((hi - lo) * ppm).astype(int)
    bev = np.full((size[1], size[0], 3), 90, np.uint8)
    rng = np.random.default_rng(0)
    bev = np.clip(bev + rng.normal(0, 6, bev.shape), 0, 255).astype(np.uint8)
    for k in range(t.count + 1):
        x = int((t.origin[0] + k * (t.width + t.gap) - lo[0]) * ppm)
        cv2.line(bev, (x, int((7.0 - lo[1]) * ppm)), (x, int((12.0 - lo[1]) * ppm)), (240, 240, 240), max(2, int(0.12 * ppm)))
    ground = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    px = np.float32(to_image(c, ground))
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]), px)
    return cv2.warpPerspective(bev, M, (W, H), borderValue=(60, 60, 60))


def test_the_lattice_snaps_to_painted_lines_and_keeps_hand_edits():
    c, t = cam(), truth()
    img = painted_picture(c, t)
    off = t.with_params(origin=(t.origin[0] + 0.22, t.origin[1]), width=2.58)  # a lattice that is a little off
    off.set_offset(2, L.BayOffset(du=0.1))
    snapped, rep = snap_lattice(img, off, c)
    assert rep["snapped"] and rep["support"] >= 0.8, rep
    assert snapped.width == pytest.approx(2.5, abs=0.03) and snapped.origin[0] == pytest.approx(-7.5, abs=0.04)
    assert snapped.offsets[2].du == 0.1  # the hand edit stays


def test_no_lines_means_no_snap():
    c, t = cam(), truth()
    flat = np.full((H, W, 3), 100, np.uint8)
    out, rep = snap_lattice(flat, t, c)
    assert not rep["snapped"] and out is t and "left as it was" in rep["reason"]


# ---- the slot label raster -----------------------------------------------------------------------------------------------------------------------
def test_the_label_raster_agrees_with_point_in_polygon():
    c, lay = cam(), truth()
    slots = lay.to_slots(c, W, H)
    sm = SlotMap(slots, W, H)
    rng = np.random.default_rng(1)
    pts = rng.random((400, 2))
    got = sm.lookup_many(pts)
    wrong = 0
    for (x, y), g in zip(pts, got):
        want = next((s["id"] for s in slots if cv2.pointPolygonTest((np.float32(s["poly"]) * np.float32([W, H])).astype(np.float32), (float(x * W), float(y * H)), False) >= 0), None)
        wrong += g != want
    assert wrong <= 8  # (1/4 resolution: only points within a pixel of an edge may differ)
    assert sm.lookup(2.0, 0.5) is None and sm.lookup(0.5, 0.5) == sm.lookup_many([[0.5, 0.5]])[0]
    assert sm.version == SlotMap(slots, W, H).version != SlotMap(slots[:-1], W, H).version


# ---- invariants over random cameras (property-style, seeded) -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(25))
def test_any_camera_gives_exact_count_equal_ground_bays_and_a_clean_round_trip(seed):
    rng = np.random.default_rng(seed)
    pitch, height = float(rng.uniform(18, 70)), float(rng.uniform(4, 14))
    f = float(rng.uniform(700, 1500))
    c = CameraModel(f=f, cx=W / 2, cy=H / 2, pitch=np.radians(pitch), roll=np.radians(float(rng.uniform(-3, 3))), h=height)
    n = int(rng.integers(3, 8))
    width, depth = float(rng.uniform(2.2, 3.0)), float(rng.uniform(4.5, 5.5))
    y0 = height / np.tan(np.radians(pitch)) - depth / 2  # the row stands where the camera looks
    lay = L.ZoneLayout(origin=(-n * width / 2, y0), width=width, depth=depth, count=n)
    g = lay.bays_ground()
    px = [to_image(c, q) for q in g]
    if any(not np.isfinite(p).all() or (p[:, 0] < 0).any() or (p[:, 0] > W).any() or (p[:, 1] < 0).any() or (p[:, 1] > H).any() for p in px):
        pytest.skip("this random row is not wholly in the picture")
    slots = lay.to_slots(c, W, H)
    assert len(slots) == n  # the count is exact
    for q, s in zip(g, slots):
        back = to_ground(c, np.array(s["poly"]) * (W, H))
        assert np.abs(back - q).max() < 0.05  # drawn -> image -> ground round trip
        assert np.ptp(back[:, 0]) == pytest.approx(width, abs=0.05)  # equal on the ground
    margin, _ = X.inset_slots(lay, c, W, H, 0.15)
    assert len(margin) == n
    fit = L.refit(slots, c, W, H)
    assert fit.offsets == {} or all(o.corners is None for o in fit.offsets.values())
    assert (fit.count, fit.rows) == (n, 1)


# ---- dots are a guide, never the row's limits (the zone polygon is the extent) -------------------------------------------------------------------------
def zone_outline(c, t):
    g = t.bays_ground()
    ring = np.array([g[0][0], g[-1][1], g[-1][2], g[0][3]])
    return (to_image(c, ring) / (W, H)).tolist()


def same_row(lay, t, tol=0.12):
    return len(lay.bays_ground()) == t.total and all(np.allclose(a, b, atol=tol) for a, b in zip(lay.bays_ground(), t.bays_ground()))


def test_dots_on_the_first_two_lines_do_not_squeeze_the_row_into_that_slot():
    c, t = cam(), truth(6)
    lay, info = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 0)), X.Dot(dot_at(c, t, 1))], 6, W, H, outline_px=zone_outline(c, t))
    assert lay.width == pytest.approx(2.5, abs=0.05) and same_row(lay, t)  # six bays across the whole 15 m zone
    assert info["extent_m"] == pytest.approx(15.0, abs=0.2) and info["covers_m"] == pytest.approx(15.0, abs=0.4) and info["warnings"] == []


def test_dots_closer_than_a_bay_set_the_start_and_direction_the_far_end_sets_the_size():
    c, t = cam(), truth(6)
    mid = lambda k: tuple((to_image(c, t.to_ground([[k, 2.5]]))[0] / (W, H)).tolist())  # noqa: E731
    lay, info = X.layout_from_dots(c, [X.Dot(mid(0.0)), X.Dot(mid(0.4))], 6, W, H, outline_px=zone_outline(c, t), dot_at="mid")
    assert info["guide_only"] and lay.width == pytest.approx(2.5, abs=0.05) and same_row(lay, t)  # Dot 1 is the start (here the zone's start): 6 bays to the far end
    assert any("closer than one bay" in w for w in info["warnings"])


def test_the_first_dot_is_the_hard_start_nothing_is_drawn_before_it():
    c, t = cam(), truth(6)
    inner = t.with_params(origin=(t.origin[0] + 2.5, t.origin[1]), count=5)  # Dot 1 is one bay in from the zone's start; the row has 5 bays after it
    lay, info = X.layout_from_dots(c, [X.Dot(dot_at(c, inner, 0)), X.Dot(dot_at(c, inner, 1))], 5, W, H, outline_px=zone_outline(c, t))
    first = lay.bays_ground()[0]
    assert first[0][0] == pytest.approx(inner.origin[0], abs=0.1)  # the first bay starts AT Dot 1, not at the zone's edge 2.5 m before it
    assert lay.width == pytest.approx(2.5, abs=0.05) and info["warnings"] == []
    assert abs(lay.bays_ground()[-1][1][0] - (t.origin[0] + 15.0)) < 0.2  # and the N bays end at the zone's far end


def test_dots_closer_than_a_bay_span_from_dot_one_to_the_far_end():
    c, t = cam(), truth(6)
    mid = lambda k: tuple((to_image(c, t.to_ground([[k, 2.5]]))[0] / (W, H)).tolist())  # noqa: E731
    lay, info = X.layout_from_dots(c, [X.Dot(mid(1.0)), X.Dot(mid(1.3))], 5, W, H, outline_px=zone_outline(c, t), dot_at="mid")
    assert lay.bays_ground()[0][0][0] == pytest.approx(t.origin[0] + 1.0, abs=0.1)  # starts at Dot 1 (1.0 m along the row), the dot 0.3 m further is only a guide
    assert info["guide_only"] and lay.width == pytest.approx(14.0 / 5, abs=0.05)  # 5 bays share the 14 m from Dot 1 to the end
    assert abs(lay.bays_ground()[-1][1][0] - (t.origin[0] + 15.0)) < 0.2


def test_stretch_to_fit_spans_from_the_first_dot_to_the_far_end_and_the_inward_margin_leaves_the_start_alone():
    c, t = cam(), truth(6)
    inner = t.with_params(origin=(t.origin[0] + 2.5, t.origin[1]))
    lay, info = X.layout_from_dots(c, [X.Dot(dot_at(c, inner, 0)), X.Dot(dot_at(c, inner, 1))], 5, W, H, outline_px=zone_outline(c, t), fit_zone=True)
    assert lay.width == pytest.approx(12.5 / 5, abs=0.05) and lay.bays_ground()[0][0][0] == pytest.approx(inner.origin[0], abs=0.1)
    assert info["available_m"] == pytest.approx(12.5, abs=0.2)
    plain, _ = lay.to_slots(c, W, H), None
    margin, _ = X.inset_slots(lay, c, W, H, 0.15)
    g0 = to_ground(c, np.array(margin[0]["poly"]) * (W, H))
    assert g0[:, 0].min() == pytest.approx(inner.origin[0] + 0.15, abs=0.06)  # the margin is inward from Dot 1's line, in metres: it never moves the row's start outward
    assert len(plain) == 5


def test_a_first_dot_past_the_zone_end_is_reported():
    c, t = cam(), truth(6)
    late = t.with_params(origin=(t.origin[0] + 15.2, t.origin[1]))
    _, info = X.layout_from_dots(c, [X.Dot(dot_at(c, late, 0)), X.Dot(dot_at(c, late, 1))], 5, W, H, width=2.5, outline_px=zone_outline(c, t))
    assert any("past the zone's far end" in w for w in info["warnings"])


def test_typed_line_numbers_still_fix_the_lattice_and_stretch_to_fit_works():
    c, t = cam(), truth(6)
    out = zone_outline(c, t)
    lay, _ = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 2), 2), X.Dot(dot_at(c, t, 3), 3)], 6, W, H, outline_px=out)
    assert same_row(lay, t)
    wide = t.with_params(width=2.7)  # dots a little too far apart (a sloppy click)
    lay2, _ = X.layout_from_dots(c, [X.Dot(dot_at(c, wide, 0)), X.Dot(dot_at(c, wide, 1))], 6, W, H, outline_px=out, fit_zone=True)
    assert lay2.width == pytest.approx(2.5, abs=0.03) and same_row(lay2, t)


def test_a_count_that_does_not_fit_the_zone_is_reported_not_hidden():
    c, t = cam(), truth(6)
    lay, info = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 0)), X.Dot(dot_at(c, t, 1))], 5, W, H, outline_px=zone_outline(c, t))
    assert lay.width == pytest.approx(2.5, abs=0.05) and any("cover" in w for w in info["warnings"])
    assert info["coverage_pct"] == pytest.approx(83.3, abs=2) and any("% of the" in w and "leave the rest" in w for w in info["warnings"])
    assert len(lay.bays_ground()) == 5  # the count is kept, the bays are not squeezed or stretched to hide the mismatch


def test_too_many_bays_run_past_the_zone_and_say_so():
    c, t = cam(), truth(6)
    lay, info = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 0)), X.Dot(dot_at(c, t, 1))], 8, W, H, outline_px=zone_outline(c, t))
    assert lay.width == pytest.approx(2.5, abs=0.05) and info["coverage_pct"] == pytest.approx(133, abs=3)
    assert any("run past the end" in w for w in info["warnings"])


# ---- dots on several slots, each with its own line number and place on the line --------------------------------------------------------------------------
def test_dots_on_the_first_middle_and_last_slot_with_mixed_places_give_the_whole_row():
    c, t = cam(), truth(6)
    out = zone_outline(c, t)
    dots = [X.Dot(dot_at(c, t, 0), 0), X.Dot(dot_at(c, t, 3), 3, "front"), X.Dot(dot_at(c, t, 6), 6)]
    lay, info = X.layout_from_dots(c, dots, 6, W, H, outline_px=out)
    assert same_row(lay, t) and info["residual_m"] < 0.05 and info["warnings"] == []


def test_some_dots_typed_and_the_rest_numbered_from_them():
    c, t = cam(), truth(6)
    dots = [X.Dot(dot_at(c, t, 1), 1), X.Dot(dot_at(c, t, 2)), X.Dot(dot_at(c, t, 5))]  # only the first has a line number
    lay, info = X.layout_from_dots(c, dots, 6, W, H, width=2.5, outline_px=zone_outline(c, t))
    assert info["indices"] == [1, 2, 5] and same_row(lay, t)


def test_a_dot_in_the_middle_of_its_line_is_placed_by_its_own_place():
    c, t = cam(), truth(6)
    mid = lambda k: tuple((to_image(c, t.to_ground([[k, 2.5]]))[0] / (W, H)).tolist())  # noqa: E731
    lay, _ = X.layout_from_dots(c, [X.Dot(mid(0), 0, "mid"), X.Dot(dot_at(c, t, 4), 4, "front")], 6, W, H, outline_px=zone_outline(c, t))
    assert same_row(lay, t)
    with pytest.raises(L.LayoutError):
        X.layout_from_dots(c, [X.Dot(mid(0), 0, "sideways"), X.Dot(mid(1), 1)], 6, W, H, outline_px=zone_outline(c, t))


def test_one_dot_runs_toward_the_zones_far_end_whichever_way_its_long_edge_is_listed():
    c, t = cam(), truth(6)
    out = zone_outline(c, t)
    for ring in (out, out[::-1]):  # the same zone listed in either direction
        lay, info = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 0))], 6, W, H, width=2.5, heading=0.0, outline_px=ring)
        assert lay.bays_ground()[0][0][0] == pytest.approx(t.origin[0], abs=0.1) and same_row(lay, t) and info["warnings"] == []


# ---- angled (diagonal) bays: a shear on the ground, read from a front / back dot pair on one separator -------------------------------------------------
def dot_on(c, lay, k, v=0.0):
    """Pixel (fractions) of separator k of a (possibly angled) lattice at across-distance v: it leans by v * cot(angle) along the row."""
    return tuple((to_image(c, lay.to_ground([[k * (lay.width + lay.gap) + v * lay.lean, v]]))[0] / (W, H)).tolist())


def slanted(angle=60.0, count=6, **kw):
    return truth(count, stall_angle=angle, **kw)


@pytest.mark.parametrize("angle", [45.0, 60.0, 75.0, 110.0])
def test_a_front_and_back_dot_on_one_line_tilt_the_whole_row_like_the_cars(angle):
    c, t = cam(), slanted(angle)
    dots = [X.Dot(dot_on(c, t, 0), 0), X.Dot(dot_on(c, t, 0, 5.0), 0, "back"), X.Dot(dot_on(c, t, 3), 3)]
    lay, info = X.layout_from_dots(c, dots, 6, W, H)
    assert lay.stall_angle == pytest.approx(angle, abs=1.0) and info["stall_source"] == "dots" and info["warnings"] == []
    assert lay.width == pytest.approx(2.5, abs=0.05) and same_row(lay, t)  # every bay leans the same, no bay edited by hand
    assert not lay.offsets


def test_the_angle_is_found_without_typed_line_numbers_and_with_an_outline():
    c, t = cam(), slanted(60.0)
    out = zone_outline(c, t)
    dots = [X.Dot(dot_on(c, t, 0)), X.Dot(dot_on(c, t, 0, 5.0), None, "back"), X.Dot(dot_on(c, t, 1))]
    lay, info = X.layout_from_dots(c, dots, 6, W, H, width=2.5, outline_px=out)
    assert lay.stall_angle == pytest.approx(60.0, abs=1.0) and same_row(lay, t)


def test_square_bays_stay_square_a_pair_on_a_square_row_and_no_pair_at_all():
    c, t = cam(), truth(6)
    lay, info = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 0), 0), X.Dot(dot_at(c, t, 0, 5.0), 0, "back"), X.Dot(dot_at(c, t, 3), 3)], 6, W, H)
    assert lay.stall_angle == pytest.approx(90.0, abs=0.5) and same_row(lay, t)
    lay2, info2 = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 0), 0), X.Dot(dot_at(c, t, 3), 3)], 6, W, H)
    assert lay2.stall_angle == 90.0 and info2["stall_source"] == "none" and same_row(lay2, t)  # nothing asked for: exactly today's bays


def test_a_typed_angle_wins_and_an_absurd_one_is_refused():
    c, t = cam(), slanted(60.0)
    dots = [X.Dot(dot_on(c, t, 0), 0), X.Dot(dot_on(c, t, 3), 3)]
    lay, info = X.layout_from_dots(c, dots, 6, W, H, stall_angle=60.0)
    assert info["stall_source"] == "typed" and lay.stall_angle == pytest.approx(60.0) and same_row(lay, t)
    with pytest.raises(L.LayoutError):
        X.layout_from_dots(c, dots, 6, W, H, stall_angle=10.0)


def test_the_back_dot_decides_which_side_the_bays_are_on():
    c, t = cam(), slanted(60.0, flip=True)  # the bays run to the other side of the row
    dots = [X.Dot(dot_on(c, t, 0), 0), X.Dot(dot_on(c, t, 0, 5.0), 0, "back"), X.Dot(dot_on(c, t, 3), 3)]
    lay, _ = X.layout_from_dots(c, dots, 6, W, H)
    assert lay.stall_angle == pytest.approx(60.0, abs=1.0) and same_row(lay, t)


def test_pairs_that_disagree_are_reported():
    c = cam()
    t60, t75 = slanted(60.0), slanted(75.0)
    dots = [X.Dot(dot_on(c, t60, 0), 0), X.Dot(dot_on(c, t60, 0, 5.0), 0, "back"), X.Dot(dot_on(c, t75, 3), 3), X.Dot(dot_on(c, t75, 3, 5.0), 3, "back")]
    _, info = X.layout_from_dots(c, dots, 6, W, H)
    assert any("disagree" in w for w in info["warnings"])


def test_a_refit_keeps_the_angle_as_the_layouts_not_as_per_bay_offsets():
    c, t = cam(), slanted(60.0)
    fit = L.refit(t.to_slots(c, W, H), c, W, H)
    assert fit.stall_angle == pytest.approx(60.0, abs=1.0) and not fit.offsets and fit.count == 6
    assert all(np.allclose(a, b, atol=0.1) for a, b in zip(fit.bays_ground(), t.bays_ground()))
    sq = L.refit(truth(6).to_slots(c, W, H), c, W, H)
    assert sq.stall_angle == 90.0  # square rows are not turned into slanted ones


def test_a_master_bay_that_is_a_parallelogram_clones_its_slant():
    c, t = cam(), slanted(60.0)
    master = (to_image(c, t.bays_ground()[2]) / (W, H)).tolist()
    lay, info = X.clone_master(c, master, 6, W, H, master_index=2)
    assert info["stall_angle_deg"] == pytest.approx(60.0, abs=1.0) and info["squareness_m"] < 0.05 and same_row(lay, t)


def test_the_inward_margin_works_on_slanted_bays_and_they_never_touch():
    c, t = cam(), slanted(60.0)
    slots, kept = X.inset_slots(t, c, W, H, 0.15)
    assert len(slots) == 6 and kept == []
    full = [np.abs(np.cross(q[1] - q[0], q[3] - q[0])) for q in t.bays_ground()]
    inner = [np.abs(np.cross(np.array(s["poly"])[1] - np.array(s["poly"])[0], np.array(s["poly"])[3] - np.array(s["poly"])[0])) for s in slots]
    assert len(full) == len(inner)


def test_the_angle_is_saved_and_an_old_layout_without_it_is_square():
    t = slanted(60.0)
    assert L.ZoneLayout.from_dict(t.to_dict()).stall_angle == pytest.approx(60.0)
    d = truth(6).to_dict()
    d.pop("stall_angle")
    assert L.ZoneLayout.from_dict(d).stall_angle == 90.0


@pytest.mark.parametrize("pitch,f,height", [(25.0, 900.0, 6.0), (35.0, 1200.0, 9.0), (50.0, 1100.0, 12.0), (30.0, 800.0, 5.0)])
def test_the_slant_is_perspective_correct_for_any_camera(pitch, f, height):
    c, t = cam(pitch=pitch, f=f, height=height), slanted(55.0)
    dots = [X.Dot(dot_on(c, t, 0), 0), X.Dot(dot_on(c, t, 0, 5.0), 0, "back"), X.Dot(dot_on(c, t, 3), 3)]
    lay, info = X.layout_from_dots(c, dots, 6, W, H)
    assert lay.stall_angle == pytest.approx(55.0, abs=1.5) and same_row(lay, t, tol=0.2)
    slots = lay.to_slots(c, W, H)  # the picture: every bay's side edges lean the same on the ground, so far bays are not drawn at a different angle than they are
    back = L.refit(slots, c, W, H)
    assert back.stall_angle == pytest.approx(55.0, abs=1.5)


def test_snap_leaves_slanted_bays_alone_and_says_why():
    c, t = cam(), slanted(60.0)
    lay, rep = snap_lattice(np.zeros((H, W, 3), np.uint8), t, c)
    assert lay is t and rep["snapped"] is False and "square bays only" in rep["reason"]


# ---- item 198: safeguards ---------------------------------------------------------------------------------------------------------------------
def test_a_gap_that_eats_the_step_is_refused_not_laid_inside_out():
    c, t = cam(), truth()
    dots = [X.Dot(dot_at(c, t, 0), 0), X.Dot(dot_at(c, t, 2), 2)]  # a 2.5 m step
    with pytest.raises(L.LayoutError, match="gap"):
        X.layout_from_dots(c, dots, 6, W, H, gap=2.4)
    lay, _ = X.layout_from_dots(c, dots, 6, W, H, gap=0.4)
    assert lay.width == pytest.approx(2.1, abs=0.05) and lay.gap == 0.4


def test_two_clicks_on_one_spot_are_one_dot():
    c, t = cam(), truth()
    dots = [X.Dot(dot_at(c, t, 0), 0), X.Dot(dot_at(c, t, 0)), X.Dot(dot_at(c, t, 3), 3)]  # the second click lands on the first
    lay, info = X.layout_from_dots(c, dots, 6, W, H)
    assert any("same spot" in w for w in info["warnings"]) and info["indices"] == [0, 3]
    assert np.allclose(lay.bays_ground()[0], t.bays_ground()[0], atol=0.08) and lay.width == pytest.approx(2.5, abs=0.04)
    one, _ = X.layout_from_dots(c, [X.Dot(dot_at(c, t, 2), 2), X.Dot(dot_at(c, t, 2))], 6, W, H, width=2.5, heading=0.0)  # both on one spot: one dot, not a wild direction
    assert one.heading == pytest.approx(0.0, abs=1e-6)


def test_a_long_row_has_no_drift_and_no_overlap():
    lay = L.ZoneLayout(origin=(-60.0, 7.0), heading=17.0, width=2.5, depth=5.0, gap=0.1, count=99, rows=3, row_gap=0.5, stall_angle=65.0)
    cells = [lay.cell(i) for i in range(lay.total)]
    for i in range(lay.total):
        r, k = divmod(i, 99)
        assert cells[i][0] == pytest.approx([k * 2.6, r * 5.5], abs=1e-12)  # placed by multiplication, not by adding up: the 99th is as exact as the first
        if k < 98:
            assert cells[i][1][0] == pytest.approx(cells[i + 1][0][0] - 0.1, abs=1e-12) and cells[i][2][0] == pytest.approx(cells[i + 1][3][0] - 0.1, abs=1e-12)
    widths = [float(np.linalg.norm(q[1] - q[0])) for q in lay.bays_ground()]
    assert max(widths) - min(widths) < 1e-9
