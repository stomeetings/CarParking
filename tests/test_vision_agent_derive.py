"""derive.py: synthetic proposals made from a known layout through a synthetic camera (no model, no network)."""
import numpy as np
import pytest

from carparking import layout as L
from carparking.geometry.camera import CameraModel, to_image
from carparking.vision_agent.base import LineEndpoint, RawVisionProposal, Visibility
from carparking.vision_agent.derive import DeriveError, derive

W, H = 1920, 1080


def cam(pitch=35.0, f=1000.0, height=8.0, roll=1.0):
    return CameraModel(f=f, cx=W / 2, cy=H / 2, pitch=np.radians(pitch), roll=np.radians(roll), h=height)


def truth(angle=90.0, count=6):
    return L.ZoneLayout(origin=(-7.5, 7.0), width=2.5, depth=5.0, count=count, rows=1, stall_angle=angle)


def frac(c, lay, k, v):
    """Full-frame fraction of line k of the lattice at across-distance v (a separator leans by v * lay.lean along the row, as in ZoneLayout.cell)."""
    return tuple((to_image(c, lay.to_ground([[k * (lay.width + lay.gap) + v * lay.lean, v]]))[0] / (W, H)).tolist())


def proposal(c, lay, ks=None, vis=None, crop=(0.0, 0.0, 1.0, 1.0), count=None, conf=0.9):
    ks = list(range(lay.count + 1)) if ks is None else ks
    x0, y0, x1, y1 = crop
    to_crop = lambda p: ((p[0] - x0) / (x1 - x0), (p[1] - y0) / (y1 - y0))  # noqa: E731
    lines = [LineEndpoint(front=to_crop(frac(c, lay, k, 0.0)), back=to_crop(frac(c, lay, k, lay.depth)), visible=(vis or {}).get(k, Visibility.CLEAR)) for k in ks]
    return RawVisionProposal(lines=lines, estimated_count=count, confidence=conf, notes="")


@pytest.mark.parametrize("angle", [90.0, 60.0, 45.0, 120.0])
def test_angle_count_and_line0_come_back_with_the_right_sign(angle):
    c, t = cam(), truth(angle)
    p, info = derive(proposal(c, t), c, W, H, slot_prefix="B")
    assert p.stall_angle == pytest.approx(angle, abs=1.0) and p.total_slots == 6 and p.slot_prefix == "B"
    assert p.line0_front_dot == pytest.approx(frac(c, t, 0, 0.0), abs=1e-6) and p.line0_back_dot == pytest.approx(frac(c, t, 0, 5.0), abs=1e-6)
    assert info["step_m"] == pytest.approx(2.5, abs=0.02) and info["row_length_m"] == pytest.approx(15.0, abs=0.1) and info["warnings"] == []


@pytest.mark.parametrize("pitch,f,height", [(25.0, 900.0, 6.0), (35.0, 1200.0, 9.0), (50.0, 1100.0, 12.0)])
def test_perspective_does_not_bend_the_angle(pitch, f, height):
    c, t = cam(pitch=pitch, f=f, height=height), truth(55.0)
    p, _ = derive(proposal(c, t), c, W, H)
    assert p.stall_angle == pytest.approx(55.0, abs=1.0) and p.total_slots == 6


def test_a_crop_is_mapped_back_to_the_full_frame():
    c, t = cam(), truth(60.0)
    crop = (0.05, 0.1, 0.95, 0.98)
    p, _ = derive(proposal(c, t, crop=crop), c, W, H, crop=crop)
    assert p.stall_angle == pytest.approx(60.0, abs=1.0) and p.line0_front_dot == pytest.approx(frac(c, t, 0, 0.0), abs=1e-6)
    with pytest.raises(DeriveError, match="crop"):
        derive(proposal(c, t), c, W, H, crop=(0.5, 0.0, 0.4, 1.0))


def test_a_missed_line_still_gives_the_right_count():
    c, t = cam(), truth(70.0)
    p, info = derive(proposal(c, t, ks=[0, 1, 2, 4, 5, 6]), c, W, H)  # line 3 missed
    assert p.total_slots == 6 and info["step_m"] == pytest.approx(2.5, abs=0.02)


def test_one_wrong_line_is_dropped_and_the_median_holds():
    c, t = cam(), truth(60.0)
    prop = proposal(c, t)
    bad = prop.lines[3].model_copy(update={"back": (prop.lines[3].front[0] - 0.2, prop.lines[3].back[1])})  # leans the wrong way
    prop = prop.model_copy(update={"lines": [*prop.lines[:3], bad, *prop.lines[4:]]})
    p, info = derive(prop, c, W, H)
    assert p.stall_angle == pytest.approx(60.0, abs=1.0) and any(i == 3 for i, _ in info["skipped"]) and 3 not in info["used"]


def test_occluded_lines_are_not_used_and_line0_is_the_first_usable():
    c, t = cam(), truth(60.0)
    p, info = derive(proposal(c, t, vis={0: Visibility.OCCLUDED, 3: Visibility.OCCLUDED}), c, W, H)
    assert (0, "occluded") in info["skipped"] and p.line0_front_dot == pytest.approx(frac(c, t, 1, 0.0), abs=1e-6)
    assert p.total_slots == 5 and any("line 0" in x for x in info["warnings"])  # lines 1..6 are 5 bays: the row now starts at line 1
    faded, _ = derive(proposal(c, t, vis={2: Visibility.FADED}), c, W, H)
    assert faded.total_slots == 6


def test_the_models_count_is_only_compared_and_the_total_is_clamped():
    c, t = cam(), truth(90.0)
    p, info = derive(proposal(c, t, count=9), c, W, H)
    assert p.total_slots == 6 and any("counted 9" in x for x in info["warnings"])
    p, info = derive(proposal(c, t), c, W, H, max_slots=4)
    assert p.total_slots == 4 and any("limited to 4" in x for x in info["warnings"])


def test_unusable_proposals_are_refused_with_a_reason():
    c, t = cam(), truth(90.0)
    with pytest.raises(DeriveError, match="at least 2"):
        derive(proposal(c, t, ks=[0]), c, W, H)
    with pytest.raises(DeriveError, match="no separator line is visible"):
        derive(proposal(c, t, vis={k: Visibility.OCCLUDED for k in range(7)}), c, W, H)
    with pytest.raises(DeriveError, match="confidence"):
        derive(proposal(c, t, conf=0.2), c, W, H, min_confidence=0.5)
    swapped = RawVisionProposal(lines=[LineEndpoint(front=ln.back, back=ln.front, visible=ln.visible) for ln in proposal(c, t).lines], estimated_count=None, confidence=0.9)
    p, _ = derive(swapped, c, W, H)  # front/back swapped: the normal follows the back points, the angle is the mirror of the lean (square stays square)
    assert p.stall_angle == pytest.approx(90.0, abs=1.0)
    near = RawVisionProposal(lines=[LineEndpoint(front=(0.5, 0.9), back=(0.5, 0.899), visible=Visibility.CLEAR), LineEndpoint(front=(0.6, 0.9), back=(0.6, 0.899), visible=Visibility.CLEAR)],
                             estimated_count=None, confidence=0.9)
    with pytest.raises(DeriveError, match="plausible length"):
        derive(near, c, W, H)
    sky = RawVisionProposal(lines=[LineEndpoint(front=(0.2, 0.05), back=(0.2, 0.02), visible=Visibility.CLEAR), LineEndpoint(front=(0.3, 0.05), back=(0.3, 0.02), visible=Visibility.CLEAR)],
                            estimated_count=None, confidence=0.9)
    with pytest.raises(DeriveError, match="plausible length"):  # above the horizon: not on the ground
        derive(sky, c, W, H)


def test_most_gaps_being_doubles_does_not_fool_the_spacing():
    c, t = cam(), truth(60.0)
    p, info = derive(proposal(c, t, ks=[0, 1, 3, 5, 6]), c, W, H)  # gaps of 1, 2, 2, 1 bays: the median gap is not one bay
    assert p.total_slots == 6 and info["step_m"] == pytest.approx(2.5, abs=0.02)


def test_two_lines_that_disagree_wildly_do_not_give_an_empty_angle():
    c, t = cam(), truth(90.0)
    prop = proposal(c, t, ks=[0, 1])
    bad = prop.lines[1].model_copy(update={"back": (prop.lines[1].front[0] + 0.25, prop.lines[1].back[1])})
    p, info = derive(prop.model_copy(update={"lines": [prop.lines[0], bad]}), c, W, H)  # 90 and about 60 degrees: no group of two
    assert 20 < p.stall_angle < 160 and len(info["used"]) == 1
