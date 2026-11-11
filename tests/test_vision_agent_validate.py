"""validate.py: guardrails and bounded retries, on synthetic proposals (no model, no network)."""
import numpy as np
import pytest

from carparking import layout as L
from carparking.geometry.camera import CameraModel, to_image
from carparking.vision_agent.base import BaseVisionAgent, LineEndpoint, RawVisionProposal, SlotCalibrationParameters, Visibility
from carparking.vision_agent.derive import derive
from carparking.vision_agent.validate import Limits, check_parameters, check_proposal, run_calibration

W, H = 1920, 1080


def cam(pitch=35.0, f=1000.0, height=8.0, roll=1.0):
    return CameraModel(f=f, cx=W / 2, cy=H / 2, pitch=np.radians(pitch), roll=np.radians(roll), h=height)


def truth(angle=90.0, count=6, width=2.5, depth=5.0):
    return L.ZoneLayout(origin=(-7.5, 7.0), width=width, depth=depth, count=count, rows=1, stall_angle=angle)  # (wide bays: fewer of them, to stay in the picture)


def frac(c, lay, k, v, shift=0.0):
    return tuple((to_image(c, lay.to_ground([[k * (lay.width + lay.gap) + v * lay.lean + shift, v]]))[0] / (W, H)).tolist())


def proposal(c, lay, ks=None, vis=None, shifts=None):
    ks = list(range(lay.count + 1)) if ks is None else ks
    lines = [LineEndpoint(front=frac(c, lay, k, 0.0, (shifts or {}).get(k, 0.0)), back=frac(c, lay, k, lay.depth, (shifts or {}).get(k, 0.0)),
                          visible=(vis or {}).get(k, Visibility.CLEAR)) for k in ks]
    return RawVisionProposal(lines=lines, estimated_count=None, confidence=0.9, notes="")


def run(c, lay, limits=Limits(), outline=None, **kw):
    p, info = derive(proposal(c, lay, **kw), c, W, H)
    return p, info, check_parameters(p, info, c, W, H, limits, outline)


def zone(c, lay, length_m):
    """A zone outline (fractions) covering the row's front line for `length_m` metres and its depth."""
    uv = [[-0.2, -0.2], [length_m, -0.2], [length_m, lay.depth + 0.2], [-0.2, lay.depth + 0.2]]
    return (to_image(c, lay.to_ground(uv)) / (W, H)).tolist()


# ---- a valid proposal passes -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [90.0, 60.0, 120.0])
def test_a_regular_row_passes_with_its_measurements(angle):
    c = cam()
    t = truth(angle, width=2.5 / np.sin(np.radians(angle)) if angle != 90.0 else 2.5)  # the same 2.5 m across the bay's own sides
    _, info, rep = run(c, t)
    assert rep.ok, rep.errors
    assert rep.metrics["width_m"] == pytest.approx(2.5, abs=0.05) and rep.metrics["depth_m"] == pytest.approx(5.0, abs=0.05)
    assert rep.metrics["residual_m"] < 0.05 and rep.metrics["engine_angle_deg"] == pytest.approx(angle, abs=2.0)


def test_coverage_is_checked_against_a_zone_outline():
    c, t = cam(), truth()
    assert run(c, t, outline=zone(c, t, 15.2))[2].ok
    rep = run(c, t, outline=zone(c, t, 24.0))[2]  # the zone is far longer than the bays the lines give
    assert not rep.ok and any("of the zone" in e for e in rep.errors)


# ---- bounds ----------------------------------------------------------------------------------------------------------------------------------
def params(**kw):
    return SlotCalibrationParameters.model_construct(**{"stall_angle": 60.0, "line0_front_dot": (0.2, 0.9), "line0_back_dot": (0.25, 0.5), "total_slots": 6, "slot_prefix": "A", **kw})


@pytest.mark.parametrize("kw,text", [({"stall_angle": 20.0}, "strictly"), ({"stall_angle": 160.0}, "strictly"), ({"stall_angle": 12.0}, "strictly"), ({"stall_angle": float("nan")}, "strictly"),
                                    ({"line0_front_dot": (1.2, 0.5)}, "outside the picture"), ({"line0_back_dot": (0.5, -0.1)}, "outside the picture"),
                                    ({"total_slots": 0}, "bays is outside"), ({"total_slots": 61}, "bays is outside")])
def test_out_of_bounds_parameters_are_rejected(kw, text):
    rep = check_parameters(params(**kw), {}, cam(), W, H)
    assert not rep.ok and any(text in e for e in rep.errors)


def test_a_proposal_with_points_outside_the_picture_or_no_length_is_rejected():
    ok = LineEndpoint(front=(0.2, 0.9), back=(0.25, 0.5), visible=Visibility.CLEAR)
    outside = LineEndpoint.model_construct(front=(1.4, 0.9), back=(0.3, 0.5), visible=Visibility.CLEAR)  # (built past the schema on purpose)
    flat = LineEndpoint.model_construct(front=(0.4, 0.9), back=(0.4, 0.9), visible=Visibility.CLEAR)
    rep = check_proposal(RawVisionProposal.model_construct(lines=[ok, outside, flat], estimated_count=None, confidence=0.5, notes=""))
    assert any("line 1" in e and "outside" in e for e in rep.errors) and any("line 2" in e and "same point" in e for e in rep.errors)


# ---- physical plausibility ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw,text", [({"width": 3.6, "count": 4}, "bay width"), ({"width": 1.9}, "bay width"), ({"depth": 8.0}, "bay depth"), ({"depth": 3.2}, "bay depth")])
def test_implausible_bay_sizes_are_rejected(kw, text):
    c = cam()
    _, _, rep = run(c, truth(**kw))
    assert not rep.ok and any(text in e for e in rep.errors), rep.errors


def test_a_slightly_odd_bay_is_a_warning_not_an_error():
    _, _, rep = run(cam(), truth(width=3.15, count=5))  # 5 % over the usual 3.0 m, inside the 10 % tolerance
    assert rep.ok and any("a little outside" in x for x in rep.warnings)


# ---- fit and residual -------------------------------------------------------------------------------------------------------------------------
def test_a_line_off_the_regular_spacing_fails_the_residual_check():
    c, t = cam(), truth()
    _, _, strict = run(c, t, Limits(max_residual_m=0.05), shifts={3: 0.45})
    assert not strict.ok and any("residual" in e for e in strict.errors)
    assert run(c, t, shifts={3: 0.45})[2].ok  # inside the default 0.25 m


def test_a_mirrored_angle_is_caught_by_the_engines_own_reading():
    c, t = cam(), truth(60.0)
    p, info = derive(proposal(c, t), c, W, H)
    bad = p.model_copy(update={"stall_angle": 120.0})
    rep = check_parameters(bad, info, c, W, H)
    assert not rep.ok and any("mirrored" in e for e in rep.errors)


# ---- occlusion / hallucination -----------------------------------------------------------------------------------------------------------------
def test_occluded_lines_never_corrupt_the_row():
    c, t = cam(), truth(60.0)
    prop = proposal(c, t, vis={2: Visibility.OCCLUDED, 4: Visibility.OCCLUDED})
    rep = check_proposal(prop)
    assert rep.ok and rep.metrics["occluded"] == 2 and rep.metrics["usable"] == 5
    p, info = derive(prop, c, W, H)
    assert p.total_slots == 6 and check_parameters(p, info, c, W, H).ok  # the row is read from the lines that are visible
    few = check_proposal(proposal(c, t, vis={k: Visibility.OCCLUDED for k in range(1, 7)}))
    assert not few.ok and any("at least 2" in e for e in few.errors) and any("occluded" in x for x in few.warnings)


def test_a_duplicate_line_is_rejected():
    c, t = cam(), truth()
    prop = proposal(c, t)
    dup = prop.model_copy(update={"lines": [*prop.lines, prop.lines[2]]})
    assert any("same line" in e for e in check_proposal(dup).errors)


# ---- the retry wrapper -------------------------------------------------------------------------------------------------------------------------
class Script(BaseVisionAgent):
    """Replies in order: a RawVisionProposal is returned, an exception is raised."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def propose(self, crop_png_path, constraints):
        self.seen.append(dict(constraints))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_a_good_reply_succeeds_on_the_first_attempt():
    c, t = cam(), truth(60.0, width=2.5 / np.sin(np.radians(60.0)))
    agent = Script(proposal(c, t))
    out = run_calibration(agent, "crop.png", {"rows": 1}, c, W, H, slot_prefix="B")
    assert out.ok and len(out.attempts) == 1 and out.params.slot_prefix == "B" and out.params.total_slots == 6 and out.reason == ""


def test_a_failure_is_retried_with_its_reasons_and_recorded():
    c, t = cam(), truth(60.0, width=2.5 / np.sin(np.radians(60.0)))
    seen, cons = [], {"rows": 1}
    agent = Script(RuntimeError("503 overloaded"), proposal(c, t))
    out = run_calibration(agent, "crop.png", cons, c, W, H, on_failure=seen.append)
    assert out.ok and len(out.attempts) == 2 and out.attempts[0].errors and "503 overloaded" in out.attempts[0].errors[0]
    assert "previous_errors" not in agent.seen[0] and "503 overloaded" in agent.seen[1]["previous_errors"][0]  # the 2nd call saw why the 1st failed
    assert cons == {"rows": 1} and [a.number for a in seen] == [1]  # the caller's dict is untouched


def test_every_attempt_failing_gives_a_reason_for_the_fallback():
    c, t = cam(), truth(width=3.6, count=4)  # bays far too wide
    notes = []
    out = run_calibration(Script(proposal(c, t), proposal(c, t)), "crop.png", {}, c, W, H, on_failure=notes.append)
    assert not out.ok and out.params is None and len(out.attempts) == 2 and len(notes) == 2 and "bay width" in out.reason


def test_an_invalid_reply_is_recorded_not_raised():
    c = cam()
    try:
        RawVisionProposal.model_validate_json('{"lines": [], "estimated_count": "many", "confidence": 3}')
    except Exception as e:  # noqa: BLE001
        err = e
    out = run_calibration(Script(err, ValueError("no json")), "crop.png", {}, c, W, H)
    assert not out.ok and "not valid" in out.attempts[0].errors[0] and "no json" in out.attempts[1].errors[0]
    assert run_calibration(Script(), "x.png", {}, c, W, H, max_attempts=0).attempts[0].errors  # at least one attempt is made (an empty script raises IndexError: recorded)
