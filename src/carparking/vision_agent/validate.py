"""Vision agent, step 3: validation and guardrails between the derived parameters and the layout engine / UI.

Nothing the model said reaches a layout until it passes here. The checks, in order, each with a reason a retry can show the model:

  1. proposal   : every point finite and inside the picture (0..1), no line without length, no duplicate lines, enough lines that are not
                  occluded (an occluded line is a guess, derive.py never uses it; here it only has to leave a row behind);
  2. parameters : stall angle strictly 20..160 degrees, line 0 dots and everything inside 0..1, count 1..MAX_SLOTS, a clean prefix;
  3. physics    : bay width (across the row, = line spacing x sin(angle)) 2.2-3.0 m and depth 4.5-6.0 m, each with a tolerance (`Limits.tol`);
  4. fit        : the parameters are handed to the layout engine's own solver (`layout_extra.layout_from_dots`) with every used line as a dot;
                  its fitting residual must be small, its coverage of the zone (when an outline is given) near 100 %, and the angle it
                  reads from the dots must agree with ours (this is what catches a mirrored angle).

`run_calibration` wraps agent -> derive -> validate with bounded retries: a failed attempt's reasons go back to the agent in
`constraints["previous_errors"]`; every failure is kept in the outcome (and logged) so a fallback to the manual tools can say why.
No network here: the agent is whatever `BaseVisionAgent` the caller passes.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import ValidationError

from carparking.layout import MAX_STALL_DEG, MIN_STALL_DEG, LayoutError
from carparking.layout_extra import Dot, layout_from_dots
from carparking.vision_agent.base import MAX_SLOTS, BaseVisionAgent, RawVisionProposal, SlotCalibrationParameters, Visibility
from carparking.vision_agent.derive import DeriveError, derive

log = logging.getLogger("carparking.vision_agent")


@dataclass(frozen=True)
class Limits:
    """The bounds a proposal must meet. width_m / depth_m: the usual bay (across the row / its length), widened by `tol` (10 % = "roughly")."""
    width_m: tuple[float, float] = (2.2, 3.0)
    depth_m: tuple[float, float] = (4.5, 6.0)
    tol: float = 0.10
    max_residual_m: float = 0.25  # RMS distance of the used lines' fronts from the fitted lattice
    coverage_pct: tuple[float, float] = (85.0, 115.0)  # the layout engine's own band, checked only when a zone outline is given
    angle_agree_deg: float = 5.0  # the engine's angle from the dots vs ours
    min_lines: int = 2  # usable (not occluded) lines
    max_occluded_share: float = 0.5  # more than this share of the lines occluded: only a warning, the row is still read from the rest


@dataclass
class Report:
    """The verdict: `errors` reject the proposal, `warnings` do not. `metrics` is what was measured."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _in_unit(*pts) -> bool:
    return all(math.isfinite(v) and 0.0 <= v <= 1.0 for p in pts for v in p)


def check_proposal(proposal: RawVisionProposal, limits: Limits = Limits()) -> Report:
    """The model's raw lines: coordinates, length, duplicates, occlusion. (The pydantic models already refuse most of this; a model built with
    `model_construct` or changed later is checked again here, because this is the gate.)"""
    rep = Report()
    usable = []
    for i, ln in enumerate(proposal.lines):
        if not _in_unit(ln.front, ln.back):
            rep.errors.append(f"line {i}: a point is outside the picture (coordinates must be fractions 0..1)")
        elif ln.front == ln.back:
            rep.errors.append(f"line {i}: front and back are the same point")
        elif ln.visible is not Visibility.OCCLUDED:
            usable.append(i)
    n_occ = sum(ln.visible is Visibility.OCCLUDED for ln in proposal.lines)
    rep.metrics.update(lines=len(proposal.lines), usable=len(usable), occluded=n_occ)
    for a in range(len(usable)):
        for b in range(a + 1, len(usable)):
            fa, fb = proposal.lines[usable[a]].front, proposal.lines[usable[b]].front
            if abs(fa[0] - fb[0]) < 1e-3 and abs(fa[1] - fb[1]) < 1e-3:
                rep.errors.append(f"lines {usable[a]} and {usable[b]} are the same line (their front points coincide)")
    if len(usable) < limits.min_lines:
        rep.errors.append(f"only {len(usable)} line(s) are clearly or faintly visible; at least {limits.min_lines} are needed (do not mark lines that are hidden)")
    if proposal.lines and n_occ / len(proposal.lines) > limits.max_occluded_share:
        rep.warnings.append(f"{n_occ} of {len(proposal.lines)} lines are occluded: the row is read from the rest")
    return rep


def check_parameters(params: SlotCalibrationParameters, info: dict, cam, w: int, h: int, limits: Limits = Limits(),
                     outline_px=None) -> Report:
    """Derived parameters: bounds, bay dimensions, and the layout engine's own fit of them. `info` is derive()'s second result."""
    rep = Report()
    if not (math.isfinite(params.stall_angle) and MIN_STALL_DEG < params.stall_angle < MAX_STALL_DEG):
        rep.errors.append(f"the bay angle {params.stall_angle:.1f} degrees is not strictly inside {MIN_STALL_DEG:.0f} to {MAX_STALL_DEG:.0f}")
    if not _in_unit(params.line0_front_dot, params.line0_back_dot):
        rep.errors.append("line 0's dots are outside the picture")
    if not 1 <= params.total_slots <= MAX_SLOTS:
        rep.errors.append(f"{params.total_slots} bays is outside 1 to {MAX_SLOTS}")
    if rep.errors:
        return rep
    step, depth = float(info["step_m"]), float(info["depth_m"])
    width = step * math.sin(math.radians(params.stall_angle))  # the bay's width across its own sides
    rep.metrics.update(width_m=round(width, 3), depth_m=round(depth, 3), step_m=round(step, 3))
    for name, v, (lo, hi) in (("bay width", width, limits.width_m), ("bay depth", depth, limits.depth_m)):
        if not lo * (1 - limits.tol) <= v <= hi * (1 + limits.tol):
            rep.errors.append(f"{name} {v:.2f} m is not a parking bay ({lo:.1f} to {hi:.1f} m, {limits.tol * 100:.0f} % tolerance): check the line positions or the calibration")
        elif not lo <= v <= hi:
            rep.warnings.append(f"{name} {v:.2f} m is a little outside the usual {lo:.1f} to {hi:.1f} m")
    # the layout engine's own fit: line 0 (front and back) first, then the other used lines' fronts, each with its line number
    pts = list(info["front_dots"])
    if not pts:
        rep.errors.append("no used line left to fit")
        return rep
    first = min(pts, key=lambda t: t[2])
    dots = [Dot((first[0], first[1]), first[2]), Dot(params.line0_back_dot, first[2], "back")] + [Dot((x, y), k) for x, y, k in pts if (x, y, k) != first]
    try:
        lay, fit = layout_from_dots(cam, dots, params.total_slots, w, h, depth=depth, outline_px=outline_px)
    except LayoutError as e:
        rep.errors.append(f"the layout engine cannot fit these lines: {e}")
        return rep
    rep.metrics.update(residual_m=fit["residual_m"], coverage_pct=fit["coverage_pct"], engine_angle_deg=fit["stall_angle_deg"], engine_warnings=fit["warnings"])
    if fit["residual_m"] > limits.max_residual_m:
        rep.errors.append(f"the lines do not sit on one regular row: fitting residual {fit['residual_m']:.2f} m (limit {limits.max_residual_m:.2f} m)")
    if fit["coverage_pct"] is not None and not limits.coverage_pct[0] <= fit["coverage_pct"] <= limits.coverage_pct[1]:
        rep.errors.append(f"{params.total_slots} bays cover {fit['coverage_pct']:.0f} % of the zone (expected {limits.coverage_pct[0]:.0f} to {limits.coverage_pct[1]:.0f} %)")
    if fit["stall_source"] == "dots" and abs(fit["stall_angle_deg"] - params.stall_angle) > limits.angle_agree_deg:
        rep.errors.append(f"the layout engine reads {fit['stall_angle_deg']:.0f} degrees from the same lines, not {params.stall_angle:.0f}: the angle may be mirrored")
    rep.warnings += list(fit["warnings"])
    return rep


# ---- bounded retries around agent -> derive -> validate ------------------------------------------------------------------------------------
@dataclass
class Attempt:
    """One try: its outcome, the reasons it failed (empty on success) and what the model said (for the fallback log; None if the call failed)."""
    number: int
    errors: list[str]
    proposal: Optional[RawVisionProposal] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    """The result of `run_calibration`: `params` is set only when `ok`; `attempts` holds every try, so a fallback can say why it was needed."""
    ok: bool
    params: Optional[SlotCalibrationParameters] = None
    info: dict = field(default_factory=dict)
    report: Optional[Report] = None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def reason(self) -> str:
        """Why it failed (the last attempt's reasons), or "" on success."""
        return "" if self.ok or not self.attempts else "; ".join(self.attempts[-1].errors)


def run_calibration(agent: BaseVisionAgent, crop_png_path: str, constraints: dict, cam, w: int, h: int,
                    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0), slot_prefix: str = "A", limits: Limits = Limits(),
                    outline_px=None, max_attempts: int = 2, on_failure: Optional[Callable[[Attempt], None]] = None) -> Outcome:
    """agent.propose -> derive -> check_proposal / check_parameters, up to `max_attempts` times. After a failure the reasons are passed back in
    `constraints["previous_errors"]` (a copy: the caller's dict is not changed). Every failure (a provider error, an invalid reply, a rejected
    proposal) is recorded as an Attempt, logged, and given to `on_failure`; nothing raised by the agent escapes. Returns an Outcome, never raises
    for these; the caller falls back to the manual layout tools when `ok` is False."""
    outcome = Outcome(ok=False)
    cons = dict(constraints)
    for n in range(1, max(1, max_attempts) + 1):
        att = Attempt(number=n, errors=[])
        outcome.attempts.append(att)
        try:
            att.proposal = agent.propose(crop_png_path, dict(cons))
            pre = check_proposal(att.proposal, limits)
            att.warnings += pre.warnings
            if not pre.ok:
                att.errors = pre.errors
            else:
                params, info = derive(att.proposal, cam, w, h, crop=crop, slot_prefix=slot_prefix)
                rep = check_parameters(params, info, cam, w, h, limits, outline_px)
                att.warnings += info["warnings"] + rep.warnings
                if rep.ok:
                    outcome.ok, outcome.params, outcome.info, outcome.report = True, params, info, rep
                    return outcome
                att.errors = rep.errors
        except DeriveError as e:
            att.errors = [str(e)]
        except ValidationError as e:  # the reply did not fit the schema
            att.errors = [f"the reply is not valid: {x['loc']}: {x['msg']}" for x in e.errors()][:5]
        except Exception as e:  # noqa: BLE001 - a provider / network error must not escape into the page; it is recorded and the manual path stays
            att.errors = [f"the vision model call failed: {type(e).__name__}: {e}"]
        log.warning("vision calibration attempt %d failed: %s", n, "; ".join(att.errors))
        if on_failure is not None:
            on_failure(att)
        cons["previous_errors"] = list(att.errors)
    return outcome
