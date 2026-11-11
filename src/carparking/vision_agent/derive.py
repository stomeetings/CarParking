"""Vision agent, step 2: from the model's observed lines to the layout parameters, on the GROUND plane.

The model only says where it sees painted separator lines (`RawVisionProposal`, points in the crop it was shown). Everything the layout
uses is worked out here, in metres, through the calibrated camera, so the perspective of the picture cannot bend the angle or the count:

  * each line's front / back points -> ground points (`geometry.camera.to_ground`);
  * the row's direction = the line through the front points, running from the first line listed toward the last; its normal points
    toward the back points;
  * stall angle    = per line atan2(across, along) of its ground vector in that frame (90 = square, `layout.ZoneLayout.stall_angle`'s
                     convention), the MEDIAN over the lines; a line far from it is dropped;
  * spacing        = the median gap between neighbouring front points along the row (a missed line makes a double gap: gaps are divided
                     by the whole number of bays they hold);
  * total_slots    = row length (first to last line) / spacing, at most `max_slots`;
  * line 0         = the first line along the row, as two full-frame points: the anchor `layout_extra.layout_from_dots` takes.

The model's own count is never used, only compared (a warning). No network, no model: a pure function of its inputs.
"""
from __future__ import annotations

import numpy as np

from carparking.geometry.camera import to_ground
from carparking.layout import MAX_STALL_DEG, MIN_STALL_DEG
from carparking.vision_agent.base import MAX_SLOTS, RawVisionProposal, SlotCalibrationParameters, Visibility

MIN_DEPTH_M, MAX_DEPTH_M = 2.0, 10.0  # a separator's length across the row, on the ground (shorter / longer is not a bay's line)
MIN_STEP_M, MAX_STEP_M = 1.5, 6.0  # the spacing of the lines along the row (a 2.5 m bay at 45 degrees is 3.5 m)
ANGLE_OUTLIER_DEG = 10.0  # a line whose angle is this far from the median is dropped
ANGLE_WARN_DEG = 5.0  # the lines' angles spread wider than this: say so (same threshold as the layout engine)
ROW_RESIDUAL_M = 0.5  # the front points stray further than this from one straight line: say so
GAP_TOL = 0.25  # a gap this far (in bays) from a whole number of bays: say so


class DeriveError(ValueError):
    """The proposal cannot give a layout (too few usable lines, not on measurable ground, implausible spacing ...). The message says why."""


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


def _bay_step(gaps: np.ndarray) -> float:
    """The bay spacing the gaps are whole multiples of. A median is wrong when missed lines make most gaps doubles, so: candidates are every gap
    divided by 1..3 inside the plausible range; the LARGEST one all gaps fit (each within GAP_TOL of a whole number of bays) wins, since a half
    step fits too; none fits: the one with the smallest mean misfit."""
    cands = sorted({round(float(g) / m, 3) for g in gaps for m in (1, 2, 3) if MIN_STEP_M <= g / m <= MAX_STEP_M}, reverse=True)
    if not cands:
        return float(np.median(gaps))
    miss = lambda c: np.abs(gaps / c - np.maximum(np.round(gaps / c), 1.0))  # noqa: E731
    for c in cands:
        if miss(c).max() <= GAP_TOL:
            return c
    return min(cands, key=lambda c: float(miss(c).mean()))


def derive(proposal: RawVisionProposal, cam, w: int, h: int, crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
           slot_prefix: str = "A", max_slots: int = MAX_SLOTS, min_confidence: float = 0.0) -> tuple[SlotCalibrationParameters, dict]:
    """-> (parameters, info). `cam`: the calibrated camera of the picture (`w` x `h` px) the crop was cut from; `crop`: (x0, y0, x1, y1) of the
    crop in full-frame fractions (the proposal's points are fractions of it). info: {"depth_m", "front_dots" (x, y, line number of each used line's
    front, full-frame fractions), "stall_deg", "angles_deg", "step_m", "row_length_m",
    "count_from_lines", "row_residual_m", "used", "skipped", "warnings"}. Raises DeriveError."""
    x0, y0, x1, y1 = crop
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise DeriveError("the crop must lie inside the picture: 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    if not 1 <= max_slots <= MAX_SLOTS:
        raise DeriveError(f"max_slots is 1 to {MAX_SLOTS}")
    if proposal.confidence < min_confidence:
        raise DeriveError(f"the model's confidence {proposal.confidence:.2f} is under {min_confidence:.2f}")
    warnings: list[str] = []
    skipped: list[tuple[int, str]] = []
    scale, origin = np.array([x1 - x0, y1 - y0]), np.array([x0, y0])

    keep: list[int] = []
    for i, ln in enumerate(proposal.lines):  # an occluded line's points are guesses: not used
        if ln.visible is Visibility.OCCLUDED:
            skipped.append((i, "occluded"))
        else:
            keep.append(i)
    if not keep:
        raise DeriveError("no separator line is visible")
    full = {i: (origin + np.array(proposal.lines[i].front) * scale, origin + np.array(proposal.lines[i].back) * scale) for i in keep}  # full-frame fractions
    px = np.array([p for i in keep for p in full[i]]) * (w, h)
    g = to_ground(cam, px).reshape(-1, 2, 2)  # (line, front|back, X|Y) in metres
    ok = np.isfinite(g).all(axis=(1, 2))
    for j in np.flatnonzero(~ok):
        skipped.append((keep[j], "not on measurable ground"))
    keep = [i for i, k in zip(keep, ok) if k]
    g = g[ok]
    if len(keep) < 2:
        raise DeriveError(f"{len(keep)} usable separator line(s): at least 2 are needed to find the row and the spacing")
    fronts, backs = g[:, 0], g[:, 1]

    # the row: the line through the front points, from the first line listed toward the last; the back points say which side the bays are on
    d = _unit(np.linalg.svd(fronts - fronts.mean(axis=0))[2][0])
    if d @ (fronts[-1] - fronts[0]) < 0:
        d = -d
    nrm = np.array([-d[1], d[0]])
    row_residual = float(np.abs((fronts - fronts.mean(axis=0)) @ nrm).max())
    if float(np.median((backs - fronts) @ nrm)) < 0:
        nrm = -nrm
    if row_residual > ROW_RESIDUAL_M:
        warnings.append(f"the front points stray {row_residual:.2f} m from one straight row: check that every line was marked at its front end")

    # the angle: each line's own, then the median of the plausible ones
    vec = backs - fronts
    across, along = vec @ nrm, vec @ d
    plausible = (across >= MIN_DEPTH_M) & (across <= MAX_DEPTH_M)
    for j in np.flatnonzero(~plausible):
        skipped.append((keep[j], f"its length across the row is {across[j]:.1f} m (expected {MIN_DEPTH_M:.0f} to {MAX_DEPTH_M:.0f})"))
    if not plausible.any():
        raise DeriveError("no separator line has a plausible length on the ground: the front / back points may be swapped or the calibration is off")
    ang = np.degrees(np.arctan2(across, along))
    # the centre of the biggest group of lines whose angles agree (within ANGLE_OUTLIER_DEG): the plain median fails when two lines disagree wildly
    cand = ang[plausible]
    med = float(np.median(cand))
    centre = max(cand, key=lambda a_: (int((np.abs(cand - a_) <= ANGLE_OUTLIER_DEG).sum()), -abs(float(a_) - med)))
    close = plausible & (np.abs(ang - centre) <= ANGLE_OUTLIER_DEG)
    med = float(centre)
    for j in np.flatnonzero(plausible & ~close):
        skipped.append((keep[j], f"its angle {ang[j]:.0f} degrees is far from the median {med:.0f}"))
    stall = float(np.median(ang[close]))
    if not MIN_STALL_DEG <= stall <= MAX_STALL_DEG:
        raise DeriveError(f"the bay angle {stall:.0f} degrees is outside {MIN_STALL_DEG:.0f} to {MAX_STALL_DEG:.0f}")
    spread = float(np.ptp(ang[close]))
    if spread > ANGLE_WARN_DEG:
        warnings.append(f"the lines' angles spread {spread:.0f} degrees ({ang[close].min():.0f} to {ang[close].max():.0f}): the median is used")

    # spacing and count, from the front points (they all lie on the row's front line)
    order = np.argsort(fronts @ d)
    u = (fronts @ d)[order]
    gaps = np.diff(u)
    gaps = gaps[gaps > 0.3]  # two points on one line (a duplicate) are not a gap
    if not len(gaps):
        raise DeriveError("the lines are all in one place: no spacing")
    step = _bay_step(gaps)
    bays = np.maximum(np.round(gaps / step), 1.0)  # a missed line: that gap holds 2 bays
    step = float(gaps.sum() / bays.sum())
    if np.abs(gaps / step - bays).max() > GAP_TOL:
        warnings.append("the gaps between lines are uneven: a line may be missing or misplaced")
    if not MIN_STEP_M <= step <= MAX_STEP_M:
        raise DeriveError(f"the lines are {step:.2f} m apart on the ground (expected {MIN_STEP_M} to {MAX_STEP_M} m): not a row of bays")
    length = float(u[-1] - u[0])
    count = int(round(length / step))
    if count > max_slots:
        warnings.append(f"{count} bays from the lines: limited to {max_slots}")
    total = max(1, min(count, max_slots))
    if proposal.estimated_count is not None and proposal.estimated_count != total:
        warnings.append(f"the model counted {proposal.estimated_count} bays, the lines give {total}: the lines are used")
    first = keep[int(order[0])]
    if any(i < first and why == "occluded" for i, why in skipped):
        warnings.append("a line listed before line 0 was occluded and not used: line 0 is the first usable line")

    f0, b0 = full[first]
    params = SlotCalibrationParameters(stall_angle=round(stall, 2), line0_front_dot=(float(f0[0]), float(f0[1])),
                                       line0_back_dot=(float(b0[0]), float(b0[1])), total_slots=total, slot_prefix=slot_prefix)
    used = [int(j) for j in np.flatnonzero(close)]
    front_dots = [(float(full[keep[j]][0][0]), float(full[keep[j]][0][1]), int(round(((fronts[j] @ d) - u[0]) / step))) for j in used]  # (x, y, line number)
    info = {"depth_m": round(float(np.median(across[close])), 3), "front_dots": front_dots, "stall_deg": round(stall, 2), "angles_deg": [round(float(a), 2) for a in ang[close]], "step_m": round(step, 3),
            "row_length_m": round(length, 3), "count_from_lines": count, "row_residual_m": round(row_residual, 3),
            "used": [keep[j] for j in used], "skipped": skipped, "warnings": warnings}
    return params, info
