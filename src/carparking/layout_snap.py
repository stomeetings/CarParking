"""Setup-time magnetic snap of a ground-metre layout to the painted lines of the picture (never at runtime: the slots stay fixed, drift is the drift module's job).

The zone's ground is warped to a bird's-eye image in the layout's own frame (x = u along the row, y = v across it, `ppm` pixels per metre), where the separators are
vertical lines. A column profile of the paint (bright thin strokes: white top-hat, plus a yellow hue gate) over the bays' depth gives the separators' positions; the
lattice's lines u_k = k * step are matched to the nearest peaks within `tol` of a bay and (u0, step) are refitted by least squares over the matches, so a missing or
occluded line does not matter and the spacing stays equal. Nothing is changed unless enough lines support the fit; the offsets of hand-edited bays are kept."""
from __future__ import annotations

import cv2
import numpy as np

from carparking.geometry.camera import to_image
from carparking.layout import ZoneLayout

PAD_M = 1.0


def birdseye(frame: np.ndarray, lay: ZoneLayout, cam, ppm: float = 60.0) -> tuple[np.ndarray, tuple[float, float]]:
    """The zone's ground as an image in the layout's (u, v) frame. -> (BGR image, (u, v) metres of its top-left pixel)."""
    uv = np.vstack([lay.cell(i) for i in range(lay.total)])
    lo, hi = uv.min(axis=0) - PAD_M, uv.max(axis=0) + PAD_M
    size = np.ceil((hi - lo) * ppm).astype(int)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    px = to_image(cam, lay.to_ground(corners))
    if not np.isfinite(px).all():
        raise ValueError("the zone is not on measurable ground")
    M = cv2.getPerspectiveTransform(np.float32(px), np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]))
    return cv2.warpPerspective(frame, M, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR), (float(lo[0]), float(lo[1]))


def paint_mask(bev: np.ndarray, ppm: float) -> np.ndarray:
    """Bright and yellow thin strokes (paint) as 0 / 255."""
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    k = max(3, int(0.25 * ppm) | 1)  # a stripe is ~10-15 cm wide: the hat is wider than that, narrower than a bay
    hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    white = (hat > max(25, float(np.percentile(hat, 92)))).astype(np.uint8) * 255
    yellow = cv2.inRange(cv2.cvtColor(bev, cv2.COLOR_BGR2HSV), (15, 90, 120), (35, 255, 255))
    return cv2.bitwise_or(white, yellow)


def snap_lattice(frame: np.ndarray, lay: ZoneLayout, cam, ppm: float = 60.0, tol_bays: float = 0.3, min_support: float = 0.4) -> tuple[ZoneLayout, dict]:
    """Snap the layout's separators to the painted lines. -> (the layout, possibly moved / re-spaced; a report
    {"snapped": bool, "support": share of the lines found, "matched": n, "du_m", "dstep_m", "reason"})."""
    rep = {"snapped": False, "support": 0.0, "matched": 0, "du_m": 0.0, "dstep_m": 0.0, "reason": ""}
    try:
        bev, (u_lo, v_lo) = birdseye(frame, lay, cam, ppm)
    except ValueError as e:
        rep["reason"] = str(e)
        return lay, rep
    mask = paint_mask(bev, ppm)
    total_v = lay.depth * lay.rows + lay.row_gap * (lay.rows - 1)
    r0 = int((0.15 * lay.depth - v_lo) * ppm)  # across the middle of the bays: the front / back ends hold other paint
    r1 = int((total_v - 0.15 * lay.depth - v_lo) * ppm)
    band = mask[max(0, r0):max(max(0, r0) + 1, min(mask.shape[0], r1))]
    prof = band.astype(float).sum(axis=0) / 255.0
    k = max(3, int(0.12 * ppm))
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    floor = 0.35 * max(float(prof.max()), 1.0)
    pu = np.array([u_lo + (i + 0.5) / ppm for i in range(1, len(prof) - 1) if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > floor])
    step = lay.width + lay.gap
    match_k, match_u = [], []
    for k_ in range(lay.count + 1):
        if len(pu) == 0:
            break
        j = int(np.argmin(np.abs(pu - k_ * step)))
        if abs(pu[j] - k_ * step) <= tol_bays * step:
            match_k.append(k_)
            match_u.append(float(pu[j]))
    rep["matched"], rep["support"] = len(match_k), round(len(match_k) / (lay.count + 1), 3)
    if len(match_k) < 2 or rep["support"] < min_support:
        rep["reason"] = f"only {len(match_k)} of {lay.count + 1} lines found ({rep['support']:.0%}): left as it was"
        return lay, rep
    u0, st = np.linalg.lstsq(np.c_[np.ones(len(match_k)), match_k], np.array(match_u), rcond=None)[0]
    if st <= 0.3 or abs(st - step) > tol_bays * step:
        rep["reason"] = "the lines give an implausible spacing: left as it was"
        return lay, rep
    d, _ = lay.axes()
    o = np.asarray(lay.origin, float) + d * u0
    out = lay.with_params(origin=(float(o[0]), float(o[1])), width=float(st - lay.gap))
    rep.update(snapped=True, du_m=round(float(u0), 4), dstep_m=round(float(st - step), 4))
    return out, rep
