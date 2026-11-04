"""Vehicle appearance signature: is the car in this bay still the car that parked there? Used by the parked lock (evidence.py) to veto a
ghost exit, and later to reopen a session after a short gap (occlusion roadmap 4.5).

A Signature is two tiny fingerprints of a vehicle crop, both cheap and neither a network:
  hist   HSV histogram, 16 x 4 x 4 bins (hue x saturation x value), L1-normalised: the car's colours, tolerant of pose and of box jitter
  thumb  32 x 16 grayscale thumbnail: the car's layout. Compared by NCC (zero-mean, unit-norm), so a gain / offset change of the whole
         picture (exposure, a cloud, auto-white-balance) does not matter; the raw thumbnail is kept for thumb_diff (local motion).

Compared with hist_distance (Bhattacharyya: 0 = same colours, 1 = nothing in common) and thumb_ncc (1 = same layout, ~0 = unrelated).
same_vehicle() needs both to agree, plus the metric footprint size when both are known.

Nothing here runs per frame: a signature is made at a lock, at an exit candidacy or at a re-appearance (~0.2 ms on a car crop).
Takes crops as BGR uint8 arrays (or None) and boxes as polygons in fractions of the frame, like the rest of the package. numpy + cv2 only."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

HIST_BINS = (16, 4, 4)  # H, S, V
THUMB = (32, 16)  # width, height (cv2 order)
MIN_CROP = 8  # px: a smaller crop (either side) carries no usable signature
TRIM = 0.1  # share cut off each side of the crop before measuring: the box edge is mostly ground and neighbours
MAX_BHATTA = 0.35  # same_vehicle: colours further apart than this = another car
MIN_NCC = 0.6  # same_vehicle: layout correlating less than this = another car
SIZE_TOL = 0.2  # same_vehicle: metric footprint sizes differing by more than this share = another car
V_MEAN = 128.0  # the mean brightness the crop is scaled to before the histogram (exposure normalisation)
MAX_GAIN = 4.0  # ... by at most this factor: a near-black crop is noise, not worth amplifying
_FLAT = 1e-3 # a thumbnail with less spread than this (grey levels) is flat: it has no layout to correlate


@dataclass(frozen=True, eq=False)
class Signature:
    hist: np.ndarray  # float32 (prod(HIST_BINS),), sums to 1
    thumb: np.ndarray  # float32 (THUMB h, w), raw grey levels 0..255
    unit: np.ndarray  # float32 (h * w,), zero mean and unit norm (all zeros for a flat thumbnail)
    size_m: float | None = None  # footprint size in metres when the camera is calibrated


def make_signature(crop, size_m: float | None = None, trim: float = TRIM) -> Signature | None:
    """The signature of a BGR vehicle crop, or None when the crop is missing, not colour, or smaller than MIN_CROP px."""
    if crop is None or crop.ndim != 3 or crop.shape[2] != 3:
        return None
    h, w = crop.shape[:2]
    dy, dx = int(h * trim), int(w * trim)
    core = crop[dy:h - dy, dx:w - dx]
    if core.shape[0] < MIN_CROP or core.shape[1] < MIN_CROP:
        return None
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    # Exposure gain: a lighting dip scales V and would push the car across the 4 coarse V bins; scale it back to a fixed mean first.
    gain = min(V_MEAN / max(float(hsv[..., 2].mean()), 1.0), MAX_GAIN)
    hsv[..., 2] = np.minimum(hsv[..., 2] * gain, 255).astype(np.uint8)
    hist =cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256]).ravel().astype(np.float32)
    total = float(hist.sum())
    if total <= 0:
        return None
    hist /= total
    thumb = cv2.resize(cv2.cvtColor(core, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
    flat = thumb.ravel() - float(thumb.mean())
    norm = float(np.linalg.norm(flat))
    unit = flat / norm if norm > _FLAT else np.zeros_like(flat)
    return Signature(hist, thumb, unit, size_m)


def hist_distance(a: Signature, b: Signature) -> float:
    """Bhattacharyya distance of the colour histograms: 0 same .. 1 disjoint."""
    return float(cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_BHATTACHARYYA))


def thumb_ncc(a: Signature, b: Signature) -> float:
    """Normalised cross-correlation of the thumbnails (-1..1). 0 when either is flat: nothing to correlate is no evidence of a match."""
    if not a.unit.any() or not b.unit.any():
        return 0.0
    return float(np.clip(np.dot(a.unit, b.unit), -1.0, 1.0))


def thumb_diff(a: Signature, b: Signature) -> float:
    """Mean absolute difference of the raw thumbnails in grey levels: how much the pixels under the box changed (local motion)."""
    return float(np.abs(a.thumb - b.thumb).mean())


def same_vehicle(a: Signature | None, b: Signature | None, max_bhatta: float = MAX_BHATTA, min_ncc: float = MIN_NCC,
                 size_tol: float = SIZE_TOL) -> bool:
    """True when colours, layout and (if both known) metric size all agree. False if either signature is missing: callers that treat
    "no signature" as "no opinion" must check for None first."""
    if a is None or b is None:
        return False
    if a.size_m and b.size_m and abs(a.size_m - b.size_m) > size_tol * max(a.size_m, b.size_m):
        return False
    return hist_distance(a, b) <= max_bhatta and thumb_ncc(a, b) >= min_ncc


def crop_box(frame: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    """The part of the frame under a box (polygon in fractions), grown by `pad` of its size on each side. A view, not a copy; may be empty."""
    h, w = frame.shape[:2]
    p = np.asarray(box, np.float64)  # float64 + round: 0.2 * 100 must be pixel 20, not 20.000000000000004 -> 21
    (x1, y1), (x2, y2) = p.min(0), p.max(0)
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    xa, xb = int(min(max(round((x1 - dx) * w), 0), w)), int(min(max(round((x2 + dx) * w), 0), w))
    ya, yb = int(min(max(round((y1 - dy) * h), 0), h)), int(min(max(round((y2 + dy) * h), 0), h))
    return frame[ya:yb, xa:xb]
