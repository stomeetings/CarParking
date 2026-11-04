"""Non-vehicle filter (post-demo item 2, first layer): a detection the detector calls a vehicle that is really clutter must not make a bay
BUSY. Pure functions on a box and (optionally) its pixels; FixedSlotTracker applies them before the frame's matching when its option
`obstacle_filter` is on (default 0 = off, and then nothing here runs). It rejects only; it never adds a detection.

A box is rejected, first reason that applies:
  too_small   its area is below `min_area_share` of the space it stands in (the bay's area, divided by the bay's capacity for a pin): a
              cone, a bin, a bag, a person's torso. Judged against the bay, not the frame: a car at the far end of the row is a few
              hundred pixels, a cone beside the camera is bigger. Skipped for a box that stands in no bay.
  aspect      width / height outside [min_aspect, max_aspect]: a standing person, a pole, a bollard (narrow); a fence, a kerb run (wide).
  pylon       the box is not a whole vehicle (area below `color_size` of its bay) and `color_share` or more of its pixels are the
              saturated orange of a traffic cone (hue band, saturation and value floors). A full-size orange car is never rejected on colour.
  shadow      most of the crop is dark (`dark_share` of the pixels under `dark_val`) and flat (mean grey gradient under `flat_grad`, camera-1 vehicles 11 .. 21): the
              shadow of a neighbour or a building, not a car (a black car still has windscreen and wheel edges).
The colour breakdown (orange / dark / bright-low-saturation / other, HSV on a 48-px-wide copy of the crop) is also returned for diagnostics.

Every number lives in ObstacleCfg and none is a camera calibration: the defaults were checked against the real vehicle boxes of the
camera-1 pixel recording (none of them is rejected, see ACTIVITY_LOG) and against synthetic clutter; there is no real footage of cones,
bins or shadows yet, so the clutter side is only as good as the synthetic scenes. Replay a recording (scripts/replay_eval.py) before
switching it on anywhere real."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ObstacleCfg:
    min_area_share: float = 0.25  # box area / its bay's area per space below this: too small to be a vehicle in that bay
    min_aspect: float = 0.5  # width / height of a vehicle box (cars on camera-1: 0.65 .. 3.2)
    max_aspect: float = 4.5
    pylon_hue: tuple[int, int] = (4, 20)  # OpenCV hue (0..179) of a safety cone's orange
    pylon_sat: int = 150  # a cone is strongly saturated
    pylon_val: int = 110
    color_share: float = 0.45  # share of pylon-coloured pixels that makes a not-whole-vehicle box a cone
    color_size: float = 0.60  # a box with at least this share of its bay's area is a whole vehicle, whatever its colour
    dark_val: int = 60  # a pixel this dark (V) counts towards a shadow
    dark_share: float = 0.75  # strict on purpose: wrongly rejecting a black car costs more than letting a faint shadow through
    flat_grad: float = 3.0  # mean |Sobel| / 8 of the grey crop below which it is flat (camera-1 vehicles: 11 .. 21)

    def __post_init__(self):
        if not (0 < self.min_area_share <= self.color_size <= 2 and 0 < self.min_aspect < self.max_aspect
                and 0 <= self.pylon_hue[0] < self.pylon_hue[1] <= 179 and 0 < self.color_share <= 1 and 0 < self.dark_share <= 1):
            raise ValueError("ObstacleCfg: needs 0 < min_area_share <= color_size, 0 < min_aspect < max_aspect, a hue band in 0..179, shares in (0, 1]")


CFG = ObstacleCfg()
SMALL = 48  # px width the crop is shrunk to before the colour statistics: a handful of ops whatever the box


def color_breakdown(crop: np.ndarray, cfg: ObstacleCfg = CFG) -> dict:
    """Shares of the crop's pixels (BGR image): 'orange' (cone orange), 'dark' (V < dark_val), 'bright' (white / grey / glare: low
    saturation, bright), 'other'; plus 'grad', the mean |Sobel| of the grey crop. They sum to ~1 (a pixel can be dark and orange: orange wins)."""
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return {"orange": 0.0, "dark": 0.0, "bright": 0.0, "other": 1.0, "grad": 0.0}
    small = cv2.resize(crop, (SMALL, max(2, round(SMALL * h / w))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    orange = (hue >= cfg.pylon_hue[0]) & (hue <= cfg.pylon_hue[1]) & (sat >= cfg.pylon_sat) & (val >= cfg.pylon_val)
    dark = (val < cfg.dark_val) & ~orange
    bright = (sat < 60) & (val >= 160) & ~orange
    n = hue.size
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grad = float((np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1))).mean() / 8.0)
    o, d, b = orange.sum() / n, dark.sum() / n, bright.sum() / n
    return {"orange": float(o), "dark": float(d), "bright": float(b), "other": float(1 - o - d - b), "grad": grad}


def judge(box, ref_area: float | None, crop_fn=None, cfg: ObstacleCfg = CFG) -> tuple[str | None, dict | None]:
    """box: (x1, y1, x2, y2) px. ref_area: px area of one space of the bay the box stands in, or None (it stands in no bay). crop_fn() ->
    the BGR crop, or None when no frame is at hand (the colour rules are then skipped). -> (reason or None, colour breakdown or None)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return "too_small", None
    share = None if not ref_area else w * h / ref_area
    if share is not None and share < cfg.min_area_share:
        return "too_small", None
    if not cfg.min_aspect <= w / h <= cfg.max_aspect:
        return "aspect", None
    if crop_fn is None:
        return None, None
    crop = crop_fn()
    if crop is None or crop.size == 0:
        return None, None
    cb = color_breakdown(crop, cfg)
    if share is not None and share < cfg.color_size and cb["orange"] >= cfg.color_share:
        return "pylon", cb
    if cb["dark"] >= cfg.dark_share and cb["grad"] < cfg.flat_grad:
        return "shadow", cb
    return None, cb


REASON_TEXT = {"too_small": "too small for the bay", "aspect": "not a vehicle shape", "pylon": "cone-coloured object", "shadow": "flat shadow"}


class RejectLog:
    """What the non-vehicle filter drops, as a few audit notices instead of one per frame: a `rejected` event for a (bay, reason) when the same
    kind of detection has been dropped for `hold` seconds and in at least `frames` frames, then silent for `cooldown` seconds (a cone that
    stays for a day is one row per 10 min at most). A (bay, reason) not seen for `gap` seconds starts over. Pure, frame-rate aware (seconds
    AND frames, the slower of the two). slot 0 = the detection stands in no bay."""

    def __init__(self, hold: float = 5.0, frames: int = 2, cooldown: float = 600.0, gap: float = 60.0):
        self.hold, self.frames, self.cooldown, self.gap = hold, frames, cooldown, gap
        self._seen: dict[tuple, list] = {}  # (slot, reason) -> [first time, last time, frames, last logged or None]

    def update(self, rejected: list[dict], now: float, names: dict | None = None) -> list[dict]:
        """rejected: FixedSlotTracker.rejected of this frame -> [{"slot", "event": "rejected", "time", "note"}]."""
        out, seen_now = [], set()
        for r in rejected:
            key = (r["slot"] if r["slot"] is not None else 0, r["reason"])
            if key in seen_now:
                continue  # one count per frame however many boxes
            seen_now.add(key)
            row = self._seen.get(key)
            if row is None or now - row[1] > self.gap:
                row = self._seen[key] = [now, now, 0, row[3] if row else None]
            row[1], row[2] = now, row[2] + 1
            if now - row[0] >= self.hold and row[2] >= self.frames and (row[3] is None or now - row[3] >= self.cooldown):
                row[3] = now
                x1, y1, x2, y2 = r["box"]
                where = f" in bay {(names or {}).get(key[0], key[0])}" if key[0] else " outside the bays"
                out.append({"slot": key[0], "event": "rejected", "time": now,
                            "note": f"{r['reason']}: {REASON_TEXT.get(r['reason'], r['reason'])}{where}, {x2 - x1}x{y2 - y1} px, not counted as a vehicle"})
        for key in [k for k, row in self._seen.items() if now - row[1] > max(self.cooldown, self.gap) and k not in seen_now]:
            del self._seen[key]
        return out
