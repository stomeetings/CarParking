"""Stream health (Phase A, item 105): is the picture the worker counts from still live? Three states, and only "active" may change a count.

  active   new frames arrive and the scene is not a repeated picture
  stale    no new frame for STALE_S (the camera or the network is down; the reader is reconnecting)
  frozen   frames arrive but are the same picture (many NVRs repeat the last good frame when the camera drops): the decoder
           reports a healthy stream, so only the pixels can tell

A frozen picture is detected from a 64x36 grayscale thumbnail of every frame, compared with an ANCHOR (the first thumbnail of the
quiet stretch), not with the previous frame: a repeated frame is identical to the anchor for as long as it lasts, while a live
scene in a quiet lot drifts away from it (exposure, shadows, noise) even if no two consecutive frames differ much. The anchor moves
on whenever a frame differs from it by FREEZE_DIFF grey levels (mean absolute difference); FREEZE_S seconds without that = frozen.

Read the value before trusting the default: Worker publishes "stream.diff" (the latest difference from the anchor) in the snapshot, so a
quiet night can be checked on the real camera first (CARPARK_FREEZE_DIFF / CARPARK_FREEZE_S; FREEZE_S 0 turns the check off).
numpy + cv2 only."""
from __future__ import annotations

import cv2
import numpy as np

STALE_S = 10.0  # seconds without a new frame before the stream counts as stale
FREEZE_DIFF = 0.5  # grey levels (mean absolute difference of the 64x36 thumbnails): below it = the same picture
FREEZE_S = 20.0  # ... for this long = frozen
THUMB = (64, 36)

ACTIVE, STALE, FROZEN = "active", "stale", "frozen"


def thumbnail(frame: np.ndarray) -> np.ndarray:
    """The frame as a THUMB-sized float grayscale picture."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


class FreezeDetector:
    """update(frame, t) -> True while the picture has been the same for freeze_s seconds of capture time (t). Not thread-safe: one per camera."""

    def __init__(self, diff: float = FREEZE_DIFF, freeze_s: float = FREEZE_S):
        self.diff_max, self.freeze_s = diff, freeze_s
        self.anchor: np.ndarray | None = None
        self.since = 0.0  # capture time of the anchor
        self.diff = 0.0  # the latest frame's difference from the anchor
        self.frozen_for = 0.0

    def reset(self):
        self.anchor, self.diff, self.frozen_for = None, 0.0, 0.0

    def update(self, frame: np.ndarray, t: float) -> bool:
        if self.freeze_s <= 0:
            return False
        thumb = thumbnail(frame)
        if self.anchor is None or thumb.shape != self.anchor.shape:
            self.anchor, self.since, self.diff = thumb, t, 0.0
        else:
            self.diff = float(np.abs(thumb - self.anchor).mean())
            if self.diff >= self.diff_max:
                self.anchor, self.since = thumb, t  # the scene moved: a new quiet stretch starts here
        self.frozen_for = max(t - self.since, 0.0)
        return self.frozen_for >= self.freeze_s


def health_from_env(env) -> "StreamHealth":
    """StreamHealth with CARPARK_STALE_S, CARPARK_FREEZE_DIFF and CARPARK_FREEZE_S from the environment (a bad value = the default;
    CARPARK_FREEZE_S=0 turns the frozen-picture check off)."""
    def num(name, default):
        try:
            return float(env.get(name, default))
        except (TypeError, ValueError):
            return default
    return StreamHealth(num("CARPARK_STALE_S", STALE_S),
                        FreezeDetector(num("CARPARK_FREEZE_DIFF", FREEZE_DIFF), num("CARPARK_FREEZE_S", FREEZE_S)))


class StreamHealth:
    """One per camera session. observe(frame, seq, t, now) -> ACTIVE / STALE / FROZEN (call it every worker step; the thumbnail is
    only taken for a frame not seen before). info() is the snapshot's "stream" entry. Before the first frame the state is ACTIVE
    (the reader's own status tells "connecting")."""

    def __init__(self, stale_s: float = STALE_S, freeze: FreezeDetector | None = None):
        self.stale_s = stale_s
        self.freeze = freeze if freeze is not None else FreezeDetector()
        self.state, self.age, self._seq, self._frozen, self._t = ACTIVE, 0.0, 0, False, 0.0
        self.since: float | None = None  # `now` when the current non-active state began

    def observe(self, frame, seq: int, t: float, now: float) -> str:
        if frame is not None:
            if seq != self._seq:
                if self._t and t - self._t > self.stale_s:
                    self.freeze.reset()  # a gap in the stream: the quiet stretch before it says nothing about the picture after it
                self._seq, self._t = seq, t
                self._frozen = self.freeze.update(frame, t)
            self.age = max(now - t, 0.0)
        if frame is None:
            state = ACTIVE
        elif self.age > self.stale_s:
            state = STALE
        else:
            state = FROZEN if self._frozen else ACTIVE
        if state != self.state:
            self.since = None if state == ACTIVE else now
        self.state = state
        return state

    def info(self, now: float) -> dict:
        age = self.age if self._t == 0.0 else max(now - self._t, 0.0)  # as of `now`, not of the last observe()
        return {"state": self.state, "age_s": round(age, 1),
                "for_s": 0.0 if self.since is None else round(max(now - self.since, 0.0), 1),
                "diff": round(self.freeze.diff, 3), "same_s": round(self.freeze.frozen_for, 1)}
