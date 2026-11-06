"""A slot-id label raster: O(1) "which bay is this point in" for every detection, instead of a point-in-polygon test over every slot.
Built once per layout version at 1/4 resolution; the first bay in the list keeps a pixel two bays share (the lists are already de-duplicated)."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


class SlotMap:
    def __init__(self, slots: list[dict], w: int, h: int, scale: float = 0.25):
        self.w, self.h = w, h
        self.sw, self.sh = max(1, round(w * scale)), max(1, round(h * scale))
        self.ids = [s["id"] for s in slots]
        self.label = np.zeros((self.sh, self.sw), np.int32)  # 0 = no bay; k = bay k - 1
        for k in range(len(slots) - 1, -1, -1):  # the first bay is drawn last: it wins a shared pixel
            p = np.round(np.asarray(slots[k]["poly"], float) * (self.sw, self.sh)).astype(np.int32)
            cv2.fillPoly(self.label, [p], k + 1)
        self.version = hashlib.sha1(repr([(s["id"], s["poly"]) for s in slots]).encode()).hexdigest()[:12]

    def lookup(self, x: float, y: float) -> int | None:
        """Slot id at a point of the picture given as fractions, or None."""
        i, j = int(y * self.sh), int(x * self.sw)
        if not (0 <= x < 1 and 0 <= y < 1 and 0 <= i < self.sh and 0 <= j < self.sw):
            return None
        k = int(self.label[i, j])
        return self.ids[k - 1] if k else None

    def lookup_many(self, pts) -> list[int | None]:
        p = np.asarray(pts, float).reshape(-1, 2)
        i = np.clip((p[:, 1] * self.sh).astype(int), 0, self.sh - 1)
        j = np.clip((p[:, 0] * self.sw).astype(int), 0, self.sw - 1)
        inside = (p[:, 0] >= 0) & (p[:, 0] < 1) & (p[:, 1] >= 0) & (p[:, 1] < 1)
        k = self.label[i, j]
        return [self.ids[a - 1] if ok and a else None for a, ok in zip(k, inside)]
