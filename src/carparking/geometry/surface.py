"""The parking surface from a segmentation mask (Phase 2): SAM returns the ground it was clicked on with holes where cars
stand; the surface that counts for capacity includes the bays under those cars. numpy + cv2 only."""
from __future__ import annotations

import cv2
import numpy as np

CONTACT_STRIP = 0.25  # the lowest quarter of a vehicle box is where it touches the ground
CONTACT_SHARE = 0.3  # a vehicle stands on the surface when this much of that strip is (next to) the mask


BRIDGE = 0.035  # share of the image width: gaps up to this are closed (a pole, a plant pot, a kerb stone)


def mask_to_polygon(mask, boxes=(), epsilon: float = 0.004, min_area: float = 0.005, bridge: float = BRIDGE) -> dict | None:
    """Binary mask (HxW) + vehicle boxes (x1, y1, x2, y2 pixels) -> {"polygon": [[x, y]...] normalised 0-1,
    "area_frac": share of the image} or None when nothing usable is left.

    Steps: (1) a vehicle whose contact strip lies on the mask is filled (a car on the lot is lot); (2) closing over a
    gap of ~1 % of the image width; (3) a second closing over `bridge` x the width joins pieces a thin obstacle splits
    (live camera-1: a steel pole cuts the pavement in two); (4) the largest connected piece only; (5) holes filled (outer
    contour); (6) the contour simplified by `epsilon` x its perimeter."""
    h, w = mask.shape[:2]
    m = (np.asarray(mask) > 0).astype(np.uint8)
    reach = cv2.dilate(m, np.ones((max(3, w // 100) | 1,) * 2, np.uint8))
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        top = int(y2 - CONTACT_STRIP * (y2 - y1))
        strip = reach[top:y2, x1:x2]
        if strip.size and strip.mean() >= CONTACT_SHARE:
            m[y1:y2, x1:x2] = 1
    k = max(3, w // 100) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    kb = max(k, int(bridge * w)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == biggest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) / (w * h)
    if area < min_area:
        return None
    c = cv2.approxPolyDP(c, epsilon * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return {"polygon": [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in c], "area_frac": round(area, 4)}
