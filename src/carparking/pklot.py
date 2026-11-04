"""PKLot (CC BY 4.0) XML annotations -> YOLO label lines. Class ids: 0 busy_slot, 1 free_slot.

Two label formats from the same polygons:
  * yolo_lines: axis-aligned 'cls cx cy w h' (the original model)
  * obb_lines:  oriented 'cls x1 y1 x2 y2 x3 y3 x4 y4' (Ultralytics OBB), tight around diagonal bays
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import numpy as np

NAMES = ["busy_slot", "free_slot"]  # order = YOLO class id; yolo_engine.state_of maps these names to BUSY/FREE


def parse_spaces(xml_path) -> list[tuple] | None:
    """-> [(class_id, points Nx2 float array)] in pixels, or None if any space has no label
    (training on it would teach the model that an unlabelled bay is background)."""
    out = []
    for sp in ET.parse(xml_path).getroot().iter("space"):
        occ = sp.get("occupied")
        if occ not in ("0", "1"):
            return None
        pts = [(float(p.get("x")), float(p.get("y"))) for p in sp.iter("point")]
        if not pts:  # no contour: fall back to the rotated rectangle
            c, s, a = (sp.find(f"rotatedRect/{t}") for t in ("center", "size", "angle"))
            pts = cv2.boxPoints(((float(c.get("x")), float(c.get("y"))),
                                 (float(s.get("w")), float(s.get("h"))), float(a.get("d"))))
        out.append((0 if occ == "1" else 1, np.asarray(pts, dtype=float)))
    return out


def yolo_lines(spaces, w: int, h: int) -> list[str]:
    """Polygons -> 'cls cx cy bw bh' (normalised, clipped to the image); boxes fully outside are dropped."""
    lines = []
    for c, pts in spaces:
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 > x0 and y1 > y0:
            lines.append(f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def obb_corners(pts) -> np.ndarray:
    """Polygon -> 4 corners of its minimum-area rotated rectangle. Degenerate contours (e.g. 2 points, a diagonal)
    fall back to their axis-aligned box, so a bay is never turned into a zero-width line."""
    (cx, cy), (rw, rh), a = cv2.minAreaRect(np.asarray(pts, dtype=np.float32))
    if rw * rh > 0:
        return cv2.boxPoints(((cx, cy), (rw, rh), a)).astype(float)
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def obb_lines(spaces, w: int, h: int) -> list[str]:
    """Polygons -> 'cls x1 y1 x2 y2 x3 y3 x4 y4' (normalised, corners clipped to the image); bays that end up with
    no area inside the image are dropped. Ultralytics re-fits a rotated box to the clipped corners when loading."""
    lines = []
    for c, pts in spaces:
        q = np.clip(obb_corners(pts), 0, [w, h])
        if cv2.contourArea(q.astype(np.float32)) > 0:
            lines.append(f"{c} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in q))
    return lines
