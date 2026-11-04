"""Fabric.js canvas JSON <-> slot polygons (fractions of the frame).

streamlit-drawable-canvas (0.13, Fabric.js v6) returns the drawing as Fabric's JSON, and a web frontend drawing with
Fabric would send the same. Only plain dicts here; no Streamlit.
* fabric_to_polys(): every Polygon / Rect object -> its corners in canvas pixels -> fractions of the canvas, using
  Fabric's own object transform (origin, position, scale, rotation; polygon points are relative to pathOffset), so a
  shape that was moved, resized, rotated or had a corner dragged comes back where it is shown.
* polys_to_fabric(): the inverse, for initial_drawing: every slot as an editable Polygon.
"""
from __future__ import annotations

import math

import numpy as np

SHAPES = {"polygon", "rect"}  # Fabric type names, compared lower-case (v6 writes "Polygon" / "Rect")
_ORIGIN = {"left": 0.0, "top": 0.0, "center": 0.5, "right": 1.0, "bottom": 1.0}


def _origin(v) -> float:
    return float(v) if isinstance(v, (int, float)) else _ORIGIN.get(v, 0.0)


def object_points(obj: dict) -> np.ndarray | None:
    """Corners of a Fabric Polygon or Rect in canvas pixels (N x 2), None for other object types."""
    kind = str(obj.get("type", "")).lower()
    if kind not in SHAPES:
        return None
    sx, sy = float(obj.get("scaleX", 1) or 1), float(obj.get("scaleY", 1) or 1)
    if obj.get("flipX"):
        sx = -sx
    if obj.get("flipY"):
        sy = -sy
    sw = float(obj.get("strokeWidth", 0) or 0) if obj.get("stroke") else 0.0
    if kind == "rect":
        w, h = float(obj.get("width", 0)), float(obj.get("height", 0))
        local = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)], float)
    else:
        pts = np.array([(p["x"], p["y"]) for p in obj.get("points", [])], float)
        if len(pts) < 3:
            return None
        lo, hi = pts.min(0), pts.max(0)
        w, h = hi - lo
        local = pts - (lo + hi) / 2  # Fabric's pathOffset: the centre of the points' bounding box
    # Fabric's bounding box includes the stroke: (size + stroke) * scale, or size * scale + stroke when uniform
    if obj.get("strokeUniform"):
        bw, bh = abs(w * sx) + sw, abs(h * sy) + sw
    else:
        bw, bh = abs((w + sw) * sx), abs((h + sw) * sy)
    a = math.radians(float(obj.get("angle", 0) or 0))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    # (left, top) is the origin point; the centre is offset from it in the object's rotated frame
    to_centre = np.array([(0.5 - _origin(obj.get("originX", "left"))) * bw,
                          (0.5 - _origin(obj.get("originY", "top"))) * bh])
    centre = np.array([float(obj.get("left", 0)), float(obj.get("top", 0))]) + rot @ to_centre
    return centre + (local * (sx, sy)) @ rot.T


def fabric_to_polys(json_data: dict | None, width: float, height: float, min_area: float = 2e-4):
    """Canvas JSON -> (polygons as fractions [[x, y], ...] clipped to the frame, skipped) where skipped counts objects
    that are not a polygon/rectangle (lines, free drawing, ...) or are smaller than min_area of the frame (a click
    with the rectangle tool)."""
    polys, skipped = [], 0
    for obj in (json_data or {}).get("objects", []):
        pts = object_points(obj)
        if pts is None:
            skipped += 1
            continue
        frac = np.clip(pts / (width, height), 0, 1)
        x, y = frac[:, 0], frac[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < min_area:
            skipped += 1
            continue
        polys.append([[round(float(a), 4), round(float(b), 4)] for a, b in frac])
    return polys, skipped


def object_poly(obj: dict, width: float, height: float) -> list[list[float]] | None:
    """One Fabric object -> its polygon as fractions of the canvas (clipped), None if not a polygon/rectangle."""
    pts = object_points(obj)
    if pts is None:
        return None
    return [[round(float(a), 4), round(float(b), 4)] for a, b in np.clip(pts / (width, height), 0, 1)]


def polys_to_fabric(polys, width: float, height: float, stroke: str = "#00ff00", fill: str = "rgba(0,255,0,0.15)",
                    stroke_width: int = 3, locked: bool = False) -> dict:
    """Polygons (fractions) -> initial_drawing {"objects": [Polygon, ...]}; fabric_to_polys() reads them back.
    locked: shown for reference only (cannot be selected, moved or reshaped)."""
    objects = []
    for poly in polys:
        pts = np.asarray(poly, float) * (width, height)
        lo, hi = pts.min(0), pts.max(0)
        obj = {"type": "Polygon", "points": [{"x": float(x), "y": float(y)} for x, y in pts],
               "left": float(lo[0] - stroke_width / 2), "top": float(lo[1] - stroke_width / 2),
               "width": float(hi[0] - lo[0]), "height": float(hi[1] - lo[1]),
               "originX": "left", "originY": "top", "scaleX": 1, "scaleY": 1, "angle": 0,
               "stroke": stroke, "strokeWidth": stroke_width, "fill": fill, "objectCaching": False}
        if locked:
            obj.update(selectable=False, evented=False, hasControls=False, lockMovementX=True, lockMovementY=True)
        objects.append(obj)
    return {"objects": objects}
