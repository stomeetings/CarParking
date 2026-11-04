"""Parking-bay geometry: load bay annotations and scale them to the frame being processed.

A bay is a polygon: an (N, 2) float array of (x, y) pixel corners, origin top-left, y pointing down.
CNR-EXT annotates bays as axis-aligned boxes, so every CNR bay is a 4-corner polygon.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# CNR-EXT camera CSVs give pixel coordinates in the original 2592x1944 camera image,
# while the distributed full frames are downsampled to 1000x750 (dataset docs).
CNR_SOURCE_SIZE = (2592, 1944)  # (width, height)


def box(x, y, w, h) -> np.ndarray:
    """Axis-aligned box -> 4-corner polygon (clockwise from top-left)."""
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def load_slots_json(path) -> dict[int, np.ndarray]:
    """slots.json: [{"x":..,"y":..,"w":..,"h":..[, "id":..]}, ...] in frame pixels.

    Missing ids default to the 1-based list position, so keep the list order fixed once ids matter.
    """
    bays = {}
    for i, s in enumerate(json.loads(Path(path).read_text()), 1):
        slot = int(s.get("id", i))
        if slot in bays:
            raise ValueError(f"duplicate slot id {slot} in {path}")
        bays[slot] = box(*(float(s[k]) for k in ("x", "y", "w", "h")))
    if not bays:
        raise ValueError(f"no slots in {path}")
    return bays


def demo_grid(frame_shape, rows=2, cols=6) -> dict[int, np.ndarray]:
    """Placeholder boxes so the pipeline runs without slots.json. These are NOT real bays."""
    h, w = frame_shape[:2]
    cw, ch = w / cols, h / (rows + 2)
    return {r * cols + c + 1: box(c * cw + 0.1 * cw, (r + 1) * ch + 0.1 * ch, 0.8 * cw, 0.8 * ch)
            for r in range(rows) for c in range(cols)}


def load_cnr_bays(csv_path) -> dict[int, np.ndarray]:
    """Read a CNR-EXT cameraN.csv (SlotId,X,Y,W,H) -> {slot_id: 4x2 polygon}, in source pixels, sorted by id."""
    bays = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            slot = int(row["SlotId"])
            x, y, w, h = (float(row[k]) for k in ("X", "Y", "W", "H"))
            if slot in bays:
                raise ValueError(f"duplicate SlotId {slot} in {csv_path}")
            bays[slot] = box(x, y, w, h)
    if not bays:
        raise ValueError(f"no bays in {csv_path}")
    return dict(sorted(bays.items()))


def scale_bays(bays: dict[int, np.ndarray], frame_shape, source_size=CNR_SOURCE_SIZE) -> dict[int, np.ndarray]:
    """Map polygons from source-image pixels to a resized frame of shape (h, w[, c]).

    Scale factors come from the real frame size, not a hard-coded 1000x750:
    for CNR-EXT, 1000/2592 == 750/1944 == 0.3858.
    """
    h, w = frame_shape[:2]
    s = np.array([w / source_size[0], h / source_size[1]])
    return {slot: poly * s for slot, poly in bays.items()}
