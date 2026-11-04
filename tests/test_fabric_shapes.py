"""Fabric.js canvas JSON <-> slot polygons (carparking/fabric_shapes.py) and relabel() of drawn shapes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.fabric_shapes import fabric_to_polys, object_points, polys_to_fabric  # noqa: E402
from carparking.fixed_slots import relabel, validate_slots  # noqa: E402

W, H = 960, 540


def rect(**kw):  # what the canvas' rectangle tool creates (origin top-left, uniform stroke 3)
    return {"type": "Rect", "left": 100, "top": 50, "width": 200, "height": 100, "originX": "left", "originY": "top",
            "scaleX": 1, "scaleY": 1, "angle": 0, "stroke": "#00ff00", "strokeWidth": 3, "strokeUniform": True, **kw}


def polygon(points, sw=3, **kw):  # what the polygon tool creates: absolute points, Fabric sets left/top from them
    p = np.array(points, float)
    return {"type": "Polygon", "points": [{"x": x, "y": y} for x, y in p], "left": p[:, 0].min() - sw / 2,
            "top": p[:, 1].min() - sw / 2, "width": np.ptp(p[:, 0]), "height": np.ptp(p[:, 1]), "originX": "left",
            "originY": "top", "scaleX": 1, "scaleY": 1, "angle": 0, "stroke": "#00ff00", "strokeWidth": sw, **kw}


def test_rectangle_tool_and_its_edits():
    # stroke centre line: the box starts half a stroke inside (left, top)
    assert np.allclose(object_points(rect()), [(101.5, 51.5), (301.5, 51.5), (301.5, 151.5), (101.5, 151.5)])
    # resized to double width with the edit toggle (uniform stroke: not scaled)
    assert np.allclose(object_points(rect(scaleX=2))[:, 0], [101.5, 501.5, 501.5, 101.5])
    # rotated 90 degrees around its top-left origin: now hangs to the left of x=100
    r = object_points(rect(angle=90))
    assert np.allclose(r.min(0), (-1.5, 51.5)) and np.allclose(r.max(0), (98.5, 251.5))
    # centre origin (other Fabric versions / frontends)
    assert np.allclose(object_points(rect(originX="center", originY="center", left=201.5, top=101.5)),
                       object_points(rect()))


def test_polygon_tool_moved_and_scaled():
    corners = [(300, 200), (520, 210), (540, 330), (280, 320)]  # an angled bay, 4 clicks
    assert np.allclose(object_points(polygon(corners)), corners)
    moved = polygon(corners)
    moved["left"] += 40
    moved["top"] -= 10
    assert np.allclose(object_points(moved), np.array(corners) + (40, -10))
    scaled = polygon(corners, scaleX=2)  # stretched from its left edge
    # the outer left edge (278.5) stays put; the stroke (not uniform) doubles too, so the centre line is 3 px inside
    assert np.isclose(object_points(scaled)[:, 0].min(), 278.5 + 1.5 * 2)
    assert np.isclose(np.ptp(object_points(scaled)[:, 0]), 2 * 260)


def test_round_trip_and_skipping():
    polys = [[[0.1, 0.2], [0.4, 0.2], [0.4, 0.5], [0.1, 0.5]], [[0.5, 0.3], [0.8, 0.35], [0.75, 0.6], [0.45, 0.55]]]
    drawing = polys_to_fabric(polys, W, H)
    tiny = rect(width=4, height=4)  # a click with the rectangle tool
    line = {"type": "Line", "x1": 0, "y1": 0, "x2": 10, "y2": 10}
    back, skipped = fabric_to_polys({"objects": drawing["objects"] + [tiny, line]}, W, H)
    assert np.allclose(back, polys, atol=1e-4) and skipped == 2
    off = fabric_to_polys({"objects": [rect(left=900, width=200)]}, W, H)[0][0]  # dragged past the right edge
    assert max(x for x, _ in off) == 1.0
    validate_slots([{"id": i + 1, "poly": p} for i, p in enumerate(back)])
    assert fabric_to_polys(None, W, H) == ([], 0)


def test_relabel_keeps_ids_while_reshaping():
    a = {"id": 7, "name": "A1", "poly": [[0.1, 0.2], [0.4, 0.2], [0.4, 0.5], [0.1, 0.5]]}
    b = {"id": 3, "name": "B", "poly": [[0.5, 0.2], [0.8, 0.2], [0.8, 0.5], [0.5, 0.5]]}
    moved_a = [[0.12, 0.22], [0.42, 0.22], [0.42, 0.52], [0.12, 0.52]]
    new = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]
    out = relabel([new, moved_a], [a, b])  # b was deleted on the canvas, a moved, one shape added
    assert [(s["id"], s["name"]) for s in out] == [(8, "S8"), (7, "A1")] and out[1]["poly"] == moved_a
    assert [s["id"] for s in relabel([new, new], [])] == [1, 2]


def test_object_poly_and_locked_reference_shapes():
    from carparking.fabric_shapes import object_poly
    poly = [[0.1, 0.2], [0.4, 0.2], [0.4, 0.5], [0.1, 0.5]]
    obj = polys_to_fabric([poly], W, H)["objects"][0]
    assert np.allclose(object_poly(obj, W, H), poly, atol=1e-4) and object_poly({"type": "Line"}, W, H) is None
    locked = polys_to_fabric([poly], W, H, locked=True)["objects"][0]
    assert locked["selectable"] is False and locked["evented"] is False and "selectable" not in obj
