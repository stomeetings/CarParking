import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carparking.dynamic_iou import intersection_area, iou, overlap  # noqa: E402
from carparking.yolo_engine import _iou_ios  # noqa: E402


def rect(x1, y1, x2, y2):
    return np.float32([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def exact(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def test_live_case_nearly_shared_edge_is_symmetric_and_sane():
    # Front Parking 02, 2026-10-22: slot 6 ends at x 0.5676, the small car box at the pillar at 0.5680.
    # cv2.intersectConvexConvex gave IoU 11.85 in one argument order: the matcher handed that car the pickup's slot
    slot6, small = rect(0.4383, 0.1431, 0.5676, 0.2139), rect(0.5344, 0.1569, 0.568, 0.1799)
    want = exact((0.4383, 0.1431, 0.5676, 0.2139), (0.5344, 0.1569, 0.568, 0.1799))
    assert abs(iou(slot6, small) - want) < 1e-4 and abs(iou(small, slot6) - want) < 1e-4
    assert abs(_iou_ios(small, slot6)[0] - want) < 1e-4 and _iou_ios(small, slot6)[1] <= 1.0


def test_overlap_matches_the_exact_iou_on_random_boxes_with_shared_edges():
    rng = np.random.default_rng(7)
    edges = np.round(rng.uniform(0, 1, 40), 4)  # few distinct coordinates: many shared / nearly shared edges
    for _ in range(3000):
        a = sorted(rng.choice(edges, 2, replace=False)), sorted(rng.choice(edges, 2, replace=False))
        b = sorted(rng.choice(edges, 2, replace=False)), sorted(rng.choice(edges, 2, replace=False))
        a = (a[0][0], a[1][0], a[0][1], a[1][1])
        b = (b[0][0] + rng.choice([0, 1e-4, -1e-4]), b[1][0], b[0][1], b[1][1])
        if b[0] >= b[2]:
            continue
        want = exact(a, b)
        for got in (iou(rect(*a), rect(*b)), iou(rect(*b), rect(*a)), _iou_ios(rect(*a), rect(*b))[0]):
            assert abs(got - want) < 1e-3, (a, b, got, want)


def test_rotated_boxes_against_a_raster():
    rng = np.random.default_rng(3)
    size = 800
    for _ in range(40):
        polys = []
        for _ in range(2):
            (cx, cy), (w, h), ang = rng.uniform(250, 550, 2), rng.uniform(80, 300, 2), rng.uniform(0, 90)
            polys.append(cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float32))
        masks = []
        for p in polys:
            m = np.zeros((size, size), np.uint8)
            cv2.fillConvexPoly(m, np.round(p).astype(np.int32), 1)
            masks.append(m.astype(bool))
        raster = float((masks[0] & masks[1]).sum())
        got = intersection_area(polys[0], polys[1])
        assert abs(got - raster) <= 0.03 * max(raster, 500) + 2 * (cv2.arcLength(polys[0], True) + cv2.arcLength(polys[1], True))
        assert abs(intersection_area(polys[1], polys[0]) - got) < 1e-6 * max(got, 1)
    a = rect(0.1, 0.1, 0.3, 0.3)
    assert overlap(a, a)[0] == 1.0 and overlap(a, rect(0.3, 0.1, 0.5, 0.3)) == (0.0, 0.0)  # touching: no overlap
