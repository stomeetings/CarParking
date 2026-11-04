"""Bird's-eye-view helpers on top of the 3x3 homography: the parking surface in metres, a vehicle's ground footprint and
the distances between footprints (Phase 2). numpy + cv2 only (no shapely: the main image stays as it is)."""
from __future__ import annotations

import cv2
import numpy as np

from .camera import ground_resolution, to_ground, to_image

CAR_LENGTH, CAR_WIDTH = 4.5, 1.8  # metres, the prior for a passenger car
MAX_RANGE = 60.0  # metres from the camera: ground farther away is not measured
MAX_RES = 0.25  # metres per pixel: where one pixel covers more than this the ground is not measured


def densify(poly, step: float = 8.0) -> np.ndarray:
    """Add points along the polygon's edges (a straight line in the image is straight on the ground, but the clip below
    works on points) -> Nx2."""
    p = np.asarray(poly, float)
    out = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t / n for t in range(n)]
    return np.array(out)


def surface_polygon(H: np.ndarray, poly_px, max_range: float = MAX_RANGE, max_res: float = MAX_RES) -> np.ndarray:
    """The image polygon (pixels) on the ground -> Nx2 metres, keeping only what can be measured: points above the
    horizon, farther than max_range from the camera or where a pixel covers more than max_res metres are dropped."""
    pts = densify(poly_px)
    g = to_ground(H, pts)
    keep = ~np.isnan(g).any(axis=1)
    keep[keep] = (np.linalg.norm(g[keep], axis=1) <= max_range) & (ground_resolution(H, pts[keep]) <= max_res)
    return g[keep]


def area(poly_m) -> float:
    """Area (m^2) of a polygon given as Nx2 metres."""
    p = np.asarray(poly_m, np.float32)
    return float(cv2.contourArea(p)) if len(p) >= 3 else 0.0


def footprint(box, H: np.ndarray, axis=(0.0, 1.0), length: float = CAR_LENGTH, width: float = CAR_WIDTH) -> np.ndarray:
    """Ground footprint of a vehicle box (x1, y1, x2, y2 pixels) -> 4x2 metres: the width across the visible contact
    line (the box's bottom edge), extended away from the camera by `length` along the bay direction `axis` (a unit
    vector on the ground; the sign is chosen to point away from the camera)."""
    x1, y1, x2, y2 = box
    a, b = to_ground(H, [(x1, y2), (x2, y2)])
    c = (a + b) / 2
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    if np.dot(u, c) < 0:
        u = -u
    n = np.array([-u[1], u[0]])
    near_l, near_r = c - n * width / 2, c + n * width / 2
    return np.array([near_l, near_r, near_r + u * length, near_l + u * length])


def centroid(poly_m) -> np.ndarray:
    return np.asarray(poly_m, float).mean(axis=0)


def closest_on_polygon(p, poly) -> tuple[float, np.ndarray]:
    """(distance, nearest point) from the point p to the BOUNDARY of a polygon (Nx2 metres); (0, p) when p is inside."""
    poly = np.asarray(poly, float)
    p = np.asarray(p, float)
    if cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(p[0]), float(p[1])), False) >= 0:
        return 0.0, p
    best, q = np.inf, p
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        c = a + ab * t
        d = float(np.linalg.norm(p - c))
        if d < best:
            best, q = d, c
    return best, q


def closest_points(a, b) -> tuple[float, np.ndarray, np.ndarray]:
    """(distance, point on a, point on b) of the two nearest points of two polygons (Nx2 metres). Overlapping or touching
    polygons: 0 and the same point twice (a vertex of one inside the other). Exact for convex shapes (a vehicle footprint
    is a rectangle): the nearest pair always involves a vertex of one of them."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    for p, q in ((a, b), (b, a)):
        for v in p:
            if cv2.pointPolygonTest(q.astype(np.float32).reshape(-1, 1, 2), (float(v[0]), float(v[1])), False) >= 0:
                return 0.0, v, v
    best = (np.inf, a[0], b[0])
    for v in a:
        d, q = closest_on_polygon(v, b)
        if d < best[0]:
            best = (d, v, q)
    for v in b:
        d, q = closest_on_polygon(v, a)
        if d < best[0]:
            best = (d, q, v)
    return float(best[0]), np.asarray(best[1], float), np.asarray(best[2], float)


def poly_distance(a, b, step: float = 0.05) -> float:
    """Smallest distance (m) between two convex polygons (Nx2 metres): 0 if they touch or overlap."""
    return closest_points(a, b)[0]


def nearest_pairs(polys: list, max_dist: float = 6.0) -> list[tuple[int, int, float]]:
    """(i, j, edge-to-edge distance m) for each pair closer than max_dist, i < j, nearest first. Pairs whose centroids
    are farther apart than max_dist + both circumradii are skipped before the exact test."""
    cs = [centroid(p) for p in polys]
    rs = [float(np.max(np.linalg.norm(np.asarray(p) - c, axis=1))) for p, c in zip(polys, cs)]
    out = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if np.linalg.norm(cs[i] - cs[j]) > max_dist + rs[i] + rs[j]:
                continue
            d = poly_distance(polys[i], polys[j])
            if d <= max_dist:
                out.append((i, j, d))
    return sorted(out, key=lambda t: t[2])


def uncertainty(H: np.ndarray, px, err_px: float = 2.0) -> float:
    """Metres a +-err_px error at the pixel is worth on the ground there (see ground_resolution)."""
    return float(ground_resolution(H, [px])[0] * err_px)


# --- the surface as a ground raster (Phase 2) -------------------------------------------------------------------------
# Projecting an image POLYGON to the ground is wrong wherever the polygon covers something that stands up: the body of a
# parked car (or a wall) is not ground, and the rays through its pixels hit the ground metres BEHIND it (on camera-1 a
# hatchback's body stretched the "surface" ten metres back to the wall). So the surface is built the other way round:
# every cell of a ground grid is projected INTO the image and kept when it lands on ground pixels (the mask, with the
# vehicles' boxes cut out); each parked car's footprint is added back (cars park on parking surface).
CELL = 0.1  # metres per raster cell
CAR_SIDE_DEPTH, CAR_END_DEPTH = 2.2, 4.6  # how far behind its near contact line a car's footprint reaches: side-on / end-on
SIDE_ON, END_ON = 3.4, 2.2  # contact-line length (m) from which a car counts as side-on / end-on (linear in between)


class Grid:
    """A boolean ground raster: cells[row = Y, col = X], the cell (0, 0) centred at (x0, y0) metres, `cell` m per cell."""

    def __init__(self, cells: np.ndarray, x0: float, y0: float, cell: float = CELL):
        self.cells, self.x0, self.y0, self.cell = cells, x0, y0, cell

    @property
    def area(self) -> float:
        return float(self.cells.sum()) * self.cell ** 2

    def to_cells(self, pts_m) -> np.ndarray:
        return (np.asarray(pts_m, float) - (self.x0, self.y0)) / self.cell

    def to_metres(self, pts_cells) -> np.ndarray:
        return np.asarray(pts_cells, float) * self.cell + (self.x0, self.y0)

    def polygon(self, min_area: float = 2.0, eps: float = 0.04) -> np.ndarray | None:
        """The largest connected piece's outer contour in metres (Nx2), simplified; None when nothing is left."""
        cs, _ = cv2.findContours(self.cells.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.contourArea(c) * self.cell ** 2 >= min_area]
        if not cs:
            return None
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps / self.cell, True).reshape(-1, 2)
        return self.to_metres(c) if len(c) >= 3 else None


OPEN = 1.2  # metres: ground strips thinner than this are not usable surface (a verge along a wall, a seam)
MIN_PIECE = 3.0  # m2: smaller pieces are dropped


def surface_grid(cam, mask, cell: float = CELL, max_range: float = MAX_RANGE, max_res: float = MAX_RES,
                 pad: float = 1.5, open_m: float = OPEN, include=()) -> Grid | None:
    """Image mask (HxW, > 0 = ground) -> the ground raster of what it covers, or None if nothing measurable. cam: a
    CameraModel (lens distortion handled). Cells farther than max_range, or where one pixel covers more than max_res
    metres, are left out (not measurable). include: polygons (metres) the raster must be able to hold, e.g. the parked
    cars' footprints that add_footprints paints later (they can lie beyond what the mask reaches)."""
    m = np.asarray(mask)
    h, w = m.shape[:2]
    vs, us = np.nonzero(m > 0)
    if len(us) == 0:
        return None
    pick = np.random.default_rng(0).choice(len(us), size=min(40_000, len(us)), replace=False)
    g = to_ground(cam, np.stack([us[pick], vs[pick]], axis=1).astype(float))
    g = g[np.isfinite(g).all(axis=1) & (np.linalg.norm(np.nan_to_num(g), axis=1) <= max_range)]
    if len(g) < 10:
        return None
    lo, hi = np.percentile(g, 0.5, axis=0) - pad, np.percentile(g, 99.5, axis=0) + pad
    for p in include:
        p = np.asarray(p, float)
        lo, hi = np.minimum(lo, p.min(axis=0) - pad), np.maximum(hi, p.max(axis=0) + pad)
    nx, ny = int((hi[0] - lo[0]) / cell) + 1, int((hi[1] - lo[1]) / cell) + 1
    if nx * ny > 4_000_000:
        return None
    gx, gy = np.meshgrid(lo[0] + np.arange(nx) * cell, lo[1] + np.arange(ny) * cell)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    px = to_image(cam, pts)
    ix, iy = np.rint(px[:, 0]), np.rint(px[:, 1])
    ok = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    hit = np.zeros(len(pts), bool)
    hit[ok] = m[iy[ok].astype(int), ix[ok].astype(int)] > 0
    near = hit & (np.linalg.norm(pts, axis=1) <= max_range)
    if near.any():
        res = ground_resolution(cam, px[near])
        keep = np.zeros(len(pts), bool)
        keep[np.nonzero(near)[0][res <= max_res]] = True
        hit = keep
    else:
        hit = near
    cells = hit.reshape(ny, nx).astype(np.uint8)
    if open_m > 0:  # thin strips are not parking surface; then small pieces go
        k = max(3, int(round(open_m / cell))) | 1
        cells = cv2.morphologyEx(cells, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cells, connectivity=8)
        small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] * cell ** 2 < MIN_PIECE]
        if small:
            cells[np.isin(labels, small)] = 0
    if not cells.any():
        return None
    return Grid(cells > 0, float(lo[0]), float(lo[1]), cell)


def explain_surface(cam, mask, max_range: float = MAX_RANGE, max_res: float = MAX_RES) -> dict:
    """Why an image mask has no measurable ground, as shares of its pixels: {"pixels", "unprojectable" (at or above the horizon), "too_far" (beyond max_range
    metres), "too_coarse" (one pixel covers more than max_res metres), "usable" (what is left)}. For the error message only; surface_grid decides."""
    m = np.asarray(mask)
    vs, us = np.nonzero(m > 0)
    if len(us) == 0:
        return {"pixels": 0, "unprojectable": 0.0, "too_far": 0.0, "too_coarse": 0.0, "usable": 0.0}
    pick = np.random.default_rng(0).choice(len(us), size=min(20_000, len(us)), replace=False)
    uv = np.stack([us[pick], vs[pick]], axis=1).astype(float)
    g = to_ground(cam, uv)
    finite = np.isfinite(g).all(axis=1)
    near = finite & (np.linalg.norm(np.nan_to_num(g), axis=1) <= max_range)
    fine = np.zeros(len(uv), bool)
    if near.any():
        fine[np.nonzero(near)[0][ground_resolution(cam, uv[near]) <= max_res]] = True
    n = float(len(uv))
    return {"pixels": int(len(us)), "unprojectable": round(float((~finite).sum()) / n, 3), "too_far": round(float((finite & ~near).sum()) / n, 3),
            "too_coarse": round(float((near & ~fine).sum()) / n, 3), "usable": round(float(fine.sum()) / n, 3)}


def car_footprint(cam, box, side_depth: float = CAR_SIDE_DEPTH, end_depth: float = CAR_END_DEPTH):
    """Ground footprint of a parked vehicle from its pixel box -> (4x2 metres, contact-line direction (unit), contact-line
    length m), or None. The box's bottom edge is the visible contact line; the car reaches behind it (away from the
    camera) by side_depth when seen side-on (a long contact line) up to end_depth when seen end-on (a short one)."""
    x1, y1, x2, y2 = box
    a, b = to_ground(cam, [(x1, y2), (x2, y2)])
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    edge = b - a
    ell = float(np.linalg.norm(edge))
    if ell < 0.5:
        return None
    t = float(np.clip((SIDE_ON - ell) / (SIDE_ON - END_ON), 0.0, 1.0))
    depth = side_depth + (end_depth - side_depth) * t
    u = edge / ell
    n = np.array([-u[1], u[0]])
    if np.dot(n, (a + b) / 2) < 0:
        n = -n  # away from the camera
    return np.array([a, b, b + n * depth, a + n * depth]), u, ell


def add_footprints(grid: Grid, polys) -> None:
    """Paint vehicle footprints (4x2 metres each) into the grid (in place), then close hairline cracks between them and
    the open ground around."""
    cells = grid.cells.astype(np.uint8)
    for p in polys:
        cv2.fillPoly(cells, [np.round(grid.to_cells(p)).astype(np.int32)], 1)
    k = max(3, int(round(0.4 / grid.cell))) | 1
    grid.cells = cv2.morphologyEx(cells, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
