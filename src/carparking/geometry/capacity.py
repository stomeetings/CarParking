"""Geometric capacity: how many bays fit in the parking surface (Phase 2). Pure geometry on the surface polygon in metres,
so it works on an empty lot on day 1. numpy + cv2 only."""
from __future__ import annotations

import cv2
import numpy as np

BAY = (2.5, 5.0)  # width, length (metres) of one bay
AISLE = 6.0  # driving aisle between two facing rows


def dominant_angle(poly_m) -> float:
    """Direction (degrees, 0-180) of the polygon's longer side, from its minimum-area rectangle. Rows of bays run along
    it, so by default the bays themselves point across it (pack_capacity adds 90)."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.asarray(poly_m, np.float32))
    return float(ang % 180 if w >= h else (ang + 90) % 180)


EDGE = 0.01  # metres: a bay may touch the surface's edge (float rounding must not reject it)


def _inside(poly32: np.ndarray, pts) -> bool:
    c = np.mean(pts, axis=0)
    pts = [c + (np.asarray(p) - c) * (1 - EDGE / np.linalg.norm(np.asarray(p) - c)) for p in pts]
    return all(cv2.pointPolygonTest(poly32, (float(x), float(y)), False) >= 0 for x, y in pts)


def pack_capacity(surface_m, angle_deg: float | None = None, bay=BAY, aisle: float = AISLE) -> list[np.ndarray]:
    """Rows of bays inside the surface -> list of 4x2 bay polygons (metres), in the surface's own frame.

    angle_deg = the direction of a bay's long side; default: across the surface's longer side (rows run along it).
    The surface is rotated so the bays' length runs along the Y axis. Bays are laid in double rows (two rows back to
    back, then an aisle) - the usual layout, and the one that gives the most bays per area - and a bay is kept only when
    all four corners lie inside the polygon. The best of the four offsets (the two row phases x the two start columns)
    is kept, since a surface rarely starts on a bay boundary."""
    s = np.asarray(surface_m, np.float64)
    if len(s) < 3:
        return []
    ang = (dominant_angle(s) + 90) % 180 if angle_deg is None else angle_deg
    # rotate by -(angle - 90) so a bay's long side (along `angle`) points along Y
    t = np.radians(90 - ang)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    r = s @ rot.T
    poly32 = r.astype(np.float32).reshape(-1, 1, 2)
    (x0, y0), (x1, y1) = r.min(axis=0), r.max(axis=0)
    bw, bl = bay
    best: list[np.ndarray] = []
    for phase in (0, 1):  # which row of the double row comes first
        for dx in (0.0, bw / 2):
            found, y, row = [], y0, phase
            while y + bl <= y1 + 1e-9:
                x = x0 + dx
                while x + bw <= x1 + 1e-9:
                    corners = [(x, y), (x + bw, y), (x + bw, y + bl), (x, y + bl)]
                    if _inside(poly32, corners):
                        found.append(np.array(corners))
                    x += bw
                y += bl + (aisle if row % 2 else 0.0)
                row += 1
            if len(found) > len(best):
                best = found
    back = rot.T  # inverse rotation
    return [b @ back.T for b in best]


def to_pixel_polys(cam, bays: list[np.ndarray]) -> list[np.ndarray]:
    """Bay polygons (metres) -> pixel polygons (Nx2) through the camera, for pre-rendered slots."""
    from .camera import to_image
    return [to_image(cam, b) for b in bays]


# --- packing on a ground raster ----------------------------------------------------------------------------------------
COVER = 0.995  # a bay needs this share of its cells to be surface: a corner nick on a slanted edge stays under ~0.35 m
LONG_CAR = 2.6  # a contact line at least this long (m) is a car seen (nearly) side-on: its long axis runs along it
NARROW_CAR = 2.0  # one at most this long is seen end-on (or is a partial box): the car runs across it. In between: no vote
SAME_AXIS = 15.0  # degrees: cars within this of each other share a parking direction


def bay_angle_from_cars(cars) -> tuple[float, float] | None:
    """The direction (degrees, 0-180) of the bays' long side from the parked cars: cars = [(contact-line direction
    (unit 2-vector), contact-line length m)]. A car whose contact line is long (side-on) lies along it, a narrow one
    (end-on) across it, one in between (an oblique view) does not vote. The answer is the biggest group of cars whose
    axes agree within SAME_AXIS degrees (an angle and its opposite are the same axis; partly hidden or far cars
    that disagree cannot outvote the rest) -> (mean angle of that group, its share of the voting cars 0-1) or None
    without a vote."""
    th = []
    for u, ell in cars:
        a = np.arctan2(u[1], u[0])
        if ell >= LONG_CAR:
            th.append(a)
        elif ell <= NARROW_CAR:
            th.append(a + np.pi / 2)
    if not th:
        return None
    th = np.array(th)
    tol = np.radians(SAME_AXIS)
    diff = np.abs(((th[:, None] - th[None, :]) + np.pi / 2) % np.pi - np.pi / 2)  # axis difference, 0..90 deg
    group = (diff <= tol).sum(axis=1).argmax()
    members = diff[group] <= tol
    c, s = np.cos(2 * th[members]).mean(), np.sin(2 * th[members]).mean()
    return float((np.degrees(np.arctan2(s, c)) / 2) % 180), float(members.sum() / len(th))


def pack_grid(grid, angle_deg: float, bay=BAY, aisle: float = AISLE, cover: float = COVER, anchors=None) -> list[np.ndarray]:
    """pack_bays without the count: the list of 4x2 bay polygons in metres."""
    return pack_bays(grid, angle_deg, bay, aisle, cover, anchors)[0]


def pack_bays(grid, angle_deg: float, bay=BAY, aisle: float = AISLE, cover: float = COVER, anchors=None):
    """Bays on a ground raster (carparking.geometry.bev.Grid) -> (list of 4x2 bay polygons in metres, n_cars): the first
    n_cars bays are the ones placed on parked cars, the rest are free bays.

    angle_deg: the direction of a bay's long side. The raster is resampled into a frame whose v axis runs along that
    direction (u across it), and an integral image makes "is this rectangle surface" O(1), so obstacles inside the surface
    (a pole) simply stop a bay.
    1. `anchors` (the centres, in metres, of cars already parked): each gets a bay centred on it; a bay that would overlap
       its neighbour's is moved to touch it, so a row of cars becomes a row of bays. No surface test: the parked car is
       the proof that the bay is usable (its footprint is only approximate, and the raster behind a car is not visible).
    2. The free bays: double rows (two rows back to back, then an aisle of `aisle` m) on the lattice of the first car's
       bay, wherever at least `cover` of the cells are surface and no car's bay is in the way. The better of the two row
       phases is kept. Without anchors the best of several start offsets is kept."""
    cells, cs = grid.cells, grid.cell
    if not cells.any():
        return [], 0
    a = np.radians(angle_deg)
    ev, eu = np.array([np.cos(a), np.sin(a)]), np.array([np.sin(a), -np.cos(a)])  # bay length / width directions
    ys, xs = np.nonzero(cells)
    pts = grid.to_metres(np.stack([xs, ys], axis=1))
    uu, vv = pts @ eu, pts @ ev
    bw, bl, ai = int(round(bay[0] / cs)), int(round(bay[1] / cs)), int(round(aisle / cs))
    # the working frame reaches one bay beyond the surface on every side: a bay centred on a car sticks out of a footprint
    u0, v0 = uu.min() - bw * cs, vv.min() - bl * cs
    nu, nv = int((uu.max() - u0) / cs) + 2 + bw, int((vv.max() - v0) / cs) + 2 + bl
    gu, gv = np.meshgrid(u0 + np.arange(nu) * cs, v0 + np.arange(nv) * cs)  # resample: rotated cell -> source cell
    world = gu[..., None] * eu + gv[..., None] * ev
    cx = np.rint((world[..., 0] - grid.x0) / cs).astype(int)
    cy = np.rint((world[..., 1] - grid.y0) / cs).astype(int)
    inside = (cx >= 0) & (cx < cells.shape[1]) & (cy >= 0) & (cy < cells.shape[0])
    rot = np.zeros((nv, nu), np.int32)
    rot[inside] = cells[cy[inside], cx[inside]]
    ii = np.pad(rot.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def share(j, i):
        return (ii[j + bl, i + bw] - ii[j, i + bw] - ii[j + bl, i] + ii[j, i]) / (bw * bl)

    def fits(j, i):
        return 0 <= i and 0 <= j and i + bw <= nu and j + bl <= nv

    taken = np.zeros((nv, nu), bool)
    placed: list[tuple[int, int]] = []  # (j, i) of the bays (v start, u start)
    marks = sorted(((p @ eu - u0) / cs, (p @ ev - v0) / cs) for p in (np.asarray(q, float) for q in (anchors if anchors is not None else ())))
    for mi, mj in marks:
        i, j = int(round(mi - bw / 2)), int(round(mj - bl / 2))
        for pj, pi in placed:  # touch a neighbour in the same row instead of overlapping it
            if abs(pj - j) < bl // 2 and abs(pi - i) < bw:
                i = pi + bw if i >= pi else pi - bw
                j = pj
                break
        if fits(j, i) and not taken[j:j + bl, i:i + bw].any() and i <= mi < i + bw + 1:
            placed.append((j, i))
            taken[j:j + bl, i:i + bw] = True
    n_cars = len(placed)

    def lattice(di, dj, phase):
        found, j, row = [], dj, phase
        while j + bl <= nv:
            found += [(j, i) for i in range(di, nu - bw + 1, bw)
                      if share(j, i) >= cover and not taken[j:j + bl, i:i + bw].any()]
            j += bl + (ai if row % 2 else 0)
            row += 1
        return found

    if placed:
        offsets = [(placed[0][1] % bw, placed[0][0] % bl)]
    else:
        offsets = [(di, dj) for dj in (0, bl // 4, bl // 2, 3 * bl // 4) for di in (0, bw // 4, bw // 2, 3 * bw // 4)]
    free: list[tuple[int, int]] = []
    for di, dj in offsets:
        for start in (dj, dj + bl):  # the frame has an empty margin of one bay: start in it or just past it
            for phase in (0, 1):
                f = lattice(di, start, phase)
                if len(f) > len(free):
                    free = f
    out = []
    for j, i in placed + free:
        c = np.array([(u0 + i * cs, v0 + j * cs), (u0 + (i + bw) * cs, v0 + j * cs),
                      (u0 + (i + bw) * cs, v0 + (j + bl) * cs), (u0 + i * cs, v0 + (j + bl) * cs)])
        out.append(c[:, :1] * eu + c[:, 1:] * ev)
    return out, n_cars
