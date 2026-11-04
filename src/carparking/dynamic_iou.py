"""Mode 4: camera-independent bays. Slots come from the painted lines in the frame itself (find_slots) or from our
trained bay model (bay_slots), never from slots.json; a slot
is BUSY when a vehicle detection covers > 25% of it (or IoU > threshold). Vehicles come from a stock COCO model
(car / bus / truck, real car boxes) or from our bay model's BUSY boxes.

Pipeline (on a copy resized to 1000 px, like view.py):
  CLAHE -> adaptive threshold (keep only pixels clearly brighter than their surroundings = paint, not soft tree
  shadows or grey asphalt texture) -> morphological closing (re-join chipped paint) -> tight Canny -> HoughLinesP
  -> keep segments within ±angle_tol of the dominant direction (the bay dividers)
  -> join collinear pieces (same offset, touching along the line) into divider lines
  -> each divider + its nearest parallel neighbour that runs alongside it = one slot (4-corner polygon)
  -> drop slots that aren't bay-like (aspect, area, mostly green = trees, optional top-of-frame cut)
  -> keep only continuous rows of >= min_row equally sized bays, each row with one common length
  -> drop overlapping duplicates.
ponytail: assumes straight, roughly parallel painted dividers. Curved rows, faded paint, strong perspective (bays
shrinking a lot along a row) or unmarked lots give missing slots; the upgrade is a learned segmentation of the
markings. The knobs are find_slots() arguments, in working-size (1000 px) pixels.
"""
from __future__ import annotations

import cv2
import numpy as np


def paint_mask(gray):
    """Bright, thin markings only: local contrast instead of a global threshold, so shadows don't matter."""
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    # pixel > local mean + 10: paint. Soft shadows / texture vary slowly or weakly and fall below it.
    m = cv2.adaptiveThreshold(cv2.GaussianBlur(g, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                              31, -10)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def segments(frame_bgr, work: int = 1000):
    """-> (N, 4) Hough segments in working-size pixels, and the scale s (working = original * s)."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    s = work / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    edges = cv2.Canny(paint_mask(g), 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10)
    return (np.zeros((0, 4)) if lines is None else np.asarray(lines, dtype=float).reshape(-1, 4)), s


def _slots_along(seg, ang, theta, angle_tol, merge_gap, min_width, min_overlap, aspect):
    """Slots bounded by divider lines running at angle theta (degrees, 0..180)
    -> ([(o1, o2, t0, t1)], t_dir, n_dir): offsets across / extents along the dividers, in that frame."""
    diff = (ang - theta + 90) % 180 - 90  # signed angle difference, wrap-safe (0° == 180°)
    near = np.abs(diff) <= angle_tol
    if not near.any():
        return [], None, None
    a = np.radians(theta + np.median(diff[near]))  # refine the 5° bin centre to the dividers' actual direction
    t_dir, n_dir = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    pieces = sorted(((x1 + x2) / 2 * n_dir[0] + (y1 + y2) / 2 * n_dir[1],  # offset across the dividers
                     *sorted((x1 * t_dir[0] + y1 * t_dir[1], x2 * t_dir[0] + y2 * t_dir[1])))  # extent along
                    for x1, y1, x2, y2 in seg[near])
    dividers: list[list[float]] = []  # [offset, t0, t1, pieces]
    for o, t0, t1 in pieces:  # same offset (both edges of one stripe) and touching along the line = one divider
        for d in dividers:
            if abs(d[0] - o) <= merge_gap and t0 <= d[2] + merge_gap and t1 >= d[1] - merge_gap:
                # running mean: the divider sits on the stripe's centre line, not on one of its edges
                d[0], d[1], d[2], d[3] = (d[0] * d[3] + o) / (d[3] + 1), min(d[1], t0), max(d[2], t1), d[3] + 1
                break
        else:
            dividers.append([o, t0, t1, 1])
    dividers = sorted(d[:3] for d in dividers)
    cands = []
    for i, (o1, a1, b1) in enumerate(dividers):
        for o2, a2, b2 in dividers[i + 1:]:
            lo, hi = max(a1, a2), min(b1, b2)
            if o2 - o1 >= min_width and hi - lo >= min_overlap * min(b1 - a1, b2 - a2):
                if aspect[0] <= (hi - lo) / (o2 - o1) <= aspect[1]:  # bay-shaped: longer than wide, not a sliver
                    cands.append((o1, o2, lo, hi))
                break  # only the nearest neighbour that runs alongside this divider
    return cands, t_dir, n_dir


def _quad(c, t_dir, n_dir):
    o1, o2, t0, t1 = c
    return np.array([t * t_dir + o * n_dir for o, t in ((o1, t0), (o1, t1), (o2, t1), (o2, t0))], np.float32)


def _rows(cands, min_row, width_step, gap):
    """Slots -> continuous rows only. Same row = similar extent along the dividers; a run continues while the next
    slot starts where the last one ended (shared divider, gap < gap x width) and its width changes by < width_step
    (gradual perspective is fine, a jump is not). Runs shorter than min_row (isolated floating boxes) are dropped;
    each kept run gets one common length (median start/end), so a row is a uniform strip of bays."""
    if not cands:
        return []
    half = np.median([hi - lo for *_, lo, hi in cands]) / 2
    rows: list[list] = []
    for c in sorted(cands, key=lambda c: c[2] + c[3]):
        row = next((r for r in rows if abs((r[0][2] + r[0][3]) - (c[2] + c[3])) / 2 < half), None)
        if row:
            row.append(c)
        else:
            rows.append([c])
    out = []
    for row in rows:
        runs = [[]]
        for c in sorted(row):
            p = runs[-1][-1] if runs[-1] else None
            if p and not (c[0] - p[1] <= gap * (p[1] - p[0]) and
                          1 / width_step <= (c[1] - c[0]) / (p[1] - p[0]) <= width_step):
                runs.append([])
            runs[-1].append(c)
        for run in runs:
            if len(run) >= min_row:
                lo, hi = float(np.median([c[2] for c in run])), float(np.median([c[3] for c in run]))
                out += [(o1, o2, lo, hi) for o1, o2, *_ in run]
    return out


def vegetation(bgr):
    """Trees / grass: saturated green. Asphalt, paint and most cars are grey or another hue."""
    return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), (35, 60, 30), (85, 255, 255))


def find_slots(frame_bgr, angle_tol: float = 15, merge_gap: float = 6, min_width: float = 15,
               min_overlap: float = 0.5, aspect=(1.3, 4.0), min_area: float = 600, max_green: float = 0.3,
               ignore_top: float = 0.0, min_row: int = 3, width_step: float = 1.35, row_gap: float = 0.5,
               max_overlap: float = 0.3) -> list[np.ndarray]:
    """-> slot polygons (4x2 float32, original pixels), long side along the dividers as in PKLot. Distances/areas are
    in working-size (1000 px) pixels.
    aspect: allowed length/width. min_area: smallest slot. max_green: drop a slot if more of it than this is
    vegetation. ignore_top: drop slots whose centre is in the top fraction of the frame (sky/trees/buildings).
    min_row / width_step / row_gap: see _rows. max_overlap: IoU above which a slot is a duplicate."""
    seg, s = segments(frame_bgr)
    if len(seg) < 2:
        return []
    h = frame_bgr.shape[0] * s
    small = cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    green = vegetation(small)

    def bay_like(c, t_dir, n_dir):
        if (c[1] - c[0]) * (c[3] - c[2]) < min_area:
            return False
        q = _quad(c, t_dir, n_dir)
        if q[:, 1].mean() < ignore_top * h:
            return False
        m = cv2.fillConvexPoly(np.zeros(green.shape, np.uint8), np.round(q).astype(np.int32), 255)
        return cv2.mean(green, m)[0] / 255 <= max_green

    ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180
    bins = np.bincount((ang // 5).astype(int) % 36, minlength=36)  # 5° bins, by segment count (dividers are many)
    top = int(bins.argmax())
    # the most common direction may be the row edge rather than the dividers: also try the next distinct one (≥30°)
    other = [int(b) for b in np.argsort(-bins) if bins[b] and min(abs(b - top), 36 - abs(b - top)) >= 6][:1]
    best: list[np.ndarray] = []
    for b in [top, *other]:
        cands, t_dir, n_dir = _slots_along(seg, ang, (b + 0.5) * 5, angle_tol, merge_gap, min_width, min_overlap,
                                           aspect)
        rows = _rows([c for c in cands if bay_like(c, t_dir, n_dir)], min_row, width_step, row_gap)
        kept: list[np.ndarray] = []
        for c in sorted(rows, key=lambda c: -(c[1] - c[0]) * (c[3] - c[2])):  # big first
            q = _quad(c, t_dir, n_dir)
            if all(iou(q, k) <= max_overlap for k in kept):  # a divider spanning two rows can pair twice
                kept.append(q)
        best = max(best, kept, key=len)
    return [p / s for p in best]


def bay_slots(frame_bgr, bay_model, sliced: bool = False) -> list[np.ndarray]:
    """Slot source B: every bay our trained model finds (FREE and BUSY alike) becomes a slot polygon, skipping
    Hough entirely. Its own FREE/BUSY verdict is ignored; occupancy() decides from the vehicle boxes.
    bay_model: a YoloDetector on models/parking_yolo*.pt (axis-aligned or OBB)."""
    return [np.array(d["poly"], np.float32) for d in bay_model(frame_bgr, sliced=sliced)]


def _rows_at(cars: list[tuple], angle: float, max_dy, max_gap, min_dx, max_angle) -> list[list[tuple]]:
    """Rows running within ±max_angle of `angle` (degrees; 0 = left-right, 90 = top-bottom). The frame is rotated so
    that direction becomes horizontal; each car's along/across size = its box projected onto the row direction /
    its normal. Then, walking along the row direction, a car may join a row whose last car is ahead of it by
    min_dx..max_gap car lengths (side by side, up to ~3 empty bays; a car level with it across is another lane):
      * row of >= 2 cars: its direction is known (first -> last car); the car must lie within max_dy x car depth of
        that line, measured perpendicular to it. This is the lane-width limit: the next parallel row is about a car
        depth away and fails it, even on a slanted row.
      * single car: the step to it must be within max_angle of the pass direction.
    Among the rows that pass, the car joins the one it fits best (smallest distance to the line)."""
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    local = sorted(((x * c + y * s, -x * s + y * c, abs(c) * w + abs(s) * h, abs(s) * w + abs(c) * h, car)
                    for car in cars for x, y, w, h in [car]), key=lambda t: t[0])
    rows: list[list[tuple]] = []
    for car in local:
        best, best_res = None, np.inf
        for r in rows:
            lx, ly, lw, lh, _ = r[-1]
            dx, dy = car[0] - lx, car[1] - ly
            if not min_dx * min(car[2], lw) < dx < max_gap * max(car[2], lw):
                continue
            if len(r) >= 2:
                slope = (ly - r[0][1]) / (lx - r[0][0])  # local x strictly increases along a row (dx > 0 above)
                res = abs(dy - slope * dx) / np.hypot(1, slope)  # perpendicular distance to the row's line
                if res >= max_dy * min(car[3], lh):
                    continue
            elif abs(np.degrees(np.arctan2(dy, dx))) > max_angle:
                continue
            else:
                res = abs(dy)
            if res < best_res:
                best, best_res = r, res
        if best is not None:
            best.append(car)
        else:
            rows.append([car])
    return [[t[4] for t in r] for r in rows]


def vehicle_rows(dets: list[dict], max_dy: float = 0.35, max_gap: float = 4.0, min_dx: float = 0.5,
                 max_angle: float = 35, angles=(0, 90, 45, 135)) -> list[list[tuple]]:
    """Vehicle boxes (state BUSY) -> rows of (cx, cy, w, h), each sorted along its own direction; single cars last.
    Any orientation: one clustering pass per direction in `angles` (each covers ±max_angle, so 0/45/90/135 cover all
    of them: left-right rows, top-bottom columns along a lot edge, both diagonals). The longest row found in any pass
    is accepted and its cars removed, then all passes run again on the rest. Longest-first stops a short wrong row
    (e.g. two cars side by side across two columns) from stealing cars from the real, longer one.
    ponytail: O(rows x angles x cars²) in Python, fine for ~100 cars per frame; accept several rows per round if slow."""
    cars = [((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
            for x1, y1, x2, y2 in (d["box"] for d in dets if d["state"] == 1)]
    rows: list[list[tuple]] = []
    while cars:
        best = max((r for a in angles for r in _rows_at(cars, a, max_dy, max_gap, min_dx, max_angle)),
                   key=len, default=[])
        if len(best) < 2:
            break
        rows.append(best)
        cars = [car for car in cars if not any(car is b for b in best)]  # identity: two identical boxes stay two cars
    return rows + [[car] for car in cars]


def _box(cx, cy, w, h):
    return np.array([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
                     (cx - w / 2, cy + h / 2)], np.float32)


def _rect(centre, u, v, length, depth):
    """Rotated rectangle: length along the row direction u, depth along its normal v."""
    du, dv = u * length / 2, v * depth / 2
    return np.array([centre - du - dv, centre + du - dv, centre + du + dv, centre - du + dv], np.float32)


def _main_lot(rows: list[list[tuple]], reach: float) -> set[int]:
    """Indices of the biggest group (by car count) of rows that touch: some car of one within `reach` car sizes of
    some car of the other. Street parking across a road is a separate, smaller group."""
    near = lambda a, b: any(max(abs(p[0] - q[0]) / max(q[2], 1), abs(p[1] - q[1]) / max(q[3], 1)) <= reach  # noqa
                            for p in a for q in b)
    seen, best = set(), set()
    for s in range(len(rows)):
        if s in seen:
            continue
        comp, todo = {s}, [s]
        while todo:  # flood fill over touching rows
            i = todo.pop()
            for j in range(len(rows)):
                if j not in comp and (near(rows[i], rows[j]) or near(rows[j], rows[i])):
                    comp.add(j)
                    todo.append(j)
        seen |= comp
        if sum(len(rows[i]) for i in comp) > sum(len(rows[i]) for i in best):
            best = comp
    return best


def anchor_slots(dets: list[dict], frame_shape, extend: int = 2, margin: float = 0.9, min_row: int = 3,
                 max_fill: int = 3, reach: float = 2.5, frame=None, max_green: float = 0.3, roi=None,
                 main_lot: bool = True, ignore_top: float = 0.0, mask=None, stay_in_hull: bool = True,
                 hull_margin: float = 1.0, car_green: float = 0.6, keep_lone: bool = True, max_overlap: float = 0.3,
                 **row_knobs) -> list[np.ndarray]:
    """Slot source C, auto-anchor grid: parked vehicles are the anchors. Per trusted row, in the row's own rotated
    frame (u = row direction from a line fit through the car centres, v = its normal):
      * each car's footprint along u / v = its axis-aligned box projected onto u / v
      * pitch = median step along u between direct neighbours (steps < 1.5x the median footprint)
      * one slot per car at its centre; each gap holds round(gap / pitch) - 1 virtual bays, spaced evenly
      * extend virtual bays continue past each end of the row
    All slots are rectangles rotated to the row's angle, margin x pitch wide (a neighbour's box doesn't spill into
    an empty bay), car-footprint deep. Rows can run in any direction (see vehicle_rows).

    Parking area = the neighbourhood of trusted rows, inside the ROI. Every slot, real car or virtual bay, must pass
    in_lot():
      * ignore_top (fraction of the height): a hard horizontal cut-off, applied like the ROI. Vehicles whose centre
        is in the top band are removed before clustering, and no slot centre there is kept (background roads).
      * roi (polygon, original pixels) and mask (uint8 H x W, 0 = not parking, any shape): absolute walls, like
        ignore_top. Vehicles outside them are removed before clustering, so they can't form rows or anchor bays; no
        slot centre outside them is ever kept.
      * stay_in_hull: the lot's outline is the convex hull of all trusted-row car centres, padded by hull_margin car
        sizes. Bays extrapolated past a row's end and lone cars must lie inside it, so a column at the lot's edge
        can't run on into the driveway; extension only fills out toward where neighbouring rows already reach.
        A lot with a single row therefore gets no end extensions (nothing says the lot continues).
      * main_lot (the default boundary when there is no ROI): only the biggest group of touching trusted rows
        (_main_lot) anchors the grid; separate rows, e.g. street parking across a road, become lone cars.
      * trusted row: >= min_row cars AND at least two of them side by side (a step < 1.5x footprint). A chain of
        cars that are all far apart is usually a false row across a driveway or a street.
      * inside the frame
      * with `frame`: not more than max_green of an empty bay is foliage (vegetation()); a detected car's slot may be
        up to car_green foliage (a car under a tree canopy has leaves and green shadow over it)
      * reach: the slot's centre must be within `reach` car sizes of a car in a trusted row (max(|dx|/w, |dy|/h)).
        Applies to virtual bays, so they can't wander into open road. Detected cars skip it: trusted-row cars by
        definition, lone cars (single, untrusted chain, outside the main lot) by default (keep_lone=True: a car
        under a tree that COCO saw is real); with keep_lone=False a lone car must also be near a row and inside the
        lot outline, the strict mode for cameras without a mask.
      * a gap of more than max_fill empty bays is not filled (an aisle or road crossing the row, not bays)
      * virtual bays overlapping a kept slot (IoU > max_overlap, e.g. two crossing rows) are dropped
    ponytail: with keep_lone=True every detected car inside the walls gets a slot, moving cars too: blocking the
    street is the mask's job (tracking, blueprint stage 3, would tell parked from passing). Without a trusted row
    there are no virtual bays; how far a row continues is a guess (extend). main_lot drops a real
    lot section separated by more than `reach` car sizes (wide road inside the lot): set an ROI and main_lot=False
    for such lots; street parking that touches the lot is only separated by an ROI. The real parking surface comes
    from blueprint stage 3 (footprints of stationary cars over time) or a learned surface mask."""
    hgt, wid = frame_shape[:2]
    green = vegetation(frame) if frame is not None else None
    roi = None if roi is None else np.asarray(roi, np.float32).reshape(-1, 1, 2)
    top = ignore_top * hgt  # hard cut-off: background road / buildings above this y
    anchors, hull, pad = [], None, 0.0  # trusted-row cars (cx, cy, w, h) and their outline: the parking area

    def allowed(x, y):
        """The hard walls: frame, top cut-off, ROI polygon, mask. Nothing outside them, ever."""
        if not (0 <= x < wid and top <= y < hgt):
            return False
        if roi is not None and cv2.pointPolygonTest(roi, (float(x), float(y)), False) < 0:
            return False
        return mask is None or mask[int(y), int(x)] > 0

    def in_lot(q, check_reach=True, check_hull=True, green_limit=max_green):
        x, y = q.mean(0)
        if not allowed(x, y):
            return False
        if check_hull and stay_in_hull and hull is not None and cv2.pointPolygonTest(hull, (float(x), float(y)), True) < -pad:
            return False  # past the outermost parked cars of the lot: open driveway, not bays
        if check_reach:
            a = np.array(anchors, float).reshape(-1, 4)
            if not len(a) or np.max(np.abs(a[:, :2] - (x, y)) / np.maximum(a[:, 2:], 1), axis=1).min() > reach:
                return False
        if green is None:
            return True
        m = cv2.fillConvexPoly(np.zeros(green.shape, np.uint8), np.round(q).astype(np.int32), 255)
        return cv2.mean(green, m)[0] / 255 <= green_limit

    # hard walls before clustering: a car outside them never forms or anchors a row
    dets = [d for d in dets if d["state"] != 1 or allowed((d["box"][0] + d["box"][2]) / 2,
                                                          (d["box"][1] + d["box"][3]) / 2)]

    lone, trusted = [], []  # trusted: (cars, their slots, virtual bays) per trusted row
    for row in vehicle_rows(dets, **row_knobs):
        c = np.array(row)
        pts = c[:, :2]
        if len(row) >= 2:
            u = np.linalg.svd(pts - pts.mean(0))[2][0]  # principal direction of the centres = row direction
            u = u if (pts[-1] - pts[0]) @ u > 0 else -u  # point along the row's order (also for top-bottom columns)
            v = np.array([-u[1], u[0]])
            along = np.abs(u[0]) * c[:, 2] + np.abs(u[1]) * c[:, 3]  # footprint of each box along the row
            depth = np.abs(v[0]) * c[:, 2] + np.abs(v[1]) * c[:, 3]  # ... and across it
            ds = np.diff(pts @ u)
            adjacent = ds < 1.5 * float(np.median(along))
        if len(row) < max(2, min_row) or not adjacent.any():  # not a trusted row: lone cars, checked below
            lone += [_box(*car) for car in row]
            continue
        pitch = float(np.median(ds[adjacent]))
        n = np.maximum(1, np.round(ds / pitch)).astype(int)  # bays per gap
        virtual = []
        for i, k_max in enumerate(n):
            if k_max - 1 > max_fill:
                continue
            for k in range(1, k_max):  # interpolate the empty bays between two parked cars
                t = k / k_max
                virtual.append(_rect(pts[i] + (pts[i + 1] - pts[i]) * t, u, v, margin * ds[i] / k_max,
                                     depth[i] + (depth[i + 1] - depth[i]) * t))
        step = float(np.median(ds / n))  # one bay further along the row
        for p, d, sign in ((pts[0], depth[0], -1), (pts[-1], depth[-1], 1)):
            virtual += [_rect(p + sign * k * step * u, u, v, margin * step, d) for k in range(1, extend + 1)]
        trusted.append((row, [_rect(p, u, v, margin * pitch, d) for p, d in zip(pts, depth)], virtual))

    if main_lot and len(trusted) > 1:  # default boundary: rows not connected to the biggest group are outside
        main = _main_lot([t[0] for t in trusted], reach)
        lone += [_box(*car) for i, t in enumerate(trusted) if i not in main for car in t[0]]
        trusted = [t for i, t in enumerate(trusted) if i in main]
    anchors[:] = [car for t in trusted for car in t[0]]
    if anchors:  # the lot's outline: hull of the trusted-row car centres, padded by hull_margin car sizes
        a = np.array(anchors, np.float32)
        hull = cv2.convexHull(np.ascontiguousarray(a[:, :2]).reshape(-1, 1, 2))  # OpenCV 5 rejects a column view
        pad = hull_margin * float(np.median((a[:, 2] + a[:, 3]) / 2))
    real, virtual = [q for t in trusted for q in t[1]], [q for t in trusted for q in t[2]]
    # detected cars: leaves and canopy shadow over a car are normal, so foliage only rules out a car slot that is
    # mostly green (car_green), unlike an empty bay (max_green)
    kept = [q for q in real if in_lot(q, check_reach=False, check_hull=False, green_limit=car_green)]  # the lot itself
    # lone cars (single, untrusted chain, or a row outside the main lot): a detected car is real, so by default it
    # only has to pass the hard walls and foliage; street cars are the mask/ROI's job. keep_lone=False restores the
    # strict rule (near a trusted row and inside the lot outline) for cameras without a mask.
    kept += [q for q in lone if in_lot(q, check_reach=not keep_lone, check_hull=not keep_lone, green_limit=car_green)]
    for q in virtual:
        if in_lot(q) and all(iou(q, k) <= max_overlap for k in kept):
            kept.append(q)
    return kept


def _convex(p):
    """Any 4-corner box (axis-aligned or OBB xyxyxyxy, any corner order) -> convex hull in consistent order, as
    cv2.intersectConvexConvex needs; a self-crossing corner order would give a wrong intersection."""
    return cv2.convexHull(np.asarray(p, np.float32).reshape(-1, 2))


def _clip_area(a, b) -> float:
    """Area of the intersection of two convex polygons (N x 2, float64): Sutherland-Hodgman, a clipped by each edge
    of b. Exact for shared or nearly collinear edges, where cv2.intersectConvexConvex can return nonsense, depending
    on the argument order (Front Parking 02, 2026-10-22: a car box ending at x 0.5680 against slot 6 ending at 0.5676
    gave 11.8x the box's own area, so the matcher gave that small car the pickup's slot and the pickup a second one)."""
    def ccw(p):
        return p if _signed(p) >= 0 else p[::-1]

    out, clip = ccw(a), ccw(b)
    for i in range(len(clip)):
        if len(out) == 0:
            return 0.0
        p, q = clip[i], clip[(i + 1) % len(clip)]
        e = q - p
        side = e[0] * (out[:, 1] - p[1]) - e[1] * (out[:, 0] - p[0])  # >= 0: inside (left of the edge)
        pts = []
        for j in range(len(out)):
            c, n, sc, sn = out[j], out[(j + 1) % len(out)], side[j], side[(j + 1) % len(out)]
            if sc >= 0:
                pts.append(c)
            if (sc >= 0) != (sn >= 0):
                pts.append(c + (n - c) * (sc / (sc - sn)))
        out = np.array(pts, np.float64).reshape(-1, 2)
    return abs(_signed(out)) if len(out) >= 3 else 0.0


def _signed(p) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def intersection_area(a, b) -> float:
    """Exact intersection area of two convex polygons (any corner order; never more than the smaller one)."""
    a, b = _convex(a), _convex(b)
    return min(_clip_area(a.reshape(-1, 2).astype(np.float64), b.reshape(-1, 2).astype(np.float64)),
               cv2.contourArea(a), cv2.contourArea(b))


def overlap(slot, car) -> tuple[float, float]:
    """-> (IoU, coverage = intersection / slot area). Bounding-box reject first, then the exact polygon intersection
    (_clip_area: symmetric, never more than the smaller polygon)."""
    a, b = _convex(slot), _convex(car)
    if (a.max((0, 1)) < b.min((0, 1))).any() or (b.max((0, 1)) < a.min((0, 1))).any():
        return 0.0, 0.0
    area_a, area_b = cv2.contourArea(a), cv2.contourArea(b)
    inter = min(_clip_area(a.reshape(-1, 2).astype(np.float64), b.reshape(-1, 2).astype(np.float64)), area_a, area_b)
    union = area_a + area_b - inter
    return (float(inter / union) if union > 0 else 0.0), (float(inter / area_a) if area_a > 0 else 0.0)


def iou(a, b) -> float:
    return overlap(a, b)[0]


def occupancy(slots, dets: list[dict], threshold: float = 0.35, cover: float = 0.25,
              min_area: float = 100) -> list[dict]:
    """Slot polygons + YoloDetector output -> draw_dets-compatible slot dicts. Vehicles = detections with state BUSY
    (COCO car/bus/truck/motorcycle, or busy_slot). A slot is BUSY (1) when a vehicle covers > cover of the slot's
    area, or their IoU > threshold; else FREE (0). Coverage is the main rule: from a CCTV angle a car's box is
    taller than its bay (roof, side), so IoU stays low even when the bay is clearly taken.
    Slots smaller than min_area (original pixels²) are dropped."""
    cars = [d["poly"] for d in dets if d["state"] == 1]
    out = []
    for poly in slots:
        if cv2.contourArea(_convex(poly)) < min_area:
            continue
        pairs = [overlap(poly, c) for c in cars]
        best_iou, best_cov = max((p[0] for p in pairs), default=0.0), max((p[1] for p in pairs), default=0.0)
        out.append({"poly": [tuple(map(int, p)) for p in np.round(poly)],
                    "state": int(best_cov > cover or best_iou > threshold),
                    "iou": round(best_iou, 3), "cover": round(best_cov, 3), "label": "slot", "conf": best_cov})
    return out
