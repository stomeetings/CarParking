"""Ways to make a ZoneLayout (carparking/layout.py) from a few user marks, and the inward margin.

  * layout_from_dots   reference dots placed anywhere on the row's separator lines (start, middle or end)
  * clone_master       one perfect master bay, cloned along the row on the ground
  * inset_slots        every bay pulled in by a margin in METRES on the ground (the cross-talk guard), projected afterwards
Pure numpy; nothing is stored."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carparking.geometry.camera import to_ground, to_image
from carparking.layout import CORNERS, LayoutError, ZoneLayout


@dataclass
class Dot:
    px: tuple[float, float]  # the dot on a separator line, fractions of the picture
    index: int | None = None  # the line it is on: 0 = the row's first line (left of bay 0), k = between bay k-1 and k; None = worked out from the spacing
    at: str | None = None  # where on its line this dot sits: front / mid / back; None = the call's dot_at (dots on different slots can be at different places)


def _zone_extent(cam, outline_px, w: int, h: int) -> np.ndarray | None:
    """The zone polygon (fractions) as ground points, or None when it is missing or not on measurable ground."""
    if outline_px is None:
        return None
    og = to_ground(cam, np.asarray(outline_px, float).reshape(-1, 2) * (w, h))
    return og if len(og) >= 3 and np.isfinite(og).all() else None


def _long_edge(og: np.ndarray) -> np.ndarray:
    """Unit direction of the polygon's longest edge on the ground (the row runs along the zone's long side)."""
    edges = [og[(i + 1) % len(og)] - og[i] for i in range(len(og))]
    e = max(edges, key=lambda v: float(np.linalg.norm(v)))
    return e / (np.linalg.norm(e) or 1.0)


def layout_from_dots(cam, dots: list[Dot], count: int, w: int, h: int, width: float | None = None, depth: float = 5.0, gap: float = 0.0, rows: int = 1,
                     row_gap: float = 0.0, dot_at: str = "front", heading: float | None = None, flip: bool = False, outline_px=None,
                     fit_zone: bool = False) -> tuple[ZoneLayout, dict]:
    """A layout from reference dots. The dots are a GUIDE (spacing, scale, alignment), never the row's limits: the zone polygon (`outline_px`, fractions) is the
    row's extent and the N bays are laid across all of it.
      * direction : the line through the dots when they span a few bays; else the zone's long side (dots in one slot cannot give a direction)
      * step      : from the dots' spacing; each dot's line number is typed or inferred from the spacing against the zone's nominal bay (zone length / N) or `width`.
                    Dots closer than one bay (all in one slot, or all on one line) give no step: the nominal bay of the zone is used
      * start     : the row STARTS AT THE DOTS: untyped, Dot 1 (the first dot along the row) is line 0, the first bay's start edge, and nothing is drawn before it;
                    typed line numbers put line 0 where they say. The zone polygon only gives the far end
      * count     : N bays of the step from the start. `fit_zone` (and dots closer than a bay) make the step (zone's far end - start) / N, so N bays span from Dot 1 to the end
    Without an outline the old behaviour stands (the extent is the dots'). dot_at: where on its separator the dot is: front (v = 0), mid (depth / 2), back (depth).
    -> (layout, {"step", "residual_m", "dropped", "indices", "extent_m", "covers_m", "guide_only", "warnings"})"""
    if not dots or count < 1:
        raise LayoutError("place at least one reference dot and give the expected count")
    if dot_at not in ("front", "mid", "back"):
        raise LayoutError("dot_at is front, mid or back")
    for dd in dots:
        if dd.at not in (None, "front", "mid", "back"):
            raise LayoutError("a dot's place is front, mid or back")
    g = to_ground(cam, np.array([dd.px for dd in dots], float) * (w, h))
    if not np.isfinite(g).all():
        raise LayoutError("a dot is not on measurable ground (above the horizon?)")
    ext = _zone_extent(cam, outline_px, w, h)
    d_out = _long_edge(ext) if ext is not None else None
    nominal = float(np.ptp(ext @ d_out)) / int(count) if ext is not None else None
    warnings: list[str] = []
    d_dots = None
    span = 0.0
    if len(g) >= 2:
        vs = np.array([{"front": 0.0, "mid": depth / 2, "back": depth}[dd.at or dot_at] for dd in dots])
        gc = g
        d_dots = np.linalg.svd(g - g.mean(axis=0))[2][0]
        for _ in range(3 if np.ptp(vs) > 1e-9 else 0):  # dots at different places on their lines: bring each to the front line before reading the row's direction
            nn = np.array([-d_dots[1], d_dots[0]]) * (-1.0 if flip else 1.0)
            gc = g - np.outer(vs, nn)
            d_dots = np.linalg.svd(gc - gc.mean(axis=0))[2][0]
        if d_dots @ (gc[-1] - gc[0]) < 0:
            d_dots = -d_dots
        span = float(np.ptp(g @ d_dots))
    if d_dots is not None and (ext is None or (span >= 3.0 and span >= 1.5 * (nominal or 0.0))):
        d = d_dots  # the dots span a few bays: their line is the row
    elif d_out is not None:
        d = d_out if d_dots is None or d_out @ d_dots >= 0 else -d_out  # the zone's long side: dots in one slot cannot give a direction
        if d_dots is None and ext is not None:  # one dot: the row runs from it toward the far end of the zone (the side with more room), the depth side kept
            ue = ext @ d
            if float(ue.max()) - float(g[0] @ d) < float(g[0] @ d) - float(ue.min()):
                d, flip = -d, not flip
        if d_dots is not None and abs(float(d_dots @ d)) < np.cos(np.radians(5.0)):  # (only worth saying when the dots' own line disagrees)
            warnings.append("the dots are close together: the zone's long side gives the row's direction")
    elif heading is not None and width:
        d = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))])
    else:
        raise LayoutError("one dot needs the bay width and the row's heading (or an outline to take the direction from)")
    heading_deg = float(np.degrees(np.arctan2(d[1], d[0])))
    u = g @ d
    typed = any(dd.index is not None for dd in dots)  # one typed line number anchors the lattice; the untyped dots are numbered from it
    ref = width or nominal
    for dd in dots:
        if dd.at not in (None, "front", "mid", "back"):
            raise LayoutError("a dot's place is front, mid or back")
    dropped: list = []
    keep = list(range(len(g)))
    guide_only = False

    def solve(ix):
        nonlocal guide_only
        us = u[ix]
        guide_only = False
        tix = [j for j, i in enumerate(ix) if dots[i].index is not None]
        if tix:
            ks = np.zeros(len(ix))
            tk = np.array([dots[ix[j]].index for j in tix], float)
            tu = us[tix]
            if np.ptp(tk) >= 1:  # two or more typed lines give the step (least squares); the others are numbered by it
                a0, st0 = np.linalg.lstsq(np.c_[np.ones(len(tk)), tk], tu, rcond=None)[0]
            else:  # one typed line: the step is the bay size we expect
                st0 = width or nominal or (float(np.ptp(us)) / max(1, len(ix) - 1) if len(ix) > 1 else 0.0)
                a0 = float(tu.mean() - tk.mean() * st0) if st0 else 0.0
            for j in range(len(ix)):
                ks[j] = dots[ix[j]].index if j in tix else (round((us[j] - a0) / st0) if st0 > 1e-6 else 0)
            ks = np.maximum(ks, 0)
        elif ref is not None and len(ix) >= 2:  # lines from the spacing against the bay size we expect, never from the dots' own minimum gap
            ks = np.round((us - us.min()) / ref)
        elif len(ix) >= 2:  # no outline, no width: the smallest gap is one bay
            gaps = np.diff(np.sort(us))
            r0 = max(float(gaps[gaps > 1e-6].min()) if (gaps > 1e-6).any() else 2.5, 0.5)
            ks = np.round((us - us.min()) / r0)
        else:
            ks = np.zeros(1)
        if len(ix) == 1 or np.ptp(ks) < 1:  # one line, or every dot closer than one bay: no step to read
            guide_only = True
            st = width or nominal
            if not st:
                raise LayoutError("the dots give no spacing: type the bay width")
            return float(us.min() - ks.min() * st), float(st), [int(k) for k in ks]  # (the first dot along the row is where the row starts)
        u0_, step_ = np.linalg.lstsq(np.c_[np.ones(len(ks)), ks], us, rcond=None)[0]
        return float(u0_), float(step_), [int(k) for k in ks]

    u0, step, ks = solve(keep)
    while len(keep) >= 4 and not guide_only:
        res = np.array([abs(u[i] - (u0 + k * step)) for i, k in zip(keep, ks)])
        worst = int(np.argmax(res))
        if res[worst] <= 0.3 * abs(step):
            break
        dropped.append(list(dots[keep[worst]].px))
        keep.pop(worst)
        u0, step, ks = solve(keep)
    # the row STARTS at the dots: untyped, Dot 1 (the first dot along the row) is line 0 and the first bay's start edge; typed, line 0 is where the typed numbers say
    u_first = u0 if (ext is None or typed) else u0 + step * float(min(ks))
    u_end = float((ext @ d).max()) if ext is not None else None
    avail = None if u_end is None else u_end - u_first  # the zone's length from the start of the row to its far end
    if avail is not None and avail <= 0.3 * int(count):
        warnings.append("the first dot is at or past the zone's far end: no room for the bays after it; move Dot 1 back or the zone's outline out")
        avail = None
    if avail is not None and (fit_zone or (guide_only and not width)):  # span the expected count from the anchor to the far end
        step = avail / int(count)
        if guide_only and not fit_zone:
            warnings.append("the dots are closer than one bay: they set the start and direction, the bay size is the distance from Dot 1 to the zone's far end over the expected count")
    if step <= 0.3:
        raise LayoutError("the dots give bays narrower than 0.3 m: check their order or line numbers")
    resid = 0.0 if guide_only else float(np.sqrt(np.mean([(u[i] - (u0 + k * step)) ** 2 for i, k in zip(keep, ks)])))
    n = np.array([-d[1], d[0]]) * (-1.0 if flip else 1.0)
    v_of = {"front": 0.0, "mid": depth / 2, "back": depth}
    base_v = float(np.mean([g[i] @ n - v_of[dots[i].at or dot_at] for i in keep]))  # the front line passes this far across (each dot knows where on its line it is)
    extent, covers, pct = None, int(count) * step, None
    if ext is not None:
        extent = float(np.ptp(ext @ d))
        pct = 100.0 * covers / avail if avail else None  # coverage of the stretch from the row's start to the zone's far end
        if pct is not None and not 85.0 <= pct <= 115.0:
            what = "leave the rest of the zone empty" if pct < 100 else "run past the end of the zone"
            warnings.append(f"{int(count)} bays of {step:.2f} m cover {covers:.1f} m = {pct:.0f} % of the {avail:.1f} m from the first dot to the zone's end: they {what}. "
                            f"Check the expected count or the dots' line numbers, or tick 'Stretch the bays to fit the zone exactly'")
    o = d * u_first + n * base_v
    lay = ZoneLayout(origin=(float(o[0]), float(o[1])), heading=heading_deg, flip=flip, width=step - gap, depth=depth, gap=gap, count=int(count), rows=int(rows), row_gap=row_gap)
    return lay, {"step": round(step, 4), "residual_m": round(resid, 4), "dropped": dropped, "indices": ks, "extent_m": None if extent is None else round(extent, 3), "available_m": None if avail is None else round(avail, 3),
                 "covers_m": round(covers, 3), "coverage_pct": None if pct is None else round(pct, 1), "guide_only": guide_only, "warnings": warnings}


def clone_master(cam, master_px, count: int, w: int, h: int, master_index: int = 0, rows: int = 1, gap: float = 0.0, row_gap: float = 0.0) -> tuple[ZoneLayout, dict]:
    """One perfect master bay (4 pixel corners, fractions, front-left first) -> the whole grid: its ground rectangle (width, depth, direction) is measured through
    the camera and replicated along the row by width + gap; far clones shrink with the perspective because they are equal on the ground. master_index: where the
    master stands in the row (0 = first; the clones then extend both ways). Reports how far the drawn master is from a rectangle on the ground."""
    g = to_ground(cam, np.asarray(master_px, float).reshape(-1, 2) * (w, h))
    if len(g) != CORNERS or not np.isfinite(g).all():
        raise LayoutError("the master bay is not on measurable ground")
    d = (g[1] - g[0] + g[2] - g[3]) / 2
    if np.linalg.norm(d) < 0.3:
        raise LayoutError("the master bay is degenerate")
    d = d / np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    flip = float(((g[3] - g[0] + g[2] - g[1]) / 2) @ nrm) < 0
    if flip:
        nrm = -nrm
    u, v = g @ d, g @ nrm
    width, depth = float(u.max() - u.min()), float(v.max() - v.min())
    if width < 0.3 or depth < 0.3:
        raise LayoutError("the master bay is smaller than 0.3 m on the ground")
    ideal = np.array([[u.min(), v.min()], [u.max(), v.min()], [u.max(), v.max()], [u.min(), v.max()]])
    err = float(np.abs(np.c_[u, v] - ideal).max())  # how far the drawn corners are from the ideal rectangle (m)
    lay = ZoneLayout(heading=float(np.degrees(np.arctan2(d[1], d[0]))), flip=flip, width=width, depth=depth, gap=gap, count=int(count), rows=int(rows), row_gap=row_gap)
    dd, nn = lay.axes()
    o = dd * (u.min() - master_index * (width + gap)) + nn * v.min()
    lay.origin = (float(o[0]), float(o[1]))
    return lay, {"width_m": round(width, 3), "depth_m": round(depth, 3), "squareness_m": round(err, 3)}


def inset_quad(quad: np.ndarray, margin: float) -> np.ndarray | None:
    """A convex quad (ground metres) pulled in by `margin` on every side (the inward buffer, without shapely). None when it collapses (under 5 % of its area left)."""
    q = np.asarray(quad, float)
    area = 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    sgn = 1.0 if area > 0 else -1.0  # counter-clockwise: the inside is to the left of each edge
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        e = (b - a) / (np.linalg.norm(b - a) or 1.0)
        lines.append((a + sgn * np.array([-e[1], e[0]]) * margin, e))
    out = []
    for i in range(4):
        (p, e), (r, f) = lines[i - 1], lines[i]
        M = np.array([e, -f]).T
        if abs(np.linalg.det(M)) < 1e-9:
            return None
        out.append(p + e * np.linalg.solve(M, r - p)[0])
    out = np.array(out)
    a2 = 0.5 * float(np.sum(out[:, 0] * np.roll(out[:, 1], -1) - np.roll(out[:, 0], -1) * out[:, 1]))
    return out if a2 * sgn > 0.05 * abs(area) else None


def inset_slots(lay: ZoneLayout, cam, w: int, h: int, margin: float = 0.15) -> tuple[list[dict], list[str]]:
    """to_slots with every bay pulled in by `margin` metres on the ground. A bay that would collapse keeps its full shape and is named in the second return value
    (flag it for review)."""
    from carparking.geometry_live import VEHICLE_HALF
    out, kept_full = [], []
    for i, gq in enumerate(lay.bays_ground()):
        ins = inset_quad(gq, margin)
        if ins is None:
            ins = gq
            kept_full.append(lay.names[i] if i < len(lay.names) else f"B{i + 1}")
        px = to_image(cam, ins) / (w, h)
        if not np.isfinite(px).all():
            continue
        mid = gq.mean(axis=0)
        far = to_image(cam, np.array([mid, mid + (0.0, VEHICLE_HALF)])) / (w, h)
        lf = float(far[0][1] - far[1][1]) if np.isfinite(far).all() else 0.0
        out.append({"id": int(lay.ids[i]) if i < len(lay.ids) else i + 1, "name": lay.names[i] if i < len(lay.names) else f"B{i + 1}",
                    "poly": np.clip(px, 0.0, 1.0).round(4).tolist(), "lift": round(max(lf, 0.0), 5)})
    return out, kept_full
