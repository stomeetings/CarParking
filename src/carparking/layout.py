"""A zone's bays as a parametric layout in GROUND METRES, with per-bay offset overrides.

The pixels of a bay depend on the camera; its place on the ground does not. So a zone is stored as
  * a lattice: origin O (ground), unit direction d along the row, n across it (away from the camera), bay `width` x `depth`, `gap` between bays in a row,
    `n` bays per row, `rows`, `row_gap`;
  * `offsets`: for any bay, what the user changed by hand, in the zone's own (u, v) metres (u along d, v along n), so they are relative to the lattice and
    survive a new origin / direction / width / count (`with_params`) and a new camera (`refit`):
      - du, dv      shift of the bay
      - dw, dd      wider / deeper (about the bay's front-left corner)
      - rot         turn in degrees about the bay's centre
      - corners     4 absolute (u, v) corners, for a bay whose vertices were dragged freely (front-left, front-right, back-right, back-left)
Pixels are always derived: `to_slots(cam, w, h)` projects the ground quads through a camera (geometry.camera.to_image) and gives the same fixed-slot dicts the
rest of the sample test uses (fractions, ids, names, lift).

Two ways a recalibration meets the user's edits:
  * the camera is the same, the lattice changes (new count, width, direction ...)  -> `with_params(...)`: the offsets stay, by bay index;
  * the camera changes (GeoCalib run again, drift re-anchor)                         -> `refit(slots, new_cam, w, h)`: the bays as they stand on the picture
    (edits included) are taken to the ground by the NEW camera and re-expressed as lattice + offsets, so what the user placed stays where they placed it.
Pure functions on numpy; nothing is read or written."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from carparking.geometry.camera import to_ground, to_image

CORNERS = 4
RECT_TOL_M = 0.03  # a fitted bay within this of its lattice cell, in every corner, needs no offset
ROT_TOL_DEG = 0.3
MIN_AREA = 1e-4  # a projected bay smaller than this share of the picture (a sliver on its edge, a bay wholly outside) is not a slot (fixed_slots.validate_slots agrees)


class LayoutError(ValueError):
    pass


def poly_area(poly) -> float:
    """Unsigned area of a polygon (shoelace), in the units of its points."""
    p = np.asarray(poly, float).reshape(-1, 2)
    return 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))


def project_bays(cam, quads: list[np.ndarray], w: int, h: int, lift: bool = True) -> list[tuple[np.ndarray, float] | None]:
    """Ground quads (each 4 x 2 m) -> per quad (its corners as picture fractions clipped to 0..1, its lift) or None when it cannot be a slot: a corner behind the
    camera / above the horizon, or less than MIN_AREA of the picture once clipped (wholly outside). One projection call for all corners and one for all lifts
    (to_image is element-wise, so the numbers equal a bay-by-bay projection; the layout is re-sent on every move of the margin slider)."""
    from carparking.geometry_live import VEHICLE_HALF
    if not quads:
        return []
    q = np.asarray(quads, float).reshape(len(quads), CORNERS, 2)
    px = (to_image(cam, q.reshape(-1, 2)) / (w, h)).reshape(len(quads), CORNERS, 2)
    lifts = np.zeros(len(quads))
    if lift:
        mid = q.mean(axis=1)
        far = (to_image(cam, np.stack([mid, mid + (0.0, VEHICLE_HALF)], axis=1).reshape(-1, 2)) / (w, h)).reshape(len(quads), 2, 2)
        lf = far[:, 0, 1] - far[:, 1, 1]
        lifts = np.where(np.isfinite(far).all(axis=(1, 2)), np.maximum(lf, 0.0), 0.0)
    out: list[tuple[np.ndarray, float] | None] = []
    for i in range(len(quads)):
        if not np.isfinite(px[i]).all():
            out.append(None)
            continue
        poly = np.clip(px[i], 0.0, 1.0)
        out.append((poly, float(lifts[i])) if poly_area(poly) >= MIN_AREA else None)
    return out


@dataclass
class BayOffset:
    du: float = 0.0
    dv: float = 0.0
    dw: float = 0.0
    dd: float = 0.0
    rot: float = 0.0
    corners: list[list[float]] | None = None  # 4 x (u, v), absolute in the zone frame

    def is_zero(self) -> bool:
        return self.corners is None and not any((self.du, self.dv, self.dw, self.dd, self.rot))

    def to_dict(self) -> dict:
        d = {k: round(float(getattr(self, k)), 4) for k in ("du", "dv", "dw", "dd", "rot") if getattr(self, k)}
        if self.corners is not None:
            d["corners"] = [[round(float(a), 4), round(float(b), 4)] for a, b in self.corners]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BayOffset":
        c = d.get("corners")
        return cls(**{k: float(d.get(k, 0.0)) for k in ("du", "dv", "dw", "dd", "rot")}, corners=None if c is None else [[float(a), float(b)] for a, b in c])


@dataclass
class ZoneLayout:
    origin: tuple[float, float] = (0.0, 0.0)  # ground metres of the lattice's first front-left corner
    heading: float = 0.0  # degrees: the row's direction d on the ground (0 = +x)
    flip: bool = False  # n = d turned +90 deg (False) or -90 deg (True): which side of the row is "back"
    width: float = 2.5
    depth: float = 5.0
    gap: float = 0.0
    count: int = 1  # bays per row
    rows: int = 1
    row_gap: float = 0.0
    stall_angle: float = 90.0  # degrees between the row direction and the bay separators: 90 = square bays, less = they lean forward along the row (angled / diagonal parking)
    offsets: dict[int, BayOffset] = field(default_factory=dict)  # by bay index (row * count + i)
    ids: list[int] = field(default_factory=list)  # the bay ids / names to hand out, by index (default 1..n, "B<n>")
    names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # what a fit had to decide for the user (refit: two bays wanted one cell ...); not stored

    @property
    def lean(self) -> float:
        """cot(stall_angle): how far a separator moves along the row per metre of depth (0 for square bays). The back corners of every bay shift by depth * lean."""
        a = float(self.stall_angle)
        return 0.0 if abs(a - 90.0) < 1e-9 else float(1.0 / np.tan(np.radians(a)))

    # ---- frame ----------------------------------------------------------------------------------------------------------------------------
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        t = np.radians(self.heading)
        d = np.array([np.cos(t), np.sin(t)])
        n = np.array([-d[1], d[0]]) * (-1.0 if self.flip else 1.0)
        return d, n

    def to_ground(self, uv) -> np.ndarray:
        d, n = self.axes()
        uv = np.asarray(uv, float).reshape(-1, 2)
        return np.asarray(self.origin, float) + uv[:, :1] * d + uv[:, 1:2] * n

    def to_uv(self, g) -> np.ndarray:
        d, n = self.axes()
        p = np.asarray(g, float).reshape(-1, 2) - np.asarray(self.origin, float)
        return np.c_[p @ d, p @ n]

    @property
    def total(self) -> int:
        return max(0, int(self.count)) * max(0, int(self.rows))

    # ---- bays -----------------------------------------------------------------------------------------------------------------------------
    def cell(self, idx: int) -> np.ndarray:
        """The lattice cell of bay idx in (u, v): front-left, front-right, back-right, back-left."""
        r, i = divmod(idx, max(1, int(self.count)))
        u0, v0 = i * (self.width + self.gap), r * (self.depth + self.row_gap)
        s = self.depth * self.lean  # a shear on the ground about the row's front line: the front edge and the step stay, the separators lean
        return np.array([[u0, v0], [u0 + self.width, v0], [u0 + self.width + s, v0 + self.depth], [u0 + s, v0 + self.depth]])

    def bay_uv(self, idx: int) -> np.ndarray:
        """Bay idx as it stands: its lattice cell with its offsets applied (in (u, v))."""
        off = self.offsets.get(idx)
        q = self.cell(idx)
        if off is None:
            return q
        if off.corners is not None:
            return np.asarray(off.corners, float)
        q = q + [[0.0, 0.0], [off.dw, 0.0], [off.dw, off.dd], [0.0, off.dd]]  # wider / deeper about the front-left corner
        if off.rot:
            c, t = q.mean(axis=0), np.radians(off.rot)
            rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            q = (q - c) @ rot.T + c
        return q + [off.du, off.dv]

    def bays_ground(self) -> list[np.ndarray]:
        return [self.to_ground(self.bay_uv(i)) for i in range(self.total)]

    def slot_of(self, i: int, poly: np.ndarray, lift: float) -> dict:
        """The fixed-slot dict of bay i from its projected corners (fractions) and lift: the id / name the layout hands out by index."""
        sid = self.ids[i] if i < len(self.ids) else i + 1
        name = self.names[i] if i < len(self.names) else f"B{i + 1}"
        return {"id": int(sid), "name": name, "poly": np.asarray(poly, float).round(4).tolist(), "lift": round(float(lift), 5)}

    def to_slots(self, cam, w: int, h: int, lift: bool = True) -> list[dict]:
        """Fixed-slot dicts (image fractions) of every bay through `cam`. A bay with a corner behind the camera / above the horizon, or wholly outside the
        picture (under MIN_AREA once clipped), is left out."""
        return [self.slot_of(i, *p) for i, p in enumerate(project_bays(cam, self.bays_ground(), w, h, lift)) if p is not None]

    # ---- edits ----------------------------------------------------------------------------------------------------------------------------
    def set_offset(self, idx: int, off: BayOffset | None) -> None:
        if not 0 <= idx < self.total:
            raise LayoutError(f"bay {idx} is not in the layout ({self.total} bays)")
        if off is None or off.is_zero():
            self.offsets.pop(idx, None)
        else:
            self.offsets[idx] = off

    def edit_bay_px(self, idx: int, quad_px, cam, w: int, h: int) -> BayOffset:
        """The user dragged bay idx to this quad (pixels): store it as an offset in the zone's metres (a free quad becomes `corners`)."""
        g = to_ground(cam, np.asarray(quad_px, float).reshape(-1, 2) * (w, h) if np.nanmax(np.asarray(quad_px)) <= 1.5 else np.asarray(quad_px, float))
        if not np.isfinite(g).all() or len(g) != CORNERS:
            raise LayoutError("the edited bay is not on measurable ground")
        off = offset_for(self.cell(idx), self.to_uv(g))
        self.set_offset(idx, off)
        return off

    # ---- regenerate / refit ---------------------------------------------------------------------------------------------------------------
    def with_params(self, **kw: Any) -> "ZoneLayout":
        """A copy with new lattice parameters; the offsets of the bays that still exist stay (by index, relative to the new lattice). Nothing is shared with
        the original (its ids, names and offsets are copied)."""
        d = {**self.__dict__, "ids": list(self.ids), "names": list(self.names), "warnings": list(self.warnings), **kw}
        out = ZoneLayout(**d)
        out.offsets = {i: BayOffset(**{**o.__dict__, "corners": None if o.corners is None else [list(c) for c in o.corners]}) for i, o in self.offsets.items() if i < out.total}
        return out

    def to_dict(self) -> dict:
        return {"origin": [round(float(x), 4) for x in self.origin], "heading": round(float(self.heading), 4), "flip": bool(self.flip), "width": round(self.width, 4),
                "depth": round(self.depth, 4), "gap": round(self.gap, 4), "count": int(self.count), "rows": int(self.rows), "row_gap": round(self.row_gap, 4),
                "stall_angle": round(float(self.stall_angle), 3),
                "offsets": {str(i): o.to_dict() for i, o in sorted(self.offsets.items()) if not o.is_zero()}, "ids": list(self.ids), "names": list(self.names)}

    @classmethod
    def from_dict(cls, d: dict) -> "ZoneLayout":
        try:
            lay = cls(origin=tuple(map(float, d["origin"])), heading=float(d["heading"]), flip=bool(d.get("flip", False)), width=float(d["width"]), depth=float(d["depth"]),
                      gap=float(d.get("gap", 0.0)), count=int(d["count"]), rows=int(d.get("rows", 1)), row_gap=float(d.get("row_gap", 0.0)), stall_angle=float(d.get("stall_angle", 90.0)),
                      offsets={int(i): BayOffset.from_dict(o) for i, o in (d.get("offsets") or {}).items()}, ids=[int(x) for x in d.get("ids", [])],
                      names=list(d.get("names", [])))
        except (KeyError, TypeError, ValueError) as e:
            raise LayoutError(f"not a layout: {e}") from None
        if len(lay.origin) != 2 or not all(np.isfinite([*lay.origin, lay.heading, lay.width, lay.depth, lay.gap, lay.row_gap, lay.stall_angle])):
            raise LayoutError("not a layout: a number is missing or not finite")
        if not MIN_STALL_DEG <= lay.stall_angle <= MAX_STALL_DEG:  # (0 or 180 would make the lean infinite: every corner NaN, every bay silently dropped)
            raise LayoutError(f"not a layout: the bay angle is {MIN_STALL_DEG:.0f} to {MAX_STALL_DEG:.0f} degrees")
        if lay.width <= 0 or lay.depth <= 0 or lay.count < 1 or lay.rows < 1:
            raise LayoutError("not a layout: the bay size, count and rows must be positive")
        return lay


def offset_for(cell_uv: np.ndarray, quad_uv: np.ndarray) -> BayOffset:
    """What turns the lattice cell into this quad (both 4 x (u, v), front-left first): the simplest offset that reproduces it within RECT_TOL_M,
    else the free corners."""
    cell, quad = np.asarray(cell_uv, float), np.asarray(quad_uv, float)
    if np.abs(quad - cell).max() <= RECT_TOL_M:
        return BayOffset()
    # a rectangle turned by `rot` about its centre, resized and shifted?
    e = (quad[1] - quad[0] + quad[2] - quad[3]) / 2  # front-ish edge direction, averaged
    rot = float(np.degrees(np.arctan2(e[1], e[0])))
    t = np.radians(-rot)
    unrot = (quad - quad.mean(axis=0)) @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T
    wd, dd = float(unrot[:, 0].max() - unrot[:, 0].min()), float(unrot[:, 1].max() - unrot[:, 1].min())
    cand = BayOffset(du=0.0, dv=0.0, dw=wd - (cell[1, 0] - cell[0, 0]), dd=dd - (cell[3, 1] - cell[0, 1]), rot=rot if abs(rot) > ROT_TOL_DEG else 0.0)
    made = _apply(cell, cand)
    cand.du, cand.dv = (quad.mean(axis=0) - made.mean(axis=0)).tolist()
    if np.abs(_apply(cell, cand) - quad).max() <= RECT_TOL_M:
        return cand
    return BayOffset(corners=quad.tolist())


def _apply(cell: np.ndarray, off: BayOffset) -> np.ndarray:
    q = cell + [[0.0, 0.0], [off.dw, 0.0], [off.dw, off.dd], [0.0, off.dd]]
    if off.rot:
        c, t = q.mean(axis=0), np.radians(off.rot)
        q = (q - c) @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T + c
    return q + [off.du, off.dv]


def layout_from_row(cam, front_px, back_px, count: int, rows: int = 1, depth: float | None = None, gap: float = 0.0, row_gap: float = 0.0,
                    w: int = 1, h: int = 1) -> ZoneLayout:
    """A default layout from a drawn row: its front edge (2 pixel points, fractions of the picture) and one point on its back side, N bays along the front edge.
    The width is the front edge's ground length over N (less the gaps); the depth is the back point's ground distance across the row unless given."""
    f = to_ground(cam, np.asarray(front_px, float).reshape(2, 2) * (w, h))
    b = to_ground(cam, np.asarray(back_px, float).reshape(-1, 2)[:1] * (w, h))
    if not (np.isfinite(f).all() and np.isfinite(b).all()):
        raise LayoutError("the row is not on measurable ground (above the horizon?)")
    d = f[1] - f[0]
    length = float(np.linalg.norm(d))
    if length < 0.5 or count < 1:
        raise LayoutError("the row is too short, or the expected count is below 1")
    d = d / length
    across = float((b[0] - f[0]) @ np.array([-d[1], d[0]]))
    width = (length - gap * (count - 1)) / count
    if width <= 0.3:
        raise LayoutError("the bays would be narrower than 0.3 m: lower the count or the gap")
    return ZoneLayout(origin=(float(f[0][0]), float(f[0][1])), heading=float(np.degrees(np.arctan2(d[1], d[0]))), flip=across < 0, width=width,
                      depth=float(depth if depth else max(abs(across), 2.0)), gap=gap, count=int(count), rows=int(rows), row_gap=row_gap)


MIN_STALL_DEG, MAX_STALL_DEG = 20.0, 160.0
LEAN_TOL = float(np.tan(np.radians(1.0)))  # under 1 degree off square is square


def lean_of(quad_uv: np.ndarray) -> float | None:
    """cot of the stall angle of one bay (4 x (u, v), front-left first, front edge along u): how far its separators lean along the row per metre of depth. None when the
    corners are not in that order (the front edge is not along the row) or the bay is shallower than 0.3 m."""
    q = np.asarray(quad_uv, float)
    front = (q[1] - q[0] + q[2] - q[3]) / 2
    side = (q[3] - q[0] + q[2] - q[1]) / 2
    if front[0] <= 0.3 or abs(front[1]) > 0.2 * abs(front[0]) or side[1] < 0.3:
        return None
    return float(side[0] / side[1])


def _median_lean(uvs: list[np.ndarray]) -> float | None:
    """The row's lean (median over its bays), or None when it is square (within a degree), unreliable (under half the bays give one) or outside 20-160 degrees."""
    ls = [x for x in (lean_of(q) for q in uvs) if x is not None]
    if len(ls) < max(1, len(uvs) // 2):
        return None
    m = float(np.median(ls))
    lim = float(1.0 / np.tan(np.radians(MIN_STALL_DEG)))
    return m if LEAN_TOL < abs(m) <= lim else None


def refit(slots: list[dict], cam, w: int, h: int, count: int | None = None) -> ZoneLayout:
    """The bays as they stand on the picture (edits included) -> a layout, through `cam` (a NEW camera after a recalibration): the lattice (direction, width,
    depth, gap, rows) is the median of the bays, each bay's difference from it is its offset, so nothing the user placed moves."""
    if not slots:
        raise LayoutError("no bays to fit")
    quads = [to_ground(cam, np.asarray(s["poly"], float) * (w, h)) for s in slots]
    keep = [i for i, g in enumerate(quads) if np.isfinite(g).all() and len(g) == CORNERS]
    if not keep:
        raise LayoutError("no bay is on measurable ground")
    quads = [quads[i] for i in keep]
    slots = [slots[i] for i in keep]
    d = np.mean([(g[1] - g[0] + g[2] - g[3]) / 2 for g in quads], axis=0)
    d = d / (np.linalg.norm(d) or 1.0)
    nrm = np.array([-d[1], d[0]])
    back = float(np.mean([((g[3] - g[0] + g[2] - g[1]) / 2) @ nrm for g in quads]))
    flip = back < 0
    if flip:
        nrm = -nrm
    lay = ZoneLayout(heading=float(np.degrees(np.arctan2(d[1], d[0]))), flip=flip)
    d, nrm = lay.axes()
    pts = np.vstack(quads)
    o = np.array([(pts @ d).min(), (pts @ nrm).min()])
    lay.origin = (float(d[0] * o[0] + nrm[0] * o[1]), float(d[1] * o[0] + nrm[1] * o[1]))
    uv = [lay.to_uv(g) for g in quads]
    lean = _median_lean(uv)
    if lean is not None:  # angled bays: the separators lean on the ground; measure each bay from its front edge, keep the angle as the layout's (not as per-bay offsets)
        lay.stall_angle = float(np.degrees(np.arctan2(1.0, lean)))
        shift = float(min(q[0, 0] for q in uv))
        lay.origin = (lay.origin[0] + d[0] * shift, lay.origin[1] + d[1] * shift)
        uv = [lay.to_uv(g) for g in quads]
        u0 = np.array([q[0, 0] for q in uv])
        wd = np.array([(q[1, 0] - q[0, 0] + q[2, 0] - q[3, 0]) / 2 for q in uv])
        dp = np.array([(q[3, 1] - q[0, 1] + q[2, 1] - q[1, 1]) / 2 for q in uv])
    else:
        u0 = np.array([q[:, 0].min() for q in uv])
        wd = np.array([q[:, 0].max() - q[:, 0].min() for q in uv])
        dp = np.array([q[:, 1].max() - q[:, 1].min() for q in uv])
    v0 = np.array([q[:, 1].min() for q in uv])
    lay.width, lay.depth = float(np.median(wd)), float(np.median(dp))
    # rows: bays whose v0 differ by more than half a depth are another row
    order_v = np.argsort(v0)
    row_of = np.zeros(len(uv), int)
    r = 0
    for a, b in zip(order_v, order_v[1:]):
        if v0[b] - v0[a] > 0.5 * lay.depth:
            r += 1
        row_of[b] = r
    rows = r + 1
    per_row = [sorted(np.where(row_of == k)[0], key=lambda i: u0[i]) for k in range(rows)]
    lay.rows = rows
    lay.count = int(count or max(len(p) for p in per_row))
    steps = [u0[p[j + 1]] - u0[p[j]] for p in per_row for j in range(len(p) - 1)]
    lay.gap = float(max(0.0, np.median(steps) - lay.width)) if steps else 0.0
    if rows > 1:
        vr = [float(np.median(v0[np.where(row_of == k)[0]])) for k in range(rows)]
        lay.row_gap = float(max(0.0, np.median(np.diff(vr)) - lay.depth))
    # bays go to their lattice index: by row, then by position along the row (a missing bay leaves a hole: its cell stays empty, not shifted)
    ids, names, off, placed = {}, {}, {}, {}
    step = max(lay.width + lay.gap, 1e-6)
    u1 = u0 + wd  # (a bay's front-edge end along the row)
    for k, p in enumerate(per_row):
        i, prev_end = max(0, int(round(u0[p[0]] / step))), None
        for j in p:
            if prev_end is not None:  # the next cell, plus one per bay-sized empty stretch between this bay and the last (a hole, not a shift)
                i += 1 + int(max(0.0, u0[j] - prev_end) / step + 0.25)
            placed[(k, i)] = j  # (i only grows along a row: no two bays share a cell)
            prev_end = u1[j]
    need = max(i for _, i in placed) + 1
    if need > lay.count:  # a row holds more bays than the count asked for: the lattice grows rather than drop a bay the user placed
        lay.warnings.append(f"a row has {need} bays (holes counted), more than the {lay.count} expected: the row was widened to {need}; check the expected count")
        lay.count = need
    for (k, i), j in placed.items():
        idx = k * lay.count + i
        ids[idx], names[idx] = int(slots[j].get("id", idx + 1)), str(slots[j].get("name", f"B{idx + 1}"))
        o_ = offset_for(lay.cell(idx), uv[j])
        if not o_.is_zero():
            off[idx] = o_
    lay.offsets = off
    # ids / names by index; a hole keeps the default
    top = max(ids) + 1 if ids else 0
    lay.ids = [ids.get(i, 1000 + i) for i in range(top)]
    lay.names = [names.get(i, f"B{i + 1}") for i in range(top)]
    return lay
