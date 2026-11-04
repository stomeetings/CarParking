"""Multi-slot encroachment ("spillage", post-demo item 1, docs/post_demo_engineering_plan.md section 1): a free bay that a neighbour's
vehicle stands on is not a usable space. Pure functions and one small state machine; nothing here touches the slot states.

  spill_map()     which FREE, unowned bays are ENCROACHED by the vehicles standing in other bays, and by whom.
  AvailTracker    the debounce in front of it: a bay becomes ENCROACHED after `hold` seconds AND `min_enter` frames, and clear again after
                  `exit_hold` seconds AND `min_leave` frames (the slower of the two, as every temporal filter of the plan: the worker can
                  take 5-60 s per frame); a bay that stops being eligible (a car arrived in it) is cleared at once.

The rules (all numbers in SpillCfg, the plan's):
  cover(v, s)  the share of the bay's ground that lies within the vehicle's lateral extent (the x-range of its ground footprint, the
               wheels), for a vehicle on the bay's row (see alignment). Not the area under the footprint strip itself: that strip is only
               the bottom 30 % of the box, so a car wholly over a bay would score ~0.4 and 0.30 would mean 73 % of its width; this way 0.30
               means 30 % of the bay, also on a bay drawn in perspective. Only vehicles BUSY-confirmed in a bay of their own count (a car
               still rolling locks nothing) - or a vehicle that holds no bay at all (owner 0: too wide for any single bay under the
               matcher's bay_share rule, a van over three bays), whose bays would otherwise all be counted FREE; the debounce below keeps
               a car merely driving past from locking anything. Only bays of capacity 1 take part (a capacity pin is left alone).
  single car   cover >= share (0.30)  -> ENCROACHED          (the 30 % single-car multi-slot boundary)
  several      sum of covers >= total (0.45)                (two cars each a hand's width over the line)
  hysteresis   once ENCROACHED, it stays until the largest cover < leave (0.20) and the sum < total * leave / share: box jitter at the
               threshold does not flap it.
  alignment    a vehicle counts for a bay only if its ground line (the bottom of its box) lies within `align` (0.30) of the bay's height
               above its top or below its bottom: same row, same depth. A car standing in front of or behind the bay is not on it.
The owner always wins: a bay with a vehicle of its own is never ENCROACHED (it is BUSY), and a BUSY or UNKNOWN bay is not a candidate.
Ground metric (GroundMeasure, post-demo item 1b): on a camera with an active calibration the same rules run on the GROUND instead of the
picture. The vehicle's ground footprint (its measured contact line plus the depth a car of that view reaches behind it, geometry.bev.
car_footprint, the wheels trimmed `wheel_inset` at each end) and the bay are polygons in metres, and cover = the AREA overlap as a share of the smaller of bay and
footprint (a car with 40 % of itself on a bay scores 0.4, whatever the depth drawn for the bay), so it works along whichever axis the neighbours lie (side by side, nose to tail, or stacked in depth in the picture): the first
version measured only the contact line's reach across the bay's short side, which is a point on a layout where the neighbour is behind, not beside
(camera-1's bays B1..B7 are stacked in depth) and so never saw anything. The footprint must reach `margin_m` into the bay (its metric edge moved
inward) to count at all. The covers are shares, so a calibration error in scale (the camera height) cancels; only the margin is in metres, and it
is capped at a quarter of the bay's short side. `mode` "union" (live option spillage_ground) also keeps the pixel cover where it is higher: the
tuning fallback when the ground reading is too cautious on some camera. Where the calibration cannot be trusted for a pair (above the horizon,
beyond CAR_MAX_RES m/px, another frame shape) that pair uses the pixel rule above, so no calibration = exactly the pixel behaviour.
Spill.free_m is then the bay's short side left, in metres."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpillCfg:
    share: float = 0.30  # one vehicle's cover that makes a free bay ENCROACHED
    total: float = 0.45  # the sum of several vehicles' covers that does
    leave: float = 0.20  # hysteresis: it stays ENCROACHED until the largest cover is below this
    align: float = 0.30  # the depth band, as a share of the bay's height
    margin_m: float = 0.15  # ground metric only: the wheels must reach this far (metres) into the bay to count at all
    wheel_inset: float = 0.06  # ground metric only: the wheels stand this share of the box width inside each end (bumpers, mirrors); untuned
    depth_trust: float = 0.85  # ground metric only: the share of the guessed depth of a vehicle's footprint (its far end is not seen) that counts

    def __post_init__(self):
        if not 0 < self.leave < self.share <= 1 or not self.share <= self.total <= 1 or not 0 <= self.align <= 2 or self.margin_m < 0 or not 0 <= self.wheel_inset < 0.4 or not 0.3 <= self.depth_trust <= 1:
            raise ValueError("SpillCfg needs 0 < leave < share <= total <= 1, 0 <= align <= 2 and margin_m >= 0 and 0 <= wheel_inset < 0.4 and 0.3 <= depth_trust <= 1")


CFG = SpillCfg()


@dataclass(frozen=True)
class Spill:
    slot: int  # the encroached bay
    by: tuple[int, ...]  # the bays whose vehicles stand on it, largest cover first (0 = a vehicle that holds no bay)
    cover: float  # the largest single cover
    total: float  # the sum of the covers
    free_m: float | None = None  # metres of the bay's width the vehicles leave (ground metric only, else None)
    metric: bool = False  # the covers were measured on the ground (every pair of this bay)


def _clip(poly: np.ndarray, p0: np.ndarray, p1: np.ndarray, inset: float = 0.0) -> np.ndarray:
    """Sutherland-Hodgman: `poly` (Nx2) kept on the left of the directed edge p0->p1 moved `inset` further left (a counter-clockwise bay: inward)."""
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)  # the left normal
    side = (poly - p0) @ n - inset
    out = []
    for i in range(len(poly)):
        a, b, sa, sb = poly[i], poly[(i + 1) % len(poly)], side[i], side[(i + 1) % len(poly)]
        if sa >= 0:
            out.append(a)
        if (sa >= 0) != (sb >= 0):
            out.append(a + (b - a) * (sa / (sa - sb)))
    return np.array(out) if out else np.zeros((0, 2))


def _area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    return poly if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0 else poly[::-1]


def _overlap(fp: np.ndarray, bay: np.ndarray, inset: float = 0.0) -> float:
    """Area of the convex footprint `fp` inside the convex counter-clockwise `bay` (its edges moved `inset` inward)."""
    out = fp
    for i in range(len(bay)):
        out = _clip(out, bay[i], bay[(i + 1) % len(bay)], inset)
        if len(out) < 3:
            return 0.0
    return _area(out)


class GroundMeasure:
    """cover(k, i) of the ground metric for one frame: bay k (fractions polygon) against vehicle footprint i, or None where the calibration
    cannot answer (the caller then uses the pixel rule for that pair). cam: a geometry CameraModel for this frame's shape; shape = (w, h).
    mode: "auto" (the ground reading alone) or "union" (spill_map also keeps the pixel cover when it is higher)."""

    def __init__(self, cam, shape, footprints: list, cfg: SpillCfg = CFG, max_res: float | None = None, mode: str = "auto"):
        from carparking.geometry.camera import ground_resolution, to_ground  # (the geometry package is only needed on a calibrated camera)
        from carparking.geometry.pipeline import CAR_MAX_RES
        self.cam, self.cfg, self.mode = cam, cfg, "union" if mode == "union" else "auto"
        self.w, self.h = int(shape[0]), int(shape[1])
        self.max_res = CAR_MAX_RES if max_res is None else max_res
        self._to_ground, self._res = to_ground, ground_resolution
        self._bays: dict[int, tuple | None] = {}
        self._cars = [self._car(fp) for fp in footprints]

    def _px(self, pts) -> np.ndarray:
        return np.asarray(pts, float).reshape(-1, 2) * (self.w, self.h)

    def _ok(self, px) -> bool:
        return bool((self._res(self.cam, px) <= self.max_res).all())  # NaN compares False: above the horizon is not ok

    def _car(self, fp):
        """The vehicle's ground footprint (Nx2 metres, counter-clockwise), or None: its bottom edge is the measured contact line (trimmed at
        each end, the wheels stand inside the body), the car reaches behind it by the depth bev.car_footprint guesses for that view."""
        from carparking.geometry.bev import CAR_END_DEPTH, CAR_SIDE_DEPTH, car_footprint
        fp = np.asarray(fp, float)
        y = float(fp[:, 1].max())
        x0, x1 = float(fp[:, 0].min()), float(fp[:, 0].max())
        inset = self.cfg.wheel_inset * (x1 - x0)  # the wheels stand inside the body's ends
        px = self._px([(x0 + inset, y), (x1 - inset, y)])
        if not self._ok(px):
            return None
        t = self.cfg.depth_trust
        try:
            quad = car_footprint(self.cam, (px[0, 0], 0.0, px[1, 0], px[0, 1]), CAR_SIDE_DEPTH * t, CAR_END_DEPTH * t)
        except Exception:  # noqa: BLE001
            return None
        if quad is None or not np.isfinite(quad[0]).all():
            return None
        return _ccw(np.asarray(quad[0], float))

    def _bay(self, k, poly):
        """(convex counter-clockwise polygon in metres, its short side in metres), or None if the bay cannot be placed."""
        if k not in self._bays:
            import cv2
            px = self._px(poly)
            res = None
            if self._ok(px):
                g = self._to_ground(self.cam, px)
                if np.isfinite(g).all():
                    hull = cv2.convexHull(g.astype(np.float32)).reshape(-1, 2).astype(float)
                    if len(hull) >= 3 and _area(hull) > 0:
                        (_, _), (rw, rh), _ = cv2.minAreaRect(g.astype(np.float32))
                        res = (_ccw(hull), float(min(rw, rh)), _area(hull))
            self._bays[k] = res
        return self._bays[k]

    def width_m(self, k, poly):
        bay = self._bay(k, poly)
        return None if bay is None else bay[1]

    def cover(self, k, poly, i) -> float | None:
        car, bay = self._cars[i], self._bay(k, poly)
        if car is None or bay is None:
            return None
        hull, short, area = bay
        got = _overlap(car, hull)
        if got <= 0:
            return 0.0
        # the footprint must really reach into the bay: past its edge by the margin (never more than a quarter of the bay's short side)
        if _overlap(car, hull, min(self.cfg.margin_m, 0.25 * short)) <= 0:
            return 0.0
        return min(1.0, got / min(area, _area(car)))  # the share of the smaller of the two: a car 40 % over a bay is 0.4 whatever the bay's depth


def spill_map(slots: dict[int, np.ndarray], footprints: list, owners: list, candidates, prev=(), cfg: SpillCfg = CFG,
              ground: "GroundMeasure | None" = None) -> dict[int, Spill]:
    """slots: bay id -> polygon (fractions). footprints[i]: the ground footprint of vehicle i (polygon, fractions); owners[i]: the bay it is
    BUSY-confirmed in, 0 for a vehicle that holds no bay, or None (such a vehicle is no source). candidates: the bays that may be encroached (free, unowned). prev: the
    bays encroached in the previous frame (the spatial hysteresis)."""
    from carparking.fixed_slots import poly_inside  # (fixed_slots imports this module: a late import breaks the cycle)
    out = {}
    sources = [(i, fp, o) for i, (fp, o) in enumerate(zip(footprints, owners)) if o is not None]
    for k in candidates:
        poly = np.asarray(slots[k], float)
        top, bottom = float(poly[:, 1].min()), float(poly[:, 1].max())
        pad = cfg.align * (bottom - top)
        covers: dict[int, float] = {}
        metric = pixel = False
        for i, fp, o in sources:
            if o == k:
                continue
            c = ground.cover(k, poly, i) if ground is not None else None
            if c is not None and ground.mode != "union":  # measured on the ground
                metric = True
                if c > 0:
                    covers[o] = max(covers.get(o, 0.0), c)
                continue
            p = 0.0  # the pixel rule
            if top - pad <= float(np.asarray(fp)[:, 1].max()) <= bottom + pad:
                x0, x1 = float(np.asarray(fp)[:, 0].min()), float(np.asarray(fp)[:, 0].max())
                if not (x1 < poly[:, 0].min() or x0 > poly[:, 0].max()):  # (the pre-filter: some lateral overlap with the bay)
                    p = poly_inside(poly, [(x0, top), (x1, top), (x1, bottom), (x0, bottom)])  # the share of the bay inside the vehicle's slab
            if c is not None:  # union: the larger of the two readings
                metric = metric or c >= p
                pixel = pixel or p > c
                p = max(p, c)
            elif p > 0:
                pixel = True
            if p > 0:
                covers[o] = max(covers.get(o, 0.0), p)
        if not covers:
            continue
        biggest, summed = max(covers.values()), sum(covers.values())
        hit = (biggest >= cfg.leave or summed >= cfg.total * cfg.leave / cfg.share) if k in prev else \
            (biggest >= cfg.share or summed >= cfg.total)
        if hit:
            width = ground.width_m(k, poly) if metric and not pixel else None
            out[k] = Spill(k, tuple(sorted(covers, key=covers.get, reverse=True)), round(biggest, 3), round(summed, 3),
                           None if width is None else round(max(0.0, width * (1 - summed)), 2), bool(metric and not pixel))
    return out


class AvailTracker:
    """The debounce of spill_map(): .active = the bays shown ENCROACHED (slot -> Spill, the latest reading)."""

    def __init__(self, min_enter: int = 2, min_leave: int = 3, events: tuple[str, str] = ("encroached", "encroach_cleared"),
                 notes: tuple[str, str] = ("cover below the leave share", "spillage switched off")):
        # (events / notes: this layer's names; blocked.py builds its own tracker with "blocked" / "unblocked", the defaults are spillage's)
        self.min_enter, self.min_leave = min_enter, min_leave
        self.events, self.notes = events, notes
        self.active: dict[int, Spill] = {}
        self._enter: dict[int, list] = {}  # slot -> [first time, frames] of an unconfirmed encroachment
        self._leave: dict[int, list] = {}  # slot -> [first time, frames] of an unconfirmed clearing

    def update(self, raw: dict[int, Spill], now: float, hold: float, exit_hold: float, eligible) -> list[dict]:
        """One frame. raw: spill_map()'s answer; eligible: the bays that may be encroached at all now. -> events
        [{"slot", "event": "encroached" / "encroach_cleared", "time", "note"}] for the bays that changed."""
        events = []
        for k in list(self.active):
            if k not in eligible:  # a car arrived, the bay was removed or went UNKNOWN: nothing to wait for
                events.append(self._clear(k, now, "bay no longer free"))
        for k in [k for k in list(self._enter) if k not in raw or k not in eligible]:
            del self._enter[k]
        for k in [k for k in list(self._leave) if k in raw or k not in self.active]:
            del self._leave[k]
        for k, sp in raw.items():
            if k not in eligible:
                continue
            if k in self.active:
                self.active[k] = sp
                continue
            t0, n = self._enter.setdefault(k, [now, 0])
            self._enter[k][1] = n = n + 1
            if now - t0 >= hold and n >= self.min_enter:
                del self._enter[k]
                self.active[k] = sp
                events.append({"slot": k, "event": self.events[0], "time": now, "note": "by " + ",".join(map(str, sp.by))})
        for k in [k for k in self.active if k not in raw]:
            t0, n = self._leave.setdefault(k, [now, 0])
            self._leave[k][1] = n = n + 1
            if now - t0 >= exit_hold and n >= self.min_leave:
                events.append(self._clear(k, now, self.notes[0]))
        return events

    def _clear(self, k: int, now: float, note: str) -> dict:
        del self.active[k]
        self._leave.pop(k, None)
        return {"slot": k, "event": self.events[1], "time": now, "note": note}

    def clear_all(self, now: float) -> list[dict]:
        events = [self._clear(k, now, self.notes[1]) for k in list(self.active)]
        self._enter.clear()
        self._leave.clear()
        return events
