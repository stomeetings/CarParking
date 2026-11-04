"""The worker's side of Phase 2 (docs/PHASE2_GEOMETRY_PLAN.md): read a camera's ACTIVE calibration from the versioned
geometry store and turn each frame's detections into metric data for the dashboard: vehicle footprints, the distances
between them (and to other objects), and the capacity of the packed bays.

Strictly opt-in. Nothing here is imported or run unless CARPARK_GEOMETRY_ENABLED is on (scripts/worker.py checks first),
and with it on but no active calibration the feed returns None and the worker behaves exactly as before. A store that
fails (the table not migrated yet, PostgreSQL restarting) is logged once and never raises into the worker's loop: the last
good calibration is served for STALE_AFTER seconds, then the camera counts as uncalibrated.

Two more jobs, both on the worker's own frames and both fail-safe:
  * the DRIFT GUARD: every DRIFT_EVERY seconds the current frame's static background is compared with the reference stored
    with the calibration (carparking/geometry/drift.py). A shift confirmed on DRIFT_CONFIRM checks in a row marks the active
    version STALE (an event in the history) and the camera falls back to the legacy snapshot until a new calibration is
    accepted. An inconclusive check (night, glare, a blocked view) is never an alarm.
  * HISTORY: an occupancy sample (capacity / occupied / vehicles) is stored every SAMPLE_EVERY seconds, and at once when
    the occupancy changes (at most one per SAMPLE_MIN_GAP), old rows pruned hourly (GeometryStore.add_sample / prune_history).

Reading is one cheap query every POLL seconds (GeometryStore.active_version()); the full record is loaded only when the
version changed, so an activation from the setup page is picked up within a few seconds with nothing restarted."""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from .geometry import CameraModel, drift, to_ground, to_image
from .geometry.capacity import BAY, pack_bays
from .geometry.bev import CELL, Grid, car_footprint, closest_on_polygon, closest_points, nearest_pairs
from .geometry.camera import ground_resolution
from .fixed_slots import MAX_EXTRAPOLATED, poly_iou  # the most an expected bay count may add: the camera model's error grows with distance
from .geometry.pipeline import CAR_MAX_RES
from .geometry_store import geometry_enabled, open_geometry

POLL = 5.0  # seconds between looks at the store
STALE_AFTER = 60.0  # seconds a failing store keeps serving the last good calibration
MAX_DIST = 6.0  # metres: nearer than this a distance is worth drawing
SAME_CAR = 1.0  # metres: two footprints with centres this close are one car seen twice (whole frame + zoom detection)
DRIFT_EVERY = 30.0  # seconds between drift checks (one ORB pass + match: ~30 ms)
DRIFT_CONFIRM = 3  # consecutive "drift" checks before a version is marked stale (a passing truck or a gust is not a bump)
STRIKE_TTL = 10  # a strike older than this many check intervals is forgotten (inconclusive nights do not carry it over)
SAMPLE_EVERY = 60.0  # seconds between occupancy samples
SAMPLE_MIN_GAP = 10.0  # ... and the shortest gap when the occupancy changes in between
PRUNE_EVERY = 3600.0
CLOSE_M = 8.0  # metres: notches in the measured surface up to this wide are filled before tiling (see ActiveGeometry.tiles)
DEFAULT_EXTRA = 1  # bays added past the lane's last measured tile when no expected count is set (extrapolated; see ActiveGeometry.tiles)
EDGE_KEEP = 0.5  # share of an extrapolated bay that must project inside the frame, else it (and those beyond it) is dropped
PIN_IOU = 0.3  # an operator-pinned bay replaces the grid bays it overlaps by at least this much (image IoU)
MEASURED, EXTRAPOLATED, MANUAL = "measured", "extrapolated", "manual"  # where a bay of the auto grid comes from
VEHICLE_HALF = 1.0  # metres from a vehicle's near wheels to its centre on the ground (a car side-on ~0.9, end-on ~2)
SNAP_DEG = 10.0  # degrees: a bay orientation this close to the camera's axes is made square to them
TILE_COVER = 0.9  # share of a tile that must be measured surface (the packing's own 0.995 drops bays on a slanted edge)


def merge_pins(slots: list[dict], pins) -> list[dict]:
    """The grid's slots with the operator's pins applied (fixed-slot dicts, polygons as image fractions). A pin replaces
    every grid bay it overlaps by at least PIN_IOU (it takes the lowest id among them, so the other bays keep theirs and the
    history of the bay it stands for goes on); a pin that overlaps none is added after the last bay, in the order of the
    pins. Pins are the operator's word: lift 0 (drawn where the wheels stand), source MANUAL. Bad pins are skipped."""
    out = [dict(s) for s in slots]
    for pin in pins or ():
        try:
            poly = np.asarray(pin["poly"], float)
            hit = [s for s in out if s.get("source") != MANUAL and poly_iou(poly, s["poly"]) >= PIN_IOU]
            sid = min((s["id"] for s in hit), default=max((s["id"] for s in out), default=0) + 1)
            fixed = int(pin["slot_id"]) if pin.get("slot_id") else None  # the pin's own id, once it has one: it never moves
            if fixed and not any(s.get("source") == MANUAL and s["id"] == fixed for s in out):
                sid = fixed
            name = str(pin.get("name") or "").strip()
            slot = {"id": sid, "name": name if name and name != str(pin.get("id")) else f"B{sid}",
                    "poly": poly.round(4).tolist(), "lift": 0.0, "source": MANUAL}
            if pin.get("capacity"):  # the operator's vehicles-per-pin (validate_slots has checked it)
                slot["capacity"] = pin["capacity"]
        except (KeyError, TypeError, ValueError):
            continue
        out = [s for s in out if s not in hit]
        for s in out:  # a bay that happens to hold the pin's own id steps aside (the pin is the one with history)
            if s.get("source") != MANUAL and s["id"] == sid:
                s["id"] = max([x["id"] for x in out] + [sid]) + 1
        out.append(slot)
    bays = [s for s in out if s.get("source") != MANUAL]
    for s in out:
        if s.get("source") == MANUAL and bays:  # the pin binds to the nearest bay of the grid (a label: it never renumbers anything)
            c = np.asarray(s["poly"], float).mean(axis=0)
            s["binds_to"] = min(bays, key=lambda b: float(np.linalg.norm(np.asarray(b["poly"], float).mean(axis=0) - c)))["id"]
    return sorted(out, key=lambda s: s["id"])


class ActiveGeometry:
    """One stored version, ready for a frame: its calibration, bays and the numbers the dashboard shows."""

    def __init__(self, row: dict):
        rec = row["record"]
        self.version, self.cal, self.bays = row["ver"], rec["calibration"], rec["capacity"]["slots"]
        self.image = (rec["image"]["width"], rec["image"]["height"])
        self.surface, self.layout = rec["bev"]["surface_m"], rec["capacity"]  # the measured ground (metres) and how it was packed
        ref = rec.get("reference")  # the calibration still's static-background features (the drift guard's reference)
        self.reference = ref if ref and ref.get("n", 0) >= drift.MIN_REFERENCE else None
        self.summary = {"version": row["ver"], "confidence": self.cal.get("confidence"), "uncertainty": self.cal.get("scale_uncertainty"),
                        "reference_used": "manual" in (self.cal.get("scale_sources") or {}), "h": self.cal.get("h"),
                        "lens": self.cal.get("lens"), "area_m2": rec["bev"]["area_m2"]}
        self.frozen = self._frozen_grid(rec.get("grid"))  # an accepted suggestion: the bays as the operator saw and fine-tuned them
        self._cams: dict[tuple[int, int], CameraModel | None] = {}
        self._grid: list[np.ndarray] | None = None  # the measured tiles, front to back
        self._lane: dict[int, list[np.ndarray]] = {}  # extra bays continuing the lane, by how many (see lane_extension)
        self._outline: np.ndarray | None = None  # the measured ground with its notches closed (metres), see surface_outline

    @staticmethod
    def _frozen_grid(grid) -> list[dict] | None:
        """record["grid"] -> clean fixed-slot dicts (id, name, poly, lift, source), or None when there is none / it is unusable."""
        if not isinstance(grid, list) or not grid:
            return None
        out = []
        for s in grid:
            try:
                poly = np.asarray(s["poly"], float)
                if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or poly.min() < 0 or poly.max() > 1:
                    return None
                out.append({"id": int(s["id"]), "name": str(s.get("name") or f"B{int(s['id'])}"), "poly": poly.round(4).tolist(),
                            "lift": max(float(s.get("lift") or 0.0), 0.0), "source": str(s.get("source") or MEASURED)})
            except (KeyError, TypeError, ValueError):
                return None
        return out if len({s["id"] for s in out}) == len(out) else None

    def tiles(self, expected: int | None = None) -> list[tuple[np.ndarray, str]]:
        """The auto grid's bays front to back as (4x2 polygon in metres, source): the measured ones (MEASURED) and, past the
        lane's last measured tile, extrapolated ones (EXTRAPOLATED). expected = how many bays the lot has: the lane is
        continued until that many exist (at most MAX_EXTRAPOLATED added, never over an existing tile); None adds
        DEFAULT_EXTRA."""
        return self.retile(None, expected)

    def retile(self, anchors, expected: int | None = None) -> list[tuple[np.ndarray, str]]:
        """tiles(), but packed around `anchors` (centres in metres: where vehicles stand NOW) instead of the calibration's own car
        bays; None = the calibration's. Not cached: a suggestion asks once."""
        if anchors is None:
            measured = self.measured()
            lane = self.lane_extension
        else:
            measured, _ = self._tile(anchors)
            lane = lambda n: self._lane_of(measured, n)  # noqa: E731
        want = DEFAULT_EXTRA if expected is None else max(0, min(int(expected) - len(measured), MAX_EXTRAPOLATED))
        tiles = [(t, MEASURED) for t in measured] + [(t, EXTRAPOLATED) for t in lane(want)]
        return sorted(tiles, key=lambda ts: (round(float(ts[0][:, 1].mean()) / 2.0), float(ts[0][:, 0].mean())))

    def surface_outline(self) -> np.ndarray | None:
        """The measured ground as one polygon (metres): the surface with the notches a too-far vehicle left closed, so a vehicle
        standing where the calibration could not measure is still on it. None without a surface."""
        self.measured()
        return self._outline

    def lane_extension(self, n: int) -> list[np.ndarray]:
        """n tiles continuing the lane's own row pitch beyond its last measured tile (extrapolated, not measured: a car stood
        there too far to be measured when the calibration was made, so its ground was cut out of the surface). Only on a lane
        of at least 3 tiles in a row whose last step is one bay wide, never a jump between pieces."""
        if n not in self._lane:
            self._lane[n] = self._lane_of(self.measured(), n)
        return self._lane[n]

    def _lane_of(self, measured: list[np.ndarray], n: int) -> list[np.ndarray]:
        lane = sorted(measured, key=lambda t: float(t[:, 1].mean()))
        ext: list[np.ndarray] = []
        if n > 0 and len(lane) >= 3:
            step = lane[-1] - lane[-2]
            bay = tuple(self.layout.get("bay") or BAY)
            if 0.8 * bay[0] <= np.linalg.norm(step.mean(axis=0)) <= 1.2 * bay[0]:
                ext = [lane[-1] + step * k for k in range(1, n + 1)]
        return ext

    def measured(self) -> list[np.ndarray]:
        """The whole measured surface tiled with standard bays (metres, 4x2 each): same bay size and orientation as the
        calibration's capacity, but with NO driving aisles (the stored `slots` leave 6 m aisles between double rows, which on
        a narrow lot is 3 bays out of 156 m2). The bays the calibration put on parked cars are kept as anchors, so the rows
        line up with how cars really stand."""
        if self._grid is None:
            self._grid, self._outline = self._tile([np.mean(b["poly_m"], axis=0) for b in self.bays if b.get("car")])
        return self._grid

    def _tile(self, anchors) -> tuple[list[np.ndarray], np.ndarray | None]:
        """measured()'s packing around the given anchors -> (tiles front to back, the surface outline with its notches closed)."""
        lay = self.layout
        surf = np.asarray(self.surface, float)
        if len(surf) < 3:
            return [], None
        outline = None
        cell = CELL
        pad = CLOSE_M + 2 * cell
        x0, y0 = surf.min(axis=0) - pad
        x1, y1 = surf.max(axis=0) + pad
        cells = np.zeros((int(np.ceil((y1 - y0) / cell)) + 1, int(np.ceil((x1 - x0) / cell)) + 1), np.uint8)
        cv2.fillPoly(cells, [np.round((surf - (x0, y0)) / cell).astype(np.int32)], 1)
        # a car too far to be measured when the calibration was made was cut out of the ground and left a notch (a vehicle's
        # body is not ground, but it stands on it): close concavities up to CLOSE_M wide so the bay under it is a bay again
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(CLOSE_M / cell) | 1,) * 2)
        cells = cv2.morphologyEx(cells, cv2.MORPH_CLOSE, k)
        found, _ = cv2.findContours(cells, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if found:  # the ground the vehicle-driven bays may stand on (see FixedSlotTracker): the outline, 0.2 m simplified
            biggest = cv2.approxPolyDP(max(found, key=cv2.contourArea), 0.2 / cell, True)[:, 0, :]
            outline = biggest.astype(float) * cell + (x0, y0)
        grid = Grid(cells.astype(bool), x0, y0, cell)
        bay, ang = tuple(lay.get("bay") or BAY), float(lay["orientation_deg"])
        snapped = round(ang / 90.0) * 90.0  # the polygon's long axis is a few degrees off the lane (a jagged outline): bays run
        if abs(ang - snapped) <= SNAP_DEG and lay.get("orientation_source") != "vehicles":  # square to the lane, unless cars show it
            ang = snapped % 180
        tiles = pack_bays(grid, ang, bay, 0.0, TILE_COVER, anchors)[0]
        # where the surface is too narrow for a bay crosswise (a strip beside a pillar), turn the bays 90 degrees to run
        # along it: a second pass over the ground the first one left, never on top of its tiles
        left = grid.cells.astype(np.uint8)
        for t in tiles:
            cv2.fillPoly(left, [np.round((t - (x0, y0)) / cell).astype(np.int32)], 0)
        tiles += pack_bays(Grid(left.astype(bool), x0, y0, cell), (ang + 90) % 180, bay, 0.0, TILE_COVER)[0]
        tiles = sorted(tiles, key=lambda t: (round(float(t[:, 1].mean()) / 2.0), float(t[:, 0].mean())))  # front to back
        return tiles, outline

    def camera(self, w: int, h: int) -> CameraModel | None:
        """The calibrated camera for a frame of w x h pixels (the calibration is in the still's pixels: f and the
        principal point scale with the frame, the angles and the lens do not), or None when the frame is not this
        picture (another aspect ratio)."""
        if (w, h) not in self._cams:
            rw, rh = self.image
            s = w / rw
            c = self.cal
            self._cams[(w, h)] = None if abs(h - rh * s) > 0.01 * h else CameraModel(
                c["f"] * s, c["cx"] * s, c["cy"] * s, c["pitch"], c["roll"], c["h"], c.get("k1", 0.0), c.get("k2", 0.0))
        return self._cams[(w, h)]


class GeometryFeed:
    """Polls the active version of one camera and computes metrics per frame. open_store() -> a GeometryStore (called on each
    poll: PostgreSQL connects per call, SQLite reuses the worker's connection). Single-threaded, like the worker."""

    def __init__(self, open_store, camera_id: str, poll: float = POLL, clock=time.time, log=print,
                 drift_every: float = DRIFT_EVERY, drift_confirm: int = DRIFT_CONFIRM, sample_every: float = SAMPLE_EVERY):
        self.open_store, self.camera_id, self.poll, self.clock, self.log = open_store, camera_id, poll, clock, log
        self.drift_every, self.drift_confirm, self.sample_every = drift_every, max(1, int(drift_confirm)), sample_every
        self.active: ActiveGeometry | None = None
        self.errors = 0
        self._checked = -np.inf
        self._good_at: float | None = None
        self._last_error = ""
        self.drift: dict | None = None  # the last drift check, as published
        self._drift_at = -np.inf
        self._strikes, self._strike_at = 0, -np.inf
        self._baseline: dict | None = None  # a version made before the drift guard has no stored reference: the first good frame
        self._baseline_for: int | None = None  # after it was loaded becomes one (in memory only)
        self._sample_at, self._sample_key, self._prune_at = -np.inf, None, -np.inf
        self._side_errors: set[str] = set()
        self._auto_cache: tuple | None = None  # (key, (version, slots)) of the last auto grid built
        self._surface_cache: tuple | None = None  # (key, polygon) of the last auto surface

    def refresh(self, now: float) -> ActiveGeometry | None:
        """The active calibration (looked up at most every `poll` seconds), or None: uncalibrated, or the store down for
        longer than STALE_AFTER. Never raises."""
        if now - self._checked < self.poll:
            return self.active
        self._checked = now
        try:
            store = self.open_store()
            ver = store.active_version()
            if ver is None:
                self.active = None
            elif self.active is None or self.active.version != ver:
                row = store.get(ver)
                if row is not None and row["state"] == "active":  # (an activation raced us: the next poll sees it)
                    self.active = ActiveGeometry(row)
                    self.drift, self._strikes, self._drift_at = None, 0, -np.inf  # a new calibration: a fresh check
                    self.log(f"geometry {self.camera_id}: calibration version {ver} is active "
                             f"({self.active.summary['confidence']}, h {self.active.summary['h']} m)")
            self._good_at, self._last_error = now, ""
        except Exception as e:  # noqa: BLE001 - a store problem must never reach the worker's loop
            self.errors += 1
            msg = f"{type(e).__name__}: {e}"[:200]
            if msg != self._last_error:  # once per distinct problem, not every poll
                self.log(f"geometry {self.camera_id}: reading the active calibration failed ({msg}); "
                         "the last good one is kept for a minute, then the camera counts as uncalibrated")
                self._last_error = msg
            if self._good_at is None or now - self._good_at > STALE_AFTER:
                self.active = None
        return self.active

    # -- the drift guard ------------------------------------------------------------------------------------------------
    def _once(self, what: str, e: Exception):
        """A side job (drift, history) failed: say so once per distinct message, never raise."""
        msg = f"{what}: {type(e).__name__}: {e}"[:200]
        if msg not in self._side_errors and len(self._side_errors) < 20:
            self._side_errors.add(msg)
            self.log(f"geometry {self.camera_id}: {msg} (this check is skipped; frames and metrics go on)")

    def observe(self, frame, detections: list[dict], now: float | None = None):
        """Called with every processed frame (cheap unless a check is due): the drift guard. Compares the frame's static
        background with the active calibration's reference every DRIFT_EVERY seconds; DRIFT_CONFIRM "drift" results in a row
        mark the version stale (the camera then counts as uncalibrated). Never raises."""
        now = self.clock() if now is None else now
        try:
            g = self.refresh(now)
            if g is None or frame is None or now - self._drift_at < self.drift_every:
                return
            self._drift_at = now
            h, w = frame.shape[:2]
            boxes = [tuple(np.asarray(d["box"], float) * (w, h, w, h)) for d in detections if d.get("box")]  # cars, people: not background
            ref, source = g.reference, "calibration"
            if ref is None:  # made before the guard existed: the first frame seen after loading is the baseline (weaker: a camera
                source = "worker"  # that moved before this worker started is taken as it is)
                if self._baseline_for != g.version:
                    base = drift.extract_reference(frame, boxes)
                    if base["n"] < drift.MIN_REFERENCE:
                        self.drift = {"status": "inconclusive", "why": "too few features in the frame", "at": now, "version": g.version, "baseline": source}
                        return
                    self._baseline, self._baseline_for = base, g.version
                    self.log(f"geometry {self.camera_id}: version {g.version} has no stored reference; this frame is the drift baseline")
                    self.drift = {"status": "ok", "shift_px": 0.0, "at": now, "version": g.version, "baseline": source}
                    return
                ref = self._baseline
            res = drift.compare(ref, frame, boxes)
            self.drift = {"status": res["status"], "shift_px": res["shift_px"], "at": now, "version": g.version, "baseline": source,
                          **({"why": res["why"]} if res.get("why") else {})}
            if res["status"] == "drift":
                if now - self._strike_at > STRIKE_TTL * self.drift_every:
                    self._strikes = 0
                self._strikes, self._strike_at = self._strikes + 1, now
                if self._strikes == 1:
                    self._event("drift_suspected", g.version, {"shift_px": res["shift_px"], "shift_frac": res["shift_frac"]}, now)
                if self._strikes >= self.drift_confirm:
                    self._declare_drift(g, res, now)
            elif res["status"] == "ok" and self._strikes:
                self._strikes = 0
                self._event("drift_cleared", g.version, {"shift_px": res["shift_px"]}, now)
        except Exception as e:  # noqa: BLE001 - an optional guard must never reach the worker's loop
            self._once("drift check failed", e)

    def _event(self, kind: str, ver: int | None, detail: dict, now: float):
        try:
            self.open_store().log_event(kind, ver, detail, now)
        except Exception as e:  # noqa: BLE001
            self._once("logging an event failed", e)

    def _declare_drift(self, g: ActiveGeometry, res: dict, now: float):
        """The camera moved: stale the version we checked (only if it is still the active one), fall back at once."""
        reason = (f"drift: the background moved {res['shift_px']:.0f} px ({res['shift_frac'] * 100:.1f} % of the width), "
                  f"confirmed on {self._strikes} checks in a row")
        try:
            done = self.open_store().mark_stale(reason, now, ver=g.version)
        except Exception as e:  # noqa: BLE001 - the store is down: try again on the next check
            self._once("marking the calibration stale failed", e)
            return
        self.log(f"geometry {self.camera_id}: version {g.version} marked STALE ({reason}); metrics stop until a new calibration is accepted")
        self.active, self._good_at, self._strikes = None, None, 0
        self.drift = {**(self.drift or {}), "status": "stale"} if done else self.drift

    # -- history --------------------------------------------------------------------------------------------------------
    def _sample(self, g: ActiveGeometry, counts: dict, now: float):
        """Store an occupancy sample: every SAMPLE_EVERY seconds, and when the occupancy changes (SAMPLE_MIN_GAP apart)."""
        key = (g.version, counts["capacity"], counts["occupied"])
        due = now - self._sample_at >= self.sample_every or (key != self._sample_key and now - self._sample_at >= SAMPLE_MIN_GAP)
        if not due:
            return
        self._sample_at, self._sample_key = now, key
        try:
            store = self.open_store()
            store.add_sample(g.version, counts, now)
            if now - self._prune_at >= PRUNE_EVERY:
                self._prune_at = now
                store.prune_history(now)
        except Exception as e:  # noqa: BLE001
            self._once("storing a history sample failed", e)

    def auto_slots(self, shape, now: float | None = None, expected: int | None = None,
                   pins=()) -> tuple[int, list[dict]] | None:
        """The slot grid for Dynamic mode: the packed bays of the ACTIVE calibration as fixed-slot dicts (image fractions),
        or None unless that calibration is HIGH confidence (a typed reference distance: the metres can be trusted), fits
        a frame of this shape and has bays. -> (calibration version, [{"id", "name", "poly", "lift", "source"}]), source =
        measured / extrapolated / manual. Never raises.

        expected: how many bays the lot has; the lane is continued past its last measured tile up to that count (see
        ActiveGeometry.tiles). An extrapolated bay that falls out of the frame is dropped, and so is everything beyond it.
        pins: the operator's bays (fixed-slot dicts, image fractions): each replaces the grid bays it overlaps (PIN_IOU) and
        takes the id of the first of them, one that overlaps none is added last. Ids of the other bays never move."""
        try:
            g = self.refresh(self.clock() if now is None else now)
            if g is None or g.summary["confidence"] != "HIGH" or g.camera(int(shape[0]), int(shape[1])) is None:
                return None
            w, h = int(shape[0]), int(shape[1])
            key = (g.version, w, h, expected, json.dumps(pins, sort_keys=True))
            if self._auto_cache is None or self._auto_cache[0] != key:  # the frame loop asks on every frame
                self._auto_cache = (key, (g.version, merge_pins(self._grid_slots(g, w, h, expected), pins)))
            slots = self._auto_cache[1][1]
            return (g.version, slots) if slots else None
        except Exception as e:  # noqa: BLE001 - optional: Dynamic mode simply carries on as it always did
            self._once("building the auto slot grid failed", e)
            return None

    def auto_surface(self, shape, now: float | None = None) -> list[list[float]] | None:
        """The measured ground as a polygon in image fractions (what the vehicle-driven bays of the auto grid must stand on), or
        None: not HIGH, another frame shape, no surface, or not all in front of the camera. Never raises."""
        try:
            g = self.refresh(self.clock() if now is None else now)
            w, h = int(shape[0]), int(shape[1])
            if g is None or g.summary["confidence"] != "HIGH" or g.camera(w, h) is None:
                return None
            key = (g.version, w, h)
            if self._surface_cache is None or self._surface_cache[0] != key:
                outline, poly = g.surface_outline(), None
                if outline is not None:
                    px = to_image(g.camera(w, h), outline) / (w, h)
                    poly = np.clip(px, 0.0, 1.0).round(4).tolist() if np.isfinite(px).all() else None
                self._surface_cache = (key, poly)
            return self._surface_cache[1]
        except Exception as e:  # noqa: BLE001
            self._once("building the auto surface failed", e)
            return None

    @staticmethod
    def _grid_slots(g: ActiveGeometry, w: int, h: int, expected: int | None) -> list[dict]:
        if g.frozen is not None:  # an accepted suggestion: the bays are fixed (the expected count was applied when it was made)
            return [dict(s) for s in g.frozen]
        return GeometryFeed.slots_from_tiles(g, g.tiles(expected), w, h)

    @staticmethod
    def slots_from_tiles(g: ActiveGeometry, tiles: list[tuple[np.ndarray, str]], w: int, h: int) -> list[dict]:
        """Tiles (metres, front to back) -> fixed-slot dicts in image fractions of a w x h frame, ids 1..n in that order."""
        cam, slots, beyond = g.camera(w, h), [], False
        frame = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        for tile, source in tiles:
            px = to_image(cam, tile) / (w, h)
            poly = np.clip(px, 0.0, 1.0)
            if not np.isfinite(px).all():
                beyond = beyond or source == EXTRAPOLATED
                continue
            if source == EXTRAPOLATED:  # it is a guess: only while most of it is still in the picture
                whole = abs(cv2.contourArea(px.astype(np.float32)))
                inside = cv2.intersectConvexConvex(px.astype(np.float32), frame)[0] if whole > 0 else 0.0
                beyond = beyond or inside < EDGE_KEEP * whole
                if beyond:
                    continue
            if cv2.contourArea(poly.astype(np.float32)) >= 1e-4:  # (validate_slots' own limits)
                mid = tile.mean(axis=0)  # a vehicle's centre stands VEHICLE_HALF farther than its near wheels (the box bottom)
                far = to_image(cam, np.array([mid, mid + (0.0, VEHICLE_HALF)])) / (w, h)
                lift = float(far[0][1] - far[1][1]) if np.isfinite(far).all() else 0.0
                i = len(slots) + 1
                slots.append({"id": i, "name": f"B{i}", "poly": poly.round(4).tolist(), "lift": max(lift, 0.0), "source": source})
        return slots

    def camera(self, shape, now: float | None = None) -> CameraModel | None:
        """The calibrated camera for a frame of `shape` ([width, height]), or None: no active calibration, another picture shape, or the
        calibration marked stale. For the spillage ground metric; never raises."""
        try:
            g = self.refresh(self.clock() if now is None else now)
            return None if g is None else g.camera(int(shape[0]), int(shape[1]))
        except Exception:  # noqa: BLE001
            return None

    def metrics(self, detections: list[dict], shape, now: float | None = None) -> dict | None:
        """The metric view of one frame, or None while the camera has no active calibration (the worker then publishes
        nothing extra). detections: the snapshot's list ({"label", "kind" vehicle / other, "box" fractions}); shape:
        [width, height] of the frame. -> {"version", "confidence", "uncertainty", "h", "area_m2", "valid", "counts",
        "vehicles": [{"i", "poly" (image fractions), "centroid_m"}], "distances": [{"kind", "m", "p", "q" (image
        fractions), "label"?}], "bays": [{"id", "poly", "state" FREE / BUSY, "car"}]}."""
        g = self.refresh(self.clock() if now is None else now)
        if g is None:
            return None
        w, h = int(shape[0]), int(shape[1])
        cam = g.camera(w, h)
        if cam is None:  # a frame of another shape than the calibrated picture: say so, draw nothing
            return {**g.summary, "valid": False, "reason": "the frame is not the picture this calibration was made on"}
        px = np.array([w, h], float)

        cars, beyond, n_vehicles = [], 0, 0
        for i, d in enumerate(detections):
            if d.get("kind", "vehicle") != "vehicle":
                continue
            n_vehicles += 1
            x1, y1, x2, y2 = (np.asarray(d["box"], float) * (w, h, w, h)).tolist()
            whole = x1 > 0.01 * w and x2 < 0.99 * w and y2 < 0.99 * h and y1 > 0.01 * h
            fp = None
            if whole and ground_resolution(cam, [((x1 + x2) / 2, y2)])[0] <= CAR_MAX_RES:  # position meaningful here
                fp = car_footprint(cam, (x1, y1, x2, y2))
            img = None if fp is None else to_image(cam, fp[0]) / px
            if fp is None or not np.isfinite(img).all():
                beyond += 1
                continue
            centre = fp[0].mean(axis=0)
            if any(np.linalg.norm(centre - c["centroid"]) < SAME_CAR for c in cars):
                n_vehicles -= 1  # the same car detected twice
                continue
            cars.append({"i": i, "m": fp[0], "centroid": centre, "img": img})

        distances = []
        # each car's nearest neighbour (one line per pair, no web of lines)
        pairs = nearest_pairs([c["m"] for c in cars], MAX_DIST)
        keep = {}
        for a, b, dist in pairs:
            for k in (a, b):
                if k not in keep or dist < keep[k][2]:
                    keep[k] = (a, b, dist)
        for a, b, _ in sorted(set(keep.values())):
            d, pa, pb = closest_points(cars[a]["m"], cars[b]["m"])
            ends = to_image(cam, np.array([pa, pb])) / px
            if np.isfinite(ends).all():
                distances.append({"kind": "vehicle", "m": round(d, 2), "p": ends[0].round(4).tolist(), "q": ends[1].round(4).tolist()})
        # other objects (people, animals) standing near a parked car
        for d in detections:
            if d.get("kind") != "other" or not cars:
                continue
            x1, y1, x2, y2 = (np.asarray(d["box"], float) * (w, h, w, h)).tolist()
            if ground_resolution(cam, [((x1 + x2) / 2, y2)])[0] > CAR_MAX_RES:
                continue
            foot = to_ground(cam, [((x1 + x2) / 2, y2)])[0]
            if not np.isfinite(foot).all():
                continue
            near = min(((*closest_on_polygon(foot, c["m"]), c) for c in cars), key=lambda t: t[0])
            if 0 < near[0] <= MAX_DIST:
                ends = to_image(cam, np.array([foot, near[1]])) / px
                if np.isfinite(ends).all():
                    distances.append({"kind": "object", "label": d.get("label", "object"), "m": round(near[0], 2),
                                      "p": ends[0].round(4).tolist(), "q": ends[1].round(4).tolist()})

        bays = []
        for b in g.bays:
            poly = np.float32(b["poly_m"]).reshape(-1, 1, 2)
            busy = any(cv2.pointPolygonTest(poly, (float(c["centroid"][0]), float(c["centroid"][1])), False) >= 0 for c in cars)
            bays.append({"id": b["id"], "poly": b["poly"], "state": "BUSY" if busy else "FREE", "car": bool(b.get("car"))})
        occupied = sum(b["state"] == "BUSY" for b in bays)
        counts = {"capacity": len(bays), "occupied": occupied, "free": len(bays) - occupied, "vehicles": n_vehicles,
                  "measured": len(cars), "beyond_range": beyond}
        self._sample(g, counts, self.clock() if now is None else now)
        return {**g.summary, "valid": True, "drift": self.drift, "drift_guard": g.reference is not None,
                "counts": counts,
                "vehicles": [{"i": c["i"], "poly": c["img"].round(4).tolist(), "centroid_m": c["centroid"].round(2).tolist()} for c in cars],
                "distances": distances, "bays": bays}


def feed_from_env(env, store, log=print) -> GeometryFeed | None:
    """The feed for this worker (its camera is CARPARK_CAMERA_ID, set by the supervisor), or None when the flag is off or
    the worker does not know its camera. store: the worker's LiveStore (the SQLite backend lives in its database;
    CARPARK_DATABASE_URL selects PostgreSQL, as for the setup endpoints)."""
    if not geometry_enabled(env):
        return None
    camera = (env.get("CARPARK_CAMERA_ID") or "").strip()
    if not camera:
        log("geometry: CARPARK_GEOMETRY_ENABLED is on but this worker has no CARPARK_CAMERA_ID: no metrics for it")
        return None
    num = lambda name, default: float(env.get(name) or default)  # noqa: E731
    try:
        every, confirm = num("CARPARK_DRIFT_EVERY", DRIFT_EVERY), int(num("CARPARK_DRIFT_CONFIRM", DRIFT_CONFIRM))
    except ValueError:
        every, confirm = DRIFT_EVERY, DRIFT_CONFIRM
    log(f"geometry: reading the active calibration of camera {camera!r} every {POLL:.0f} s; drift check every {every:.0f} s "
        f"(stale after {confirm} in a row), occupancy history every {SAMPLE_EVERY:.0f} s")
    return GeometryFeed(lambda: open_geometry(store, camera, env), camera, log=log, drift_every=every, drift_confirm=confirm)
