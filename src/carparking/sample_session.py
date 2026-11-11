"""A sample picture as its own throw-away camera: calibrate it from a few clicks, lay a bay grid on it, hand the grid to sample_test.run_sample.

The picture of a random lot has no configured slots, so it gets a calibration of its own, built the way camera-1's was:
  * rectangle method  four clicked corners of a ground rectangle of known size (a bay, a painted box): geometry.pins.camera_from_rectangle gives the focal
                      length, tilt and height exactly, no model needed;
  * GeoCalib method   the setup service's GeoCalib (and its depth) for the lens and tilt, the scale from a typed distance between two clicked ground points
                      (that is what makes the calibration HIGH);
then the same geometry.pipeline.build_record packs the bays on the parking surface (a clicked polygon, or SAM's), anchored on the cars found in the picture.
Everything here is pure functions on a frame and a few settings: nothing is stored, nothing is read from or written to a live camera. The API keeps the
result in memory only (api/sample_sessions.py).
"""
from __future__ import annotations

import cv2
import numpy as np

from carparking.geometry.camera import to_image
from carparking.geometry.capacity import AISLE, BAY
from carparking.geometry.pins import PinsError, camera_from_rectangle
from carparking.geometry.pipeline import SetupError, build_record, usable_vehicle_boxes

MAX_WIDTH = 2560  # a bigger picture is scaled down on upload: every size-based threshold was tuned for camera-1's 2560 px
MIN_POLYGON_POINTS = 3


class SessionError(ValueError):
    """The picture or the clicks cannot make a calibration: the message says what to change."""


def load_frame(raw: bytes) -> np.ndarray:
    """JPEG / PNG bytes -> BGR frame, upright (OpenCV applies the EXIF orientation), at most MAX_WIDTH wide."""
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise SessionError("the picture could not be decoded (a JPEG or PNG)")
    h, w = frame.shape[:2]
    if min(h, w) < 240:
        raise SessionError(f"the picture is too small ({w}x{h}): at least 240 px on the short side")
    if w > MAX_WIDTH:
        frame = cv2.resize(frame, (MAX_WIDTH, round(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return frame


def _fractions(points, name: str, minimum: int = 1) -> np.ndarray:
    """Clicked points (fractions of the picture) -> an Nx2 array, or SessionError."""
    try:
        p = np.asarray(points, float)
    except (TypeError, ValueError):
        raise SessionError(f"{name}: at least {minimum} point(s) as [x, y] between 0 and 1") from None
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < minimum or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise SessionError(f"{name}: at least {minimum} point(s) as [x, y] between 0 and 1")
    return p


def vehicle_boxes(raw: list[dict], size: tuple[int, int]) -> tuple[list[tuple], list[tuple]]:
    """The detector's raw detections -> (the boxes good for a scale estimate, every vehicle box): pixels. Vehicles are the detections with a state."""
    veh = [{"box": d["box"], "conf": d.get("conf", 1.0)} for d in raw if d.get("state") is not None]
    return usable_vehicle_boxes(veh, size[0], size[1]), [tuple(d["box"]) for d in veh]


def rectangle_record(frame: np.ndarray, raw: list[dict], *, pins, sides, polygon, bay=BAY, aisle: float = AISLE, meta: dict | None = None,
                     row_outline=None) -> dict:
    """The calibration record from four ground pins. pins: 4 [x, y] fractions in click order around a ground rectangle; sides: (metres from pin 1 to pin 2,
    metres from pin 2 to pin 3); polygon: the parking surface, fractions. row_outline: a zone's outline (its bays are laid from it, so no minimum lot
    area is asked for). -> build_record's record plus {"pins_fit": {...}}."""
    h, w = frame.shape[:2]
    p = _fractions(pins, "pins", 4)
    if len(p) != 4:
        raise SessionError("pins: exactly four corners")
    poly = _fractions(polygon, "the parking surface", MIN_POLYGON_POINTS)
    try:
        pose = camera_from_rectangle(p * (w, h), float(sides[0]), float(sides[1]), (w, h))
        usable, every = vehicle_boxes(raw, (w, h))
        rec = build_record(width=w, height=h, polygon=poly.tolist(), f=pose["f"], up=pose["up"], vehicles=usable, obstacles=every,
                           reference={"a": p[0].tolist(), "b": p[1].tolist(), "metres": float(sides[0])}, bay=tuple(bay), aisle=aisle, meta=meta or {},
                           row_outline=row_outline, min_surface_m2=0.0 if row_outline is not None else 20.0)
    except (SetupError, PinsError) as e:
        raise SessionError(str(e)) from None
    rec["pins_fit"] = {k: pose[k] for k in ("reprojection_px", "hfov_deg", "h")}
    if pose["reprojection_px"] > 4.0:
        rec["warnings"].append(f"the four pins do not quite make a rectangle seen by one camera ({pose['reprojection_px']:.1f} px off): re-click the corners "
                               "closer to the true ones, or use a bigger rectangle")
    return rec


def geocalib_record(frame: np.ndarray, raw: list[dict], *, geo: dict, depth: dict | None, reference: dict, polygon, bay=BAY, aisle: float = AISLE,
                    meta: dict | None = None, row_outline=None) -> dict:
    """The calibration record from the setup service's GeoCalib answer `geo` ({"f", "up", "k1", "k2", "model"}) and its depth estimate, with the scale from
    `reference` ({"a": [x, y], "b": [x, y], "metres": m}: two clicked ground points a known distance apart)."""
    h, w = frame.shape[:2]
    poly = _fractions(polygon, "the parking surface", MIN_POLYGON_POINTS)
    _fractions([reference.get("a"), reference.get("b")], "reference", 2)
    if not float(reference.get("metres", 0)) > 0:
        raise SessionError("reference: the distance between the two points must be positive (metres)")
    try:
        usable, every = vehicle_boxes(raw, (w, h))
        return build_record(width=w, height=h, polygon=poly.tolist(), f=geo["f"], up=geo["up"], depth=depth, vehicles=usable, obstacles=every,
                            reference=reference, bay=tuple(bay), aisle=aisle, k1=geo.get("k1", 0.0), k2=geo.get("k2", 0.0),
                            lens=geo.get("model", "pinhole"), meta=meta or {}, row_outline=row_outline,
                            min_surface_m2=0.0 if row_outline is not None else 20.0)
    except SetupError as e:
        raise SessionError(str(e)) from None


def topdown_record(frame: np.ndarray, raw: list[dict], *, reference: dict, polygon, bay=BAY, aisle: float = AISLE, meta: dict | None = None,
                   row_outline=None) -> dict:
    """The calibration record of a picture taken looking straight down (aerial / drone): no GeoCalib, so no estimated tilt. The camera is modelled as a
    nadir one (pitch 90 deg, no roll), whose ground scale is the same everywhere: metres per pixel = height / focal length: the focal length is
    set from the typed `reference` ({"a", "b", "metres"}: two ground points a known distance apart) for a nominal camera height, so the scale is exactly the reference's."""
    h, w = frame.shape[:2]
    poly = _fractions(polygon, "the parking surface", MIN_POLYGON_POINTS)
    _fractions([reference.get("a"), reference.get("b")], "reference", 2)
    if not float(reference.get("metres", 0)) > 0:
        raise SessionError("reference: the distance between the two points must be positive (metres)")
    ppm = float(np.linalg.norm((np.asarray(reference["a"], float) - np.asarray(reference["b"], float)) * (w, h))) / float(reference["metres"])  # picture pixels per metre
    if ppm < 1e-3:
        raise SessionError("reference: the two points are the same: click two ground points a known distance apart")
    try:
        usable, every = vehicle_boxes(raw, (w, h))
        rec = build_record(width=w, height=h, polygon=poly.tolist(), f=TOPDOWN_HEIGHT * ppm, up=TOPDOWN_UP, vehicles=usable, obstacles=every,
                           reference=reference, bay=tuple(bay), aisle=aisle, meta=meta or {}, row_outline=row_outline,
                           min_surface_m2=0.0 if row_outline is not None else 20.0)
    except SetupError as e:
        raise SessionError(str(e)) from None
    rec["warnings"].append("measured as a top-down view: the ground scale is taken as the same everywhere; use this only for pictures looking straight down")
    return rec


def grid_slots(record: dict, w: int, h: int) -> list[dict]:
    """The bays packed on the surface as fixed-slot dicts (image fractions, ids 1..n front to back, 'B<n>'): the measured tiles only, no guessed
    extension of the lane."""
    from carparking.geometry_live import MEASURED, ActiveGeometry, GeometryFeed
    g = ActiveGeometry({"ver": 0, "record": record})
    if g.camera(w, h) is None:
        raise SessionError("the calibration does not fit the picture's shape")
    return GeometryFeed.slots_from_tiles(g, [(t, MEASURED) for t in g.measured()], w, h)


def ground_camera(record: dict, w: int, h: int):
    """The calibrated CameraModel for spillage's ground metric, or None."""
    from carparking.geometry_live import ActiveGeometry
    return ActiveGeometry({"ver": 0, "record": record}).camera(w, h)


def session_run(profile_run: dict, polygon=None) -> dict:
    """A detection / monitor run for the picture: the profile camera's detector, confidence, classes and timings, but no parking-area cut at all (neither that camera's
    margins nor the surface outline: a fixed map keeps a box only if its CENTRE is in the mask, and on an oblique view a car's centre is on its body, above the ground the
    outline covers) and no fixed map (sample_test.run_sample gets the slots). What counts is decided by the slots and, with `auto_surface`, by the surface the vehicle stands on."""
    m5 = dict(profile_run["m5"])
    m5.update(top=0.0, bottom=0.0, left=0.0, right=0.0, roi=None)
    return {**profile_run, "fixed_mode": False, "fixed": None, "auto": None, "m5": m5}


TOPDOWN_HEIGHT = 20.0  # the nominal camera height (m) of the top-down camera: its focal length is set from the reference so that height is exact (the scale is what matters)
TOPDOWN_UP = (0.0, -0.0087, -1.0)  # straight down, a hair off so the pitch is 89.5 deg (a pitch of exactly 90 has no horizon direction to build a roll from)
MAX_ZONE_BAYS = 99  # slot ids of zone z are z * 100 + n
ZONE_OVERLAP = 0.3  # a bay of a later zone this much over a bay of an earlier zone (IoU) is the same bay: kept once


def zone_letter(zone: int) -> str:
    return chr(64 + ((zone - 1) % 26) + 1)


def _row_axes(outline, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """The zone outline's longer side (the row's direction) and the side across it, unit vectors in pixels; the row runs left to right
    (or top to bottom when it stands upright)."""
    (_, _), (rw, rh), ang = cv2.minAreaRect(np.float32(np.asarray(outline, float) * (w, h)))
    t = np.radians(ang if rw >= rh else ang + 90)
    u = np.array([np.cos(t), np.sin(t)])
    if u[0] < -1e-9 or (abs(u[0]) <= 1e-9 and u[1] < 0):
        u = -u
    return u, np.array([-u[1], u[0]])


def order_along_row(slots: list[dict], outline, w: int, h: int) -> list[dict]:
    """Bays in reading order of the zone: rows across the outline first (front to back), then along the row, so A1, A2, A3 ... follow each other."""
    if not slots:
        return []
    u, v = _row_axes(outline, w, h)
    cs = [np.asarray(x["poly"], float).mean(axis=0) * (w, h) for x in slots]
    depth = [float(np.ptp(np.asarray(x["poly"], float) * (w, h) @ v)) for x in slots]
    band = 0.6 * float(np.median(depth)) if depth else 1.0
    order = sorted(range(len(slots)), key=lambda i: float(cs[i] @ v))
    rows, last = [], None
    for i in order:  # a new row across the outline where the bays' centres jump by more than ~half a bay
        if last is None or float(cs[i] @ v) - last > band:
            rows.append([])
        rows[-1].append(i)
        last = float(cs[i] @ v)
    return [slots[i] for r in rows for i in sorted(r, key=lambda i: float(cs[i] @ u))]


def is_pin(slot: dict) -> bool:
    """A capacity pin: a bay the operator says holds more than one vehicle (fixed_slots.validate_slots keeps `capacity` only above 1). It is placed by hand, not laid by the grid."""
    try:
        return int(slot.get("capacity") or 1) > 1
    except (TypeError, ValueError):
        return False


def split_pins(slots: list[dict]) -> tuple[list[dict], list[dict]]:
    """A zone's bays -> (the laid bays, its capacity pins): the grid steps (extend, re-fit, re-lay) work on the laid bays only, so a hand-placed pin neither bends the row nor is renamed A5."""
    return [s for s in slots if not is_pin(s)], [s for s in slots if is_pin(s)]


def with_pins(bays: list[dict], pins: list[dict], zone: int) -> list[dict]:
    """The zone's laid bays plus its pins again, in id order. A pin keeps its id and name unless a new bay took that id (or it is not one of this zone's ids): it then gets the next free one."""
    used = {b["id"] for b in bays}
    out = list(bays)
    for p in pins:
        pid = p["id"] if p["id"] not in used and p["id"] // 100 == zone else next((zone * 100 + n for n in range(1, MAX_ZONE_BAYS + 1) if zone * 100 + n not in used), None)
        if pid is None:
            break
        used.add(pid)
        out.append({**p, "id": pid})
    return sorted(out, key=lambda b: b["id"])


def zone_bays(slots: list[dict], zone: int, outline=None, w: int = 1, h: int = 1) -> tuple[list[dict], int]:
    """A zone's own calibrated grid -> its bays in row order (see order_along_row, when the outline is given), renumbered zone * 100 + n and named
    after the zone (A1, A2 ...). -> (bays, how many were left out above MAX_ZONE_BAYS)."""
    letter = zone_letter(zone)
    ordered = order_along_row(slots, outline, w, h) if outline is not None else list(slots)
    out = [{**s, "id": zone * 100 + n, "name": f"{letter}{n}"} for n, s in enumerate(ordered[:MAX_ZONE_BAYS], 1)]
    return out, max(0, len(slots) - MAX_ZONE_BAYS)


ROW_STRETCH = 0.12  # the bays may be widened / narrowed this much to fill the zone's length exactly; beyond it they keep the bay width
def _outline_sides(outline_px) -> tuple[np.ndarray, np.ndarray]:
    """A zone outline (pixels) -> its front edge (F0, F1: the long side nearer the camera, i.e. lower in the picture) and its back edge (B0, B1),
    both running the same way along the row (F0 and B0 at the same end). An outline of more than 4 corners is simplified to its 4 main ones."""
    p = np.asarray(outline_px, float)
    if len(p) != 4:
        hull = cv2.convexHull(np.float32(p)).reshape(-1, 2)
        for eps in (0.01, 0.02, 0.04, 0.08, 0.15):
            a = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True).reshape(-1, 2)
            if len(a) == 4:
                p = a.astype(float)
                break
        else:
            p = cv2.boxPoints(cv2.minAreaRect(np.float32(p))).astype(float)
    if _crossed(p):  # (clicked in a bow-tie order: the corners in order round the outline)
        p = cv2.convexHull(np.float32(p)).reshape(-1, 2).astype(float)
        if len(p) != 4:
            p = cv2.boxPoints(cv2.minAreaRect(np.float32(p))).astype(float)
    L = [float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)]
    i, j = (0, 2) if L[0] + L[2] >= L[1] + L[3] else (1, 3)
    e1, e2 = (p[i], p[(i + 1) % 4]), (p[j], p[(j + 1) % 4])
    k = i if (e1[0][1] + e1[1][1]) >= (e2[0][1] + e2[1][1]) else j  # the front edge starts at corner k; its ends' neighbours across the outline are the back's
    front, back = (p[k], p[(k + 1) % 4]), (p[(k + 3) % 4], p[(k + 2) % 4])  # (back[0] next to front[0]: never by distance, which crosses on a thin slanted outline)
    if front[0][0] > front[1][0]:  # left to right
        front, back = (front[1], front[0]), (back[1], back[0])
    return np.array([front[0], front[1]]), np.array([back[0], back[1]])


def _crossed(p) -> bool:
    """Do two opposite sides of the 4-corner outline p cross (a bow-tie)?"""
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def meets(a, b, c, d):
        return cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0
    return bool(meets(p[0], p[1], p[2], p[3]) or meets(p[1], p[2], p[3], p[0]))


def _meet(p, d, q, e) -> np.ndarray | None:
    """Where the line p + s d meets the line q + r e (pixels), or None when they are parallel."""
    m = np.array([[d[0], -e[0]], [d[1], -e[1]]], float)
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    s, _ = np.linalg.solve(m, np.asarray(q, float) - np.asarray(p, float))
    return np.asarray(p, float) + s * np.asarray(d, float)


DUPLICATE_GAP = 0.45  # two cars closer than this share of the row's typical spacing (or of a bay of the zone frame) are one car seen twice
EMPTY_GAP = 1.8  # without a zone frame: a gap this many typical spacings wide holds empty bays between the cars
MULTIPLE = 2.5  # with a zone frame, when some of the zone's cars stand next to each other (a gap under ADJACENT frame bays): another gap holds
#                 empty bays only from this many frame bays. The frame was measured 1.3-1.4x long on the sample pictures, locally more where an
#                 outline corner was cut by the picture's edge; at 2.0 one car of image2.png was split, at 2.5 none (7 bays for 7 cars at both resolutions)
ADJACENT = 1.5  # a gap under this many frame bays is two cars side by side; with no such gap the cars are sparse and every gap is cut into
#                 round(gap / frame bay) bays (a frame that is right then keeps one car in every other bay from making double-wide bays)
MIN_ROW_CARS = 2  # cars needed to lay a zone from its cars (fewer: the metric grid)


class ZoneFrame:
    """A zone's own metric frame, from its outline as a rectangle on the ground, the camera's focal length and the typed reference distance.

    A rectangle's picture is the homography H of the unit square (u along the row, v across it). With the intrinsics K (GeoCalib's f, at the
    picture's own size: f / width measured constant across 640-2508 px), K^-1 H = [s L r1, s D r2, t] for the rectangle's length L and depth D
    and orthonormal r1, r2, so L / D = |K^-1 h1| / |K^-1 h2|: the zone's proportions need no camera tilt, roll or height (the quantities GeoCalib
    got wrong on the sample pictures). The typed reference (two ground points `metres` apart) mapped through H^-1 then gives D, hence L, in
    metres. Every quantity is a ratio of pixels to pixels or a typed length, so the picture's resolution cancels.
    `squareness_deg`: how far the outline's corners are from right angles as seen by this camera (a check of the drawing, not used)."""

    def __init__(self, outline, w: int, h: int, cam, reference: dict):
        out_px = np.asarray(outline, float) * (w, h)
        (self.f0, self.f1), (self.b0, self.b1) = _outline_sides(out_px)
        self.w, self.h, self.cam = w, h, cam
        ideal = cam.undistort(np.array([self.f0, self.f1, self.b1, self.b0]))
        self.H = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), np.float32(ideal))
        self.Hinv = np.linalg.inv(self.H)
        m = np.linalg.inv(cam.K()) @ self.H
        r1, r2 = m[:, 0], m[:, 1]
        self.aspect = float(np.linalg.norm(r1) / np.linalg.norm(r2))
        self.squareness_deg = float(np.degrees(np.arcsin(min(1.0, abs(r1 @ r2) / np.linalg.norm(r1) / np.linalg.norm(r2)))))
        a = np.asarray(reference["a"], float) * (w, h)
        b = np.asarray(reference["b"], float) * (w, h)
        uv = self.to_uv(cam.undistort(np.array([a, b])))
        d = uv[1] - uv[0]
        span = float(np.hypot(d[0] * self.aspect, d[1]))  # the reference in units of the zone's depth
        if not (np.isfinite(span) and span > 1e-9 and float(reference["metres"]) > 0):
            raise SessionError("the reference distance cannot be measured in this zone")
        self.depth_m = float(reference["metres"]) / span
        self.length_m = self.aspect * self.depth_m

    def to_uv(self, ideal_px) -> np.ndarray:
        q = np.c_[np.asarray(ideal_px, float).reshape(-1, 2), np.ones(len(np.asarray(ideal_px).reshape(-1, 2)))] @ self.Hinv.T
        return q[:, :2] / q[:, 2:3]

    def to_px(self, uv) -> np.ndarray:
        q = np.c_[np.asarray(uv, float).reshape(-1, 2), np.ones(len(np.asarray(uv).reshape(-1, 2)))] @ self.H.T
        return self.cam.distort(q[:, :2] / q[:, 2:3])

    def t_of_u(self, u: float) -> float:
        """A point u of the row's front edge (0..1 of the zone's length) -> its share t along the drawn front edge in the picture."""
        p = self.to_px([[u, 0.0]])[0]
        fd = self.f1 - self.f0
        return float(np.dot(p - self.f0, fd) / np.dot(fd, fd))

    def u_of_t(self, t: float) -> float:
        return float(self.to_uv(self.cam.undistort((self.f0 + t * (self.f1 - self.f0))[None]))[0, 0])

    def bay_t(self, t: float, width_m: float) -> float:
        """How long, in t along the drawn front edge, a bay `width_m` wide is at t (the zone's own perspective)."""
        u = self.u_of_t(t)
        return self.t_of_u(u + width_m / self.length_m) - t


def zone_frame(outline, w: int, h: int, cam, reference: dict | None) -> ZoneFrame | None:
    """The zone's metric frame (ZoneFrame), or None without a reference or when the outline cannot be one (degenerate, beyond the horizon)."""
    if not reference or cam is None:
        return None
    try:
        fr = ZoneFrame(outline, w, h, cam, reference)
    except (SessionError, np.linalg.LinAlgError, cv2.error, ValueError):
        return None
    ok = np.isfinite([fr.length_m, fr.depth_m]).all() and fr.length_m > 0.5 and fr.depth_m > 0.5
    return fr if ok else None


def car_position(box) -> tuple[float, float]:
    """Where a parked vehicle stands for its bay, in the picture: the middle of its box across, a quarter of its height above the box's bottom.
    The bottom centre is its nearest bumper corner on an angled view and the box centre is up on its body; this point stays on the car's own
    footprint on both, and is the same share of the box at any resolution."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2 - 0.25 * (y2 - y1)


def anchored_edges(ts, bay_t=None) -> list[float]:
    """Car positions along the row's front edge (t: 0 at the outline's first corner, 1 at its last) -> the bay edges along it. The bays start at
    the outline's first corner (the zone is drawn from the first bay's edge) and each is as wide as the cars standing there are apart: the
    spacing is read between neighbouring cars (two detections of one car are one car; a gap of several spacings holds that many empty bays) and
    follows the row, nearer or farther, by interpolation. A spacing is a DIFFERENCE of two car positions, so whatever a car's position is off by
    (its box reaches over its roof on an angled view) cancels: no scale, no tilt, no resolution enters.
    bay_t(t): the length of one bay at t from the zone's metric frame (ZoneFrame.bay_t), or None. With it the gaps between cars are read in
    frame bays: when some cars stand side by side (a gap under ADJACENT bays) another gap holds empty bays only from MULTIPLE bays (a frame a
    little off never splits a car); when none do (sparse cars, missed detections) every gap is cut into round(gap / bay) bays (a car in every
    other bay never makes one double-wide bay). Two detections closer than DUPLICATE_GAP of a bay are one car. Without it the cars' own median spacing is the yardstick (EMPTY_GAP).
    Empty with fewer than MIN_ROW_CARS cars."""
    t = np.sort(np.asarray([x for x in ts if -0.05 <= x <= 1.05], float))
    if len(t) < MIN_ROW_CARS:
        return []
    gaps = np.diff(t)
    if not (gaps > 0).any():
        return []
    typical = float(np.median(gaps[gaps > 0]))
    unit = (lambda x: bay_t(x)) if bay_t is not None else (lambda x: typical)  # noqa: E731
    cars = [float(t[0])]
    for x in t[1:]:
        if x - cars[-1] < DUPLICATE_GAP * unit((x + cars[-1]) / 2):
            cars[-1] = (cars[-1] + float(x)) / 2
        else:
            cars.append(float(x))
    if len(cars) < MIN_ROW_CARS:
        return []
    typical = float(np.median(np.diff(cars)))
    ratios = [(b - a) / max(bay_t((a + b) / 2), 1e-9) for a, b in zip(cars, cars[1:])] if bay_t is not None else []
    split_from = MULTIPLE if any(r < ADJACENT for r in ratios) else ADJACENT  # dense rows: only clear gaps hold empty bays; sparse rows: every gap
    mids, spacing = [], []
    for i, (a, b) in enumerate(zip(cars, cars[1:])):
        if bay_t is not None:
            r = ratios[i]
            n = max(1, int(round(r))) if r >= split_from else 1
        else:
            n = max(1, int(round((b - a) / typical))) if (b - a) >= EMPTY_GAP * typical else 1
        mids.append((a + b) / 2)
        spacing.append((b - a) / n)
    edges = [0.0]
    while len(edges) <= MAX_ZONE_BAYS:
        x = edges[-1]
        step = float(np.interp(x + np.interp(x, mids, spacing) / 2, mids, spacing))  # the spacing at the bay's own middle
        if step <= 0 or x + step / 2 > 1.0:  # the last bay belongs to the zone only if most of it is in it
            break
        edges.append(x + step)
    return edges


def _perspective_edges(edges: list[float], f0, f1, b0, b1) -> list[float]:
    """The bay edges the cars gave (t along the front edge) -> the same row as equal bays on the ground, seen in the zone's own perspective. The outline
    is a ground rectangle: its four corners fix the homography of the unit square (u along the row, v across it), with no camera model needed. Equal
    steps in u between the row's first and last edge are equal bays on the ground; carried back to the picture they shrink steadily towards the far end
    exactly as the zone's separating lines converge (and stay equal along a row at one depth). The cars say how many bays there are and where the
    row starts and ends; the perspective says how wide each looks, so one parked car's odd spacing never makes a far bay as wide as a near one."""
    if len(edges) < 3:
        return edges
    try:
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), np.float32([f0, f1, b1, b0]))
        Hinv = np.linalg.inv(H)
        pts = np.asarray(f0, float) + np.asarray(edges, float)[:, None] * (np.asarray(f1, float) - f0)
        q = np.c_[pts, np.ones(len(pts))] @ Hinv.T
        u = q[:, 0] / q[:, 2]
        uk = u[0] + (u[-1] - u[0]) * np.arange(len(u)) / (len(u) - 1)
        back = np.c_[uk, np.zeros(len(uk)), np.ones(len(uk))] @ H.T
        px = back[:, :2] / back[:, 2:3]
        fd = np.asarray(f1, float) - f0
        fit = (((px - f0) @ fd) / float(fd @ fd)).tolist()
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return edges
    return fit if np.isfinite(fit).all() and all(b > a for a, b in zip(fit, fit[1:])) else edges


def car_row_grid(outline, w: int, h: int, cars, rows: int = 1, cam=None, frame: ZoneFrame | None = None, width_m: float = BAY[0],
                 edges: list[float] | None = None) -> tuple[list[dict], dict] | None:
    """A zone's bays from its own parked cars, in the picture itself: no metres, so neither the typed scale nor GeoCalib's tilt and roll (nor a
    picture resized out of its aspect) can make them too narrow or too wide. The outline's front and back edges are the row's front and back;
    its two short sides meet at the vanishing point of the bays' separating lines. Every car (car_position) is carried along those lines onto
    the front edge; the bays start at the outline's first corner and are as wide as the cars there are apart (anchored_edges): one car, one
    bay. None with fewer than MIN_ROW_CARS cars in the zone (the caller lays the metric grid then).
    edges: the bays' edges given (t along the front edge, 0..1): no car decides anything, the same lattice is cut with these (a fixed bay count)."""
    from carparking.geometry.camera import to_ground
    from carparking.geometry_live import VEHICLE_HALF
    out_px = np.asarray(outline, float) * (w, h)
    (f0, f1), (b0, b1) = _outline_sides(out_px)
    fd = f1 - f0
    flen = float(np.linalg.norm(fd))
    if flen < 1:
        return None
    vp = _meet(f0, b0 - f0, f1, b1 - f1)
    side = ((b0 - f0) + (b1 - f1)) / 2  # (parallel sides: the separators keep this direction)
    ts = []
    for box in cars:
        c = np.array(car_position(box), float)
        m = _meet(c, (vp - c) if vp is not None else side, f0, fd)
        if m is not None:
            ts.append(float(np.dot(m - f0, fd) / flen ** 2))  # 0 at F0, 1 at F1
    given = edges is not None
    if not given:
        edges = anchored_edges(ts, (lambda x: frame.bay_t(x, width_m)) if frame is not None else None)
        if len(edges) < 2:
            return None
        edges = _perspective_edges(edges, f0, f1, b0, b1)

    def back_of(pf):
        q = _meet(pf, (vp - pf) if vp is not None else side, b0, b1 - b0)
        return q if q is not None else pf + side

    rows = max(1, int(rows))
    slots = []
    for r in range(rows):
        for i in range(len(edges) - 1):
            pa, pb = f0 + edges[i] * fd, f0 + edges[i + 1] * fd
            qa, qb = back_of(pa), back_of(pb)
            lo, hi = r / rows, (r + 1) / rows
            quad = np.array([pa + lo * (qa - pa), pb + lo * (qb - pb), pb + hi * (qb - pb), pa + hi * (qa - pa)])
            poly = np.clip(quad / (w, h), 0.0, 1.0)
            if cv2.contourArea(poly.astype(np.float32)) < 1e-4:
                continue
            lift = 0.0
            if cam is not None:  # a vehicle's centre stands VEHICLE_HALF beyond its near wheels (as the live grid's lift)
                mid = to_ground(cam, quad.mean(axis=0, keepdims=True))[0]
                if np.isfinite(mid).all():
                    far = to_image(cam, np.array([mid, mid + (0.0, VEHICLE_HALF)])) / (w, h)
                    lift = float(far[0][1] - far[1][1]) if np.isfinite(far).all() else 0.0
            slots.append({"id": len(slots) + 1, "name": f"B{len(slots) + 1}", "poly": poly.round(4).tolist(), "lift": round(max(lift, 0.0), 5)})
    width = None
    if frame is not None and slots:  # the bays' width in the zone's own metric frame (the median one)
        us = [frame.u_of_t(e) for e in edges]
        width = round(float(np.median(np.diff(us))) * frame.length_m, 2)
    elif cam is not None and slots:
        g = to_ground(cam, np.array([f0 + edges[0] * fd, f0 + edges[1] * fd]))
        width = round(float(np.linalg.norm(g[1] - g[0])), 2) if np.isfinite(g).all() else None
    return slots, {"width_m": width, "length_m": None, "source": ("the expected bay count" if given else f"fitted to the zone's {len(ts)} cars in the picture"),
                   "bays_along": len(edges) - 1, "rows": rows}


def frame_grid(frame: ZoneFrame, w: int, h: int, bay=BAY, rows: int = 1) -> tuple[list[dict], dict]:
    """A zone without (enough) cars: its length cut into whole bays of the bay width in its own metric frame (ZoneFrame): n = round(L / width),
    at least one, so a bay is the width within half a bay over the zone; the depth into `rows` equal rows. Equal steps in the rectified frame are
    the row's true perspective in the picture."""
    from carparking.geometry.camera import to_ground
    from carparking.geometry_live import VEHICLE_HALF
    n = min(MAX_ZONE_BAYS, max(1, int(round(frame.length_m / float(bay[0])))))
    rows = max(1, int(rows))
    slots = []
    for r in range(rows):
        for i in range(n):
            uv = np.array([[i / n, r / rows], [(i + 1) / n, r / rows], [(i + 1) / n, (r + 1) / rows], [i / n, (r + 1) / rows]])
            quad = frame.to_px(uv)
            poly = np.clip(quad / (w, h), 0.0, 1.0)
            if not np.isfinite(poly).all() or cv2.contourArea(poly.astype(np.float32)) < 1e-4:
                continue
            lift = 0.0
            mid = to_ground(frame.cam, quad.mean(axis=0, keepdims=True))[0]
            if np.isfinite(mid).all():
                far = to_image(frame.cam, np.array([mid, mid + (0.0, VEHICLE_HALF)])) / (w, h)
                lift = float(far[0][1] - far[1][1]) if np.isfinite(far).all() else 0.0
            slots.append({"id": len(slots) + 1, "name": f"B{len(slots) + 1}", "poly": poly.round(4).tolist(), "lift": round(max(lift, 0.0), 5)})
    return slots, {"width_m": round(frame.length_m / n, 2), "length_m": round(frame.depth_m / rows, 2),
                   "source": f"the zone's own frame ({frame.length_m:.1f} x {frame.depth_m:.1f} m from its outline, the focal length and the reference)",
                   "bays_along": n, "rows": rows}


MAX_ROWS = 6  # rows of bays in one zone
SLOPE_MAX_SHIFT = 0.25  # the slope-corrected outline may move a corner at most this share of the picture's diagonal, else the drawn one is kept


def ground_outline(cam, outline, w: int, h: int, depth_m: float | None = None) -> list[list[float]] | None:
    """The zone's outline as the camera sees a true ground rectangle: the drawn front edge is kept (it is on the ground, where the row stands), the
    sides are the ground perpendicular to it carried through the camera (GeoCalib's tilt, roll and lens), not the drawn sides, which a hand-drawn
    rectangle leaves at any angle. Its depth is `depth_m` (the zone frame's), else the drawn back edge's distance on the ground. The sides then
    converge on the vanishing point the camera really has, so the bays' separators follow the perspective slope. None when it cannot be built (a corner
    above the horizon, a degenerate edge, a corner moved too far): the caller keeps the drawn outline."""
    from carparking.geometry.camera import to_ground
    out_px = np.asarray(outline, float) * (w, h)
    (f0, f1), (b0, b1) = _outline_sides(out_px)
    g = to_ground(cam, np.array([f0, f1, b0, b1]))
    if not np.isfinite(g).all():
        return None
    d = g[1] - g[0]
    length = float(np.linalg.norm(d))
    if length < 0.5:
        return None
    d = d / length
    n = np.array([-d[1], d[0]])
    off = float(((g[2] + g[3]) / 2 - g[0]) @ n)
    if abs(off) < 0.1:
        return None
    depth = float(depth_m) if depth_m and np.isfinite(depth_m) and depth_m > 0.5 else abs(off)
    n = n * (1.0 if off > 0 else -1.0)
    ground = np.array([g[0], g[1], g[1] + n * depth, g[0] + n * depth])
    px = to_image(cam, ground)
    if not np.isfinite(px).all():
        return None
    old = np.array([f0, f1, b1, b0])
    if float(np.linalg.norm(px - old, axis=1).max()) > SLOPE_MAX_SHIFT * float(np.hypot(w, h)):
        return None
    if px[:, 0].min() < -0.05 * w or px[:, 0].max() > 1.05 * w or px[:, 1].min() < -0.05 * h or px[:, 1].max() > 1.05 * h:
        return None
    keep = np.array([f0, f1, px[2], px[3]]) / (w, h)  # (the drawn front edge exactly as it was; only the back corners are the camera's)
    return np.clip(keep, 0.0, 1.0).round(4).tolist()


def count_edges(frame: "ZoneFrame | None", f0, f1, b0, b1, n: int) -> list[float]:
    """n equal bays along a zone's front edge as edges t in 0..1: equal on the ground in the zone's frame, else in the outline's own perspective."""
    if frame is not None:
        return [0.0, *(frame.t_of_u(i / n) for i in range(1, n)), 1.0] if n > 1 else [0.0, 1.0]
    return _perspective_edges([i / n for i in range(n + 1)], f0, f1, b0, b1) if n > 2 else [i / n for i in range(n + 1)]


def auto_rows(frame: "ZoneFrame | None", length_m: float) -> int:
    """How many rows of bays a zone is deep: its depth on the ground (the zone frame's, from the outline, the focal length and the reference) over a bay's
    length, rounded: a zone drawn over two cars side by side (10-12 m deep) is two rows, one bay per car, not one bay across both. 1 without a frame."""
    if frame is None or not np.isfinite(frame.depth_m) or length_m <= 0:
        return 1
    return int(min(MAX_ROWS, max(1, round(frame.depth_m / length_m))))


def row_grid(record: dict, outline, w: int, h: int, bay=BAY, rows: int = 1, cars=(), reference: dict | None = None, count: int | None = None,
             slope: bool = False) -> tuple[list[dict], dict]:
    """A zone's bays laid straight along its row, on the ground, by the zone's own calibration. No car-anchored lattice: the bays follow the zone.
      * the row's direction is the outline's longer side IN THE PICTURE (a zone is drawn along its row), measured on the ground where the outline
        stands. (Its longer side on the ground is not used: on an oblique view the outline's top, drawn over the cars' bodies, lands metres behind
        them and can make the depth look longer than the row, which turned the bays across the cars.)
      * the bays are `bay` wide and long, in `rows` rows from the outline's front edge (the one nearest the camera) away from the camera;
      * along the row they fill the outline's length, widened / narrowed by at most ROW_STRETCH to share their lines exactly, else exactly the bay
        width (never squeezed to a sliver).
    With MIN_ROW_CARS or more cars in the zone the bays are laid from the cars in the picture instead (car_row_grid), their spacing checked
    against the zone's own metric frame (ZoneFrame, when a reference is given); without cars but with a reference, the frame alone lays them
    (frame_grid). The camera grid below is the last resort: it is right only as far as GeoCalib's tilt, roll and height are, and on the sample
    pictures they were wrong enough to put two bays on one car at the row's far end (image2.png: 16 bays for 7 cars).
    rows: how many rows of bays from the front edge; 0 = automatic (auto_rows: the zone's depth over the bay length, with a reference).
    count: the expected number of bays in the zone (the real lot's): fixed, whatever the cars or the zone's length say. The row is cut into
      ceil(count / rows) equal bays on the ground and the last cells are dropped to make exactly `count`.
    slope: re-draw the outline's sides along the camera's ground perpendicular (ground_outline) before laying anything: the bays then follow the
      true perspective slope, not the drawn rectangle's. The outline used is returned in the sizing as "outline" (fractions) so the page can show it.
    -> (fixed-slot dicts: fractions, ids 1..n, front row first; {"width_m", "length_m", "source", "bays_along", "rows"})."""
    from carparking.geometry.bev import densify
    from carparking.geometry.camera import to_ground
    from carparking.geometry_live import VEHICLE_HALF
    cam = ground_camera(record, w, h)
    if cam is None:
        raise SessionError("the calibration does not fit the picture's shape")
    frame = zone_frame(outline, w, h, cam, reference)
    shown = None
    if slope:
        shown = ground_outline(cam, outline, w, h, frame.depth_m if frame is not None else None)
        if shown is not None:
            outline = shown
            frame = zone_frame(outline, w, h, cam, reference)
    if not rows:
        rows = auto_rows(frame, float(bay[1]))
    if count:
        along = -(-int(count) // max(1, int(rows)))
        (f0, f1), (b0, b1) = _outline_sides(np.asarray(outline, float) * (w, h))
        made = car_row_grid(outline, w, h, [], rows, cam, frame, float(bay[0]), edges=count_edges(frame, f0, f1, b0, b1, along))
        if made is not None:
            made[0][:] = made[0][:int(count)]
            made[1]["bays_along"] = along
            made[1]["outline"] = shown
            return made
    by_cars = car_row_grid(outline, w, h, list(cars), rows, cam, frame, float(bay[0]))
    if by_cars is not None:
        by_cars[1]["outline"] = shown
        return by_cars
    if frame is not None:
        made = frame_grid(frame, w, h, bay, rows)
        made[1]["outline"] = shown
        return made
    out_px = np.asarray(outline, float) * (w, h)
    g = to_ground(cam, densify(out_px))
    g = g[np.isfinite(g).all(axis=1)]
    if len(g) < 3:
        raise SessionError("the zone's outline is not on measurable ground (above the horizon?): draw it on the parking ground")
    u_px, _ = _row_axes(outline, w, h)
    low = out_px[np.argsort(-out_px[:, 1])[: max(2, len(out_px) // 2)]].mean(axis=0)  # the outline's lower (nearer) part: on the ground for sure
    gd = to_ground(cam, np.array([low, low + 20 * u_px]))
    d = gd[1] - gd[0] if np.isfinite(gd).all() else np.array([1.0, 0.0])
    d = d / (np.linalg.norm(d) or 1.0)
    nrm = np.array([-d[1], d[0]])
    a, b = g @ d, g @ nrm
    front, back = (b.min(), b.max()) if abs(b.min()) <= abs(b.max()) else (b.max(), b.min())  # the camera stands over the ground's origin
    away = 1.0 if back >= front else -1.0
    width, length = float(bay[0]), float(bay[1])
    span = float(a.max() - a.min())
    n = max(1, int(round(span / width)))
    step = span / n
    if abs(step - width) > ROW_STRETCH * width:
        step = width
        n = max(1, int(np.floor(span / width + 0.25)))
    start = float(a.min())
    source = "bay width" if step == width else "bay width, fitted to the zone"
    slots = []
    for r in range(max(1, int(rows))):
        for i in range(n):
            u0, v0 = start + i * step, front + away * r * length
            tile = np.array([d * u0 + nrm * v0, d * (u0 + step) + nrm * v0, d * (u0 + step) + nrm * (v0 + away * length), d * u0 + nrm * (v0 + away * length)])
            px = to_image(cam, tile) / (w, h)
            if not np.isfinite(px).all():
                continue
            poly = np.clip(px, 0.0, 1.0)
            if cv2.contourArea(poly.astype(np.float32)) < 1e-4:
                continue
            mid = tile.mean(axis=0)  # a vehicle's centre stands VEHICLE_HALF beyond its near wheels (as the live grid's lift)
            far = to_image(cam, np.array([mid, mid + (0.0, VEHICLE_HALF)])) / (w, h)
            lift = float(far[0][1] - far[1][1]) if np.isfinite(far).all() else 0.0
            slots.append({"id": len(slots) + 1, "name": f"B{len(slots) + 1}", "poly": poly.round(4).tolist(), "lift": round(max(lift, 0.0), 5)})
    return slots, {"width_m": round(float(step), 2), "length_m": length, "source": source, "bays_along": n, "rows": max(1, int(rows)), "outline": shown}


def in_zone(raw: list[dict], outline, w: int, h: int, margin: float = 0.02) -> list[dict]:
    """The detections standing in the zone: a box's bottom centre (where it touches the ground) inside the outline, or within `margin` of the
    picture's width of it. Cars of other rows must not orient or anchor this zone's grid."""
    poly = np.float32(np.asarray(outline, float) * (w, h)).reshape(-1, 1, 2)
    out = []
    for d in raw:
        x1, _, x2, y2 = d["box"]
        if cv2.pointPolygonTest(poly, ((x1 + x2) / 2, y2), True) >= -margin * w:
            out.append(d)
    return out


def _extend_in_picture(ordered: list[dict], count: int, w: int, h: int) -> list[np.ndarray]:
    """Bays laid from the cars (front-left, front-right, back-right, back-left in the picture) continued in the picture: each next bay shares the
    last one's far side and is as wide as the last one, times the last two bays' width ratio (the row's own perspective, at most +-15 % a bay)."""
    last = np.asarray(ordered[-1]["poly"], float) * (w, h)
    r = 1.0
    if len(ordered) >= 2:
        prev = np.asarray(ordered[-2]["poly"], float) * (w, h)
        a, b = np.linalg.norm(last[1] - last[0]), np.linalg.norm(prev[1] - prev[0])
        r = float(np.clip(a / b, 0.85, 1.15)) if b > 0 else 1.0
    out, cur = [], last
    for k in range(1, count + 1):
        front, back = (cur[1] - cur[0]) * r, (cur[2] - cur[3]) * r
        nxt = np.array([cur[1], cur[1] + front, cur[2] + back, cur[2]])
        out.append(nxt)
        cur = nxt
    return out


def extend_zone(record: dict, slots: list[dict], outline, zone: int, count: int, w: int, h: int, in_picture: bool = False) -> tuple[list[dict], int]:
    """`count` more bays after the zone's last one, stepped on the ground by the zone's own calibration: the step is the last two bays'
    centre to centre distance (one bay: its own width along the row), so the new bays continue the row's spacing and perspective.
    in_picture: the bays were laid from the zone's cars (car_row_grid): continue them in the picture (_extend_in_picture), not in metres.
    A bay that would lie wholly outside the picture is not added. -> (the zone's bays with the new ones, how many were added)."""
    from carparking.geometry.camera import to_ground
    cam = ground_camera(record, w, h)
    if cam is None:
        raise SessionError("this zone has no usable calibration: calibrate it again")
    ordered = order_along_row(slots, outline, w, h)
    if not ordered:
        raise SessionError("this zone has no bay to continue from")
    if in_picture:
        tiles = _extend_in_picture(ordered, count, w, h)
        return _add_bays(slots, ordered, [t / (w, h) for t in tiles], outline, zone, w, h)
    ground = [to_ground(cam, np.asarray(x["poly"], float) * (w, h)) for x in ordered]
    last = ground[-1]
    if not np.isfinite(last).all():
        raise SessionError("the zone's last bay cannot be placed on the ground")
    if len(ground) >= 2 and np.isfinite(ground[-2]).all():
        step = last.mean(axis=0) - ground[-2].mean(axis=0)
    else:
        u, _ = _row_axes(outline, w, h)
        c = np.asarray(ordered[-1]["poly"], float).mean(axis=0) * (w, h)  # the row's direction on the ground, at the last bay
        g = to_ground(cam, np.array([c, c + 10 * u]))
        g_u = g[1] - g[0]
        d = g_u / (np.linalg.norm(g_u) or 1.0)
        step = d * float(np.ptp(last @ d))
    if not np.isfinite(step).all() or np.linalg.norm(step) < 0.5:
        raise SessionError("the last bays of the zone do not give a direction to continue in: remove an odd one or calibrate again")
    return _add_bays(slots, ordered, [to_image(cam, last + k * step) / (w, h) for k in range(1, count + 1)], outline, zone, w, h)


def _add_bays(slots: list[dict], ordered: list[dict], tiles, outline, zone: int, w: int, h: int) -> tuple[list[dict], int]:
    """New bays (fractions) after a zone's bays: each gets the zone's next free id; the first one wholly outside the picture ends it. The zone's
    bays are then named along the row again. -> (bays, how many were added)."""
    used = {x["id"] for x in slots}
    out, added = list(slots), 0
    for px in tiles:
        free = next((zone * 100 + n for n in range(1, MAX_ZONE_BAYS + 1) if zone * 100 + n not in used), None)
        if free is None or not np.isfinite(px).all():
            break
        poly = np.clip(px, 0.0, 1.0)
        if cv2.contourArea(poly.astype(np.float32)) < 1e-4:  # wholly outside the picture (or a sliver at its edge)
            break
        used.add(free)
        out.append({"id": free, "name": "", "poly": poly.round(4).tolist(), **({"lift": ordered[-1]["lift"]} if ordered[-1].get("lift") else {})})
        added += 1
    letter = zone_letter(zone)
    renamed = order_along_row(out, outline, w, h)
    return [{**x, "name": f"{letter}{n}"} for n, x in enumerate(renamed, 1)], added


def _iou(a, b, w: int, h: int, scale: float = 0.25) -> float:
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    ma, mb = np.zeros((sh, sw), np.uint8), np.zeros((sh, sw), np.uint8)
    cv2.fillPoly(ma, [np.round(np.asarray(a, float) * (sw, sh)).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(np.asarray(b, float) * (sw, sh)).astype(np.int32)], 1)
    inter, union = int((ma & mb).sum()), int((ma | mb).sum())
    return inter / union if union else 0.0


def merge_zones(zones: dict, w: int, h: int, skip=()) -> tuple[list[dict], list[dict]]:
    """Every calibrated zone's bays as ONE list for the occupancy run (zone order). A bay of a later zone that lies on a bay already in the list
    (IoU >= ZONE_OVERLAP: the two zone outlines overlapped) is left out, so one parked car is never two bays. `skip`: bay ids the user removed.
    -> (merged bays, [{"id", "name", "same_as"}] left out as duplicates)."""
    kept, dropped = [], []
    for zid in sorted(zones):
        for s in zones[zid]["slots"]:
            twin = next((m for m in kept if m["id"] // 100 != zid and _iou(m["poly"], s["poly"], w, h) >= ZONE_OVERLAP), None)
            if twin is not None:
                dropped.append({"id": s["id"], "name": s["name"], "same_as": twin["name"]})
            else:
                kept.append(s)
    # a removed bay still owns its place: its twin in another zone does not come back
    return [s for s in kept if s["id"] not in skip], dropped


# ---- the crop the vision model is shown ------------------------------------------------------------------------------------------------------
CROP_PAD = 0.10  # the zone's box grows by this share of its size each way: the lines at its ends are not cut off
CROP_MIN = 0.15  # no crop narrower / shorter than this share of the picture
CROP_SIDE = 1280  # the crop is scaled down so its long side is at most this many pixels


def zone_crop_box(shape) -> tuple[float, float, float, float]:
    """The zone's outline (fractions) -> its padded bounding box (x0, y0, x1, y1) in picture fractions, at least CROP_MIN wide / high; the whole picture
    when there is no outline."""
    if not shape:
        return (0.0, 0.0, 1.0, 1.0)
    xs, ys = [float(p[0]) for p in shape], [float(p[1]) for p in shape]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    px, py = max((x1 - x0) * CROP_PAD, 0.0), max((y1 - y0) * CROP_PAD, 0.0)
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    if x1 - x0 < CROP_MIN:
        c = (x0 + x1) / 2
        x0, x1 = c - CROP_MIN / 2, c + CROP_MIN / 2
    if y1 - y0 < CROP_MIN:
        c = (y0 + y1) / 2
        y0, y1 = c - CROP_MIN / 2, c + CROP_MIN / 2
    x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1))
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return (0.0, 0.0, 1.0, 1.0)
    return (x0, y0, x1, y1)


def crop_for_model(frame, box):
    """The picture cut to `box` (fractions), scaled down to CROP_SIDE on its long side (never up): what the vision model is shown."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    c = frame[int(round(y0 * h)):max(int(round(y1 * h)), int(round(y0 * h)) + 1), int(round(x0 * w)):max(int(round(x1 * w)), int(round(x0 * w)) + 1)]
    k = CROP_SIDE / max(c.shape[:2])
    return cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else c
