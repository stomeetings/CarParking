"""Fixed (pre-defined) parking slots: a hand-drawn slot map per camera, stored as JSON, and the tracker that checks
vehicles against it instead of learning slots from detections.

* FixedSlotStore: config/fixed_slots.json, keyed by camera name. Polygons are fractions (0..1) of the frame, so the
  main and the sub-stream share them. Written atomically (temp file + rename).
    {"version": 1, "cameras": {"Front Parking 02": {"updated": "2026-10-19T12:00:00+05:00",
      "slots": [{"id": 1, "name": "A1", "poly": [[0.22, 0.33], [0.56, 0.33], [0.56, 0.59], [0.22, 0.59]]}]}}}
* validate_slots(): the one check used by the editor, the store and (later) an API: ids unique positive ints,
  >= 3 points inside the frame, non-zero area. Raises ValueError with a message naming the slot.
* FixedSlotTracker: the SlotTracker state machine (hold, exit buffer, occlusion, events) on the fixed map, with
  learning off and its own matching rule (MATCH_RULES). Events and history therefore work exactly as in dynamic mode.
* draft_slots(): one rectangle per detected vehicle, as a starting point for the editor.
No Streamlit here; everything takes and returns plain lists / dicts (JSON-ready).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from carparking import blocked as blk
from carparking import evidence as evd
from carparking import obstacle_filter as obf
from carparking import spillage as spl
from carparking.dynamic_iou import overlap
from carparking.history import SlotTracker
from carparking.signature import crop_box, make_signature
from carparking.stream_health import thumbnail

# how a vehicle box is matched to a fixed slot (each vehicle fills at most one slot, each slot holds one vehicle)
MATCH_RULES = {
    "ground": "Ground point: the box's bottom centre (just above the wheels) lies in the slot, or just outside it while "
              "the box covers the slot (parked slightly forward or crooked). Best for low-angle views, where a car's "
              "box also covers the slots behind it.",
    "center": "Centre point: the box centre lies in the slot (or just outside it while the box covers the slot). Good "
              "for high, top-down views.",
    "iou": "Overlap: box/slot IoU above the threshold. For slots drawn around the whole vehicle.",
}


def parse_points(text: str) -> list[list[float]]:
    """'x1,y1,x2,y2' (a rectangle) or 'x,y; x,y; x,y; ...' (a polygon), as fractions -> [[x, y], ...].
    ValueError if it is neither."""
    text = (text or "").strip()
    if not text:
        raise ValueError("no points")
    try:
        if ";" not in text:
            v = [float(s) for s in text.split(",")]
            if len(v) != 4:
                raise ValueError("a rectangle needs 4 numbers 'x1,y1,x2,y2'; a polygon needs 'x,y; x,y; x,y'")
            x1, x2, y1, y2 = min(v[0], v[2]), max(v[0], v[2]), min(v[1], v[3]), max(v[1], v[3])
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        pts = [[float(s) for s in p.split(",")] for p in text.split(";") if p.strip()]
    except ValueError as e:
        raise ValueError(str(e) if "needs" in str(e) else f"not numbers: {text!r}") from None
    if any(len(p) != 2 for p in pts):
        raise ValueError("each polygon point is 'x,y'")
    return pts


def format_points(poly) -> str:
    """Inverse of parse_points: an axis-aligned rectangle as 'x1,y1,x2,y2', anything else as 'x,y; x,y; ...'."""
    p = np.asarray(poly, float)
    xs, ys = np.unique(p[:, 0]), np.unique(p[:, 1])
    if len(p) == 4 and len(xs) == 2 and len(ys) == 2:
        return f"{xs[0]:.3f},{ys[0]:.3f},{xs[1]:.3f},{ys[1]:.3f}"
    return "; ".join(f"{x:.3f},{y:.3f}" for x, y in p)


def validate_slots(slots: list[dict]) -> list[dict]:
    """-> clean copies {"id": int, "name": str, "poly": [[x, y], ...]} sorted by id. ValueError naming the first bad slot."""
    out, seen = [], set()
    for i, s in enumerate(slots, 1):
        try:
            sid = int(s["id"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"row {i}: the slot id must be a whole number") from None
        if sid < 1 or sid != float(s["id"]):
            raise ValueError(f"row {i}: the slot id must be a whole number >= 1")
        if sid in seen:
            raise ValueError(f"slot {sid}: id used twice")
        seen.add(sid)
        poly = np.asarray(s.get("poly", []), float)
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
            raise ValueError(f"slot {sid}: needs at least 3 points")
        if (poly < 0).any() or (poly > 1).any():
            raise ValueError(f"slot {sid}: points must be fractions of the frame (0..1)")
        if cv2.contourArea(np.float32(poly)) < 1e-4:  # < 0.01% of the frame: a line or a dot
            raise ValueError(f"slot {sid}: the polygon has no area")
        name = str(s.get("name") or "").strip() or str(sid)
        slot = {"id": sid, "name": name, "poly": [[round(float(x), 4), round(float(y), 4)] for x, y in poly]}
        if s.get("lift"):  # the auto grid's: how far up the image (frame fraction) a vehicle's centre lies from its box bottom here
            slot["lift"] = round(float(s["lift"]), 5)
        if s.get("capacity") not in (None, "", 1):  # how many vehicles the operator says this slot holds (a big pin); 1 = the default
            try:
                cap = int(s["capacity"])
                ok = cap == float(s["capacity"]) and 1 <= cap <= MAX_CAPACITY
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"slot {sid}: the capacity must be a whole number from 1 to {MAX_CAPACITY}")
            if cap > 1:
                slot["capacity"] = cap
        if s.get("slot_id") not in (None, ""):  # a pin's own slot id: fixed once set, so the pin keeps its history (auto grid)
            try:
                pid = int(s["slot_id"])
                ok = pid == float(s["slot_id"]) and 1 <= pid < DYNAMIC_BASE
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"slot {sid}: the slot id of a pin must be a whole number from 1 to {DYNAMIC_BASE - 1}")
            slot["slot_id"] = pid
        out.append(slot)
    return sorted(out, key=lambda s: s["id"])


class FixedSlotStore:
    """config/fixed_slots.json: the fixed slot maps of all cameras."""

    def __init__(self, path):
        self.path = Path(path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "cameras": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.setdefault("cameras", {})
        return data

    def cameras(self) -> list[str]:
        return sorted(self._read()["cameras"])

    def load(self, camera: str) -> list[dict]:
        """The camera's slots (validated), [] if none saved."""
        return validate_slots(self._read()["cameras"].get(camera, {}).get("slots", []))

    def get(self, camera: str) -> dict:
        """{"camera", "updated", "slots"} for an API; updated None if never saved."""
        entry = self._read()["cameras"].get(camera, {})
        return {"camera": camera, "updated": entry.get("updated"), "slots": validate_slots(entry.get("slots", []))}

    def upsert_slot(self, camera: str, slot: dict, tz=None) -> list[dict]:
        """Add or replace one slot (matched by id); the camera's other slots are untouched -> all its slots."""
        sid = validate_slots([slot])[0]["id"]
        return self.save(camera, [s for s in self.load(camera) if s["id"] != sid] + [slot], tz)

    def delete_slot(self, camera: str, slot_id: int, tz=None) -> list[dict]:
        """Remove one slot; the camera's other slots are untouched -> the remaining slots. KeyError if unknown."""
        slots = self.load(camera)
        if all(s["id"] != slot_id for s in slots):
            raise KeyError(f"camera {camera!r} has no slot {slot_id}")
        return self.save(camera, [s for s in slots if s["id"] != slot_id], tz)

    def save(self, camera: str, slots: list[dict], tz=None) -> list[dict]:
        """Validate and store the camera's slots (other cameras untouched) -> the stored slots. [] removes the camera."""
        clean = validate_slots(slots)
        data = self._read()
        if clean:
            data["cameras"][camera] = {"updated": datetime.now(tz).astimezone(tz).isoformat(timespec="seconds"),
                                       "slots": clean}
        else:
            data["cameras"].pop(camera, None)
        self._write(data)
        return clean

    def _write(self, data: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: a crash never leaves half a file

    # -- the auto grid's operator input (Dynamic mode on a HIGH calibration, geometry_live.GeometryFeed.auto_slots) ----------
    # Kept under their own top-level key, apart from the fixed map: save() above rewrites a camera's "cameras" entry (and the
    # registry mirrors into it), which must never take the pins with it.
    #   "auto_grid": {"<camera>": {"updated", "expected": 11 | null, "pins": [{"id", "name", "poly"}]}}
    def get_auto_grid(self, camera: str) -> dict:
        """{"camera", "updated", "expected" (int or None), "pins"} for an API; updated None if never saved."""
        entry = self._read().get("auto_grid", {}).get(camera, {})
        return {"camera": camera, "updated": entry.get("updated"), "expected": clean_expected(entry.get("expected")),
                "pins": validate_slots(entry.get("pins", []))}

    def save_auto_grid(self, camera: str, expected, pins: list[dict], tz=None) -> dict:
        """Validate and store the camera's expected bay count and pinned bays (everything else untouched) -> what was
        stored. Nothing to keep (no count, no pins) removes the camera's entry. ValueError naming the bad value."""
        expected, clean = clean_expected(expected, strict=True), validate_slots(pins)
        data = self._read()
        grids = data.setdefault("auto_grid", {})
        if expected is not None or clean:
            grids[camera] = {"updated": datetime.now(tz).astimezone(tz).isoformat(timespec="seconds"),
                             "expected": expected, "pins": clean}
        else:
            grids.pop(camera, None)
        if not grids:
            data.pop("auto_grid", None)
        self._write(data)
        return self.get_auto_grid(camera)


MAX_EXPECTED = 64  # bays: an expected count above this is a typo
MAX_EXTRAPOLATED = 3  # bays the auto grid may add past its last measured tile to reach the expected count (geometry_live)


def clean_expected(value, strict: bool = False) -> int | None:
    """The expected bay count: a whole number 1..MAX_EXPECTED, or None (not set). A bad stored value reads as None; a bad
    value being saved (strict) is a ValueError."""
    if value is None or value == "":
        return None
    try:
        n = int(value)
        if n != float(value) or not 1 <= n <= MAX_EXPECTED:
            raise ValueError
    except (TypeError, ValueError):
        if strict:
            raise ValueError(f"expected bays: a whole number from 1 to {MAX_EXPECTED}") from None
        return None
    return n


# the auto slot grid's own rule (Dynamic mode on a HIGH calibration, see Worker._auto_run); not offered in the UIs, so not in
# MATCH_RULES. Its bays are measured ground tiles: the vehicle stands where its bottom edge meets the ground, so the point is
# the box's bottom centre, not "ground"'s 10 % of the box height, which for a truck far from the lens is several metres of
# ground away. A far tile is only a few pixels tall, so a box covers little of it: the "just outside" tolerance needs no coverage
FOOT = "foot"


def _point(box, rule: str, lift: float = 0.0):
    """Reference point of a detection box (fractions) for the point rules. lift (FOOT only): the box bottom is the vehicle's near
    wheels; its centre stands about half its length / width farther, which is `lift` higher up the image at this slot."""
    (x1, y1), (x2, y2) = box.min(0), box.max(0)
    if rule == FOOT:
        return float((x1 + x2) / 2), float(y2 - lift)
    if rule == "ground":  # 10% above the bottom edge: on the near wheels, not on the floor in front of them
        return float((x1 + x2) / 2), float(y2 - 0.1 * (y2 - y1))
    return float((x1 + x2) / 2), float((y1 + y2) / 2)


# point rules, a car parked slightly forward or crooked: its point just outside the slot (within POINT_TOL of the
# box's own height) still counts when the box covers >= POINT_COVER of the slot. Front Parking 02, 2026-10-22: the
# maroon hatchback's wheels stood 0.5 % of the frame below S1's drawn edge
POINT_TOL, POINT_COVER = 0.15, 0.3


def match_score(box, slot, rule: str = "ground", min_iou: float = 0.15, lift: float = 0.0, share: float = 0.0) -> float:
    """> 0 if the detection box occupies the slot under the rule; higher = better (for one-to-one matching).
    Point rules: the point inside the slot -> 1 + coverage; just outside it (POINT_TOL, POINT_COVER) -> the coverage
    alone, below every inside match, so a slot that holds the point always wins the vehicle. share (0 = off): that near
    miss also needs this share of the vehicle's ground footprint (footprint_poly) inside the slot, with the foot rule too:
    a tyre or bumper over a bay's line is not a vehicle in it (Front Parking, 2026-10-26: a white SUV in the next bay);
    "on" is ground_on."""
    box, slot = np.float32(box), np.float32(slot)
    v_iou, cover = overlap(slot, box)
    if rule == "iou":
        return v_iou if v_iou >= min_iou else 0.0
    if rule not in ("ground", "center", FOOT):
        raise ValueError(f"unknown match rule {rule!r}; use one of {list(MATCH_RULES)}")
    poly, point = slot.reshape(-1, 1, 2), _point(box, rule, lift)
    if cv2.pointPolygonTest(poly, point, False) >= 0:
        # (the point is inside: a lifted foot point or a box centre can land in a bay the vehicle's ground is mostly outside of.
        # Real vehicles in their bays measure 0.74-0.77 on camera-1, so this one is held to INSIDE_SHARE at most, not to `share`)
        if share > 0 and ground_on(box, slot) < min(share, INSIDE_SHARE):
            return 0.0
        return 1.0 + cover  # cover breaks ties between neighbouring slots
    height = float(box[:, 1].max() - box[:, 1].min())
    if (cover >= POINT_COVER or rule == FOOT) and -cv2.pointPolygonTest(poly, point, True) <= POINT_TOL * height and \
            (share <= 0 or ground_on(box, slot) >= share):
        return max(cover, 0.01)  # (a far tile the box barely overlaps: still above 0, still below every inside match)
    return 0.0


# --- vehicle-driven bays of the auto grid (Dynamic mode on a HIGH calibration, Worker._auto_run) ------------------------------
# The grid is the calibration's measured ground; a vehicle that parks off it (between bays, or where the ground could not be
# measured) still has to count. So, on an auto grid only (FixedSlotTracker(auto=...)):
#  * a vehicle that no bay claims by the rule above is claimed by the bay its GROUND FOOTPRINT (the bottom FOOT_SHARE of its box:
#    the wheels, not the roof) overlaps by IoU >= FOOT_IOU; between two bays: the one whose centre is nearest the footprint's;
#  * one that still has no bay, stands on the measured surface, has conf >= SPAWN_CONF and a footprint overlapping every bay by
#    IoU < SPAWN_IOU, for SPAWN_HOLD s, gets a bay of its own (ids from DYNAMIC_BASE, names D1, D2 ...);
#  * such a bay goes when no vehicle was in it for CLEAR_FRAMES frames in a row and it is not BUSY. The grid's bays and the
#    operator's pins are never touched.
# CAPACITY: a slot (or pin) with capacity N > 1 (validate_slots) is N spaces: the total counts N, BUSY counts the distinct vehicles
# on it (at most N), and it makes no auto sub-bays (FixedSlotTracker._count_loads). For a distant view where a sub-bay's
# polygon is unreliable.
# SUB-BAYS (FixedSlotTracker(split=True), any map): a slot is one space, but a big pin can hold two vehicles (a truck and a car).
# One-to-one matching gave the slot to one of them and ignored the other. A vehicle with no bay whose footprint lies >= SPLIT_INSIDE
# inside a slot another vehicle holds (and is a different vehicle: footprint IoU < SPLIT_DISTINCT) gets a sub-bay of that parent,
# named "<parent>.2", ".3" ...: same id pool, debounce and decay as a vehicle-driven bay. The parent keeps its own vehicle, so busy =
# vehicles. Every vehicle-driven bay is saved (HistoryDB.save_slots) and restored after a restart; its id is never reused in the scope.
DYNAMIC_BASE = 1000
FOOT_SHARE, FOOT_IOU = 0.30, 0.25
SPAWN_CONF, SPAWN_IOU, SPAWN_HOLD, SPAWN_SAME = 0.45, 0.20, 3.0, 0.5
SPAWN_FRAMES = 3  # consecutive frames a vehicle must stand in no bay (or in a shared one) before it gets a bay of its own
SPLIT_INSIDE, SPLIT_DISTINCT = 0.9, 0.3  # sub-bays: share of the footprint inside the shared slot; max footprint IoU with its holder
CLEAR_FRAMES = 30
MAX_CAPACITY = 20  # the most vehicles an operator can say one slot holds
CAPACITY_FRAMES = 5  # frames a capacity slot's vehicle count is judged over (see FixedSlotTracker._count_loads)
CAPACITY_DISTINCT = 0.5  # footprint IoU from which two boxes in a capacity slot are one vehicle
INSIDE_SHARE = 0.5  # the most ground_on a vehicle whose reference point lies inside a bay needs to hold it (match_score)
SPAWN_PAD = 0.1  # the new bay reaches this share of the footprint's height below the box bottom (the foot point is inside it)


def _rasters(a, b, size: int):
    """Two polygons (fractions) filled on one raster of `size` px over their joint box -> (mask a, mask b), or None when their
    boxes do not meet."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.max(axis=0) < b.min(axis=0)).any() or (b.max(axis=0) < a.min(axis=0)).any():
        return None
    lo, hi = np.minimum(a.min(axis=0), b.min(axis=0)), np.maximum(a.max(axis=0), b.max(axis=0))
    s = size / max(float((hi - lo).max()), 1e-9)
    shape = tuple(int(np.ceil(v)) + 1 for v in ((hi - lo) * s)[::-1])
    return tuple(cv2.fillPoly(np.zeros(shape, np.uint8), [np.round((p - lo) * s).astype(np.int32)], 1) for p in (a, b))


def poly_iou(a, b, size: int = 200) -> float:
    """Intersection over union of two polygons (any shape, fractions of the frame), on a raster of `size` px over their joint box."""
    m = _rasters(a, b, size)
    if m is None:
        return 0.0
    union = int(np.count_nonzero(m[0] | m[1]))
    return int(np.count_nonzero(m[0] & m[1])) / union if union else 0.0


def poly_inside(a, b, size: int = 400) -> float:
    """The share of polygon a that lies inside polygon b (0..1)."""
    m = _rasters(a, b, size)
    area = 0 if m is None else int(np.count_nonzero(m[0]))
    return int(np.count_nonzero(m[0] & m[1])) / area if area else 0.0


def ground_on(box, slot) -> float:
    """How much a vehicle box stands on a slot (0..1): the larger of the share of its ground footprint (footprint_poly) inside the
    slot and the share of the slot under that footprint. A car in its bay is high on one or both (a thin far tile is covered but
    the footprint reaches past it); a tyre or bumper over the bay's line is low on both."""
    fp = footprint_poly(box)
    return max(poly_inside(fp, slot), poly_inside(slot, fp))


def footprint_poly(box, share: float = FOOT_SHARE, below: float = 0.0) -> np.ndarray:
    """The ground footprint of a vehicle box (polygon in fractions, 4x2): its bottom `share` (the wheels), reaching `below` of
    its own height past the box bottom."""
    (x1, y1), (x2, y2) = np.float32(box).min(0), np.float32(box).max(0)
    top, low = y2 - share * (y2 - y1), y2 + below * (y2 - y1)
    return np.float32([[x1, top], [x2, top], [x2, low], [x1, low]])


class FixedSlotTracker(SlotTracker):
    """SlotTracker on a fixed slot map: never learns, matches by rule (MATCH_RULES). slots: validated dicts
    (FixedSlotStore.load). Everything else (hold, exit_hold, occlusion, absent = FREE, events) is SlotTracker's.

    evidence_filter=1 (post-demo item 4, docs/post_demo_engineering_plan.md; default 0 = off, and then nothing below runs): every
    slot's observation goes through evidence.py before the debounce: an EMA with a hysteresis band (a missed frame or a jittering box
    is not an exit), a parked lock on a car that has not moved, and an exit veto on a locked bay (a re-look, the car's appearance, the
    local-vs-global change of the picture). The veto needs pixels: the monitor sets .frame (and .relook) for each update; without
    them the EMA still runs and a locked bay's exit is not vetoed."""

    def __init__(self, slots: list[dict], rule: str = "ground", min_iou: float = 0.15, auto: dict | None = None,
                 split: bool = False, evidence_filter: int = 0, evidence_cfg: evd.EvidenceCfg | None = None,
                 spillage: int = 0, spill_cfg: spl.SpillCfg | None = None, obstacle_filter: int = 0,
                 obstacle_cfg: obf.ObstacleCfg | None = None, blocked: int = 0, blocked_cfg: blk.BlockedCfg | None = None, **kwargs):
        kwargs.pop("learn", None)
        kwargs.pop("learn_hold", None)
        super().__init__({s["id"]: s["poly"] for s in slots}, learn=False, suppress_phantoms=False,
                         share_cover=None, **kwargs)
        if rule not in MATCH_RULES and rule != FOOT:
            raise ValueError(f"unknown match rule {rule!r}; use one of {list(MATCH_RULES)}")
        self.rule, self.min_iou = rule, min_iou
        self.names = {s["id"]: s["name"] for s in slots}
        self.lift = {s["id"]: s.get("lift", 0.0) for s in slots}
        self.capacity = {s["id"]: s.get("capacity", 1) for s in slots}  # the operator's vehicles-per-slot (see CAPACITY above)
        self.load: dict[int, int] = {}  # slot -> vehicles counted in it now (only slots with capacity > 1)
        self._seen_load: dict[int, list[int]] = {}
        self.load_notes: dict[int, str] = {}  # slot -> why boxes on it were or were not counted (diagnostics)
        # the auto grid's vehicle-driven rules (see DYNAMIC_BASE): auto = {"surface": the measured ground, polygon in fractions,
        # or None: footprint matching only, no new bays}
        self.auto = auto
        surface = (auto or {}).get("surface")
        self.surface = None if not surface else np.float32(surface).reshape(-1, 1, 2)
        self.split = split  # sub-bays for vehicles sharing a slot (see SUB-BAYS above)
        self._spawns: list[dict] = []  # vehicles standing in no bay: {"fp", "parent", "first", "last", "n"}
        self._dyn_next = DYNAMIC_BASE
        self.held: dict[int, float] = {}  # slot -> ground_on of the vehicle matched to it in the last match (diagnostics)
        self.parent: dict[int, int] = {}  # sub-bay -> the slot it was split from
        self.unplaced: list[dict] = []  # vehicles in the last frame that hold no bay yet, with the reason (diagnostics)
        self._unplaced_i: list[int] = []  # (their indices in the frame's detections, parallel to .unplaced)
        self._counted: set[int] = set()  # detections counted by a capacity slot in the last frame
        self._born: list[int] = []  # vehicle-driven bays made in the last frame (-> new_slots, saved by the monitor)
        self.evidence_filter = 1 if evidence_filter else 0  # (SlotTracker.update calls _evidence_obs only when 1)
        self.evidence_cfg = evidence_cfg or evd.CFG
        self.ev: dict[int, evd.SlotEvidence] = {}  # slot -> its evidence (EMA, parked lock, veto bookkeeping)
        self._ev_t: dict[int, float] = {}  # slot -> capture time of its previous observation
        self._ev_g: dict[int, np.ndarray] = {}  # slot -> 64x36 thumbnail of the whole frame when its lock was taken
        self.frame: np.ndarray | None = None  # the frame being updated (set by the monitor; dropped after update): crops, global change
        self.relook = None  # (slot id) -> True / False / None: a forced zoom on the slot found a vehicle (set by the monitor)
        # spillage = 1 (post-demo item 1, spillage.py; default 0 = off, and then nothing below runs): a FREE bay that neighbours' vehicles
        # stand on is .avail ENCROACHED (the state stays FREE: the history, sync_open and the counts of states are unaffected)
        self.spillage = 1 if spillage else 0
        self.spill_cfg = spill_cfg or spl.CFG
        self.avail_tracker = spl.AvailTracker()
        # the camera model of the frame being updated (set by the worker from the active calibration; None = uncalibrated): spillage then
        # measures on the ground (spillage.GroundMeasure), per pair falling back to the pixel rule where the calibration cannot answer
        self.ground_cam = None
        self.ground_mode = "auto"  # "auto" = the ground reading, "union" = also the pixel cover where higher (live option spillage_ground)
        self.avail: dict[int, spl.Spill] = {}  # slot -> its confirmed encroachment (empty while off)
        self.avail_events: list[dict] = []  # the last update's "encroached" / "encroach_cleared" (spillage.AvailTracker.update)
        self._avail_pending: list[dict] = []  # clearings made by switching off, handed out by the next update
        # blocked = 1 (post-demo item 5, blocked.py; default 0 = off, and then nothing below runs): a FREE bay with a non-vehicle object standing in it
        # (a potted plant, a vase, a box ...: the monitor hands the frame's non-vehicle detections over in .objects) is .block BLOCKED: the state stays
        # FREE (history, sessions, sync_open unchanged), the bay is just not usable and not counted free
        self.blocked = 1 if blocked else 0
        self.blocked_cfg = blocked_cfg or blk.CFG
        self.block_tracker = spl.AvailTracker(events=("object_blocked", "object_cleared"), notes=("object gone", "blocked-object detection switched off"))
        self.block: dict[int, blk.Block] = {}  # slot -> its confirmed blockage (empty while off)
        self.block_events: list[dict] = []  # the last update's "object_blocked" / "object_cleared"
        self._block_pending: list[dict] = []  # clearings made by switching off, handed out by the next update
        self.objects: list[dict] = []  # the frame's non-vehicle detections (set by the monitor; dropped after update)
        # obstacle_filter = 1 (post-demo item 2, obstacle_filter.py; default 0 = off, and then nothing below runs): vehicle detections that
        # are clutter (too small for the bay, an impossible aspect, a cone's orange, a flat shadow) are dropped before anything matches them
        self.obstacle_filter = 1 if obstacle_filter else 0
        self.obstacle_cfg = obstacle_cfg or obf.CFG
        self.rejected: list[dict] = []  # the last update's dropped detections: {"reason", "slot", "box", "colors"}

    def set_evidence(self, on) -> None:
        """Switch the evidence layer on or off while running (the worker follows the live option "evidence_filter"). Either way it
        starts from nothing: the next observation of each slot is seeded from its confirmed state."""
        on = 1 if on else 0
        if on == self.evidence_filter:
            return
        self.evidence_filter = on
        self.ev, self._ev_t, self._ev_g = {}, {}, {}
        self.frame = self.relook = None

    def set_obstacle_filter(self, on) -> None:
        """Switch the non-vehicle filter on or off while running (the worker follows the live option "obstacle_filter"). It keeps no
        state between frames, so there is nothing to reset but the last frame's list."""
        on = 1 if on else 0
        if on != self.obstacle_filter:
            self.obstacle_filter, self.rejected = on, []

    def _reject(self, dets, frame_shape):
        """The detections the non-vehicle filter lets through (obstacle_filter.judge). A box standing in a bay that is BUSY is never
        judged: the filter keeps clutter from making a bay busy, it does not take a parked car away. The colour rules need the frame
        (set by the monitor); without it only size and aspect apply."""
        h, w = frame_shape[:2]
        cfg, frame, keep, self.rejected = self.obstacle_cfg, self.frame, [], []
        polys = {k: (np.float32(s["poly"]) * (w, h)).astype(np.float32) for k, s in self.slots.items()}
        for d in dets:
            if d["state"] != 1:
                keep.append(d)
                continue
            pts = np.float32(d["poly"])
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            point = (float((x1 + x2) / 2), float(y2 - 0.1 * (y2 - y1)))  # the "ground" point: where the wheels stand
            at = [k for k, p in polys.items() if cv2.pointPolygonTest(p.reshape(-1, 1, 2), point, False) >= 0]
            k = min(at, key=lambda j: cv2.contourArea(polys[j])) if at else None
            if k is not None and self.slots[k]["state"] == 1:
                keep.append(d)
                continue
            ref = None
            if k is not None:
                px = polys[k]
                ref = float((px[:, 0].max() - px[:, 0].min()) * (px[:, 1].max() - px[:, 1].min())) / max(1, self.capacity.get(k, 1))
            reason, colors = obf.judge((x1, y1, x2, y2), ref, None if frame is None else (lambda: crop_box(frame, pts / (w, h))), cfg)
            if reason is None:
                keep.append(d)
            else:
                self.rejected.append({"reason": reason, "slot": k, "colors": colors,  # (the detection's own box: the key the overlay matches it by)
                                      "box": [round(float(v)) for v in (d["box"] if d.get("box") is not None else (x1, y1, x2, y2))]})
        return keep

    def set_spillage(self, on, now: float | None = None) -> None:
        """Switch the spillage layer on or off while running (the worker follows the live option "spillage"). Off clears every
        encroachment (its clearing events come out with the next update); on starts from nothing."""
        on = 1 if on else 0
        if on == self.spillage:
            return
        self.spillage = on
        if not on:
            self._avail_pending += self.avail_tracker.clear_all(time.time() if now is None else now)
        self.avail = {}

    def set_blocked(self, on, now: float | None = None) -> None:
        """Switch the blocked-by-object layer on or off while running (the worker follows the live option "blocked_objects"). Off clears every
        blockage (its "object_cleared" events come out with the next update); on starts from nothing."""
        on = 1 if on else 0
        if on == self.blocked:
            return
        self.blocked = on
        if not on:
            self._block_pending += self.block_tracker.clear_all(time.time() if now is None else now)
            self.objects = []
        self.block = {}

    def avail_name(self, k) -> str:
        """The usable state of slot k: BLOCKED (an object stands in it), else ENCROACHED, else what shown() says (FREE / BUSY / UNKNOWN)."""
        if k in self.block:
            return "BLOCKED"
        return "ENCROACHED" if k in self.avail else {0: "FREE", 1: "BUSY", None: "UNKNOWN"}[self.shown(k)]

    def _block(self, dets, frame_shape, now) -> None:
        """After SlotTracker.update: which FREE, unowned bays a non-vehicle object stands in (blocked.blocked_map), debounced. A bay a vehicle owns
        (the one-to-one _match of this frame's vehicles, re-run on the same boxes; .held, its diagnostics, is kept) is never blocked."""
        h, w = frame_shape[:2]
        ids = [k for k in self.slots]
        polys = [np.float32(d["poly"]) / (w, h) for d in dets if d["state"] == 1]
        held = self.held
        hit = self._match(polys, ids) if polys and ids else {}
        self.held = held
        taken = {ids[j] for j in hit.values()}
        eligible = {k for k in ids if k < DYNAMIC_BASE and self.slots[k]["state"] == 0 and k not in taken and self.capacity.get(k, 1) == 1}
        objs = [{"poly": np.float32(o["poly"]) / (w, h), "label": o.get("label"), "conf": o.get("conf", 0.0)} for o in self.objects if o.get("poly") is not None]
        raw = blk.blocked_map({k: self.slots[k]["poly"] for k in eligible}, objs, eligible, self.block, self.blocked_cfg) if objs else {}
        self.block_events = self._block_pending + self.block_tracker.update(raw, now, self.hold, self.exit_hold, eligible)
        self._block_pending = []
        self.block = dict(self.block_tracker.active)

    def _spill(self, dets, frame_shape, now) -> None:
        """After SlotTracker.update: which FREE, unowned bays do the vehicles standing in other bays cover (spillage.spill_map), debounced.
        Ownership is the one-to-one _match of the frame's vehicles (re-run here on the same boxes; .held, its diagnostics, is kept)."""
        h, w = frame_shape[:2]
        ids = [k for k in self.slots]
        polys = [np.float32(d["poly"]) / (w, h) for d in dets if d["state"] == 1]
        held = self.held
        hit = self._match(polys, ids) if polys and ids else {}
        self.held = held
        taken = {ids[j] for j in hit.values()}
        # a source is a vehicle BUSY-confirmed in a bay of its own, or one that holds no bay at all (owner 0: a van wider than the
        # bay_share rule lets any single bay own sits over several bays and would otherwise leave them all counted FREE); a vehicle
        # matched to a bay that is not BUSY yet (still arriving) or to a capacity pin is not
        owners = [(0 if i not in hit else ids[hit[i]] if self.slots[ids[hit[i]]]["state"] == 1 and self.capacity.get(ids[hit[i]], 1) == 1
                   else None) for i in range(len(polys))]
        # (on the ground metric an UNKNOWN bay can be encroached too: a bay behind a vehicle is UNKNOWN because that vehicle's box hides it, and it
        # is exactly the bay its far end may stand on; with the pixel rule only FREE bays, as before)
        open_states = (0, None) if self.ground_cam is not None else (0,)
        eligible = {k for k in ids if k < DYNAMIC_BASE and self.slots[k]["state"] in open_states and k not in taken
                    and self.capacity.get(k, 1) == 1}
        fps = [footprint_poly(p) for p in polys]
        ground = None
        if self.ground_cam is not None and polys:
            try:
                ground = spl.GroundMeasure(self.ground_cam, (w, h), fps, self.spill_cfg, mode=self.ground_mode)
            except Exception:  # noqa: BLE001 - a calibration problem means the pixel rule, never a lost frame
                ground = None
        raw = spl.spill_map({k: self.slots[k]["poly"] for k in eligible}, fps, owners, eligible,
                            self.avail, self.spill_cfg, ground) if polys else {}
        self.avail_events = self._avail_pending + self.avail_tracker.update(raw, now, self.hold, self.exit_hold, eligible)
        self._avail_pending = []
        self.avail = dict(self.avail_tracker.active)

    def _evidence_obs(self, k, s, o, box, conf, now):
        """SlotTracker's hook: the raw observation o (1 / 0 / None) of slot k -> the one the debounce sees. None (not judged:
        hidden, covered, unverifiable) passes through untouched, as without the filter."""
        if o is None:
            return None
        cfg = self.evidence_cfg
        ev = self.ev.get(k)
        if ev is None:  # a new bay's evidence starts from what it is confirmed to be: enabling the filter or editing the map is not an exit
            ev = self.ev[k] = evd.SlotEvidence()
            if s["state"] is not None:
                ev.e, ev.n, ev.seen = (0.9 if s["state"] == 1 else 0.0), 1, s["state"]
        dt = now - self._ev_t.get(k, now)
        self._ev_t[k] = now
        seen = evd.step(ev, evd.observation(o, conf, cfg), 1.0, dt, cfg)
        if o == 1 and box is not None and s["state"] == 1 and not ev.locked:
            frame = self.frame
            if evd.track_anchor(ev, box, now, cfg, crop_fn=None if frame is None else (lambda: crop_box(frame, box))) \
                    and frame is not None:
                self._ev_g[k] = thumbnail(frame)  # the whole picture at the lock: the baseline of "did the picture change as much"
        if seen is None:
            return o  # no decisive reading yet: the raw one
        return evd.gate_exit(ev, seen, now, cfg, probe=None if self.frame is None else (lambda: self._probe(k, ev)))

    def _probe(self, k, ev):
        """(signature of the crop at the anchor now, change of the whole picture since the lock, re-look hit) for exit_veto, cheapest
        first: the forced zoom (a detector call) only when the two cheap checks found no reason to veto."""
        frame, cfg = self.frame, self.evidence_cfg
        sig = None if ev.anchor is None else make_signature(crop_box(frame, ev.anchor))
        ref, glob = self._ev_g.get(k), None
        if ref is not None:
            now_t = thumbnail(frame)
            glob = float(np.abs(now_t - ref).mean()) if now_t.shape == ref.shape else None
        hit = None
        if self.relook is not None and evd.exit_veto(ev, sig, glob, None, cfg) is None:
            hit = self.relook(k)
        return sig, glob, hit

    def _evidence_after(self, events):
        """After SlotTracker.update: a bay confirmed FREE forgets its parked car; evidence of removed slots goes; the frame is dropped."""
        for e in events:
            if e["state"] == 0 and e["slot"] in self.ev:
                evd.release(self.ev[e["slot"]])
                self._ev_g.pop(e["slot"], None)
        for k in [k for k in self.ev if k not in self.slots]:
            for d in (self.ev, self._ev_t, self._ev_g):
                d.pop(k, None)
        self.frame = None

    def set_slots(self, slots: list[dict]) -> list[int]:
        """Swap in an edited map without losing state: a kept slot (same id) keeps its FREE/BUSY state and pending
        change with the new polygon and name, a new one starts unknown. -> ids of removed slots (the caller closes
        their open parking sessions)."""
        new = {s["id"]: s for s in validate_slots(slots)}
        dyn = {k: self.names.get(k, str(k)) for k in self.slots  # vehicle-driven bays outlive a new map, a sub-bay with its parent
               if k >= DYNAMIC_BASE and k not in new and (k not in self.parent or self.parent[k] in new)}
        removed = [k for k in self.slots if k not in new and k not in dyn]  # vehicle-driven bays outlive a new grid
        for k in removed:
            del self.slots[k]
        for k, s in new.items():
            if k in self.slots:
                self.slots[k]["poly"] = np.float32(s["poly"])
            else:
                self.slots[k] = {"poly": np.float32(s["poly"]), "state": None, "cand": None, "since": 0.0,
                                 "occluded": False}
        self.names = {k: s["name"] for k, s in new.items()} | dyn
        self.lift = {k: s.get("lift", 0.0) for k, s in new.items()} | {k: 0.0 for k in dyn}
        self.capacity = {k: s.get("capacity", 1) for k, s in new.items()}
        self.load = {k: v for k, v in self.load.items() if self.capacity.get(k, 1) > 1}
        self.hidden &= set(new) | set(dyn)
        self.parent = {k: p for k, p in self.parent.items() if k in self.slots}
        return removed

    def _match(self, polys, ids) -> dict[int, int]:
        pairs = sorted(((v, i, j) for i, p in enumerate(polys) for j, k in enumerate(ids)
                        if (v := match_score(p, self.slots[k]["poly"], self.rule, self.min_iou, self.lift.get(k, 0.0),
                                             self.bay_share)) > 0), reverse=True)
        out, used = {}, set()
        for _, i, j in pairs:
            if i not in out and j not in used:
                out[i] = j
                used.add(j)
        if self.auto is not None:  # a vehicle no bay claimed: the bay its wheels overlap, the nearest one between two
            near = []
            for i, p in enumerate(polys):
                if i in out:
                    continue
                fp = footprint_poly(p)
                near += [(float(np.linalg.norm(fp.mean(0) - self.slots[k]["poly"].mean(0))), i, j) for j, k in enumerate(ids)
                         if j not in used and poly_iou(fp, self.slots[k]["poly"]) >= FOOT_IOU and
                         ground_on(polys[i], self.slots[k]["poly"]) >= self.bay_share]
            for _, i, j in sorted(near):
                if i not in out and j not in used:
                    out[i] = j
                    used.add(j)
        self.held = {ids[j]: round(ground_on(polys[i], self.slots[ids[j]]["poly"]), 2) for i, j in out.items()}
        return out

    def update(self, dets: list[dict], frame_shape, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        if self.obstacle_filter:
            dets = self._reject(dets, frame_shape)
        active = self.auto is not None or self.split
        if active:
            self._spawn(dets, frame_shape, now)
        events = super().update(dets, frame_shape, now)
        if self.evidence_filter:
            self._evidence_after(events)
        if self.obstacle_filter and not self.evidence_filter:
            self.frame = None  # (the monitor hands it over for the colour rules; held for one update only)
        if self.spillage:
            self._spill(dets, frame_shape, now)
        elif self._avail_pending or self.avail_events:
            self.avail_events, self._avail_pending = self._avail_pending, []
        if self.blocked:
            self._block(dets, frame_shape, now)
            self.objects = []
        elif self._block_pending or self.block_events:
            self.block_events, self._block_pending = self._block_pending, []
        self._counted = set()
        if any(c > 1 for c in self.capacity.values()):
            self._count_loads(dets, frame_shape)
        if active:
            self._clear()
            self.new_slots, self._born = self._born, []  # (SlotTracker.update resets it): the monitor saves them
            keep = [n for n, i in enumerate(self._unplaced_i) if i not in self._counted]  # (a capacity slot counted it after all)
            self.unplaced, self._unplaced_i = [self.unplaced[n] for n in keep], [self._unplaced_i[n] for n in keep]
        else:
            self._unplaced_fixed(dets, frame_shape)
        return events

    def _unplaced_fixed(self, dets, frame_shape):
        """A Fixed map holds exactly the operator's slots and capacities: a vehicle that no slot holds or counts (a second
        vehicle in a capacity-1 slot, one standing in no slot) is not made a bay; it is reported in .unplaced with the reason."""
        h, w = frame_shape[:2]
        idx = [i for i, d in enumerate(dets) if d["state"] == 1]
        polys = [np.float32(dets[i]["poly"]) / (w, h) for i in idx]
        ids = list(self.slots)
        held = self._match(polys, ids).keys() if polys and ids else ()
        self.unplaced, self._unplaced_i = [], []
        for n, i in enumerate(idx):
            if n in held or i in self._counted:
                continue
            near = max(ids, key=lambda k: ground_on(polys[n], self.slots[k]["poly"]), default=None)
            on = near is not None and ground_on(polys[n], self.slots[near]["poly"]) >= FOOT_SHARE
            self.unplaced.append({"conf": round(dets[i].get("conf", 1.0), 2),
                                  "why": f"no free slot (Fixed map: capacity {self.capacity.get(near, 1)})" if on
                                  else "in no slot of the Fixed map"})
            self._unplaced_i.append(i)

    def vehicles_in(self, k: int) -> int:
        """Vehicles to count in a BUSY slot: 1, or its counted boxes up to its capacity (see _count_loads)."""
        return max(1, min(self.load.get(k, 0), self.capacity.get(k, 1)))

    def _count_loads(self, dets, frame_shape):
        """CAPACITY: a slot the operator gave capacity N holds up to N vehicles: the vehicle boxes standing on it that no other slot
        holds, one per vehicle. Standing on it = the middle of the box's ground footprint is inside the slot, or ground_on >=
        min(bay_share, INSIDE_SHARE) (a car half over the line still counts). A vehicle held by no slot whose footprint middle is within
        pin_reach x its height outside the slot counts too (auto grid only; the pin is drawn a little tight, as on camera 1's P1). One vehicle = boxes whose footprints overlap by
        less than CAPACITY_DISTINCT (a duplicate box, e.g. whole frame + zoom, overlaps almost completely; two cars beside each
        other with overlapping boxes do not). Steady: a count is taken when seen in 2 of the last CAPACITY_FRAMES frames and kept
        until 4 frames show fewer, so a box that drops out for a frame, or a flicker, does not move it. Why a box was left out is
        in .load_notes[slot]."""
        h, w = frame_shape[:2]
        idx = [i for i, d in enumerate(dets) if d["state"] == 1]
        cars = [dets[i] for i in idx]
        polys = [np.float32(d["poly"]) / (w, h) for d in cars]
        ids = list(self.slots)
        held_by = {ids[j]: i for i, j in self._match(polys, ids).items()}
        elsewhere = {i: k for k, i in held_by.items()}
        need = min(self.bay_share, INSIDE_SHARE)
        for k, cap in self.capacity.items():
            if cap <= 1 or k not in self.slots:
                continue
            slot = self.slots[k]["poly"]
            kept, notes = [], []
            for i in sorted(range(len(cars)), key=lambda i: -cars[i].get("conf", 1.0)):
                fp = footprint_poly(polys[i])
                centre = (float(fp[:, 0].mean()), float(fp[:, 1].mean()))
                reach = cv2.pointPolygonTest(slot.reshape(-1, 1, 2), centre, True)  # + inside, - outside (frame fractions)
                height = float(polys[i][:, 1].max() - polys[i][:, 1].min())
                if reach >= 0 or ground_on(polys[i], slot) >= need:
                    pass
                elif self.auto is not None and reach >= -self.pin_reach * height:  # just past the pin's edge (a pin drawn a little
                    # tight): the auto grid only, a Fixed map counts strictly inside its drawn slots
                    notes.append(f"counted: stands {-reach * 100:.1f}% of the frame outside its edge (draw the pin larger)")
                else:
                    continue  # not on this slot (a neighbour's car): nothing to report
                if elsewhere.get(i, k) != k:
                    notes.append(f"a vehicle on it is held by slot {elsewhere[i]}")
                elif any(poly_iou(fp, footprint_poly(polys[j])) >= CAPACITY_DISTINCT for j in kept):
                    notes.append("a box left out: same vehicle as a stronger one (footprints overlap)")
                else:
                    kept.append(i)
                    self._counted.add(idx[i])
            hist = (self._seen_load.get(k, []) + [len(kept)])[-CAPACITY_FRAMES:]
            self._seen_load[k] = hist
            # (seen in 2 frames, gone after 4; the first frame alone counts one at most)
            self.load[k] = min(sorted(hist, reverse=True)[1] if len(hist) > 1 else min(hist[0], 1), cap)
            self.load_notes[k] = f"{len(kept)} counted now" + ("; " + ", ".join(notes) if notes else "")

    def _spawn(self, dets, frame_shape, now: float):
        """Give a bay to each vehicle that has none (see DYNAMIC_BASE and SUB-BAYS above): one standing on the measured surface
        in no bay (auto grids), or in a slot another vehicle holds (split), for SPAWN_FRAMES consecutive frames over SPAWN_HOLD s
        (a car driving past, a flicker, never gets one). Added before the frame's matching, so the vehicle is in its bay from
        that frame on."""
        h, w = frame_shape[:2]
        polys = [np.float32(d["poly"]) / (w, h) for d in dets]
        ids = list(self.slots)
        hit = self._match(polys, ids)
        holder = {j: i for i, j in hit.items()}  # slot index -> the detection that holds it
        lone = []
        self.unplaced, self._unplaced_i = [], []
        for i, d in enumerate(dets):
            if i in hit or d["state"] != 1:
                continue
            if d.get("conf", 1.0) < SPAWN_CONF:
                self.unplaced.append({"conf": round(d.get("conf", 1.0), 2), "why": f"confidence under {SPAWN_CONF}"})
                self._unplaced_i.append(i)
                continue
            ground = footprint_poly(polys[i])
            fp = np.clip(footprint_poly(polys[i], below=SPAWN_PAD), 0.0, 1.0)
            parent, best, why = None, SPLIT_INSIDE, "in no bay"
            for j, k in enumerate(ids if self.split else ()):
                if k >= DYNAMIC_BASE or self.capacity.get(k, 1) > 1 or j not in holder or (share := poly_inside(ground, self.slots[k]["poly"])) < best:
                    continue
                # (the holders of the slot and of its sub-bays: a second box of any of those vehicles is not a new vehicle)
                held = [holder[x] for x, kx in enumerate(ids) if x in holder and (kx == k or self.parent.get(kx) == k)]
                if all(poly_iou(ground, footprint_poly(polys[h])) < SPLIT_DISTINCT for h in held):
                    parent, best = k, share
                else:
                    why = "the same vehicle as one already counted (footprints overlap)"
            if parent is not None:
                lone.append((fp, parent))
            elif self.surface is not None:
                foot = (float(polys[i][:, 0].mean()), float(polys[i][:, 1].max()))
                if cv2.pointPolygonTest(self.surface, foot, False) >= 0 and \
                        all(poly_iou(ground, s["poly"]) < SPAWN_IOU for s in self.slots.values()):
                    lone.append((fp, None))
                    continue
            if parent is None:
                self.unplaced.append({"conf": round(d.get("conf", 1.0), 2), "why": why})
                self._unplaced_i.append(i)
        seen = []
        for fp, parent in lone:
            c = next((c for c in self._spawns if c["parent"] == parent and poly_iou(c["fp"], fp) >= SPAWN_SAME), None)
            seen.append({**c, "fp": fp, "last": now, "n": c["n"] + 1} if c else
                        {"fp": fp, "parent": parent, "first": now, "last": now, "n": 1})
        self._spawns = []  # a vehicle that is not there on this frame is forgotten: its count starts over
        for c in seen:
            if c["n"] >= SPAWN_FRAMES and now - c["first"] >= SPAWN_HOLD and cv2.contourArea(c["fp"]) >= 1e-4:
                self._born.append(self._add(c["fp"], c["parent"]))
            else:
                self._spawns.append(c)
                self.unplaced.append({"conf": None, "why": f"waiting for its own bay ({c['n']}/{SPAWN_FRAMES} steady frames)"})
                self._unplaced_i.append(None)

    def _add(self, poly, parent: int | None, k: int | None = None) -> int:
        """A vehicle-driven bay (a new one, or one restored under its saved id) -> its id."""
        if k is None:
            k, self._dyn_next = self._dyn_next, self._dyn_next + 1
        else:
            self._dyn_next = max(self._dyn_next, k + 1)
        self.slots[k] = {"poly": np.float32(poly), "state": None, "cand": None, "since": 0.0, "occluded": False}
        self.lift[k] = 0.0
        if parent is None:
            self.names[k] = f"D{k - DYNAMIC_BASE + 1}"
        else:
            self.parent[k] = parent
            used = {n for j, n in self.names.items() if self.parent.get(j) == parent}
            suffix = next(n for n in range(2, len(self.slots) + 3) if f"{self.names.get(parent, parent)}.{n}" not in used)
            self.names[k] = f"{self.names.get(parent, parent)}.{suffix}"
        return k

    def restore(self, saved: dict, next_id: int = DYNAMIC_BASE) -> list[int]:
        """The vehicle-driven bays saved by an earlier run of this map (HistoryDB.load_slots: {id: polygon}); next_id: the lowest
        id never used in the scope, so an id is never reused. A bay whose centre lies inside a slot is that slot's sub-bay.
        A Fixed map (no auto grid, no sub-bays) has no such bays: they are not restored -> their ids (the caller retires them)."""
        self._dyn_next = max(self._dyn_next, next_id)
        if self.auto is None and not self.split:
            return [k for k in sorted(saved) if k >= DYNAMIC_BASE and k not in self.slots]
        for k, poly in sorted(saved.items()):
            if k < DYNAMIC_BASE or k in self.slots:
                continue
            centre = tuple(float(v) for v in np.float32(poly).mean(0))  # (a sub-bay reaches past its parent's near edge)
            inside = [j for j, s in self.slots.items() if self.split and j < DYNAMIC_BASE and
                      cv2.pointPolygonTest(s["poly"].reshape(-1, 1, 2), centre, False) >= 0]
            self._add(poly, min(inside, key=lambda j: cv2.contourArea(self.slots[j]["poly"])) if inside else None, k)
        return []

    def _clear(self):
        """Take back the vehicle-driven bays nobody has been in for CLEAR_FRAMES frames (and that are not BUSY)."""
        for k in [k for k in self.slots if k >= DYNAMIC_BASE]:
            if self.slots[k]["state"] != 1 and self.slots[k].get("empty_n", 0) >= CLEAR_FRAMES:
                del self.slots[k]
                self.names.pop(k, None)
                self.lift.pop(k, None)
                self.parent.pop(k, None)
                self.hidden.discard(k)
                self.retired.append(k)
                self.retire_notes[k] = "vehicle-driven bay cleared (its vehicle left)"


def relabel(polys, labelled: list[dict], min_iou: float = 0.3) -> list[dict]:
    """Shapes from a drawing tool (no ids) -> slot dicts: a shape that overlaps a labelled slot (IoU >= min_iou, best
    pairs first, one-to-one) keeps its id and name, so moving or reshaping a slot keeps its history; a new shape
    gets the next free id and the name "S<id>". Not validated (the editor reports problems per row)."""
    pairs = sorted(((overlap(np.float32(s["poly"]), np.float32(p))[0], i, j)
                    for i, p in enumerate(polys) for j, s in enumerate(labelled)), reverse=True)
    keep, used = {}, set()
    for v, i, j in pairs:
        if v < min_iou:
            break
        if i not in keep and j not in used:
            keep[i] = labelled[j]
            used.add(j)
    next_id = max([s["id"] for s in labelled] + [0]) + 1
    out = []
    for i, p in enumerate(polys):
        if i in keep:
            out.append({"id": keep[i]["id"], "name": keep[i]["name"], "poly": p})
        else:
            out.append({"id": next_id, "name": f"S{next_id}", "poly": p})
            next_id += 1
    return out


def draft_slots(dets: list[dict], frame_shape, start_id: int = 1) -> list[dict]:
    """One rectangle slot per vehicle detection (pixel boxes -> fractions), front to back (lowest box first), then left
    to right, as a starting point for the editor. Shown-only classes (state None) are skipped."""
    h, w = frame_shape[:2]
    boxes = sorted((d["box"] for d in dets if d.get("state") is not None), key=lambda b: (-b[3], b[0]))
    return validate_slots([{"id": start_id + i, "name": f"S{start_id + i}",
                            "poly": [[x1 / w, y1 / h], [x2 / w, y1 / h], [x2 / w, y2 / h], [x1 / w, y2 / h]]}
                           for i, (x1, y1, x2, y2) in enumerate(boxes)])
