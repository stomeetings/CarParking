"""Parking monitor service: one camera's frames -> vehicle detections -> slot map -> logged entry/exit events.

No Streamlit (or any UI) code here. scripts/worker.py drives a ParkingMonitor from its RTSP loop and the FastAPI app serves, as JSON:
  * ParkingMonitor.state()          live slots and counts       (e.g. GET /cameras/{id}/state)
  * FrameResult.to_json()           what the last frame showed  (e.g. a WebSocket push per processed frame)
  * HistoryDB.sessions() / events() / summary()                 (e.g. GET /cameras/{id}/sessions, /events?since_id=)
  * fixed_slots.FixedSlotStore.get() / save() + validate_slots() (e.g. GET / PUT /cameras/{id}/fixed-slots)
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from carparking.fixed_slots import FixedSlotTracker, validate_slots
from carparking.dynamic_iou import iou
from carparking.history import HistoryDB, SlotTracker, _hidden
from carparking.obstacle_filter import RejectLog
from carparking.yolo_engine import _iou_ios, merge_dets

STATE_NAMES = {0: "FREE", 1: "BUSY", None: "UNKNOWN"}
# the parking-area wall (in_area): share of a vehicle box that must lie inside the area; EDGE_INSIDE for a box cut by
# the frame's left / right side (a piece of something: the 2026-10-19 scaffolding box)
MIN_INSIDE, EDGE_INSIDE = 0.7, 0.9
# merged boxes (ParkingMonitor._split): whole-frame vehicle boxes at least SPLIT_MIN_H of the frame tall are checked
# with a second whole-frame pass at SPLIT_IMGSZ; a result (split or not) is reused for SPLIT_EVERY s while the box stays
SPLIT_MIN_H, SPLIT_IMGSZ, SPLIT_EVERY = 0.25, 1280, 30.0
# a vehicle box rising above its own slot's top by more than this share of the slot's height has swallowed the car
# parked behind it (ParkingMonitor._apart)
SWALLOW = 0.15
INSIDE = 0.8  # a zoom box this much inside a vehicle box already found is a piece of that vehicle (ParkingMonitor._zoom)
RELOOK_CONF = 0.15  # confidence of the evidence layer's forced re-look at a locked bay about to be released (ParkingMonitor._relook)


def parse_roi(text: str):
    """'x,y; x,y; x,y; ...' as fractions (0..1) of width/height -> (N, 2) array, None if empty; ValueError if bad."""
    if not text.strip():
        return None
    pts = np.array([[float(v) for v in p.split(",")] for p in text.split(";") if p.strip()], float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or (pts < 0).any() or (pts > 1).any():
        raise ValueError("need at least 3 'x,y' points between 0 and 1, separated by ';'")
    return pts


@dataclass
class ParkingArea:
    """Where vehicles count: margins cut from each edge (shares of the frame), then the optional ROI polygon
    (fractions). Outside is ignored (building, stairs, timestamp, road): dynamic mode uses in_area (centre, ground
    point and 90 % of the box inside) for detections and learned slots; fixed mode keeps the centre test (inside)."""
    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0
    roi: np.ndarray | None = None

    @classmethod
    def from_settings(cls, s: dict) -> ParkingArea:
        """From the UI's settings dict (keys top/bottom/left/right/roi; missing = no cut)."""
        return cls(s.get("top", 0.0), s.get("bottom", 0.0), s.get("left", 0.0), s.get("right", 0.0), s.get("roi"))

    def mask(self, shape) -> np.ndarray:
        """uint8 mask for a frame of this shape, 255 = parking area."""
        h, w = shape[:2]
        m = np.zeros((h, w), np.uint8)
        m[int(self.top * h):int(round((1 - self.bottom) * h)), int(self.left * w):int(round((1 - self.right) * w))] = 255
        if self.roi is not None:
            poly = np.zeros_like(m)
            cv2.fillPoly(poly, [np.round(np.asarray(self.roi) * (w, h)).astype(np.int32)], 255)
            m &= poly
        return m


def inside(mask: np.ndarray, poly) -> bool:
    """The centre of the polygon's bounding box (pixels) lies in the mask."""
    p = np.asarray(poly, float)
    h, w = mask.shape
    x, y = (p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
    return mask[min(max(int(y), 0), h - 1), min(max(int(x), 0), w - 1)] > 0


def in_area(mask: np.ndarray, poly, min_inside: float = MIN_INSIDE, edge_inside: float = EDGE_INSIDE) -> bool:
    """Hard wall for dynamic slots and their detections (pixels): the box centre and its ground point (bottom centre,
    where the vehicle stands) lie in the mask, and >= min_inside of the box area does (0.7: a car parked a little
    forward or shifted overhangs the wall while it stands inside it; Front Parking 02, 2026-10-22: the maroon
    hatchback, 87 % inside the left cut, was dropped at the former 0.9 and never counted). A box cut by the frame's
    left / right side needs edge_inside (0.9): it is a piece of something (the box over the scaffolding at the left
    frame edge, 2026-10-19, centre just inside the wall)."""
    p = np.asarray(poly, float)
    h, w = mask.shape
    x1, y1 = max(int(p[:, 0].min()), 0), max(int(p[:, 1].min()), 0)
    x2, y2 = min(int(np.ceil(p[:, 0].max())), w), min(int(np.ceil(p[:, 1].max())), h)
    if x2 <= x1 or y2 <= y1 or not inside(mask, p):
        return False
    if not mask[y2 - 1, min((x1 + x2) // 2, w - 1)]:
        return False
    whole = p[:, 0].min() > 0.01 * w and p[:, 0].max() < 0.99 * w
    return mask[y1:y2, x1:x2].mean() >= 255 * (min_inside if whole else max(min_inside, edge_inside))


def det_json(d: dict) -> dict:
    """A detection as plain JSON: {"label", "conf", "box" [x1, y1, x2, y2] pixels, "src"}."""
    return {"label": d["label"], "conf": round(float(d["conf"]), 3), "box": [int(v) for v in d["box"]],
            "src": d.get("src", "")}


@dataclass
class FrameResult:
    """One processed frame. vehicles: what fed the slot map; others: shown-only classes (person, dog, ...);
    events: slot transitions confirmed in this frame (logged=True if they opened/closed a session)."""
    vehicles: list[dict]
    others: list[dict]
    events: list[dict]
    mask: np.ndarray = field(repr=False)
    new_slots: list[int] = field(default_factory=list)
    retired: list[int] = field(default_factory=list)  # phantom slots removed from the learned map in this frame

    @property
    def logged(self) -> bool:
        return any(e["logged"] for e in self.events)

    def to_json(self) -> dict:
        return {"vehicles": [det_json(d) for d in self.vehicles], "others": [det_json(d) for d in self.others],
                "events": self.events, "new_slots": list(self.new_slots),
                "retired": list(self.retired)}


class ParkingMonitor:
    """One camera: detector + parking area + slot tracker + HistoryDB, in one of two slot mapping modes:
    * dynamic (fixed=None): SlotTracker learns the slot map from detections; it is loaded from / saved to the
      database. Saved slots outside the parking area are skipped (their IDs never reused, so old history rows keep
      pointing at the right slot).
    * fixed (fixed=[slot dicts] from FixedSlotStore.load): FixedSlotTracker on that map, matched by match_rule
      (fixed_slots.MATCH_RULES); nothing is learned or saved. Give its HistoryDB its own detector scope (e.g.
      "yolov8s | fixed"), so fixed slot IDs never mix with learned ones in the history.
    detector: any callable frame -> [{"poly", "box", "label", "conf", "state"}] (YoloDetector, HybridDetector);
    state None = shown only, never a slot. targets: class names to keep (None = all the detector returns).
    tracker: SlotTracker keyword arguments (hold, exit_hold, learn_hold, absent_free, learn, occlusion_cover).
    Zoom re-check (zoom_max slots per frame, 0 = off; detectors with detect_region): a slot that holds a car (BUSY),
    or is UNKNOWN while hidden / a duplicate, and got no detection now is looked at again on a crop around it at
    zoom_conf (0.30; the whole frame keeps the normal threshold); a FREE slot only while the vehicles in front hide
    it (occlusion_cover), at zoom_free_conf (0.40). A box found there counts only if its centre lies in
    that slot, it fits the slot (IoU >= match_iou) and it repeats no box already found (merge_dets); it then goes
    through the same filters (hard wall, phantom, tiny) as any other. A visible FREE slot is never re-checked: an
    empty bay in view can't turn BUSY from a low-confidence box; a slot whose car left frees itself on the normal
    timers (no box, zoomed or not, for exit_hold / hidden_max / exit_frames)."""

    def __init__(self, detector, db: HistoryDB, area: ParkingArea | None = None, targets=None, sliced: bool = False,
                 conf: float | None = None, fixed: list[dict] | None = None, match_rule: str = "ground",
                 match_iou: float = 0.15, zoom_max: int = 2, zoom_conf: float = 0.30, zoom_free_conf: float = 0.40,
                 **tracker):
        self.detector, self.db, self.area = detector, db, area or ParkingArea()
        self.targets, self.sliced, self.conf, self.tracker_kwargs = targets, sliced, conf, tracker
        self.fixed, self.match_rule, self.match_iou = fixed, match_rule, match_iou
        self.tracker: SlotTracker | None = None
        self.ignored = 0  # saved slots skipped because they lie outside the parking area
        self.zoom_max, self.zoom_conf, self.zoom_free_conf = zoom_max, zoom_conf, zoom_free_conf
        self._zoom_turn = 0  # round robin when more slots need a re-check than zoom_max
        self._splits: list[dict] = []  # merged-box checks: {"box", "at", "parts" (front, rear) or None}
        self._mask_key = self._mask = self._shape = None
        # the history against the live count (see _check_sessions): {"expected", "open", "since" (when they started to differ)}
        self.session_check: dict | None = None
        self._rejects = RejectLog()  # (non-vehicle filter on: the dropped detections as a few audit notices, see obstacle_filter.RejectLog)
        self.notice_error = ""  # why the last notice could not be written to the event log, else ""
        self.session_error = ""  # why the last repair of the open sessions failed (a locked / read-only history file), else ""
        # the stream's health (stream_health): anything but "active" shows every slot UNKNOWN (state()); the tracker, the history and
        # its open sessions are untouched meanwhile (the worker does not process frames of a stale / frozen stream)
        self.stream = "active"

    @property
    def mode(self) -> str:
        return "dynamic" if self.fixed is None else "fixed"

    def mask(self, shape) -> np.ndarray:
        if self._mask_key != shape[:2]:
            self._mask_key, self._mask = shape[:2], self.area.mask(shape)
        return self._mask

    def start(self, shape):
        """Build the tracker for frames of this shape (done by the first process()): the fixed map, or the learned map
        saved in the database."""
        if self.fixed is not None:
            self.tracker = FixedSlotTracker(validate_slots(self.fixed), self.match_rule, self.match_iou,
                                            **self.tracker_kwargs)
            # the vehicle-driven bays of earlier runs of this map; a Fixed map (no auto grid, no sub-bays) has none: the ones an
            # earlier run saved are retired, which closes their open sessions
            for k in self.tracker.restore(self.db.load_slots(), self.db.max_slot_id() + 1):
                self.db.retire_slots([k], note="vehicle-driven bay of an earlier run: a Fixed map has no sub-bays")
            return
        self.tracker = SlotTracker(self.db.load_slots(), proven=self.db.proven_slots(),
                                   distinct=self.db.distinct_pairs(), area=self._area_test(shape),
                                   **self.tracker_kwargs)
        self.ignored = len(self.tracker.outside)
        self.tracker.next_id = self.db.max_slot_id() + 1  # retired and skipped slots' IDs are never reused
        for k, j in self.tracker.merged.items():  # duplicates in the saved map (slot NMS on load)
            self.db.retire_slots([k], note=f"duplicate slot merged into slot {j} (same space)")

    def _area_test(self, shape):
        """Dynamic mode: SlotTracker's hard wall, a test on polygons in fractions of a frame of this shape."""
        mask, (h, w) = self.mask(shape), shape[:2]
        return lambda poly: in_area(mask, np.asarray(poly) * (w, h))

    def set_area(self, area: ParkingArea) -> list[int]:
        """Apply a changed parking area while running (e.g. PUT /cameras/{id}/area). Dynamic mode: slots outside
        the new area are set aside at once (their IDs and state kept, back when the area grows). -> ids outside."""
        self.area, self._mask_key = area, None
        if self.tracker is None or self.fixed is not None:
            return []
        self.tracker.set_area(self._area_test(self._shape))
        self.ignored = len(self.tracker.outside)
        return list(self.tracker.outside)

    def update_fixed(self, slots: list[dict], now: float | None = None) -> list[int]:
        """Fixed mode: apply an edited slot map while running (e.g. PUT /cameras/{id}/fixed-slots). Kept slots keep
        their state; deleted slots' open sessions are closed in the history. -> removed ids."""
        if self.fixed is None:
            raise ValueError("update_fixed() needs a monitor in fixed mode")
        self.fixed = validate_slots(slots)
        if self.tracker is None:  # not started yet: start() builds the tracker from self.fixed
            return []
        removed = self.tracker.set_slots(self.fixed)
        for k in removed:
            self.db.close_slot(k, now, "slot deleted")
        return removed

    def process(self, frame, t: float | None = None) -> FrameResult:
        """Detect, filter to the parking area, update the slot map, log confirmed entries/exits. t: capture time
        (epoch seconds; events are timed by it, not by when inference finished)."""
        if self.tracker is None:
            self.start(frame.shape)
        self._shape = frame.shape
        if self.conf is not None:
            self.detector.conf = self.conf
        if self.targets is not None and hasattr(self.detector, "set_targets"):
            self.detector.set_targets(self.targets)  # in-model filter (whole frame); the label check covers SAHI
        mask = self.mask(frame.shape)
        keep = inside if self.fixed is not None else in_area  # dynamic: strict hard wall (no slot learned outside)
        found = [d for d in self.detector(frame, sliced=self.sliced)
                 if (self.targets is None or d["label"] in self.targets) and keep(mask, d["poly"])]
        vehicles = [d for d in found if d["state"] is not None]
        others = [d for d in found if d["state"] is None]
        if self.fixed is None:
            vehicles = self._split(frame, vehicles, time.time() if t is None else t)
        vehicles += self._zoom(frame, vehicles, lambda d: (self.targets is None or d["label"] in self.targets) and
                               keep(mask, d["poly"]))
        if getattr(self.tracker, "blocked", 0):  # (blocked by object: the frame's non-vehicle detections)
            self.tracker.objects = others
        if getattr(self.tracker, "obstacle_filter", 0):  # (non-vehicle filter: pixels for the colour rules)
            self.tracker.frame = frame
        if getattr(self.tracker, "evidence_filter", 0):  # (evidence layer: pixels for the parked car's signature and the exit veto)
            self.tracker.frame = frame
            self.tracker.relook = lambda k: self._relook(frame, k, lambda d: (self.targets is None or d["label"] in self.targets) and
                                                         keep(mask, d["poly"]))
        events = self.tracker.update(vehicles, frame.shape, now=t)
        if self.tracker.new_slots or self.tracker.updated_slots:  # learned, or grown for a larger vehicle
            self.db.save_slots(self.tracker, self.tracker.new_slots + self.tracker.updated_slots)
        if self.tracker.new_distinct:  # slots that held two vehicles at once: never merged as duplicates
            self.db.add_distinct_pairs(self.tracker.new_distinct)
        for k in self.tracker.retired:  # phantoms and tiny strays: closes an open session with the reason
            note = self.tracker.retire_notes.get(k)
            self.db.retire_slots([k], t, **({"note": note} if note else {}))
        out = []
        for e in events:
            logged = self.db.apply(e)
            # entry / exit = a session opened / closed; busy / free = state confirmed without one (car already logged
            # before a restart, slot empty at start)
            kind = ("entry" if e["state"] == 1 else "exit") if logged else STATE_NAMES[e["state"]].lower()
            out.append({"slot_id": e["slot"], "event": kind, "state": STATE_NAMES[e["state"]],
                        "time": self.db.iso(e["time"]), "ts": e["time"], "initial": e["initial"],
                        "occluded": e["occluded"], "logged": logged})
        when = time.time() if t is None else t
        self._log_notices(when)
        if hasattr(self.tracker, "vehicles_in"):  # a fixed map / auto grid: the open sessions are the live BUSY count, every frame
            tr = self.tracker
            want = {k: tr.vehicles_in(k) if s["state"] == 1 else 0 for k, s in tr.slots.items() if s["state"] is not None}
            undecided = {k for k, s in tr.slots.items() if s["state"] is None}
            try:
                for kind, k, _ in self.db.sync_open(want, undecided, when):
                    out.append({"slot_id": k, "event": kind, "state": "BUSY" if kind == "entry" else "FREE",
                                "time": self.db.iso(when), "ts": when, "initial": False, "occluded": False, "logged": True})
                self.session_error = ""
            except sqlite3.Error as e:  # a locked / read-only history file: the frame goes on, the check below shows it
                self.session_error = f"{type(e).__name__}: {e}"[:200]
        self._check_sessions(when)
        return FrameResult(vehicles, others, out, mask, list(self.tracker.new_slots), list(self.tracker.retired))

    def set_blocked(self, on, labels=()) -> None:
        """Blocked by object (blocked.py): the detector must report the object classes for the layer to see them, so they join the target
        classes while it is on (the base targets are restored when it goes off). Idempotent: the worker calls it every frame. A monitor with
        no target list returns every class already."""
        if not on and getattr(self, "_targets_base", None) is None:
            return  # never switched on: the targets are exactly what they were
        if getattr(self, "_targets_base", None) is None and self.targets is not None:
            self._targets_base = list(self.targets)
        base = getattr(self, "_targets_base", None)
        if base is None:
            return
        self.targets = base + [l for l in labels if l not in base] if on else list(base)

    def _log_notices(self, when: float):
        """The tracker's notices into the event log (never a session): `encroached` / `encroach_cleared` of the spillage layer and
        `rejected` of the non-vehicle filter. A write that fails (a locked / read-only history file) is reported in .notice_error and
        never stops the frame. Both layers are off by default, and then there is nothing here."""
        tr = self.tracker
        names = getattr(tr, "names", {})
        notices = []
        for e in getattr(tr, "avail_events", ()):
            note = e["note"]
            if e["event"] == "encroached":
                sp = tr.avail.get(e["slot"])
                who = ", ".join(names.get(o, str(o)) if o else "an unplaced vehicle" for o in (sp.by if sp else ()))
                note = f"by {who}" + (f", {round(sp.cover * 100)} % of the bay" if sp else "") + ": free but not usable"
            notices.append({"slot": e["slot"], "event": e["event"], "time": e["time"], "note": note})
        for e in getattr(tr, "block_events", ()):
            note = e["note"]
            if e["event"] == "object_blocked":
                bk = tr.block.get(e["slot"])
                note = f"by {', '.join(bk.by) if bk else 'an object'}: free but not usable"
            notices.append({"slot": e["slot"], "event": e["event"], "time": e["time"], "note": note})
        if getattr(tr, "obstacle_filter", 0):
            notices += self._rejects.update(tr.rejected, when, names)
        try:
            for n in notices:
                self.db.log_notice(n["slot"], n["event"], n["time"], n["note"])
            if notices:
                self.notice_error = ""
        except sqlite3.Error as e:
            self.notice_error = f"{type(e).__name__}: {e}"[:200]

    def _check_sessions(self, when: float):
        """The history against the live count, no extra thread: every frame, the open sessions of the slots whose state is decided
        (UNKNOWN ones keep theirs) against the vehicles the tracker counts BUSY (a capacity slot counts its vehicles). They are
        equal by construction on a fixed map / auto grid (sync_open repairs a difference in the same frame); a difference that stays
        is a failure to write (see .session_error) or, on a learned map (which is only reported, never repaired), a merge / retire
        the log has not followed. .session_check = {"expected", "open", "since": when they began to differ, else None, "at": the frame checked}."""
        tr = self.tracker
        vehicles_in = getattr(tr, "vehicles_in", None)
        expected = sum(vehicles_in(k) if vehicles_in else 1 for k, s in tr.slots.items() if s["state"] == 1)
        undecided = {k for k, s in tr.slots.items() if s["state"] is None}
        try:
            opened = sum(n for k, n in self.db.open_by_slot().items() if k not in undecided)
        except sqlite3.Error as e:
            self.session_error = self.session_error or f"{type(e).__name__}: {e}"[:200]
            return
        since = (self.session_check or {}).get("since") if opened != expected else None
        if opened != expected and since is None:
            since = when
        self.session_check = {"expected": expected, "open": opened, "since": since, "at": when}

    def _split(self, frame, vehicles, now: float) -> list[dict]:
        """Dynamic mode: one whole-frame box over two parked cars, the one in front and the one behind it (Front Parking
        02, 2026-10-22: the white sedan and the white hatchback behind it, one box at 1920 px even at conf 0.05 / NMS
        0.9; the slot learned from it after a map reset held both, the hatchback was never counted) -> the two boxes.
        A tall box (>= SPLIT_MIN_H of the frame) is merged when a second whole-frame pass at SPLIT_IMGSZ boxes the front
        car alone (the same bottom, its top lower by >= 15 % of the box: 1280 px does on 3 of 4 live frames) AND the
        behind crop over the strip above it finds a car there (as wide as half the box, its bottom near the front
        car's top). Then: the front car's box + that car's ("src" "split"). A lone tall car: the second pass gives
        the same box, nothing is split. A result (split or not) is reused for SPLIT_EVERY s while the box stays (IoU
        >= 0.9): one extra pass and crop per half minute, and the car behind is seen on every frame meanwhile."""
        alt, look = getattr(self.detector, "_whole", None), getattr(self.detector, "detect_region", None)
        if alt is None or look is None:
            return vehicles
        h, w = frame.shape[:2]
        self._splits = [c for c in self._splits if now - c["at"] < SPLIT_EVERY]
        out, others = [], None
        for d in vehicles:
            x1, y1, x2, y2 = d["box"]
            bh, poly = y2 - y1, np.float32(d["poly"]) / (w, h)
            if d["state"] != 1 or d.get("src") or bh < SPLIT_MIN_H * h:
                out.append(d)
                continue
            cached = next((c for c in self._splits if iou(c["box"], poly) >= 0.9), None)
            if cached is None:
                if others is None:
                    others = [a for a in alt(frame, SPLIT_IMGSZ) if a["state"] == 1]
                cached = {"box": poly, "at": now, "parts": self._split_one(frame, d, others, look)}
                self._splits.append(cached)
            out += cached["parts"] or [d]
        return out

    def _split_one(self, frame, d, others, look):
        """(front car, car behind) for one merged box d, or None (see _split)."""
        x1, y1, x2, y2 = d["box"]
        bh, bw = y2 - y1, x2 - x1
        fronts = [a for a in others if abs(a["box"][3] - y2) <= 0.08 * bh and a["box"][1] >= y1 + 0.15 * bh and
                  min(a["box"][2], x2) - max(a["box"][0], x1) >= 0.8 * bw]
        if not fronts:
            return None
        front = max(fronts, key=lambda a: a["conf"])
        top = front["box"][1]
        rears = [r for r in look(frame, (x1, y1, x2, top + 0.1 * (top - y1)), conf=self.zoom_conf, behind=True)
                 if y1 + 0.1 * bh <= r["box"][3] <= top + 0.15 * bh and r["box"][1] <= y1 + 0.15 * bh and
                 r["box"][2] - r["box"][0] >= 0.4 * bw and _iou_ios(r["poly"], d["poly"])[1] >= 0.7]
        if not rears:
            return None
        rear = max(rears, key=lambda r: r["conf"])
        return [{**front, "src": "split"}, {**rear, "src": "split"}]

    def _relook(self, frame, k, ok) -> bool | None:
        """The evidence layer's forced re-look at slot k (a locked bay about to be vetoed or released): does a zoom on it, at
        RELOOK_CONF (below the whole-frame threshold), find a vehicle standing in it? Same crop and fit as _zoom. None = no zoom
        possible (detector without detect_region, or zoom off by zoom_max = 0). Called only when a locked bay wants to leave."""
        look, tr = getattr(self.detector, "detect_region", None), self.tracker
        if look is None or not self.zoom_max or k not in tr.slots:
            return None
        h, w = frame.shape[:2]
        poly = tr.slots[k]["poly"]
        (x1, y1), (x2, y2) = poly.min(0) * (w, h), poly.max(0) * (w, h)
        pts = np.float32(poly.reshape(-1, 1, 2) * (w, h))
        return any(ok(d) and iou(np.float32(d["poly"]) / (w, h), poly) >= tr.match_iou and
                   cv2.pointPolygonTest(pts, ((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2), False) >= 0
                   for d in look(frame, (x1, y1, x2, y2), conf=RELOOK_CONF) if d["state"] == 1)

    def _zoom(self, frame, vehicles, ok) -> list[dict]:
        """Vehicles found by the zoom re-check of slots that lost their detection (see the class doc)."""
        look, tr = getattr(self.detector, "detect_region", None), self.tracker
        if look is None or not self.zoom_max:
            return []
        h, w = frame.shape[:2]
        found = [np.float32(d["poly"]) / (w, h) for d in vehicles]
        flagged = tr.hidden | set(getattr(tr, "covered", {}))
        cars = [p for p, d in zip(found, vehicles) if d["state"] == 1]
        # FREE slots only where the vehicles in front hide them (the whole frame can't see a car there), and at the
        # stricter zoom_free_conf: a visible empty bay is never re-checked, so it can't turn BUSY from a weak box
        hidden_free = {k for k, s in tr.slots.items() if s["state"] == 0 and self.zoom_free_conf is not None and
                       _hidden(s["poly"], cars, tr.occlusion_cover)}
        # a proven bay (it has held a car) taken for a phantom, the gap between the cars around it: the car hidden in
        # it gets no box of its own (Front Parking 02, 2026-10-22: the white hatchback behind the white sedan, inside
        # the sedan's box), so it is re-checked too, at the stricter threshold; never zoomed before, it stayed UNKNOWN
        if self.zoom_free_conf is not None:
            hidden_free |= {k for k in getattr(tr, "phantoms", ()) if k in tr.proven}
        # slots that got a box in the whole frame, by the tracker's own one-to-one matching: the sedan's box overlaps
        # the bay of the car hidden behind it (IoU 0.37), but it is the sedan's slot's box, not that bay's
        ids = list(tr.slots)
        boxed = [(p, d) for p, d in zip(found, vehicles) if d["state"] is not None]
        owner = {id(boxed[i][1]): ids[j] for i, j in tr._match([p for p, _ in boxed], ids).items()}  # box -> its slot
        matched = set(owner.values())
        want = [k for k, s in tr.slots.items()  # + a pending entry: its confirming look comes next (see below)
                if (s["state"] == 1 or s["cand"] == 1 or (s["state"] is None and k in flagged) or k in hidden_free)
                and k not in matched]
        if not want:
            return []
        # BUSY slots that lost their box first (each in turn): unconfirmed for hidden_max they would be closed as left
        # (Front Parking 02, 2026-10-22: the hatchback behind the sedan waited for its turn among 6 hidden slots and
        # its BUSY lapsed), and slots with a pending entry (seen once by the zoom: the next look confirms it within
        # the tracker's window, instead of a turn ~30 s later); the hidden FREE / unknown ones share what is left
        self._zoom_turn += 1
        turn = lambda ks: ks[self._zoom_turn % len(ks):] + ks[:self._zoom_turn % len(ks)] if ks else []  # noqa: E731
        busy = [k for k in want if tr.slots[k]["state"] == 1 or tr.slots[k]["cand"] == 1]
        want = (turn(busy) + turn([k for k in want if k not in busy]))[:self.zoom_max]
        added = []
        for k in want:
            poly = tr.slots[k]["poly"]
            x1, y1 = poly.min(0) * (w, h)
            x2, y2 = poly.max(0) * (w, h)
            conf = self.zoom_free_conf if k in hidden_free else self.zoom_conf
            fit = lambda d: (ok(d) and iou(np.float32(d["poly"]) / (w, h), poly) >= tr.match_iou and  # noqa: E731
                             cv2.pointPolygonTest(np.float32(poly.reshape(-1, 1, 2) * (w, h)),
                                                  ((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2),
                                                  False) >= 0)
            fits = [d for d in look(frame, (x1, y1, x2, y2), conf=conf) if fit(d)]
            if not fits and (k in hidden_free or k in flagged):
                # behind a car in front: cut the crop at the slot's ground line, else the two cars are one box
                # (YoloDetector.detect_region behind)
                fits = [d for d in look(frame, (x1, y1, x2, y2), conf=conf, behind=True) if fit(d)]
            if not fits:
                continue
            best = max(fits, key=lambda d: d["conf"])
            # a box already found (the merge), or a piece of one (mostly inside it): that vehicle's, not one for slot
            # k, unless that box swallowed a car behind it (_apart). Front Parking 02, 2026-10-22: slot 24, a strip
            # on the white car of slot 18, kept itself BUSY for 14 min on the zoom's crop of that car's upper part
            # (inside: the whole frame's boxes only; another zoom box is loosely cut itself: the hatchback's includes
            # the silver car's roof, which then would never count)
            twins = [v for v in vehicles + added if len(merge_dets([v, best])) == 1 or
                     (v in vehicles and _iou_ios(v["poly"], best["poly"])[1] >= INSIDE)]
            if not twins or self._apart(k, best, poly, twins, owner, h):
                added.append(best)
                owner[id(best)] = k
        return added

    def _apart(self, k, d, poly, twins, owner, h) -> bool:
        """A zoom box of slot k that repeats or lies inside vehicles already found (twins) is a vehicle of its own
        when that front box has swallowed it: Front Parking 02, 2026-10-22, the white sedan's box (slot 11) also spans
        the white hatchback behind it, so it rises far above the sedan's own slot (top 0.25 vs 0.333), and the
        hatchback's own box (0.26, as wide) is that extra part. So: every vehicle it would merge into is matched to
        a slot and rises above that slot's top by > SWALLOW of its height, and this box reaches above that top too.
        A front box that fits its own slot (the pickup, its upper half cut at a slot behind it) or a lone car's roof
        (starts at its own slot's top) is never split."""
        tr, top = self.tracker, d["box"][1] / h
        for v in twins:
            j = owner.get(id(v))
            if j is None or j == k or j not in tr.slots:
                return False
            front = tr.slots[j]["poly"]
            f_top, f_h = float(front[:, 1].min()), float(np.ptp(front[:, 1]))
            if not (v["box"][1] / h < f_top - SWALLOW * f_h and top < f_top):
                return False
        return bool(twins)

    def state(self) -> dict:
        """Live slot map as JSON: {"camera", "detector", "mode" (dynamic / fixed), "slots": [{"id", "name", "state",
        "hidden", "blocked", "phantom", "duplicate_of" (the slot of the vehicle it lies on, see SlotTracker.covered,
        else None), "poly" (fractions)}], "counts": {"free", "busy", "unknown", "hidden", "phantom", "duplicate",
        "slots", "learning"}}. A FREE slot that isn't visibly free (SlotTracker.unseen: another vehicle stands in it =
        blocked, or it is behind vehicles in front = hidden) is shown and counted UNKNOWN."""
        tr = self.tracker
        names = getattr(tr, "names", {})
        unseen = getattr(tr, "unseen", {})
        shown = getattr(tr, "shown", lambda k: tr.slots[k]["state"])  # = the video banner's states (view())
        slots = [] if tr is None else [{"id": k, "name": names.get(k, str(k)), "state": STATE_NAMES[shown(k)],
                                         "hidden": k in tr.hidden or unseen.get(k) == "hidden",
                                         "blocked": unseen.get(k) == "blocked", "note": unseen.get(k),
                                         "phantom": k in tr.phantoms,
                                         "duplicate_of": tr.covered.get(k),
                                         "parent": getattr(tr, "parent", {}).get(k),
                                         "capacity": getattr(tr, "capacity", {}).get(k, 1),
                                         "count_note": getattr(tr, "load_notes", {}).get(k),
                                         "ground": getattr(tr, "held", {}).get(k),  # fixed maps: share of the matched vehicle's ground on it
                                         "poly": np.round(s["poly"], 4).tolist()}
                                        for k, s in sorted(tr.slots.items())]
        avail = getattr(tr, "avail", None) if getattr(tr, "spillage", 0) else None  # spillage layer on (fixed maps / auto grids)
        block = getattr(tr, "block", None) if getattr(tr, "blocked", 0) else None  # blocked-by-object layer on
        layer = avail is not None or block is not None
        if layer:  # additive: "state" is unchanged, "avail" / "display" say whether the FREE bay is usable
            for s in slots:
                sp = (avail or {}).get(s["id"])
                bk = (block or {}).get(s["id"])
                s["avail"] = "BLOCKED" if bk is not None else "ENCROACHED" if sp is not None else None
                s["display"] = s["avail"] or s["state"]
                s["encroached_by"] = [names.get(o, str(o)) if o else "unplaced vehicle" for o in sp.by] if sp is not None else []
                s["cover"] = sp.cover if sp is not None else None
                if sp is not None and sp.free_m is not None:  # (ground metric only)
                    s["free_m"] = sp.free_m
                if block is not None:
                    s["blocked_by"] = list(bk.by) if bk is not None else []
        vehicles_in = getattr(tr, "vehicles_in", lambda k: 1)
        for s in slots:  # vehicles in the slot: its counted boxes (capacity slots, at least 1 while BUSY), else 1 while BUSY
            s["vehicles"] = vehicles_in(s["id"]) if s["state"] == "BUSY" else 0
        n = {"FREE": 0, "BUSY": 0, "UNKNOWN": 0}  # spaces, not slots: a slot of capacity N is N spaces
        for s in slots:
            if s["state"] == "BUSY":
                n["BUSY"] += s["vehicles"]
                n["FREE"] += s["capacity"] - s["vehicles"]
            else:
                n[s["state"]] += s["capacity"]
        if self.stream != "active":  # stale / frozen picture: nothing is known about any space (the tracker keeps what it had)
            for s in slots:
                s["state"], s["vehicles"], s["hidden"], s["blocked"], s["note"] = "UNKNOWN", 0, False, False, f"stream {self.stream}"
            n = {"FREE": 0, "BUSY": 0, "UNKNOWN": sum(s["capacity"] for s in slots)}
        lost = lost_blocked = 0
        if layer and self.stream == "active":  # a bay a neighbour's vehicle or an object stands in is FREE but not usable
            lost = sum(s["avail"] == "ENCROACHED" for s in slots)  # (an UNKNOWN bay can be encroached on the ground metric: counted here...)
            lost_blocked = sum(s["avail"] == "BLOCKED" for s in slots)
            n["FREE"] -= sum(s["avail"] in ("ENCROACHED", "BLOCKED") and s["state"] == "FREE" for s in slots)  # ...but only a FREE one leaves the free count
        elif layer:  # (stale / frozen picture: nothing is known, so nothing is encroached or blocked either)
            for s in slots:
                s["avail"], s["display"], s["encroached_by"], s["cover"] = None, s["state"], [], None
                if block is not None:
                    s["blocked_by"] = []
        subs = sum(s["parent"] is not None for s in slots)  # sub-bays (each is one more space of its slot)
        return {"camera": self.db.camera, "detector": self.db.detector, "mode": self.mode, "slots": slots,
                "unplaced": list(getattr(tr, "unplaced", ())),  # vehicles that hold no bay yet, with the reason
                **({"rejected": list(tr.rejected)} if getattr(tr, "obstacle_filter", 0) else {}),  # non-vehicle filter on: what it dropped last frame
                "counts": {"free": n["FREE"], "busy": n["BUSY"], "unknown": n["UNKNOWN"],
                           "hidden": sum(s["hidden"] for s in slots), "blocked": sum(s["blocked"] for s in slots),
                           "phantom": 0 if tr is None else len(tr.phantoms),
                           "duplicate": 0 if tr is None else len(tr.covered), "slots": sum(s["capacity"] for s in slots),  # total spaces: a sub-bay or a slot's capacity N adds to it (9 + 1 = 10)
                           "sub_bays": subs, "unplaced": len(getattr(tr, "unplaced", ())),
                           "learning": 0 if tr is None else len(tr.candidates),
                           # spillage on: counts.free no longer holds the encroached bays; "available" = the drivable free spaces
                           **({"encroached": lost, "available": n["FREE"]} if layer else {}),
                           **({"blocked_objects": lost_blocked} if block is not None else {})}}
