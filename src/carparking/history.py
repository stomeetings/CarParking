"""Entry/exit history for a fixed camera: a persistent slot map, debounced FREE/BUSY per slot, and a SQLite log.

Detectors give no stable IDs: box i in one frame is not box i in the next. SlotTracker keeps its own slot map and
matches every frame's detections to it by polygon IoU, so a slot keeps its ID for as long as the map exists (it is
stored in the database, so across restarts too).

* Learning: a detection that matches no slot becomes a candidate; a candidate seen again for >= learn_hold seconds
  becomes a slot. A car driving through moves away from its first box, so it never becomes one.
* Debounce: a slot changes state only after the new state has been observed for >= hold seconds (and in at least
  two processed frames). The event time is when the new state was first seen, not when it was confirmed.
* Absent = FREE: vehicle-only detectors (COCO) never report free bays, so a learned slot without a matching vehicle
  is FREE. Bay models (PKLot OBB) report free bays, so "no detection" there means unknown and keeps the last state.
* Occlusion (low-angle cameras): a slot with no matching vehicle is HIDDEN, not empty, when the vehicle boxes in
  front of it (bottom edge lower in the image = nearer the camera on a ground-plane view) together cover
  >= occlusion_cover (0.6) of it. A hidden slot keeps its state and its FREE countdown does not run; it starts once
  the slot is visible and empty, so the logged exit is "first seen empty" (flagged in the event). Bounded: after
  hidden_max seconds (30) without any vehicle in it, hidden or not, the slot counts as empty and the exit is timed
  at the first frame without the car ("expired" in the event): a car that left a bay half behind the car in front
  doesn't keep it BUSY until a restart.
* Exit buffer: BUSY -> FREE needs exit_hold seconds (longer than hold) of visible emptiness: a person, a car door or
  a detector miss for a minute does not end a parking session.
* Phantoms (dynamic maps, tightly parked rows): a box that is not a vehicle of its own never feeds the map
  (phantom_reason). On a low-angle view a detector sometimes stretches the box of a car down over the car in front
  of it; that box's ground point (bottom centre) then sits on the front car's body, between two parked cars where
  no slot fits, and a slot learned from it shows FREE whenever the stretched box is gone. A box is a phantom when
  - it holds the box of a vehicle behind it, and that box plus the vehicles in front cover almost all of it
    (>= phantom_cover): it is that vehicle's box stretched down over the car in front ('stretched'), or
  - its ground point lies between the ground points of two vehicles parked side by side that are closer together
    than room x their mean width (no space for a slot between them).
  Candidates are learned only where the ground point stays put (within ground_tol of the box size): a vehicle
  resting in a bay, not a box that stretches and shrinks. A learned slot that is not BUSY and is a phantom against
  the current vehicles is UNKNOWN (never FREE). If it has never held a car (not in proven) it is retired after
  retire_hold seconds (.retired; the monitor marks it in the database, its ID is never reused). A real bay whose car
  left can look the same while it is hidden in a dense row, so a proven slot is only kept UNKNOWN.
* Slot NMS (dynamic maps, any camera): two slots are one space learned twice when they overlap by IoU > merge_iou
  (0.5), or by IoU > near_iou (0.35) with their ground points at the same depth (same_depth: within 0.2 x the taller
  one's height; a duplicate of one space stands on the same ground line). They are merged every frame (the proven /
  BUSY / oldest one stays, the other is retired and its open session closed), and a candidate that is such a
  duplicate of a slot (or IoU > dup_iou 0.4) never becomes a second slot. Not a car parked behind another one: on a
  low-angle view it overlaps the front car's slot heavily but stands further back (Front Parking 02: slot 11, the
  white sedan behind the silver car in slot 20, IoU 0.33, 74 % inside it, ground y 0.59 vs 0.86; the live row's
  slots 2 / 5: IoU > 0.35, ground 0.30 x the height apart). Never slots that have held two vehicles in the same frame
  (.distinct), whatever their overlap. Likewise a candidate whose overlapping slot is held by ANOTHER vehicle box in
  the same frame is a second vehicle, not a duplicate: it is learned and the pair recorded as distinct. Unless the two
  boxes are one vehicle (same_vehicle: IoU >= 0.5 at one depth, a split box): then it stays a duplicate, and such a
  pair never counts as two vehicles at once.
* One vehicle box, one slot (dynamic maps): a detection occupies only the slot it matches (greedy one-to-one IoU,
  best pairs first = each box its highest-IoU slot); the detector's class-agnostic NMS already leaves one box per car.
  A detection that matched no slot (its box changed shape) may occupy ONE unmatched slot: the one it overlaps most
  among those holding its centre, overlapping it with IoU >= share_cover (0.25) or covering >= claim_cover (30 %)
  of its area (.shared). A neighbour's box spilling over a bay
  (the maroon car's box covers 44 % of the motorbike's bay next to it, IoU 0.14) never keeps that bay BUSY, so the
  bay frees itself once its own vehicle leaves. Vehicles in front of a slot are occlusion (HIDDEN, bounded by
  hidden_max), never occupancy.
* Shown state (dynamic maps, any camera; display only, the history is unchanged): view() and the monitor's state()
  both use shown(), so the video banner, the overlay and the dashboard's counts are always the same numbers.
  One vehicle box counts once: a BUSY slot that no box matched in this frame, while a vehicle matched to ANOTHER slot
  overlaps it (IoU > vehicle_iou 0.2) or stands in it (stands_in), is shown UNKNOWN ("on the vehicle of slot N"), not
  a second BUSY (Front Parking 02: the pickup matched slot 6 and stands in slot 5). Unless it is hidden behind
  vehicles in front (a car may stand there unseen) or an unmatched box stands in it (.shared).
  FREE needs a visibly empty ground: judged on the slot's ground footprint (the bottom 40 % of its box, where a
  vehicle's wheels stand; footprint()), a slot in state FREE is shown UNKNOWN (.unseen) while
  - "blocked": another vehicle stands in it: a vehicle box's ground point (bottom centre, where its wheels are) lies
    in the slot's floor (the bottom 40 % of its box, inner 80 % of its width). In an unmarked lot cars park at
    slightly different spots, so slots learned from earlier cars overlap: the car now standing across them counts
    once (in its own slot), the others are neither FREE nor BUSY (Front Parking 02: the pickup's wheels stand in
    slots 5 and 18, the hatchback's in slot 2).
  - "hidden": vehicles nearer the camera cover > hidden_ground (80 %) of its footprint, also after hidden_max (its
    session is closed then as before, timed at the departure; it just isn't shown FREE while nobody can see its
    ground): the far end of a low-angle row (slots 13 / 22, wholly inside the pickup's box in front of them).
  - "overlaps a vehicle": a vehicle box not in front of it (at its depth or behind) covers > ground_share (40 %) of
    its footprint: a car stands on that ground, no room for another one. A neighbour's box spilling over the side
    of a bay (30 %) leaves it FREE. Otherwise the slot is FREE once its exit buffer has run.
* Tiny strays (min_size, is_tiny): a vehicle box far too small for a vehicle at its depth (< 15 % of the median
  area of the slots / vehicles standing there) is dropped before matching and learning; a slot that small is
  retired at once, proven or not, and its open session closed. So no slot is BUSY without a vehicle of its size.
* Hard wall (area): a slot or detection outside the parking area (monitor.in_area: centre and ground point inside,
  and >= 90 % of the box) is set aside: never learned, matched, displayed or logged. set_area() re-applies a
  changed area while running; slots set aside come back when it grows again.
Polygons are stored as fractions of the frame size, so main and sub stream (different resolutions) share one map.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, tzinfo

import cv2
import numpy as np

from carparking.dynamic_iou import iou, overlap


# a box that overlaps the vehicle box a BUSY slot last had by at least this IoU is that vehicle (SlotTracker._owned)
OWN_IOU = 0.3

# Display / counting thresholds of the shown state (SlotTracker.shown, dynamic maps), adjustable at run time from
# the dashboard's Settings (the worker reads the "tuning" option every step and applies it with SlotTracker.tune). They
# only change what is shown and counted (FREE / BUSY / UNKNOWN), never the slot map or the history, so every change
# is reversible. {name: (default, min, max)}
DISPLAY_TUNING = {
    "hidden_ground": (0.8, 0.5, 1.0),  # vehicles in front cover more of a FREE slot's ground footprint: UNKNOWN
    "ground_share": (0.4, 0.1, 0.9),   # a vehicle at its depth / behind covers more of the footprint: not FREE
    "vehicle_iou": (0.2, 0.05, 0.9),   # BUSY without its own box, a vehicle of another slot overlaps more: UNKNOWN
    # fixed maps / auto grids only (FixedSlotTracker): a vehicle standing just outside a bay, or matched by its footprint alone, holds
    # the bay (BUSY) only when this share of its ground footprint lies inside it: a tyre or bumper over the line does not
    "bay_share": (0.5, 0.1, 0.95),
    # auto grid only (FixedSlotTracker with `auto`): a capacity pin also counts a vehicle no bay holds when the middle of its ground
    # footprint lies outside the pin by at most this share of the vehicle's height (a pin drawn a little tight); 0 = strictly inside.
    # A Fixed map is always strict
    "pin_reach": (0.25, 0.0, 0.5),
}


def tuning_values(raw=None) -> dict[str, float]:
    """{name: value} for DISPLAY_TUNING: raw (a dict, e.g. the stored option) where valid and in range, else the
    default."""
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for name, (default, lo, hi) in DISPLAY_TUNING.items():
        try:
            v = float(raw.get(name, default))
        except (TypeError, ValueError):
            v = default
        out[name] = min(max(v, lo), hi) if v == v else default  # NaN -> default
    return out


def _hidden(slot, boxes, cover: float, margin: float = 0.01, grid: int = 30) -> bool:
    """The vehicle boxes in front of the slot (bottom edge lower in the frame) together cover >= cover of it: the
    camera can't see whether it is empty. Together, not one box: in a low-angle row every slot is half covered by
    the car in front (Front Parking 02: 42-50 %), so a low share would hide every bay for good."""
    x1, y1, x2, y2 = _bounds(slot)
    front = [_bounds(b) for b in boxes if b[:, 1].max() >= y2 + margin]
    if not front:
        return False
    xs, ys = np.meshgrid(np.linspace(x1, x2, grid), np.linspace(y1, y2, grid))
    m = np.zeros_like(xs, bool)
    for bx1, by1, bx2, by2 in front:
        m |= (xs >= bx1) & (xs <= bx2) & (ys >= by1) & (ys <= by2)
    return m.mean() >= cover


def footprint(slot, band: float = 0.4):
    """The slot's ground footprint (fractions): the bottom band of its box, where a vehicle parked in it stands."""
    x1, y1, x2, y2 = _bounds(slot)
    return x1, y2 - band * (y2 - y1), x2, y2


def _share(region, boxes, grid: int = 30) -> float:
    """Share of region (x1, y1, x2, y2) covered by the boxes (fractions) together."""
    if not boxes:
        return 0.0
    x1, y1, x2, y2 = region
    xs, ys = np.meshgrid(np.linspace(x1, x2, grid), np.linspace(y1, y2, grid))
    m = np.zeros_like(xs, bool)
    for b in boxes:
        bx1, by1, bx2, by2 = _bounds(b)
        m |= (xs >= bx1) & (xs <= bx2) & (ys >= by1) & (ys <= by2)
    return float(m.mean())


def stands_in(slot, box, band: float = 0.4, inset: float = 0.1, below: float = 0.05) -> bool:
    """The vehicle box's ground point (bottom centre) lies in the slot's floor: the bottom band of its box (fractions),
    inset from its left and right edges. A car parked behind the slot stands above that band (further back), a car in
    front of it below, a neighbour beside it outside its width."""
    gx, gy = _ground(box)
    x1, y1, x2, y2 = _bounds(slot)
    w, h = x2 - x1, y2 - y1
    return x1 + inset * w <= gx <= x2 - inset * w and y2 - band * h <= gy <= y2 + below * h


def _occupied_by(slot, box, min_iou: float, min_cover: float | None = None, depth: float = 0.3) -> float:
    """How strongly box (fractions) stands in slot: its IoU with the slot if any of these holds, else 0:
    the box centre lies inside the slot, or the box stands in the slot's column at its depth (centre between the
    slot's left and right edges, bottom edges within depth x the slot's height) and its IoU is >= min_iou or it covers
    >= min_cover of the slot's area (a box of the slot's own car that changed shape: taller, cut, shifted). Not a
    vehicle parked in FRONT of a bay (covers 87 % of it on a low-angle view, IoU 0.27: occlusion, HIDDEN) nor one
    BESIDE it (covers 33-44 % at the same depth): both would keep the bay BUSY after its own car left.
    Only for a box that matched no slot (SlotTracker._claims), and it then holds one slot only (the best score): a
    neighbour's box spilling over a bay never holds it while its own car matches its own slot (Front Parking 02: the
    maroon car's box covers 44 % of the motorbike's bay, IoU 0.14, centre outside)."""
    x1, y1, x2, y2 = _bounds(box)
    v, cover = overlap(slot, box)
    centre_in = cv2.pointPolygonTest(np.float32(slot).reshape(-1, 1, 2), ((x1 + x2) / 2, (y1 + y2) / 2), False) >= 0
    sx1, sy1, sx2, sy2 = _bounds(slot)
    level = abs(y2 - sy2) <= depth * (sy2 - sy1) and sx1 <= (x1 + x2) / 2 <= sx2
    return v if centre_in or (level and (v >= min_iou or (min_cover is not None and cover >= min_cover))) else 0.0


def _greedy(polys_a, polys_b, threshold: float) -> dict[int, int]:
    """One-to-one matching by IoU, best pairs first: {index in a: index in b} for pairs >= threshold."""
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(polys_a) for j, b in enumerate(polys_b)), reverse=True)
    out, used = {}, set()
    for v, i, j in pairs:
        if v < threshold:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def _bounds(p):
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def _ground(p):
    """Bottom centre of the box: where the vehicle stands, nearest to the camera."""
    x1, _, x2, y2 = _bounds(p)
    return (x1 + x2) / 2, y2


def phantom_reason(poly, others, cover: float = 0.8, room: float = 1.5, depth: float = 0.15,
                   margin: float = 0.05, inner: float = 0.7, grid: int = 40) -> str | None:
    """poly (fractions): a detection or a learned slot; others: the other vehicle boxes of the same frame.
    -> why poly is not a vehicle of its own ('stretched' / 'no room'), None if it may be one.
    'stretched': poly holds the box of a vehicle behind it (>= inner of that box inside poly) and, together with the
    vehicles in front of it, that box covers >= cover of poly: poly is that vehicle's box stretched down over the car
    in front. Only boxes inside poly can explain it: rectangles overstate occlusion (a real car's visible strip often
    lies inside the box of the car behind it, whose box doesn't fit inside the real car's)."""
    x1, y1, x2, y2 = _bounds(poly)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    gx, gy = _ground(poly)
    boxes = [_bounds(o) for o in others]
    front = [b for b in boxes if b[3] > y2 + margin * h]  # bottom lower = nearer the camera
    # inside poly and behind it (bottom higher), or beside it (same depth) but a different object: a duplicate box
    # of the same car (IoU >= 0.5 at the same depth) explains nothing
    inside = [b for b, o in zip(boxes, others) if overlap(o, poly)[1] >= inner and
              (b[3] < y2 - margin * h or (abs(b[3] - y2) <= margin * h and iou(poly, o) < 0.5))]
    xs, ys = np.meshgrid(np.linspace(x1, x2, grid), np.linspace(y1, y2, grid))

    def hit(bs):
        m = np.zeros_like(xs, bool)
        for bx1, by1, bx2, by2 in bs:
            m |= (xs >= bx1) & (xs <= bx2) & (ys >= by1) & (ys <= by2)
        return m

    visible = ~hit(front)
    # >= 10% visible: a box almost wholly behind others can't be judged (the car hiding it may be the phantom)
    if inside and visible.mean() >= 0.1 and (~visible | hit(inside)).mean() >= cover:
        return "stretched"
    side = [(_ground(o), _bounds(o)[2] - _bounds(o)[0]) for o in others]
    side = [(g, bw) for g, bw in side if abs(g[1] - gy) <= depth * h]  # same depth = the same row
    for i, (ga, wa) in enumerate(side):
        for gb, wb in side[i + 1:]:
            if min(ga[0], gb[0]) < gx < max(ga[0], gb[0]) and abs(ga[0] - gb[0]) < room * (wa + wb) / 2:
                return "no room"
    return None


def duplicate_of(poly, slots: dict, dup_iou: float = 0.4, inside: float = 0.8, depth: float = 0.1,
                 near_iou: float | None = 0.35):
    """The existing slot poly (fractions) duplicates, or None: IoU > dup_iou with it, or IoU > near_iou at the same
    depth (same_depth), or one of the two lies >= inside within the other with their ground points at the same depth
    (within depth x the taller one's height): a bike's bay inside the box of the car now parked there, or a narrower
    box of the same car. Highest IoU first."""
    best, best_v = None, 0.0
    gy, h = _ground(poly)[1], _bounds(poly)[3] - _bounds(poly)[1]
    for k, p in slots.items():
        v, slot_in = overlap(p, poly)  # IoU, share of the slot inside poly
        if v > dup_iou or (near_iou is not None and v > near_iou and same_depth(p, poly)) or (
                v > 0 and (slot_in >= inside or overlap(poly, p)[1] >= inside) and
                abs(_ground(p)[1] - gy) <= depth * max(h, _bounds(p)[3] - _bounds(p)[1])):
            if v > best_v:
                best, best_v = k, v
    return best


def covered_by(poly, boxes, share: float = 0.6, margin: float = 0.01):
    """Index of the vehicle box (fractions) whose body poly stands on, or None: >= share of poly lies inside the box,
    poly's ground point is inside it too and the box stands at least as near the camera. A second slot over one car
    (Front Parking 02, 2026-10-20: the maroon car's old bay is 74 % inside the white sedan parked further forward,
    IoU 0.27). A real bay behind a parked car is only partly covered by it (42-52 % on that camera). Most covered
    first."""
    gx, gy = _ground(poly)
    best, best_v = None, share
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = _bounds(b)
        if x1 <= gx <= x2 and y1 <= gy <= y2 and y2 >= gy - margin:
            v = overlap(poly, b)[1]
            if v >= best_v:
                best, best_v = i, v
    return best


def same_depth(a, b, depth: float = 0.2) -> bool:
    """The ground points (bottom centres) of two boxes / slots (fractions) lie at the same depth: within depth x the
    taller one's height."""
    h = max(_bounds(a)[3] - _bounds(a)[1], _bounds(b)[3] - _bounds(b)[1])
    return abs(_ground(a)[1] - _ground(b)[1]) <= depth * h


def same_vehicle(a, b, min_iou: float = 0.5, depth: float = 0.25) -> bool:
    """Two vehicle boxes (fractions) of ONE vehicle: IoU >= min_iou with their ground points at the same depth (within
    depth x the taller box's height), e.g. a car split into two boxes the detector's NMS missed. Two real vehicles
    never overlap that much at one depth (Front Parking 02: at most IoU 0.39 side by side); a car parked behind
    another stands at another depth (silver car / white sedan: ground y 0.86 vs 0.57)."""
    h = max(_bounds(a)[3] - _bounds(a)[1], _bounds(b)[3] - _bounds(b)[1])
    return iou(a, b) >= min_iou and abs(_ground(a)[1] - _ground(b)[1]) <= depth * h


def dedupe_slots(slots: dict, merge_iou: float = 0.5, rank=None, apart=(), near_iou: float | None = 0.35) -> dict[int, int]:
    """Slot NMS: {dropped id: kept id} for slots overlapping a better one by IoU > merge_iou, or by IoU > near_iou at
    the same depth (same_depth). rank(id) -> sort key, best first (default: the oldest, lowest ID); every slot is
    compared with the slots kept so far. apart: pairs (frozensets) that are two spaces whatever their overlap (they
    have held two vehicles at once): never merged."""
    kept, out = [], {}
    for k in sorted(slots, key=rank or (lambda k: k)):
        twins = {j: v for j in kept if frozenset((k, j)) not in apart
                 if (v := iou(slots[k], slots[j])) > merge_iou or
                 (near_iou is not None and v > near_iou and same_depth(slots[k], slots[j]))}
        j = max(twins, key=twins.get, default=None)
        if j is not None:
            out[k] = j
        else:
            kept.append(k)
    return out


def _area(p) -> float:
    x1, y1, x2, y2 = _bounds(p)
    return float((x2 - x1) * (y2 - y1))


def is_tiny(poly, refs, ratio: float = 0.15, depth: float = 0.5) -> bool:
    """poly (fractions) is far too small for a vehicle at its depth: its area is < ratio x the median area of the
    reference boxes (other slots of the map, current vehicles) standing at the same depth (ground points within
    depth x the reference's height) that aren't the same object (IoU < 0.3). No reference at that depth = can't
    judge = False. Relative, not absolute: far cars are small too (Front Parking 02: the far car's slot is 0.55 % of
    the frame; the stray box by the stairs is 0.14 %, 6 % of the cars at its depth; the smallest real ratio, the
    dark car behind the SUV, is 28 %)."""
    gy = _ground(poly)[1]
    at_depth = [_area(r) for r in refs
                if abs(_ground(r)[1] - gy) <= depth * (_bounds(r)[3] - _bounds(r)[1]) and iou(poly, r) < 0.3]
    return bool(at_depth) and _area(poly) < ratio * float(np.median(at_depth))


class SlotTracker:
    """update(dets, frame_shape) -> state changes [{"slot", "state", "time", "initial", "occluded"}] and new slots in
    .new_slots; .hidden = slots occluded in the last frame.
    initial=True: the slot's first known state (car already parked / bay already empty when tracking started).
    occluded=True on an exit: the slot was hidden at some point since it turned BUSY, so the car may have left earlier
    than the logged time. exit_hold=None: same as hold. .phantoms: slots held UNKNOWN in the last frame as phantoms,
    .retired: slots removed in the last frame (phantoms for retire_hold seconds, None = learn_hold).
    suppress_phantoms=False (fixed maps): every detection counts, no slot is retired. .shared: slots observed BUSY
    in the last frame through an unmatched vehicle (_claims; share_cover=None: off, as in fixed maps)."""

    def __init__(self, slots: dict[int, np.ndarray] | None = None, hold: float = 10.0, learn_hold: float = 10.0,
                 match_iou: float = 0.3, absent_free: bool = False, learn: bool = True,
                 exit_hold: float | None = None, occlusion_cover: float = 0.6, hidden_max: float | None = 30.0,
                 suppress_phantoms: bool = True,
                 phantom_cover: float = 0.8, room: float = 1.5, ground_tol: float = 0.2,
                 retire_hold: float | None = None, proven=None, share_cover: float | None = 0.25, area=None,
                 min_size: float | None = 0.15, dup_iou: float | None = 0.4, merge_iou: float | None = 0.5,
                 near_iou: float | None = 0.35,
                 grow: float = 1.25, cover_share: float = 0.6, cover_hold: float = 60.0, distinct=None,
                 claim_cover: float | None = 0.30, exit_frames: int = 0, seen_window: float = 15.0,
                 vehicle_iou: float = DISPLAY_TUNING["vehicle_iou"][0],
                 hidden_ground: float = DISPLAY_TUNING["hidden_ground"][0],
                 ground_share: float = DISPLAY_TUNING["ground_share"][0],
                 bay_share: float = DISPLAY_TUNING["bay_share"][0],
                 pin_reach: float = DISPLAY_TUNING["pin_reach"][0]):
        self.hold, self.learn_hold, self.match_iou, self.absent_free, self.learn = (
            hold, learn_hold, match_iou, absent_free, learn)
        self.exit_hold = hold if exit_hold is None else exit_hold
        # a pending BUSY of a mostly hidden car survives frames where it isn't seen for this long, and a state first
        # seen this soon after the start is "initial" (already there): at least seen_window, so a short entry hold
        # (fast entries) neither loses hidden cars seen on some frames only nor logs parked cars as new arrivals
        self.window = max(hold, seen_window)
        self.occlusion_cover, self.hidden_max = occlusion_cover, hidden_max
        self.suppress_phantoms, self.phantom_cover, self.room, self.ground_tol = (
            suppress_phantoms, phantom_cover, room, ground_tol)
        self.retire_hold = learn_hold if retire_hold is None else retire_hold
        self.proven = set(proven or ())  # slots that have held a car (BUSY): never retired, only kept UNKNOWN
        self.share_cover, self.claim_cover = share_cover, claim_cover
        # flicker guard: BUSY -> FREE also needs this many consecutive processed frames without a vehicle in the slot
        # (whole frame or zoom), on top of exit_hold seconds; a single missed frame restarts the count
        self.exit_frames = exit_frames
        self.shared: set[int] = set()
        self.min_size = min_size if suppress_phantoms else None  # is_tiny ratio; None = off (fixed maps)
        self.retire_notes: dict[int, str] = {}  # why a slot in .retired went (the session note in the history)
        # id -> {"poly" (fractions), "state" None/0/1, "cand" pending state, "since" first time cand was seen,
        #        "occluded" hidden at least once since the last confirmed state, "phantom" since when it is one}
        self.slots = {k: {"poly": np.float32(p), "state": None, "cand": None, "since": 0.0, "occluded": False}
                      for k, p in (slots or {}).items()}
        self.candidates: list[dict] = []  # {"poly", "state", "first", "last", "n", "ground"}
        self.next_id = max(self.slots, default=0) + 1
        self.started = None
        self.new_slots: list[int] = []
        self.hidden: set[int] = set()
        self.phantoms: set[int] = set()
        self.unseen: dict[int, str] = {}  # slots shown UNKNOWN, with the reason (see shown() and the module doc)
        self.vehicle_iou = vehicle_iou
        # shown FREE needs its ground: UNKNOWN when vehicles in front cover > hidden_ground of the footprint, or a
        # vehicle at its depth / behind covers > ground_share of it
        self.hidden_ground, self.ground_share = hidden_ground, ground_share
        self.bay_share = bay_share  # (matching on a fixed map, see FixedSlotTracker._match; unused by learned maps)
        self.pin_reach = pin_reach  # (capacity pins of an auto grid, see FixedSlotTracker._count_loads; unused otherwise)
        # 1 = each slot's observation goes through _evidence_obs before the debounce below (post-demo item 4, evidence.py); only
        # FixedSlotTracker implements it. 0 = this class runs exactly as before: nothing of it is evaluated
        self.evidence_filter = 0
        self.retired: list[int] = []
        self.dropped = 0  # phantom detections ignored in the last frame
        self.outside: dict[int, dict] = {}  # slots outside the parking area: kept (IDs, state), never shown or matched
        # duplicates (dynamic maps; fixed maps are drawn by hand): a candidate overlapping a slot (duplicate_of,
        # IoU > dup_iou) never becomes a new slot, a larger one (>= grow x the area: a car in a bike's bay) replaces
        # the slot's box (.updated_slots, same ID); slots overlapping each other by IoU > merge_iou are merged (every
        # frame), unless they have held two vehicles at once (.distinct)
        self.dup_iou, self.grow = (dup_iou, grow) if suppress_phantoms else (None, grow)
        self.merge_iou = merge_iou if suppress_phantoms else None
        self.near_iou = near_iou  # IoU above which two slots / boxes at the same depth are one space
        # a slot with no vehicle of its own standing on another vehicle's body (covered_by, >= cover_share) is that
        # vehicle's duplicate: never FREE nor logged, merged into the vehicle's slot after cover_hold (.covered)
        self.cover_share = cover_share if suppress_phantoms else None
        self.cover_hold = cover_hold
        self.covered: dict[int, int | None] = {}  # slot -> slot of the vehicle covering it in the last frame
        # overlapping slots that have held two separate vehicles in one frame: two spaces (a car behind another one
        # on a low-angle view lies wholly inside its box), never duplicates; .new_distinct: pairs found in the last
        # frame (HistoryDB.add_distinct_pairs)
        self.distinct = {frozenset(p) for p in (distinct or ())}
        self.new_distinct: list[tuple[int, int]] = []
        self.updated_slots: list[int] = []
        self.set_area(area)
        self.merged = self._dedupe()  # {dropped: kept} in the loaded map; the monitor retires them in the database

    def set_area(self, area) -> list[int]:
        """Hard wall: area(poly in fractions) -> True if a box belongs to the parking area (None = everywhere).
        Slots outside it are set aside in .outside (not displayed, matched, learned or logged; their open session,
        if any, stays open) and come back if the area grows again; detections outside it are ignored.
        -> ids of the slots now outside."""
        self.area = area
        every = {**self.slots, **self.outside}
        self.slots = {k: s for k, s in sorted(every.items()) if area is None or area(s["poly"])}
        self.outside = {k: s for k, s in sorted(every.items()) if k not in self.slots}
        self.next_id = max(self.next_id, max(every, default=0) + 1)
        self.hidden &= set(self.slots)
        return list(self.outside)

    def _dedupe(self) -> dict[int, int]:
        """Merge overlapping slots (dedupe_slots, IoU > merge_iou; pairs in .distinct never): keeps the proven one (it
        has held a car), then the BUSY one, then the oldest; the others are removed (.retired, with the kept ID in
        .retire_notes)."""
        if self.merge_iou is None:
            return {}
        rank = lambda k: (k not in self.proven, self.slots[k]["state"] != 1, k)  # noqa: E731
        out = dedupe_slots({k: s["poly"] for k, s in self.slots.items()}, self.merge_iou, rank, self.distinct,
                           self.near_iou)
        for k, j in out.items():
            del self.slots[k]
            self.retired.append(k)
            self.retire_notes[k] = f"duplicate slot merged into slot {j} (same space)"
        self.hidden -= set(out)
        return out

    def _covering(self, k, cars, car_idx, hit, ids) -> int | None:
        """The slot whose vehicle slot k lies on (covered_by), if k is its duplicate: that vehicle matches a slot of its
        own, and the two slots have never held two separate vehicles at once (.distinct). Only for slots that are
        not BUSY (see update)."""
        if self.cover_share is None:
            return None
        c = covered_by(self.slots[k]["poly"], cars, self.cover_share)
        owner = None if c is None or car_idx[c] not in hit else ids[hit[car_idx[c]]]
        return None if owner is None or {k, owner} in self.distinct else owner

    def _phantom(self, poly, others, room=None) -> str | None:
        return phantom_reason(poly, others, self.phantom_cover, self.room if room is None else room)

    def _real(self, obs):
        """Drop phantom vehicle boxes. The 'stretched' test runs first, so a stretched box can't make a real car
        beside it look squeezed."""
        cars = [i for i, o in enumerate(obs) if o[1] == 1]
        others = lambda i, keep: [obs[j][0] for j in keep if j != i]  # noqa: E731
        keep = [i for i in cars if not self._phantom(obs[i][0], others(i, cars), room=0)]
        keep = [i for i in keep if not self._phantom(obs[i][0], others(i, keep))]
        self.dropped = len(cars) - len(keep)
        return [o for i, o in enumerate(obs) if o[1] != 1 or i in keep]

    def _claims(self, obs, hit, free_ids) -> set[int]:
        """Slots occupied by vehicles that matched no slot (their box changed shape, IoU < match_iou): each such box
        occupies at most ONE slot, the unmatched one it overlaps most among those holding its centre or with IoU >
        share_cover. A box matched to a slot occupies that slot only, so a neighbour's box spilling over a bay never
        keeps it BUSY after its own vehicle left."""
        if self.share_cover is None:
            return set()
        claimed = set()
        for i, (box, state) in enumerate(obs):
            if state != 1 or i in hit:
                continue
            scores = {k: _occupied_by(self.slots[k]["poly"], box, self.share_cover, self.claim_cover) for k in free_ids
                      if k not in claimed}
            best = max(scores, key=scores.get, default=None)
            if best is not None and scores[best] > 0:
                claimed.add(best)
        return claimed

    def _full_size(self, obs):
        """Tiny boxes (is_tiny against the slot map and the other vehicles): a vehicle box far too small for its
        depth is a stray (a railing, a mirror, a piece of a car) and is dropped before matching and learning. A slot
        that small is retired at once, BUSY or not, proven or not: its size never changes, so waiting changes
        nothing, and a tiny box can't hold a car (its open session is closed)."""
        slots = {k: s["poly"] for k, s in self.slots.items()}
        polys = [o[0] for o in obs]
        keep = [o for i, o in enumerate(obs)
                if o[1] != 1 or not is_tiny(o[0], polys[:i] + polys[i + 1:] + list(slots.values()), self.min_size)]
        self.dropped += len(obs) - len(keep)
        cars = [o[0] for o in keep if o[1] == 1]
        for k, p in slots.items():
            if is_tiny(p, [q for j, q in slots.items() if j != k and j in self.slots] + cars, self.min_size):
                del self.slots[k]
                self.retired.append(k)
                self.retire_notes[k] = "tiny slot retired (far too small for a vehicle at its depth)"
        return keep

    def update(self, dets: list[dict], frame_shape, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        self.started = now if self.started is None else self.started
        self.new_slots, self.updated_slots, self.retired, self.retire_notes, self.new_distinct = [], [], [], {}, []
        h, w = frame_shape[:2]
        obs = [(np.float32(d["poly"]) / (w, h), d["state"]) for d in dets]
        conf_of = {id(o): d.get("conf", 1.0) for o, d in zip(obs, dets)} if self.evidence_filter else None  # (the filters below keep the tuples)
        if self.area is not None:  # outside the hard wall: never matched, learned or counted as a neighbour
            obs = [o for o in obs if self.area(o[0])]
        if self.suppress_phantoms:
            obs = self._real(obs)
        if self.min_size:
            obs = self._full_size(obs)
        ids = list(self.slots)
        hit = self._match([o[0] for o in obs], ids)
        seen = {ids[j]: obs[i][1] for i, j in hit.items()}
        if self.cover_share is not None:
            busy = sorted(k for k, v in seen.items() if v == 1)
            box_of = {ids[j]: obs[i][0] for i, j in hit.items()}
            for n, a in enumerate(busy):
                for b in busy[n + 1:]:
                    if {a, b} not in self.distinct and overlap(self.slots[a]["poly"], self.slots[b]["poly"])[1] > 0 \
                            and not same_vehicle(box_of[a], box_of[b]):
                        self.distinct.add(frozenset((a, b)))
                        self.new_distinct.append((a, b))
        if self.learn:
            self._learn([o for i, o in enumerate(obs) if i not in hit], now,
                        taken={ids[j]: obs[i][0] for i, j in hit.items()})
        # slot NMS every frame: a duplicate (two slots of one space, never two vehicles at once) goes as soon as it
        # exists, not only at the next restart
        merged = self._dedupe()
        hit = {i: j for i, j in hit.items() if ids[j] not in merged}
        seen = {k: v for k, v in seen.items() if k not in merged}
        ev_box = {ids[j]: (obs[i][0], conf_of.get(id(obs[i]), 1.0)) for i, j in hit.items()} if self.evidence_filter else None
        # one vehicle, one BUSY slot: a box that would make another slot busy while it is the vehicle of a BUSY slot
        # that has no box of its own in this frame keeps that slot instead (dynamic maps)
        claimed = self._owned(obs, hit, ids, seen) if self.suppress_phantoms else set()
        car_idx = [i for i, o in enumerate(obs) if o[1] == 1]
        cars = [obs[i][0] for i in car_idx]
        events, self.hidden, self.phantoms, self.covered = [], set(), set(), {}
        self.shared = self._claims(obs, hit, [k for k in ids if k not in seen and k in self.slots])
        for k in ids:  # slots learned in this frame already carry their pending state
            if k not in self.slots:  # merged into another slot just now
                continue
            s = self.slots[k]
            o = seen.get(k, 0 if self.absent_free else None)
            if k in claimed:
                o = None  # the vehicle is another BUSY slot's: no evidence for this one (never an entry, never FREE)
            if k in seen:
                s.pop("phantom", None)  # a vehicle matches it again
            elif k in self.shared:
                o = 1  # an unmatched vehicle stands in it (centre inside or IoU > share_cover): occupied
                s.pop("phantom", None)
            elif s["state"] != 1 and (owner := self._covering(k, cars, car_idx, hit, ids)) is not None:
                # standing on the body of another vehicle: a duplicate of that vehicle's slot, not a space of its own.
                # Not while BUSY: a car that left a bay now behind a van in front frees it on the hidden_max timer
                s.setdefault("covered", now)
                if now - s["covered"] >= self.cover_hold:
                    del self.slots[k]
                    self.retired.append(k)
                    self.retire_notes[k] = f"duplicate slot merged into slot {owner} (it lies on that vehicle)"
                    continue
                self.covered[k] = owner
                o, s["cand"] = None, None  # no countdown; UNKNOWN, never FREE on top of a car
                s["state"] = None
            elif s["state"] != 1 and self.suppress_phantoms and ("phantom" in s or self._phantom(s["poly"], cars)):
                # learned from a phantom box (e.g. between two tightly parked cars): never FREE, retired in time.
                # Sticky until a detection matches it: the test is borderline on some frames
                s.setdefault("phantom", now)
                if k not in self.proven and now - s["phantom"] >= self.retire_hold:
                    del self.slots[k]
                    self.retired.append(k)
                    continue
                self.phantoms.add(k)
                o, s["state"], s["cand"] = None, None, None
            if k not in self.covered:
                s.pop("covered", None)
            s["empty_n"] = 0 if k in seen or k in self.shared else s.get("empty_n", 0) + 1
            if k in seen or k in self.shared:
                s["gone"] = None  # a vehicle is in it now
                s.pop("expired", None)
            elif k not in self.phantoms and k not in self.covered:
                s["gone"] = s.get("gone") or now  # first frame without a vehicle in it
                expired = self.hidden_max is not None and now - s["gone"] >= self.hidden_max
                if not expired and _hidden(s["poly"], cars, self.occlusion_cover):
                    o = None  # mostly behind vehicles in front: not evidence of an empty slot, keeps its state
                    self.hidden.add(k)
                    s["occluded"] = s["occluded"] or s["state"] == 1
                elif expired and o == 0 and s["cand"] != 0:
                    # no vehicle in it for hidden_max: hidden or not, it is empty; the FREE countdown runs from the
                    # first frame without the car, so the exit is timed at the departure, not at the expiry
                    s["cand"], s["since"], s["expired"] = 0, s["gone"], True
            if o == 1:
                s["seen_at"] = now
            if self.evidence_filter:  # the filtered observation feeds the debounce below (everything above judged the raw one)
                o = self._evidence_obs(k, s, o, *ev_box.get(k, (None, None)), now)
            if (o is None and (k in self.hidden or k in self.phantoms) and s["cand"] == 1 and
                    now - s.get("seen_at", now) <= self.window):
                pass  # a car mostly hidden is seen on some frames only (the zoom's turn): those frames don't restart
                # a pending BUSY
            elif o is None or o == s["state"]:
                s["cand"] = None  # an occlusion also restarts a pending FREE countdown
            elif o != s["cand"]:
                s["cand"], s["since"] = o, now
            elif now - s["since"] >= (self.exit_hold if s["state"] == 1 and o == 0 else self.hold) and not (
                    s["state"] == 1 and o == 0 and s["empty_n"] < self.exit_frames):
                initial = s["state"] is None and s["since"] - self.started < self.window
                events.append({"slot": k, "state": o, "time": s["since"], "initial": initial,
                               "occluded": s["occluded"], "expired": s.pop("expired", False)})
                s["state"], s["cand"], s["occluded"] = o, None, False
                if o == 1:
                    self.proven.add(k)
        matched = {ids[j]: obs[i][0] for i, j in hit.items() if obs[i][1] == 1 and ids[j] in self.slots and
                   ids[j] not in claimed}
        self.unseen = self._unseen(cars, matched) if self.suppress_phantoms else {}
        return events

    def _evidence_obs(self, k, s, o, box, conf, now):
        """Hook of the evidence layer (evidence_filter = 1): this frame's raw observation o of slot k (1 / 0 / None), the box matched
        to it (polygon in fractions, or None) with its confidence -> the observation the debounce should see. FixedSlotTracker
        overrides it; here nothing is filtered."""
        return o

    def _owned(self, obs, hit, ids, seen) -> set[int]:
        """Slots whose box in this frame is the vehicle of a BUSY slot without a box of its own (-> not their own
        evidence; seen is updated: that BUSY slot counts as seen, its timers refreshed). A box is slot j's vehicle when
        it overlaps the vehicle box j last had (IoU >= OWN_IOU) or j's slot (IoU >= match_iou). Entries only: a slot
        already BUSY keeps its own box. Front Parking 02, 2026-10-22: the white hatchback hidden behind the sedan is
        found by the zoom with a differently cut box each time (its upper part with the silver car's roof, cut at the
        sedan's roof line, with the sedan's windows), each matching another learned slot (2, 1, 21): every one of them
        turned BUSY in turn, 7 BUSY for 6 cars and short false sessions. Never for distinct pairs: those boxes of one
        car in one frame were recorded as two vehicles. Also records each slot's last vehicle box (.slots[k]["last"])."""
        box_of = {ids[j]: obs[i][0] for i, j in hit.items() if obs[i][1] == 1 and ids[j] in self.slots}
        unboxed = [j for j, s in self.slots.items() if s["state"] == 1 and j not in seen]
        out = set()
        for k, b in box_of.items():
            if self.slots[k]["state"] != 1:
                for j in unboxed:
                    last = self.slots[j].get("last")
                    if (last is not None and iou(b, last) >= OWN_IOU) or iou(b, self.slots[j]["poly"]) >= self.match_iou:
                        out.add(k)
                        seen[j] = 1  # j's car, seen again
                        self.slots[j]["last"] = b
                        break
            if k not in out:
                self.slots[k]["last"] = b
        return out

    def _unseen(self, cars, matched) -> dict[int, str]:
        """Slots shown UNKNOWN (the state and the history stay), with the reason:
        BUSY without a box of its own on a vehicle matched to another slot -> "on the vehicle of slot N" (one box,
        one BUSY slot), unless that vehicle stands in front of it and the two slots have held two cars at once
        (.distinct: the car behind is just hidden); FREE with a vehicle standing in it -> "blocked", behind vehicles in front -> "hidden", any
        vehicle box on its ground (> ground_share of the footprint, not in front) -> "overlaps a vehicle". Judged on
        the footprint: a visible, empty ground is FREE."""
        out = {}
        for k, s in self.slots.items():
            if k in matched and s["state"] == 0:
                out[k] = "a vehicle is in it (entry being confirmed)"  # never green under a car during the hold
                continue
            if k in matched or k in self.phantoms or k in self.covered:
                continue
            poly = s["poly"]
            if s["state"] == 1:
                if k in self.hidden or k in self.shared:
                    continue  # a car may stand there unseen / an unmatched box stands in it
                bottom = float(poly[:, 1].max())
                # not a vehicle in front of it in a pair that has held two cars at once (.distinct): two spaces, the
                # car behind is only hidden (the white hatchback behind the white sedan, Front Parking 02)
                on = {j: iou(poly, b) for j, b in matched.items()
                      if j != k and not (frozenset((k, j)) in self.distinct and b[:, 1].max() >= bottom + 0.01) and
                      (iou(poly, b) > self.vehicle_iou or stands_in(poly, b))}
                owner = max(on, key=on.get, default=None)  # the vehicle overlapping it most
                if owner is not None:
                    out[k] = f"on the vehicle of slot {owner}"
            elif s["state"] == 0:
                foot = footprint(poly)
                front = [c for c in cars if c[:, 1].max() >= foot[3] + 0.01]  # bottom lower = nearer the camera
                if any(stands_in(poly, c) for c in cars):
                    out[k] = "blocked"
                elif _share(foot, front) > self.hidden_ground:
                    out[k] = "hidden"
                elif _share(foot, [c for c in cars if c[:, 1].max() < foot[3] + 0.01]) > self.ground_share:
                    out[k] = "overlaps a vehicle"
        return out

    def tune(self, values: dict) -> dict[str, float]:
        """Apply display thresholds (DISPLAY_TUNING names; clamped, missing / invalid = default) from the next frame
        on. -> the values now in use."""
        values = tuning_values(values)
        for name, v in values.items():
            setattr(self, name, v)
        return values

    def shown(self, k):
        """The state shown for slot k (0 FREE / 1 BUSY / None UNKNOWN): its state, or UNKNOWN while in .unseen.
        The one source for the video banner, the overlay and the dashboard's counts."""
        return None if k in self.unseen else self.slots[k]["state"]

    def _match(self, polys, ids) -> dict[int, int]:
        """Detections (fractions) -> slots: {index in polys: index in ids}, one-to-one. Learned slots: IoU (a learned
        slot is a detection box, so the same car's next box overlaps it heavily). Fixed slots override this."""
        return _greedy(polys, [self.slots[k]["poly"] for k in ids], self.match_iou)

    def _learn(self, obs, now, taken=None):
        """obs: the boxes that matched no slot. taken: {slot: the box matched to it} in this frame (a candidate over
        one of them is a second vehicle there, not its duplicate, unless the two boxes are one vehicle)."""
        taken = taken or {}
        hit = _greedy([o[0] for o in obs], [c["poly"] for c in self.candidates], self.match_iou)
        for i, j in hit.items():
            c = self.candidates[j]
            x1, y1, x2, y2 = _bounds(c["poly"])
            g = _ground(obs[i][0])
            if abs(g[0] - c["ground"][0]) > self.ground_tol * (x2 - x1) or \
                    abs(g[1] - c["ground"][1]) > self.ground_tol * (y2 - y1):
                # the ground point moved: not a vehicle resting in a bay (yet); start over from this box
                c.update(poly=obs[i][0], ground=g, first=now, n=0)
            c["last"], c["n"], c["state"] = now, c["n"] + 1, obs[i][1]
        self.candidates += [{"poly": o[0], "state": o[1], "first": now, "last": now, "n": 1, "ground": _ground(o[0])}
                            for i, o in enumerate(obs) if i not in hit]
        keep = []
        for c in self.candidates:
            if c["n"] >= 2 and c["last"] - c["first"] >= self.learn_hold:
                twin = None if self.dup_iou is None else duplicate_of(c["poly"], {
                    k: s["poly"] for k, s in self.slots.items()}, self.dup_iou, near_iou=self.near_iou)
                fresh = c["last"] == now  # seen in this very frame, next to the vehicles matched in it
                if twin is not None and fresh and twin in taken and not same_vehicle(taken[twin], c["poly"]):
                    twin = None  # another vehicle stands in that slot right now: two vehicles, two spaces
                    pair = (self.next_id, *[k for k, b in taken.items()
                                            if overlap(self.slots[k]["poly"], c["poly"])[0] > 0 and
                                            not same_vehicle(b, c["poly"])])
                    for k in pair[1:]:
                        self.distinct.add(frozenset((pair[0], k)))
                        self.new_distinct.append((min(pair[0], k), max(pair[0], k)))
                if twin is not None:  # the space has a slot already: never a second one over the same vehicle
                    if _area(c["poly"]) >= self.grow * _area(self.slots[twin]["poly"]):
                        self.slots[twin]["poly"] = c["poly"]  # a larger vehicle in a smaller one's bay: grow it
                        self.updated_slots.append(twin)
                    continue
                # the slot starts with its state pending since the candidate was first seen, so an entry is timed
                # from the car's arrival, not from the end of the learning period
                self.slots[self.next_id] = {"poly": c["poly"], "state": None, "cand": c["state"], "since": c["first"],
                                            "occluded": False}
                self.new_slots.append(self.next_id)
                self.next_id += 1
            elif now - c["last"] < self.learn_hold:  # not refreshed for learn_hold = gone (passing car, noise)
                keep.append(c)
        self.candidates = keep

    def view(self, frame_shape):
        """-> ({id: pixel polygon}, {id: shown state}) for viz.draw: the same states as the dashboard's."""
        h, w = frame_shape[:2]
        return ({k: s["poly"] * (w, h) for k, s in self.slots.items()}, {k: self.shown(k) for k in self.slots})


MOVES = ("entry", "exit")  # the events of a vehicle moving: sessions open and close with them (what every reader of the log expects)
# notices (post-demo items 1 and 2): a free bay a neighbour's vehicle stands on (encroached / encroach_cleared) and a detection the
# non-vehicle filter dropped (rejected), an object standing in a free bay (object_blocked / object_cleared). Same table, never a session, never counted as a movement
NOTICES = ("encroached", "encroach_cleared", "rejected", "object_blocked", "object_cleared")


def _kinds_clause(kinds) -> tuple[str, tuple]:
    """(" AND event IN (?, ?)", kinds) for a kinds filter; ("", ()) for None (all kinds). A bare string is one kind."""
    if kinds is None:
        return "", ()
    kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    return f" AND event IN ({', '.join('?' * len(kinds))})", kinds


class HistoryDB:
    """SQLite log, per scope = camera name + detector ("Front Parking 02 | yolov8s"):
    * history: one row per parking session (entry -> exit, duration, status Occupied/Available)
    * events: append-only, one row per slot transition ('entry' = FREE -> BUSY, 'exit' = BUSY -> FREE), linked to
      its session; never updated, so a client can poll for new rows with since_id
    * slots: the learned slot map
    Slots learned by one detector (e.g. the PKLot bay model's false 'bays' on a low-angle view) are never loaded for
    another, and slot IDs in the history always refer to the map they came from. UI settings (the parking-area
    polygon) are per camera name only. Rows written before scopes existed keep the bare camera name and stay readable
    under detector="".
    sessions() / events() / summary() return plain JSON-serialisable dicts (ISO 8601 times with UTC offset plus epoch
    seconds) for an API or a web frontend; recent() / csv() / events_csv() serve the Streamlit tables and downloads.
    Live slot states come from carparking.monitor.ParkingMonitor.state()."""

    COLUMNS = ["id", "slot_id", "entry_time", "exit_time", "duration_min", "status", "note"]

    def __init__(self, path, camera: str, tz: tzinfo | None = None, detector: str = ""):
        self.camera, self.detector, self.tz = camera, detector, tz
        self.scope = f"{camera} | {detector}" if detector else camera
        self.con = sqlite3.connect(str(path), check_same_thread=False)  # Streamlit may rerun on another thread
        had_events = self.con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'events'").fetchone()
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, camera TEXT, slot_id INTEGER,
                entry_time TEXT, exit_time TEXT, duration_min REAL, status TEXT, note TEXT, entry_ts REAL, exit_ts REAL);
            CREATE TABLE IF NOT EXISTS slots (camera TEXT, slot_id INTEGER, poly TEXT, PRIMARY KEY (camera, slot_id));
            CREATE TABLE IF NOT EXISTS settings (camera TEXT, key TEXT, value TEXT, PRIMARY KEY (camera, key));
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, camera TEXT, slot_id INTEGER,
                event TEXT, ts REAL, time TEXT, session_id INTEGER, note TEXT);
            CREATE INDEX IF NOT EXISTS events_scope ON events (camera, id);
            CREATE TABLE IF NOT EXISTS slot_pairs (camera TEXT, a INTEGER, b INTEGER, PRIMARY KEY (camera, a, b));
        """)
        if "retired" not in [r[1] for r in self.con.execute("PRAGMA table_info(slots)")]:
            with self.con:  # retired phantom slots stay in the table, so their IDs are never reused
                self.con.execute("ALTER TABLE slots ADD COLUMN retired REAL")
        if not had_events:  # database from before the event log: one entry (+ exit) event per logged session
            with self.con:
                self.con.execute("INSERT INTO events (camera, slot_id, event, ts, time, session_id, note) "
                                 "SELECT camera, slot_id, event, ts, time, id, note FROM ("
                                 " SELECT camera, slot_id, 'entry' AS event, entry_ts AS ts, entry_time AS time, id, "
                                 "  '' AS note FROM history"
                                 " UNION ALL SELECT camera, slot_id, 'exit', exit_ts, exit_time, id, COALESCE(note, '') "
                                 "  FROM history WHERE exit_ts IS NOT NULL) ORDER BY ts, event")

    def setting(self, key: str, default: str = "") -> str:
        """Per-camera UI setting (e.g. the parking-area polygon), so it survives page reloads and restarts."""
        row = self.con.execute("SELECT value FROM settings WHERE camera = ? AND key = ?", (self.camera, key)).fetchone()
        return default if row is None else row[0]

    def set_setting(self, key: str, value: str):
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO settings VALUES (?, ?, ?)", (self.camera, key, value))

    def close(self):
        self.con.close()

    def _fmt(self, t: float) -> str:
        return datetime.fromtimestamp(t, self.tz).strftime("%Y-%m-%d %H:%M:%S")

    def load_slots(self) -> dict[int, np.ndarray]:
        rows = self.con.execute("SELECT slot_id, poly FROM slots WHERE camera = ? AND retired IS NULL", (self.scope,))
        return {k: np.float32(json.loads(p)) for k, p in rows}

    def max_slot_id(self) -> int:
        """Highest slot ID ever used in this scope, retired slots included (0 if none)."""
        return self.con.execute("SELECT COALESCE(MAX(slot_id), 0) FROM slots WHERE camera = ?",
                                (self.scope,)).fetchone()[0]

    def save_slots(self, tracker: SlotTracker, ids):
        with self.con:
            self.con.executemany("INSERT OR REPLACE INTO slots (camera, slot_id, poly) VALUES (?, ?, ?)",
                                 [(self.scope, k, json.dumps(tracker.slots[k]["poly"].tolist())) for k in ids])

    def distinct_pairs(self) -> set[tuple[int, int]]:
        """Overlapping slots that have held two separate vehicles at once (SlotTracker.distinct): never duplicates."""
        return set(self.con.execute("SELECT a, b FROM slot_pairs WHERE camera = ?", (self.scope,)))

    def add_distinct_pairs(self, pairs):
        with self.con:
            self.con.executemany("INSERT OR IGNORE INTO slot_pairs VALUES (?, ?, ?)",
                                 [(self.scope, *sorted(p)) for p in pairs])

    def proven_slots(self) -> set[int]:
        """Slots that have had a parking session (a car really stood there)."""
        return {k for (k,) in self.con.execute("SELECT DISTINCT slot_id FROM history WHERE camera = ?", (self.scope,))}

    def retire_slots(self, ids, now: float | None = None,
                     note: str = "phantom slot retired (no room for a vehicle there)"):
        """Drop slots from the learned map (SlotTracker.retired) and close any session still open on them."""
        now = time.time() if now is None else now
        for k in ids:
            self.close_slot(k, now, note)
        with self.con:
            self.con.executemany("UPDATE slots SET retired = ? WHERE camera = ? AND slot_id = ?",
                                 [(now, self.scope, k) for k in ids])

    def _open(self, slot: int):
        return self.con.execute("SELECT id, entry_ts FROM history WHERE camera = ? AND slot_id = ? AND exit_ts IS NULL "
                                "ORDER BY id DESC LIMIT 1", (self.scope, slot)).fetchone()

    def _open_all(self, slot: int) -> list:
        """Every open session of the slot, newest first (a capacity slot has one per counted vehicle)."""
        return self.con.execute("SELECT id, entry_ts FROM history WHERE camera = ? AND slot_id = ? AND exit_ts IS NULL "
                                "ORDER BY id DESC", (self.scope, slot)).fetchall()

    def open_by_slot(self) -> dict[int, int]:
        """{slot: open sessions} of this scope."""
        return dict(self.con.execute("SELECT slot_id, COUNT(*) FROM history WHERE camera = ? AND exit_ts IS NULL "
                                     "GROUP BY slot_id", (self.scope,)))

    def sync_open(self, want: dict[int, int], undecided: set[int], t: float) -> list[tuple[str, int, int]]:
        """Make the open sessions match a fixed map's live state, whatever happened before (a restart, a map change, a capacity
        change): want = {slot: vehicles it holds} for slots decided BUSY (>= 1) or FREE (0); undecided = slots still unknown
        (their sessions are left alone). A slot with fewer open sessions than vehicles gets the missing ones (an entry event
        each; the first of a slot that is BUSY with none is a backfill), one with more loses the newest extras, a FREE slot
        or one that is no longer on the map (open sessions of an earlier run) loses all of them. -> [(entry / exit, slot,
        session id)] for what was logged."""
        counts = self.open_by_slot()
        out = []
        todo = {k: n for k, n in want.items() if counts.get(k, 0) != n}
        todo |= {k: 0 for k in counts if k not in want and k not in undecided}
        if not todo:
            return out
        with self.con:
            for k, target in sorted(todo.items()):
                rows = self._open_all(k)
                orphan = k not in want
                for row in rows[:max(len(rows) - target, 0)]:  # (newest first: the vehicle that came last leaves first)
                    note = ("no longer on the slot map (closed by the sync)" if orphan else
                            "the slot is empty (closed by the sync)" if target == 0 else
                            "one of several vehicles in the slot left")
                    self._close(row, k, t, note)
                    out.append(("exit", k, row[0]))
                for _ in range(max(target - len(rows), 0)):
                    note = ("already in the slot when the sync started (entry = first seen)" if not rows else
                            "another vehicle in the same slot (capacity)")
                    cur = self.con.execute("INSERT INTO history (camera, slot_id, entry_time, status, note, entry_ts) "
                                           "VALUES (?, ?, ?, 'Occupied', ?, ?)", (self.scope, k, self._fmt(t), note, t))
                    self._event(k, "entry", t, cur.lastrowid, note)
                    out.append(("entry", k, cur.lastrowid))
        return out

    def apply(self, ev: dict) -> bool:
        """One SlotTracker event -> insert (car in) or close (car out) a session, plus its row in the event log.
        -> True if something was logged. BUSY with a session already open (app restarted while the car stayed) and
        FREE without one change nothing."""
        open_row = self._open(ev["slot"])
        with self.con:
            if ev["state"] == 1 and open_row is None:
                note = "already parked when tracking started (entry = first seen)" if ev["initial"] else ""
                cur = self.con.execute("INSERT INTO history (camera, slot_id, entry_time, status, note, entry_ts) "
                                       "VALUES (?, ?, ?, 'Occupied', ?, ?)",
                                       (self.scope, ev["slot"], self._fmt(ev["time"]), note, ev["time"]))
                self._event(ev["slot"], "entry", ev["time"], cur.lastrowid, note)
                return True
            if ev["state"] == 0 and open_row is not None:
                note = ("left while tracking was off (exit = first seen empty)" if ev["initial"] else
                        "no vehicle seen in it while partly hidden (exit = last seen)" if ev.get("expired") else
                        "was hidden behind a vehicle in front (exit = first seen empty)" if ev.get("occluded") else None)
                for row in self._open_all(ev["slot"]):  # empty: every vehicle of a capacity slot is gone
                    self._close(row, ev["slot"], ev["time"], note)
                return True
        return False

    def log_notice(self, slot: int, kind: str, t: float, note: str = "", scope: str | None = None) -> None:
        """Append a notice to the event log (NOTICES): not a movement of a vehicle, so no session and nothing a reader of entries and
        exits sees unless it asks for these kinds. slot 0 = it concerns no bay."""
        if kind not in NOTICES:
            raise ValueError(f"unknown notice {kind!r}; use one of {NOTICES}")
        with self.con:
            self._event(slot, kind, t, None, note, scope)

    def _event(self, slot: int, kind: str, t: float, session: int | None, note: str | None, scope: str | None = None):
        self.con.execute("INSERT INTO events (camera, slot_id, event, ts, time, session_id, note) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)", (scope or self.scope, slot, kind, t, self._fmt(t), session, note or ""))

    def _close(self, row, slot: int, t: float, note: str | None, scope: str | None = None):
        rid, entry_ts = row
        self.con.execute("UPDATE history SET exit_time = ?, exit_ts = ?, duration_min = ?, status = 'Available', "
                         "note = COALESCE(?, note) WHERE id = ?",
                         (self._fmt(t), t, round((t - entry_ts) / 60, 1), note, rid))
        self._event(slot, "exit", t, rid, note, scope)

    def close_open(self, note: str, t: float | None = None) -> int:
        """Close every open session of this scope (each with its exit event and the note); nothing is deleted -> how many."""
        t = time.time() if t is None else t
        rows = self.con.execute("SELECT id, entry_ts, slot_id FROM history WHERE camera = ? AND exit_ts IS NULL",
                                (self.scope,)).fetchall()
        with self.con:
            for rid, entry_ts, slot in rows:
                self._close((rid, entry_ts), slot, t, note)
        return len(rows)

    def stale_open(self, only=None) -> dict[str, int]:
        """{scope: open sessions} of this camera's OTHER scopes (an earlier detector or auto grid version, whose worker is gone:
        nothing will ever close them). only: a filter on the scope name."""
        pre = f"{self.camera} | "
        rows = self.con.execute("SELECT camera, COUNT(*) FROM history WHERE exit_ts IS NULL AND camera != ? AND "
                                "(camera = ? OR substr(camera, 1, ?) = ?) GROUP BY camera",
                                (self.scope, self.camera, len(pre), pre))
        return {scope: n for scope, n in rows if only is None or only(scope)}

    def close_stale(self, note: str, t: float | None = None, dry_run: bool = False, only=None) -> dict[str, int]:
        """Close the open sessions of this camera's other scopes (see stale_open), each with an exit event under its own scope and
        the note; never deletes a row. dry_run: only count. -> {scope: sessions} (closed, or that would be)."""
        found = self.stale_open(only)
        if dry_run or not found:
            return found
        t = time.time() if t is None else t
        with self.con:
            for scope in found:
                rows = self.con.execute("SELECT id, entry_ts, slot_id FROM history WHERE camera = ? AND exit_ts IS NULL",
                                        (scope,)).fetchall()
                for rid, entry_ts, slot in rows:
                    self._close((rid, entry_ts), slot, t, note, scope)
        return found

    def close_slot(self, slot: int, now: float | None = None, note: str = "slot deleted") -> bool:
        """Close the slot's open session (the slot was deleted from a fixed map) -> True if one was open. The exit
        is logged like any other, with the note."""
        rows = self._open_all(slot)
        with self.con:
            for row in rows:
                self._close(row, slot, time.time() if now is None else now, note)
        return bool(rows)

    def reset(self, now: float | None = None):
        """Forget the slot map (new camera angle): open sessions are closed now, history rows are kept."""
        now = time.time() if now is None else now
        rows = self.con.execute("SELECT id, entry_ts, slot_id FROM history WHERE camera = ? AND exit_ts IS NULL",
                                (self.scope,)).fetchall()
        with self.con:
            for rid, entry_ts, slot in rows:
                self._close((rid, entry_ts), slot, now, "closed by slot map reset")
            self.con.execute("DELETE FROM slots WHERE camera = ?", (self.scope,))
            self.con.execute("DELETE FROM slot_pairs WHERE camera = ?", (self.scope,))  # IDs start over

    def recent(self, n: int = 50) -> list[dict]:
        rows = self.con.execute(f"SELECT {', '.join(self.COLUMNS)} FROM history WHERE camera = ? ORDER BY id DESC "
                                "LIMIT ?", (self.scope, n))
        return [dict(zip(self.COLUMNS, r)) for r in rows]

    def csv(self) -> str:
        """Every session of this camera as CSV text (download button)."""
        import pandas as pd
        return pd.read_sql_query("SELECT * FROM history WHERE camera = ? ORDER BY id", self.con,
                                 params=(self.scope,)).to_csv(index=False)

    def events_csv(self, kinds=MOVES) -> str:
        """The event log of this camera as CSV text (download button): entries and exits; kinds=None: every kind (see NOTICES)."""
        import pandas as pd
        where, args = _kinds_clause(kinds)
        return pd.read_sql_query(f"SELECT id, slot_id, event, time, ts, session_id, note FROM events WHERE camera = ?{where} "
                                 "ORDER BY id", self.con, params=(self.scope, *args)).to_csv(index=False)

    # JSON-ready views: plain dicts of str/int/float/None (no numpy, no pandas)
    def iso(self, t: float | None) -> str | None:
        """Epoch seconds -> ISO 8601 with UTC offset in this camera's time zone ("2026-10-19T10:03:27+05:00")."""
        if t is None:
            return None
        dt = datetime.fromtimestamp(t, self.tz)
        return (dt if self.tz else dt.astimezone()).isoformat(timespec="seconds")  # no tz: the process clock's

    def sessions(self, limit: int = 50, open_only: bool = False) -> list[dict]:
        """Parking sessions, newest first: {"id", "camera", "detector", "slot_id", "in_time", "out_time", "in_ts",
        "out_ts", "duration_min", "status" ("Occupied" / "Available"), "note"}. Open sessions: out_* None."""
        rows = self.con.execute("SELECT id, slot_id, entry_ts, exit_ts, duration_min, status, note FROM history "
                                f"WHERE camera = ? {'AND exit_ts IS NULL' if open_only else ''} ORDER BY id DESC "
                                "LIMIT ?", (self.scope, limit))
        return [{"id": rid, "camera": self.camera, "detector": self.detector, "slot_id": slot,
                 "in_time": self.iso(t_in), "out_time": self.iso(t_out), "in_ts": t_in, "out_ts": t_out,
                 "duration_min": dur, "status": status, "note": note or ""}
                for rid, slot, t_in, t_out, dur, status, note in rows]

    def sessions_page(self, offset: int = 0, limit: int = 20, all_scopes: bool = False) -> dict:
        """One page of sessions, the open ones (still parked) first, then newest first: {"items" (as sessions()),
        "total", "open"}. A car parked for days stays on page 1 next to the live BUSY slots. all_scopes: every scope of this
        camera (each detector / auto grid version it ran with); the items then carry their "scope"."""
        pre = f"{self.camera} | "
        where, args = (("(camera = ? OR substr(camera, 1, ?) = ?)", (self.camera, len(pre), pre)) if all_scopes
                       else ("camera = ?", (self.scope,)))
        rows = self.con.execute("SELECT id, slot_id, entry_ts, exit_ts, duration_min, status, note, camera FROM history "
                                f"WHERE {where} ORDER BY exit_ts IS NULL DESC, id DESC LIMIT ? OFFSET ?",
                                (*args, limit, offset))
        items = [{"id": rid, "camera": self.camera, "detector": self.detector, "slot_id": slot,
                  "in_time": self.iso(t_in), "out_time": self.iso(t_out), "in_ts": t_in, "out_ts": t_out,
                  "duration_min": dur, "status": status, "note": note or "", **({"scope": scope} if all_scopes else {})}
                 for rid, slot, t_in, t_out, dur, status, note, scope in rows]
        total, n_open = self.con.execute("SELECT COUNT(*), COALESCE(SUM(exit_ts IS NULL), 0) FROM history "
                                         f"WHERE {where}", args).fetchone()
        return {"items": items, "total": total, "open": n_open}

    def events(self, limit: int = 100, since_id: int = 0, kinds=MOVES) -> list[dict]:
        """Event log, newest first: {"id", "camera", "detector", "slot_id", "event" ("entry" / "exit"), "time", "ts",
        "session_id", "note"}. since_id: only events with a larger id (polling for new ones). kinds: the event kinds returned
        (default MOVES, what every older reader expects; NOTICES for the encroachment / obstacle notices; None = all); notices
        have no session (session_id None) and slot_id 0 when they concern no bay."""
        where, args = _kinds_clause(kinds)
        rows = self.con.execute(f"SELECT id, slot_id, event, ts, session_id, note FROM events WHERE camera = ? AND "
                                f"id > ?{where} ORDER BY id DESC LIMIT ?", (self.scope, since_id, *args, limit))
        return [{"id": eid, "camera": self.camera, "detector": self.detector, "slot_id": slot, "event": kind,
                 "time": self.iso(t), "ts": t, "session_id": sid, "note": note or ""}
                for eid, slot, kind, t, sid, note in rows]

    def summary(self) -> dict:
        """{"camera", "detector", "sessions", "open_sessions", "events", "last_event"} for a status endpoint."""
        n, n_open = self.con.execute("SELECT COUNT(*), SUM(exit_ts IS NULL) FROM history WHERE camera = ?",
                                     (self.scope,)).fetchone()
        n_ev = self.con.execute("SELECT COUNT(*) FROM events WHERE camera = ? AND event IN ('entry', 'exit')", (self.scope,)).fetchone()[0]
        last = self.events(limit=1)
        return {"camera": self.camera, "detector": self.detector, "sessions": n, "open_sessions": n_open or 0,
                "events": n_ev, "last_event": last[0] if last else None}
