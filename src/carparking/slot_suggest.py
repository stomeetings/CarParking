"""Slot suggestions (Task 2): an AI-drawn bay grid shown as a DRAFT next to the slots that run now, to compare, fine-tune, accept
or discard. numpy / cv2 only, no database, no model: api/geometry_routes.py runs the jobs and stores the drafts as child
versions of the active calibration (geometry_store: record["suggest"] = {parent, mode, ...}, record["grid"] = the bays).

A draft never reaches the worker: it reads the ACTIVE version only. Accepting activates the draft (an auto-grid camera, see
ActiveGeometry.frozen) or merges chosen bays into the hand-drawn map (a Fixed camera, merge_into_fixed). Ids are carried over by
overlap with what runs now (carry_ids), so a bay that stays where it is keeps its id and its history."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .fixed_slots import poly_inside, poly_iou
from .geometry.bev import car_footprint
from .geometry.camera import ground_resolution
from .geometry.pipeline import CAR_MAX_RES, usable_vehicle_boxes
from .geometry_live import MANUAL, ActiveGeometry, GeometryFeed

MATCH_IOU = 0.5  # a suggested bay over a running one by this much is the same bay (it keeps the id)
SAME_IOU = 0.9  # ... and from this much it has not moved
MIN_ANCHORS = 2  # fewer vehicles than this on the still and the bays line up as the calibration's own cars did
EDITED = "edited"  # a bay the operator reshaped in the draft
PROTECTED = "running"  # a Fixed camera's drawn slot carried into the draft exactly as it runs (see protect_baseline)
PROTECT_OVERLAP = 0.10  # a suggested bay is dropped when this share of the smaller of it and a running slot lies on the other


def vehicle_anchors(g: ActiveGeometry, dets: list[dict], w: int, h: int) -> list[np.ndarray]:
    """Centres (metres) of the whole, confident vehicles on a w x h still that the calibration can place: where the rows
    of bays should line up with today's cars. dets: the detector's output (boxes as frame fractions)."""
    cam = g.camera(w, h)
    if cam is None:
        return []
    px = [{**d, "box": [d["box"][0] * w, d["box"][1] * h, d["box"][2] * w, d["box"][3] * h]} for d in dets]
    out = []
    for x1, y1, x2, y2 in usable_vehicle_boxes(px, w, h):
        if ground_resolution(cam, [((x1 + x2) / 2, y2)])[0] > CAR_MAX_RES:
            continue  # too far for the car's position to mean anything
        fp = car_footprint(cam, (x1, y1, x2, y2))
        if fp is not None:
            out.append(fp[0].mean(axis=0))
    return out


def suggest_slots(g: ActiveGeometry, dets: list[dict], w: int, h: int, expected: int | None) -> tuple[list[dict], str]:
    """The Quick suggestion: the active calibration's surface re-tiled around the vehicles standing now, continued up to the
    expected bay count -> (fixed-slot dicts in image fractions, ids 1..n front to back; "vehicles" | "calibration" = what the
    rows were lined up with)."""
    anchors = vehicle_anchors(g, dets, w, h)
    lined = len(anchors) >= MIN_ANCHORS
    tiles = g.retile(anchors if lined else None, expected)
    return GeometryFeed.slots_from_tiles(g, tiles, w, h), "vehicles" if lined else "calibration"


def carry_ids(new: list[dict], old: list[dict], reserved=()) -> list[dict]:
    """The suggested bays with the id (and name) of the running bay each one overlaps (greedy, best overlap first, at least
    MATCH_IOU, one to one); the others get fresh ids above everything in use, never one in `reserved` (a pin's own id)."""
    pairs = sorted(((poly_iou(n["poly"], o["poly"]), i, j) for i, n in enumerate(new) for j, o in enumerate(old)), reverse=True)
    took_new, took_old = {}, set()
    for iou, i, j in pairs:
        if iou < MATCH_IOU:
            break
        if i not in took_new and j not in took_old:
            took_new[i] = j
            took_old.add(j)
    used = {old[j]["id"] for j in took_new.values()} | set(reserved)
    fresh = max([o["id"] for o in old] + list(used) + [0])
    out = []
    for i, n in enumerate(new):
        if i in took_new:
            sid, name = old[took_new[i]]["id"], old[took_new[i]].get("name")
        else:
            fresh += 1
            while fresh in used:
                fresh += 1
            sid, name = fresh, None
        out.append({**n, "id": sid, "name": name or f"B{sid}"})
    return sorted(out, key=lambda s: s["id"])


def overlap_share(a, b) -> float:
    """How much two polygons (fractions of the frame, any shape) lie on each other: the share of the smaller one inside the other (0..1).
    Bays side by side, touching along an edge, score about 0."""
    return max(poly_inside(a, b), poly_inside(b, a))


def protect_baseline(new: list[dict], running: list[dict], limit: float = PROTECT_OVERLAP, avoid: list[dict] = ()) -> tuple[list[dict], list[int]]:
    """The draft of a camera, built around the slots that run: every running slot (B1..B8, a Fixed map's pin P1 ...) is carried in exactly as it
    is, same id, name, shape, capacity and source (`protected` True), so the draft never shows one as removed or moved and never shifts it,
    whatever the suggestion would have matched it to; of the suggested bays only those that do NOT lie on a running slot (more than `limit` of
    the smaller one) are added, with fresh ids above every running id. `avoid`: slots that are not part of the draft's bays but that nothing may
    be laid on either and whose ids stay theirs (an auto grid's pins, which the worker merges on top of the grid).
    -> (the draft's bays sorted by id, the protected ids)."""
    base = [{"id": s["id"], "name": s["name"], "poly": s["poly"], "lift": s.get("lift", 0.0), "source": s.get("source") or PROTECTED, "protected": True,
             **({"capacity": s["capacity"]} if int(s.get("capacity") or 1) > 1 else {})} for s in running]
    held = [*running, *avoid]
    keep = [n for n in new if not any(overlap_share(n["poly"], r["poly"]) > limit for r in held)]
    added = carry_ids(keep, [], [s["id"] for s in held])
    return sorted(base + added, key=lambda s: s["id"]), [s["id"] for s in base]


def diff(grid: list[dict], baseline: list[dict]) -> dict:
    """What accepting the draft would change against the running bays (pins left out of both): {"added": [id], "removed": [id],
    "moved": [id], "unchanged": [id]}; ids are the carried ones, so a bay is "moved" when it overlaps its old self by less than
    SAME_IOU."""
    old = {s["id"]: s for s in baseline if s.get("source") != MANUAL}
    new = {s["id"]: s for s in grid if s.get("source") != MANUAL}
    both = sorted(set(old) & set(new))
    moved = [i for i in both if poly_iou(new[i]["poly"], old[i]["poly"]) < SAME_IOU]
    return {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
            "moved": moved, "unchanged": [i for i in both if i not in moved]}


def baseline_hash(slots: list[dict]) -> str:
    """A short fingerprint of the running bays (id + polygon): the draft remembers it, and Accept refuses when the running
    map changed since (someone drew or accepted something else meanwhile)."""
    rows = sorted((s["id"], np.round(np.asarray(s["poly"], float), 3).tolist()) for s in slots)
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:16]


def bind_pin_ids(pins: list[dict], running: list[dict]) -> list[dict]:
    """Every pin gets the slot id it has on the running grid (`running` = the grid with the pins merged in), if it has none yet:
    the id is then the pin's for good, whatever grid comes next, so its history and capacity stay with it."""
    manual = [s for s in running if s.get("source") == MANUAL]
    out = []
    for p in pins:
        if not p.get("slot_id"):
            best = max(manual, key=lambda s: poly_iou(p["poly"], s["poly"]), default=None)
            if best is not None and poly_iou(p["poly"], best["poly"]) >= MATCH_IOU:
                p = {**p, "slot_id": best["id"]}
        out.append(p)
    return out


def merge_into_fixed(drawn: list[dict], grid: list[dict], select: list[int] | None) -> list[dict]:
    """The hand-drawn map with the chosen suggested bays (select = their ids; None = all): a chosen bay carrying a drawn slot's
    id takes that slot's place (its name and capacity stay, so does its history), any other chosen bay is added. Slots that
    are not chosen are returned exactly as they were."""
    chosen = {s["id"]: s for s in grid if select is None or s["id"] in select}
    out = []
    for s in drawn:
        new = chosen.pop(s["id"], None)
        out.append({**s, "poly": new["poly"]} if new else dict(s))
    for sid, s in sorted(chosen.items()):
        out.append({"id": sid, "name": s.get("name") or f"B{sid}", "poly": s["poly"]})
    return sorted(out, key=lambda s: s["id"])
