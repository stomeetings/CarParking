"""Blocked by object ("BLOCKED", post-demo item 5): a FREE bay with a non-vehicle object standing in it (a potted plant, a vase, a suitcase / box, a
chair, a bench) is not a usable space. Pure functions; nothing here touches the slot states.

  blocked_map()   which FREE, unowned bays an object's footprint stands in, and by what.
  (the debounce is spillage.AvailTracker, constructed with this layer's event names: an object is BLOCKED after `hold` seconds AND 2 frames, and
   clear again after `exit_hold` seconds AND 3 frames, the same temporal filters as every layer; a passing person or a trolley does not block.)

The rules (BlockedCfg):
  object   a detection whose label is in `labels` (the detector's COCO names: 'potted plant', 'vase', 'suitcase', 'chair', 'bench') with a confidence of
           at least `min_conf`. People, animals and vehicles are never objects here (a vehicle makes a bay BUSY, a person walks away).
  in a bay the share of the object's footprint (the bottom 30 % of its box: where it stands) inside the bay is at least `share` (0.50) to block
           it; once BLOCKED it stays while the share is above `leave` (0.25): box jitter at the threshold does not flap it. An object over
           two bays blocks the one it stands in most.
Only FREE bays that no vehicle owns and that hold a single space are candidates (a capacity pin is counted by its vehicles). BLOCKED never changes the
state: a blocked bay is FREE for the history and the sessions, but not usable: out of the free count, drawn as blocked."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABELS = ("potted plant", "vase", "suitcase", "chair", "bench")


@dataclass(frozen=True)
class BlockedCfg:
    labels: tuple[str, ...] = LABELS
    share: float = 0.50  # the object's footprint inside the bay that blocks it
    leave: float = 0.25  # hysteresis: once BLOCKED it stays until the share is below this
    min_conf: float = 0.30  # detections under this are not objects

    def __post_init__(self):
        if not 0 < self.leave < self.share <= 1 or not 0 <= self.min_conf <= 1:
            raise ValueError("BlockedCfg needs 0 < leave < share <= 1 and 0 <= min_conf <= 1")


CFG = BlockedCfg()


@dataclass(frozen=True)
class Block:
    slot: int  # the blocked bay
    by: tuple[str, ...]  # what stands in it (labels, the best-covering first)
    cover: float  # the largest share of an object's footprint inside the bay
    conf: float  # that object's detection confidence


def blocked_map(slots: dict[int, np.ndarray], objects: list[dict], candidates, prev=(), cfg: BlockedCfg = CFG) -> dict[int, Block]:
    """slots: bay id -> polygon (fractions). objects: [{"poly": fractions of the frame (the object's box corners), "label", "conf"}] - the frame's
    non-vehicle detections. candidates: the bays that may be blocked (free, unowned). prev: the bays BLOCKED in the previous frame (the hysteresis)."""
    from carparking.fixed_slots import footprint_poly, poly_inside  # (fixed_slots imports this package's modules: a late import breaks the cycle)
    found: dict[int, list[tuple[float, str, float]]] = {}
    for o in objects:
        if o.get("label") not in cfg.labels or float(o.get("conf", 0.0)) < cfg.min_conf:
            continue
        fp = footprint_poly(np.asarray(o["poly"], np.float32))
        best = max(((poly_inside(fp, slots[k]), k) for k in candidates), default=(0.0, None))
        share, k = best
        if k is None:
            continue
        need = cfg.leave if k in prev else cfg.share
        if share >= need:
            found.setdefault(k, []).append((share, str(o["label"]), float(o.get("conf", 0.0))))
    out = {}
    for k, hits in found.items():
        hits.sort(reverse=True)
        out[k] = Block(k, tuple(dict.fromkeys(label for _, label, _ in hits)), round(hits[0][0], 3), round(hits[0][2], 3))
    return out
