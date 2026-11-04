"""Slot evidence (post-demo item 4): keeps a parked car parked. A stationary car must not log an exit / entry pair because of a missed
frame, a jittering box or a lighting dip. Per bay, in front of the existing hold / exit_hold debounce (history.py SlotTracker.update):

  step()          asymmetric EMA of the per-frame observation o_t (conf x cover, 0 when visibly empty, None when not judged) with a
                  hysteresis band: E > HI = BUSY, E < LO = FREE, between = unchanged. Slow to leave (tau_exit 30 s), quick to enter
                  (tau_enter 3 s). alpha is capped (alpha_max) so one very slow frame (the CPU worker can take 5-60 s) never flips it:
                  from E = 0.9, two misses in a row still leave BUSY standing, three are needed to cross LO.
  track_anchor()  the parked lock: a BUSY bay whose matched box stays within IoU LOCK_IOU of its first parked box for lock_s seconds and
                  lock_frames frames is locked, and a signature of the car is taken then (lazily, via crop_fn).
  gate_exit()     a locked bay's exit (filtered state 0) is held back until exit_veto() finds nothing saying the car is still there:
                    relook       a forced zoom on the bay found a vehicle
                    appearance   the crop at the anchor still looks like the car that parked (signature.same_vehicle)
                    no_motion    the pixels under the box changed no more than the whole frame did (lighting / exposure, not a car
                                 driving off: a real exit changes the bay far more than it changes the picture)
                  A veto starts a cooldown (no new probe for veto_cooldown_s), so the zoom / crop is not paid every frame.

All pure functions on a SlotEvidence per bay; the expensive evidence (crop signature, zoom, global change) comes through a `probe`
callable that gate_exit calls only when a locked bay actually wants to leave. Per frame the cost is O(bays) scalar maths.
With the layer off, SlotTracker never calls it and behaves as before. numpy only (signature.py brings cv2 for the crops)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from carparking import signature as sg


@dataclass(frozen=True)
class EvidenceCfg:
    tau_enter: float = 3.0  # s: time constant of the EMA while the observation is above it (a car arriving)
    tau_exit: float = 30.0  # s: ... while below it (a car apparently leaving): slow, a gap is not an exit
    alpha_max: float = 0.5  # cap of the per-frame step, whatever dt
    conf_full: float = 0.5  # detector confidence that counts as full evidence: a steady 0.30 box (the default threshold) = 0.6 > hi, BUSY
    hi: float = 0.55  # E above: BUSY
    lo: float = 0.20  # E below: FREE (between: unchanged)
    lock_iou: float = 0.7  # box IoU with the first parked box that counts as "has not moved"
    lock_s: float = 60.0  # s the box must stay put ...
    lock_frames: int = 3  # ... and in at least this many frames
    max_bhatta: float = sg.MAX_BHATTA  # appearance veto: colours closer than this ...
    min_ncc: float = sg.MIN_NCC  # ... and layout correlating at least this = still the same car
    motion_ratio: float = 2.0  # local change must exceed this x the global change to count as a car leaving
    motion_floor: float = 3.0  # grey levels: below this much local change nothing happened under the box, whatever the global change
    veto_cooldown_s: float = 30.0  # s without a new probe after a veto
    max_vetoes: int = 8  # consecutive vetoes after which the exit goes through (a veto must not make a ghost immortal)


CFG = EvidenceCfg()


@dataclass
class SlotEvidence:
    """The evidence state of one bay. seen: the filtered observation (1 busy / 0 free / None = no decisive reading yet); feed it to the
    hold machine in place of the raw one. The rest is the parked lock and the exit veto bookkeeping."""
    e: float = 0.0  # EMA evidence 0..1
    n: int = 0  # observations taken into e
    seen: int | None = None
    anchor: np.ndarray | None = None  # first parked box (polygon in fractions)
    anchor_since: float = 0.0  # time of the first parked box
    anchor_n: int = 0  # frames the box has stayed within lock_iou of the anchor
    locked_at: float | None = None
    sig: sg.Signature | None = None  # the parked car, taken when the lock is
    vetoes: int = 0  # consecutive exit vetoes
    veto_until: float = 0.0  # no new probe before this time
    last_veto: str | None = None

    @property
    def locked(self) -> bool:
        return self.locked_at is not None


def observation(raw: int | None, conf: float | None, cfg: EvidenceCfg = CFG) -> float | None:
    """The evidence value o_t of a slot's raw observation: None = not judged, 0 = visibly empty, else the matched box's confidence
    scaled so that cfg.conf_full is full evidence. (The box already passed the matching rules, so cover is not multiplied in: a car
    half over the line is still a car.)"""
    if raw is None:
        return None
    if not raw:
        return 0.0
    return 1.0 if conf is None else min(max(float(conf), 0.0) / cfg.conf_full, 1.0)


def step(ev: SlotEvidence, o: float | None, v: float, dt: float, cfg: EvidenceCfg = CFG) -> int | None:
    """Fold one observation into the bay's evidence and return the filtered state (1 / 0 / None).

    o: this frame's evidence of a car (max over matched boxes of conf x cover; 0 = visibly empty); None = the bay was not judged
       (hidden, unverifiable stream): nothing changes, as for the hold machine today.
    v: visibility 0..1 of the bay (visible share of its footprint, far-bay pixel factor); a half-hidden bay moves half as fast. 0 = skip.
    dt: seconds since this bay's previous observation (capture time), so uneven frame rates do not change the maths.
    The first observation seeds E directly: a worker restart on a full bay does not need seconds to climb back over HI."""
    if o is None or v <= 0:
        return ev.seen
    o, v = min(max(float(o), 0.0), 1.0), min(float(v), 1.0)
    if ev.n == 0:
        ev.e = o
    else:
        tau = cfg.tau_enter if o > ev.e else cfg.tau_exit
        alpha = min(1.0 - math.exp(-max(dt, 0.0) / tau), cfg.alpha_max) * v
        ev.e += alpha * (o - ev.e)
    ev.n += 1
    if ev.e > cfg.hi:
        ev.seen, ev.vetoes = 1, 0  # clearly there: any run of vetoes is over
    elif ev.e < cfg.lo:
        ev.seen = 0
    return ev.seen


def box_iou(a, b) -> float:
    """IoU of the axis-aligned bounds of two boxes (polygons in fractions): the cheap overlap used to ask "has the car moved"."""
    pa, pb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    (ax1, ay1), (ax2, ay2) = pa.min(0), pa.max(0)
    (bx1, by1), (bx2, by2) = pb.min(0), pb.max(0)
    iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def track_anchor(ev: SlotEvidence, box, now: float, cfg: EvidenceCfg = CFG, crop_fn=None) -> bool:
    """Feed the box matched to a BUSY bay this frame (only frames with a match: a missed frame neither breaks nor extends the run).
    Returns True on the call that takes the lock. crop_fn() -> BGR crop of the car, called only then, to take the signature.

    Unlocked: a box within lock_iou of the anchor extends the run; any other box becomes the new anchor (the car is still settling).
    Locked: the anchor stays, a box that moved is for gate_exit to judge, not for re-anchoring; release() is the way out."""
    if ev.locked:
        return False
    if ev.anchor is None or box_iou(box, ev.anchor) < cfg.lock_iou:
        ev.anchor, ev.anchor_since, ev.anchor_n = np.asarray(box, np.float32), now, 1
        return False
    ev.anchor_n += 1
    if now - ev.anchor_since < cfg.lock_s or ev.anchor_n < cfg.lock_frames:
        return False
    ev.locked_at = now
    ev.sig = sg.make_signature(crop_fn()) if crop_fn is not None else None
    return True


def release(ev: SlotEvidence):
    """The bay is confirmed FREE (exit logged) or its slot was redrawn: forget the parked car so the next one starts from nothing."""
    ev.anchor, ev.anchor_n, ev.locked_at, ev.sig = None, 0, None, None
    ev.vetoes, ev.veto_until, ev.last_veto = 0, 0.0, None


def exit_veto(ev: SlotEvidence, sig_now: sg.Signature | None, global_change: float | None, relook_hit: bool | None,
              cfg: EvidenceCfg = CFG) -> str | None:
    """Why a locked bay may NOT start its exit countdown, or None when nothing says the car is still there. Cheapest check first.

    relook_hit    True if a forced zoom on the bay found a vehicle; None = no zoom was done (skipped, not a "no")
    sig_now       signature of the crop at the anchor now; None = no crop (skipped)
    global_change mean absolute grey-level change of the whole frame over the same interval (stream_health.thumbnail difference);
                  None = unknown (skipped). The local change is thumb_diff(parked signature, sig_now).
    A bay that is not locked is never vetoed: without a settled car there is nothing to protect."""
    if not ev.locked:
        return None
    if relook_hit:
        return "relook"
    if sig_now is not None and ev.sig is not None:
        if sg.same_vehicle(ev.sig, sig_now, cfg.max_bhatta, cfg.min_ncc):
            return "appearance"
        if global_change is not None:
            local = sg.thumb_diff(ev.sig, sig_now)
            if local < max(cfg.motion_ratio * global_change, cfg.motion_floor):
                return "no_motion"
    return None


def gate_exit(ev: SlotEvidence, seen: int | None, now: float, cfg: EvidenceCfg = CFG, probe=None) -> int | None:
    """The filtered state as the hold machine should see it: a locked bay's 0 is turned back into 1 while exit_veto says the car is
    still there. probe() -> (sig_now, global_change, relook_hit) is called only when a locked bay wants to leave and its cooldown is
    over, so the crop, the signature and the forced zoom are paid at exit candidacy, never per frame. Without a probe the exit goes
    through. After max_vetoes in a row it goes through as well."""
    if seen != 0 or not ev.locked:
        return seen
    if now < ev.veto_until:
        return 1
    if probe is None or ev.vetoes >= cfg.max_vetoes:
        return seen
    reason = exit_veto(ev, *probe(), cfg=cfg)
    if reason is None:
        return seen
    ev.vetoes += 1
    ev.last_veto = reason
    ev.veto_until = now + cfg.veto_cooldown_s
    ev.seen = 1
    ev.e = max(ev.e, (cfg.hi + cfg.lo) / 2)  # back inside the band: it takes a fresh run of misses to ask again
    return 1
