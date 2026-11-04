"""The evidence layer wired into FixedSlotTracker (evidence_filter, post-demo item 4): a stationary car with missed detection frames and
exposure shifts logs no exit with the layer on (and does with it off: the ghost this removes), a real exit and a real entry still go
through, the exit veto's re-look works, and with evidence_filter = 0 the tracker is exactly what it was. One frame per second."""
import random
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.fixed_slots import FixedSlotTracker, parse_points  # noqa: E402

SHAPE = (1000, 2000, 3)
SLOT = {"id": 1, "name": "A1", "poly": parse_points("0.20,0.45,0.56,0.60")}  # px x 400..1120, y 450..600
OTHER = {"id": 2, "name": "A2", "poly": parse_points("0.60,0.45,0.90,0.60")}
CAR = (420, 440, 1100, 595)
T0 = 1000.0


def det(box=CAR, conf=0.8):
    x1, y1, x2, y2 = box
    return {"poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], "box": box, "label": "car", "conf": conf, "state": 1}


@lru_cache(maxsize=None)
def scene(car: bool = True, gain: float = 1.0) -> np.ndarray:
    """Grey ground, with a red car (windscreen, two wheels) in the slot; gain = exposure. Cached: the tracker only reads frames."""
    img = np.full(SHAPE, 110, np.uint8)
    if car:
        x1, y1, x2, y2 = CAR
        cv2.rectangle(img, (x1 + 30, y1 + 20), (x2 - 30, y2 - 5), (40, 40, 200), -1)
        cv2.rectangle(img, (x1 + 150, y1 + 30), (x2 - 150, y1 + 80), (30, 30, 30), -1)
        for cx in (x1 + 130, x2 - 130):
            cv2.circle(img, (cx, y2 - 5), 30, (10, 10, 10), -1)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def tracker(on=1, slots=(SLOT,), **kw):
    return FixedSlotTracker(list(slots), "ground", hold=2, exit_hold=5, absent_free=True, evidence_filter=on, **kw)


def play(tr, plan, give_frame=True):
    """plan: one (dets, frame) per second -> [(t, event)] for every event the tracker reports."""
    out = []
    for i, (dets, frame) in enumerate(plan):
        if give_frame and tr.evidence_filter:
            tr.frame = frame
        out += [(T0 + i, e) for e in tr.update(dets, SHAPE, now=T0 + i)]
    return out


def exits(events):
    return [(t, e) for t, e in events if e["state"] == 0 and not e["initial"]]


def entries(events):
    return [(t, e) for t, e in events if e["state"] == 1]


def stationary_plan():
    """A car parked from t=0, 420 s. Detection drops: 7 s every 60 s (a door, a shadow, a missed frame); at t=200..275 the light
    dips to 55 % and the detector loses the car altogether, though it is still there."""
    plan = []
    for t in range(420):
        dip = 200 <= t < 275
        dropped = dip or (t % 60) < 7 and t >= 20
        plan.append(([] if dropped else [det()], scene(True, 0.55 if dip else 1.0)))
    return plan


# ---- the point of it all -------------------------------------------------------------------------------------------------------

def test_without_the_layer_a_stationary_car_logs_ghost_exits():
    ev = play(tracker(0), stationary_plan())
    assert len(exits(ev)) >= 1  # what the evaluators saw


def test_stationary_car_with_missed_frames_and_exposure_dip_logs_no_exit():
    tr = tracker(1)
    ev = play(tr, stationary_plan())
    assert len(entries(ev)) == 1 and exits(ev) == []
    assert tr.slots[1]["state"] == 1
    assert tr.ev[1].locked  # the parked lock was taken ...
    assert tr.ev[1].vetoes or tr.ev[1].last_veto  # ... and the dip was held off by the veto, not just by the EMA


def test_dip_veto_reason_is_appearance():
    tr = tracker(1)
    play(tr, stationary_plan())
    assert tr.ev[1].last_veto == "appearance"  # the same car, only darker


def test_missed_frames_alone_are_absorbed_by_the_ema():
    """Long gaps below the veto: no frame needed at all (no signature, no probe), the EMA alone keeps the car."""
    tr = tracker(1)
    plan = [([] if (t % 40) < 10 and t >= 15 else [det()], None) for t in range(300)]
    ev = play(tr, plan, give_frame=False)
    assert len(entries(ev)) == 1 and exits(ev) == []
    assert not tr.ev[1].locked or tr.ev[1].sig is None


def test_real_exit_still_goes_through_once():
    tr = tracker(1)
    plan = [([det()], scene(True)) for _ in range(150)] + [([], scene(False)) for _ in range(250)]
    ev = play(tr, plan)
    assert len(entries(ev)) == 1
    out = exits(ev)
    assert len(out) == 1
    assert 150 <= out[0][0] - T0 <= 150 + 120  # delayed by the EMA, not lost
    assert tr.ev[1].locked is False  # released: the next car starts from nothing


def test_replacement_car_is_not_vetoed_as_the_old_one():
    """Another colour parks in the bay straight after the first leaves (no detection in between)."""
    tr = tracker(1)
    other = scene(False).copy()  # (scene() is cached: never draw on it)
    x1, y1, x2, y2 = CAR
    cv2.rectangle(other, (x1 + 30, y1 + 20), (x2 - 30, y2 - 5), (200, 120, 30), -1)  # a blue car, no windscreen or wheels
    plan = [([det()], scene(True)) for _ in range(150)] + [([], scene(False)) for _ in range(150)] + [([det()], other) for _ in range(60)]
    ev = play(tr, plan)
    assert len(exits(ev)) == 1 and len(entries(ev)) == 2


def test_entry_of_a_new_car_is_not_delayed_by_the_layer():
    plain, filtered = tracker(0), tracker(1)
    plan = [([], scene(False)) for _ in range(20)] + [([det()], scene(True)) for _ in range(30)]
    t_plain = entries(play(plain, plan))[0][0]
    t_filtered = entries(play(filtered, plan))[0][0]
    assert t_filtered - t_plain <= 3  # the EMA climbs in a few frames (tau_enter 3 s)


def test_relook_hit_vetoes_an_exit_until_it_stops_seeing_a_car():
    tr = tracker(1)
    clock = {"t": 0}
    tr.relook = lambda k: clock["t"] < 260  # the zoom keeps finding something in the bay until t=260
    plan = [([det()], scene(True)) for _ in range(120)] + [([], scene(False)) for _ in range(300)]
    out = []
    for i, (dets, frame) in enumerate(plan):
        clock["t"], tr.frame = i, frame
        out += [(T0 + i, e) for e in tr.update(dets, SHAPE, now=T0 + i)]
    ex = exits(out)
    assert len(ex) == 1 and ex[0][0] - T0 >= 260  # not before the re-look let go


def test_no_pixels_no_veto_but_the_ema_still_runs():
    tr = tracker(1)
    plan = [([det()], None) for _ in range(120)] + [([], None) for _ in range(250)]
    ev = play(tr, plan, give_frame=False)
    assert len(exits(ev)) == 1  # a real exit without a frame goes through (nothing to veto it with)


def test_far_bay_is_independent():
    """Evidence is per bay: a second, empty bay is unaffected by the first one's car and its dropouts."""
    tr = tracker(1, slots=(SLOT, OTHER))
    ev = play(tr, stationary_plan())
    assert [e["slot"] for _, e in entries(ev)] == [1] and exits(ev) == []
    assert tr.slots[2]["state"] == 0


# ---- switching on / off, editing the map ---------------------------------------------------------------------------------------

def test_switching_on_with_a_busy_bay_does_not_read_a_missed_frame_as_an_exit():
    tr = tracker(0)
    play(tr, [([det()], scene(True)) for _ in range(30)])
    assert tr.slots[1]["state"] == 1
    tr.set_evidence(1)
    ev = []
    for i in range(30, 70):  # on, and the detector misses the car for 40 s straight
        tr.frame = scene(True)
        ev += tr.update([], SHAPE, now=T0 + i)
    assert [e for e in ev if e["state"] == 0] == [] and tr.slots[1]["state"] == 1


def test_switching_off_forgets_everything_and_restores_the_plain_behaviour():
    tr = tracker(1)
    play(tr, [([det()], scene(True)) for _ in range(100)])
    assert tr.ev
    tr.set_evidence(0)
    assert tr.evidence_filter == 0 and not tr.ev and not tr._ev_t and not tr._ev_g and tr.frame is None
    ev = play(tr, [([], scene(False)) for _ in range(30)])
    assert len(exits(ev)) == 1  # plain exit_hold again


def test_removed_slot_drops_its_evidence():
    tr = tracker(1, slots=(SLOT, OTHER))
    play(tr, [([det()], scene(True)) for _ in range(10)])
    assert 1 in tr.ev
    tr.set_slots([OTHER])
    tr.update([], SHAPE, now=T0 + 99)
    assert 1 not in tr.ev and 1 not in tr._ev_t


def test_frame_is_dropped_after_each_update():
    tr = tracker(1)
    tr.frame = scene(True)
    tr.update([det()], SHAPE, now=T0)
    assert tr.frame is None  # no 6 MB frame kept alive between updates


def test_unjudged_slot_passes_through_untouched():
    """A hidden slot (o = None) keeps its state exactly as without the layer: the evidence does not see it."""
    tr = tracker(1)
    assert tr._evidence_obs(1, {"state": 1}, None, None, None, T0) is None
    assert tr.ev == {}


# ---- evidence_filter = 0: exactly what it was ------------------------------------------------------------------------------------

def noisy_plan(seed, n=500):
    rng = random.Random(seed)
    plan, cars = [], {1: False, 2: False}
    for t in range(n):
        if rng.random() < 0.04:
            cars[rng.choice((1, 2))] ^= True  # a car arrives or leaves
        dets = []
        for k, here in cars.items():
            if here and rng.random() < 0.85:  # 15 % of frames the detector misses it
                j = rng.randint(-6, 6)
                box = (420 + j, 440, 1100 + j, 595) if k == 1 else (1220 + j, 440, 1790 + j, 595)
                dets.append(det(box, round(rng.uniform(0.3, 0.95), 2)))
        plan.append((dets, None))
    return plan


class Identity(FixedSlotTracker):
    """The hook with the layer's plumbing on, but doing nothing: any difference from the plain tracker would be the plumbing's."""

    def _evidence_obs(self, k, s, o, box, conf, now):
        return o


def snapshot_of(tr, plan):
    out = []
    for i, (dets, _) in enumerate(plan):
        ev = tr.update(dets, SHAPE, now=T0 + i)
        out.append((ev, {k: (s["state"], s["cand"], s["since"], s["occluded"]) for k, s in tr.slots.items()}, tr.shown(1), tr.shown(2)))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_plumbing_alone_changes_nothing(seed):
    plan = noisy_plan(seed)
    kw = dict(hold=2, exit_hold=5, absent_free=True)
    off = snapshot_of(FixedSlotTracker([SLOT, OTHER], "ground", **kw), plan)
    ident = snapshot_of(Identity([SLOT, OTHER], "ground", evidence_filter=1, **kw), plan)
    assert off == ident
    assert any(ev for ev, *_ in off)  # the scenario does produce events


def test_off_never_reaches_the_evidence_code(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("evidence code ran with evidence_filter = 0")

    for name in ("_evidence_obs", "_evidence_after", "_probe"):
        monkeypatch.setattr(FixedSlotTracker, name, boom)
    tr = FixedSlotTracker([SLOT, OTHER], "ground", hold=2, exit_hold=5, absent_free=True)
    assert tr.evidence_filter == 0
    play(tr, noisy_plan(7, 200))
    assert tr.ev == {} and tr._ev_t == {} and tr._ev_g == {}


def test_filter_is_off_by_default_in_the_constructor():
    assert FixedSlotTracker([SLOT], "ground").evidence_filter == 0
    assert FixedSlotTracker([SLOT], "ground", evidence_filter=True).evidence_filter == 1
