"""Spillage / multi-slot encroachment (spillage.py wired into FixedSlotTracker, monitor.state(), live option "spillage"; post-demo item 1):
a free bay that a neighbour's vehicle stands on is ENCROACHED (not usable), the state stays FREE, and with the option off nothing of it
runs. Synthetic row of four bays on a 2000 x 1000 frame, one frame per second; no model, no network, no ./data."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.fixed_slots import FixedSlotTracker  # noqa: E402
from carparking.history import HistoryDB  # noqa: E402
from carparking.live import Worker  # noqa: E402
from carparking.monitor import ParkingMonitor  # noqa: E402
from carparking.spillage import CFG, AvailTracker, Spill, SpillCfg, spill_map  # noqa: E402

SHAPE = (1000, 2000, 3)
W = 2000
BAY_W, PITCH, TOP, BOTTOM = 0.18, 0.20, 0.50, 0.70  # bays A1..A4: x 0.05-0.23, 0.25-0.43, 0.45-0.63, 0.65-0.83
SLOTS = [{"id": i + 1, "name": f"A{i + 1}", "poly": [[0.05 + PITCH * i, TOP], [0.05 + PITCH * i + BAY_W, TOP],
                                                    [0.05 + PITCH * i + BAY_W, BOTTOM], [0.05 + PITCH * i, BOTTOM]]} for i in range(4)]
T0 = 1000.0


def bay_x(i, share=0.0):
    """x (fraction) of bay i's left edge + share of its width."""
    return 0.05 + PITCH * i + share * BAY_W


def det(x0, x1, bottom=0.695, top=0.42, conf=0.8):
    """A car box from fraction x0 to x1 (as px)."""
    x1p, x2p, y1, y2 = round(x0 * W), round(x1 * W), round(top * 1000), round(bottom * 1000)
    return {"poly": [(x1p, y1), (x2p, y1), (x2p, y2), (x1p, y2)], "box": (x1p, y1, x2p, y2), "label": "car", "conf": conf, "state": 1}


def car_in(i):
    return det(bay_x(i, 0.06), bay_x(i, 0.94))


def over(i, share):
    """A car wholly in bay i that also stands `share` of bay i+1's width into it."""
    return det(bay_x(i, 0.06), bay_x(i + 1, share))


def tracker(on=1, **kw):
    return FixedSlotTracker(SLOTS, "ground", hold=2, exit_hold=3, absent_free=True, spillage=on, **kw)


def run(tr, frames, start=0):
    out = []
    for n, dets in enumerate(frames, start):
        out.append(([(e["slot"], e["state"], e["time"], e["initial"]) for e in tr.update(dets, SHAPE, now=T0 + n)], list(tr.avail_events)))
    return out


def settled(tr, dets, n=12, start=0):
    dets = [dets] if isinstance(dets, dict) else dets
    run(tr, [dets] * n, start)
    return start + n


# ---- spill_map: the rules ---------------------------------------------------------------------------------------------------------
POLYS = {k["id"]: np.float32(k["poly"]) for k in SLOTS}


def fp(x0, x1, bottom=0.695):
    return np.float32([[x0, bottom - 0.08], [x1, bottom - 0.08], [x1, bottom], [x0, bottom]])


def spill(footprints, owners, candidates=(2,), prev=(), cfg=CFG):
    return spill_map(POLYS, footprints, owners, candidates, prev, cfg)


def test_a_single_car_over_thirty_percent_encroaches_and_below_does_not():
    assert 2 in spill([fp(bay_x(0, .06), bay_x(1, .40))], [1])
    assert spill([fp(bay_x(0, .06), bay_x(1, .25))], [1]) == {}
    sp = spill([fp(bay_x(0, .06), bay_x(1, .40))], [1])[2]
    assert sp.by == (1,) and sp.cover == pytest.approx(0.40, abs=0.01) and sp.free_m is None


def test_several_cars_each_under_the_line_encroach_by_their_sum():
    both = [fp(bay_x(0, .06), bay_x(1, .24)), fp(bay_x(1, .76), bay_x(2, .94))]  # 24 % from the left, 24 % from the right
    assert spill(both, [1, 3])[2].total == pytest.approx(0.48, abs=0.02) and spill(both, [1, 3])[2].by in ((1, 3), (3, 1))
    small = [fp(bay_x(0, .06), bay_x(1, .20)), fp(bay_x(1, .80), bay_x(2, .94))]  # 20 % + 20 % = 40 % < 45 %
    assert spill(small, [1, 3]) == {}


def test_hysteresis_keeps_it_between_leave_and_share_and_drops_it_below_leave():
    between = [fp(bay_x(0, .06), bay_x(1, .25))]  # 25 %: under 30 % to enter, over 20 % to stay
    assert spill(between, [1]) == {} and 2 in spill(between, [1], prev={2})
    below = [fp(bay_x(0, .06), bay_x(1, .15))]
    assert 2 not in spill(below, [1], prev={2})


def test_only_vehicles_on_the_bays_row_count():
    front = fp(bay_x(0, .06), bay_x(1, .6), bottom=0.95)  # a van far in front: its ground line is 0.25 below the bay
    behind = fp(bay_x(0, .06), bay_x(1, .6), bottom=0.40)
    assert spill([front], [1]) == {} and spill([behind], [1]) == {}
    assert 2 in spill([fp(bay_x(0, .06), bay_x(1, .6), bottom=0.74)], [1])  # inside the 30 % band below the bay
    assert spill([fp(bay_x(0, .06), bay_x(1, .6), bottom=0.80)], [1]) == {}  # past it


def test_unowned_vehicles_and_a_bays_own_vehicle_are_no_source():
    assert spill([fp(bay_x(0, .06), bay_x(1, .6))], [None]) == {}
    assert spill([fp(bay_x(1, .06), bay_x(1, .94))], [2]) == {}  # the owner of the bay is not encroaching on it


def test_cfg_is_validated():
    for bad in (dict(leave=0.4), dict(share=0.5, total=0.4), dict(align=-1), dict(leave=0)):
        with pytest.raises(ValueError):
            SpillCfg(**bad)


# ---- AvailTracker: the debounce ------------------------------------------------------------------------------------------------------
def sp(k=2, by=(1,)):
    return Spill(k, by, 0.4, 0.4)


def test_enter_needs_the_seconds_and_the_frames():
    a, ev = AvailTracker(min_enter=2), []
    ev += a.update({2: sp()}, 0.0, hold=3.0, exit_hold=10.0, eligible={2})
    ev += a.update({2: sp()}, 1.0, 3.0, 10.0, {2})  # 2 frames but only 1 s
    assert not a.active
    ev += a.update({2: sp()}, 3.0, 3.0, 10.0, {2})  # 3 s and 3 frames
    assert 2 in a.active and [e["event"] for e in ev] == ["encroached"] and ev[0]["note"] == "by 1"
    b = AvailTracker(min_enter=2)  # a slow worker: 20 s between frames still needs two frames
    b.update({2: sp()}, 0.0, 3.0, 10.0, {2})
    assert not b.active
    b.update({2: sp()}, 20.0, 3.0, 10.0, {2})
    assert 2 in b.active


def test_a_gap_resets_the_entry_and_leaving_needs_exit_hold_and_frames():
    a = AvailTracker()
    a.update({2: sp()}, 0.0, 2.0, 6.0, {2})
    a.update({}, 1.0, 2.0, 6.0, {2})  # one clear frame: starts over
    a.update({2: sp()}, 2.0, 2.0, 6.0, {2})
    a.update({2: sp()}, 3.0, 2.0, 6.0, {2})
    assert not a.active
    a.update({2: sp()}, 4.0, 2.0, 6.0, {2})
    assert 2 in a.active
    out = []
    for t in (5.0, 7.0, 9.0):  # clear for 4 s only
        out += a.update({}, t, 2.0, 6.0, {2})
    assert 2 in a.active and not out
    a.update({2: sp()}, 9.5, 2.0, 6.0, {2})  # back: the pending clearing is forgotten
    out = []
    for t in (10.0, 12.0, 14.0, 16.5):
        out += a.update({}, t, 2.0, 6.0, {2})
    assert not a.active and [e["event"] for e in out] == ["encroach_cleared"]


def test_a_bay_that_stops_being_eligible_is_cleared_at_once_and_clear_all():
    a = AvailTracker()
    for t in (0.0, 1.0, 2.0):
        a.update({2: sp(), 3: sp(3)}, t, 1.0, 30.0, {2, 3})
    assert set(a.active) == {2, 3}
    ev = a.update({2: sp(), 3: sp(3)}, 3.0, 1.0, 30.0, {2})  # a car arrived in bay 3
    assert [(e["slot"], e["event"]) for e in ev] == [(3, "encroach_cleared")] and set(a.active) == {2}
    assert [e["slot"] for e in a.clear_all(4.0)] == [2] and not a.active


# ---- FixedSlotTracker -----------------------------------------------------------------------------------------------------------------
def test_a_van_over_three_bays_leaves_only_the_free_bay_usable():
    """Too wide for any single bay under the matcher's bay_share rule: it holds none (unplaced), yet its three bays are lost."""
    tr = tracker()
    van = det(bay_x(0, .04), bay_x(2, .96))  # wholly over A1, A2, A3
    settled(tr, van)
    assert tr.unplaced or any(s["state"] == 1 for s in tr.slots.values())
    names = {k: tr.avail_name(k) for k in tr.slots}
    assert all(names[k] in ("BUSY", "ENCROACHED") for k in (1, 2, 3)) and names[4] == "FREE"
    lost = set(tr.avail)
    assert lost and lost <= {1, 2, 3} and all(tr.slots[k]["state"] == 0 for k in lost)  # the state stays FREE; only .avail says no
    assert all(sp.by == (0,) or sp.by[0] in (1, 2, 3) for sp in tr.avail.values())
    assert sum(1 for n in names.values() if n in ("FREE",)) == 1


def test_a_moving_vehicle_passing_the_row_locks_nothing():
    tr = tracker()
    run(tr, [[det(bay_x(0, .0) + 0.03 * t, bay_x(0, .0) + 0.03 * t + 0.16)] for t in range(10)])  # drives along the row, 1 s per bay-ish
    assert tr.avail == {}


def test_a_car_half_over_the_line_costs_the_neighbour_and_one_in_its_place_does_not():
    tr = tracker()
    settled(tr, over(0, 0.40))
    assert set(tr.avail) == {2} and tr.avail[2].cover >= 0.30 and tr.avail_name(2) == "ENCROACHED"
    tr2 = tracker()
    settled(tr2, over(0, 0.20))
    assert tr2.avail == {} and tr2.avail_name(2) == "FREE"


def test_jitter_at_the_threshold_does_not_flap():
    tr = tracker()
    n = settled(tr, over(0, 0.40))
    flips = 0
    rng = random.Random(3)
    for t in range(n, n + 120):
        share = rng.choice((0.27, 0.33, 0.25, 0.31))  # around 0.30, never below the 0.20 leave share
        before = set(tr.avail)
        tr.update([over(0, share)], SHAPE, now=T0 + t)
        flips += before != set(tr.avail)
    assert flips == 0 and set(tr.avail) == {2}


def test_it_clears_when_the_car_moves_back_but_not_before_the_exit_hold():
    tr = tracker()
    n = settled(tr, over(0, 0.40))
    ev = run(tr, [[over(0, 0.05)]] * 2, n)  # 2 s of clear: exit_hold is 3 s
    assert set(tr.avail) == {2}
    ev += run(tr, [[over(0, 0.05)]] * 6, n + 2)
    assert tr.avail == {} and [e["event"] for _, evs in ev for e in evs] == ["encroach_cleared"]


def test_a_car_still_arriving_locks_nothing_until_it_is_busy_in_its_own_bay():
    tr = tracker()
    run(tr, [[over(0, 0.45)]], 0)  # first frame: nothing confirmed yet
    assert tr.avail == {} and tr.slots[1]["state"] != 1
    settled(tr, over(0, 0.45), 12, 1)
    assert set(tr.avail) == {2}


def test_a_car_arriving_in_the_encroached_bay_clears_it_and_makes_it_busy():
    tr = tracker()
    n = settled(tr, over(0, 0.40))
    ev = run(tr, [[over(0, 0.40), car_in(1)]] * 12, n)
    assert tr.avail == {} and tr.slots[2]["state"] == 1
    assert "encroach_cleared" in [e["event"] for _, evs in ev for e in evs]


def test_the_states_and_events_of_the_tracker_are_the_same_with_the_layer_on():
    rng = random.Random(7)
    frames = []
    for t in range(300):
        cars = []
        if t > 5:
            cars.append(over(0, 0.40) if t < 150 else car_in(0))
        if 40 < t < 220 and rng.random() > 0.1:
            cars.append(car_in(3))
        frames.append(cars)
    a, b = tracker(0), tracker(1)
    ra, rb = run(a, frames), run(b, frames)
    assert [x for x, _ in ra] == [x for x, _ in rb]
    assert {k: (s["state"], s["since"]) for k, s in a.slots.items()} == {k: (s["state"], s["since"]) for k, s in b.slots.items()}
    assert [a.shown(k) for k in a.slots] == [b.shown(k) for k in b.slots]
    assert a.avail == {} and all(not evs for _, evs in ra) and any(evs for _, evs in rb)


def test_capacity_pins_and_vehicle_driven_bays_are_left_alone():
    slots = [dict(s) for s in SLOTS]
    slots[1]["capacity"] = 3
    tr = FixedSlotTracker(slots, "ground", hold=2, exit_hold=3, absent_free=True, spillage=1)
    settled(tr, over(0, 0.45))
    assert tr.avail == {}  # A2 is a capacity pin


def test_switching_it_off_clears_with_an_event_and_on_starts_from_nothing():
    tr = tracker()
    n = settled(tr, over(0, 0.45))
    assert tr.avail
    tr.set_spillage(0, now=T0 + n)
    assert tr.avail == {} and tr.spillage == 0
    ev = run(tr, [[over(0, 0.45)]], n)
    assert [e["event"] for e in ev[0][1]] == ["encroach_cleared"]
    assert run(tr, [[over(0, 0.45)]], n + 1)[0][1] == []  # handed out once
    tr.set_spillage(1)
    assert tr.avail == {}
    tr.set_spillage(1)  # unchanged: no reset
    assert tr.spillage == 1


# ---- the monitor's state() and the live option -----------------------------------------------------------------------------------------
class Static:
    def __init__(self, dets):
        self.dets, self.conf, self.targets = dets, None, None

    def __call__(self, frame, sliced=False):
        return [dict(d) for d in self.dets]


def monitor(tmp_path, dets, **tracker_kw):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    return ParkingMonitor(Static(dets), db, fixed=SLOTS, match_rule="ground", hold=2, exit_hold=3, absent_free=True, **tracker_kw)


def test_state_counts_only_usable_free_bays_and_lists_who_blocks_them(tmp_path):
    frame = np.zeros(SHAPE, np.uint8)
    on = monitor(tmp_path, [over(0, 0.45)], spillage=1)
    for t in range(12):
        on.process(frame, t=T0 + t)
    st = on.state()
    by_id = {s["id"]: s for s in st["slots"]}
    assert by_id[1]["display"] == "BUSY" and by_id[2]["state"] == "FREE" and by_id[2]["display"] == "ENCROACHED"
    assert by_id[2]["avail"] == "ENCROACHED" and by_id[2]["encroached_by"] == ["A1"] and by_id[2]["cover"] >= 0.3
    assert by_id[3]["avail"] is None and by_id[3]["display"] == "FREE"
    c = st["counts"]
    assert (c["busy"], c["encroached"], c["free"], c["available"]) == (1, 1, 2, 2) and c["slots"] == 4
    on.stream = "frozen"  # a stale picture knows nothing, and so nothing is encroached
    stale = on.state()
    assert stale["counts"]["encroached"] == 0 and all(s["avail"] is None for s in stale["slots"])


def test_state_counts_a_van_over_three_bays_as_three_lost_spaces(tmp_path):
    frame = np.zeros(SHAPE, np.uint8)
    mon = monitor(tmp_path, [det(bay_x(0, .04), bay_x(2, .96))], spillage=1)
    for t in range(12):
        mon.process(frame, t=T0 + t)
    c = mon.state()["counts"]
    assert c["available"] == 1 and c["free"] == 1 and c["busy"] + c["encroached"] == 3 and c["slots"] == 4


def test_state_is_what_it_was_with_the_layer_off(tmp_path):
    frame = np.zeros(SHAPE, np.uint8)
    off = monitor(tmp_path, [over(0, 0.45)])
    for t in range(12):
        off.process(frame, t=T0 + t)
    st = off.state()
    assert all(k not in st["counts"] for k in ("encroached", "available"))
    assert all("avail" not in s and "display" not in s for s in st["slots"])
    assert st["counts"]["free"] == 3 and st["counts"]["busy"] == 1  # A2 is counted FREE, as before


def test_the_worker_follows_the_spillage_option():
    tr = tracker(0)
    opts = {"spillage": "0"}
    w = SimpleNamespace(mon=SimpleNamespace(tracker=tr), store=SimpleNamespace(option=lambda k, d: opts.get(k, d)))
    Worker._spillage(w)
    assert tr.spillage == 0
    opts["spillage"] = "1"
    Worker._spillage(w)
    assert tr.spillage == 1
    opts["spillage"] = "0"
    Worker._spillage(w)
    assert tr.spillage == 0
    Worker._spillage(SimpleNamespace(mon=None, store=w.store))  # no monitor yet: nothing to do
    Worker._spillage(SimpleNamespace(mon=SimpleNamespace(tracker=object()), store=w.store))  # a learned map's tracker has no such layer
