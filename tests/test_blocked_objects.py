"""Blocked by object (blocked.py wired into FixedSlotTracker, ParkingMonitor.state(), live option "blocked_objects"; post-demo item 5): a FREE bay
with a non-vehicle object (potted plant, vase, suitcase ...) standing in it is BLOCKED: the state stays FREE for the history and the sessions, the bay is
left out of the free count, and with the option off nothing of it runs. Synthetic row of four bays; no model, no network, no ./data."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from test_spillage import SHAPE, SLOTS, T0, W, bay_x, car_in, settled  # noqa: E402

from carparking.blocked import CFG, BlockedCfg, blocked_map  # noqa: E402
from carparking.fixed_slots import FixedSlotTracker  # noqa: E402
from carparking.history import HistoryDB  # noqa: E402
from carparking.live import Worker  # noqa: E402
from carparking.monitor import ParkingMonitor  # noqa: E402
from carparking.spillage import AvailTracker  # noqa: E402


def thing(i, label="potted plant", conf=0.7, share=0.5, size=0.04):
    """A non-vehicle detection standing in bay i (its box around the middle of the bay, `size` of the frame wide)."""
    x0 = bay_x(i, share) - size / 2
    x1p, x2p, y1, y2 = round(x0 * W), round((x0 + size) * W), 560, 690
    return {"poly": [(x1p, y1), (x2p, y1), (x2p, y2), (x1p, y2)], "box": (x1p, y1, x2p, y2), "label": label, "conf": conf, "state": None}


def tracker(on=1, **kw):
    return FixedSlotTracker(SLOTS, "ground", hold=2, exit_hold=3, absent_free=True, blocked=on, **kw)


def play(tr, frames, objects, start=0):
    """frames: the vehicle detections per frame; objects: the non-vehicle detections per frame (set the way the monitor does)."""
    for n, (dets, objs) in enumerate(zip(frames, objects), start):
        tr.objects = list(objs)
        tr.update(dets, SHAPE, now=T0 + n)
    return start + len(frames)


# ---- blocked_map: the rules ----------------------------------------------------------------------------------------------------------------
POLYS = {k["id"]: np.float32(k["poly"]) for k in SLOTS}


def obj(i, label="potted plant", conf=0.7, share=0.5, size=0.04):
    t = thing(i, label, conf, share, size)
    return {"poly": np.float32(t["poly"]) / (W, 1000), "label": label, "conf": conf}


def test_an_object_standing_in_a_free_bay_blocks_it_and_names_what():
    out = blocked_map(POLYS, [obj(1)], {1, 2, 3, 4})
    assert list(out) == [2] and out[2].by == ("potted plant",) and out[2].cover >= CFG.share and out[2].conf == 0.7


def test_only_the_configured_object_classes_count_and_only_when_confident():
    for label in ("person", "dog", "car", "backpack"):
        assert blocked_map(POLYS, [obj(1, label)], {1, 2, 3, 4}) == {}  # not an obstacle class
    for label in ("potted plant", "vase", "suitcase", "chair", "bench"):
        assert 2 in blocked_map(POLYS, [obj(1, label)], {1, 2, 3, 4})
    assert blocked_map(POLYS, [obj(1, conf=0.2)], {1, 2, 3, 4}) == {}
    assert 2 in blocked_map(POLYS, [obj(1, "bottle")], {2}, cfg=BlockedCfg(labels=("bottle",)))  # the classes are configurable


def test_an_object_must_stand_in_the_bay_and_the_one_it_stands_in_most_is_blocked():
    assert blocked_map(POLYS, [obj(1, share=-0.055, size=0.015)], {1, 2, 3, 4}) == {}  # in the gap between bays 1 and 2: in neither
    assert blocked_map(POLYS, [obj(1, share=-0.055, size=0.06)], {1, 2, 3, 4}) == {}  # astride the gap: a third in each, not half of it in one
    assert list(blocked_map(POLYS, [obj(1, share=0.97, size=0.06)], {1, 2, 3, 4})) == [2]  # across the edge of bay 2 into bay 3: bay 2 only


def test_candidates_limit_it_and_hysteresis_keeps_a_blocked_bay_a_little_longer():
    assert blocked_map(POLYS, [obj(1)], {1, 3, 4}) == {}  # bay 2 is not a candidate (a car owns it, it is not FREE ...)
    edge = obj(1, share=1.05, size=0.06)  # about a third of the object inside bay 2
    assert blocked_map(POLYS, [edge], {2, 3}) == {}  # not enough to start blocking ...
    assert 2 in blocked_map(POLYS, [edge], {2, 3}, prev=(2,))  # ... but enough to keep it


def test_the_config_is_validated():
    with pytest.raises(ValueError):
        BlockedCfg(share=0.2, leave=0.3)
    with pytest.raises(ValueError):
        BlockedCfg(min_conf=2)


def test_the_debounce_is_the_spillage_one_with_its_own_event_names():
    t = AvailTracker(events=("object_blocked", "object_cleared"), notes=("object gone", "off"))
    from carparking.blocked import Block
    raw = {2: Block(2, ("potted plant",), 0.9, 0.7)}
    assert t.update(raw, 0.0, 2.0, 3.0, {2}) == []  # not yet: hold seconds AND frames
    t.update(raw, 1.0, 2.0, 3.0, {2})
    ev = t.update(raw, 2.5, 2.0, 3.0, {2})
    assert ev == [{"slot": 2, "event": "object_blocked", "time": 2.5, "note": "by potted plant"}]
    out = [e for n in range(4) for e in t.update({}, 3.0 + n * 2, 2.0, 3.0, {2})]  # gone for 3 frames and 3 s: cleared
    assert out == [{"slot": 2, "event": "object_cleared", "time": 7.0, "note": "object gone"}]
    t2 = AvailTracker()  # the default is still spillage's
    assert t2.events == ("encroached", "encroach_cleared") and t2.notes[1] == "spillage switched off"


# ---- the tracker ---------------------------------------------------------------------------------------------------------------------------
def test_a_pot_in_a_free_bay_blocks_it_for_the_dashboard_but_the_state_stays_free():
    tr = tracker()
    play(tr, [[]] * 12, [[thing(1)]] * 12)
    assert list(tr.block) == [2] and tr.block[2].by == ("potted plant",)
    assert tr.slots[2]["state"] == 0 and tr.shown(2) == 0  # FREE for the history and the sessions
    assert tr.avail_name(2) == "BLOCKED" and tr.avail_name(1) == "FREE"


def test_it_takes_a_while_to_block_and_a_while_to_clear():
    tr = tracker()
    n = play(tr, [[]] * 2, [[thing(1)]] * 2)
    assert tr.block == {}  # two frames and 1 s: not yet
    n = play(tr, [[]] * 6, [[thing(1)]] * 6, n)
    assert 2 in tr.block
    n = play(tr, [[]] * 2, [[]] * 2, n)  # gone for a moment: still blocked
    assert 2 in tr.block
    play(tr, [[]] * 10, [[]] * 10, n)
    assert tr.block == {}


def test_a_bay_a_vehicle_stands_in_is_never_blocked_and_a_pin_is_left_alone():
    tr = tracker()
    n = settled(tr, car_in(1), 12)  # a car parks in bay 2 (BUSY)
    play(tr, [[car_in(1)]] * 8, [[thing(1)]] * 8, n)
    assert tr.block == {} and tr.slots[2]["state"] == 1  # the bay is BUSY: the pot beside the car means nothing
    pin = FixedSlotTracker([*SLOTS[:3], {**SLOTS[3], "capacity": 4}], "ground", hold=2, exit_hold=3, absent_free=True, blocked=1)
    play(pin, [[]] * 12, [[thing(3)]] * 12)
    assert pin.block == {}  # a capacity bay is counted by its vehicles


def test_a_person_or_a_dog_in_a_bay_does_not_block_it():
    tr = tracker()
    play(tr, [[]] * 12, [[thing(1, "person"), thing(2, "dog")]] * 12)
    assert tr.block == {}


def test_off_by_default_nothing_runs_and_the_events_are_what_they_were():
    off, plain = tracker(0), FixedSlotTracker(SLOTS, "ground", hold=2, exit_hold=3, absent_free=True)
    assert off.blocked == 0 and plain.blocked == 0
    for n in range(12):
        off.objects = [thing(1)]  # even if handed over, an off layer ignores it
        a, b = off.update([car_in(0)] if n > 2 else [], SHAPE, now=T0 + n), plain.update([car_in(0)] if n > 2 else [], SHAPE, now=T0 + n)
        assert a == b and off.block == {} and off.block_events == []
    assert off.avail_name(2) == plain.avail_name(2) == "FREE"


def test_switching_it_off_clears_the_blockage_with_an_event_and_on_starts_clean():
    tr = tracker()
    n = play(tr, [[]] * 12, [[thing(1)]] * 12)
    assert 2 in tr.block
    tr.set_blocked(0, now=T0 + n)
    assert tr.block == {} and tr.avail_name(2) == "FREE"
    tr.update([], SHAPE, now=T0 + n)
    assert [(e["slot"], e["event"], e["note"]) for e in tr.block_events] == [(2, "object_cleared", "blocked-object detection switched off")]
    tr.set_blocked(1)
    assert tr.block == {}  # a fresh start


def test_blocked_wins_over_encroached_in_the_shown_name():
    tr = FixedSlotTracker(SLOTS, "ground", hold=2, exit_hold=3, absent_free=True, spillage=1, blocked=1)
    tr.avail = {2: object()}
    tr.block = {2: object()}
    assert tr.avail_name(2) == "BLOCKED"


# ---- the monitor: targets, state(), the event log ---------------------------------------------------------------------------------------
class Scene:
    """A detector that reports whatever the test puts in .dets (the monitor filters by its target classes)."""
    conf = targets = None

    def __init__(self):
        self.dets = []

    def __call__(self, frame, sliced=False):
        return [dict(d) for d in self.dets]


def monitor(tmp_path, scene, **kw):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    return ParkingMonitor(scene, db, fixed=SLOTS, match_rule="ground", hold=2, exit_hold=3, absent_free=True, targets=["car", "truck"], **kw)


def feed(mon, n, start=0):
    for t in range(n):
        mon.process(np.zeros(SHAPE, np.uint8), t=T0 + start + t)
    return start + n


def test_the_monitor_only_sees_object_classes_while_the_layer_is_on():
    mon = ParkingMonitor(Scene(), HistoryDB(":memory:", "cam", detector="fake"), fixed=SLOTS, targets=["car", "truck"])
    mon.set_blocked(False, CFG.labels)
    assert mon.targets == ["car", "truck"]  # never switched on: untouched
    mon.set_blocked(True, CFG.labels)
    assert mon.targets == ["car", "truck", *CFG.labels]
    mon.set_blocked(True, CFG.labels)  # idempotent
    assert mon.targets == ["car", "truck", *CFG.labels]
    mon.set_blocked(False, CFG.labels)
    assert mon.targets == ["car", "truck"]
    free = ParkingMonitor(Scene(), HistoryDB(":memory:", "cam", detector="fake"), fixed=SLOTS, targets=None)
    free.set_blocked(True, CFG.labels)
    assert free.targets is None  # a monitor with no class filter already sees everything


def test_state_shows_the_blocked_bay_out_of_the_free_count_and_the_history_has_no_session(tmp_path):
    scene = Scene()
    mon = monitor(tmp_path, scene, blocked=1)
    mon.start(SHAPE)
    mon.set_blocked(True, CFG.labels)
    scene.dets = [thing(1), thing(1, "person")]
    feed(mon, 12)
    st = mon.state()
    by = {s["id"]: s for s in st["slots"]}
    assert by[2]["state"] == "FREE" and by[2]["avail"] == "BLOCKED" and by[2]["display"] == "BLOCKED" and by[2]["blocked_by"] == ["potted plant"]
    assert by[1]["avail"] is None and by[1]["blocked_by"] == [] and by[1]["display"] == "FREE"
    c = st["counts"]
    assert (c["free"], c["blocked_objects"], c["encroached"], c["available"], c["slots"]) == (3, 1, 0, 3, 4)
    assert mon.db.sessions(limit=10) == [] and mon.db.open_by_slot() == {}  # FREE for the sessions: nothing opened, nothing closed
    assert [(e["event"], e["slot_id"]) for e in mon.db.events(limit=10, kinds=None)] == [("object_blocked", 2)]
    assert "potted plant" in mon.db.events(limit=1, kinds=None)[0]["note"] and "not usable" in mon.db.events(limit=1, kinds=None)[0]["note"]
    assert mon.db.events(limit=10) == []  # entries and exits: none
    scene.dets = []
    feed(mon, 12, 12)
    assert mon.state()["counts"]["free"] == 4 and mon.state()["counts"]["blocked_objects"] == 0
    assert [e["event"] for e in mon.db.events(limit=10, kinds=None)] == ["object_cleared", "object_blocked"]


def test_the_object_is_not_a_vehicle_even_when_the_layer_is_on(tmp_path):
    scene = Scene()
    mon = monitor(tmp_path, scene, blocked=1)
    mon.start(SHAPE)
    mon.set_blocked(True, CFG.labels)
    scene.dets = [thing(1), car_in(0)]
    res = None
    for t in range(12):
        res = mon.process(np.zeros(SHAPE, np.uint8), t=T0 + t)
    assert [d["label"] for d in res.vehicles] == ["car"] and [d["label"] for d in res.others] == ["potted plant"]
    st = {s["id"]: s for s in mon.state()["slots"]}
    assert st[1]["state"] == "BUSY" and st[2]["avail"] == "BLOCKED"


def test_with_the_option_off_the_state_has_none_of_it_and_a_pot_is_not_even_reported(tmp_path):
    scene = Scene()
    off = monitor(tmp_path, scene)
    off.start(SHAPE)
    off.set_blocked(False, CFG.labels)
    scene.dets = [thing(1)]
    feed(off, 12)
    st = off.state()
    assert all("blocked_by" not in s and "avail" not in s for s in st["slots"]) and "blocked_objects" not in st["counts"] and "available" not in st["counts"]
    assert st["counts"]["free"] == 4 and off.db.events(limit=10, kinds=None) == []


def test_a_stale_stream_blocks_nothing(tmp_path):
    scene = Scene()
    mon = monitor(tmp_path, scene, blocked=1)
    mon.start(SHAPE)
    mon.set_blocked(True, CFG.labels)
    scene.dets = [thing(1)]
    feed(mon, 12)
    mon.stream = "frozen"
    st = mon.state()
    assert all(s["avail"] is None and s["blocked_by"] == [] for s in st["slots"]) and st["counts"]["blocked_objects"] == 0


def test_spillage_and_blocked_together_count_each_bay_once(tmp_path):
    scene = Scene()
    mon = monitor(tmp_path, scene, blocked=1, spillage=1)
    mon.start(SHAPE)
    mon.set_blocked(True, CFG.labels)
    scene.dets = [thing(3), car_in(0)]
    feed(mon, 14)
    c = mon.state()["counts"]
    assert (c["busy"], c["blocked_objects"], c["free"]) == (1, 1, 2) and c["available"] == 2 and c["slots"] == 4


# ---- the worker follows the option -----------------------------------------------------------------------------------------------------------
def test_the_worker_applies_the_option_every_frame_without_a_restart():
    opts = {}
    tr = tracker(0)
    mon = SimpleNamespace(tracker=tr, targets=["car"], set_blocked=lambda on, labels: opts.__setitem__("targets", (on, tuple(labels))))
    w = SimpleNamespace(mon=mon, store=SimpleNamespace(option=lambda k, d="": opts.get(k, d)))
    Worker._blocked(w)
    assert tr.blocked == 0 and opts["targets"][0] is False
    opts["blocked_objects"] = "1"
    Worker._blocked(w)
    assert tr.blocked == 1 and opts["targets"] == (True, CFG.labels)
    opts["blocked_objects"] = "0"
    Worker._blocked(w)
    assert tr.blocked == 0 and opts["targets"][0] is False
    Worker._blocked(SimpleNamespace(mon=SimpleNamespace(tracker=None), store=SimpleNamespace(option=lambda k, d="": "1")))  # a learned map: nothing to do
