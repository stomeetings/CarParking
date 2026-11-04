"""SlotTracker / HistoryDB (carparking/history.py) and the Mode 5 URL helper, on synthetic detections."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.history import HistoryDB, SlotTracker, phantom_reason  # noqa: E402
from carparking.rtsp import mask_url  # noqa: E402

SHAPE = (1000, 2000, 3)


def car(x, y, state=1, w=200, h=100):
    return {"poly": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], "state": state}


def run(tracker, frames, start=0.0, step=1.0):
    """frames: list of detection lists, one per second -> all events."""
    events = []
    for i, dets in enumerate(frames):
        events += tracker.update(dets, SHAPE, now=start + i * step)
    return events


def test_parked_car_at_start_is_learned_then_leaves():
    tr = SlotTracker(hold=5, learn_hold=5, absent_free=True)
    ev = run(tr, [[car(100, 100)]] * 12 + [[]] * 8)
    assert list(tr.slots) == [1]
    assert [(e["slot"], e["state"], e["initial"]) for e in ev] == [(1, 1, True), (1, 0, False)]
    assert ev[0]["time"] == 0  # timed from the first sighting, not from when the slot was learned
    assert ev[1]["time"] == 12  # first frame without the car


def test_passing_car_never_becomes_a_slot():
    tr = SlotTracker(hold=5, learn_hold=5, absent_free=True)
    ev = run(tr, [[car(100 + 150 * i, 100)] for i in range(10)] + [[]] * 10)
    assert tr.slots == {} and ev == [] and tr.candidates == []


def test_short_blip_does_not_log_an_exit():
    tr = SlotTracker({1: np.float32([(0.05, 0.1), (0.15, 0.1), (0.15, 0.2), (0.05, 0.2)])}, hold=5,
                     absent_free=True)
    # car parked; the detector misses it for 3 s (person walking past) -> no exit
    ev = run(tr, [[car(100, 100)]] * 8 + [[]] * 3 + [[car(100, 100)]] * 5)
    assert [(e["state"], e["initial"]) for e in ev] == [(1, True)]


def test_late_arrival_is_a_real_entry():
    tr = SlotTracker(hold=5, learn_hold=5, absent_free=True)
    ev = run(tr, [[]] * 20 + [[car(100, 100)]] * 15)
    assert [(e["state"], e["initial"], e["time"]) for e in ev] == [(1, False, 20)]


def test_bay_model_absent_keeps_state():
    tr = SlotTracker(hold=3, learn_hold=3, absent_free=False)
    ev = run(tr, [[car(100, 100, state=1)]] * 6 + [[]] * 10)
    assert [e["state"] for e in ev] == [1]  # missing bay = unknown, not FREE
    ev = run(tr, [[car(100, 100, state=0)]] * 5, start=20)
    assert [e["state"] for e in ev] == [0]


def test_slot_map_is_resolution_independent():
    tr = SlotTracker({1: np.float32([(0.05, 0.1), (0.15, 0.1), (0.15, 0.2), (0.05, 0.2)])}, hold=0)
    # the same car on the 640x360 sub-stream (fractions 0.05..0.15 x 0.1..0.2)
    tr.update([{"poly": [(32, 36), (96, 36), (96, 72), (32, 72)], "state": 1}], (360, 640, 3), now=0)
    ev = tr.update([{"poly": [(32, 36), (96, 36), (96, 72), (32, 72)], "state": 1}], (360, 640, 3), now=1)
    assert [(e["slot"], e["state"]) for e in ev] == [(1, 1)] and tr.candidates == []


REAR = {1: np.float32([(0.40, 0.30), (0.55, 0.30), (0.55, 0.45), (0.40, 0.45)])}  # = car(800, 300, 300, 150)
PICKUP = car(800, 300, w=300, h=150)
VAN = car(850, 350, w=400, h=250)  # in front (bottom 600 > 450), covers 56% of the rear slot, IoU 0.21: no match
BIG_VAN = car(780, 320, w=420, h=330)  # in front (bottom 650), covers 87% of the rear slot (IoU 0.27): hidden
NEIGHBOUR = car(1000, 300, w=300, h=150)  # beside it (same bottom), covers 33%


def occl_tracker(**kw):
    tr = SlotTracker(dict(REAR), hold=5, learn_hold=5, absent_free=True, learn=False, **kw)
    assert [e["state"] for e in run(tr, [[PICKUP]] * 7)] == [1]  # parked when tracking starts
    return tr


def test_vehicle_in_front_does_not_end_the_session():
    tr = occl_tracker(exit_hold=60, hidden_max=120)  # hidden 100 s < the cap
    ev = run(tr, [[BIG_VAN]] * 100 + [[PICKUP]] * 10, start=10)  # rear car hidden 100 s (< hidden_max), visible again
    assert ev == []
    assert tr.slots[1]["state"] == 1


def test_car_that_left_while_hidden_exits_after_buffer():
    tr = occl_tracker(exit_hold=60, hidden_max=120)  # hidden 100 s < the cap
    tr.update([BIG_VAN], SHAPE, now=10)
    assert tr.hidden == {1}
    ev = run(tr, [[BIG_VAN]] * 100 + [[]] * 70, start=10)  # hidden 100 s; the van leaves, slot empty at t=110
    assert [(e["state"], e["time"], e["occluded"]) for e in ev] == [(0, 110, True)]  # first seen empty, flagged


def test_half_behind_the_car_in_front_is_not_hidden():
    # in a low-angle row every bay is about half behind the car in front (live: 42-50 %): when its car leaves, the
    # bay must go FREE after the normal exit buffer, timed at the departure (it used to stay HIDDEN, BUSY for good)
    tr = occl_tracker(exit_hold=60)
    ev = run(tr, [[VAN]] * 70, start=10)  # the pickup left at t=10; the van in front covers 56 % of its bay
    assert tr.hidden == set() and [(e["state"], e["time"]) for e in ev] == [(0, 10)]


def test_hidden_slot_without_a_car_expires_on_its_own():
    tr = occl_tracker(exit_hold=60, hidden_max=120)
    ev = run(tr, [[BIG_VAN]] * 119, start=10)  # the pickup left at t=10, a van parked in front hides the bay
    assert ev == [] and tr.hidden == {1} and tr.slots[1]["state"] == 1  # HIDDEN (kept BUSY) up to hidden_max
    ev = run(tr, [[BIG_VAN]] * 5, start=129)  # hidden_max reached: FREE on its own, no restart
    assert [(e["state"], e["time"], e["occluded"], e["expired"]) for e in ev] == [(0, 10, True, True)]
    assert tr.slots[1]["state"] == 0  # and the exit is timed when the car was last seen, not at the expiry
    run(tr, [[BIG_VAN, PICKUP]] * 8, start=200)  # a car parks there again: a new entry as usual
    assert tr.slots[1]["state"] == 1


def test_default_hidden_max_is_30_seconds():
    tr = occl_tracker(exit_hold=10)  # hidden_max not given: the default
    assert tr.hidden_max == 30
    ev = run(tr, [[BIG_VAN]] * 29, start=10)  # the pickup left at t=10, a van hides its bay
    assert ev == [] and tr.slots[1]["state"] == 1
    ev = run(tr, [[BIG_VAN]] * 3, start=39)  # 30 s without a vehicle in it: FREE, exit timed at t=10
    assert [(e["state"], e["time"], e["expired"]) for e in ev] == [(0, 10, True)]


def test_every_car_leaving_the_live_row_frees_its_slot(tmp_path):
    # the live map after the reset (2026-10-19): any one car leaves while the 6 others stay -> its slot FREE, the
    # session closed at the departure, without a restart. Before: 6 of 7 stuck (HIDDEN or forced BUSY)
    live = {1: (0.077, 0.434, 0.494, 0.840), 2: (0.327, 0.212, 0.531, 0.344), 3: (0.209, 0.331, 0.561, 0.587),
            4: (0.446, 0.181, 0.568, 0.260), 5: (0.332, 0.278, 0.544, 0.384), 6: (0.446, 0.212, 0.530, 0.301),
            7: (0.465, 0.152, 0.568, 0.206)}
    for gone in live:
        db = HistoryDB(tmp_path / f"h{gone}.db", "cam")
        tr = SlotTracker({k: frac(b) for k, b in live.items()}, hold=10, exit_hold=60, absent_free=True,
                         learn=False)
        cars = {k: px(b) for k, b in live.items()}
        for e in run(tr, [list(cars.values())] * 15):
            db.apply(e)
        left = [c for k, c in cars.items() if k != gone]
        evs = run(tr, [left] * 200, start=100)  # the car left at t=100
        for e in evs:
            db.apply(e)
        assert [(e["slot"], e["state"]) for e in evs] == [(gone, 0)], gone
        assert {k for k, s in tr.slots.items() if s["state"] == 1} == set(live) - {gone}
        out = next(s for s in db.sessions() if s["slot_id"] == gone)
        assert out["out_ts"] == 100 and out["status"] == "Available", gone  # out time = the departure


def test_neighbour_beside_is_not_an_occlusion():
    tr = occl_tracker(exit_hold=20)
    ev = run(tr, [[NEIGHBOUR]] * 30, start=10)  # rear car gone, a neighbour at the same depth overlaps the slot
    assert tr.hidden == set()
    assert [(e["state"], e["time"], e["occluded"]) for e in ev] == [(0, 10, False)]


def test_exit_buffer_is_longer_than_entry_hold():
    tr = occl_tracker(exit_hold=60)
    assert run(tr, [[]] * 50 + [[PICKUP]] * 5, start=10) == []  # 50 s missed by the detector: still parked
    ev = run(tr, [[]] * 61, start=100)
    assert [(e["state"], e["time"]) for e in ev] == [(0, 100)]


def test_slot_map_and_history_per_detector(tmp_path):
    path = tmp_path / "h.db"
    bay = HistoryDB(path, "Front Parking 02", detector="bay_obb")
    cam = HistoryDB(path, "Front Parking 02", detector="cam_front_parking_02")
    tr = SlotTracker(dict(REAR), hold=0)
    bay.save_slots(tr, [1])  # junk 'bay' learned by the bay model
    bay.apply({"slot": 1, "state": 1, "time": 0.0, "initial": False})
    assert cam.load_slots() == {} and cam.recent() == []  # the camera model starts clean
    assert not cam.apply({"slot": 1, "state": 0, "time": 9.0, "initial": False})  # can't close the bay model's row
    bay.set_setting("roi", "0,0; 1,0; 1,1")
    assert cam.setting("roi") == "0,0; 1,0; 1,1"  # the parking area is per camera, shared by all detectors
    assert list(HistoryDB(path, "Front Parking 02", detector="bay_obb").load_slots()) == [1]


def test_occluded_exit_note(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "cam")
    db.apply({"slot": 1, "state": 1, "time": 0.0, "initial": False})
    db.apply({"slot": 1, "state": 0, "time": 600.0, "initial": False, "occluded": True})
    assert "hidden behind a vehicle" in db.recent()[0]["note"]
    db.apply({"slot": 2, "state": 1, "time": 0.0, "initial": False})  # exit after hidden_max: timed at last seen
    db.apply({"slot": 2, "state": 0, "time": 50.0, "initial": False, "occluded": True, "expired": True})
    assert db.recent()[0]["note"] == "no vehicle seen in it while partly hidden (exit = last seen)"
    assert db.recent()[0]["duration_min"] == 0.8


def test_history_db_sessions(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "cam")
    tr = SlotTracker(hold=5, learn_hold=5, absent_free=True)
    run(tr, [[car(100, 100)]] * 12)
    db.save_slots(tr, tr.slots)
    assert db.apply({"slot": 1, "state": 1, "time": 1000.0, "initial": True})
    assert not db.apply({"slot": 1, "state": 1, "time": 1100.0, "initial": True})  # restart while parked: same session
    assert db.apply({"slot": 1, "state": 0, "time": 1000.0 + 45 * 60, "initial": False})
    assert not db.apply({"slot": 1, "state": 0, "time": 5000.0, "initial": False})
    (row,) = db.recent()
    assert row["status"] == "Available" and row["duration_min"] == 45.0 and row["exit_time"]
    assert "already parked" in row["note"]
    # the slot map survives a restart, under the camera name only
    assert list(HistoryDB(tmp_path / "h.db", "cam").load_slots()) == [1]
    assert HistoryDB(tmp_path / "h.db", "other").load_slots() == {}
    # reset closes open sessions, keeps the rows, forgets the slots
    db.apply({"slot": 1, "state": 1, "time": 6000.0, "initial": False})
    db.reset(now=6600.0)
    rows = db.recent()
    assert len(rows) == 2 and rows[0]["duration_min"] == 10.0 and db.load_slots() == {}
    assert "slot_id" in db.csv().splitlines()[0]


def test_mask_url():
    assert mask_url("rtsp://user:p%40ss@192.168.20.10:554/a") == "rtsp://***@192.168.20.10:554/a"
    assert mask_url("rtsp://192.168.20.10/a") == "rtsp://192.168.20.10/a"


def test_event_log_links_entries_and_exits_to_sessions(tmp_path):
    import json
    from zoneinfo import ZoneInfo
    db = HistoryDB(tmp_path / "h.db", "Front Parking 02", ZoneInfo("Asia/Karachi"), detector="yolov8s")
    tr = SlotTracker(hold=5, learn_hold=5, exit_hold=10, absent_free=True)
    for e in run(tr, [[car(100, 100)]] * 12 + [[]] * 15 + [[car(100, 100)]] * 8, start=1_760_850_000):
        db.apply(e)
    ev = db.events()
    assert [(e["event"], e["slot_id"]) for e in ev] == [("entry", 1), ("exit", 1), ("entry", 1)]  # newest first
    s = db.sessions()
    assert [(r["status"], r["out_time"] is None) for r in s] == [("Occupied", True), ("Available", False)]
    assert ev[1]["session_id"] == ev[2]["session_id"] == s[1]["id"] and ev[0]["session_id"] == s[0]["id"]
    assert s[1]["in_time"] == "2025-10-19T10:00:00+05:00" and s[1]["in_ts"] == 1_760_850_000  # camera time zone
    assert s[1]["out_ts"] - s[1]["in_ts"] == 12 and s[1]["duration_min"] == 0.2
    assert db.events(since_id=ev[1]["id"]) == ev[:1]  # polling for new events
    assert db.summary() == {"camera": "Front Parking 02", "detector": "yolov8s", "sessions": 2, "open_sessions": 1,
                            "events": 3, "last_event": ev[0]}
    json.dumps([s, ev, db.summary()])  # plain JSON, no numpy / pandas types
    assert db.events_csv().splitlines()[0] == "id,slot_id,event,time,ts,session_id,note"
    db.reset(now=1_760_850_100)
    assert db.events()[0]["event"] == "exit" and db.events()[0]["note"] == "closed by slot map reset"
    assert db.sessions(open_only=True) == []


def test_event_log_backfilled_from_an_older_database(tmp_path):
    import sqlite3
    con = sqlite3.connect(tmp_path / "old.db")  # schema before the event log
    con.executescript("""
        CREATE TABLE history (id INTEGER PRIMARY KEY AUTOINCREMENT, camera TEXT, slot_id INTEGER, entry_time TEXT,
            exit_time TEXT, duration_min REAL, status TEXT, note TEXT, entry_ts REAL, exit_ts REAL);
        INSERT INTO history VALUES (1, 'cam', 3, 'a', 'b', 1.0, 'Available', 'n', 100, 160);
        INSERT INTO history VALUES (2, 'cam', 4, 'c', NULL, NULL, 'Occupied', '', 130, NULL);""")
    con.commit()
    con.close()
    db = HistoryDB(tmp_path / "old.db", "cam")
    assert [(e["event"], e["slot_id"], e["ts"], e["session_id"]) for e in db.events()] == [
        ("exit", 3, 160, 1), ("entry", 4, 130, 2), ("entry", 3, 100, 1)]
    HistoryDB(tmp_path / "old.db", "cam")  # opening again does not backfill twice
    assert len(db.events()) == 3


# Front Parking 02, 2026-10-19 live frame (yolov8s, 1920 px), shares of the frame: a row of cars running away from the
# camera, each box overlapping the next. STRETCHED is the red car's box stretched down over the sedan in front: its
# ground point (bottom centre) lies on the sedan's body, between two cars with no room for a slot.
LIVE = {"hatchback": (0.208, 0.399, 0.495, 0.847), "sedan": (0.250, 0.327, 0.562, 0.587),
        "red": (0.289, 0.276, 0.545, 0.383), "suv": (0.328, 0.212, 0.534, 0.344), "dark": (0.447, 0.212, 0.530, 0.314),
        "grey": (0.443, 0.182, 0.568, 0.261), "far": (0.467, 0.148, 0.568, 0.201)}
STRETCHED = (0.269, 0.279, 0.557, 0.423)


def frac(b):
    return np.float32([(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])])


def px(b):
    x1, y1, x2, y2 = (round(v * s) for v, s in zip(b, (SHAPE[1], SHAPE[0], SHAPE[1], SHAPE[0])))
    return car(x1, y1, w=x2 - x1, h=y2 - y1)


def test_phantom_reason_on_the_live_row():
    boxes = {k: frac(b) for k, b in LIVE.items()}
    for k, p in boxes.items():  # every real car stays, incl. the dark car whose box lies inside the SUV's
        assert phantom_reason(p, [q for j, q in boxes.items() if j != k]) is None, k
    assert phantom_reason(frac(STRETCHED), list(boxes.values())) == "stretched"
    # a later frame: the red car's box is narrower (its visible strip lies inside the SUV's box behind it, but the
    # SUV's box doesn't fit inside the red car's, so the red car stays), and the stretched slot is still caught
    later = dict(LIVE, red=(0.319, 0.277, 0.544, 0.383), sedan=(0.246, 0.324, 0.561, 0.587))
    boxes = {k: frac(b) for k, b in later.items()}
    for k, p in boxes.items():
        assert phantom_reason(p, [q for j, q in boxes.items() if j != k]) is None, k
    assert phantom_reason(frac(STRETCHED), list(boxes.values())) == "stretched"
    # the merge case from test_engine (2560x1440): a car parked behind, its box ~100% inside the front car's box
    front = frac((794 / 2560, 302 / 1440, 1382 / 2560, 547 / 1440))
    behind = frac((998 / 2560, 317 / 1440, 1382 / 2560, 475 / 1440))
    assert phantom_reason(front, [behind]) is None and phantom_reason(behind, [front]) is None


def test_stretched_box_never_becomes_a_free_slot():
    tr = SlotTracker(hold=5, learn_hold=5, absent_free=True)
    real = [px(b) for b in LIVE.values()]
    # the stretched box comes and goes with the real ones for a minute, then only the real cars stay
    ev = run(tr, [real + [px(STRETCHED)] if i % 3 else real for i in range(60)] + [real] * 60)
    assert len(tr.slots) == len(LIVE) and tr.dropped == 0 and tr.phantoms == set()
    assert {e["state"] for e in ev} == {1}  # only entries: no FREE slot anywhere in the row
    tr.update(real + [px(STRETCHED)], SHAPE, now=200)
    assert tr.dropped == 1


def side_row(*xs, w=200, y=500, h=150):
    return [car(x, y, w=w, h=h) for x in xs]


def test_no_room_between_tightly_parked_neighbours():
    a, b = side_row(100, 360)  # ground points 260 px apart: less than 1.5 car widths, no slot fits between
    spanning = car(220, 490, w=220, h=160)  # a box over the right edge of a, the gap and the left edge of b
    as_frac = lambda d: np.float32(d["poly"]) / (SHAPE[1], SHAPE[0])  # noqa: E731
    assert phantom_reason(as_frac(spanning), [as_frac(a), as_frac(b)]) == "no room"
    tr = SlotTracker(hold=5, learn_hold=5, absent_free=True)
    ev = run(tr, [[a, b, spanning]] * 30)
    assert len(tr.slots) == 2 and tr.dropped == 1 and {e["state"] for e in ev} == {1}


def test_real_middle_car_and_a_real_free_bay_between_neighbours():
    # three cars side by side: the middle one is real (its neighbours' ground points are 2.2 widths apart)
    tr = SlotTracker(hold=5, learn_hold=5, exit_hold=10, absent_free=True)
    run(tr, [side_row(100, 320, 540)] * 20)
    assert len(tr.slots) == 3 and tr.dropped == 0
    ev = run(tr, [side_row(100, 540)] * 20, start=20)  # the middle car leaves: its bay really is FREE
    assert [(e["slot"], e["state"], e["time"]) for e in ev] == [(2, 0, 20)] and tr.phantoms == set()


def test_slot_is_learned_only_where_the_ground_point_rests():
    tr = SlotTracker(hold=5, learn_hold=5, absent_free=True)
    # a box whose bottom edge keeps jumping (stretching over the car in front and back): it overlaps itself
    # (IoU >= 0.3) but its ground point never rests, so no slot
    ev = run(tr, [[car(800, 300, w=300, h=150 if i % 2 else 260)] for i in range(30)])
    assert tr.slots == {} and ev == []
    run(tr, [[car(800, 300, w=300, h=150)]] * 10, start=30)
    assert len(tr.slots) == 1


def test_learned_phantom_slot_is_unknown_then_retired_unless_proven():
    real = [px(b) for b in LIVE.values()]
    for proven, kept in ((set(), False), ({3}, True)):
        slots = {1: frac(LIVE["red"]), 2: frac(LIVE["sedan"]), 3: frac(STRETCHED)}
        tr = SlotTracker(slots, hold=2, learn_hold=5, absent_free=True, learn=False, proven=proven,
                         merge_iou=None, cover_share=None)  # the phantom path (by default: a duplicate)
        ev = run(tr, [real] * 3)
        assert tr.phantoms == {3} and tr.slots[3]["state"] is None and 3 not in {e["slot"] for e in ev}
        run(tr, [real] * 5, start=3)
        assert (3 in tr.slots) == kept and (tr.phantoms == {3}) == kept  # proven: kept, but UNKNOWN, never FREE


def test_slot_lying_within_cars_is_never_free_while_covered():
    # live slot 13: the part of the white sedan seen above the hatchback in front (95% inside sedan + hatchback).
    # It used to be forced BUSY ("inside the cars in front"), which also kept emptied bays BUSY for good; now it is
    # HIDDEN (unknown, never FREE) while covered, and FREE once the cars in front leave
    sedan_piece = (0.216, 0.349, 0.481, 0.495)
    # live slot 11: a fragment box (0.03 wide) on the far car: 10 % of the cars at its depth, retired as tiny
    fragment = (0.536, 0.160, 0.566, 0.179)
    own = {i: frac(b) for i, b in enumerate(LIVE.values(), 1)}  # every car has its own slot (1..7), as live
    tr = SlotTracker({**own, 11: frac(fragment), 13: frac(sedan_piece)}, hold=2, exit_hold=2, absent_free=True,
                     learn=False)
    ev = run(tr, [[px(b) for b in LIVE.values()]] * 10)
    # 13 lies on the sedan (slot 2): its duplicate (SlotTracker.covered), unknown, never FREE, until it's merged
    assert 11 not in tr.slots and tr.covered == {13: 2} and tr.slots[13]["state"] is None
    assert {e["slot"] for e in ev if e["state"] == 1} == set(own) and not [e for e in ev if e["slot"] == 13]
    ev = run(tr, [[]] * 5, start=10)  # the cars leave: the spots are visible and empty
    assert {e["slot"] for e in ev if e["state"] == 0} == set(own) | {13}
    # a slot only partly behind a car in front (live slot 4 = the red car's bay, 55% behind the sedan) stays FREE
    tr = SlotTracker({4: frac(LIVE["red"])}, hold=2, absent_free=True, learn=False)
    ev = run(tr, [[px(LIVE["sedan"]), px(LIVE["hatchback"])]] * 5)
    assert [(e["slot"], e["state"]) for e in ev] == [(4, 0)] and tr.shared == set()


def test_one_vehicle_box_occupies_one_slot_only():
    # live slot 12 (2026-10-19): learned from the rear half of the white sedan. The sedan's box matches slot 2;
    # one box, one slot: it doesn't also hold slot 12 (item 29 had it do so, which let neighbours' boxes keep bays
    # BUSY after their own car left)
    rear_half = (0.421, 0.331, 0.561, 0.582)
    sedan = px(LIVE["sedan"])
    tr = SlotTracker({2: frac(LIVE["sedan"]), 12: frac(rear_half)}, hold=2, exit_hold=10, absent_free=True,
                     learn=False)
    # slot NMS: IoU > 0.35 at the sedan's depth = the same space: merged into slot 2 as the map loads (it used to wait
    # as UNKNOWN for cover_hold), never FREE on the sedan and never BUSY with it
    assert tr.merged == {12: 2} and list(tr.slots) == [2]
    ev = run(tr, [[sedan]] * 5)
    assert [(e["slot"], e["state"]) for e in ev] == [(2, 1)] and tr.shared == set()
    # a box that matched no slot (shape changed: IoU < 0.3 with its bay) still occupies the bay holding its centre,
    # and only the best one of the bays it qualifies for
    a = np.float32([(0.10, 0.50), (0.20, 0.50), (0.20, 0.70), (0.10, 0.70)])
    b = np.float32([(0.20, 0.50), (0.30, 0.50), (0.30, 0.70), (0.20, 0.70)])
    tr = SlotTracker({1: a, 2: b}, hold=2, absent_free=True, learn=False)
    wide = car(120, 300, w=240, h=500)  # x 0.06-0.18, y 0.30-0.80: centre (0.12, 0.55) in bay 1, IoU 0.25 < 0.3
    ev = run(tr, [[wide]] * 5)
    assert sorted((e["slot"], e["state"]) for e in ev) == [(1, 1), (2, 0)] and tr.shared == {1}


def test_bike_leaving_frees_its_bay_despite_the_neighbours_box():
    # live map 2026-10-19 evening: slot 3 = the motorbike, slot 1 = the maroon car beside it whose box covers 44 %
    # of the bike's bay (IoU 0.14, centre outside). When the bike left, slot 3 stayed "3 BUSY"
    live = {1: (0.331, 0.276, 0.543, 0.447), 2: (0.326, 0.212, 0.532, 0.346), 3: (0.276, 0.339, 0.390, 0.461),
            4: (0.444, 0.211, 0.529, 0.308), 5: (0.437, 0.144, 0.568, 0.217), 6: (0.476, 0.181, 0.567, 0.260)}
    dets = {1: (0.332, 0.277, 0.544, 0.445), 2: (0.326, 0.212, 0.534, 0.344), 3: (0.277, 0.335, 0.390, 0.465),
            4: (0.443, 0.211, 0.529, 0.308), 5: (0.470, 0.149, 0.567, 0.206), 6: (0.442, 0.181, 0.567, 0.258)}
    tr = SlotTracker({k: frac(b) for k, b in live.items()}, hold=10, exit_hold=60, absent_free=True, learn=False)
    run(tr, [[px(b) for b in dets.values()]] * 15)
    assert {k for k, s in tr.slots.items() if s["state"] == 1} == set(live)
    ev = run(tr, [[px(b) for k, b in dets.items() if k != 3]] * 70, start=100)  # the bike leaves at t=100
    assert [(e["slot"], e["state"], e["time"]) for e in ev] == [(3, 0, 100)]
    assert tr.slots[3]["state"] == 0 and tr.slots[1]["state"] == 1 and 3 not in tr.shared | tr.hidden


def test_shared_occupancy_is_not_triggered_by_cars_beside_in_front_or_behind():
    slot = {1: np.float32([(0.40, 0.30), (0.55, 0.30), (0.55, 0.45), (0.40, 0.45)])}  # = car(800, 300, w=300, h=150)
    for other in (NEIGHBOUR,  # beside, covers 33%: a neighbour's box spilling over, not a car in this bay
                  VAN,  # in front: occlusion (hidden), not occupancy
                  car(820, 200, w=260, h=180)):  # behind (bottom 380, above the slot's 450): its box overlaps 53%
        tr = SlotTracker({k: v.copy() for k, v in slot.items()}, hold=2, absent_free=True, learn=False)
        run(tr, [[other]] * 6)
        assert tr.shared == set() and tr.slots[1]["state"] != 1, other


def test_is_tiny_is_relative_to_the_vehicles_at_that_depth():
    from carparking.history import is_tiny
    boxes = {k: frac(b) for k, b in LIVE.items()}
    for k, p in boxes.items():  # every real car of the live row, the far ones and the dark car (28 %) included
        assert not is_tiny(p, [q for j, q in boxes.items() if j != k]), k
    stray = frac((0.292, 0.352, 0.325, 0.394))  # live slot 8: 6 % of the cars at its depth
    assert is_tiny(stray, list(boxes.values()))
    assert not is_tiny(stray, [boxes["hatchback"]])  # no reference at its depth: can't judge
    far_alone = frac((0.47, 0.15, 0.57, 0.21))  # a lone far car beside big near ones is not tiny
    assert not is_tiny(far_alone, [boxes["hatchback"], boxes["sedan"]])


def test_tiny_detections_are_never_learned_and_never_busy():
    tr = SlotTracker(hold=2, learn_hold=3, absent_free=True)
    real = [px(b) for b in LIVE.values()]
    stray = px((0.292, 0.352, 0.325, 0.394))
    ev = run(tr, [real + [stray]] * 20)
    assert len(tr.slots) == len(LIVE) and tr.dropped == 1
    assert sum(e["state"] == 1 for e in ev) == len(LIVE)  # BUSY = one per real vehicle


def test_retired_slots_stay_in_the_database(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="yolov8s")
    tr = SlotTracker({1: frac(STRETCHED), 2: frac(LIVE["red"])}, hold=0, merge_iou=None)
    db.save_slots(tr, [1, 2])
    db.apply({"slot": 1, "state": 1, "time": 0.0, "initial": True})
    assert db.proven_slots() == {1}
    db.retire_slots([1], now=60.0)
    assert list(db.load_slots()) == [2] and db.max_slot_id() == 2
    assert db.sessions()[0]["out_ts"] == 60.0 and "phantom" in db.sessions()[0]["note"]


def as_frac(d):
    return np.float32(d["poly"]) / (SHAPE[1], SHAPE[0])


def test_candidate_overlapping_a_slot_never_becomes_a_second_slot():
    # one sedan, two boxes each frame (the detector split it; merge_dets missed it): the second box overlaps slot 1
    # by IoU 0.6 but loses the one-to-one match, so it used to be learned as a duplicate slot over the same car
    frames = [[car(100, 100), car(150, 100)]] * 30
    old = SlotTracker(hold=5, learn_hold=5, absent_free=True, dup_iou=None, near_iou=None, merge_iou=None)
    run(old, frames)
    assert len(old.slots) == 2  # the bug: slot 1 and slot 2 over one car
    tr = SlotTracker(hold=5, learn_hold=5, absent_free=True)
    ev = run(tr, frames)
    assert list(tr.slots) == [1] and [(e["slot"], e["state"]) for e in ev] == [(1, 1)]


def test_narrower_box_of_a_parked_car_is_not_a_new_slot():
    tr = SlotTracker({1: as_frac(car(100, 100, w=200, h=120))}, hold=5, learn_hold=5, absent_free=True)
    # the car's box shrinks to its visible half (IoU 0.25 < match_iou): lies inside slot 1 at its depth
    ev = run(tr, [[car(100, 100, w=200, h=120)]] * 10 + [[car(100, 100, w=50, h=120)]] * 30)
    assert list(tr.slots) == [1] and tr.slots[1]["state"] == 1
    assert [(e["slot"], e["state"]) for e in ev] == [(1, 1)]


def test_car_parking_in_a_bike_bay_grows_the_slot_instead_of_a_duplicate():
    bike, sedan = car(300, 400, w=60, h=100), car(250, 380, w=200, h=120)  # same ground line (y 500)
    tr = SlotTracker(hold=3, learn_hold=5, exit_hold=5, absent_free=True)
    run(tr, [[bike]] * 15)
    assert list(tr.slots) == [1] and tr.slots[1]["state"] == 1
    run(tr, [[]] * 10, start=15)  # the bike leaves
    assert tr.slots[1]["state"] == 0
    grown, events = [], []
    for i in range(20):  # a car parks there: IoU 0.25 with the bike's bay, which lies wholly inside the car's box
        events += tr.update([sedan], SHAPE, now=25 + i)
        grown += tr.updated_slots
    assert list(tr.slots) == [1] and grown == [1] and tr.new_slots == []
    assert np.allclose(tr.slots[1]["poly"], as_frac(sedan)) and tr.slots[1]["state"] == 1
    assert [(e["slot"], e["state"]) for e in events] == [(1, 1)]


def test_overlapping_slots_are_merged_on_load():
    a, twin, other = car(100, 100), car(130, 100), car(600, 100)  # twin: IoU 0.74 with a
    slots = {1: as_frac(a), 2: as_frac(twin), 3: as_frac(other)}
    tr = SlotTracker(slots)
    assert sorted(tr.slots) == [1, 3] and tr.merged == {2: 1} and "slot 1" in tr.retire_notes[2]
    tr = SlotTracker(slots, proven={2})  # the one that has held a car is the one kept
    assert sorted(tr.slots) == [2, 3] and tr.merged == {1: 2}
    # IoU 0.45 at the same depth (> near_iou 0.35): one space learned twice, merged; unless the two have held two
    # vehicles at once
    assert SlotTracker({1: as_frac(a), 2: as_frac(car(176, 100))}).merged == {2: 1}
    assert SlotTracker({1: as_frac(a), 2: as_frac(car(176, 100))}, distinct=[(1, 2)]).merged == {}
    assert SlotTracker({1: as_frac(a), 2: as_frac(car(200, 100))}).merged == {}  # IoU 0.33: below the line
    # the same overlap with one slot further back (a car behind another one): two spaces, kept
    assert SlotTracker({1: as_frac(a), 2: as_frac(car(130, 130))}).merged == {}  # IoU 0.42, ground 0.3 x h back


def test_one_car_box_never_turns_two_overlapping_slots_busy():
    a, b, box = car(100, 100), car(176, 100), car(138, 100)  # slots: IoU 0.45 (both kept); box: IoU 0.68 with each
    for kw in ({"distinct": [(1, 2)]}, {"merge_iou": None, "dup_iou": None}):  # two real bays (they held 2 cars)
        tr = SlotTracker({1: as_frac(a), 2: as_frac(b)}, hold=3, learn_hold=5, absent_free=True, **kw)
        events = []
        for i in range(30):
            events += tr.update([box], SHAPE, now=i)
            assert sum(s["state"] == 1 for s in tr.slots.values()) <= 1
        assert len(tr.slots) == 2 and [e["state"] for e in events if e["state"] == 1] == [1]
        assert {s["state"] for s in tr.slots.values()} - {1} <= {0, None}  # the other one: FREE (or UNKNOWN)


def box(b):
    return frac(b)


def test_slot_lying_on_another_vehicle_is_merged_into_its_slot(tmp_path):
    # Front Parking 02, 2026-10-20 09:37: the white sedan (slot 11) parked further forward over the maroon car's
    # old bay (slot 1, 74 % inside the sedan's box, IoU 0.27) and the motorbike's (slot 3): three slots, one car
    db = HistoryDB(tmp_path / "h.db", "cam", detector="yolov8s")
    slots = {1: box((0.331, 0.277, 0.544, 0.446)), 3: box((0.277, 0.337, 0.389, 0.463)),
             11: box((0.252, 0.333, 0.569, 0.594))}
    sedan = px((0.232, 0.321, 0.564, 0.590))
    tr = SlotTracker(slots, hold=3, learn_hold=5, absent_free=True, proven={1, 3, 11})
    tr.update([], SHAPE, now=0)  # slot 1 BUSY from an earlier (double-counted) entry
    tr.slots[1]["state"] = 1
    db.apply({"slot": 1, "state": 1, "time": 0.0, "initial": True})
    states, retired = [], {}
    exits = []
    for t in range(1, 130):
        for e in tr.update([sedan], SHAPE, now=t):
            db.apply(e)
            exits += [(t, e["slot"], e["time"])] if e["state"] == 0 else []
        states.append({k: s["state"] for k, s in tr.slots.items()})
        retired.update(tr.retire_notes)
        if t == 10:
            assert tr.covered == {3: 11} and 1 in tr.hidden  # BUSY: no duplicate until its session is closed
    # slot 1's (double-counted) session ends like any car that left a hidden bay: hidden_max 30 s, exit at t=1
    assert exits == [(31, 1, 1)] and [r["out_ts"] for r in db.sessions() if r["slot_id"] == 1] == [1]
    assert all(st.get(3) != 0 for st in states) and sum(st.get(1) == 0 for st in states) <= 1  # FREE one frame
    assert list(tr.slots) == [11] and tr.slots[11]["state"] == 1
    assert retired == {1: "duplicate slot merged into slot 11 (it lies on that vehicle)",
                       3: "duplicate slot merged into slot 11 (it lies on that vehicle)"}


def test_bay_behind_a_parked_car_is_not_a_duplicate():
    # the live row: the red car's bay is half covered by the white sedan in front (52 %): a real bay of its own
    slots = {1: frac(LIVE["red"]), 2: frac(LIVE["sedan"])}
    tr = SlotTracker(slots, hold=3, learn_hold=5, exit_hold=5, absent_free=True, hidden_max=10)
    ev = run(tr, [[px(LIVE["red"]), px(LIVE["sedan"])]] * 10 + [[px(LIVE["sedan"])]] * 90)
    assert tr.covered == {} and sorted(tr.slots) == [1, 2]
    assert [(e["slot"], e["state"]) for e in ev if not e["initial"]] == [(1, 0)]  # the red car's exit


def test_slots_seen_with_two_cars_at_once_are_never_duplicates(tmp_path):
    # the deep row: the dark car stands wholly inside the SUV's box. Both were parked at once, so when the dark car
    # leaves its bay is a space of its own (FREE), never merged into the SUV's slot, also after a restart
    db = HistoryDB(tmp_path / "h.db", "cam", detector="yolov8s")
    slots = {1: frac(LIVE["suv"]), 2: frac(LIVE["dark"])}
    tr = SlotTracker(slots, hold=2, exit_hold=5, absent_free=True, learn=False)
    run(tr, [[px(LIVE["suv"]), px(LIVE["dark"])]] * 3)
    db.add_distinct_pairs(tr.distinct)
    assert db.distinct_pairs() == {(1, 2)}
    tr = SlotTracker(slots, hold=2, exit_hold=5, absent_free=True, learn=False, distinct=db.distinct_pairs())
    run(tr, [[px(LIVE["suv"])]] * 80)
    assert tr.covered == {} and sorted(tr.slots) == [1, 2] and tr.slots[2]["state"] == 0
    tr = SlotTracker(slots, hold=2, exit_hold=5, absent_free=True, learn=False)  # never seen together
    run(tr, [[px(LIVE["suv"])]] * 80)
    assert list(tr.slots) == [1]  # merged: one car, one slot


def test_hidden_car_seen_on_some_frames_only_turns_busy():
    # the car behind the white sedan: the detector finds it on one frame in three, the others leave its slot HIDDEN
    # behind the van in front. Hidden frames no longer restart the pending entry, so it turns BUSY after hold
    tr = SlotTracker(dict(REAR), hold=10, absent_free=True, learn=False)
    ev = run(tr, [[BIG_VAN, PICKUP] if i % 3 == 0 else [BIG_VAN] for i in range(20)])
    assert [(e["slot"], e["state"], e["time"]) for e in ev] == [(1, 1, 0)] and tr.slots[1]["state"] == 1
    # one sighting, then nothing but the van for longer than the window (max(hold, seen_window 15 s)): no entry (no
    # ghost from a single box)
    tr = SlotTracker(dict(REAR), hold=10, absent_free=True, learn=False)
    ev = run(tr, [[BIG_VAN, PICKUP]] + [[BIG_VAN]] * 17 + [[BIG_VAN, PICKUP]] + [[BIG_VAN]] * 5)
    assert ev == [] and tr.slots[1]["state"] is None


def test_short_entry_hold_keeps_the_hidden_car_window():
    # a short hold (fast entries) must not lose the hidden car seen on one frame in four: the window in which hidden
    # frames don't restart its pending entry stays seen_window (15 s) long
    frames = [[BIG_VAN, PICKUP] if i % 4 == 0 else [BIG_VAN] for i in range(20)]
    tr = SlotTracker(dict(REAR), hold=2, absent_free=True, learn=False)
    ev = run(tr, frames)
    assert [(e["slot"], e["state"], e["time"]) for e in ev] == [(1, 1, 0)]
    tr = SlotTracker(dict(REAR), hold=2, absent_free=True, learn=False, seen_window=0)  # the window = hold (before)
    assert run(tr, frames) == [] and tr.slots[1]["state"] is None


def test_reshaped_box_of_the_slots_own_car_still_occupies_it():
    # the rear car's box suddenly much taller (cut by a shadow, merged with the wall behind): IoU 0.27 < match_iou,
    # its centre above the slot; but it stands in the slot's column at its depth and covers 60 % of it: still BUSY
    tall = car(860, 0, w=180, h=455)
    tr = SlotTracker(dict(REAR), hold=2, exit_hold=3, absent_free=True, learn=False)
    run(tr, [[PICKUP]] * 5 + [[tall]] * 10)
    assert tr.slots[1]["state"] == 1 and tr.shared == {1}
    old = SlotTracker(dict(REAR), hold=2, exit_hold=3, absent_free=True, learn=False, share_cover=0.5,
                      claim_cover=None)  # centre or IoU > 0.5 only (before)
    run(old, [[PICKUP]] * 5 + [[tall]] * 10)
    assert old.slots[1]["state"] == 0


def test_busy_slot_needs_consecutive_empty_frames_before_free():
    frames = [[PICKUP]] * 5 + [[]] * 10 + [[PICKUP]] + [[]] * 20  # missed for 10 frames, seen once, then gone
    tr = SlotTracker(dict(REAR), hold=2, exit_hold=3, absent_free=True, learn=False, exit_frames=15)
    ev = run(tr, frames)
    assert [(e["state"], e["time"]) for e in ev if not e["initial"]] == [(0, 16)]  # 15 empty frames after the blip
    tr = SlotTracker(dict(REAR), hold=2, exit_hold=3, absent_free=True, learn=False)  # exit_frames=0: time only
    ev = run(tr, frames)
    assert [(e["state"], e["time"]) for e in ev if not e["initial"]][0] == (0, 5)


def test_duplicate_slot_is_merged_while_running_not_only_on_load():
    # two slots of one space (IoU 0.45, never two vehicles at once), e.g. learned before the 0.35 rule: the next frame
    # merges them; the one holding the car stays, the duplicate is retired with the reason
    a, twin = car(100, 100), car(176, 100)
    tr = SlotTracker({1: as_frac(a)}, hold=2, learn_hold=5, absent_free=True)
    tr.slots[2] = {"poly": as_frac(twin), "state": None, "cand": None, "since": 0.0, "occluded": False}
    run(tr, [[a]] * 5)
    assert list(tr.slots) == [1] and tr.slots[1]["state"] == 1
    tr = SlotTracker({1: as_frac(a)}, hold=2, learn_hold=5, absent_free=True)
    tr.slots[2] = {"poly": as_frac(twin), "state": None, "cand": None, "since": 0.0, "occluded": False}
    tr.update([a], SHAPE, now=0)
    assert tr.retired == [2] and "merged into slot 1" in tr.retire_notes[2]


def test_car_parked_behind_another_one_gets_its_own_slot_and_both_count():
    # Front Parking 02, 2026-10-21: the silver car (ground y 0.86) and the white sedan parked right behind it
    # (ground y 0.57): the sedan's box overlaps the silver car's slot heavily (IoU 0.36 here > dup_iou), but the two
    # stand at once, at different depths: two vehicles, two slots, both BUSY; never merged afterwards
    silver, sedan = px((0.176, 0.437, 0.486, 0.859)), px((0.234, 0.331, 0.541, 0.567))
    tr = SlotTracker({20: as_frac(silver)}, hold=2, learn_hold=5, absent_free=True)
    tr.next_id = 21
    ev = run(tr, [[silver, sedan]] * 15)
    assert sorted(tr.slots) == [20, 21] and {tr.slots[k]["state"] for k in tr.slots} == {1}
    assert frozenset((20, 21)) in tr.distinct
    assert sorted(e["slot"] for e in ev if e["state"] == 1) == [20, 21]
    run(tr, [[silver, sedan]] * 30, start=15)
    assert sorted(tr.slots) == [20, 21]  # kept: they have held two vehicles at once
    assert sum(s["state"] == 1 for s in tr.slots.values()) == 2  # two cars, two BUSY: correct, not a double count


def test_split_box_of_one_car_never_makes_a_second_busy_slot():
    # one car, two boxes every frame at one depth (IoU 0.6: the detector's NMS missed the split): one slot, one BUSY
    one, split = car(100, 100), car(150, 100)
    tr = SlotTracker(hold=2, learn_hold=5, absent_free=True)
    run(tr, [[one, split]] * 40)
    assert list(tr.slots) == [1] and tr.slots[1]["state"] == 1 and not tr.distinct


def test_one_box_marks_only_its_highest_iou_slot_busy():
    # two real bays overlapping (held two cars at once); one car standing between them: only the slot with the
    # highest IoU turns BUSY, the other stays FREE
    a, b, box = car(100, 100), car(220, 100), car(140, 100)  # box: IoU 0.67 with a, 0.43 with b
    tr = SlotTracker({1: as_frac(a), 2: as_frac(b)}, hold=2, learn_hold=5, absent_free=True, distinct=[(1, 2)])
    run(tr, [[box]] * 20)
    assert tr.slots[1]["state"] == 1 and tr.slots[2]["state"] == 0


def test_free_slot_with_another_vehicle_standing_in_it_is_shown_unknown():
    # Front Parking 02, 2026-10-21: an unmarked lot, slots learned from earlier cars overlap. The pickup (slot 6) now
    # stands with its wheels in slot 5's floor; the hatchback (slot 12) in slot 2's: those are not free spaces.
    # Shown UNKNOWN ("blocked"), never FREE, and never BUSY (the car counts once, in its own slot)
    slots = {2: box((0.327, 0.213, 0.535, 0.344)), 5: box((0.444, 0.181, 0.568, 0.260)),
             6: box((0.438, 0.143, 0.568, 0.214)), 12: box((0.420, 0.215, 0.530, 0.301))}
    pickup, hatchback = px((0.437, 0.144, 0.568, 0.237)), px((0.398, 0.210, 0.528, 0.310))
    tr = SlotTracker(slots, hold=2, exit_hold=2, absent_free=True, learn=False, distinct=[(2, 12), (5, 6), (2, 5)])
    run(tr, [[pickup, hatchback]] * 5)
    assert {k: s["state"] for k, s in tr.slots.items()} == {2: 0, 5: 0, 6: 1, 12: 1}  # the history: unchanged
    assert tr.unseen == {2: "blocked", 5: "blocked"}
    run(tr, [[pickup]] * 5, start=5)  # the hatchback leaves: slot 2's floor is empty and visible again
    assert 2 not in tr.unseen and tr.slots[2]["state"] == 0


def test_free_slot_behind_vehicles_is_never_shown_free_after_hidden_max():
    # the far end: a slot wholly behind the car in front. After hidden_max its session closes (FREE, timed at the
    # departure, as before), but nobody can see it: shown UNKNOWN ("hidden") until the car in front leaves
    tr = occl_tracker(exit_hold=10)
    ev = run(tr, [[BIG_VAN]] * 40, start=10)
    assert [(e["state"], e["time"]) for e in ev] == [(0, 10)] and tr.slots[1]["state"] == 0
    assert tr.unseen == {1: "hidden"}
    run(tr, [[]] * 2, start=50)
    assert tr.unseen == {}  # visible and empty: FREE


def test_car_behind_or_beside_does_not_block_a_free_bay():
    # a car parked behind the bay stands above its floor, a neighbour beside it outside its width (IoU 0.18, under
    # vehicle_iou 0.2): the bay is FREE
    tr = SlotTracker(dict(REAR), hold=2, absent_free=True, learn=False)
    behind = car(800, 150, w=300, h=150)  # ground y 300 = the bay's top edge
    run(tr, [[behind, car(1010, 300, w=300, h=150)]] * 5)
    assert tr.slots[1]["state"] == 0 and tr.unseen == {}


def test_video_banner_and_dashboard_use_the_same_shown_states():
    # one source: view() (the video banner / overlay) == shown() (the monitor's state() = the dashboard's cards)
    slots = {2: box((0.327, 0.213, 0.535, 0.344)), 5: box((0.444, 0.181, 0.568, 0.260)),
             6: box((0.438, 0.143, 0.568, 0.214)), 12: box((0.420, 0.215, 0.530, 0.301))}
    tr = SlotTracker(slots, hold=2, exit_hold=2, absent_free=True, learn=False, distinct=[(2, 12), (5, 6), (2, 5)])
    run(tr, [[px((0.437, 0.144, 0.568, 0.237)), px((0.398, 0.210, 0.528, 0.310))]] * 5)
    _, states = tr.view(SHAPE)
    assert states == {k: tr.shown(k) for k in tr.slots} and states[2] is None and states[5] is None


def test_one_vehicle_never_makes_two_busy_slots():
    # Front Parking 02, 2026-10-22: the pickup's box matches slot 6; slot 5 (an earlier car's spot, BUSY from then)
    # has no box of its own while the pickup stands in it: shown UNKNOWN, not a second BUSY for the pickup
    slots = {5: box((0.444, 0.181, 0.568, 0.260)), 6: box((0.438, 0.143, 0.568, 0.214))}
    tr = SlotTracker(slots, hold=2, exit_hold=600, absent_free=True, learn=False, distinct=[(5, 6)])
    for k in slots:
        tr.slots[k]["state"] = 1
    pickup = px((0.437, 0.144, 0.568, 0.237))
    run(tr, [[pickup]] * 5)
    assert tr.slots[5]["state"] == 1  # the history keeps its session until the exit buffer runs out
    assert tr.shown(6) == 1 and tr.shown(5) is None and tr.unseen[5] == "on the vehicle of slot 6"
    # a car really parked there again (its own box): BUSY again
    run(tr, [[pickup, px((0.444, 0.181, 0.568, 0.260))]] * 2, start=5)
    assert tr.shown(5) == 1 and 5 not in tr.unseen


def test_busy_car_hidden_behind_the_car_in_front_is_not_on_that_vehicle():
    # Front Parking 02, 2026-10-22: the white hatchback (slot 1) behind the white sedan (slot 11), both one box in the
    # whole frame. The hatchback's BUSY must not read "on the vehicle of slot 11" (the sedan stands IN FRONT of it,
    # and the two slots have held two cars at once); a pair never seen together still does (one box, one BUSY)
    slots = {1: box((0.331, 0.277, 0.544, 0.446)), 11: box((0.252, 0.333, 0.569, 0.594))}
    merged = px((0.248, 0.250, 0.556, 0.593))
    for distinct, shown in (([(1, 11)], 1), ([], None)):
        tr = SlotTracker(slots, hold=2, exit_hold=600, hidden_max=2, absent_free=True, learn=False, distinct=distinct)
        for k in slots:
            tr.slots[k]["state"] = 1
        run(tr, [[merged]] * 5)  # past hidden_max: no longer "hidden", the display rules decide
        assert tr.slots[1]["state"] == 1 and tr.shown(11) == 1
        assert tr.shown(1) == shown, distinct


def test_one_hidden_car_cut_differently_by_the_zoom_never_makes_a_second_busy_slot():
    # Front Parking 02, 2026-10-22 (fractions): the white hatchback hidden behind the sedan, BUSY in slot 2 (its box
    # then: its upper part with the silver car's roof). The zoom finds it again cut at the sedan's roof line (slot 1's
    # box) or with the sedan's windows (slot 21's box): those boxes are its car, slot 2 stays BUSY, 1 and 21 never
    # enter (before: each turned BUSY in turn, 7 BUSY for 6 cars, short false sessions)
    slots = {1: box((0.331, 0.277, 0.544, 0.446)), 2: box((0.327, 0.213, 0.535, 0.344)),
             21: box((0.163, 0.348, 0.471, 0.504))}
    cut_roof, with_windows = px((0.263, 0.259, 0.554, 0.444)), px((0.229, 0.293, 0.549, 0.503))
    tr = SlotTracker(slots, hold=2, exit_hold=30, absent_free=True, learn=False, distinct=[(1, 2), (1, 21), (2, 21)])
    tr.slots[2].update(state=1, last=box((0.332, 0.211, 0.558, 0.390)))
    for k in (1, 21):
        tr.slots[k]["state"] = 0
    ev = run(tr, [[cut_roof] if i % 2 == 0 else [with_windows] for i in range(30)])
    assert ev == [] and {k: s["state"] for k, s in tr.slots.items()} == {1: 0, 2: 1, 21: 0}
    assert tr.shown(2) == 1 and tr.shown(1) != 1 and tr.shown(21) != 1
    # a second, separate car next to a BUSY slot that keeps its own box: it enters
    sedan = px((0.248, 0.250, 0.556, 0.593))
    tr = SlotTracker({**slots, 11: box((0.252, 0.333, 0.569, 0.594))}, hold=2, absent_free=True, learn=False)
    tr.slots[11]["state"] = 1
    ev = run(tr, [[sedan, cut_roof]] * 6)
    assert [(e["slot"], e["state"]) for e in ev if not e["initial"]] == [] and tr.slots[1]["state"] == 1


def test_free_slot_overlapping_a_vehicle_box_is_never_green():
    # a FREE slot whose ground a vehicle box covers (> 40 % of the footprint, at its depth; the car's wheels beside
    # it, its centre outside): shown UNKNOWN
    slot, car_box = box((0.30, 0.40, 0.50, 0.55)), px((0.40, 0.38, 0.66, 0.549))  # 50 % of the ground
    tr = SlotTracker({1: slot}, hold=2, absent_free=True, learn=False)
    tr.slots[1]["state"] = 0
    run(tr, [[car_box]] * 3)
    assert tr.slots[1]["state"] == 0 and tr.shown(1) is None and tr.unseen[1] == "overlaps a vehicle"
    run(tr, [[]] * 2, start=3)
    assert tr.shown(1) == 0  # the car gone: FREE


def test_slot_is_not_green_while_its_cars_entry_is_being_confirmed():
    # a car parks in a FREE bay: its box matches the slot at once, the entry is confirmed after hold. Meanwhile the
    # slot is shown UNKNOWN, never FREE under the car (Front Parking 02, slot 22: green under its car for ~20 s)
    tr = SlotTracker({1: frac((0.40, 0.30, 0.55, 0.45))}, hold=10, absent_free=True, learn=False)
    tr.slots[1]["state"] = 0
    run(tr, [[PICKUP]] * 3)
    assert tr.slots[1]["state"] == 0 and tr.shown(1) is None and "confirmed" in tr.unseen[1]
    run(tr, [[PICKUP]] * 10, start=3)
    assert tr.shown(1) == 1


def test_visible_empty_bay_is_free_even_partly_behind_a_car_in_front():
    # the user's rule: UNKNOWN only when vehicles in front cover > 80 % of the bay's ground footprint. A bay whose
    # car left, 58 % of its ground behind the car in front: FREE after the exit buffer, also after hidden_max
    tr = SlotTracker(dict(REAR), hold=2, exit_hold=5, absent_free=True, learn=False, hidden_max=10)
    run(tr, [[PICKUP]] * 5)
    front = car(800, 400, w=210, h=250)  # bottom 650 > the bay's 450: in front; covers 58 % of its ground
    run(tr, [[front]] * 30, start=5)
    assert tr.slots[1]["state"] == 0 and tr.shown(1) == 0 and tr.unseen == {}
    tr.update([car(780, 320, w=420, h=330)], SHAPE, now=40)  # a van covering all of its ground: can't see it
    assert tr.shown(1) is None and tr.unseen == {1: "hidden"}


def test_display_thresholds_can_be_tuned_at_run_time():
    from carparking.history import DISPLAY_TUNING, tuning_values
    assert tuning_values(None) == {k: v[0] for k, v in DISPLAY_TUNING.items()}  # the current behaviour by default
    assert tuning_values({"hidden_ground": 0.1, "ground_share": "0.5", "vehicle_iou": float("nan")}) == {
        "hidden_ground": 0.5, "ground_share": 0.5, "vehicle_iou": 0.2, "bay_share": 0.5, "pin_reach": 0.25}
    assert tuning_values({"pin_reach": 0.9})["pin_reach"] == 0.5 and tuning_values({"pin_reach": "x"})["pin_reach"] == 0.25  # clamped / default
    assert tuning_values({"pin_reach": 0})["pin_reach"] == 0.0  # 0 is a value (strictly inside), not "missing"
    # the bay's ground 58 % behind the car in front: FREE with the default 80 %, UNKNOWN once the user sets 55 %
    tr = SlotTracker(dict(REAR), hold=2, exit_hold=5, absent_free=True, learn=False, hidden_max=10)
    run(tr, [[PICKUP]] * 5)
    front = car(800, 400, w=210, h=250)
    run(tr, [[front]] * 30, start=5)
    assert tr.shown(1) == 0
    assert tr.tune({"hidden_ground": 0.55})["hidden_ground"] == 0.55
    tr.update([front], SHAPE, now=40)
    assert tr.shown(1) is None and tr.unseen == {1: "hidden"} and tr.slots[1]["state"] == 0  # history unchanged


def test_stale_sessions_of_other_scopes_are_closed_never_deleted(tmp_path):
    path = tmp_path / "h.db"
    now = HistoryDB(path, "Gate", detector="yolo | auto v10")  # the scope in use
    old = HistoryDB(path, "Gate", detector="yolo | auto v3")  # an earlier grid version: nothing will close it
    plain = HistoryDB(path, "Gate", detector="yolo")
    other = HistoryDB(path, "Gate B", detector="yolo")  # another camera whose name starts the same
    for db, slots in ((now, (1, 2)), (old, (7, 8, 9)), (plain, (4,)), (other, (5,))):
        for slot in slots:
            db.apply({"slot": slot, "state": 1, "time": 100.0 + slot, "initial": False})
    assert now.stale_open() == {"Gate | yolo | auto v3": 3, "Gate | yolo": 1}  # (not Gate B, not the scope in use)
    assert now.stale_open(only=lambda s: s.endswith("auto v3")) == {"Gate | yolo | auto v3": 3}
    rows = lambda: now.con.execute("SELECT COUNT(*) FROM history").fetchone()[0]  # noqa: E731
    before = rows()
    assert now.close_stale("x", 500.0, dry_run=True) == {"Gate | yolo | auto v3": 3, "Gate | yolo": 1}
    assert len(old.sessions(open_only=True)) == 3  # a dry run changes nothing
    assert now.close_stale("grid replaced by v10", 500.0, only=lambda s: s.endswith("auto v3")) == {"Gate | yolo | auto v3": 3}
    assert old.sessions(open_only=True) == [] and len(plain.sessions(open_only=True)) == 1  # only the chosen scope
    assert {s["note"] for s in old.sessions()} == {"grid replaced by v10"} and old.sessions()[0]["out_ts"] == 500.0
    assert old.sessions()[0]["duration_min"] == round((500.0 - 109.0) / 60, 1)
    assert sorted(e["event"] for e in old.events(20)) == ["entry"] * 3 + ["exit"] * 3  # each exit under the scope it belongs to
    assert now.events(20)[0]["event"] == "entry" and len(now.sessions(open_only=True)) == 2  # the scope in use untouched
    assert now.close_stale("closed by the admin") == {"Gate | yolo": 1} and plain.sessions(open_only=True) == []
    assert now.close_stale("again") == {} and rows() == before  # nothing is ever deleted
    assert len(other.sessions(open_only=True)) == 1  # another camera's name: untouched
    assert now.close_open("end", 600.0) == 2 and now.sessions(open_only=True) == []  # this scope only


def test_open_by_slot_counts_this_scopes_open_sessions(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "Gate", detector="yolo")
    other = HistoryDB(tmp_path / "h.db", "Gate", detector="older")
    for slot in (1, 1, 2):
        db.con.execute("INSERT INTO history (camera, slot_id, entry_time, status, note, entry_ts) VALUES (?, ?, 'x', 'Occupied', '', 1.0)",
                       (db.scope, slot))
    db.con.commit()
    other.apply({"slot": 5, "state": 1, "time": 2.0, "initial": False})
    db.apply({"slot": 3, "state": 1, "time": 3.0, "initial": False})
    db.apply({"slot": 3, "state": 0, "time": 4.0, "initial": False})  # closed: not counted
    assert db.open_by_slot() == {1: 2, 2: 1} and other.open_by_slot() == {5: 1}
