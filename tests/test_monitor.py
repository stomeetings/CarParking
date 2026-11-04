"""ParkingMonitor (carparking/monitor.py): the UI-free service behind Mode 5, with a fake detector."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.history import HistoryDB  # noqa: E402
from carparking.monitor import ParkingArea, ParkingMonitor, in_area, inside, parse_roi  # noqa: E402

FRAME = np.zeros((1000, 2000, 3), np.uint8)


def det(label, x1, y1, x2, y2, state=1, conf=0.8):
    return {"poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], "box": (x1, y1, x2, y2), "label": label, "conf": conf,
            "state": state}


class FakeDetector:
    def __init__(self, script):
        self.script, self.i, self.conf, self.targets = script, 0, None, None

    def set_targets(self, names):
        self.targets = set(names)

    def __call__(self, frame, sliced=False):
        out = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return [dict(d) for d in out]


def test_monitor_filters_tracks_and_logs(tmp_path):
    parked = det("car", 200, 400, 600, 700)
    road = det("car", 1500, 400, 1800, 600)  # right of the 0.40 cut
    person = det("person", 700, 300, 760, 500, state=None)
    bus = det("bus", 100, 100, 300, 200)  # not a target class
    fake = FakeDetector([[parked, road, person, bus]] * 12 + [[person]] * 15)
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    mon = ParkingMonitor(fake, db, ParkingArea(right=0.40), targets=["car", "person"], conf=0.3,
                         hold=5, learn_hold=5, exit_hold=10, absent_free=True)
    results = [mon.process(FRAME, t=1000.0 + i) for i in range(27)]
    assert fake.conf == 0.3 and fake.targets == {"car", "person"}
    assert [d["label"] for d in results[0].vehicles] == ["car"] and [d["label"] for d in results[0].others] == ["person"]
    events = [e for r in results for e in r.events]
    assert [(e["event"], e["slot_id"], e["ts"], e["logged"]) for e in events] == [("entry", 1, 1000.0, True),
                                                                                  ("exit", 1, 1012.0, True)]
    assert [e["event"] for e in db.events()] == ["exit", "entry"]
    state = mon.state()
    assert state["counts"] == {"free": 1, "busy": 0, "unknown": 0, "hidden": 0, "blocked": 0, "phantom": 0, "duplicate": 0, "slots": 1, "sub_bays": 0, "unplaced": 0,
                               "learning": 0}
    assert state["slots"][0]["id"] == 1 and state["slots"][0]["poly"][0] == [0.1, 0.4]
    json.dumps([state, [r.to_json() for r in results]])
    assert results[0].to_json()["others"] == [{"label": "person", "conf": 0.8, "box": [700, 300, 760, 500], "src": ""}]


def test_saved_slots_outside_the_area_are_ignored_and_ids_not_reused(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    db.con.executemany("INSERT INTO slots (camera, slot_id, poly) VALUES (?, ?, ?)", [
        (db.scope, 1, json.dumps([[0.1, 0.4], [0.3, 0.4], [0.3, 0.7], [0.1, 0.7]])),
        (db.scope, 7, json.dumps([[0.8, 0.4], [0.9, 0.4], [0.9, 0.6], [0.8, 0.6]]))])  # on the road
    db.con.commit()
    mon = ParkingMonitor(FakeDetector([[]]), db, ParkingArea(right=0.40))
    mon.start(FRAME.shape)
    assert list(mon.tracker.slots) == [1] and mon.ignored == 1 and mon.tracker.next_id == 8


# the live map after "Reset slot map" (2026-10-19 17:05, 'Camera 1 | yolov8s'): 7 cars + slot 8, a stray box by
# the stairs above the white sedan's hood that read BUSY (7 vehicles, BUSY 8)
LIVE_MAP = {1: (0.077, 0.434, 0.494, 0.840), 2: (0.327, 0.212, 0.531, 0.344), 3: (0.209, 0.331, 0.561, 0.587),
            4: (0.446, 0.181, 0.568, 0.260), 5: (0.332, 0.278, 0.544, 0.384), 6: (0.446, 0.212, 0.530, 0.301),
            7: (0.465, 0.152, 0.568, 0.206), 8: (0.292, 0.352, 0.325, 0.394)}


def test_tiny_stray_slot_is_retired_and_busy_matches_the_vehicles(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    poly = lambda b: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]  # noqa: E731
    db.con.executemany("INSERT INTO slots (camera, slot_id, poly) VALUES (?, ?, ?)",
                       [(db.scope, k, json.dumps(poly(b))) for k, b in LIVE_MAP.items()])
    db.con.commit()
    db.apply({"slot": 8, "state": 1, "time": 0.0, "initial": False})  # its session, opened at 17:11 live
    cars = [px(b) for k, b in LIVE_MAP.items() if k != 8]
    stray = det("car", 600, 360, 640, 400, conf=0.31)  # another tiny box by the stairs, every frame
    mon = ParkingMonitor(FakeDetector([cars + [stray]]), db, hold=2, learn_hold=3, absent_free=True)
    results = [mon.process(FRAME, t=100.0 + i) for i in range(8)]
    assert results[0].retired == [8] and sorted(mon.tracker.slots) == [1, 2, 3, 4, 5, 6, 7]
    s8 = next(s for s in db.sessions() if s["slot_id"] == 8)
    assert "tiny" in s8["note"] and s8["out_ts"] == 100.0  # its session closed
    assert all(len(r.vehicles) == 8 for r in results)  # the detector's output is shown as is...
    counts = mon.state()["counts"]
    assert counts["busy"] == 7 and counts["slots"] == 7  # ...but BUSY = the 7 real vehicles, nothing learned
    assert sorted(db.load_slots()) == [1, 2, 3, 4, 5, 6, 7] and mon.tracker.next_id == 9


# live slot 17 (2026-10-19, 'Camera 1 | yolov8s'): a box over the scaffolding at the left frame edge
SCAFFOLD = (0.000, 0.451, 0.333, 0.903)
SEDAN = (0.250, 0.327, 0.562, 0.587)


def px(b, label="car"):
    return det(label, *(round(v * s) for v, s in zip(b, (2000, 1000, 2000, 1000))))


def test_in_area_is_strict_about_boxes_straddling_the_wall():
    mask = ParkingArea(left=0.15, right=0.40).mask(FRAME.shape)
    scaffold = px(SCAFFOLD)["poly"]
    assert inside(mask, scaffold)  # the old centre test let it in: centre x 0.167 is just inside a 15 % cut
    assert not in_area(mask, scaffold)  # 45 % of the box is outside the wall
    assert not in_area(ParkingArea(left=0.05).mask(FRAME.shape), scaffold)  # even 15 % outside is too much
    assert in_area(mask, px(SEDAN)["poly"])  # wholly inside
    # ground point outside (a box whose bottom crosses a 20 % bottom cut) / most of the box outside the right cut
    assert not in_area(ParkingArea(bottom=0.2).mask(FRAME.shape), det("car", 400, 500, 800, 900)["poly"])
    assert not in_area(mask, det("car", 1050, 400, 1500, 600)["poly"])


def test_a_whole_car_may_overhang_the_wall_a_piece_at_the_frame_edge_may_not():
    # Front Parking 02, 2026-10-22 13:58 (camera-1's area: left 12 %, right 42 %): the maroon hatchback parked a little
    # forward, its box 87 % inside (bonnet over the left cut, centre 0.14 of the frame inside it): kept (70 % rule; it
    # was dropped at 90 %, never learned or counted). The scaffolding box of 2026-10-19 reaches the frame's left side:
    # still out (90 % for those), even 85 % inside. A whole box only 65 % inside: out
    mask = ParkingArea(top=0.12, left=0.12, right=0.42).mask(FRAME.shape)
    maroon = px((0.070, 0.450, 0.452, 0.817))["poly"]
    assert in_area(mask, maroon) and not in_area(mask, maroon, min_inside=0.9)  # the rule, and at 90 % (before)
    assert in_area(mask, px((0.070, 0.200, 0.452, 0.567))["poly"])  # the same overhang further away
    assert not in_area(ParkingArea(left=0.05).mask(FRAME.shape), px(SCAFFOLD)["poly"])  # cut by the frame's side
    assert not in_area(mask, px((0.020, 0.450, 0.310, 0.817))["poly"])  # 66 % inside (centre inside): out


def test_dynamic_slots_outside_the_hard_wall_are_never_learned_or_shown(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    poly = lambda b: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]  # noqa: E731
    db.con.executemany("INSERT INTO slots (camera, slot_id, poly) VALUES (?, ?, ?)",
                       [(db.scope, 2, json.dumps(poly(SEDAN))), (db.scope, 17, json.dumps(poly(SCAFFOLD)))])
    db.con.commit()
    db.apply({"slot": 17, "state": 1, "time": 0.0, "initial": True})  # learned and logged before the wall moved
    other = det("car", 0, 450, 600, 920)  # another box over the scaffolding, centre inside
    mon = ParkingMonitor(FakeDetector([[px(SEDAN), px(SCAFFOLD), other]]), db, ParkingArea(left=0.15, right=0.40),
                         hold=2, learn_hold=3, absent_free=True)
    results = [mon.process(FRAME, t=100.0 + i) for i in range(8)]
    assert sorted(mon.tracker.slots) == [2] and list(mon.tracker.outside) == [17] and mon.ignored == 1
    assert all(len(r.vehicles) == 1 for r in results) and sorted(db.load_slots()) == [2, 17]  # nothing new learned
    assert [s["id"] for s in mon.state()["slots"]] == [2]  # not displayed
    assert all(e["slot_id"] == 2 for r in results for e in r.events)  # slot 17 never logs anything
    # the wall moves back while running: slot 17 returns with its ID; moved in again: set aside again
    assert mon.set_area(ParkingArea(right=0.40)) == [] and sorted(mon.tracker.slots) == [2, 17]
    assert mon.set_area(ParkingArea(left=0.15, right=0.40)) == [17] and sorted(mon.tracker.slots) == [2]


def test_phantom_slot_between_parked_cars_is_retired_and_its_id_not_reused(tmp_path):
    # Front Parking 02 (2026-10-19, shares of the frame): slot 2 was learned from the red car's box stretched down
    # over the white sedan in front; it never held a car of its own and showed FREE between the two
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    red, sedan, suv = (0.289, 0.276, 0.545, 0.383), (0.250, 0.327, 0.562, 0.587), (0.328, 0.212, 0.534, 0.344)
    stretched = (0.269, 0.279, 0.557, 0.423)
    poly = lambda b: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]  # noqa: E731
    db.con.executemany("INSERT INTO slots (camera, slot_id, poly) VALUES (?, ?, ?)",
                       [(db.scope, k, json.dumps(poly(b))) for k, b in ((1, red), (2, stretched), (3, sedan))])
    db.con.commit()
    px = lambda b: det("car", *(round(v * s) for v, s in zip(b, (2000, 1000, 2000, 1000))))  # noqa: E731
    mon = ParkingMonitor(FakeDetector([[px(red), px(sedan), px(suv)]]), db, hold=2, learn_hold=5, absent_free=True,
                         merge_iou=None, cover_share=None)  # the phantom path (by default: a duplicate)
    first = mon.process(FRAME, t=100.0)
    assert [s["state"] for s in mon.state()["slots"] if s["phantom"]] == ["UNKNOWN"] and first.retired == []
    results = [first] + [mon.process(FRAME, t=101.0 + i) for i in range(7)]
    assert [r.retired for r in results if r.retired] == [[2]] and 2 not in mon.tracker.slots
    assert all(e["slot_id"] != 2 for r in results for e in r.events)  # never logged FREE (or anything)
    assert sorted(db.load_slots()) == [1, 3, 4]  # 4 = the SUV, learned meanwhile
    mon2 = ParkingMonitor(FakeDetector([[]]), db)
    mon2.start(FRAME.shape)
    assert 2 not in mon2.tracker.slots and mon2.tracker.next_id == 5


def test_parking_area_mask_and_roi():
    area = ParkingArea(top=0.1, right=0.5, roi=parse_roi("0,0; 1,0; 1,1"))  # triangle above the diagonal
    m = area.mask((100, 200))
    assert m[50, 150] == 0 and m[5, 50] == 0 and m[20, 90] == 255 and m[80, 20] == 0
    assert parse_roi("  ") is None


def test_duplicate_slots_in_the_saved_map_are_merged_and_retired(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    sedan, twin = (0.10, 0.30, 0.30, 0.50), (0.11, 0.30, 0.31, 0.52)  # slot 1 and slot 11 over one white sedan
    poly = lambda b: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]  # noqa: E731
    db.con.executemany("INSERT INTO slots (camera, slot_id, poly) VALUES (?, ?, ?)",
                       [(db.scope, k, json.dumps(poly(b))) for k, b in ((1, sedan), (11, twin))])
    db.con.commit()
    db.apply({"slot": 11, "state": 1, "time": 50.0, "initial": True})  # the duplicate had an open session
    db.apply({"slot": 1, "state": 1, "time": 50.0, "initial": True})
    mon = ParkingMonitor(FakeDetector([[det("car", 200, 300, 600, 500)]]), db, hold=2, learn_hold=5,
                         absent_free=True)
    mon.process(FRAME, t=100.0)
    assert list(mon.tracker.slots) == [1] and list(db.load_slots()) == [1] and db.max_slot_id() == 11
    closed = [s for s in db.sessions() if s["slot_id"] == 11]
    assert closed[0]["out_ts"] is not None and "merged into slot 1" in closed[0]["note"]


class ZoomDetector(FakeDetector):
    """FakeDetector with a zoomed look (detect_region): returns `zoom` for any region; records the calls."""

    def __init__(self, script, zoom):
        super().__init__(script)
        self.zoom, self.calls = zoom, []

    def detect_region(self, frame, box, conf=0.30, behind=False):
        self.calls.append((self.i, conf))
        return [dict(d, src="zoom") for d in self.zoom]


def test_zoom_recheck_keeps_a_hidden_car_busy(tmp_path):
    # Front Parking 02, 2026-10-20: the car behind the white sedan shows only its roof and tail lights; the whole
    # frame gives it no box even at conf 0.05, a crop around its slot does (0.60). Its slot must stay BUSY
    parked = det("car", 200, 300, 600, 500)
    behind = det("car", 220, 310, 590, 500, conf=0.6)  # what the zoom finds: fits the slot (IoU 0.83)
    elsewhere = det("car", 1000, 300, 1300, 500, conf=0.6)  # a box that doesn't fit the slot
    script = [[parked]] * 10 + [[]] * 120
    runs = {}
    for name, d in (("zoom", ZoomDetector(script, [behind])), ("off", FakeDetector(script)),
                    ("misfit", ZoomDetector(script, [elsewhere]))):
        db = HistoryDB(tmp_path / f"{name}.db", "cam", detector="fake")
        mon = ParkingMonitor(d, db, hold=2, learn_hold=5, exit_hold=10, absent_free=True)
        res = [mon.process(FRAME, t=float(i)) for i in range(130)]
        runs[name] = ([(e["event"], e["slot_id"]) for r in res for e in r.events], mon, res)
    assert runs["zoom"][0] == [("entry", 1)] and runs["zoom"][1].state()["slots"][0]["state"] == "BUSY"
    assert runs["zoom"][2][-1].vehicles[0]["src"] == "zoom"
    calls = runs["zoom"][1].detector.calls
    assert calls[0][0] > 10 and {c for _, c in calls} == {0.30}  # only once the whole frame lost the car
    assert runs["off"][0] == runs["misfit"][0] == [("entry", 1), ("exit", 1)]
    assert len(runs["zoom"][1].tracker.slots) == 1  # the zoom box never becomes a slot of its own
    mon = ParkingMonitor(ZoomDetector(script, [behind]), HistoryDB(tmp_path / "z0.db", "cam", detector="fake"),
                         hold=2, learn_hold=5, exit_hold=10, absent_free=True, zoom_max=0)
    assert [e for i in range(130) for e in mon.process(FRAME, t=float(i)).events][-1]["event"] == "exit"


def test_zoom_never_makes_an_empty_slot_busy(tmp_path):
    # a FREE slot is never re-checked: a low-confidence box there (a shadow, a person, a piece of a neighbour) can't
    # turn an empty bay BUSY; only the whole frame at the normal threshold can
    parked = det("car", 200, 300, 600, 500)
    ghost = det("car", 220, 310, 590, 500, conf=0.31)
    d = ZoomDetector([[parked]] * 10 + [[]] * 100, [])
    mon = ParkingMonitor(d, HistoryDB(tmp_path / "h.db", "cam", detector="fake"), hold=2, learn_hold=5,
                         exit_hold=30, absent_free=True)
    for i in range(80):  # the car leaves at t=10: no box, zoomed or not -> FREE
        mon.process(FRAME, t=float(i))
    assert mon.state()["slots"][0]["state"] == "FREE"
    d.zoom, n = [ghost], len(d.calls)
    events = [e for i in range(80, 110) for e in mon.process(FRAME, t=float(i)).events]
    assert events == [] and len(d.calls) == n and mon.state()["slots"][0]["state"] == "FREE"


def test_departing_car_frees_its_slot_within_30_seconds(tmp_path):
    # the car drives out at t=10; nothing in its slot after that, not even on the zoomed re-check: FREE by t=40,
    # the exit logged at t=10. Also when the slot is hidden behind a van parked in front (hidden_max 30 s)
    parked, van = det("car", 200, 300, 600, 500), det("truck", 150, 340, 650, 900)  # covers 80 %, IoU 0.22
    for name, after in (("visible", []), ("hidden", [van])):
        d = ZoomDetector([[parked]] * 10 + [after] * 100, [])
        db = HistoryDB(tmp_path / f"{name}.db", "cam", detector="fake")
        mon = ParkingMonitor(d, db, hold=2, learn_hold=5, exit_hold=30, hidden_max=30, absent_free=True, learn=True)
        out = [(i, e) for i in range(60) for e in mon.process(FRAME, t=float(i)).events if e["slot_id"] == 1]
        exits = [(i, e["ts"]) for i, e in out if e["event"] == "exit"]
        assert exits and exits[0][0] <= 41 and exits[0][1] == 10.0, (name, exits)
        assert d.calls, name  # the re-check ran and found nothing


class SlotZoom(FakeDetector):
    """Whole frame: `script`; zoomed look: the boxes of `zoom` whose centre lies in the region and whose confidence
    reaches the requested threshold (as YoloDetector.detect_region)."""

    def __init__(self, script, zoom):
        super().__init__(script)
        self.zoom, self.calls = zoom, []

    def detect_region(self, frame, box, conf=0.30, behind=False):
        self.calls.append(conf)
        x1, y1, x2, y2 = box
        return [dict(d, src="zoom") for d in self.zoom if d["conf"] >= conf and
                x1 <= (d["box"][0] + d["box"][2]) / 2 <= x2 and y1 <= (d["box"][1] + d["box"][3]) / 2 <= y2]


def live_map(db, slots):
    poly = lambda b: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]  # noqa: E731
    db.con.executemany("INSERT INTO slots (camera, slot_id, poly) VALUES (?, ?, ?)",
                       [(db.scope, k, json.dumps(poly(b))) for k, b in slots.items()])
    db.con.commit()


def test_four_cars_three_boxes_slot_1_busy_via_the_zoom(tmp_path):
    # Front Parking 02, 2026-10-20 12:16 (shares of the frame): 4 cars, the whole frame boxes 3 (sedan 0.88,
    # hatchback 0.94, pickup 0.63); the car between the sedan and the hatchback (roof and tail lights only) is found by
    # the zoom on slot 1 at 0.33-0.59 (0.52 here). Slot 1 had gone FREE (hidden 30 s without a box), so before the
    # fix it was never looked at again: BUSY 3 | FREE 2. Now a FREE slot hidden behind cars is re-checked at 0.40
    db = HistoryDB(tmp_path / "h.db", "Camera 1", detector="fake")
    live_map(db, {1: (0.331, 0.277, 0.544, 0.446), 2: (0.327, 0.213, 0.535, 0.344), 5: (0.444, 0.181, 0.568, 0.26),
                  6: (0.438, 0.143, 0.568, 0.214), 7: (0.589, 0.118, 0.6, 0.147), 11: (0.252, 0.333, 0.569, 0.594)})
    px = lambda b, conf, label="car": det(label, *(round(v * s) for v, s in zip(b, (2000, 1000, 2000, 1000))),  # noqa: E731
                                          conf=conf)
    whole = [px((0.254, 0.279, 0.565, 0.590), 0.88), px((0.358, 0.215, 0.529, 0.336), 0.94),
             px((0.438, 0.145, 0.569, 0.240), 0.63, "truck")]
    hidden_car = px((0.337, 0.277, 0.560, 0.392), 0.52)
    area = ParkingArea(top=0.10, left=0.12, right=0.42)  # the saved walls: slot 7 lies outside
    for zoom, busy in (([hidden_car], {1, 2, 6, 11}), ([dict(hidden_car, conf=0.35)], {2, 6, 11})):
        d = SlotZoom([whole], zoom)
        mon = ParkingMonitor(d, db, area, hold=15, exit_hold=30, absent_free=True)
        mon.process(FRAME, t=0.0)
        mon.tracker.slots[1].update(state=0, cand=None)  # as live: slot 1 FREE
        for i in range(1, 40):
            mon.process(FRAME, t=float(i))
        st = mon.state()
        assert {s["id"] for s in st["slots"] if s["state"] == "BUSY"} == busy, zoom[0]["conf"]
        assert st["counts"]["slots"] == 5 and 0.40 in d.calls  # slot 1 re-checked at the stricter threshold
    assert st["counts"]["busy"] == 3  # 0.35 < 0.40: a weak box never turns a FREE slot BUSY


class BehindZoom(FakeDetector):
    """Whole frame: `script`; zoomed look: nothing on the normal crop, `behind` on the crop cut at the slot's ground
    line (as YoloDetector.detect_region behind=True); records the calls."""

    def __init__(self, script, behind):
        super().__init__(script)
        self.behind, self.calls = behind, []

    def detect_region(self, frame, box, conf=0.30, behind=False):
        self.calls.append(behind)
        return [dict(d, src="zoom") for d in self.behind] if behind else []


def test_white_car_swallowed_by_the_sedans_box_is_split_off_by_the_behind_zoom(tmp_path):
    # Front Parking 02, 2026-10-22 09:37 (shares of the frame): 6 cars, 5 BUSY. The whole frame boxes the white sedan
    # (slot 11) and the white hatchback behind it (slot 1) as ONE box reaching far above slot 11 (0.25 vs 0.333);
    # the crop cut at slot 1's ground line finds the hatchback alone (0.92), as wide as that box
    px = lambda b, conf, label="car": det(label, *(round(v * s) for v, s in zip(b, (2000, 1000, 2000, 1000))),  # noqa: E731
                                          conf=conf)
    slots = {1: (0.331, 0.277, 0.544, 0.446), 6: (0.438, 0.143, 0.568, 0.214), 11: (0.252, 0.333, 0.569, 0.594),
             13: (0.475, 0.149, 0.543, 0.187)}
    pickup = px((0.467, 0.139, 0.567, 0.232), 0.5, "truck")
    cases = {  # whole frame, behind-crop boxes -> BUSY
        "swallowed": ([px((0.248, 0.250, 0.556, 0.593), 0.83), pickup], [px((0.261, 0.257, 0.556, 0.444), 0.92)],
                      {1, 6, 11}),
        # slot 1 empty: the sedan's own box starts at its slot's top, the crop sees its roof only -> stays FREE
        "lone sedan": ([px((0.210, 0.330, 0.550, 0.590), 0.83), pickup], [px((0.250, 0.330, 0.550, 0.446), 0.70)],
                       {6, 11}),
        # the pickup's upper half cut at slot 13 behind it: its box fits its own slot 6 -> never a car of its own
        "pickup slice": ([px((0.210, 0.330, 0.550, 0.590), 0.83), pickup], [px((0.460, 0.140, 0.570, 0.190), 0.71)],
                         {6, 11}),
    }
    for name, (whole, behind, busy) in cases.items():
        db = HistoryDB(tmp_path / f"{name}.db", "Camera 1", detector="fake")
        live_map(db, slots)
        d = BehindZoom([whole], behind)
        mon = ParkingMonitor(d, db, hold=2, exit_hold=30, absent_free=True, learn=False)
        mon.process(FRAME, t=0.0)
        for k in (1, 13):
            mon.tracker.slots[k].update(state=0, cand=None)  # as live: FREE, hidden behind the cars in front
        for i in range(1, 40):
            mon.process(FRAME, t=float(i))
        assert {s["id"] for s in mon.state()["slots"] if s["state"] == "BUSY"} == busy, name
        assert True in d.calls, name  # the behind crop was tried


class RegionLog(FakeDetector):
    """Whole frame: `script`; zoomed look: nothing; records each region (frame pixels) and whether it was behind."""

    def __init__(self, script):
        super().__init__(script)
        self.regions = []

    def detect_region(self, frame, box, conf=0.30, behind=False):
        self.regions.append((round(box[0]), behind, conf))
        return []


def test_zoom_rechecks_the_busy_slot_that_lost_its_box_first_and_proven_phantoms_too(tmp_path):
    # one zoom per frame (zoom_max 1), 3 hidden FREE slots and 1 BUSY slot without a box: the BUSY one is looked at
    # on every frame (else it lapses after hidden_max among the others' turns), the FREE ones never
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    live_map(db, {1: (0.10, 0.20, 0.20, 0.30), 2: (0.30, 0.20, 0.40, 0.30), 3: (0.50, 0.20, 0.60, 0.30),
                  4: (0.70, 0.20, 0.80, 0.30)})
    front = det("car", 100, 150, 1700, 600)  # in front of all four: they are hidden (claim rules off: not in them)
    d = RegionLog([[front]])
    mon = ParkingMonitor(d, db, ParkingArea(top=0, bottom=0, left=0, right=0), hold=2, exit_hold=600, hidden_max=600,
                         absent_free=True, learn=False, zoom_max=1, share_cover=None, claim_cover=None)
    mon.process(FRAME, t=0.0)
    for k, s in mon.tracker.slots.items():
        s.update(state=1 if k == 3 else 0, cand=None)
    for i in range(1, 6):
        mon.process(FRAME, t=float(i))
    assert [x for x, behind, _ in d.regions if not behind] == [1000] * 5  # slot 3 (x 0.50 of 2000 px), every frame
    # a proven bay taken for a phantom (no box of its own, between / inside the cars around it) is re-checked, at
    # the stricter threshold
    tr, d.regions = mon.tracker, []
    tr.slots[3].update(state=0)
    tr.proven = {2}
    real_phantom = tr._phantom
    tr._phantom = lambda poly, cars, room=None: "gap" if abs(float(poly[0][0]) - 0.30) < 1e-3 else real_phantom(poly, cars, room)
    for k in (1, 3, 4):
        tr.slots[k]["state"] = 0
    tr.slots[2].update(state=None, cand=None)
    for i in range(6, 12):
        mon.process(FRAME, t=float(i))
    assert 2 in tr.phantoms and (600, False, 0.40) in d.regions


def test_zoomed_piece_of_a_found_car_never_keeps_a_second_slot_busy(tmp_path):
    # Front Parking 02, 2026-10-22 12:38-12:52: slot 24, a strip on top of the white car of slot 18, stayed BUSY for
    # 14 min: as a BUSY slot without a box it was zoomed every frame, and the crop found the white car's upper part
    # (inside its box, not a merge duplicate: other bottom, other width). Now that piece is the white car's: slot 24
    # is released by the hidden / exit timers, slot 18 stays BUSY
    px = lambda b, conf, label="car": det(label, *(round(v * s) for v, s in zip(b, (2000, 1000, 2000, 1000))),  # noqa: E731
                                          conf=conf)
    db = HistoryDB(tmp_path / "h.db", "Camera 1", detector="fake")
    live_map(db, {18: (0.402, 0.185, 0.512, 0.258), 24: (0.439, 0.185, 0.520, 0.217)})
    white = px((0.400, 0.180, 0.513, 0.251), 0.87)
    piece = px((0.440, 0.180, 0.515, 0.222), 0.70)
    d = ZoomDetector([[white]], [piece])
    mon = ParkingMonitor(d, db, ParkingArea(top=0, bottom=0, left=0, right=0), hold=2, exit_hold=10, hidden_max=30,
                         absent_free=True, learn=False)
    mon.process(FRAME, t=0.0)
    for k in (18, 24):
        mon.tracker.slots[k].update(state=1, cand=None)
    events = []
    for i in range(1, 60):
        res = mon.process(FRAME, t=float(i))
        events += [(e["slot_id"], e["state"]) for e in res.events]
    assert d.calls and not any(v.get("src") == "zoom" for v in res.vehicles)  # zoomed, the piece not taken
    assert (24, "FREE") in events and (18, "FREE") not in events  # slot 24 left BUSY (then: the white car's duplicate)
    assert mon.tracker.slots[18]["state"] == 1 and mon.tracker.slots[24]["state"] != 1
    assert mon.state()["counts"]["busy"] == 1


class SplitDetector(FakeDetector):
    """Whole frame: `script`; _whole(frame, imgsz): `alt` (the second whole-frame pass); detect_region(behind=True):
    `rear`; records the calls."""

    def __init__(self, script, alt, rear):
        super().__init__(script)
        self.alt, self.rear, self.calls = alt, rear, []

    def _whole(self, frame, imgsz):
        self.calls.append(("whole", imgsz))
        return [dict(d) for d in self.alt]

    def detect_region(self, frame, box, conf=0.30, behind=False):
        self.calls.append(("region", behind))
        return [dict(d, src="zoom") for d in self.rear] if behind else []


def test_one_box_over_the_sedan_and_the_hatchback_behind_it_is_split(tmp_path):
    # Front Parking 02, 2026-10-22 (shares of the frame): after the slot map reset the 1920 px box over the white sedan
    # and the white hatchback behind it [0.25, 0.225, 0.546, 0.591] became one slot: 1 BUSY for 2 cars. The 1280 px
    # pass boxes the sedan alone [0.206, 0.34, 0.543, 0.59], the behind crop over the strip above it finds the
    # hatchback [0.285, 0.206, 0.531, 0.34]: two boxes, two slots, 2 BUSY
    px = lambda b, conf=0.8: det("car", *(round(v * s) for v, s in zip(b, (2000, 1000, 2000, 1000))), conf=conf)  # noqa: E731
    merged, sedan, hatch = px((0.25, 0.225, 0.546, 0.591)), px((0.206, 0.34, 0.543, 0.59)), px((0.285, 0.206, 0.531, 0.34))
    area = ParkingArea(top=0, bottom=0, left=0, right=0)
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    d = SplitDetector([[merged]], [sedan], [hatch])
    mon = ParkingMonitor(d, db, area, hold=2, learn_hold=3, exit_hold=30, absent_free=True)
    res = [mon.process(FRAME, t=float(i)) for i in range(12)]
    assert sorted((v["src"], v["box"][1]) for v in res[0].vehicles) == [("split", 206), ("split", 340)]
    assert mon.state()["counts"]["busy"] == 2 and mon.state()["counts"]["slots"] == 2
    assert d.calls.count(("whole", 1280)) == 1  # checked once, the result reused while the box stays (SPLIT_EVERY)
    # a lone tall car: the second pass gives the same box (its own top) -> never split, the strip never looked at
    db2 = HistoryDB(tmp_path / "h2.db", "cam", detector="fake")
    lone = SplitDetector([[sedan]], [sedan], [hatch])
    mon2 = ParkingMonitor(lone, db2, area, hold=2, learn_hold=3, exit_hold=30, absent_free=True)
    res2 = [mon2.process(FRAME, t=float(i)) for i in range(12)]
    assert all(v.get("src") != "split" for r in res2 for v in r.vehicles) and ("region", True) not in lone.calls
    assert mon2.state()["counts"]["busy"] == 1


def test_the_history_is_checked_against_the_live_count_on_a_learned_map_reported_never_repaired(tmp_path):
    parked = det("car", 200, 400, 600, 700)
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    mon = ParkingMonitor(FakeDetector([[parked]] * 60), db, hold=5, learn_hold=5, exit_hold=10, absent_free=True)
    for i in range(12):
        mon.process(FRAME, t=1000.0 + i)
    assert mon.session_check == {"expected": 1, "open": 1, "since": None, "at": 1011.0}  # one vehicle, one open session
    db.con.execute("DELETE FROM history WHERE exit_ts IS NULL")  # the log loses the session (a restored / edited file)
    db.con.commit()
    mon.process(FRAME, t=1012.0)
    assert mon.session_check == {"expected": 1, "open": 0, "since": 1012.0, "at": 1012.0}
    mon.process(FRAME, t=1042.0)
    assert mon.session_check["since"] == 1012.0 and mon.session_check["at"] == 1042.0  # the difference has lasted 30 s
    assert db.sessions(open_only=True) == []  # a learned map is only reported: its merges and retirements are its own
