"""Fixed slot mode (carparking/fixed_slots.py + ParkingMonitor fixed=...): store, validation, matching, events."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.fixed_slots import (FixedSlotStore, FixedSlotTracker, draft_slots, format_points,  # noqa: E402
                                    match_score, parse_points, validate_slots)
from carparking.dynamic_iou import overlap  # noqa: E402
from carparking.history import HistoryDB  # noqa: E402
from carparking.monitor import ParkingArea, ParkingMonitor  # noqa: E402

SHAPE = (1000, 2000, 3)
FRAME = np.zeros(SHAPE, np.uint8)
# Front Parking 02 style: front car low in the frame, a second car parked behind it (higher, partly covered)
FRONT_SLOT = {"id": 1, "name": "A1", "poly": parse_points("0.20,0.45,0.56,0.60")}
BACK_SLOT = {"id": 2, "name": "A2", "poly": parse_points("0.30,0.30,0.54,0.40")}


def det(x1, y1, x2, y2, label="car", state=1):
    """Pixel box on a 2000 x 1000 frame."""
    return {"poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], "box": (x1, y1, x2, y2), "label": label, "conf": 0.8,
            "state": state}


FRONT_CAR = det(440, 330, 1120, 590)  # whole image box: also covers most of the back slot
BACK_CAR = det(620, 220, 1080, 390)


def test_parse_and_format_points():
    assert parse_points("0.6,0.5,0.2,0.1") == [[0.2, 0.1], [0.6, 0.1], [0.6, 0.5], [0.2, 0.5]]
    assert parse_points("0,0; 1,0; 0.5,1") == [[0, 0], [1, 0], [0.5, 1]]
    assert format_points(parse_points("0.2,0.1,0.6,0.5")) == "0.200,0.100,0.600,0.500"
    assert parse_points(format_points([[0, 0], [1, 0], [0.5, 1]])) == [[0, 0], [1, 0], [0.5, 1]]
    for bad in ("", "1,2,3", "a,b,c,d", "0,0; 1"):
        with pytest.raises(ValueError):
            parse_points(bad)


@pytest.mark.parametrize("slots, msg", [
    ([{"id": 1, "poly": [[0, 0], [1, 0], [1, 1]]}, {"id": 1, "poly": [[0, 0], [1, 0], [1, 1]]}], "used twice"),
    ([{"id": 0, "poly": [[0, 0], [1, 0], [1, 1]]}], ">= 1"),
    ([{"id": 2.5, "poly": [[0, 0], [1, 0], [1, 1]]}], "whole number"),
    ([{"id": float("nan"), "poly": [[0, 0], [1, 0], [1, 1]]}], "whole number"),
    ([{"id": 1, "poly": [[0, 0], [1.2, 0], [1, 1]]}], "fractions"),
    ([{"id": 1, "poly": [[0, 0], [1, 0]]}], "3 points"),
    ([{"id": 1, "poly": [[0, 0], [0.5, 0.5], [1, 1]]}], "no area"),
])
def test_validate_rejects_bad_slots(slots, msg):
    with pytest.raises(ValueError, match=msg):
        validate_slots(slots)


def test_validate_cleans_and_sorts():
    out = validate_slots([{"id": "3", "name": " ", "poly": [[0, 0], [1, 0], [1, 1]]}, BACK_SLOT, FRONT_SLOT])
    assert [(s["id"], s["name"]) for s in out] == [(1, "A1"), (2, "A2"), (3, "3")]


def test_store_round_trip_per_camera(tmp_path):
    store = FixedSlotStore(tmp_path / "config" / "fixed_slots.json")
    assert store.load("Front Parking 02") == [] and store.get("x")["updated"] is None
    store.save("Front Parking 02", [BACK_SLOT, FRONT_SLOT])
    store.save("Gate", [FRONT_SLOT])
    assert [s["id"] for s in store.load("Front Parking 02")] == [1, 2] and store.cameras() == ["Front Parking 02", "Gate"]
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["cameras"]["Gate"]["slots"][0]["name"] == "A1"
    store.save("Gate", [])  # empty = remove the camera, others untouched
    assert store.cameras() == ["Front Parking 02"] and not store.path.with_name("fixed_slots.json.tmp").exists()
    with pytest.raises(ValueError):
        store.save("Front Parking 02", [{"id": 1, "poly": [[0, 0]]}])
    assert len(store.load("Front Parking 02")) == 2  # a rejected save changes nothing


def test_ground_rule_assigns_each_car_to_its_own_slot():
    front, back = (np.float32(d["poly"]) / (2000, 1000) for d in (FRONT_CAR, BACK_CAR))
    # the front car's whole box covers most of the back slot too; its ground point is only in the front slot
    assert overlap(np.float32(BACK_SLOT["poly"]), front)[1] > 0.6
    assert match_score(front, BACK_SLOT["poly"], "ground") == 0
    assert match_score(front, FRONT_SLOT["poly"], "ground") > 0 and match_score(back, BACK_SLOT["poly"], "ground") > 0
    assert match_score(front, FRONT_SLOT["poly"], "iou", min_iou=0.15) > 0
    with pytest.raises(ValueError):
        match_score(front, FRONT_SLOT["poly"], "nearest")


def test_point_rules_accept_a_car_parked_slightly_forward_but_never_over_the_true_slot():
    # Front Parking 02, 2026-10-22 (camera-2, fractions): S1's drawn bottom edge at 0.8253; a car whose wheels stand
    # a little below it, its box covering most of S1
    s1 = [[0.0978, 0.4791], [0.4794, 0.4791], [0.4794, 0.8253], [0.0978, 0.8253]]
    rect = lambda x1, y1, x2, y2: np.float32([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])  # noqa: E731
    forward = rect(0.09, 0.50, 0.47, 0.88)  # ground point (0.28, 0.842): 0.017 outside, box height 0.38
    assert 0 < match_score(forward, s1, "ground") < 1  # lenient: below any slot that holds the point
    assert match_score(forward, s1, "center") > 1  # its centre is inside anyway
    assert match_score(rect(0.09, 0.62, 0.47, 1.0), s1, "ground") == 0  # 0.137 outside: a car in front of the bay
    assert match_score(rect(0.44, 0.70, 0.60, 0.84), s1, "ground") == 0  # just outside but covering little of it
    s0 = [[0.0978, 0.8253], [0.4794, 0.8253], [0.4794, 0.99], [0.0978, 0.99]]  # a slot in front: holds the point
    assert match_score(forward, s0, "ground") > match_score(forward, s1, "ground")


def run(tracker, frames, start=0.0):
    events = []
    for i, dets in enumerate(frames):
        events += tracker.update(dets, SHAPE, now=start + i)
    return events


def test_fixed_tracker_never_learns_and_debounces():
    tr = FixedSlotTracker([FRONT_SLOT, BACK_SLOT], "ground", hold=3, exit_hold=5, absent_free=True,
                          learn=True, learn_hold=1)  # learning args from the UI are ignored
    stray = det(1500, 100, 1700, 200)  # matches no slot: never becomes one
    ev = run(tr, [[FRONT_CAR, stray]] * 6 + [[stray]] * 8)
    assert sorted(tr.slots) == [1, 2] and tr.candidates == [] and tr.names == {1: "A1", 2: "A2"}
    # back bay A2 is hidden behind the front car at start: unknown (not FREE) until the front car leaves at t=6
    assert sorted((e["slot"], e["state"], e["time"]) for e in ev) == [(1, 0, 6), (1, 1, 0), (2, 0, 6)]
    with pytest.raises(ValueError):
        FixedSlotTracker([FRONT_SLOT], "nearest")


FAR_SLOTS = [{"id": 1, "name": "106", "poly": parse_points("0.30,0.19,0.40,0.21")},
             {"id": 2, "name": "107", "poly": parse_points("0.30,0.11,0.40,0.13")}]  # a thin far tile, and another one behind it
FAR_CAR = det(620, 150, 780, 250)  # its foot point (y 0.24) is 0.03 below the tile: just outside


def test_cover_pass_gives_a_far_car_its_thin_bay():
    off = FixedSlotTracker(FAR_SLOTS, "ground", hold=2, exit_hold=3, absent_free=True)
    run(off, [[FAR_CAR]] * 5)
    assert off.slots[1]["state"] != 1  # the point rules miss it (the live trackers keep doing so)
    on = FixedSlotTracker(FAR_SLOTS, "ground", hold=2, exit_hold=3, absent_free=True)
    on.cover_fallback = True
    run(on, [[FAR_CAR]] * 5)
    assert on.slots[1]["state"] == 1 and on.slots[2]["state"] != 1  # the tile behind it is not taken for it


def test_back_slot_hidden_by_the_front_car_stays_busy():
    tr = FixedSlotTracker([FRONT_SLOT, BACK_SLOT], "ground", hold=2, exit_hold=5, absent_free=True)
    run(tr, [[FRONT_CAR, BACK_CAR]] * 4 + [[FRONT_CAR]] * 10)  # back car no longer detected behind the front one
    assert tr.slots[2]["state"] == 1 and tr.hidden == {2}


def test_monitor_fixed_mode_logs_history_like_dynamic(tmp_path):
    class Fake:
        conf = None

        def __init__(self, script):
            self.script, self.i = script, 0

        def __call__(self, frame, sliced=False):
            self.i += 1
            return [dict(d) for d in self.script[min(self.i - 1, len(self.script) - 1)]]

    road = det(1700, 300, 1900, 500)  # right of the 0.40 cut
    db = HistoryDB(tmp_path / "h.db", "Front Parking 02", detector="yolov8s | fixed")
    mon = ParkingMonitor(Fake([[FRONT_CAR, BACK_CAR, road]] * 8 + [[BACK_CAR]] * 10), db, ParkingArea(right=0.40),
                         fixed=[FRONT_SLOT, BACK_SLOT], match_rule="ground", hold=3, exit_hold=5, absent_free=True)
    events = [e for i in range(18) for e in mon.process(FRAME, t=100.0 + i).events]
    assert [(e["slot_id"], e["event"]) for e in events] == [(1, "entry"), (2, "entry"), (1, "exit")]
    assert [(s["slot_id"], s["status"]) for s in db.sessions()] == [(2, "Occupied"), (1, "Available")]
    assert [e["event"] for e in db.events()] == ["exit", "entry", "entry"]
    assert db.load_slots() == {}  # the fixed map lives in the JSON file, not in the learned-slot table
    state = mon.state()
    assert state["mode"] == "fixed" and [(s["id"], s["name"], s["state"]) for s in state["slots"]] == [
        (1, "A1", "FREE"), (2, "A2", "BUSY")]
    json.dumps(state)


def test_draft_slots_from_detections():
    person = det(100, 100, 150, 300, label="person", state=None)
    out = draft_slots([BACK_CAR, person, FRONT_CAR], SHAPE)
    assert [(s["id"], s["name"]) for s in out] == [(1, "S1"), (2, "S2")]
    assert out[0]["poly"][0] == [0.22, 0.33] and out[1]["poly"][2] == [0.54, 0.39]  # front car first


def test_store_edits_and_deletes_one_slot_at_a_time(tmp_path):
    store = FixedSlotStore(tmp_path / "fixed_slots.json")
    store.save("Front Parking 02", [FRONT_SLOT, BACK_SLOT])
    store.save("Gate", [FRONT_SLOT])
    moved = {**BACK_SLOT, "name": "B2", "poly": parse_points("0.32,0.30,0.56,0.41")}
    out = store.upsert_slot("Front Parking 02", moved)
    assert [(s["id"], s["name"]) for s in out] == [(1, "A1"), (2, "B2")] and out[1]["poly"][1] == [0.56, 0.3]
    assert store.load("Front Parking 02")[0] == validate_slots([FRONT_SLOT])[0]  # the other slot untouched
    out = store.upsert_slot("Front Parking 02", {"id": 5, "name": "", "poly": parse_points("0.6,0.6,0.7,0.7")})
    assert [s["id"] for s in out] == [1, 2, 5]
    assert [s["id"] for s in store.delete_slot("Front Parking 02", 2)] == [1, 5]
    assert [s["id"] for s in store.load("Gate")] == [1]  # other cameras untouched
    with pytest.raises(KeyError):
        store.delete_slot("Front Parking 02", 2)
    with pytest.raises(ValueError):
        store.upsert_slot("Front Parking 02", {"id": 1, "poly": [[0, 0], [1, 1]]})
    assert [s["id"] for s in store.load("Front Parking 02")] == [1, 5]


def test_set_slots_keeps_state_of_kept_slots():
    tr = FixedSlotTracker([FRONT_SLOT, BACK_SLOT], "ground", hold=2, exit_hold=5, absent_free=True)
    run(tr, [[FRONT_CAR, BACK_CAR]] * 4)
    assert tr.slots[1]["state"] == tr.slots[2]["state"] == 1
    new = {"id": 9, "name": "C1", "poly": parse_points("0.7,0.5,0.9,0.7")}
    reshaped = {**FRONT_SLOT, "name": "Front", "poly": parse_points("0.21,0.46,0.57,0.61")}
    assert tr.set_slots([reshaped, new]) == [2]
    assert sorted(tr.slots) == [1, 9] and tr.slots[1]["state"] == 1 and tr.slots[9]["state"] is None
    assert tr.names == {1: "Front", 9: "C1"} and np.allclose(tr.slots[1]["poly"][0], (0.21, 0.46))


def test_monitor_update_fixed_closes_the_deleted_slots_session(tmp_path):
    class Fake:
        conf = None

        def __call__(self, frame, sliced=False):
            return [dict(FRONT_CAR), dict(BACK_CAR)]

    db = HistoryDB(tmp_path / "h.db", "Front Parking 02", detector="yolov8s | fixed")
    mon = ParkingMonitor(Fake(), db, fixed=[FRONT_SLOT, BACK_SLOT], hold=2, exit_hold=5, absent_free=True)
    for i in range(4):
        mon.process(FRAME, t=100.0 + i)
    assert len(db.sessions(open_only=True)) == 2
    assert mon.update_fixed([FRONT_SLOT], now=200.0) == [2]
    assert [(s["slot_id"], s["status"]) for s in db.sessions()] == [(2, "Available"), (1, "Occupied")]
    assert db.events()[0] == {**db.events()[0], "slot_id": 2, "event": "exit", "ts": 200.0, "note": "slot deleted"}
    assert [s["id"] for s in mon.state()["slots"]] == [1]
    assert db.close_slot(2) is False  # nothing open any more
    with pytest.raises(ValueError):
        ParkingMonitor(Fake(), db).update_fixed([FRONT_SLOT])


def test_foot_rule_uses_the_box_bottom_and_needs_no_coverage_just_outside():
    from carparking.fixed_slots import FOOT, FixedSlotTracker, match_score

    thin = [[0.4, 0.229], [0.6, 0.229], [0.6, 0.244], [0.4, 0.244]]  # a far tile: a few pixels tall
    truck = [[0.44, 0.139], [0.57, 0.139], [0.57, 0.2313], [0.44, 0.2313]]  # its wheels stand in the tile, its body far above
    assert match_score(truck, thin, FOOT) > 1  # the bottom centre is inside
    assert match_score(truck, thin, "ground") == 0  # "ground" looks 10 % of the box height up: past the tile
    near = [[0.44, 0.139], [0.57, 0.139], [0.57, 0.2275], [0.44, 0.2275]]  # a little above the tile: within 15 % of the height
    assert 0 < match_score(near, thin, FOOT) < 1
    FixedSlotTracker([{"id": 1, "name": "B1", "poly": thin}], FOOT)  # accepted by the tracker, though not a UI rule


def test_foot_rule_lift_puts_the_vehicle_centre_not_its_near_wheels_in_the_bay():
    from carparking.fixed_slots import FOOT, match_score, validate_slots

    bay = [[0.3, 0.30], [0.7, 0.30], [0.7, 0.34], [0.3, 0.34]]  # the bay a car's centre stands in
    car = [[0.45, 0.24], [0.55, 0.24], [0.55, 0.40], [0.45, 0.40]]  # its near wheels are below the bay's near edge
    assert match_score(car, bay, FOOT) == 0  # the bottom edge alone is outside the bay and too far for the tolerance
    assert match_score(car, bay, FOOT, lift=0.08) > 1  # lifted to the vehicle's centre: inside
    assert validate_slots([{"id": 1, "poly": bay, "lift": 0.0123456}])[0]["lift"] == 0.01235
    assert "lift" not in validate_slots([{"id": 1, "poly": bay}])[0]  # a hand-drawn slot is as it always was


def test_auto_grid_input_is_stored_apart_from_the_fixed_map(tmp_path):
    store = FixedSlotStore(tmp_path / "fixed_slots.json")
    pin = {"id": 1, "name": "Far", "poly": [[0.9, 0.05], [0.95, 0.05], [0.95, 0.1], [0.9, 0.1]]}
    assert store.get_auto_grid("cam") == {"camera": "cam", "updated": None, "expected": None, "pins": []}
    got = store.save_auto_grid("cam", 11, [pin])
    assert got["expected"] == 11 and got["pins"] == [pin] and got["updated"]
    # the camera's fixed map is rewritten and removed without touching the pins (the registry mirrors into it)
    bay = {"id": 1, "name": "A1", "poly": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]}
    store.save("cam", [bay])
    store.save("cam", [])
    assert store.get_auto_grid("cam")["pins"] == [pin] and store.cameras() == []
    with pytest.raises(ValueError, match="expected bays"):
        store.save_auto_grid("cam", 0, [])
    with pytest.raises(ValueError, match="slot 1"):
        store.save_auto_grid("cam", 11, [{"id": 1, "poly": [[0, 0], [1, 0]]}])
    assert store.get_auto_grid("cam")["expected"] == 11  # a refused save changes nothing
    assert store.save_auto_grid("cam", None, [])["pins"] == [] and "auto_grid" not in json.loads(store.path.read_text())


# --- the auto grid's vehicle-driven rules (FixedSlotTracker(auto=...)) -------------------------------------------------------
THIN = [{"id": 1, "name": "B1", "poly": [[0.40, 0.56], [0.55, 0.56], [0.55, 0.60], [0.40, 0.60]]},
        {"id": 2, "name": "B2", "poly": [[0.45, 0.56], [0.60, 0.56], [0.60, 0.60], [0.45, 0.60]]}]
SURFACE = [[0.10, 0.20], [0.90, 0.20], [0.90, 0.90], [0.10, 0.90]]


def test_a_vehicle_no_bay_claims_goes_to_the_bay_its_wheels_overlap_nearest_centre_first():
    from carparking.fixed_slots import FOOT
    car = det(860, 350, 1160, 650)  # wheels 5 % of the frame below the bays: past the point rule's tolerance, over both bays
    plain = FixedSlotTracker(THIN, FOOT, hold=2, exit_hold=5, absent_free=True)
    run(plain, [[car]] * 5)
    assert 1 not in (plain.slots[1]["state"], plain.slots[2]["state"])  # the strict rule alone: no bay is taken
    tr = FixedSlotTracker(THIN, FOOT, hold=2, exit_hold=5, absent_free=True, auto={"surface": None})
    run(tr, [[car]] * 5)
    assert tr.slots[2]["state"] == 1 and tr.slots[1]["state"] != 1  # one vehicle, one bay: the one whose centre is nearest


def test_a_vehicle_on_the_surface_in_no_bay_gets_a_bay_of_its_own_that_goes_when_it_leaves():
    tr = FixedSlotTracker(THIN, "foot", hold=2, exit_hold=3, absent_free=True, auto={"surface": SURFACE})
    car = det(1400, 400, 1700, 600)  # nowhere near a bay
    events = run(tr, [[car]] * 9)
    assert sorted(tr.slots) == [1, 2, 1000] and tr.names[1000] == "D1"  # after SPAWN_HOLD s, not at once
    assert tr.slots[1000]["state"] == 1 and [e["slot"] for e in events if e["state"] == 1] == [1000]  # an entry, like any bay
    assert [tr.slots[k]["state"] for k in (1, 2)] == [0, 0]  # the grid's own bays are as they were
    retired = []
    for i in range(80):
        tr.update([], SHAPE, now=100.0 + i)
        retired += tr.retired
    assert 1000 not in tr.slots and retired == [1000]
    assert sorted(tr.slots) == [1, 2]  # only the vehicle-driven bay went


@pytest.mark.parametrize("what, car", [
    ("a low-confidence box", {**det(1400, 400, 1700, 600), "conf": 0.3}),
    ("off the measured surface", det(1400, 700, 1700, 980)),
    ("a shown-only object", det(1000, 400, 1300, 600, state=None)),
])
def test_no_bay_is_made_for(what, car):
    tr = FixedSlotTracker(THIN, "foot", hold=2, exit_hold=3, absent_free=True, auto={"surface": SURFACE})
    run(tr, [[car]] * 9)
    assert sorted(tr.slots) == [1, 2], what


def test_a_vehicle_seen_once_or_only_in_passing_gets_no_bay_and_a_pin_or_grid_bay_is_never_removed():
    tr = FixedSlotTracker(THIN, "foot", hold=2, exit_hold=3, absent_free=True, auto={"surface": SURFACE})
    car = det(1400, 400, 1700, 600)
    run(tr, [[car], [], [car], [], [car], []] * 3)  # keeps vanishing: the count starts over each time
    assert sorted(tr.slots) == [1, 2]
    run(tr, [[]] * 60, start=100.0)  # empty for far longer than CLEAR_FRAMES: grid bays stay
    assert sorted(tr.slots) == [1, 2]


def test_a_new_grid_keeps_the_vehicle_driven_bays_and_no_auto_means_no_new_rules():
    tr = FixedSlotTracker(THIN, "foot", hold=2, exit_hold=3, absent_free=True, auto={"surface": SURFACE})
    run(tr, [[det(1400, 400, 1700, 600)]] * 9)
    assert tr.set_slots(THIN[:1]) == [2]  # bay 2 went with the new grid, the vehicle's bay did not
    assert sorted(tr.slots) == [1, 1000] and tr.names == {1: "B1", 1000: "D1"}
    fixed = FixedSlotTracker(THIN, "foot", hold=2, exit_hold=3, absent_free=True)  # a hand-drawn / plain map
    run(fixed, [[det(1400, 400, 1700, 600)]] * 9)
    assert sorted(fixed.slots) == [1, 2]


# --- a tyre or bumper over a bay's line does not make it BUSY (bay_share) -------------------------------------------------------
EDGE_A = {"id": 1, "name": "A", "poly": parse_points("0.20,0.45,0.40,0.60")}
EDGE_B = {"id": 2, "name": "B", "poly": parse_points("0.42,0.45,0.62,0.60")}


def test_a_vehicle_spilling_over_a_bays_line_is_not_in_it_but_one_standing_in_it_is():
    from carparking.fixed_slots import FOOT, ground_on, match_score

    def frac(d):
        return [(x / 2000, y / 1000) for x, y in d["poly"]]

    spill = frac(det(160, 330, 520, 580))  # centre 3 % of the frame left of bay A, a third of its width over the line (the white SUV)
    assert match_score(spill, EDGE_A["poly"], FOOT) > 0  # the near-miss rule alone takes it ...
    assert 0 < ground_on(spill, EDGE_A["poly"]) < 0.4
    assert match_score(spill, EDGE_A["poly"], FOOT, share=0.5) == 0  # ... a share of its ground does not
    inside = frac(det(500, 330, 700, 580))
    assert match_score(inside, EDGE_A["poly"], FOOT, share=0.95) > 1  # the point inside: held to 50 % at most, not to the slider
    # a lifted foot point inside bay A while the vehicle's ground is far in front of it (box bottom 10 % below the bay): no match
    far = frac(det(500, 400, 700, 700))
    assert match_score(far, EDGE_A["poly"], FOOT, lift=0.12) > 1  # the point alone says "in the bay" ...
    assert match_score(far, EDGE_A["poly"], FOOT, lift=0.12, share=0.5) == 0  # ... its ground (the bottom of the box) is not


@pytest.mark.parametrize("auto", [None, {"surface": None}])
def test_the_bay_share_threshold_decides_who_holds_a_bay_and_is_applied_live(auto):
    from carparking.fixed_slots import FOOT

    spill = det(160, 330, 520, 580)
    tr = FixedSlotTracker([EDGE_A, EDGE_B], FOOT, hold=2, exit_hold=5, absent_free=True, auto=auto)
    run(tr, [[spill]] * 5)
    assert tr.slots[1]["state"] != 1 and tr.slots[2]["state"] != 1  # bay A is not BUSY, on any frame
    low = det(500, 380, 700, 630)  # wheels 3 % below bay A's near edge, 60 % of its ground inside
    tr = FixedSlotTracker([EDGE_A, EDGE_B], FOOT, hold=2, exit_hold=5, absent_free=True, auto=auto)
    run(tr, [[low]] * 5)
    assert tr.slots[1]["state"] == 1  # a car a little forward in its bay still holds it
    tr = FixedSlotTracker([EDGE_A, EDGE_B], FOOT, hold=2, exit_hold=5, absent_free=True, auto=auto)
    assert tr.tune({"bay_share": 0.8})["bay_share"] == 0.8  # the Settings slider
    run(tr, [[low]] * 5)
    assert tr.slots[1]["state"] != 1  # asked for 80 % of its ground: 60 % is not enough


# --- sub-bays: two vehicles in one big slot (FixedSlotTracker(split=True)) -----------------------------------------------------
BIG = {"id": 9, "name": "9P", "poly": parse_points("0.30,0.40,0.70,0.60")}
TRUCK = det(620, 200, 900, 590)
SMALL_CAR = det(1000, 380, 1300, 590)


def big_tracker(**kw):
    return FixedSlotTracker([BIG], "ground", hold=2, exit_hold=3, absent_free=True, split=True, **kw)


def test_two_vehicles_in_one_big_pin_get_a_sub_bay_for_the_second():
    tr = big_tracker()
    births = []
    for i in range(8):
        tr.update([TRUCK, SMALL_CAR], SHAPE, now=float(i))
        births.append(list(tr.new_slots))
    assert sorted(tr.slots) == [9, 1000] and tr.names == {9: "9P", 1000: "9P.2"} and tr.parent == {1000: 9}
    assert [tr.slots[k]["state"] for k in (9, 1000)] == [1, 1]  # two vehicles, two BUSY: the count is right
    assert births[:3] == [[], [], []] and births.count([1000]) == 1  # not before SPAWN_FRAMES frames / SPAWN_HOLD s; saved once
    plain = FixedSlotTracker([BIG], "ground", hold=2, exit_hold=3, absent_free=True)  # split off (the default): as it always was
    run(plain, [[TRUCK, SMALL_CAR]] * 8)
    assert sorted(plain.slots) == [9] and plain.slots[9]["state"] == 1


def test_one_vehicle_seen_twice_or_for_two_frames_gets_no_sub_bay():
    tr = big_tracker()
    twin = det(1002, 382, 1302, 592)  # the same car boxed twice (whole frame + zoom): the footprints are one
    run(tr, [[SMALL_CAR, twin]] * 8)
    assert sorted(tr.slots) == [9]
    tr = big_tracker()
    run(tr, [[TRUCK, SMALL_CAR], [TRUCK, SMALL_CAR], [TRUCK], [TRUCK, SMALL_CAR], [TRUCK, SMALL_CAR], [TRUCK]] * 2)
    assert sorted(tr.slots) == [9]  # never three frames in a row


def test_a_sub_bay_goes_when_its_vehicle_leaves_and_with_its_parent():
    tr = big_tracker()
    run(tr, [[TRUCK, SMALL_CAR]] * 8)
    retired = []
    stays = SMALL_CAR if tr.slots[1000]["poly"][:, 0].mean() < 0.5 else TRUCK  # one of them holds the parent, the other the sub-bay
    for i in range(60):
        tr.update([stays], SHAPE, now=100.0 + i)  # the sub-bay's vehicle left, the other stays
        retired += tr.retired
    assert retired == [1000] and sorted(tr.slots) == [9] and tr.slots[9]["state"] == 1 and tr.parent == {}
    tr = big_tracker()
    run(tr, [[TRUCK, SMALL_CAR]] * 8)
    other = {"id": 5, "name": "X", "poly": parse_points("0.1,0.1,0.2,0.2")}
    assert tr.set_slots([other]) == [9, 1000] and sorted(tr.slots) == [5]  # the parent's map changed: the sub-bay goes with it


def test_vehicle_driven_bays_are_saved_restored_after_a_restart_and_their_ids_never_reused(tmp_path):
    class Fake:
        conf = None

        def __init__(self):
            self.dets = [dict(TRUCK), dict(SMALL_CAR)]

        def __call__(self, frame, sliced=False):
            return [dict(d) for d in self.dets]

    def monitor(db, fake):
        return ParkingMonitor(fake, db, fixed=[BIG], hold=2, exit_hold=3, absent_free=True, split=True)

    db, fake = HistoryDB(tmp_path / "h.db", "Gate", detector="yolo | fixed"), Fake()
    mon = monitor(db, fake)
    for i in range(8):
        mon.process(FRAME, t=float(i))
    assert [s["id"] for s in mon.state()["slots"]] == [9, 1000] and mon.state()["slots"][1]["parent"] == 9
    assert sorted(db.load_slots()) == [1000]  # saved when it was made
    db2 = HistoryDB(tmp_path / "h.db", "Gate", detector="yolo | fixed")  # the worker restarted
    mon2 = monitor(db2, Fake())
    mon2.start(SHAPE)
    assert sorted(mon2.tracker.slots) == [9, 1000] and mon2.tracker.names[1000] == "9P.2" and mon2.tracker._dyn_next == 1001
    fake.dets = []  # everyone leaves: the bay is cleared, closed in the history and not restored again
    for i in range(60):
        mon.process(FRAME, t=100.0 + i)
    assert [s["id"] for s in mon.state()["slots"]] == [9] and db.load_slots() == {}
    mon3 = monitor(HistoryDB(tmp_path / "h.db", "Gate", detector="yolo | fixed"), Fake())
    mon3.start(SHAPE)
    assert sorted(mon3.tracker.slots) == [9] and mon3.tracker._dyn_next == 1001  # (1000 is never handed out again)


@pytest.mark.parametrize("share", [0.5, 0.8, 0.95])
def test_sub_bays_keep_every_vehicle_busy_whatever_bay_held_is_and_while_the_boxes_jitter(share):
    other = {"id": 3, "name": "3", "poly": parse_points("0.05,0.40,0.25,0.60")}
    tr = FixedSlotTracker([BIG, other], "ground", hold=2, exit_hold=3, absent_free=True, split=True, bay_share=share)
    side = det(60, 380, 440, 590)  # the fourth car, in its own slot
    for i in range(40):
        j = (i % 5) - 2  # the detector's box wobbles a few px frame to frame
        tr.update([det(620 + j, 200, 900 + j, 590 + j), det(1000 + j, 380, 1300 + j, 590 + j), side], SHAPE, now=float(i))
    busy = [k for k, s in tr.slots.items() if s["state"] == 1]
    assert sorted(tr.slots) == [3, 9, 1000] and busy == [3, 9, 1000], (tr.names, tr.held, busy)
    assert len([k for k in tr.slots if k < 1000]) == 2  # capacity fixed: the sub-bay is not a new slot


def test_a_sub_bay_counts_busy_but_not_as_a_new_space_and_an_unplaced_vehicle_says_why(tmp_path):
    class Fake:
        conf = None
        dets = [dict(TRUCK), dict(SMALL_CAR)]

        def __call__(self, frame, sliced=False):
            return [dict(d) for d in self.dets]

    db = HistoryDB(tmp_path / "h.db", "Gate", detector="yolo | fixed")
    fake = Fake()
    mon = ParkingMonitor(fake, db, fixed=[BIG], hold=2, exit_hold=3, absent_free=True, split=True)
    mon.process(FRAME, t=0.0)
    st = mon.state()  # first frame: the second vehicle waits for its sub-bay
    assert st["counts"]["slots"] == 1 and st["counts"]["unplaced"] == 1 and "steady frames" in st["unplaced"][0]["why"]
    for i in range(1, 8):
        mon.process(FRAME, t=float(i))
    c = mon.state()["counts"]
    assert (c["busy"], c["slots"], c["sub_bays"], c["unplaced"]) == (2, 2, 1, 0)  # 2 vehicles BUSY of 2 spaces
    tr = big_tracker()  # a weak box is not counted: it says so
    tr.update([dict(TRUCK, conf=0.3), dict(SMALL_CAR, conf=0.3)], SHAPE, now=0.0)  # (one holds the slot, the other has no bay)
    assert [u["why"] for u in tr.unplaced] == ["confidence under 0.45"]


def test_a_second_box_of_the_sub_bays_vehicle_does_not_make_another_sub_bay():
    tr = big_tracker()
    twin = det(1004, 384, 1304, 594)
    run(tr, [[TRUCK, SMALL_CAR, twin]] * 10)
    assert sorted(tr.slots) == [9, 1000]


# --- capacity: the operator says a pin holds N vehicles ------------------------------------------------------------------------
THIRD = det(1320, 420, 1400, 590)


def test_capacity_is_validated_stored_and_default_one():
    poly = BIG["poly"]
    assert "capacity" not in validate_slots([{"id": 1, "poly": poly}])[0]
    assert "capacity" not in validate_slots([{"id": 1, "poly": poly, "capacity": 1}])[0]
    assert validate_slots([{"id": 1, "poly": poly, "capacity": "4"}])[0]["capacity"] == 4
    for bad in (0, -1, 2.5, "x", 21):
        with pytest.raises(ValueError, match="capacity"):
            validate_slots([{"id": 1, "poly": poly, "capacity": bad}])


def _cap_monitor(tmp_path, cap, dets):
    class Fake:
        conf = None

        def __call__(self, frame, sliced=False):
            return [dict(d) for d in dets]

    return ParkingMonitor(Fake(), HistoryDB(tmp_path / "c.db", "Gate", detector="yolo | fixed"), fixed=[{**BIG, "capacity": cap}],
                          hold=2, exit_hold=3, absent_free=True, split=True)


@pytest.mark.parametrize("cap, busy", [(4, 3), (2, 2)])
def test_a_pin_with_capacity_counts_its_vehicles_up_to_the_limit_and_adds_to_the_total(tmp_path, cap, busy):
    mon = _cap_monitor(tmp_path, cap, [TRUCK, SMALL_CAR, THIRD])
    for i in range(8):
        mon.process(FRAME, t=float(i))
    st = mon.state()
    assert (st["counts"]["slots"], st["counts"]["busy"], st["counts"]["free"]) == (cap, busy, cap - busy)
    assert [s["id"] for s in st["slots"]] == [9]  # no sub-bays: the capacity does it, with no lines
    assert st["slots"][0]["capacity"] == cap and st["slots"][0]["vehicles"] == busy


def test_capacity_pin_total_busy_free_with_other_slots(tmp_path):
    other = {"id": 3, "name": "3", "poly": parse_points("0.05,0.40,0.25,0.60")}

    class Fake:
        conf = None

        def __call__(self, frame, sliced=False):
            return [dict(d) for d in (TRUCK, SMALL_CAR, det(60, 380, 440, 590))]

    mon = ParkingMonitor(Fake(), HistoryDB(tmp_path / "d.db", "Gate", detector="yolo | fixed"), fixed=[{**BIG, "capacity": 2}, other],
                         hold=2, exit_hold=3, absent_free=True, split=True)
    for i in range(8):
        mon.process(FRAME, t=float(i))
    c = mon.state()["counts"]
    assert (c["slots"], c["busy"], c["free"]) == (3, 3, 0)


def test_an_auto_grid_pin_keeps_its_capacity():
    from carparking.geometry_live import merge_pins
    out = merge_pins([], [{"id": 1, "name": "P1", "poly": BIG["poly"], "capacity": 2}, {"id": 2, "name": "P2", "poly": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}])
    assert [s.get("capacity") for s in out] == [2, None]


def cap_tracker(cap=4):
    return FixedSlotTracker([{**BIG, "capacity": cap}], "ground", hold=2, exit_hold=3, absent_free=True, split=True)


def test_capacity_count_is_steady_a_dropped_box_or_a_flicker_does_not_move_it():
    three = [TRUCK, SMALL_CAR, THIRD]
    for frames in ([three[:2]], [three[:2]] * 3, [three, three[:2], three]):  # a box missing for 1-3 frames
        tr = cap_tracker()
        run(tr, [three] * 6)
        assert tr.load[9] == 3
        run(tr, frames, start=10.0)
        assert tr.load[9] == 3, frames
    run(tr, [three[:2]] * 4, start=20.0)  # four frames with one fewer: really gone
    assert tr.load[9] == 2
    flick = cap_tracker()
    run(flick, [two := [TRUCK, SMALL_CAR]] * 6)
    run(flick, [two + [THIRD], two, two, two, two])  # one frame with a phantom third box
    assert flick.load[9] == 2
    run(flick, [two + [THIRD]] * 2)  # a new car that stays: counted from the second frame
    assert flick.load[9] == 3


def test_capacity_counts_cars_with_overlapping_boxes_but_merges_duplicate_boxes_and_says_why():
    tr = cap_tracker()
    a, b = det(1000, 380, 1300, 590), det(1100, 390, 1400, 590)  # side by side, boxes overlap (footprint IoU ~0.5 or less)
    run(tr, [[TRUCK, a, b]] * 6)
    assert tr.load[9] == 3
    twin = det(1004, 384, 1304, 594)  # the same car boxed twice
    tr = cap_tracker()
    run(tr, [[TRUCK, a, twin]] * 6)
    assert tr.load[9] == 2 and "same vehicle" in tr.load_notes[9]


def test_capacity_counts_a_car_half_over_the_pins_edge_but_not_a_neighbours():
    tr = cap_tracker()
    half = det(1200, 380, 1560, 590)  # its middle is inside the pin
    neighbour = det(1500, 380, 1800, 590)  # beside the pin
    run(tr, [[TRUCK, half, neighbour]] * 6)
    assert tr.load[9] == 2


def test_history_sessions_follow_the_capacity_slots_vehicles_and_match_the_dashboard(tmp_path):
    dets = []

    class Fake:
        conf = None

        def __call__(self, frame, sliced=False):
            return [dict(d) for d in dets]

    db = HistoryDB(tmp_path / "s.db", "Gate", detector="yolo | fixed")
    mon = ParkingMonitor(Fake(), db, fixed=[{**BIG, "capacity": 4}], hold=2, exit_hold=3, absent_free=True, split=True)
    t = [0.0]

    def frames(n):
        for _ in range(n):
            mon.process(FRAME, t=t[0])
            t[0] += 1.0

    def check(vehicles):
        st = mon.state()["counts"]
        open_ = len(db.sessions(open_only=True))
        assert (st["busy"], open_) == (vehicles, vehicles), (st, open_)

    dets[:] = [TRUCK]
    frames(8)
    check(1)
    dets[:] = [TRUCK, SMALL_CAR]
    frames(8)
    check(2)
    dets[:] = [TRUCK, SMALL_CAR, THIRD]
    frames(8)
    check(3)
    frames(1)
    dets[:] = [TRUCK, SMALL_CAR]  # one leaves
    frames(8)
    check(2)
    dets[:] = [TRUCK, SMALL_CAR, THIRD]  # one box for a single frame only: no spike in the history
    frames(1)
    dets[:] = [TRUCK, SMALL_CAR]
    frames(8)
    check(2)
    dets[:] = []  # all gone
    frames(10)
    assert db.sessions(open_only=True) == [] and mon.state()["counts"]["busy"] == 0
    ev = db.events()
    assert sum(e["event"] == "entry" for e in ev) == sum(e["event"] == "exit" for e in ev) >= 3


def test_open_sessions_are_synced_to_the_live_count_at_start_and_on_every_count_change(tmp_path):
    dets = [TRUCK, SMALL_CAR, THIRD]

    class Fake:
        conf = None

        def __call__(self, frame, sliced=False):
            return [dict(d) for d in dets]

    path = tmp_path / "sync.db"
    other = {"id": 3, "name": "3", "poly": parse_points("0.05,0.40,0.25,0.60")}
    db = HistoryDB(path, "Gate", detector="yolo | fixed")
    # what an earlier run left behind: one session in the pin, one in a slot that is not on the map any more, one in an empty slot
    for slot, t in ((9, 1.0), (77, 2.0), (3, 3.0)):
        db.apply({"slot": slot, "state": 1, "time": t, "initial": False, "occluded": False})
    mon = ParkingMonitor(Fake(), db, fixed=[{**BIG, "capacity": 4}, other], hold=2, exit_hold=3, absent_free=True, split=True)
    for i in range(10):
        mon.process(FRAME, t=100.0 + i)
    open_ = db.sessions(open_only=True)
    assert sorted(s["slot_id"] for s in open_) == [9, 9, 9]  # 3 vehicles in the pin; 77 (gone) and 3 (empty) closed
    assert mon.state()["counts"]["busy"] == len(open_) == 3
    notes = {s["slot_id"]: s["note"] for s in db.sessions(limit=20) if s["out_ts"] is not None}
    assert "no longer on the slot map" in notes[77] and "empty" in notes[3]
    dets[:] = dets[:2]  # a count change alone (the slot stays BUSY): one session closes at once the count settles
    for i in range(10, 20):
        mon.process(FRAME, t=100.0 + i)
    assert len(db.sessions(open_only=True)) == mon.state()["counts"]["busy"] == 2
    db.con.execute("DELETE FROM history WHERE slot_id = 9 AND exit_ts IS NULL")  # sessions lost (a restored database)
    db.con.commit()
    mon.process(FRAME, t=200.0)
    assert len(db.sessions(open_only=True)) == mon.state()["counts"]["busy"] == 2  # backfilled in the next frame


def test_capacity_counts_an_unheld_vehicle_just_past_the_pins_edge_but_not_one_far_away():
    tight = det(1000, 440, 1300, 640)  # its wheels stand 4 % of the frame below the pin's lower edge (y 0.60): a pin drawn tight
    far = det(1000, 500, 1300, 700)  # well beyond that
    grid = lambda: FixedSlotTracker([{**BIG, "capacity": 4}], "ground", hold=2, exit_hold=3, absent_free=True,  # noqa: E731
                                    auto={"surface": None}, split=True)  # (the auto grid)
    tr = grid()
    run(tr, [[TRUCK, tight]] * 6)
    assert tr.load[9] == 2 and "outside its edge" in tr.load_notes[9]
    tr = grid()
    run(tr, [[TRUCK, far]] * 6)
    assert tr.load[9] == 1
    held = FixedSlotTracker([{**BIG, "capacity": 4}, {"id": 4, "name": "4", "poly": parse_points("0.45,0.60,0.80,0.80")}], "ground",
                            hold=2, exit_hold=3, absent_free=True)
    run(held, [[TRUCK, tight]] * 6)
    assert held.load[9] == 1  # a vehicle another slot holds is that slot's, however near


# --- Fixed maps are strict: no sub-bays, no grace margin, extra vehicles reported --------------------------------------------
def fixed_tracker(slots, **kw):
    return FixedSlotTracker(slots, "ground", hold=2, exit_hold=3, absent_free=True, **kw)  # (no auto grid, no sub-bays)


def test_a_fixed_map_caps_busy_at_its_slots_and_reports_the_extra_vehicle():
    a = {"id": 1, "name": "S1", "poly": parse_points("0.05,0.40,0.25,0.60")}
    b = {"id": 2, "name": "S2", "poly": parse_points("0.30,0.40,0.70,0.60")}
    tr = fixed_tracker([a, b])
    extra = det(1000, 380, 1300, 590)  # a second vehicle in S2, beside TRUCK
    run(tr, [[det(60, 380, 440, 590), TRUCK, extra]] * 10)
    assert sorted(tr.slots) == [1, 2] and not tr.parent  # no sub-bay, however long it stays
    assert [tr.slots[k]["state"] for k in (1, 2)] == [1, 1]  # BUSY 2 on 2 slots
    assert [u["why"] for u in tr.unplaced] == ["no free slot (Fixed map: capacity 1)"]
    run(tr, [[det(60, 380, 440, 590)]] * 3, start=20.0)
    assert tr.unplaced == []  # nobody extra
    run(tr, [[det(60, 380, 440, 590), det(1500, 100, 1700, 200)]], start=30.0)  # a vehicle standing in no slot
    assert [u["why"] for u in tr.unplaced] == ["in no slot of the Fixed map"]


def test_a_fixed_capacity_pin_counts_strictly_inside_and_the_auto_grids_counts_the_margin():
    tight = det(1000, 440, 1300, 640)  # just past the pin's lower edge
    fixed = fixed_tracker([{**BIG, "capacity": 4}])
    run(fixed, [[TRUCK, tight]] * 6)
    assert fixed.load[9] == 1 and [u["why"] for u in fixed.unplaced] == ["no free slot (Fixed map: capacity 4)"]
    grid = fixed_tracker([{**BIG, "capacity": 4}], auto={"surface": None}, split=True)
    run(grid, [[TRUCK, tight]] * 6)
    assert grid.load[9] == 2 and grid.unplaced == []  # counted by the pin, so no longer "unplaced"


def test_restore_skips_the_vehicle_driven_bays_of_a_fixed_map_and_the_monitor_retires_them(tmp_path):
    class Fake:
        conf = None

        def __call__(self, frame, sliced=False):
            return [dict(TRUCK)]

    poly = parse_points("0.60,0.45,0.70,0.58")
    tr = fixed_tracker([BIG])
    assert tr.restore({1001: np.float32(poly), 9: np.float32(BIG["poly"])}, 1002) == [1001]
    assert sorted(tr.slots) == [9] and tr._dyn_next == 1002  # (its id is still never handed out again)
    grid = fixed_tracker([BIG], auto={"surface": None}, split=True)
    assert grid.restore({1001: np.float32(poly)}, 1002) == [] and sorted(grid.slots) == [9, 1001]  # the auto grid keeps them
    db = HistoryDB(tmp_path / "r.db", "Gate", detector="yolo | fixed")
    db.save_slots(type("T", (), {"slots": {1001: {"poly": np.float32(poly)}}})(), [1001])
    db.apply({"slot": 1001, "state": 1, "time": 1.0, "initial": False, "occluded": False})
    mon = ParkingMonitor(Fake(), db, fixed=[BIG], hold=2, exit_hold=3, absent_free=True)  # a Fixed map: split off
    mon.start(SHAPE)
    assert sorted(mon.tracker.slots) == [9] and db.load_slots() == {}  # retired
    assert db.sessions(open_only=True) == []  # its open session was closed
    assert "no sub-bays" in db.sessions()[0]["note"]


def test_the_pin_grace_setting_moves_how_far_past_the_edge_a_capacity_pin_counts_on_the_auto_grid_only():
    near = det(1000, 440, 1300, 640)  # footprint middle 1 % of the frame below the pin's lower edge (y 0.60), height 0.2
    mid = det(1000, 470, 1300, 670)  # 4 %
    far = det(1000, 500, 1300, 700)  # 7 %
    grid = lambda: FixedSlotTracker([{**BIG, "capacity": 4}], "ground", hold=2, exit_hold=3, absent_free=True,  # noqa: E731
                                    auto={"surface": None}, split=True)

    def counted(car, reach=None, make=grid):
        tr = make()
        if reach is not None:
            assert tr.tune({"pin_reach": reach})["pin_reach"] == reach
        run(tr, [[TRUCK, car]] * 6)
        return tr.load[9]

    assert (counted(near), counted(mid), counted(far)) == (2, 2, 1)  # the default: 0.25 x the car's 0.2 height = 5 % of the frame
    assert counted(mid, 0.0) == 1 and counted(near, 0.0) == 1  # 0: strictly inside
    assert counted(far, 0.5) == 2 and counted(mid, 0.5) == 2  # 0.5 x 0.2 = 10 %
    strict = lambda: fixed_tracker([{**BIG, "capacity": 4}])  # noqa: E731  (a Fixed map)
    assert counted(near, 0.5, strict) == 1  # a Fixed map ignores the setting


def test_a_failed_repair_of_the_history_is_visible_and_clears_when_writing_works_again(tmp_path):
    import sqlite3
    dets = [TRUCK, SMALL_CAR, THIRD]

    class Fake:
        conf = None

        def __call__(self, frame, sliced=False):
            return [dict(d) for d in dets]

    db = HistoryDB(tmp_path / "m.db", "Gate", detector="yolo | fixed")
    mon = ParkingMonitor(Fake(), db, fixed=[{**BIG, "capacity": 4}], hold=2, exit_hold=3, absent_free=True, split=True)
    for i in range(8):
        mon.process(FRAME, t=float(i))
    assert mon.session_check["expected"] == mon.session_check["open"] == 3 and mon.session_check["since"] is None
    assert mon.session_error == ""
    real = db.sync_open

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    db.sync_open = locked
    dets[:] = dets[:2]  # one vehicle leaves while the history cannot be written
    for i in range(8, 16):
        mon.process(FRAME, t=float(i))  # (the frame goes on: no exception)
    chk = mon.session_check
    assert (chk["expected"], chk["open"]) == (2, 3) and chk["since"] is not None and chk["at"] - chk["since"] >= 3
    assert "database is locked" in mon.session_error
    db.sync_open = real
    mon.process(FRAME, t=16.0)
    chk = mon.session_check
    assert (chk["expected"], chk["open"], chk["since"]) == (2, 2, None) and mon.session_error == ""  # repaired in the next frame


def test_unknown_slots_keep_their_sessions_and_do_not_count_as_a_mismatch(tmp_path):
    class Fake:
        conf = None

        def __call__(self, frame, sliced=False):
            return [dict(TRUCK)]

    db = HistoryDB(tmp_path / "u.db", "Gate", detector="yolo | fixed")
    other = {"id": 3, "name": "3", "poly": parse_points("0.05,0.40,0.25,0.60")}
    mon = ParkingMonitor(Fake(), db, fixed=[{**BIG, "capacity": 1}, other], hold=2, exit_hold=3, absent_free=True)
    db.apply({"slot": 3, "state": 1, "time": 0.5, "initial": False})  # a session of a slot the tracker has not decided yet
    mon.process(FRAME, t=0.0)
    assert mon.tracker.slots[3]["state"] is None
    assert mon.session_check["expected"] == 0 and mon.session_check["open"] == 0 and mon.session_check["since"] is None
