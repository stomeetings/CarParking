"""Slot suggestions (carparking/slot_suggest.py) and the pieces of the live grid they stand on: the quick re-tile around today's
vehicles, id carry-over, the diff against the running bays, stable pin ids with nearest-bay binding, the frozen grid of an accepted
suggestion, and the merge into a hand-drawn map."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_geometry_live import CARS, H, W, box_of, det, record  # noqa: E402

from carparking.fixed_slots import validate_slots
from carparking.geometry_live import MANUAL, ActiveGeometry, GeometryFeed, merge_pins
from carparking.slot_suggest import (EDITED, baseline_hash, bind_pin_ids, carry_ids, diff, merge_into_fixed, suggest_slots,
                                     vehicle_anchors)


def sq(x, y, s=0.1):
    x2, y2 = round(x + s, 4), round(y + s, 4)
    return [[x, y], [x2, y], [x2, y2], [x, y2]]


def bay(i, x, y=0.5, **kw):
    return {"id": i, "name": f"B{i}", "poly": sq(x, y), "lift": 0.01, "source": "measured", **kw}


@pytest.fixture(scope="module")
def geo():
    rec = record()
    rec["calibration"]["confidence"] = "HIGH"
    return ActiveGeometry({"ver": 1, "record": rec})


def test_retiling_around_no_vehicles_is_the_live_grid(geo):
    live = GeometryFeed._grid_slots(geo, W, H, None)
    sugg, lined = suggest_slots(geo, [], W, H, None)
    assert lined == "calibration" and [s["poly"] for s in sugg] == [s["poly"] for s in live]  # nothing new to line up with


def test_vehicles_on_the_still_become_anchors_and_far_or_cut_ones_do_not(geo):
    dets = [det(box_of(x, y)) for x, y in CARS]
    assert len(vehicle_anchors(geo, dets, W, H)) == len(CARS)
    cut = det((0, 100, 200, H))  # touches the frame edge: its position is not trusted
    person = det(box_of(0, 12), label="person", kind="other")
    assert len(vehicle_anchors(geo, dets + [cut, person], W, H)) == len(CARS)
    sugg, lined = suggest_slots(geo, dets, W, H, None)
    assert lined == "vehicles" and len(sugg) >= len(CARS) and [s["id"] for s in sugg] == list(range(1, len(sugg) + 1))


def test_one_vehicle_is_not_enough_to_line_the_rows_up(geo):
    assert suggest_slots(geo, [det(box_of(*CARS[0]))], W, H, None)[1] == "calibration"


def test_the_expected_count_continues_the_lane_in_a_suggestion_too(geo):
    from carparking.geometry_live import MAX_EXTRAPOLATED
    base, _ = suggest_slots(geo, [], W, H, None)
    measured = [s for s in base if s["source"] == "measured"]
    met, _ = suggest_slots(geo, [], W, H, len(measured))  # the count is met: nothing is guessed
    assert [s["source"] for s in met] == ["measured"] * len(measured)
    many, _ = suggest_slots(geo, [], W, H, len(measured) + 50)  # never more than the camera model is trusted for
    assert len(measured) <= len(many) <= len(measured) + MAX_EXTRAPOLATED
    assert all(s["source"] == "extrapolated" for s in many[len(measured):])


def test_ids_are_carried_by_overlap_and_new_bays_get_fresh_ones():
    old = [bay(3, 0.1), bay(7, 0.3, name="Door")]
    new = [bay(1, 0.11), bay(2, 0.31), bay(3, 0.8)]  # the first two sit over old bays, the third is new
    out = carry_ids(new, old, reserved=[8])
    assert [(s["id"], s["name"]) for s in out] == [(3, "B3"), (7, "Door"), (9, "B9")]  # fresh ids skip the reserved one
    assert carry_ids([bay(1, 0.11)], [bay(5, 0.1), bay(6, 0.1)])[0]["id"] in (5, 6)  # one to one: one old bay is taken once
    assert [s["id"] for s in carry_ids([bay(1, 0.1), bay(2, 0.1)], [bay(5, 0.1)])] == [5, 6]


def test_diff_names_what_accepting_would_change():
    old = [bay(1, 0.1), bay(2, 0.3), bay(3, 0.5), bay(9, 0.7, source=MANUAL)]
    new = [bay(1, 0.1), bay(2, 0.3, 0.52), bay(4, 0.8)]
    d = diff(new, old)
    assert d == {"added": [4], "removed": [3], "moved": [2], "unchanged": [1]}  # the pin (manual) is on neither side


def test_a_baseline_fingerprint_changes_with_any_bay():
    a = [bay(1, 0.1), bay(2, 0.3)]
    assert baseline_hash(a) == baseline_hash(list(reversed(a)))
    assert baseline_hash(a) != baseline_hash([bay(1, 0.1), bay(2, 0.31)])
    assert baseline_hash(a) != baseline_hash(a[:1])


# --- pins: a stable id of their own and the nearest bay -----------------------------------------------------------------------
def test_a_pin_keeps_its_own_id_when_the_grid_changes_and_binds_to_the_nearest_bay():
    pin = {"id": 1, "name": "P1", "poly": sq(0.52, 0.5), "capacity": 4, "slot_id": 9}
    grid = [bay(i, 0.1 * i) for i in (1, 2, 3, 4, 5, 6)]  # bay 5 lies under the pin
    out = merge_pins(grid, [pin])
    p = next(s for s in out if s["source"] == MANUAL)
    assert p["id"] == 9 and p["capacity"] == 4 and p["name"] == "P1"  # not the overlapped bay's id any more
    assert 5 not in [s["id"] for s in out] and p["binds_to"] in (4, 6)  # bay 5 was replaced, the nearest left is a neighbour
    again = merge_pins([bay(i, 0.1 * i + 0.02) for i in (1, 2, 3)], [pin])  # a different grid: the pin is still slot 9
    assert next(s for s in again if s["source"] == MANUAL)["id"] == 9


def test_a_bay_holding_the_pins_id_steps_aside_and_a_pin_without_an_id_behaves_as_before():
    pin = {"id": 1, "name": "P1", "poly": sq(0.8, 0.2), "slot_id": 2}
    out = merge_pins([bay(1, 0.1), bay(2, 0.3)], [pin])  # the pin does not overlap bay 2, which holds its id
    ids = {s["id"]: s["source"] for s in out}
    assert ids[2] == MANUAL and sorted(ids) == [1, 2, 3] and ids[3] == "measured"
    plain = merge_pins([bay(1, 0.1), bay(2, 0.3)], [{"id": 1, "name": "1", "poly": sq(0.305, 0.5)}])
    assert [(s["id"], s["source"]) for s in plain] == [(1, "measured"), (2, MANUAL)]  # lowest overlapped id, as always


def test_binding_a_pin_to_its_running_id_and_validation_keeps_it():
    pin = {"id": 1, "name": "P1", "poly": sq(0.52, 0.5), "capacity": 4}
    running = merge_pins([bay(i, 0.1 * i) for i in (1, 2, 3, 4, 5, 6)], [pin])
    bound = bind_pin_ids([pin, {"id": 2, "poly": sq(0.05, 0.05)}], running)
    assert bound[0]["slot_id"] == 5 and "slot_id" not in bound[1]  # the second pin stands for nothing running
    assert bind_pin_ids([{**pin, "slot_id": 9}], running)[0]["slot_id"] == 9  # already bound: never rebound
    assert validate_slots(bound[:1])[0]["slot_id"] == 5
    for bad in (0, -1, 1000, 2.5, "x"):
        with pytest.raises(ValueError):
            validate_slots([{**pin, "slot_id": bad}])


# --- the frozen grid of an accepted suggestion ----------------------------------------------------------------------------------
def test_an_accepted_suggestion_is_the_grid_exactly_as_saved(geo):
    rec = record()
    rec["calibration"]["confidence"] = "HIGH"
    frozen = [bay(4, 0.2, source=EDITED), bay(5, 0.4)]
    rec["grid"] = frozen
    g = ActiveGeometry({"ver": 2, "record": rec})
    slots = GeometryFeed._grid_slots(g, W, H, 50)  # the expected count was applied when the suggestion was made
    assert [(s["id"], s["source"]) for s in slots] == [(4, EDITED), (5, "measured")] and slots[0]["poly"] == frozen[0]["poly"]
    assert slots is not g.frozen and slots[0] is not g.frozen[0]  # a copy: the pins merged on top never touch the record
    for broken in ([{"id": 1}], [bay(1, 0.1), bay(1, 0.3)], [], "x", [bay(1, 1.5)]):
        rec["grid"] = broken
        assert ActiveGeometry({"ver": 3, "record": rec}).frozen is None  # unusable: the live tiling as before
    rec["grid"] = None
    assert GeometryFeed._grid_slots(ActiveGeometry({"ver": 4, "record": rec}), W, H, None) == GeometryFeed._grid_slots(geo, W, H, None)


# --- a Fixed camera: merge chosen bays into the hand-drawn map ---------------------------------------------------------------------
def test_chosen_bays_replace_or_join_the_drawn_map_and_the_rest_is_untouched():
    drawn = [{"id": 1, "name": "S1", "poly": sq(0.1, 0.5), "capacity": 3}, {"id": 2, "name": "S2", "poly": sq(0.3, 0.5)}]
    grid = [bay(1, 0.12), bay(2, 0.32), bay(3, 0.7)]
    out = merge_into_fixed(drawn, grid, [1, 3])
    assert [s["id"] for s in out] == [1, 2, 3]
    assert out[0]["poly"] == grid[0]["poly"] and out[0]["name"] == "S1" and out[0]["capacity"] == 3  # replaced, still S1
    assert out[1] == drawn[1]  # not chosen: exactly as drawn
    assert out[2]["poly"] == grid[2]["poly"] and set(out[2]) == {"id", "name", "poly"}
    assert merge_into_fixed(drawn, grid, [])  == drawn  # nothing chosen, nothing changes
    assert [s["id"] for s in merge_into_fixed(drawn, grid, None)] == [1, 2, 3]
    assert drawn[0]["poly"] == sq(0.1, 0.5)  # the input is untouched
    assert np.allclose(validate_slots(out)[2]["poly"], grid[2]["poly"], atol=1e-4)
