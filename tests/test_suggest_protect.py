"""A Fixed camera's running slots (B1..B8, the pin P1 ...) are protected in a slot suggestion (slot_suggest.protect_baseline, the suggest job and the
fine-tune endpoint in api/geometry_routes.py): the draft carries them in exactly as they run, never marks one removed or moved, adds only bays that do not
lie on them, and refuses to move or remove them. An auto-grid camera's draft is unchanged. Synthetic polygons and a mock network; no models, no ./data."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from test_setup_api import AUTH, Net, cam_id, make  # noqa: E402
from test_suggest_api import CARS, ready, suggest, worker_sees  # noqa: E402

from carparking.slot_suggest import PROTECTED, carry_ids, diff, overlap_share, protect_baseline  # noqa: E402


@pytest.fixture
def env(tmp_path):
    return {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
            "CARPARK_RTSP_URL": "rtsp://admin:secret@cam.local:554/Streaming/Channels/101", "CARPARK_API_TOKEN": "t0k3n",
            "CARPARK_SPA_DIR": str(tmp_path / "dist"), "CARPARK_GEOMETRY_ENABLED": "1", "CARPARK_SETUP_URL": "http://setup:8100",
            "CARPARK_GO2RTC_INTERNAL": "http://g"}


def box(x, y, w=0.1, h=0.1):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def shrunk(poly, k=0.55):
    p = np.asarray(poly, float)
    c = p.mean(axis=0)
    return (c + (p - c) * k).round(4).tolist()


def slot(i, poly, **kw):
    return {"id": i, "name": f"B{i}", "poly": poly, "lift": 0.0, **kw}


# ---- the pure rule -----------------------------------------------------------------------------------------------------------------------------
def test_the_overlap_measure_calls_neighbours_apart_and_a_bay_on_a_bay_together():
    assert overlap_share(box(0.1, 0.1), box(0.2, 0.1)) < 0.02  # side by side, touching along an edge
    assert overlap_share(box(0.1, 0.1), box(0.1, 0.2)) < 0.02  # one behind the other
    assert overlap_share(box(0.1, 0.1), box(0.1, 0.1)) == pytest.approx(1.0, abs=0.01)
    assert overlap_share(shrunk(box(0.1, 0.1)), box(0.1, 0.1)) > 0.95  # a small bay inside a big one: the IoU is low, this is not
    assert 0.15 < overlap_share(box(0.1, 0.1), box(0.18, 0.1)) < 0.25  # a fifth over


def test_the_old_matching_called_a_running_slot_removed_and_laid_a_bay_on_it_the_protected_draft_does_not():
    running = [slot(i, shrunk(box(0.05 + 0.12 * i, 0.5))) for i in range(1, 6)]  # slots that sit inside the calibration's tiles
    suggested = [slot(i, box(0.05 + 0.12 * i, 0.5)) for i in range(1, 6)]  # the tiles the suggestion lays (IoU ~0.3 with each running slot)
    old = carry_ids(suggested, running)
    d = diff(old, running)
    assert d["removed"] and any(o["id"] > 5 for o in old)  # the bug: running slots "removed", fresh bays over them
    grid, protected = protect_baseline(suggested, running)
    assert protected == [1, 2, 3, 4, 5] and [g["id"] for g in grid] == [1, 2, 3, 4, 5]  # nothing laid on them
    d = diff(grid, running)
    assert d["removed"] == [] and d["moved"] == [] and d["added"] == [] and d["unchanged"] == [1, 2, 3, 4, 5]


def test_running_slots_are_carried_in_exactly_and_only_non_overlapping_bays_are_added_beyond_them():
    running = [slot(1, box(0.1, 0.5)), slot(2, box(0.2, 0.5)), slot(9, box(0.5, 0.1, 0.2, 0.1), name="P1", capacity=4)]
    running[2]["name"] = "P1"
    suggested = [slot(1, box(0.1, 0.5)),  # on B1
                 slot(2, box(0.205, 0.5)),  # on B2, nearly
                 slot(3, box(0.3, 0.5)),  # adjacent to B2: new
                 slot(4, box(0.4, 0.5)),  # new
                 slot(5, box(0.55, 0.12, 0.1, 0.08))]  # on the pin
    grid, protected = protect_baseline(suggested, running)
    by = {g["id"]: g for g in grid}
    assert protected == [1, 2, 9]
    for r in running:  # exactly as they run
        assert by[r["id"]]["poly"] == r["poly"] and by[r["id"]]["name"] == r["name"] and by[r["id"]]["source"] == PROTECTED and by[r["id"]]["protected"] is True
    assert by[9]["capacity"] == 4 and "capacity" not in by[1]
    new = [g for g in grid if not g.get("protected")]
    assert [g["poly"] for g in new] == [box(0.3, 0.5), box(0.4, 0.5)]  # only the ones that lie on nothing
    assert all(g["id"] > 9 for g in new)  # fresh ids above every running id
    assert all(overlap_share(n["poly"], r["poly"]) <= 0.1 for n in new for r in running)
    assert [g["id"] for g in grid] == sorted(g["id"] for g in grid)


def test_no_running_slots_means_every_suggested_bay_is_added_and_nothing_is_protected():
    grid, protected = protect_baseline([slot(1, box(0.1, 0.1)), slot(2, box(0.3, 0.1))], [])
    assert protected == [] and [g["id"] for g in grid] == [1, 2]


# ---- through the API on a Fixed camera -----------------------------------------------------------------------------------------------------
def draw_fixed(c, cid, sid):
    """The camera is Fixed with 5 drawn slots INSIDE the suggestion's tiles (a smaller slot each: IoU ~0.3, the case that used to be 'removed') and a
    capacity pin P1 on the last tile. -> (the drawn slots, the suggested tiles)."""
    first = suggest(c, cid, sid)["result"]["version"]
    tiles = c.get(f"/api/cameras/{cid}/suggest/{first}", headers=AUTH).json()["grid"]
    assert c.delete(f"/api/cameras/{cid}/suggest/{first}", headers=AUTH).status_code == 200
    assert len(tiles) >= 8
    drawn = [{"id": i + 1, "name": f"B{i + 1}", "poly": shrunk(t["poly"])} for i, t in enumerate(tiles[:5])]
    drawn.append({"id": 9, "name": "P1", "poly": tiles[-1]["poly"], "capacity": 4})
    assert c.put("/api/settings", headers=AUTH, json={"slot_mode": "fixed"}).status_code == 200
    assert c.put("/api/fixed-slots", headers=AUTH, json={"slots": drawn}).status_code == 200
    return drawn, tiles


def test_a_fixed_cameras_draft_never_marks_a_running_slot_removed_or_moved_and_lays_nothing_over_it(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        drawn, tiles = draw_fixed(c, cid, sid)
        before = (*worker_sees(env, cid), c.get("/api/fixed-slots", headers=AUTH).json())
        res = suggest(c, cid, sid)["result"]
        assert res["diff"]["removed"] == [] and res["diff"]["moved"] == []
        view = c.get(f"/api/cameras/{cid}/suggest/{res['version']}", headers=AUTH).json()
        assert view["meta"]["protected"] == [1, 2, 3, 4, 5, 9] and view["meta"]["kind"] == "fixed"
        grid = {s["id"]: s for s in view["grid"]}
        for d in drawn:  # carried in exactly as drawn
            assert grid[d["id"]]["poly"] == d["poly"] and grid[d["id"]]["name"] == d["name"] and grid[d["id"]]["protected"] is True
        assert grid[9]["capacity"] == 4
        assert view["diff"]["removed"] == [] and view["diff"]["moved"] == [] and set(view["diff"]["unchanged"]) == {1, 2, 3, 4, 5, 9}
        new = [s for s in view["grid"] if not s.get("protected")]
        assert new and all(s["id"] > 9 for s in new) and sorted(view["diff"]["added"]) == sorted(s["id"] for s in new)
        for n in new:  # no green bay over a running one
            assert all(overlap_share(n["poly"], d["poly"]) <= 0.1 for d in drawn)
        assert (*worker_sees(env, cid), c.get("/api/fixed-slots", headers=AUTH).json()) == before  # a draft only


def test_a_protected_slot_cannot_be_moved_or_removed_in_the_draft_but_a_new_bay_can_be_fine_tuned(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        drawn, tiles = draw_fixed(c, cid, sid)
        res = suggest(c, cid, sid)["result"]
        url = f"/api/cameras/{cid}/suggest/{res['version']}"
        grid = c.get(url, headers=AUTH).json()["grid"]
        moved = [{**s, "poly": [[x + 0.01, y] for x, y in s["poly"]]} if s["id"] == 1 else s for s in grid]
        r = c.put(url, headers=AUTH, json={"grid": moved})
        assert r.status_code == 422 and "running slot" in r.json()["detail"] and "moved or reshaped" in r.json()["detail"]
        r = c.put(url, headers=AUTH, json={"grid": [s for s in grid if s["id"] != 9]})
        assert r.status_code == 422 and "P1" in r.json()["detail"] and "removed" in r.json()["detail"]
        new = [s for s in grid if not s.get("protected")]
        tuned = [{**s, "poly": [[x + 0.002, y] for x, y in s["poly"]]} if s["id"] == new[0]["id"] else s for s in grid]
        assert c.put(url, headers=AUTH, json={"grid": tuned}).status_code == 200
        after = {s["id"]: s for s in c.get(url, headers=AUTH).json()["grid"]}
        assert after[new[0]["id"]]["source"] == "edited"
        assert all(after[d["id"]]["poly"] == d["poly"] and after[d["id"]]["protected"] for d in drawn)  # untouched, still protected
        assert after[9]["capacity"] == 4


def test_accepting_a_fixed_draft_adds_the_new_bays_and_leaves_every_running_slot_as_drawn(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        drawn, tiles = draw_fixed(c, cid, sid)
        res = suggest(c, cid, sid)["result"]
        view = c.get(f"/api/cameras/{cid}/suggest/{res['version']}", headers=AUTH).json()
        new_ids = [s["id"] for s in view["grid"] if not s.get("protected")][:3]
        # even if the client ticks the running slots too, they come out as drawn
        r = c.post(f"/api/cameras/{cid}/suggest/{res['version']}/accept", headers=AUTH, json={"select": [1, 2, 9, *new_ids]})
        assert r.status_code == 200, r.text
        now = {s["id"]: s for s in c.get("/api/fixed-slots", headers=AUTH).json()["slots"]}
        for d in drawn:
            assert now[d["id"]]["poly"] == d["poly"] and now[d["id"]]["name"] == d["name"]
        assert now[9]["capacity"] == 4 and set(new_ids) <= set(now) and len(now) == len(drawn) + 3


def test_an_auto_grid_cameras_draft_keeps_its_running_bays_too_and_its_pin_out_of_the_way(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        res = suggest(c, cid, sid)["result"]
        assert res["diff"]["removed"] == [] and res["diff"]["moved"] == []
        view = c.get(f"/api/cameras/{cid}/suggest/{res['version']}", headers=AUTH).json()
        assert view["meta"]["kind"] == "auto"
        running = [s for s in view["baseline"]["slots"] if s.get("source") != "manual"]
        grid = {s["id"]: s for s in view["grid"]}
        assert sorted(view["meta"]["protected"]) == sorted(s["id"] for s in running)
        for r in running:  # as they run: same id, shape and source
            assert grid[r["id"]]["poly"] == r["poly"] and grid[r["id"]]["protected"] is True and grid[r["id"]]["source"] == r["source"]
        assert set(view["diff"]["unchanged"]) == {s["id"] for s in running}


# ---- camera-1 as it really ran (tests/fixtures/camera1_auto_grid.json) ---------------------------------------------------------------------------------
def camera1():
    import json
    from carparking.geometry_live import ActiveGeometry
    from carparking.slot_suggest import suggest_slots
    fx = json.loads((ROOT / "tests" / "fixtures" / "camera1_auto_grid.json").read_text(encoding="utf-8"))
    ag = ActiveGeometry({"ver": 10, "record": fx["record"]})
    w, h = fx["shape"]
    slots, _ = suggest_slots(ag, fx["dets"], w, h, fx["expected"])
    return fx["running"], slots


def test_on_camera_1s_real_grid_the_old_matching_removed_and_moved_slots_and_the_protected_draft_does_not():
    running, slots = camera1()
    assert [r["name"] for r in running] == ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "P1"] and running[-1]["capacity"] == 4
    bays = [r for r in running if r["source"] != "manual"]
    old = diff(carry_ids(slots, bays, [9]), running)
    assert old["removed"] and old["moved"]  # what the operator saw: valid slots red, others shifted
    grid, protected = protect_baseline(slots, bays, avoid=[r for r in running if r["source"] == "manual"])
    d = diff(grid, running)
    assert protected == [1, 2, 3, 4, 5, 6, 7, 8]
    assert d["removed"] == [] and d["moved"] == [] and d["unchanged"] == [1, 2, 3, 4, 5, 6, 7, 8]
    by = {g["id"]: g for g in grid}
    for r in bays:  # every running bay exactly as it runs
        assert by[r["id"]]["poly"] == r["poly"] and by[r["id"]]["name"] == r["name"] and by[r["id"]]["source"] == r["source"]
    pin = next(r for r in running if r["name"] == "P1")
    for g in grid:  # nothing new on a running bay or on the pin
        if not g.get("protected"):
            assert g["id"] > 9 and all(overlap_share(g["poly"], r["poly"]) <= 0.1 for r in running)
    assert pin["id"] not in by  # the pin stays a pin (the worker merges it on top), it is not turned into a grid bay
