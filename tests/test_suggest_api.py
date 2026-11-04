"""Slot suggestions through the API (api/geometry_routes.py, carparking/slot_suggest.py): a quick and a full suggestion are DRAFTS next to
the running slots and change nothing the worker reads; fine-tune, discard, and Accept on an auto-grid camera (the draft becomes the
active version, pins keep their ids) and on a Fixed camera (chosen bays merged into the drawn map). Mock network, no models."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from test_setup_api import AUTH, URL, VIEW, VIEW_AUTH, Net, calibrate, car_fractions, cam_id, make, wait  # noqa: E402

from carparking.geometry_live import ActiveGeometry, GeometryFeed  # noqa: E402
from carparking.live import LiveStore  # noqa: E402



@pytest.fixture
def env(tmp_path):
    return {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
            "CARPARK_RTSP_URL": URL, "CARPARK_API_TOKEN": "t0k3n", "CARPARK_SPA_DIR": str(tmp_path / "dist"),
            "CARPARK_GEOMETRY_ENABLED": "1", "CARPARK_SETUP_URL": "http://setup:8100", "CARPARK_GO2RTC_INTERNAL": "http://g"}


CARS = [car_fractions(-3, 11), car_fractions(1, 13), car_fractions(-1, 17)]


def ready(c, cid):
    """A HIGH calibration (version 1, typed reference) accepted and active -> a snapshot id."""
    sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
    calibrate(c, cid, sid)
    assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 1}).status_code == 200
    return sid


def suggest(c, cid, sid=None, **body):
    r = c.post(f"/api/cameras/{cid}/suggest", headers=AUTH, json={**({"still": sid} if sid else {}), **body})
    assert r.status_code == 200, r.text
    return wait(c, r.json()["id"], timeout=30)


def worker_sees(env, cid):
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    return store.config(), store.geometry(cid).active_version()


def test_suggestions_need_the_flag_the_admin_token_and_a_calibration(env, monkeypatch):
    with make({**env, "CARPARK_GEOMETRY_ENABLED": "0"}, Net(), monkeypatch)[0] as c:
        cid = cam_id(c)
        assert c.post(f"/api/cameras/{cid}/suggest", headers=AUTH, json={}).status_code == 503
        assert c.get(f"/api/cameras/{cid}/suggest", headers=AUTH).status_code == 503
    with make({**env, "CARPARK_VIEW_TOKEN": VIEW}, Net(), monkeypatch)[0] as c:
        cid = cam_id(c)
        for headers in ({}, VIEW_AUTH):
            assert c.post(f"/api/cameras/{cid}/suggest", headers=headers, json={}).status_code in (401, 403)
            assert c.get(f"/api/cameras/{cid}/suggest/1", headers=headers).status_code in (401, 403)
        job = suggest(c, cid)  # uncalibrated
        assert job["state"] == "error" and "no active calibration" in job["error"]
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        calibrate(c, cid, sid, reference=False)
        assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 1, "accept_unverified": True}).status_code == 200
        job = suggest(c, cid, sid)
        assert job["state"] == "error" and "HIGH" in job["error"]  # the metres of a prior-only calibration are not good enough
        assert c.get(f"/api/cameras/{cid}/suggest/99", headers=AUTH).status_code == 404
        assert c.post(f"/api/cameras/{cid}/suggest", headers=AUTH, json={"still": "a" * 32}).status_code == 410


def test_a_quick_suggestion_is_a_draft_and_changes_nothing_the_worker_reads(env, monkeypatch):
    net = Net()
    c, detected = make(env, net, monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        before = (*worker_sees(env, cid), c.get("/api/fixed-slots", headers=AUTH).json(), c.get("/api/auto-pins", headers=AUTH).json())
        calls = len(net.setup_calls)
        job = suggest(c, cid, sid, note="try")
        assert job["state"] == "done", job
        res = job["result"]
        assert res["mode"] == "quick" and res["bays"] >= 6 and res["lined_up"] == "vehicles" and res["version"] == 2
        assert len(net.setup_calls) == calls  # quick needs no setup service
        assert (*worker_sees(env, cid), c.get("/api/fixed-slots", headers=AUTH).json(), c.get("/api/auto-pins", headers=AUTH).json()) == before
        assert LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid).get(2)["state"] == "draft"

        view = c.get(f"/api/cameras/{cid}/suggest/2", headers=AUTH).json()
        ids = [s["id"] for s in view["grid"]]
        assert len(ids) == len(set(ids)) == res["bays"] and view["state"] == "draft" and view["stale"] is False
        assert view["baseline"]["kind"] == "auto" and view["calibration"] == {"confidence": "HIGH", "h": pytest.approx(6.0, rel=0.1), "version": 1}
        assert set(view["diff"]) == {"added", "removed", "moved", "unchanged"} and view["pins"] == []
        assert all(set(s) >= {"id", "name", "poly", "lift", "source"} for s in view["grid"])
        assert view["meta"]["parent"] == 1 and view["meta"]["kind"] == "auto" and view["meta"]["mode"] == "quick"

        listed = c.get(f"/api/cameras/{cid}/suggest", headers=AUTH).json()["suggestions"]
        assert [(s["version"], s["bays"], s["note"]) for s in listed] == [(2, res["bays"], "try")]
        versions = c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["versions"]
        assert versions[0]["ver"] == 2 and versions[0]["suggest"]["parent"] == 1 and versions[1]["suggest"] is None
        events = LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid).events()
        assert any(e["kind"] == "draft" and e["detail"].get("suggest") == "quick" for e in events)


def test_fine_tuning_keeps_the_draft_a_draft_and_the_running_bays_exactly_as_they_run(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        suggest(c, cid, sid)
        grid = c.get(f"/api/cameras/{cid}/suggest/2", headers=AUTH).json()["grid"]
        protected = [s for s in grid if s.get("protected")]
        assert protected  # the bays that run now are in the draft as they run
        moved = [{**s, "poly": [[x + 0.01, y] for x, y in s["poly"]]} if s["id"] == protected[0]["id"] else s for s in grid]
        r = c.put(f"/api/cameras/{cid}/suggest/2", headers=AUTH, json={"grid": moved})  # a running bay cannot be moved ...
        assert r.status_code == 422 and "running slot" in r.json()["detail"]
        r = c.put(f"/api/cameras/{cid}/suggest/2", headers=AUTH, json={"grid": [s for s in grid if s["id"] != protected[0]["id"]]})  # ... or removed
        assert r.status_code == 422 and "removed" in r.json()["detail"]
        r = c.put(f"/api/cameras/{cid}/suggest/2", headers=AUTH, json={"grid": grid})  # the draft as it is saves fine
        assert r.status_code == 200, r.text
        after = c.get(f"/api/cameras/{cid}/suggest/2", headers=AUTH).json()
        assert after["grid"] == grid and after["meta"]["edited"] is True and after["diff"]["removed"] == [] and after["diff"]["moved"] == []
        assert c.put(f"/api/cameras/{cid}/suggest/2", headers=AUTH, json={"grid": [{"id": 1, "poly": [[0, 0], [1, 1]]}]}).status_code == 422
        assert c.put(f"/api/cameras/{cid}/suggest/1", headers=AUTH, json={"grid": moved}).status_code == 404  # not a suggestion
        assert worker_sees(env, cid)[1] == 1


def test_discard_removes_only_the_suggestion(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        suggest(c, cid, sid)
        assert c.delete(f"/api/cameras/{cid}/suggest/1", headers=AUTH).status_code == 404  # the active calibration is not one
        assert c.delete(f"/api/cameras/{cid}/suggest/2", headers=AUTH).json() == {"discarded": [2]}
        assert c.get(f"/api/cameras/{cid}/suggest", headers=AUTH).json() == {"suggestions": []}
        assert worker_sees(env, cid)[1] == 1


def test_accepting_on_an_auto_grid_activates_the_draft_and_the_pin_keeps_its_id(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        suggest(c, cid, sid)
        first = c.get(f"/api/cameras/{cid}/suggest/2", headers=AUTH).json()
        target = first["grid"][len(first["grid"]) // 2]
        pin = {"id": 1, "name": "P1", "poly": [[min(x + 0.012, 1.0), y] for x, y in target["poly"]], "capacity": 4}
        assert c.put("/api/auto-pins", headers=AUTH, json={"expected": None, "pins": [pin]}).status_code == 200
        assert c.get(f"/api/cameras/{cid}/suggest/2", headers=AUTH).json()["stale"] is True  # the running slots changed meanwhile
        r = c.post(f"/api/cameras/{cid}/suggest/2/accept", headers=AUTH, json={})
        assert r.status_code == 409 and "changed" in r.json()["detail"]
        assert c.delete(f"/api/cameras/{cid}/suggest/2", headers=AUTH).status_code == 200

        v = suggest(c, cid, sid)["result"]["version"]
        view = c.get(f"/api/cameras/{cid}/suggest/{v}", headers=AUTH).json()
        assert view["stale"] is False and [p["name"] for p in view["pins"]] == ["P1"] and view["pins"][0]["capacity"] == 4
        pin_id = view["pins"][0]["id"]
        assert pin_id == target["id"] and pin_id not in [s["id"] for s in view["grid"]]  # the pin's id is kept free in the draft
        assert view["pins"][0]["binds_to"] in [s["id"] for s in view["grid"]]  # bound to the nearest bay of the grid

        r = c.post(f"/api/cameras/{cid}/suggest/{v}/accept", headers=AUTH, json={})
        assert r.status_code == 200, r.text
        assert r.json()["applied"] == "auto" and r.json()["previous"] == 1 and r.json()["active"]["ver"] == v
        store = LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid)
        active = store.active()
        assert active["ver"] == v and active["record"]["grid"] == view["grid"] and store.get(1)["state"] == "superseded"
        saved = c.get("/api/auto-pins", headers=AUTH).json()["pins"]
        assert saved[0]["slot_id"] == pin_id and saved[0]["capacity"] == 4  # the pin has an id of its own from now on
        g = ActiveGeometry(active)
        slots = GeometryFeed._grid_slots(g, 1920, 1080, 99)
        assert [s["id"] for s in slots] == [s["id"] for s in view["grid"]]  # the worker's grid is exactly the accepted one
        assert any(e["kind"] == "activated" and e["detail"].get("suggest") == "quick" and e["detail"]["parent"] == 1 for e in store.events())
        assert c.post(f"/api/cameras/{cid}/suggest/{v}/accept", headers=AUTH, json={}).status_code in (404, 409)  # only once


def test_accept_is_refused_when_the_calibration_or_the_camera_changed(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        suggest(c, cid, sid)
        calibrate(c, cid, sid)  # version 3 (typed reference) replaces version 1 as the active calibration
        assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 3}).status_code == 200
        r = c.post(f"/api/cameras/{cid}/suggest/2/accept", headers=AUTH, json={})
        assert r.status_code == 409 and "active calibration is not version 1" in r.json()["detail"]
        assert worker_sees(env, cid)[1] == 3
        suggest(c, cid, sid)  # now a suggestion of version 3
        assert c.put("/api/settings", headers=AUTH, json={"slot_mode": "fixed"}).status_code == 200
        r = c.post(f"/api/cameras/{cid}/suggest/4/accept", headers=AUTH, json={})
        assert r.status_code == 409 and "fixed mode" in r.json()["detail"]


def test_a_fixed_camera_takes_the_chosen_bays_into_its_drawn_map(env, monkeypatch):
    c, _ = make(env, Net(), monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        assert c.put("/api/settings", headers=AUTH, json={"slot_mode": "fixed"}).status_code == 200
        drawn = [{"id": 1, "name": "S1", "poly": [[0.05, 0.05], [0.15, 0.05], [0.15, 0.15], [0.05, 0.15]], "capacity": 2},
                 {"id": 2, "name": "S2", "poly": [[0.2, 0.05], [0.3, 0.05], [0.3, 0.15], [0.2, 0.15]]}]
        assert c.put("/api/fixed-slots", headers=AUTH, json={"slots": drawn}).status_code == 200
        res = suggest(c, cid, sid)["result"]
        view = c.get(f"/api/cameras/{cid}/suggest/{res['version']}", headers=AUTH).json()
        assert view["baseline"]["kind"] == "fixed" and [s["id"] for s in view["baseline"]["slots"]] == [1, 2] and view["pins"] == []
        new_ids = [s["id"] for s in view["grid"] if s["id"] > 2]
        assert len(new_ids) >= 4
        take = new_ids[:2]
        assert c.post(f"/api/cameras/{cid}/suggest/{res['version']}/accept", headers=AUTH, json={"select": [999]}).status_code == 422
        r = c.post(f"/api/cameras/{cid}/suggest/{res['version']}/accept", headers=AUTH, json={"select": take})
        assert r.status_code == 200, r.text
        assert r.json()["applied"] == "fixed"
        now = c.get("/api/fixed-slots", headers=AUTH).json()["slots"]
        assert [s["id"] for s in now] == [1, 2, *take]
        assert now[0]["poly"] == drawn[0]["poly"] and now[0].get("capacity") == 2 and now[1]["poly"] == drawn[1]["poly"]  # as drawn
        store = LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid)
        assert store.get(res["version"]) is None and worker_sees(env, cid)[1] == 1  # the draft is consumed, the calibration untouched
        assert any(e["kind"] == "applied_to_fixed" and e["detail"]["taken"] == take for e in store.events())


def test_a_full_suggestion_remeasures_with_the_setup_service_and_keeps_the_scale(env, monkeypatch):
    net = Net(depth=9.0)  # the depth prior now disagrees wildly: it must not move the metres
    c, _ = make(env, net, monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        old_h = LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid).active()["record"]["calibration"]["h"]
        calls = len(net.setup_calls)
        job = suggest(c, cid, sid, mode="full")
        assert job["state"] == "done", job
        assert [p for p, _ in net.setup_calls[calls:]] == ["/calibrate"]  # the stored surface: no new SAM pass
        draft = LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid).get(job["result"]["version"])["record"]
        assert draft["calibration"]["confidence"] == "HIGH" and draft["calibration"]["h"] == pytest.approx(old_h)
        assert draft["suggest"]["mode"] == "full" and draft["suggest"]["parent"] == 1 and draft["grid"]
        job = suggest(c, cid, sid, mode="full", clicks=[{"x": 0.5, "y": 0.6, "label": 1}])
        assert job["state"] == "done" and [p for p, _ in net.setup_calls[-2:]] == ["/segment", "/calibrate"]
        assert worker_sees(env, cid)[1] == 1  # still only drafts

        # an active calibration without a typed distance cannot lend its metres
        sid2 = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        loose = calibrate(c, cid, sid2, reference=False)["version"]
        assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": loose, "accept_unverified": True}).status_code == 200
        job = suggest(c, cid, sid2, mode="full")
        assert job["state"] == "error" and "HIGH" in job["error"]


def test_a_full_suggestion_needs_the_setup_service(env, monkeypatch):
    net = Net()
    c, _ = make(env, net, monkeypatch, dets=CARS)
    with c:
        cid = cam_id(c)
        sid = ready(c, cid)
        net.setup_up = False
        job = suggest(c, cid, sid, mode="full")
        assert job["state"] == "error" and "setup service" in job["error"]
        assert suggest(c, cid, sid)["state"] == "done"  # quick does not
