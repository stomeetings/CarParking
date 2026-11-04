"""Phase 2 history on the API: the drift guard's reference stored with each new calibration, the occupancy history and the
lifecycle events endpoints (viewer token, behind CARPARK_GEOMETRY_ENABLED). Mock setup service and camera, no models."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from carparking.geometry import drift
from carparking.live import LiveStore
from test_geometry_drift import scene
from test_setup_api import AUTH, POLYGON, URL, Net, cam_id, make, wait

TOKEN = "t0k3n"
VIEW = "v1ew"
VIEW_AUTH = {"Authorization": f"Bearer {VIEW}"}


@pytest.fixture
def env(tmp_path):
    return {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
            "CARPARK_RTSP_URL": URL, "CARPARK_API_TOKEN": TOKEN, "CARPARK_SPA_DIR": str(tmp_path / "dist"),
            "CARPARK_GEOMETRY_ENABLED": "1", "CARPARK_SETUP_URL": "http://setup:8100", "CARPARK_GO2RTC_INTERNAL": "http://g",
            "CARPARK_VIEW_TOKEN": VIEW}


TEXTURED = cv2.imencode(".jpg", scene())[1].tobytes()


def calibrate(c, cid, sid):
    job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH, json={"still": sid, "polygon": POLYGON}).json()["id"])
    assert job["state"] == "done", job
    return job["result"]


def test_each_new_calibration_keeps_the_stills_background_as_the_drift_reference(env, monkeypatch):
    with make(env, Net(jpeg=TEXTURED), monkeypatch)[0] as c:
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        res = calibrate(c, cid, sid)
        assert res["drift_guard"] is True and not any("drift guard" in w for w in res["warnings"])
        rec = c.get(f"/api/cameras/{cid}/setup/geometry/{res['version']}", headers=AUTH).json()
        assert rec["drift_guard"] is True and rec["record"]["reference"]["n"] >= drift.MIN_REFERENCE
        assert rec["record"]["reference"]["size"] == [1920, 1080]
        assert c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["versions"][0]["drift_guard"] is True
        json.dumps(rec["record"]["reference"])  # it lives in the record: plain JSON
    with make(env, Net(), monkeypatch)[0] as c:  # a still with nothing to hold on to (the mock's black picture)
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        res = calibrate(c, cid, sid)
        assert res["drift_guard"] is False and any("too few stable features for the drift guard" in w for w in res["warnings"])


def seed(env, cam, samples=0, t0=None):
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    g = store.geometry(cam)
    t0 = int(time.time()) - 3 * 3600 if t0 is None else t0
    for i in range(samples):  # a sample every 5 minutes
        g.add_sample(1 + i // 24, {"capacity": 4, "occupied": i % 5, "vehicles": i % 5 + 1, "measured": i % 5}, now=t0 + i * 300)
    return g


def test_history_is_bucketed_in_the_database_and_behind_the_flag_and_the_viewer_token(env, monkeypatch):
    net = Net()
    with make(env, net, monkeypatch)[0] as c:
        cid = cam_id(c)
        seed(env, cid, samples=36)  # three hours
        seed(env, "someone-else", samples=10)  # another camera: never mixed in
        h = c.get(f"/api/cameras/{cid}/geometry/history?hours=4&points=48", headers=VIEW_AUTH).json()
        assert h["bucket_s"] == 300 and h["until"] > h["since"] and h["until"] - h["since"] == pytest.approx(4 * 3600, abs=5)
        assert sum(p["samples"] for p in h["points"]) == 36 and h["versions"] == [1, 2]
        assert all({"t", "samples", "occupied_avg", "occupied_max", "occupied_min", "capacity_avg", "capacity_max", "vehicles_avg", "ver"} <= set(p)
                   for p in h["points"])
        assert [p["t"] for p in h["points"]] == sorted(p["t"] for p in h["points"])
        coarse = c.get(f"/api/cameras/{cid}/geometry/history?hours=4&points=10", headers=AUTH).json()  # the admin token reads too
        assert coarse["bucket_s"] == 1440 and 1 <= len(coarse["points"]) <= 10
        assert sum(p["samples"] for p in coarse["points"]) == 36 and max(p["occupied_max"] for p in coarse["points"]) == 4
        assert c.get(f"/api/cameras/{cid}/geometry/history", headers=VIEW_AUTH).json()["bucket_s"] == 360  # 24 h in 240 points
        assert c.get(f"/api/cameras/{cid}/geometry/history?hours=0.01&points=1000", headers=VIEW_AUTH).json()["bucket_s"] == 60  # never finer than a minute
        for bad in ("hours=0", "hours=3000", "points=5", "points=5000"):
            assert c.get(f"/api/cameras/{cid}/geometry/history?{bad}", headers=VIEW_AUTH).status_code == 422
        assert c.get(f"/api/cameras/{cid}/geometry/history").status_code == 401  # a viewer token is needed
        assert c.get("/api/cameras/nope/geometry/history", headers=VIEW_AUTH).status_code == 404
    with make({**env, "CARPARK_GEOMETRY_ENABLED": "0"}, net, monkeypatch)[0] as off:  # the flag is off: no history, no events
        cid = cam_id(off)
        for path in ("history", "events"):
            r = off.get(f"/api/cameras/{cid}/geometry/{path}", headers=VIEW_AUTH)
            assert r.status_code == 503 and "CARPARK_GEOMETRY_ENABLED" in r.json()["detail"]
    fresh = {**env, "CARPARK_HISTORY_DB": str(Path(env["CARPARK_HISTORY_DB"]).with_name("fresh.db"))}
    with make(fresh, net, monkeypatch)[0] as empty:  # nothing sampled yet
        assert empty.get(f"/api/cameras/{cam_id(empty)}/geometry/history?hours=1", headers=VIEW_AUTH).json()["points"] == []


def test_the_events_endpoint_tells_the_calibrations_story(env, monkeypatch):
    with make(env, Net(), monkeypatch)[0] as c:
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        v1, v2 = calibrate(c, cid, sid)["version"], calibrate(c, cid, sid)["version"]
        c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": v1, "accept_unverified": True})
        c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": v2, "accept_unverified": True})
        c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": v1, "accept_unverified": True})  # a rollback
        v3 = calibrate(c, cid, sid)["version"]
        c.delete(f"/api/cameras/{cid}/setup/draft?version={v3}", headers=AUTH)
        LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid).mark_stale("drift: the background moved 31 px")  # what the worker's guard does
        ev = c.get(f"/api/cameras/{cid}/geometry/events", headers=VIEW_AUTH).json()["events"]
        assert [(e["kind"], e["ver"]) for e in ev] == [("stale", 1), ("discarded", 3), ("draft", 3), ("activated", 1), ("activated", 2),
                                                          ("activated", 1), ("draft", 2), ("draft", 1)]  # newest first
        assert ev[0]["detail"] == {"reason": "drift: the background moved 31 px"}
        assert [e["detail"] for e in ev if e["kind"] == "activated"] == [{"previous": 2, "rollback": True}, {"previous": 1, "rollback": False},
                                                                          {"previous": None, "rollback": False}]
        assert all(set(e) == {"id", "t", "kind", "ver", "detail"} and e["t"] > 0 for e in ev)
        assert [e["kind"] for e in c.get(f"/api/cameras/{cid}/geometry/events?limit=2", headers=VIEW_AUTH).json()["events"]] == ["stale", "discarded"]
        cut = ev[2]["t"]
        assert all(e["t"] > cut for e in c.get(f"/api/cameras/{cid}/geometry/events?since={cut}", headers=VIEW_AUTH).json()["events"])
        for bad in ("limit=0", "limit=900"):
            assert c.get(f"/api/cameras/{cid}/geometry/events?{bad}", headers=VIEW_AUTH).status_code == 422
        assert c.get(f"/api/cameras/{cid}/geometry/events").status_code == 401
        assert np.sum([e["kind"] == "stale" for e in ev]) == 1
