"""Re-anchoring through the API (POST /api/cameras/{id}/setup/reanchor): calibrate on a textured still, the camera is bumped, a new
still is matched to the calibration's stored reference, a linked DRAFT results and the Admin accepts it. Mock camera and setup
service, real feature matching; nothing touches ./data."""
import cv2
import numpy as np
import pytest

from carparking.geometry import drift
from carparking.live import LiveStore
from test_geometry_drift import scene
from test_geometry_reanchor import truth, turned_picture
from test_geometry_live import CAM
from test_setup_api import AUTH, VIEW, VIEW_AUTH, Net, cam_id, calibrate, make, wait

@pytest.fixture
def env(tmp_path):
    return {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
            "CARPARK_RTSP_URL": "rtsp://admin:secret@cam.local:554/Streaming/Channels/101", "CARPARK_API_TOKEN": "t0k3n",
            "CARPARK_SPA_DIR": str(tmp_path / "dist"), "CARPARK_GEOMETRY_ENABLED": "1", "CARPARK_SETUP_URL": "http://setup:8100",
            "CARPARK_GO2RTC_INTERNAL": "http://g"}


SCENE = scene()
JPG = cv2.imencode(".jpg", SCENE, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def moved_jpeg(dp=2.0, dr=-1.0, yaw=4.0):
    _, _, R = truth(CAM, dp, dr, yaw)
    return cv2.imencode(".jpg", turned_picture(SCENE, R), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def still(c, cid):
    return c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]


def reanchor(c, cid, sid, **extra):
    r = c.post(f"/api/cameras/{cid}/setup/reanchor", headers=AUTH, json={"still": sid, **extra})
    assert r.status_code == 200, r.text
    return wait(c, r.json()["id"], timeout=60)


def calibrated(c, net):
    """An accepted calibration made on the textured picture -> (camera id, its version)."""
    cid = cam_id(c)
    ver = calibrate(c, cid, still(c, cid))["version"]
    assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": ver}).status_code == 200
    return cid, ver


def test_a_bumped_camera_is_reanchored_to_a_linked_draft_then_accepted(env, monkeypatch):
    net = Net(jpeg=JPG)
    with make({**env, "CARPARK_VIEW_TOKEN": VIEW}, net, monkeypatch)[0] as c:
        cid, v1 = calibrated(c, net)
        store = LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid)
        assert c.get(f"/api/cameras/{cid}/setup/geometry/{v1}", headers=AUTH).json()["record"]["reference"]["n"] >= drift.MIN_REFERENCE
        store.mark_stale("camera moved")  # what the drift guard does
        assert store.active_version() is None

        net.jpeg = moved_jpeg()
        calls = len(net.setup_calls)
        job = reanchor(c, cid, still(c, cid), note="after the storm")
        assert job["state"] == "done", job
        res = job["result"]
        assert res["version"] == v1 + 1 and res["parent"] == v1 and res["parent_state"] == "stale" and res["state"] == "draft"
        assert res["confidence"] == "HIGH" and res["drift_guard"]
        assert res["reanchor"]["inliers"] > 100 and res["reanchor"]["rms_px"] < 2 and res["reanchor"]["yaw_deg"] == pytest.approx(4.0, abs=0.3)
        assert len(net.setup_calls) == calls  # no SAM, no GeoCalib, no depth: the setup service was not asked
        assert store.active_version() is None  # a draft: nothing is used yet

        # the draft links to its parent in the summary, the full record and the events
        listing = c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["versions"]
        draft = next(v for v in listing if v["ver"] == res["version"])
        assert draft["state"] == "draft" and draft["reanchor"]["parent"] == v1 and draft["note"] == "after the storm"
        assert next(v for v in listing if v["ver"] == v1)["reanchor"] is None
        events = c.get(f"/api/cameras/{cid}/geometry/events", headers=VIEW_AUTH).json()
        kinds = [(e["kind"], e["ver"]) for e in events["events"]]
        assert ("reanchored", res["version"]) in kinds
        assert next(e for e in events["events"] if e["kind"] == "reanchored")["detail"]["parent"] == v1

        # Accept: active at once; the stale parent stays in the history as stale
        r = c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": res["version"]})
        assert r.status_code == 200 and r.json()["active"]["ver"] == res["version"]
        assert store.active_version() == res["version"]
        states = {v["ver"]: v["state"] for v in c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["versions"]}
        assert states == {v1: "stale", res["version"]: "active"}
        act = [e for e in c.get(f"/api/cameras/{cid}/geometry/events", headers=VIEW_AUTH).json()["events"] if e["kind"] == "activated"]
        assert act[0]["ver"] == res["version"] and act[0]["detail"]["parent"] == v1

        # the new reference is the moved picture: the guard now says ok about it
        rec = store.get(res["version"])["record"]
        assert drift.compare(rec["reference"], cv2.imdecode(np.frombuffer(net.jpeg, np.uint8), 1))["status"] == "ok"


def test_the_parent_can_be_chosen_and_bad_choices_are_refused(env, monkeypatch):
    net = Net(jpeg=JPG)
    with make(env, net, monkeypatch)[0] as c:
        cid, v1 = calibrated(c, net)
        net.jpeg = moved_jpeg()
        sid = still(c, cid)
        job = reanchor(c, cid, sid, from_version=v1)  # the active one is a fine parent too
        assert job["state"] == "done" and job["result"]["parent"] == v1 and job["result"]["parent_state"] == "active"
        draft = job["result"]["version"]
        bad = reanchor(c, cid, sid, from_version=draft)  # a draft has no calibration history to carry over
        assert bad["state"] == "error" and "cannot be re-anchored" in bad["error"]
        gone = reanchor(c, cid, sid, from_version=99)
        assert gone["state"] == "error" and "no geometry version 99" in gone["error"]


def test_it_refuses_what_is_not_a_clean_turn_and_leaves_no_draft(env, monkeypatch):
    net = Net(jpeg=JPG)
    with make(env, net, monkeypatch)[0] as c:
        cid, v1 = calibrated(c, net)
        before = len(c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["versions"])
        net.jpeg = cv2.imencode(".jpg", scene(seed=7))[1].tobytes()  # another place altogether
        job = reanchor(c, cid, still(c, cid))
        assert job["state"] == "error" and "full setup" in job["error"]
        assert len(c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["versions"]) == before


def test_a_version_without_a_reference_cannot_be_reanchored(env, monkeypatch):
    plain = cv2.imencode(".jpg", np.zeros((1080, 1920, 3), np.uint8))[1].tobytes()  # a black still: no features, no reference
    net = Net(jpeg=plain)
    with make(env, net, monkeypatch)[0] as c:
        cid, v1 = calibrated(c, net)
        net.jpeg = JPG
        job = reanchor(c, cid, still(c, cid))
        assert job["state"] == "error" and "none has a stored reference" in job["error"]
        job = reanchor(c, cid, still(c, cid), from_version=v1)
        assert job["state"] == "error" and "cannot be re-anchored" in job["error"]


def test_the_endpoint_is_admin_only_flagged_and_validated(env, monkeypatch):
    net = Net(jpeg=JPG)
    with make({**env, "CARPARK_VIEW_TOKEN": VIEW}, net, monkeypatch)[0] as c:
        cid = cam_id(c)
        sid = still(c, cid)
        url = f"/api/cameras/{cid}/setup/reanchor"
        assert c.post(url, json={"still": sid}).status_code == 401
        assert c.post(url, headers=VIEW_AUTH, json={"still": sid}).status_code in (401, 403)
        assert c.post(url, headers=AUTH, json={"still": "nope"}).status_code == 422
        assert c.post(url, headers=AUTH, json={"still": sid, "from_version": 0}).status_code == 422
        assert c.post(url, headers=AUTH, json={"still": "0" * 32}).status_code == 410  # unknown / expired still (as for segment and calibrate)
        assert c.post("/api/cameras/nope/setup/reanchor", headers=AUTH, json={"still": sid}).status_code == 404
    with make({k: v for k, v in env.items() if k != "CARPARK_GEOMETRY_ENABLED"}, net, monkeypatch)[0] as c:
        assert c.post(f"/api/cameras/{cam_id(c)}/setup/reanchor", headers=AUTH, json={"still": "0" * 32}).status_code == 503


def test_a_second_reanchoring_while_one_runs_is_a_conflict(env, monkeypatch):
    net = Net(jpeg=JPG)
    with make(env, net, monkeypatch)[0] as c:
        cid, _ = calibrated(c, net)
        net.jpeg = moved_jpeg()
        sid = still(c, cid)
        first = c.post(f"/api/cameras/{cid}/setup/reanchor", headers=AUTH, json={"still": sid})
        second = c.post(f"/api/cameras/{cid}/setup/reanchor", headers=AUTH, json={"still": sid})
        assert first.status_code == 200
        assert second.status_code in (200, 409)  # 409 while the first still runs (it may already be done on a fast machine)
        wait(c, first.json()["id"], timeout=60)


def test_accepting_refuses_when_the_camera_moved_again_after_the_reanchoring(env, monkeypatch):
    net = Net(jpeg=JPG)
    with make(env, net, monkeypatch)[0] as c:
        cid, v1 = calibrated(c, net)
        net.jpeg = moved_jpeg(2.0, -1.0, 4.0)
        res = reanchor(c, cid, still(c, cid))["result"]
        # the worker's latest frame shows ANOTHER bump (not what the draft was made on)
        again = cv2.imencode(".jpg", turned_picture(SCENE, truth(CAM, -3.0, 1.0, -9.0)[2]))[1].tobytes()
        import sqlite3
        import time
        con = sqlite3.connect(env["CARPARK_HISTORY_DB"])
        con.execute("INSERT OR IGNORE INTO live_state (id, owner, heartbeat) VALUES (1, 'w', ?)", (time.time(),))
        con.execute("UPDATE live_state SET raw = ?, raw_ts = ? WHERE id = 1", (again, time.time()))
        con.commit()
        con.close()
        r = c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": res["version"]})
        assert r.status_code == 409 and "moved again" in r.json()["detail"]
        # a worker frame that agrees with the draft lets it through
        con = sqlite3.connect(env["CARPARK_HISTORY_DB"])
        con.execute("UPDATE live_state SET raw = ?, raw_ts = ? WHERE id = 1", (net.jpeg, time.time()))
        con.commit()
        con.close()
        assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": res["version"]}).status_code == 200
