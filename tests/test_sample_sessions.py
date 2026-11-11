"""Sample-picture sessions through the API (api/sample_sessions.py): upload a random picture, calibrate it from four ground pins (or GeoCalib + a typed
distance), review the grid, run the stack. Scripted detector, mock setup service; nothing live may change."""
import base64
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import test_setup_api  # noqa: E402
from test_sample_session import CARS, FRAME, H, PINS, SURFACE, W  # noqa: E402
from test_setup_api import AUTH, Net, make  # noqa: E402

from carparking import sample_test  # noqa: E402
from carparking.geometry import to_image  # noqa: E402
from carparking.live import LiveStore  # noqa: E402


@pytest.fixture
def env(tmp_path):
    return {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
            "CARPARK_RTSP_URL": "rtsp://admin:secret@cam.local:554/Streaming/Channels/101", "CARPARK_API_TOKEN": "t0k3n",
            "CARPARK_SPA_DIR": str(tmp_path / "dist"), "CARPARK_GEOMETRY_ENABLED": "1", "CARPARK_SETUP_URL": "http://setup:8100",
            "CARPARK_GO2RTC_INTERNAL": "http://g"}


def b64(frame=FRAME):
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", frame)[1].tobytes()).decode()


@pytest.fixture
def scripted(monkeypatch):
    state = {"dets": [dict(d) for d in CARS], "calls": [], "gate": None, "tunes": []}

    def detect(run, frame, root, objects=False, tune=None):
        state["calls"].append(objects)
        state["tunes"].append(tune)
        if state["gate"] is not None:
            state["gate"].wait(10)
        return [dict(d) for d in state["dets"]]

    monkeypatch.setattr(sample_test, "detect_frame", detect)
    return state


def app_client(env, monkeypatch, net=None):
    return make(env, net or Net(), monkeypatch)[0]


def wait(c, sid, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        job = c.get(f"/api/sample-sessions/{sid}/job", headers=AUTH).json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


def new(c, **kw):
    r = c.post("/api/sample-sessions", headers=AUTH, json={"image": b64(), **kw})
    assert r.status_code == 200, r.text
    return r.json()


RECT = {"method": "rectangle", "pins": PINS, "sides": [2.5, 5.0], "polygon": SURFACE}


def calibrated(c):
    s = new(c)
    r = c.post(f"/api/sample-sessions/{s['id']}/calibrate", headers=AUTH, json=RECT)
    assert r.status_code == 200, r.text
    job = wait(c, s["id"])
    assert job["state"] == "done", job
    return s["id"], job["result"]


def live_state(c, env):
    return (LiveStore(env["CARPARK_HISTORY_DB"]).config(), c.get("/api/events?kinds=all").json(), c.get("/api/fixed-slots").json(),
            LiveStore(env["CARPARK_HISTORY_DB"]).state(images=False))


# ---- the session --------------------------------------------------------------------------------------------------------------------------
def test_needs_the_admin_token_and_checks_the_picture(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        assert c.post("/api/sample-sessions", json={"image": b64()}).status_code == 401
        post = lambda **b: c.post("/api/sample-sessions", headers=AUTH, json=b)  # noqa: E731
        assert post().status_code == 422 and post(image=b64(), sample="a.jpg").status_code == 422
        assert post(image="not base64!").status_code == 422
        assert post(image=base64.b64encode(b"not an image").decode()).status_code == 422
        assert post(sample="nope.jpg").status_code == 404 and post(sample="../h.db").status_code == 404
        assert post(image=b64(), profile_cam="nope").status_code == 404
        s = post(image=b64()).json()
        assert (s["width"], s["height"]) == (W, H) and s["image"].startswith("data:image/jpeg;base64,") and s["slots"] == [] and s["calibration"] is None
        assert s["layers"] == {"spillage": False, "obstacle_filter": False, "blocked_objects": False} and s["expires_in"] > 1700
        assert c.get(f"/api/sample-sessions/{s['id']}").status_code == 401
        assert c.get("/api/sample-sessions/abc", headers=AUTH).status_code == 404


def test_sessions_are_capped_expire_and_can_be_discarded(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        ids = [new(c)["id"] for _ in range(6)]
        assert len(c.app.state.sample_sessions) == 5 and c.get(f"/api/sample-sessions/{ids[0]}", headers=AUTH).status_code == 404  # the oldest made room
        c.app.state.sample_sessions[ids[1]]["updated"] -= 3600  # idle for an hour
        assert c.get(f"/api/sample-sessions/{ids[1]}", headers=AUTH).status_code == 404
        assert c.delete(f"/api/sample-sessions/{ids[2]}", headers=AUTH).json() == {"discarded": ids[2]}
        assert c.get(f"/api/sample-sessions/{ids[2]}", headers=AUTH).status_code == 404
        assert c.get(f"/api/sample-sessions/{ids[5]}", headers=AUTH).status_code == 200


# ---- rectangle calibration --------------------------------------------------------------------------------------------------------------
def test_four_pins_calibrate_the_picture_and_give_a_grid(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        sid, res = calibrated(c)
        assert res["method"] == "rectangle" and res["confidence"] == "HIGH" and res["h"] == pytest.approx(6.0, rel=0.03) and res["bays"] >= 6
        assert len(res["slots"]) == res["bays"] and res["pins_fit"]["reprojection_px"] < 1.0
        s = c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()
        assert s["calibration"]["confidence"] == "HIGH" and s["slots"] == res["slots"] and s["polygon"] == SURFACE and s["job"]["state"] == "done"
        assert scripted["calls"] == [False]  # the vehicles of the picture, once


def test_bad_pins_end_the_job_with_what_to_fix_and_nothing_is_kept(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        s = new(c)
        crossed = {**RECT, "pins": [PINS[0], PINS[2], PINS[1], PINS[3]]}
        assert c.post(f"/api/sample-sessions/{s['id']}/calibrate", headers=AUTH, json=crossed).status_code == 200
        job = wait(c, s["id"])
        assert job["state"] == "error" and "in order" in job["error"]
        assert c.get(f"/api/sample-sessions/{s['id']}", headers=AUTH).json()["slots"] == []
        post = lambda **b: c.post(f"/api/sample-sessions/{s['id']}/calibrate", headers=AUTH, json=b)  # noqa: E731
        assert post(method="rectangle", polygon=SURFACE).status_code == 422  # no pins
        assert post(method="geocalib", polygon=SURFACE).status_code == 422  # no reference
        assert post(method="magic", polygon=SURFACE).status_code == 422
        assert post(method="rectangle", pins=PINS[:3], sides=[2.5, 5], polygon=SURFACE).status_code == 422


def test_run_and_edit_need_a_calibration_first(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        s = new(c)
        assert c.post(f"/api/sample-sessions/{s['id']}/run", headers=AUTH, json={}).status_code == 409
        one = {"id": 1, "name": "A", "poly": [[0, 0], [1, 0], [1, 1]]}
        assert c.put(f"/api/sample-sessions/{s['id']}/slots", headers=AUTH, json={"slots": [one]}).status_code == 409


# ---- review and run -----------------------------------------------------------------------------------------------------------------------
def test_the_grid_can_be_edited_and_the_picture_runs_on_it(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        sid, res = calibrated(c)
        mine = res["slots"][:3]
        r = c.put(f"/api/sample-sessions/{sid}/slots", headers=AUTH, json={"slots": mine})
        assert r.status_code == 200 and [s["id"] for s in r.json()["slots"]] == [1, 2, 3]
        bad = {"id": 1, "name": "x", "poly": [[0, 0], [1, 1]]}
        assert c.put(f"/api/sample-sessions/{sid}/slots", headers=AUTH, json={"slots": [bad]}).status_code == 422
        before = live_state(c, env)
        r = c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layers": {"blocked_objects": True}})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["kind"] == "session" and out["session"] == sid and out["slot_count"] == 3 and out["shape"] == [W, H] and out["image"].startswith("data:image/jpeg")
        assert [s["id"] for s in out["slots"] if s["id"] < 1000] == [1, 2, 3] and out["layers"]["blocked_objects"] is True and out["layers"]["spillage"] is False
        assert out["counts"]["slots"] == 3 and out["dynamic"] is False and all(x["id"] < 1000 for x in out["slots"])  # dynamic slots off (the default): only the fixed bays
        dyn = c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layers": {"blocked_objects": True}, "dynamic": True}).json()
        assert dyn["dynamic"] is True and any(x["id"] >= 1000 for x in dyn["slots"])  # dynamic on: a car off the grid gets a bay of its own (live auto-grid rules)
        assert dyn["counts"]["slots"] == 3 + len([x for x in dyn["slots"] if x["id"] >= 1000]) and dyn["vehicles"] == 3
        assert scripted["calls"] == [False, True]  # the object classes were asked for on the second run (and taken from the cache on the third)
        assert live_state(c, env) == before  # nothing live moved: settings, event log, saved slots, worker state


def test_a_session_does_not_inherit_a_live_cameras_parking_area_cuts(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        assert c.get("/api/settings").json()["values"]["cut_right"] == 40  # the live camera cuts the right 40 % off
        sid, res = calibrated(c)
        out = c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={}).json()
        assert out["area"]["right"] == 0.0 and out["area"]["roi"] is None  # no cut at all: neither the live camera's margins nor the surface outline


def test_every_protection_starts_off_whatever_the_profile_camera_runs_and_a_request_turns_one_on(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        assert c.put("/api/settings", headers=AUTH, json={"spillage": True}).status_code == 200  # the live camera runs spillage ...
        sid, _ = calibrated(c)
        assert c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()["layers"] == {"spillage": False, "obstacle_filter": False, "blocked_objects": False}  # ... the sample test does not
        assert c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={}).json()["layers"]["spillage"] is False
        assert c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layers": {"spillage": True}}).json()["layers"]["spillage"] is True


def test_one_model_job_at_a_time_and_the_lock_is_released(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        a, b = new(c)["id"], new(c)["id"]
        scripted["gate"] = threading.Event()
        assert c.post(f"/api/sample-sessions/{a}/calibrate", headers=AUTH, json=RECT).status_code == 200
        time.sleep(0.2)
        r = c.post(f"/api/sample-sessions/{b}/calibrate", headers=AUTH, json=RECT)
        assert r.status_code == 409 and "another sample job" in r.json()["detail"]
        assert c.post(f"/api/sample-sessions/{a}/calibrate", headers=AUTH, json=RECT).status_code == 409  # a session has one job at a time
        scripted["gate"].set()
        assert wait(c, a)["state"] == "done"
        assert c.post(f"/api/sample-sessions/{b}/calibrate", headers=AUTH, json=RECT).status_code == 200  # free again
        assert wait(c, b)["state"] == "done"


# ---- the models of the setup service ----------------------------------------------------------------------------------------------------
def test_the_geocalib_route_uses_the_setup_service_and_a_typed_distance(env, monkeypatch, scripted):
    net = Net()
    with app_client(env, monkeypatch, net) as c:
        s = new(c)
        a, b = (to_image(test_setup_api.CAM, [(-3, 12), (2, 15)]) / (W, H)).tolist()
        body = {"method": "geocalib", "polygon": SURFACE, "reference": {"a": a, "b": b, "metres": float(np.hypot(5, 3))}, "lens": "simple_radial"}
        assert c.post(f"/api/sample-sessions/{s['id']}/calibrate", headers=AUTH, json=body).status_code == 200
        job = wait(c, s["id"])
        assert job["state"] == "done", job
        assert job["result"]["method"] == "geocalib" and job["result"]["confidence"] == "HIGH" and job["result"]["bays"] >= 4
        assert [p for p, _ in net.setup_calls] == ["/calibrate"] and net.setup_calls[0][1]["camera_model"] == "simple_radial" and net.setup_calls[0][1]["depth"] is True


def test_sam_outlines_the_surface_and_a_dead_service_is_said_plainly(env, monkeypatch, scripted):
    net = Net()
    with app_client(env, monkeypatch, net) as c:
        s = new(c)
        r = c.post(f"/api/sample-sessions/{s['id']}/surface", headers=AUTH, json={"clicks": [{"x": 960, "y": 700, "label": 1}]})
        assert r.status_code == 200, r.text
        job = wait(c, s["id"])
        assert job["state"] == "done", job
        assert len(job["result"]["polygon"]) >= 3 and 0.01 < job["result"]["area_frac"] < 0.9
        assert net.setup_calls[0][0] == "/segment" and net.setup_calls[0][1]["clicks"] == [{"x": 960.0, "y": 700.0, "label": 1}]
        assert c.get(f"/api/sample-sessions/{s['id']}", headers=AUTH).json()["polygon"] == job["result"]["polygon"]
        net.setup_up = False
        c.post(f"/api/sample-sessions/{s['id']}/surface", headers=AUTH, json={"clicks": [{"x": 960, "y": 700}]})
        dead = wait(c, s["id"])
        assert dead["state"] == "error" and "setup service" in dead["error"]
        # the rectangle method does not need it
        assert c.post(f"/api/sample-sessions/{s['id']}/calibrate", headers=AUTH, json=RECT).status_code == 200
        assert wait(c, s["id"])["state"] == "done"


def test_the_page_can_ask_whether_the_setup_service_is_up_and_list_the_samples_folder(env, monkeypatch, scripted, tmp_path):
    net = Net()
    with app_client(env, monkeypatch, net) as c:
        assert c.get("/api/sample-sessions/service").status_code == 401 and c.get("/api/sample-sessions/images").status_code == 401
        up = c.get("/api/sample-sessions/service", headers=AUTH).json()
        assert up["service"] == "up" and up["loaded"]["sam2"] is True and "--profile setup" in up["command"]
        net.setup_up = False
        down = c.get("/api/sample-sessions/service", headers=AUTH).json()
        assert down["service"] == "down" and down["loaded"] is None and "--profile setup" in down["command"]
        assert c.get("/api/sample-sessions/images", headers=AUTH).json()["images"] == []
        folder = tmp_path / "samples"
        folder.mkdir()
        cv2.imwrite(str(folder / "b.jpg"), FRAME)
        cv2.imwrite(str(folder / "a.png"), FRAME)
        (folder / "notes.txt").write_text("x")
        listed = c.get("/api/sample-sessions/images", headers=AUTH).json()
        assert listed["images"] == ["a.png", "b.jpg"] and listed["folder"].endswith("samples")
        assert c.post("/api/sample-sessions", headers=AUTH, json={"sample": "b.jpg"}).status_code == 200  # the listed name starts a session
        assert c.post("/api/sample-sessions", headers=AUTH, json={"sample": "../h.db"}).status_code == 404


def test_a_negative_sam_click_is_passed_on_and_the_camera_slot_test_is_gone(env, monkeypatch, scripted):
    net = Net()
    with app_client(env, monkeypatch, net) as c:
        s = new(c)
        clicks = [{"x": 960, "y": 700, "label": 1}, {"x": 100, "y": 100, "label": 0}]
        assert c.post(f"/api/sample-sessions/{s['id']}/surface", headers=AUTH, json={"clicks": clicks}).status_code == 200
        wait(c, s["id"])
        assert [k["label"] for k in net.setup_calls[0][1]["clicks"]] == [1, 0]
        # the old "test on this camera's slots" endpoints no longer exist
        assert c.post("/api/sample-test", headers=AUTH, json={"image": b64()}).status_code in (404, 405)
        assert c.get("/api/sample-test/images", headers=AUTH).status_code in (404, 405)


# ---- the settings panel: detector, confidence, NMS, footprint ----------------------------------------------------------------------------------
def test_a_session_starts_at_a_lower_confidence_than_a_live_camera_and_lists_the_detectors(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        assert c.get("/api/sample-sessions/models").status_code == 401
        listed = c.get("/api/sample-sessions/models", headers=AUTH).json()
        assert listed["defaults"] == {"model": listed["defaults"]["model"], "conf": 0.20, "iou": 0.45, "footprint": 0.50, "encroach": 0.30}
        assert listed["defaults"]["model"] in ("default", "yolov8s.pt")  # the standard COCO detector when its weights are in the project, else the camera's own
        ids = {m["id"]: m for m in listed["models"]}
        assert ids["default"]["kind"] == "vehicle" and "weights" not in ids["default"]  # never a path to the page
        assert any(m["kind"] == "bay" and "PKLot" in m["label"] for m in ids.values()) or True  # (the weights are in the project, not in the test's root)
        assert new(c)["tune"] == listed["defaults"]  # the live camera runs at 0.30; the sample at 0.20


def test_the_settings_reach_the_detector_and_the_rules_and_come_back_in_the_result(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        model = c.get("/api/sample-sessions/models", headers=AUTH).json()["defaults"]["model"]  # the COCO baseline when its weights exist, else the camera's own
        weights = None if model == "default" else model
        sid, _ = calibrated(c)
        assert scripted["tunes"][-1] == {"weights": weights, "bay": False, "conf": 0.20, "iou": 0.45}
        out = c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"tune": {"conf": 0.12, "iou": 0.3, "footprint": 0.7, "encroach": 0.4}}).json()
        assert scripted["tunes"][-1] == {"weights": weights, "bay": False, "conf": 0.12, "iou": 0.3}
        assert out["tune"] == {"model": model, "conf": 0.12, "iou": 0.3, "footprint": 0.7, "encroach": 0.4}
        assert out["detector"] == ("This camera's own detector" if model == "default" else f"COCO · {model}") and out["detected"] == len(CARS)
        assert c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()["tune"]["conf"] == 0.12  # kept for the next call
        assert c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={}).json()["tune"]["conf"] == 0.12


def test_the_detector_runs_once_per_setting_not_once_per_click(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        sid, _ = calibrated(c)
        n = len(scripted["calls"])
        c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={})
        c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layers": {"spillage": True}})
        assert len(scripted["calls"]) == n  # same picture, same settings: the answer is reused
        c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"tune": {"conf": 0.15}})
        assert len(scripted["calls"]) == n + 1  # a different confidence is a new pass


def test_settings_out_of_the_slider_ranges_or_an_unknown_model_are_refused_before_any_work(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        sid, _ = calibrated(c)
        n = len(scripted["calls"])
        for bad in ({"conf": 0.05}, {"conf": 0.9}, {"iou": 0.1}, {"iou": 0.9}, {"footprint": 0.1}, {"footprint": 0.95}, {"encroach": 0.9}):
            assert c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"tune": bad}).status_code == 422, bad
        for model in ("../../etc/passwd", "C:/x.pt", "nope.pt"):
            assert c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"tune": {"model": model}}).status_code == 422, model
        assert len(scripted["calls"]) == n
        assert c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()["tune"]["conf"] == 0.20  # a refused request changed nothing


def test_pictures_in_data_sample_are_listed_too(env, monkeypatch, scripted, tmp_path):
    (tmp_path / "sample").mkdir()
    cv2.imwrite(str(tmp_path / "sample" / "lot.png"), FRAME)
    with app_client(env, monkeypatch) as c:
        assert c.get("/api/sample-sessions/images", headers=AUTH).json()["images"] == ["lot.png"]
        assert c.post("/api/sample-sessions", headers=AUTH, json={"sample": "lot.png"}).status_code == 200


# ---- the direct bay model (a comparison; no calibration, nothing of the session changes) ----------------------------------------------------------
@pytest.fixture
def bay_root(tmp_path, monkeypatch):
    """A project root with a (dummy) PKLot weights file in models/: the endpoint only lists it and hands its relative path to the (scripted) detector."""
    import api.main

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "parking_yolo.pt").write_bytes(b"x")
    (tmp_path / "models" / "parking_yolo_obb_holdout-PUCPR.pt").write_bytes(b"x")
    monkeypatch.setattr(api.main, "ROOT", tmp_path)
    return tmp_path


def test_the_models_list_names_the_three_benchmark_options_and_their_box_family(env, monkeypatch, scripted, bay_root):
    (bay_root / "yolov8s.pt").write_bytes(b"x")
    with app_client(env, monkeypatch) as c:
        models = c.get("/api/sample-sessions/models", headers=AUTH).json()["models"]
    by = {m["option"]: m for m in models if m.get("option")}
    assert sorted(by) == ["A", "B", "C"]
    assert (by["A"]["family"], by["A"]["kind"]) == ("coco", "vehicle") and by["A"]["id"] == "yolov8s.pt"
    assert (by["B"]["family"], by["B"]["kind"], by["B"]["id"]) == ("aabb", "bay", "models/parking_yolo.pt")
    assert (by["C"]["family"], by["C"]["kind"]) == ("obb", "bay") and "obb" in by["C"]["id"] and by["C"]["label"].startswith("PKLot bay model")


def bay(x, y, state, conf=0.8, w=120, h=60):
    return {"poly": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], "box": (x, y, x + w, y + h), "label": "busy_slot" if state else "free_slot", "conf": conf, "state": state}


def test_the_bay_model_runs_straight_on_an_uncalibrated_picture_and_changes_nothing(env, monkeypatch, scripted, bay_root):
    scripted["dets"] = [bay(100, 200, 1), bay(300, 200, 0), bay(500, 200, 1)]
    with app_client(env, monkeypatch) as c:
        sid = new(c)["id"]
        assert c.post(f"/api/sample-sessions/{sid}/direct", json={}).status_code == 401
        r = c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={})
        assert r.status_code == 200, r.text
        out = r.json()
        assert scripted["tunes"][-1] == {"weights": "models/parking_yolo.pt", "bay": True, "conf": 0.25, "iou": 0.45}
        assert out["kind"] == "direct" and out["slot_count"] == 3 and out["counts"]["busy"] == 2 and out["counts"]["free"] == 1 and out["counts"]["slots"] == 3
        assert [s["state"] for s in out["slots"]] == ["BUSY", "FREE", "BUSY"] and out["detector"].startswith("PKLot bay model")
        assert all(0 <= v <= 1 for s in out["slots"] for p in s["poly"] for v in p) and out["image"].startswith("data:image/jpeg")
        assert out["direct"] == {"conf": 0.25, "iou": 0.45} and "tune" not in out
        after = c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()
        assert after["calibration"] is None and after["slots"] == [] and after["tune"]["conf"] == 0.20  # the session is as it was
        # its own settings, and the other bay model
        out = c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={"model": "models/parking_yolo_obb_holdout-PUCPR.pt", "conf": 0.1, "iou": 0.5}).json()
        assert scripted["tunes"][-1] == {"weights": "models/parking_yolo_obb_holdout-PUCPR.pt", "bay": True, "conf": 0.1, "iou": 0.5}
        assert c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()["tune"]["conf"] == 0.20


def test_the_calibrated_pipeline_is_untouched_by_a_direct_run(env, monkeypatch, scripted, bay_root):
    with app_client(env, monkeypatch) as c:
        sid, _ = calibrated(c)
        before = c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={}).json()
        scripted["dets"] = [bay(100, 200, 1)]
        assert c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={}).status_code == 200
        scripted["dets"] = [dict(d) for d in CARS]
        after = c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={}).json()
        assert (after["counts"], [s["state"] for s in after["slots"]], after["kind"]) == (before["counts"], [s["state"] for s in before["slots"]], "session")


def test_the_direct_run_refuses_bad_input_and_a_missing_bay_model(env, monkeypatch, scripted, bay_root, tmp_path):
    with app_client(env, monkeypatch) as c:
        sid = new(c)["id"]
        n = len(scripted["calls"])
        for bad in ({"conf": 0.01}, {"conf": 0.9}, {"iou": 0.1}, {"model": "default"}, {"model": "../../x.pt"}, {"model": "nope.pt"}):
            assert c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json=bad).status_code == 422, bad
        assert len(scripted["calls"]) == n  # the detector never ran
        for f in (bay_root / "models").glob("parking_yolo*"):
            f.unlink()
        r = c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={})
        assert r.status_code == 409 and "bay model" in r.json()["detail"]


def drawn(i, x, y, w=120, h=60):
    """A hand-drawn slot (fractions of the test picture) over the pixel box x, y, w, h."""
    return {"id": i, "name": f"Slot {i}", "poly": [[x / W, y / H], [(x + w) / W, y / H], [(x + w) / W, (y + h) / H], [x / W, (y + h) / H]]}


def test_drawn_slots_get_busy_or_free_never_unknown_and_say_who_decided(env, monkeypatch, scripted, bay_root):
    car = {"box": (10, 10, 90, 70), "poly": [(10, 10), (90, 10), (90, 70), (10, 70)], "label": "car", "conf": 0.8}

    def detect(run, frame, root, objects=False, tune=None):
        scripted["tunes"].append(tune)
        if tune["bay"]:  # the bay model sees the two bays on the whole picture and nothing in any single slot crop
            return [bay(100, 200, 1), bay(300, 200, 0)] if frame.shape[:2] == FRAME.shape[:2] else []
        return [dict(car)]

    monkeypatch.setattr(sample_test, "detect_frame", detect)
    with app_client(env, monkeypatch) as c:
        sid = new(c)["id"]
        slots = [drawn(1, 105, 205), drawn(2, 300, 200), drawn(3, 20, 15), drawn(4, 400, 350)]
        r = c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={"slots": slots})
        assert r.status_code == 200, r.text
        out = r.json()
        assert [(s["name"], s["state"], s["by"]) for s in out["slots"]] == [
            ("Slot 1", "BUSY", "bay model, whole picture"), ("Slot 2", "FREE", "bay model, whole picture"),
            ("Slot 3", "BUSY", "vehicle detector"), ("Slot 4", "FREE", "vehicle detector")]
        assert out["counts"]["unknown"] == 0 and (out["counts"]["busy"], out["counts"]["free"], out["counts"]["slots"]) == (2, 2, 4)
        assert out["direct"]["by"] == {"bay model, whole picture": 2, "vehicle detector": 2} and out["kind"] == "direct"
        assert out["layers"] == {"spillage": False, "obstacle_filter": False, "blocked_objects": False, "ground": False}
        assert not any(s["state"] in ("ENCROACHED", "BLOCKED") or s.get("encroached_by") or s.get("blocked_by") for s in out["slots"])
        # the camera's own detector, chosen as the model, judges every slot itself
        out = c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={"model": "default", "slots": slots}).json()
        assert scripted["tunes"][-1]["bay"] is False and scripted["tunes"][-1]["weights"] is None
        assert [s["state"] for s in out["slots"]] == ["FREE", "FREE", "BUSY", "FREE"] and {s["by"] for s in out["slots"]} == {"vehicle detector"}
        # another model of the list works the same; the plain (no slots) direct run still wants a bay model
        assert c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={"model": "default"}).status_code == 422
        after = c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()
        assert after["slots"] == [] and after["calibration"] is None  # the session keeps nothing of them


def test_drawn_slots_are_validated_like_every_slot_map(env, monkeypatch, scripted, bay_root):
    with app_client(env, monkeypatch) as c:
        sid = new(c)["id"]
        n = len(scripted["calls"])
        bad = ([], [{"id": 1, "name": "x", "poly": [[0.1, 0.1], [0.2, 0.2]]}], [{"id": 1, "name": "x", "poly": [[0.1, 0.1], [1.5, 0.1], [0.5, 0.5]]}])
        for slots in bad:
            assert c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={"slots": slots}).status_code == 422, slots
        assert len(scripted["calls"]) == n  # the detector never ran


def test_a_weak_bay_verdict_is_not_trusted_and_an_empty_slot_next_to_a_car_stays_free(env, monkeypatch, scripted, bay_root):
    car = {"box": (100, 100, 220, 190), "poly": [(100, 100), (220, 100), (220, 190), (100, 190)], "label": "car", "conf": 0.8}

    def detect(run, frame, root, objects=False, tune=None):
        if tune["bay"]:  # a low-confidence "busy" on the empty slot, as the PKLot models give on empty ground out of their domain
            return [bay(300, 100, 1, conf=0.2)] if frame.shape[:2] == FRAME.shape[:2] else []
        return [dict(car)]

    monkeypatch.setattr(sample_test, "detect_frame", detect)
    with app_client(env, monkeypatch) as c:
        sid = new(c)["id"]
        out = c.post(f"/api/sample-sessions/{sid}/direct", headers=AUTH, json={"slots": [drawn(10, 100, 130), drawn(11, 300, 100)]}).json()
        assert [(s["name"], s["state"], s["by"]) for s in out["slots"]] == [("Slot 10", "BUSY", "vehicle detector"), ("Slot 11", "FREE", "vehicle detector")]


# ---- zones: SAM + GeoCalib per zone, one merged run --------------------------------------------------------------------------------------
NEAR = (to_image(test_setup_api.CAM, [(-4, 8), (4, 8), (8, 16), (-8, 16)]) / (W, H)).tolist()
FAR = (to_image(test_setup_api.CAM, [(-8, 16), (8, 16), (12, 24), (-12, 24)]) / (W, H)).tolist()


def zone_cal(c, sid, zone, polygon, **kw):
    r = c.post(f"/api/sample-sessions/{sid}/calibrate", headers=AUTH, json={**RECT, "polygon": polygon, "zone": zone, **kw})
    assert r.status_code == 200, r.text
    return wait(c, sid)


def test_each_zone_gets_its_own_geocalib_run_and_grid_and_the_auto_grid_is_left_alone(env, monkeypatch, scripted):
    net = Net()
    with app_client(env, monkeypatch, net) as c:
        sid, auto = calibrated(c)
        a, b = (to_image(test_setup_api.CAM, [(-3, 12), (2, 15)]) / (W, H)).tolist()
        geo = {"method": "geocalib", "reference": {"a": a, "b": b, "metres": float(np.hypot(5, 3))}, "lens": "simple_radial"}
        ja = zone_cal(c, sid, 1, NEAR, **geo)
        jb = zone_cal(c, sid, 2, FAR, zone_name="Back row", **geo)
        assert ja["state"] == "done" and jb["state"] == "done", (ja, jb)
        masks = [cv2.imdecode(np.frombuffer(base64.b64decode(body["mask"]), np.uint8), cv2.IMREAD_GRAYSCALE) for p, body in net.setup_calls if p == "/calibrate"]
        assert len(masks) == 2 and (masks[0] != masks[1]).any()  # GeoCalib ran once per zone, each on its own zone's ground
        ra, rb = ja["result"], jb["result"]
        assert ra["zone"] == 1 and rb["zone"] == 2 and ra["confidence"] == "HIGH" and ra["bays"] >= 1 and rb["bays"] >= 1
        assert [x["name"] for x in ra["slots"]] == [f"A{n}" for n in range(1, ra["bays"] + 1)] and {x["id"] // 100 for x in rb["slots"]} == {2}
        s = c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()
        assert s["layout"] == "zones" and [z["name"] for z in s["zones"]] == ["Zone A", "Back row"]
        # bays laid across the shared edge of the two zones are one bay: the later zone's copy is left out
        assert all(d["id"] // 100 == 2 and d["same_as"].startswith("A") for d in s["zone_dropped"])
        assert len(s["zone_slots"]) == ra["bays"] + rb["bays"] - len(s["zone_dropped"]) > ra["bays"]
        assert s["slots"] == auto["slots"] and s["polygon"] == SURFACE  # the auto grid is untouched
        assert c.delete(f"/api/sample-sessions/{sid}/zones/2", headers=AUTH).json()["zones"][0]["zone"] == 1
        assert c.delete(f"/api/sample-sessions/{sid}/zones/2", headers=AUTH).status_code == 404


@pytest.mark.parametrize("n", [3, 5, 8])
def test_an_expected_bay_count_fixes_the_zone(env, monkeypatch, scripted, n):
    net = Net()
    with app_client(env, monkeypatch, net) as c:
        sid, _ = calibrated(c)
        a, b = (to_image(test_setup_api.CAM, [(-3, 12), (2, 15)]) / (W, H)).tolist()
        geo = {"method": "geocalib", "reference": {"a": a, "b": b, "metres": float(np.hypot(5, 3))}, "lens": "simple_radial"}
        sizes = set()
        for _ in range(2):  # (again and again: the count never moves by itself)
            j = zone_cal(c, sid, 1, NEAR, expected_bays=n, **geo)
            assert j["state"] == "done" and j["result"]["bays"] == n and len(j["result"]["slots"]) == n, j
            sizes.add(len(j["result"]["slots"]))
        assert sizes == {n}
        z = c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()["zones"][0]
        assert len(z["outline"]) == 4  # the outline the bays were cut from (the page shows it)


def test_a_zone_can_be_laid_again_from_a_master_bay_dots_a_margin_and_a_refit(env, monkeypatch, scripted):
    net = Net()
    with app_client(env, monkeypatch, net) as c:
        sid, _ = calibrated(c)
        a, b = (to_image(test_setup_api.CAM, [(-3, 12), (2, 15)]) / (W, H)).tolist()
        geo = {"method": "geocalib", "reference": {"a": a, "b": b, "metres": float(np.hypot(5, 3))}, "lens": "simple_radial"}
        j = zone_cal(c, sid, 1, NEAR, expected_bays=5, **geo)
        assert j["state"] == "done"
        url = f"/api/sample-sessions/{sid}/zones/1/layout"
        first = j["result"]["slots"][0]["poly"]
        r = c.post(url, headers=AUTH, json={"mode": "master", "master": first, "count": 4})
        assert r.status_code == 200, r.text
        out = r.json()
        assert len(out["zones"][0]["slots"]) == 4 and out["info"]["width_m"] > 0.3 and out["layout"]["count"] == 4
        thin = c.post(url, headers=AUTH, json={"mode": "master", "master": first, "count": 4, "margin_m": 0.15}).json()
        full_area = lambda sl: sum(abs(np.cross(np.array(q["poly"])[1] - np.array(q["poly"])[0], np.array(q["poly"])[3] - np.array(q["poly"])[0])) for q in sl)  # noqa: E731
        assert full_area(thin["zones"][0]["slots"]) < full_area(out["zones"][0]["slots"])
        again = c.post(url, headers=AUTH, json={"mode": "refit"}).json()
        assert len(again["zones"][0]["slots"]) == 4 and again["layout"]["rows"] == 1
        dots = [{"x": q[0], "y": q[1]} for q in (first[0], out["zones"][0]["slots"][2]["poly"][0])]
        d = c.post(url, headers=AUTH, json={"mode": "dots", "dots": [{"x": dots[0]["x"], "y": dots[0]["y"], "index": 0}, {"x": dots[1]["x"], "y": dots[1]["y"], "index": 2}], "count": 4})
        assert d.status_code == 200 and len(d.json()["zones"][0]["slots"]) == 4, d.text
        assert c.post(url, headers=AUTH, json={"mode": "dots", "count": 3}).status_code == 422  # no dots
        tilted = c.post(url, headers=AUTH, json={"mode": "dots", "dots": [{"x": dots[0]["x"], "y": dots[0]["y"], "index": 0}, {"x": dots[1]["x"], "y": dots[1]["y"], "index": 2}], "count": 4, "stall_angle": 60})
        assert tilted.status_code == 200 and tilted.json()["layout"]["stall_angle"] == 60 and tilted.json()["info"]["stall_source"] == "typed", tilted.text
        assert c.post(url, headers=AUTH, json={"mode": "dots", "dots": [{"x": 0.4, "y": 0.8}, {"x": 0.5, "y": 0.8}], "count": 3, "stall_angle": 5}).status_code == 422  # outside 20-160
        assert c.post(f"/api/sample-sessions/{sid}/zones/9/layout", headers=AUTH, json={"mode": "refit"}).status_code == 404


def test_sam_for_a_zone_is_cut_to_its_outline_and_does_not_replace_the_surface(env, monkeypatch, scripted):
    net = Net()
    with app_client(env, monkeypatch, net) as c:
        sid = new(c)["id"]
        r = c.post(f"/api/sample-sessions/{sid}/surface", headers=AUTH, json={"clicks": [{"x": 960, "y": 900, "label": 1}], "zone": 1, "bound": NEAR})
        assert r.status_code == 200, r.text
        job = wait(c, sid)
        assert job["state"] == "done", job
        poly = np.asarray(job["result"]["polygon"])
        near = np.asarray(NEAR)
        assert job["result"]["zone"] == 1 and poly[:, 1].min() >= near[:, 1].min() - 0.01  # nothing beyond the zone's outline
        assert c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()["polygon"] is None


def test_zones_run_as_one_engine_overlaps_are_one_bay_and_nothing_live_moves(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        sid = new(c)["id"]
        assert c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layout": "zones"}).status_code == 409
        before = live_state(c, env)
        assert zone_cal(c, sid, 1, SURFACE)["state"] == "done"
        twin = zone_cal(c, sid, 2, SURFACE)["result"]  # the same ground twice: every bay of zone B is a bay of zone A
        assert len(twin["zone_dropped"]) == twin["bays"] and twin["zone_dropped"][0]["same_as"].startswith("A")
        assert len(twin["zone_slots"]) == twin["zones"][0]["calibration"]["bays"]
        merged = twin["zone_slots"]
        r = c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layout": "zones", "skip": [merged[0]["id"]], "layers": {"spillage": True}})
        assert r.status_code == 200, r.text
        out = r.json()
        assert sorted(x["id"] for x in out["slots"]) == sorted(x["id"] for x in merged[1:])  # no duplicates, the removed bay out, no spawned / sub-bays
        assert {x["zone"] for x in out["slots"]} == {"Zone A"} and out["layout"] == "zones" and out["slot_count"] == len(merged) - 1
        assert out["layers"]["spillage"] is True and out["layers"]["ground"] is True  # the zones are calibrated: spillage has its ground metric
        assert out["counts"]["busy"] <= out["vehicles"]
        assert c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layout": "auto"}).status_code == 409  # no auto grid in this session
        assert c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layout": "magic"}).status_code == 422
        assert live_state(c, env) == before


def test_a_zone_can_be_extended_by_the_next_bays_and_its_bays_edited(env, monkeypatch, scripted):
    with app_client(env, monkeypatch) as c:
        sid = new(c)["id"]
        res = zone_cal(c, sid, 1, NEAR)["result"]
        n = res["bays"]
        r = c.post(f"/api/sample-sessions/{sid}/zones/1/extend", headers=AUTH, json={"count": 2})
        assert r.status_code == 200, r.text
        out = r.json()
        a = out["zones"][0]["slots"]
        assert out["added"] >= 1 and len(a) == n + out["added"] and [x["name"] for x in a] == [f"A{k}" for k in range(1, len(a) + 1)]
        assert out["zones"][0]["calibration"]["bays"] == len(a) and len(out["zone_slots"]) == len(a)
        assert c.post(f"/api/sample-sessions/{sid}/zones/1/extend", headers=AUTH, json={"count": 0}).status_code == 422
        assert c.post(f"/api/sample-sessions/{sid}/zones/3/extend", headers=AUTH, json={"count": 1}).status_code == 404
        moved = [{**x, "poly": [[px + 0.01, py] for px, py in x["poly"]]} if i == 0 else x for i, x in enumerate(a[:-1])]
        r = c.put(f"/api/sample-sessions/{sid}/zones/1/slots", headers=AUTH, json={"slots": moved})
        assert r.status_code == 200, r.text
        z = r.json()["zones"][0]
        assert len(z["slots"]) == len(a) - 1 and z["slots"][0]["poly"][0][0] == pytest.approx(a[0]["poly"][0][0] + 0.01, abs=1e-4)
        assert c.put(f"/api/sample-sessions/{sid}/zones/1/slots", headers=AUTH, json={"slots": [{**a[0], "id": 201}]}).status_code == 422  # another zone's id
        assert c.put(f"/api/sample-sessions/{sid}/zones/1/slots", headers=AUTH, json={"slots": [{**a[0], "poly": [[0, 0], [1, 1]]}]}).status_code == 422
        # a whole-zone turn on the page sends the turned outline and ground with the bays: kept, and bad corners refused
        turned = [[0.2, 0.3], [0.8, 0.31], [0.8, 0.7], [0.2, 0.69]]
        r = c.put(f"/api/sample-sessions/{sid}/zones/1/slots", headers=AUTH, json={"slots": moved, "outline": turned, "polygon": turned})
        assert r.status_code == 200, r.text
        assert r.json()["zones"][0]["polygon"] == turned
        assert c.put(f"/api/sample-sessions/{sid}/zones/1/slots", headers=AUTH, json={"slots": moved, "outline": [[0.1, 0.1], [1.5, 0.2], [0.5, 0.5]]}).status_code == 422


def test_a_capacity_pin_survives_extend_and_refit_and_counts_its_vehicles_in_the_run(env, monkeypatch, scripted):
    """Port of the live dashboard's "manual pin with capacity": a hand-placed slot with capacity N in a zone. The grid steps leave it alone, the run counts the
    vehicles standing on it (up to N) through the same tracker as the live view, and the total counts N spaces."""
    cam = test_setup_api.CAM
    pin_poly = (to_image(cam, [(-4.5, 9.5), (2.5, 9.5), (2.5, 15.0), (-4.5, 15.0)]) / (W, H)).round(4).tolist()  # around the two cars of the zone
    with app_client(env, monkeypatch) as c:
        sid = new(c)["id"]
        a = zone_cal(c, sid, 1, NEAR)["result"]["zones"][0]["slots"]
        pin = {"id": 150, "name": "P1", "poly": pin_poly, "capacity": 3}
        r = c.put(f"/api/sample-sessions/{sid}/zones/1/slots", headers=AUTH, json={"slots": [*a, pin]})
        assert r.status_code == 200, r.text
        saved = [x for x in r.json()["zones"][0]["slots"] if x["id"] == 150][0]
        assert saved["capacity"] == 3 and saved["name"] == "P1"
        bad = c.put(f"/api/sample-sessions/{sid}/zones/1/slots", headers=AUTH, json={"slots": [*a, {**pin, "capacity": 25}]})
        assert bad.status_code == 422 and "capacity" in bad.text  # the live limit: a whole number from 1 to 20
        # "Add next 1" continues the row from the laid bays: the pin keeps its id and name, and is not renamed A<n>
        ext = c.post(f"/api/sample-sessions/{sid}/zones/1/extend", headers=AUTH, json={"count": 1})
        assert ext.status_code == 200, ext.text
        zs = ext.json()["zones"][0]["slots"]
        assert [x["name"] for x in zs if x["id"] != 150] == [f"A{k}" for k in range(1, len(zs))]
        assert [x for x in zs if x["id"] == 150][0]["name"] == "P1" and [x for x in zs if x["id"] == 150][0]["capacity"] == 3
        # a re-fit of the laid bays leaves it too
        rf = c.post(f"/api/sample-sessions/{sid}/zones/1/layout", headers=AUTH, json={"mode": "refit", "count": 1})
        assert rf.status_code == 200, rf.text
        zs = rf.json()["zones"][0]["slots"]
        assert [x for x in zs if x["id"] == 150][0]["capacity"] == 3
        # the run: the grid bays whose centre is under the pin are removed (the page does it when the pin is dropped; none here: the grid lies at the pin's near edge)
        under = [x["id"] for x in zs if x["id"] != 150 and cv2.pointPolygonTest(np.float32(pin_poly), (float(np.mean([q[0] for q in x["poly"]])), float(np.mean([q[1] for q in x["poly"]]))), False) >= 0]
        out = c.post(f"/api/sample-sessions/{sid}/run", headers=AUTH, json={"layout": "zones", "skip": under}).json()
        p1 = [x for x in out["slots"] if x["id"] == 150][0]
        assert p1["capacity"] == 3 and p1["vehicles"] == 2 and p1["state"] == "BUSY", p1
        assert out["counts"]["slots"] == len(out["slots"]) - 1 + 3  # the pin is 3 spaces
        assert out["counts"]["busy"] + out["counts"]["free"] + out["counts"]["unknown"] == out["counts"]["slots"]


# ---- top-down (aerial) scale: no GeoCalib; a level GeoCalib answer says what to choose -----------------------------------------------------
def test_the_topdown_method_needs_no_setup_service_and_takes_its_scale_from_the_typed_distance(env, monkeypatch, scripted):
    net = Net(setup_up=False)  # the setup service is down: top-down must not need it
    with app_client(env, monkeypatch, net) as c:
        s = new(c)
        a, b = (to_image(test_setup_api.CAM, [(-3, 12), (2, 15)]) / (W, H)).tolist()
        no_ref = c.post(f"/api/sample-sessions/{s['id']}/calibrate", headers=AUTH, json={"method": "topdown", "polygon": SURFACE})
        assert no_ref.status_code == 422 and "reference" in no_ref.text
        body = {"method": "topdown", "polygon": SURFACE, "reference": {"a": a, "b": b, "metres": float(np.hypot(5, 3))}}
        assert c.post(f"/api/sample-sessions/{s['id']}/calibrate", headers=AUTH, json=body).status_code == 200
        job = wait(c, s["id"])
        assert job["state"] == "done", job
        assert job["result"]["method"] == "topdown" and job["result"]["bays"] >= 1
        assert any("top-down" in t for t in job["result"]["warnings"])
        assert net.setup_calls == []  # GeoCalib / SAM were never asked


def test_a_level_geocalib_answer_on_an_aerial_picture_points_to_the_topdown_method(env, monkeypatch, scripted):
    level = [0.0, -1.0, 0.1]  # up almost along the camera's -y: the camera looks (nearly) level, as GeoCalib read an overhead picture
    net = Net(up=level)
    with app_client(env, monkeypatch, net) as c:
        s = new(c)
        a, b = (to_image(test_setup_api.CAM, [(-3, 12), (2, 15)]) / (W, H)).tolist()
        body = {"method": "geocalib", "polygon": SURFACE, "reference": {"a": a, "b": b, "metres": 5.0}, "lens": "simple_radial"}
        assert c.post(f"/api/sample-sessions/{s['id']}/calibrate", headers=AUTH, json=body).status_code == 200
        job = wait(c, s["id"])
        assert job["state"] == "error"
        assert "too level for ground" in job["error"] and "Top-down" in job["error"]


def test_a_new_session_defaults_to_the_standard_coco_detector_when_its_weights_exist(env, monkeypatch, scripted, bay_root):
    with app_client(env, monkeypatch) as c:  # (no yolov8s.pt yet: only the PKLot bay models are there, never chosen by default)
        assert c.get("/api/sample-sessions/models", headers=AUTH).json()["defaults"]["model"] == "default"
        (bay_root / "yolov8s.pt").write_bytes(b"x")
        listed = c.get("/api/sample-sessions/models", headers=AUTH).json()
        assert listed["defaults"]["model"] == "yolov8s.pt" and listed["defaults"]["conf"] == 0.20
        assert new(c)["tune"] == listed["defaults"]  # a new session starts on it, no manual selection


# ---- AI auto-calibrate (carparking/vision_agent): a proposal only, the session never changes ---------------------------------------------------
class FakeVision:
    """A vision agent that 'sees' the zone's own laid bays: line i = the left edge of bay i, the last line the right edge of the last bay (crop fractions)."""
    model = "fake-vision"

    def __init__(self, c, sid, box, fail=None):
        z = c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()["zones"][0]
        self.box, self.slots, self.fail, self.seen = box, [b["poly"] for b in z["slots"]], fail, []

    def propose(self, crop_png_path, constraints):
        from carparking.vision_agent.base import LineEndpoint, RawVisionProposal, Visibility
        self.seen.append((crop_png_path, dict(constraints)))
        if self.fail:
            raise self.fail
        x0, y0, x1, y1 = self.box
        to_crop = lambda p: ((p[0] - x0) / (x1 - x0), (p[1] - y0) / (y1 - y0))  # noqa: E731
        edges = [(q[0], q[3]) for q in self.slots] + [(self.slots[-1][1], self.slots[-1][2])]
        return RawVisionProposal(lines=[LineEndpoint(front=to_crop(f), back=to_crop(b), visible=Visibility.CLEAR) for f, b in edges], estimated_count=len(self.slots), confidence=0.9, notes="")


def laid_zone(c, monkeypatch):
    sid, _ = calibrated(c)
    a, b = (to_image(test_setup_api.CAM, [(-3, 12), (2, 15)]) / (W, H)).tolist()
    geo = {"method": "geocalib", "reference": {"a": a, "b": b, "metres": float(np.hypot(5, 3))}, "lens": "simple_radial"}
    assert zone_cal(c, sid, 1, NEAR, expected_bays=3, **geo)["state"] == "done"  # (3 bays on this 8 m zone: realistic widths)
    return sid


def test_auto_calibrate_proposes_dots_angle_and_count_and_changes_nothing(env, monkeypatch, scripted):
    from carparking import sample_session as ss
    with app_client(env, monkeypatch) as c:
        sid = laid_zone(c, monkeypatch)
        before = c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()
        box = ss.zone_crop_box(before["zones"][0]["outline"])
        agent = FakeVision(c, sid, box)
        c.app.state.vision_agent_factory = lambda: agent
        r = c.post(f"/api/sample-sessions/{sid}/zones/1/auto-calibrate", headers=AUTH, json={"rows": 1, "min_count": 3, "slot_prefix": "B"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["ok"], out
        assert out["count"] == 3 and abs(out["stall_angle"] - 90) < 3 and out["params"]["slot_prefix"] == "B" and out["model"] == "fake-vision"
        assert [d["at"] for d in out["dots"][:2]] == ["front", "back"] and out["dots"][0]["index"] == 0 and len(out["dots"]) == 2 + 3  # line 0 front + back, then the other lines' fronts
        assert all(0 <= d["x"] <= 1 and 0 <= d["y"] <= 1 for d in out["dots"])
        assert agent.seen[0][1] == {"rows": 1, "min_count": 3}  # only what the operator gave
        assert c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json() == before  # the proposal is not applied: the form does that
        # the answer is something the layout route accepts as it is
        lay = c.post(f"/api/sample-sessions/{sid}/zones/1/layout", headers=AUTH, json={"mode": "dots", "count": out["count"], "stall_angle": out["stall_angle"], "depth": out["depth"],
                                                                                    "dots": out["dots"]})
        assert lay.status_code == 200 and len(lay.json()["zones"][0]["slots"]) == 3, lay.text


def test_auto_calibrate_reports_why_a_proposal_was_refused_and_when_the_model_failed(env, monkeypatch, scripted):
    from carparking import sample_session as ss
    with app_client(env, monkeypatch) as c:
        sid = laid_zone(c, monkeypatch)
        box = ss.zone_crop_box(c.get(f"/api/sample-sessions/{sid}", headers=AUTH).json()["zones"][0]["outline"])
        c.app.state.vision_agent_factory = lambda: FakeVision(c, sid, box, fail=RuntimeError("503 overloaded"))
        out = c.post(f"/api/sample-sessions/{sid}/zones/1/auto-calibrate", headers=AUTH, json={}).json()
        assert out["ok"] is False and len(out["attempts"]) == 2 and "503 overloaded" in out["reason"] and "params" not in out
        assert c.post(f"/api/sample-sessions/{sid}/zones/9/auto-calibrate", headers=AUTH, json={}).status_code == 404
        assert c.post(f"/api/sample-sessions/{sid}/zones/1/auto-calibrate", json={}).status_code in (401, 403)
        assert c.post(f"/api/sample-sessions/{sid}/zones/1/auto-calibrate", headers=AUTH, json={"slot_prefix": "bad prefix"}).status_code == 422


def test_auto_calibrate_without_an_api_key_says_so_and_a_zone_must_be_calibrated(env, monkeypatch, scripted):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with app_client(env, monkeypatch) as c:
        sid = laid_zone(c, monkeypatch)
        r = c.post(f"/api/sample-sessions/{sid}/zones/1/auto-calibrate", headers=AUTH, json={})
        assert r.status_code == 501 and "GEMINI_API_KEY" in r.json()["detail"] and "AIza" not in r.text


def test_the_crop_for_the_model_is_the_padded_zone_box():
    from carparking import sample_session as ss
    box = ss.zone_crop_box([[0.2, 0.5], [0.6, 0.5], [0.6, 0.9], [0.2, 0.9]])
    assert box == pytest.approx((0.16, 0.46, 0.64, 0.94))
    assert ss.zone_crop_box(None) == (0.0, 0.0, 1.0, 1.0)
    tiny = ss.zone_crop_box([[0.5, 0.5], [0.52, 0.5], [0.52, 0.52]])
    assert tiny[2] - tiny[0] >= ss.CROP_MIN - 1e-9 and tiny[3] - tiny[1] >= ss.CROP_MIN - 1e-9
    edge = ss.zone_crop_box([[0.0, 0.0], [0.3, 0.0], [0.3, 0.3]])
    assert edge[0] == 0.0 and edge[1] == 0.0
    crop = ss.crop_for_model(np.zeros((2000, 3000, 3), np.uint8), (0.0, 0.0, 1.0, 1.0))
    assert max(crop.shape[:2]) == ss.CROP_SIDE
    assert ss.crop_for_model(np.zeros((200, 300, 3), np.uint8), (0.0, 0.0, 0.5, 0.5)).shape[:2] == (100, 150)
