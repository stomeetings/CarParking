"""Phase 2 setup endpoints (api/geometry_routes.py): the flag, the admin token, still -> segment -> calibrate jobs against a
mock setup service, drafts in the geometry store, and that nothing the worker uses is touched. No models, no network."""
import base64
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import Config, create_app  # noqa: E402
from carparking import slot_preview  # noqa: E402
from carparking.geometry import CameraModel, to_image  # noqa: E402
from carparking.live import LiveStore  # noqa: E402

URL = "rtsp://admin:secret@cam.local:554/Streaming/Channels/101"
TOKEN = "t0k3n"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
W, H = 1920, 1080
CAM = CameraModel(f=1500.0, cx=W / 2, cy=H / 2, pitch=np.radians(32), roll=np.radians(0.5), h=6.0)
GROUND = [(-4, 8), (4, 8), (12, 24), (-12, 24)]
POLYGON = (to_image(CAM, GROUND) / (W, H)).tolist()
UP = (-np.array([np.cos(CAM.pitch) * np.sin(CAM.roll), np.cos(CAM.pitch) * np.cos(CAM.roll), np.sin(CAM.pitch)])).tolist()
JPEG = cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))[1].tobytes()


def names_of(weights):
    return ["car", "motorcycle", "bus", "truck"], ["person", "dog"]


@pytest.fixture
def env(tmp_path):
    return {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
            "CARPARK_RTSP_URL": URL, "CARPARK_API_TOKEN": TOKEN, "CARPARK_SPA_DIR": str(tmp_path / "dist"),
            "CARPARK_GEOMETRY_ENABLED": "1", "CARPARK_SETUP_URL": "http://setup:8100", "CARPARK_GO2RTC_INTERNAL": "http://g"}


class Net:
    """Mock network: the camera's snapshot, and the setup service (what it was asked, what it answers)."""

    def __init__(self, up=UP, depth=6.1, setup_up=True, k1=0.0, jpeg=None):
        self.up, self.depth, self.setup_up, self.setup_calls, self.camera_calls, self.k1 = up, depth, setup_up, [], 0, k1
        self.jpeg = jpeg or JPEG  # what the camera's snapshot returns

    def handler(self, request: httpx.Request):
        if request.url.host == "cam.local":
            self.camera_calls += 1
            return httpx.Response(200, content=self.jpeg, headers={"Content-Type": "image/jpeg"})
        if request.url.host == "setup":
            if not self.setup_up:
                raise httpx.ConnectError("refused")
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True, "loaded": {"sam2": True, "geocalib": False, "depth": False},
                                                 "pins": {"sam2": {"commit": "abc"}}})
            body = __import__("json").loads(request.content)
            self.setup_calls.append((request.url.path, body))
            if request.url.path == "/segment":
                mask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(mask, [np.round(np.array(POLYGON) * (W, H)).astype(np.int32)], 255)
                png = base64.b64encode(cv2.imencode(".png", mask)[1].tobytes()).decode()
                return httpx.Response(200, json={"size": [W, H], "seconds": 1.5,
                                                 "masks": [{"score": 0.93, "png": png}, {"score": 0.5, "png": png}]})
            return httpx.Response(200, json={"geocalib": {"f": 1500.0, "up": self.up, "n": len(body["images"]),
                                                          "up_spread_deg": 0.1, "k1": self.k1, "k2": 0.0,
                                                          "model": body["camera_model"]},
                                             "depth": None if self.depth is None else {"h": self.depth, "n": 9000},
                                             "pins": {"sam2": {"commit": "abc"}}})
        return httpx.Response(404)


def make(env, net, monkeypatch, dets=()):
    detected = []
    monkeypatch.setattr(slot_preview, "detect_on_demand",
                        lambda run, frame, root: detected.append(frame.shape) or list(dets))
    http = httpx.AsyncClient(transport=httpx.MockTransport(net.handler))
    app = create_app(Config(env), names_of=names_of, http=http, setup_http=http)
    return TestClient(app), detected


def wait(c, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        job = c.get(f"/api/setup/jobs/{job_id}", headers=AUTH).json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish: " + str(job))


def cam_id(c):
    return c.get("/api/cameras", headers=AUTH).json()[0]["id"]


def test_everything_is_off_without_the_flag_and_needs_the_admin_token(env, monkeypatch):
    net = Net()
    with make({**env, "CARPARK_GEOMETRY_ENABLED": "0"}, net, monkeypatch)[0] as c:
        cid = cam_id(c)
        for method, path, body in (("post", f"/api/cameras/{cid}/setup/still", None),
                                   ("post", f"/api/cameras/{cid}/setup/segment", {"still": "a" * 32, "clicks": [{"x": .5, "y": .5}]}),
                                   ("post", f"/api/cameras/{cid}/setup/calibrate", {"still": "a" * 32, "polygon": POLYGON}),
                                   ("get", "/api/setup/jobs/abc", None)):
            r = getattr(c, method)(path, headers=AUTH, **({"json": body} if body else {}))
            assert r.status_code == 503 and "CARPARK_GEOMETRY_ENABLED" in r.json()["detail"], path
        assert net.camera_calls == 0 and net.setup_calls == []  # nothing was touched
    with make(env, net, monkeypatch)[0] as c:  # on, but no / wrong token
        cid = cam_id(c)
        for headers in ({}, {"Authorization": "Bearer nope"}):
            assert c.post(f"/api/cameras/{cid}/setup/still", headers=headers).status_code == 401
            assert c.get("/api/setup/jobs/abc", headers=headers).status_code == 401
        assert c.post("/api/cameras/nope/setup/still", headers=AUTH).status_code == 404
    with make({**env, "CARPARK_API_TOKEN": ""}, net, monkeypatch)[0] as c:  # no admin token configured: writes disabled
        assert c.post(f"/api/cameras/{cam_id(c)}/setup/still").status_code == 503


def test_still_segment_and_calibrate_make_a_draft_and_change_nothing_else(env, monkeypatch):
    net = Net()
    car = {"label": "car", "conf": 0.9, "src": "", "kind": "vehicle", "box": [0.40, 0.40, 0.50, 0.50]}
    c, detected = make(env, net, monkeypatch, dets=[car])
    with c:
        cid = cam_id(c)
        before_settings, before_slots = LiveStore(env["CARPARK_HISTORY_DB"]).config(), c.get("/api/fixed-slots").json()
        still = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH)
        assert still.status_code == 200, still.text
        still = still.json()
        assert (still["source"], still["width"], still["height"]) == ("camera", W, H) and detected == []  # no detection yet
        assert still["image"].startswith("data:image/jpeg;base64,")

        # segment: clicks are sent to SAM in pixels; the best candidate comes back as a polygon
        r = c.post(f"/api/cameras/{cid}/setup/segment", headers=AUTH,
                   json={"still": still["id"], "clicks": [{"x": 0.5, "y": 0.6, "label": 1}, {"x": 0.9, "y": 0.1, "label": 0}]})
        assert r.status_code == 200, r.text
        job = wait(c, r.json()["id"])
        assert job["state"] == "done", job
        seg = job["result"]
        assert seg["score"] == 0.93 and len(seg["candidates"]) == 2 and seg["model"].startswith("sam2.1")
        assert 0.2 < seg["area_frac"] < 0.5 and len(seg["polygon"]) >= 3
        path, sent = net.setup_calls[0]
        assert path == "/segment" and sent["clicks"] == [{"x": 960.0, "y": 648.0, "label": 1}, {"x": 1728.0, "y": 108.0, "label": 0}]
        assert detected == [(H, W, 3)]  # the vehicles on the still (for filling the surface), once

        # calibrate with the accepted polygon -> a draft
        body = {"still": still["id"], "polygon": seg["polygon"], "meta": {"score": seg["score"], "junk": 1}, "note": "demo"}
        r = c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH, json=body)
        assert r.status_code == 200, r.text
        job = wait(c, r.json()["id"])
        assert job["state"] == "done", job
        res = job["result"]
        assert res["version"] == 1 and res["state"] == "draft" and res["h"] == 6.1 and res["confidence"] == "LOW"  # depth alone
        assert res["pitch_deg"] == pytest.approx(32, abs=0.1) and res["area_m2"] == pytest.approx(256, rel=0.1)
        assert res["capacity"] >= 6 and res["scale_sources"] == {"depth": 6.1}
        path, sent = net.setup_calls[1]
        assert path == "/calibrate" and len(sent["images"]) == 1 and sent["depth"] and sent["mask"]

        store = LiveStore(env["CARPARK_HISTORY_DB"])
        geometry = store.geometry(cid)
        saved = geometry.get(1)
        assert saved["state"] == "draft" and saved["note"] == "demo" and saved["created_by"] == "admin"
        assert saved["record"]["surface"]["score"] == 0.93 and "junk" not in saved["record"]["surface"]
        assert saved["record"]["calibration"]["pins"] == {"sam2": {"commit": "abc"}} and URL not in str(saved)
        assert geometry.active() is None  # a draft is not used by anything
        # a second calibration is another draft; nothing the worker reads changed
        job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH, json=body).json()["id"])
        assert job["result"]["version"] == 2 and [v["ver"] for v in geometry.list()] == [2, 1]
        assert store.config() == before_settings and c.get("/api/fixed-slots").json() == before_slots


def car_fractions(x, y, half=0.9, height=1.45):
    """A car-sized upright billboard at ground point (x, y) of CAM as the detector's box (fractions)."""
    base = to_image(CAM, [(x - half, y), (x + half, y)])
    top = to_image(CAM.with_height(CAM.h - height), [(x - half, y), (x + half, y)])
    pts = np.vstack([base, top])
    (x1, y1), (x2, y2) = pts.min(axis=0), pts.max(axis=0)
    return {"label": "car", "conf": 0.9, "src": "", "kind": "vehicle", "box": [x1 / W, y1 / H, x2 / W, y2 / H]}


def test_the_cars_of_every_still_are_pooled_for_the_scale(env, monkeypatch):
    # two cars per still are too few for a height estimate (it needs 3); the second still adds two more: one camera, one geometry
    net = Net(depth=None)
    c, detected = make(env, net, monkeypatch, dets=[car_fractions(-3, 11), car_fractions(1, 13)])
    with c:
        cid = cam_id(c)
        one = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        two = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        alone = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH, json={"still": one, "polygon": POLYGON}).json()["id"])
        assert alone["state"] == "error" and "no scale source" in alone["error"]  # 2 cars, no depth
        pooled = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH,
                                json={"still": one, "extra_stills": [two], "polygon": POLYGON}).json()["id"])
        assert pooled["state"] == "done", pooled
        assert set(pooled["result"]["scale_sources"]) == {"vehicles"} and pooled["result"]["h"] == pytest.approx(6.0, rel=0.05)
        assert len(net.setup_calls[-1][1]["images"]) == 2 and len(detected) == 2  # both stills went to GeoCalib, each detected once


def test_the_lens_model_is_forwarded_and_stored(env, monkeypatch):
    net = Net(k1=-0.05)
    with make(env, net, monkeypatch)[0] as c:
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH,
                             json={"still": sid, "polygon": POLYGON}).json()["id"])  # default: a radial lens
        assert job["state"] == "done", job
        assert net.setup_calls[-1][1]["camera_model"] == "simple_radial"
        cal = LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid).get(1)["record"]["calibration"]
        assert (cal["lens"], cal["k1"], cal["k2"]) == ("simple_radial", -0.05, 0.0)
        job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH,
                             json={"still": sid, "polygon": POLYGON, "camera_model": "pinhole"}).json()["id"])
        assert net.setup_calls[-1][1]["camera_model"] == "pinhole"
        assert c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH,
                      json={"still": sid, "polygon": POLYGON, "camera_model": "fisheye"}).status_code == 422


def test_failures_are_reported_in_the_job(env, monkeypatch):
    with make(env, Net(setup_up=False), monkeypatch)[0] as c:  # the setup service isn't running
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        job = wait(c, c.post(f"/api/cameras/{cid}/setup/segment", headers=AUTH,
                             json={"still": sid, "clicks": [{"x": .5, "y": .5}]}).json()["id"])
        assert job["state"] == "error" and "docker compose --profile setup up -d setup" in job["error"]
        job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH,
                             json={"still": sid, "polygon": POLYGON}).json()["id"])
        assert job["state"] == "error" and "setup service isn't reachable" in job["error"]
        assert LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid).list() == []  # no draft
    with make(env, Net(up=(0, -1, 0)), monkeypatch)[0] as c:  # GeoCalib says the camera looks level: refused with a reason
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH,
                             json={"still": sid, "polygon": POLYGON}).json()["id"])
        assert job["state"] == "error" and "too level" in job["error"]
        assert LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid).list() == []
    with make(env, Net(depth=None), monkeypatch)[0] as c:  # no scale source: the Admin is asked for a distance
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH,
                             json={"still": sid, "polygon": POLYGON}).json()["id"])
        assert job["state"] == "error" and "reference distance" in job["error"]
        a, b = to_image(CAM, [(-3, 12), (2, 15)]) / (W, H)  # ...which resolves it
        typed = {"still": sid, "polygon": POLYGON, "reference": {"a": a.tolist(), "b": b.tolist(), "metres": float(np.hypot(5, 3))}}
        job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH, json=typed).json()["id"])
        assert job["state"] == "done", job
        assert job["result"]["confidence"] == "HIGH" and job["result"]["h"] == pytest.approx(6.0, rel=0.02)


def test_bad_requests(env, monkeypatch):
    with make(env, Net(), monkeypatch)[0] as c:
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        gone = "f" * 32
        assert c.post(f"/api/cameras/{cid}/setup/segment", headers=AUTH,
                      json={"still": gone, "clicks": [{"x": .5, "y": .5}]}).status_code == 410
        assert c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH,
                      json={"still": gone, "polygon": POLYGON}).status_code == 410
        for body in ({"still": sid, "clicks": []}, {"still": sid, "clicks": [{"x": 1.5, "y": .5}]},
                     {"still": "xyz", "clicks": [{"x": .5, "y": .5}]}):
            assert c.post(f"/api/cameras/{cid}/setup/segment", headers=AUTH, json=body).status_code == 422
        for body in ({"still": sid, "polygon": POLYGON[:2]}, {"still": sid, "polygon": [[0, 0], [2, 0], [1, 1]]},
                     {"still": sid, "polygon": POLYGON, "bay_width": 9}):
            assert c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH, json=body).status_code == 422
        assert c.get("/api/setup/jobs/nope", headers=AUTH).status_code == 404


# --- Accept / discard / read: the Admin's side of the setup ---------------------------------------------------------------
VIEW = "v1ew"
VIEW_AUTH = {"Authorization": f"Bearer {VIEW}"}
# the Admin's typed distance: two ground points and what they are apart
REFERENCE = {"a": (to_image(CAM, [(-3, 12)])[0] / (W, H)).tolist(), "b": (to_image(CAM, [(2, 15)])[0] / (W, H)).tolist(),
             "metres": float(np.hypot(5, 3))}


def calibrate(c, cid, sid, reference=True):
    body = {"still": sid, "polygon": POLYGON, **({"reference": REFERENCE} if reference else {})}
    job = wait(c, c.post(f"/api/cameras/{cid}/setup/calibrate", headers=AUTH, json=body).json()["id"])
    assert job["state"] == "done", job
    return job["result"]


def test_accept_rollback_and_discard_flow(env, monkeypatch):
    with make({**env, "CARPARK_VIEW_TOKEN": VIEW}, Net(), monkeypatch)[0] as c:
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        before = (LiveStore(env["CARPARK_HISTORY_DB"]).config(), c.get("/api/fixed-slots", headers=AUTH).json())

        # a reference-anchored draft is HIGH; one without is not
        first, second, loose = calibrate(c, cid, sid), calibrate(c, cid, sid), calibrate(c, cid, sid, reference=False)
        assert (first["version"], second["version"], loose["version"]) == (1, 2, 3)
        assert first["confidence"] == "HIGH" and first["reference_used"]
        assert loose["confidence"] == "LOW" and not loose["reference_used"] and loose["scale_uncertainty"] > 0.25
        assert c.get(f"/api/cameras/{cid}/geometry", headers=VIEW_AUTH).json() == {"active": None}  # uncalibrated

        # the versions: summaries newest first; one in full
        listing = c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()
        assert listing["active"] is None and [v["ver"] for v in listing["versions"]] == [3, 2, 1]
        assert listing["versions"][2]["confidence"] == "HIGH" and "record" not in listing["versions"][0]
        full = c.get(f"/api/cameras/{cid}/setup/geometry/1", headers=AUTH).json()
        assert full["record"]["calibration"]["confidence"] == "HIGH" and "slots" in full["record"]["capacity"]
        assert c.get(f"/api/cameras/{cid}/setup/geometry/99", headers=AUTH).status_code == 404

        # a version without a typed reference is refused unless the Admin accepts it as it is
        r = c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 3})
        assert r.status_code == 409 and "typed reference" in r.json()["detail"]
        assert c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["active"] is None  # nothing changed

        # Accept version 1: active at once, seen by the very next read of the same running app (nothing restarted)
        r = c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 1})
        assert r.status_code == 200, r.text
        assert r.json()["previous"] is None and r.json()["active"]["ver"] == 1 and r.json()["active"]["state"] == "active"
        live = c.get(f"/api/cameras/{cid}/geometry", headers=VIEW_AUTH).json()["active"]
        assert live["ver"] == 1 and live["confidence"] == "HIGH" and live["h"] == first["h"]
        store = LiveStore(env["CARPARK_HISTORY_DB"]).geometry(cid)
        assert store.active_version() == 1

        # a newer one replaces it; the old one is kept
        r = c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 2})
        assert r.json()["previous"] == 1 and store.active_version() == 2
        states = [v["state"] for v in c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["versions"]]
        assert states == ["draft", "active", "superseded"]
        r = c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 1})  # rollback
        assert r.json()["previous"] == 2 and store.active_version() == 1
        assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 1}).json()["previous"] is None  # no-op
        # unverified, on purpose
        r = c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 3, "accept_unverified": True})
        assert r.status_code == 200 and r.json()["active"]["confidence"] == "LOW" and r.json()["previous"] == 1
        assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 99}).status_code == 404
        assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 0}).status_code == 422

        # a version the drift guard marked stale can't be accepted again
        store.mark_stale("moved")
        assert c.get(f"/api/cameras/{cid}/geometry", headers=AUTH).json() == {"active": None}
        r = c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": 3, "accept_unverified": True})
        assert r.status_code == 409 and "stale" in r.json()["detail"]

        # discard: drafts only
        extra = calibrate(c, cid, sid)["version"]
        assert c.delete(f"/api/cameras/{cid}/setup/draft?version=1", headers=AUTH).status_code == 409  # superseded: history
        assert c.delete(f"/api/cameras/{cid}/setup/draft?version=99", headers=AUTH).status_code == 404
        assert c.delete(f"/api/cameras/{cid}/setup/draft?version={extra}", headers=AUTH).json() == {"discarded": [extra]}
        d1, d2 = calibrate(c, cid, sid)["version"], calibrate(c, cid, sid)["version"]
        assert c.delete(f"/api/cameras/{cid}/setup/draft", headers=AUTH).json() == {"discarded": [d1, d2]}  # every draft
        assert c.delete(f"/api/cameras/{cid}/setup/draft", headers=AUTH).json() == {"discarded": []}
        kept = {v["state"] for v in c.get(f"/api/cameras/{cid}/setup/geometry", headers=AUTH).json()["versions"]}
        assert kept == {"superseded", "stale"}

        # none of this touched what the worker reads
        assert (LiveStore(env["CARPARK_HISTORY_DB"]).config(), c.get("/api/fixed-slots", headers=AUTH).json()) == before


def paths(cid):
    return (("post", f"/api/cameras/{cid}/setup/activate", {"version": 1}), ("delete", f"/api/cameras/{cid}/setup/draft", None),
            ("get", f"/api/cameras/{cid}/setup/geometry", None), ("get", f"/api/cameras/{cid}/setup/geometry/1", None),
            ("get", "/api/setup/status", None))


def test_new_endpoints_need_the_admin_token_and_the_flag(env, monkeypatch):
    net = Net()
    with make({**env, "CARPARK_VIEW_TOKEN": VIEW}, net, monkeypatch)[0] as c:
        cid = cam_id(c)
        for method, path, body in paths(cid):  # no token / only the viewer token: refused
            for headers in ({}, VIEW_AUTH):
                r = getattr(c, method)(path, headers=headers, **({"json": body} if body else {}))
                assert r.status_code == 401, (path, r.status_code)
        assert c.get(f"/api/cameras/{cid}/geometry").status_code == 401  # the read needs the viewer (or admin) token
        assert c.get(f"/api/cameras/{cid}/geometry", headers=AUTH).status_code == 200
    with make({**env, "CARPARK_GEOMETRY_ENABLED": "0", "CARPARK_VIEW_TOKEN": VIEW}, net, monkeypatch)[0] as c:  # flag off
        cid = cam_id(c)
        for method, path, body in paths(cid)[:4]:
            r = getattr(c, method)(path, headers=AUTH, **({"json": body} if body else {}))
            assert r.status_code == 503 and "CARPARK_GEOMETRY_ENABLED" in r.json()["detail"], path
        assert c.get(f"/api/cameras/{cid}/geometry", headers=VIEW_AUTH).status_code == 503
        # the status answers while the flag is off (so the page can say why) and contacts nothing
        assert c.get("/api/setup/status", headers=AUTH).json() == {"enabled": False, "service": "unchecked", "loaded": None, "pins": None}
        assert net.setup_calls == [] and net.camera_calls == 0


def test_status_says_whether_the_setup_service_answers(env, monkeypatch):
    with make(env, Net(), monkeypatch)[0] as c:
        assert c.get("/api/setup/status", headers=AUTH).json() == {
            "enabled": True, "service": "up", "loaded": {"sam2": True, "geocalib": False, "depth": False}, "pins": {"sam2": {"commit": "abc"}}}
    with make(env, Net(setup_up=False), monkeypatch)[0] as c:
        assert c.get("/api/setup/status", headers=AUTH).json() == {"enabled": True, "service": "down", "loaded": None, "pins": None}


def test_two_stills_activation_does_not_need_a_restart_of_anything(env, monkeypatch):
    # the activation writes only the geometry store; the app is the same object before and after
    net = Net()
    app_http, _ = make(env, net, monkeypatch)
    with app_http as c:
        cid = cam_id(c)
        sid = c.post(f"/api/cameras/{cid}/setup/still", headers=AUTH).json()["id"]
        v = calibrate(c, cid, sid)["version"]
        before = c.get(f"/api/cameras/{cid}/geometry", headers=AUTH).json()
        assert c.post(f"/api/cameras/{cid}/setup/activate", headers=AUTH, json={"version": v}).status_code == 200
        after = c.get(f"/api/cameras/{cid}/geometry", headers=AUTH).json()
        assert before == {"active": None} and after["active"]["ver"] == v
        assert c.get("/api/health").status_code == 200  # still the same, serving process


# --- Phase 2 on the dashboard stream: the worker's "geometry" in the snapshot, only while the flag is on ---------------------------
def publish_geometry(env, geometry=None):
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    snap = {"status": "streaming", "seq": 1, "frame_ts": time.time(), "slots": [], "counts": {"free": 1, "busy": 0, "unknown": 0, "slots": 1}}
    if geometry is not None:
        snap["geometry"] = geometry
    store.publish("w1", snap, JPEG, JPEG)


METRICS = {"version": 3, "valid": True, "confidence": "HIGH", "counts": {"capacity": 4, "occupied": 1, "free": 3, "vehicles": 1, "measured": 1, "beyond_range": 0},
           "vehicles": [], "distances": [], "bays": []}


def test_the_dashboard_stream_carries_metrics_only_while_the_flag_is_on(env, monkeypatch):
    from api.main import state_key

    publish_geometry(env, METRICS)
    with make(env, Net(), monkeypatch)[0] as on:
        assert on.get("/api/status", headers=AUTH).json()["snapshot"]["geometry"] == METRICS
    with make({**env, "CARPARK_GEOMETRY_ENABLED": "0"}, Net(), monkeypatch)[0] as off:
        snap = off.get("/api/status", headers=AUTH).json()["snapshot"]
        assert "geometry" not in snap and snap["counts"]["free"] == 1  # a worker that published it anyway: not passed on
    publish_geometry(env)  # no calibration: the snapshot is as it always was
    with make(env, Net(), monkeypatch)[0] as on:
        assert "geometry" not in on.get("/api/status", headers=AUTH).json()["snapshot"]
    # a new calibration or a change of the bay occupancy is worth sending at once on a throttled stream
    base = {"connected": True, "settings_version": 1, "snapshot": {"enabled": True, "status": "streaming", "counts": {"free": 1}, "slots": []}}
    with_g = lambda g: {**base, "snapshot": {**base["snapshot"], "geometry": g}}  # noqa: E731
    assert state_key(base) != state_key(with_g(METRICS))
    assert state_key(with_g(METRICS)) != state_key(with_g({**METRICS, "version": 4}))
    assert state_key(with_g(METRICS)) != state_key(with_g({**METRICS, "counts": {**METRICS["counts"], "occupied": 2}}))
    assert state_key(with_g(METRICS)) == state_key(with_g({**METRICS, "distances": [{"m": 1.0}]}))  # distances wait for the interval
