"""FastAPI backend (api/main.py): settings round trip and parity with the page's builder, masking, token, versions,
commands, history, frames, the WebSocket and the SPA fallback. No models are loaded (class lists stubbed)."""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import Config, create_app  # noqa: E402
from carparking.history import HistoryDB  # noqa: E402
from carparking.fixed_slots import FixedSlotStore  # noqa: E402
from carparking.live import LiveStore, build_settings  # noqa: E402

URL = "rtsp://admin:secret@cam.local:554/Streaming/Channels/101"
TOKEN = "t0k3n"
BAY1 = {"id": 1, "name": "G1", "poly": [[0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.1, 0.5]]}
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def names_of(weights):
    return ["car", "motorcycle", "bus", "truck"], ["person", "dog"]


@pytest.fixture
def env(tmp_path):
    spa = tmp_path / "dist"
    return {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
            "CARPARK_RTSP_URL": URL, "CARPARK_API_TOKEN": TOKEN, "CARPARK_SPA_DIR": str(spa)}


@pytest.fixture
def client(env):
    with TestClient(create_app(Config(env), names_of=names_of)) as c:
        yield c


def stored(env):
    return LiveStore(env["CARPARK_HISTORY_DB"]).config()


def test_settings_defaults_and_round_trip_with_page_parity(client, env):
    r = client.get("/api/settings").json()
    assert r["version"] == 0 and r["values"]["camera"] == "Camera 1" and r["values"]["stream"] == "main"
    assert r["values"]["hold"] == 15 and r["values"]["cut_right"] == 40 and URL not in str(r)
    r = client.put("/api/settings", headers=AUTH, json={"hold": 20, "stream": "sub", "cut_top": 10, "enabled": True,
                                                        "classes": ["car", "truck"]})
    assert r.status_code == 200, r.text
    v = r.json()["values"]
    assert r.json()["version"] == 1 and v["hold"] == 20 and v["stream"] == "sub" and v["cut_top"] == 10
    cfg, ver = stored(env)
    assert ver == 1 and cfg["enabled"] and cfg["run"]["m5"]["top"] == 0.10 and cfg["run"]["m5"]["hold"] == 20
    # exactly what the Streamlit page stores for these widget values
    assert build_settings(cfg["ui"], ROOT, env["CARPARK_HISTORY_DB"], URL, env["CARPARK_FIXED_SLOTS"], names_of) == cfg
    # a re-save with nothing new is no new version: the worker never restarts for it
    assert client.put("/api/settings", headers=AUTH, json={"hold": 20}).json()["version"] == 1


def test_camera_password_never_leaves_and_masked_value_keeps_it(client, env):
    r = client.put("/api/settings", headers=AUTH, json={"stream": "custom", "custom_url": "rtsp://u:pw@h2/ch1"})
    assert r.json()["values"]["custom_url"] == "rtsp://***@h2/ch1" and "pw" not in r.text
    assert "pw@" not in client.get("/api/settings").text and "secret" not in client.get("/api/settings").text
    client.put("/api/settings", headers=AUTH, json={"custom_url": "rtsp://***@h2/ch1", "hold": 25})  # sent back
    assert stored(env)[0]["run"]["custom_url"] == "rtsp://u:pw@h2/ch1"


def test_writes_need_the_token(client, env):
    assert client.put("/api/settings", json={"hold": 20}).status_code == 401
    assert client.put("/api/settings", headers={"X-API-Token": "wrong"}, json={"hold": 20}).status_code == 401
    assert client.post("/api/commands", json={"action": "reset"}).status_code == 401
    assert client.put("/api/settings", headers={"X-API-Token": TOKEN}, json={"hold": 21}).status_code == 200
    with TestClient(create_app(Config({**env, "CARPARK_API_TOKEN": ""}), names_of=names_of)) as ro:
        assert ro.put("/api/settings", json={"hold": 20}).status_code == 503  # no token configured: read-only
        assert ro.get("/api/settings").status_code == 200


def test_validation_and_version_conflicts(client):
    assert client.put("/api/settings", headers=AUTH, json={"conf": 0.9}).status_code == 422
    assert client.put("/api/settings", headers=AUTH, json={"stream": "4k"}).status_code == 422
    assert client.put("/api/settings", headers=AUTH, json={"detector": "nope"}).status_code == 422
    assert client.put("/api/settings", headers=AUTH, json={"roi": "1,2;3"}).status_code == 422
    v = client.put("/api/settings", headers=AUTH, json={"hold": 30}).json()["version"]
    assert client.put("/api/settings", headers=AUTH, json={"version": v - 1, "hold": 35}).status_code == 409
    assert client.put("/api/settings", headers=AUTH, json={"version": v, "hold": 35}).status_code == 200


def test_options_list_every_choice(client):
    o = client.get("/api/settings/options").json()
    assert o["stream"] == ["main", "sub", "custom"] and o["inference"] == ["1280", "1920", "sahi"]
    assert o["classes"] == ["car", "motorcycle", "bus", "truck"] and o["ranges"]["conf"] == [0.05, 0.5]
    assert o["detector_selected"] in o["detector"] and o["detector"][-1].startswith("🅿️")


def test_commands_go_to_a_running_worker_else_reset_directly(client, env):
    db = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector="yolov8s")
    assert client.post("/api/commands", headers=AUTH, json={"action": "snapshot"}).status_code == 409
    assert client.post("/api/commands", headers=AUTH, json={"action": "reset"}).json() == {"done": True,
                                                                                            "action": "reset"}
    assert client.post("/api/commands", headers=AUTH, json={"action": "explode"}).status_code == 422
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    store.beat("w1", {"status": "streaming"})
    r = client.post("/api/commands", headers=AUTH, json={"action": "snapshot"}).json()
    assert r["action"] == "snapshot" and store.commands(0)[-1] == (r["queued"], "snapshot")
    db.close()


def test_status_frames_and_history(client, env):
    assert client.get("/api/status").json()["connected"] is False
    assert client.get("/api/frame.jpg").status_code == 404
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    store.publish("w1", {"status": "streaming", "seq": 3, "slots": []}, b"\xff\xd8jpeg")
    s = client.get("/api/status").json()
    assert s["connected"] and s["snapshot"]["seq"] == 3
    r = client.get("/api/frame.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content == b"\xff\xd8jpeg"
    client.get("/api/settings")
    scope = client.get("/api/settings").json()["effective"]["db_detector"]
    db = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=scope)
    db.apply({"slot": 1, "state": 1, "time": 100.0, "initial": False})
    assert [e["event"] for e in client.get("/api/events").json()] == ["entry"]
    assert client.get("/api/sessions").json()[0]["slot_id"] == 1
    assert client.get("/api/summary").json()["open_sessions"] == 1
    for t in (200.0, 300.0):  # two newer, closed sessions: the open one stays on top of page 1
        db.apply({"slot": 2, "state": 1, "time": t, "initial": False})
        db.apply({"slot": 2, "state": 0, "time": t + 50, "initial": False})
    page = client.get("/api/sessions/page?size=2").json()
    assert (page["total"], page["open"], page["page"], page["size"], page["pages"]) == (3, 1, 1, 2, 2)
    assert [(x["slot_id"], x["out_time"] is None) for x in page["items"]] == [(1, True), (2, False)]
    assert [x["in_ts"] for x in client.get("/api/sessions/page?size=2&page=2").json()["items"]] == [200.0]
    assert client.get("/api/sessions/page?page=9").json()["items"] == []
    assert client.get("/api/sessions/page?page=0").status_code == 422


def test_history_follows_the_scope_the_worker_writes_to_including_the_auto_grids_version(client, env):
    """camera-1, 2026-10-26: the worker wrote to "<detector> | auto v10" but the pages read the settings' own scope, an older
    one: the History page showed 25 October's rows (3 open) while the live count was different, whatever was clicked."""
    client.get("/api/settings")
    base = client.get("/api/settings").json()["effective"]["db_detector"]
    old = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=base)
    old.apply({"slot": 1, "state": 1, "time": 100.0, "initial": False})  # the old scope: one open session
    assert [x["slot_id"] for x in client.get("/api/sessions/page").json()["items"]] == [1]  # (no worker yet: the settings' scope)
    live = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=f"{base} | auto v10")
    for slot in (7, 9, 9):
        live.con.execute("INSERT INTO history (camera, slot_id, entry_time, status, note, entry_ts) VALUES (?, ?, 'x', 'Occupied', '', ?)",
                         (live.scope, slot, 200.0 + slot))
    live.con.commit()
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    store.publish("w1", {"status": "streaming", "seq": 1, "scope": live.scope}, b"\xff\xd8x")
    page = client.get("/api/sessions/page").json()
    assert (page["total"], page["open"]) == (3, 3) and sorted(x["slot_id"] for x in page["items"]) == [7, 9, 9]
    assert client.get("/api/summary").json()["open_sessions"] == 3
    store.publish("w1", {"status": "streaming", "seq": 2, "scope": "Another camera | yolo"}, b"\xff\xd8x")  # not this camera's: ignored
    assert [x["slot_id"] for x in client.get("/api/sessions/page").json()["items"]] == [1]


def test_websocket_pushes_worker_state_events_and_frames(client, env):
    client.put("/api/settings", headers=AUTH, json={"hold": 20})
    scope = client.get("/api/settings").json()["effective"]["db_detector"]
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    with client.websocket_connect("/ws/stream?frames=1") as ws:
        first = ws.receive_json()
        assert first["type"] == "state" and first["connected"] and first["settings_version"] == 1
        store.publish("w1", {"status": "streaming", "seq": 1}, b"\xff\xd8one")
        msgs = [ws.receive_json(), ws.receive_bytes()]
        assert msgs[0]["snapshot"]["seq"] == 1 and msgs[1] == b"\xff\xd8one"
        # a worker that makes clean frames (the page draws the layers itself): that image is the one sent
        store.publish("w1", {"status": "streaming", "seq": 2, "clean_frames": True}, b"\xff\xd8baked", b"\xff\xd8clean")
        msgs = [ws.receive_json(), ws.receive_bytes()]
        assert msgs[0]["snapshot"]["seq"] == 2 and msgs[1] == b"\xff\xd8clean"
        time.sleep(0.6)  # the hub has seen the history; a new entry is pushed as an event
        HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=scope).apply(
            {"slot": 2, "state": 1, "time": 200.0, "initial": False})
        ev = ws.receive_json()
        assert ev["type"] == "event" and ev["event"] == "entry" and ev["slot_id"] == 2


def test_spa_is_served_with_a_fallback_for_client_routes(env, tmp_path):
    with TestClient(create_app(Config(env), names_of=names_of)) as c:
        assert "not built yet" in c.get("/").text
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    with TestClient(create_app(Config(env), names_of=names_of)) as c:
        assert c.get("/").text == "<div id=root></div>" and c.get("/dashboard/live").text == "<div id=root></div>"
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/api/nope").status_code == 404 and c.get("/../h.db").status_code in (404, 200)
        assert "slot" not in c.get("/../h.db").text


def test_viewer_token_guards_every_read_but_health(env):
    view = "v1ew"
    with TestClient(create_app(Config({**env, "CARPARK_VIEW_TOKEN": view}), names_of=names_of)) as c:
        assert c.get("/api/health").status_code == 200
        for path in ("/api/status", "/api/settings", "/api/sessions", "/api/events", "/api/summary", "/api/video",
                     "/api/frame.jpg", "/api/sessions.csv", "/api/settings/options"):
            assert c.get(path).status_code == 401, path
        assert c.get("/api/status", params={"token": view}).status_code == 200  # <img src=...?token=>
        assert c.get("/api/status", headers={"Authorization": f"Bearer {view}"}).status_code == 200
        assert c.get("/api/status", headers=AUTH).status_code == 200  # the admin token may view too
        assert c.put("/api/settings", headers={"Authorization": f"Bearer {view}"}, json={"hold": 20}).status_code == 401
        assert c.get("/api/auth", params={"token": view}).json() == {"view": True, "admin": False,
                                                                     "view_required": True, "writes_enabled": True}
        assert c.get("/api/auth").json()["view"] is False
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect("/ws/stream") as ws:
                ws.receive_json()
        assert e.value.code == 4401
        with c.websocket_connect(f"/ws/stream?token={view}") as ws:
            assert ws.receive_json()["type"] == "state"


def test_webrtc_signalling_is_proxied_to_go2rtc(env):
    import httpx
    seen = []

    def go2rtc(request: httpx.Request):
        seen.append((str(request.url), request.headers["content-type"], request.content))
        return httpx.Response(201, content=b"v=0 answer", headers={"Content-Type": "application/sdp"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(go2rtc))
    cfg = Config({**env, "CARPARK_VIEW_TOKEN": "v", "CARPARK_GO2RTC_INTERNAL": "http://go2rtc:1984"})
    with TestClient(create_app(cfg, names_of=names_of, http=http)) as c:
        assert c.post("/api/webrtc?src=parking", content=b"v=0 offer").status_code == 401  # no viewer token
        r = c.post("/api/webrtc?src=parking&token=v", content=b"v=0 offer", headers={"Content-Type": "application/sdp"})
        assert r.status_code == 201 and r.content == b"v=0 answer" and r.headers["content-type"] == "application/sdp"
        assert seen == [("http://go2rtc:1984/api/webrtc?src=parking", "application/sdp", b"v=0 offer")]
        assert c.post("/api/webrtc?src=../api/config&token=v", content=b"x").status_code == 404
        assert c.get("/api/video?token=v").json()["default"] == "parking_hd"


def test_history_csv_downloads(client, env):
    scope = client.get("/api/settings").json()["effective"]["db_detector"]
    HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=scope).apply(
        {"slot": 3, "state": 1, "time": 100.0, "initial": False})
    r = client.get("/api/sessions.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv") and "slot_id" in r.text
    assert "entry" in client.get("/api/events.csv").text


def test_model_variant_is_a_worker_option_not_a_settings_version(client, env):
    assert client.get("/api/settings").json()["values"]["model_variant"] == "auto"
    assert client.get("/api/settings/options").json()["model_variant"] == ["auto", "default", "yolo11m"]
    assert client.put("/api/settings", json={"model_variant": "yolo11m"}).status_code == 401
    assert client.put("/api/settings", headers=AUTH, json={"model_variant": "yolo99x"}).status_code == 422
    r = client.put("/api/settings", headers=AUTH, json={"model_variant": "yolo11m"})
    assert r.status_code == 200 and r.json()["values"]["model_variant"] == "yolo11m" and r.json()["version"] == 0
    assert LiveStore(env["CARPARK_HISTORY_DB"]).option("model_variant") == "yolo11m"
    assert stored(env)[1] == 0  # the page-shared settings untouched: Streamlit never rewrites the choice away
    r = client.put("/api/settings", headers=AUTH, json={"model_variant": "default", "hold": 20})
    assert r.json()["values"]["model_variant"] == "default" and r.json()["version"] == 1


def test_fixed_slot_map_drawn_in_the_dashboard(client, env):
    bay = {"id": 1, "name": "A1", "poly": [[0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.1, 0.5]]}
    r = client.get("/api/fixed-slots").json()
    assert r["camera"] == "Camera 1" and r["slots"] == [] and r["in_use"] is False
    assert client.put("/api/fixed-slots", json={"camera": "Camera 1", "slots": [bay]}).status_code == 401
    # Dynamic mode (the default): the editor is read-only and the API refuses drawn slots
    assert r["mode"] == "dynamic" and r["editable"] is False and r["locked"].startswith("Dynamic Mode automatically")
    r = client.put("/api/fixed-slots", headers=AUTH, json={"camera": "Camera 1", "slots": [bay]})
    assert r.status_code == 409 and "Switch to Fixed Mode" in r.json()["detail"]
    assert client.get("/api/fixed-slots").json()["slots"] == [] and stored(env)[1] == 0  # nothing saved or rebuilt
    # switch the settings to fixed mode (allowed with no map yet: the worker idles), then save a map: the worker's
    # settings carry it at once
    client.put("/api/settings", headers=AUTH, json={"slot_mode": "fixed"})
    got = client.get("/api/fixed-slots").json()
    assert got["mode"] == "fixed" and got["editable"] is True and got["locked"] is None
    bad = {"id": 2, "name": "B", "poly": [[0.1, 0.1], [1.4, 0.1], [0.2, 0.3]]}  # outside the frame
    r = client.put("/api/fixed-slots", headers=AUTH, json={"camera": "Camera 1", "slots": [bay, bad]})
    assert r.status_code == 422 and "slot 2" in r.json()["detail"]
    r = client.put("/api/fixed-slots", headers=AUTH, json={"camera": "Camera 1", "slots": [bay]}).json()
    assert r["slots"] == [bay] and r["settings_version"] == 2
    cfg, ver = stored(env)
    assert ver == 2 and cfg["run"]["fixed"] == [bay] and cfg["run"]["fixed_mode"]
    # the same file the Streamlit editor uses, and exactly what the page would store
    assert build_settings(cfg["ui"], ROOT, env["CARPARK_HISTORY_DB"], URL, env["CARPARK_FIXED_SLOTS"], names_of) == cfg
    got = client.get("/api/fixed-slots").json()
    assert got["slots"] == [bay] and got["in_use"] and got["cameras"] == ["Camera 1"] and got["updated"]
    # another camera's map: saved, the running settings untouched
    r = client.put("/api/fixed-slots", headers=AUTH, json={"camera": "Gate", "slots": [bay]}).json()
    assert r["settings_version"] == 2 and client.get("/api/fixed-slots?camera=Gate").json()["slots"] == [bay]


def test_multi_camera_registry_isolation_and_streams(env, tmp_path):
    import httpx
    calls = []

    def go2rtc(request: httpx.Request):
        calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/api/webrtc":
            return httpx.Response(201, content=b"v=0 answer", headers={"Content-Type": "application/sdp"})
        return httpx.Response(200, json={})

    http = httpx.AsyncClient(transport=httpx.MockTransport(go2rtc))
    with TestClient(create_app(Config({**env, "CARPARK_GO2RTC_INTERNAL": "http://g"}), names_of=names_of, http=http)) as c:
        cams = c.get("/api/cameras").json()
        assert [(x["id"], x["name"], x["primary"]) for x in cams] == [("camera-1", "Camera 1", True)]
        assert cams[0]["streams"] == {"hd": "parking_hd", "sd": None, "native": "parking_native"} and "secret" not in str(cams)
        new = {"name": "Gate", "rtsp_url": "rtsp://u:pw@cam2/Streaming/Channels/101",
               "rtsp_sub_url": "rtsp://u:pw@cam2/Streaming/Channels/102", "metadata": {"location": "east gate"}}
        assert c.post("/api/cameras", json=new).status_code == 401
        r = c.post("/api/cameras", headers=AUTH, json=new)
        assert r.status_code == 201, r.text
        gate = r.json()
        assert gate["id"] == "gate" and gate["rtsp_url"] == "rtsp://***@cam2/Streaming/Channels/101" and "pw" not in r.text
        assert gate["status"]["enabled"] is False and gate["status"]["slot_mode"] == "dynamic"  # starts off
        assert {p["name"] for m, path, p in calls if m == "PUT"} == {"gate_hd", "gate_sd", "gate_native"}
        assert c.post("/api/cameras", headers=AUTH, json={**new, "id": "gate-2"}).status_code == 422  # name taken
        # isolated settings, files and history
        c.put("/api/settings?cam=gate", headers=AUTH, json={"hold": 40, "slot_mode": "dynamic", "enabled": True})
        assert c.get("/api/settings?cam=gate").json()["values"]["hold"] == 40
        assert c.get("/api/settings").json()["values"]["hold"] == 15  # the primary camera untouched
        assert c.get("/api/settings?cam=gate").json()["values"]["camera"] == "Gate"
        assert c.put("/api/settings?cam=gate", headers=AUTH, json={"camera": "Other"}).status_code == 422
        assert (tmp_path / "cameras" / "gate" / "parking_history.db").exists()
        assert c.put("/api/fixed-slots?cam=gate", headers=AUTH, json={"slots": [BAY1]}).status_code == 409  # dynamic
        c.put("/api/settings?cam=gate", headers=AUTH, json={"slot_mode": "fixed"})
        assert c.put("/api/fixed-slots?cam=gate", headers=AUTH, json={"slots": [BAY1]}).json()["cam"] == "gate"
        assert c.get("/api/fixed-slots?cam=gate").json()["slots"] == [BAY1]
        assert c.get("/api/fixed-slots").json()["slots"] == []  # Camera 1's map untouched
        assert FixedSlotStore(tmp_path / "cameras" / "gate" / "fixed_slots.json").load("Gate") == [BAY1]
        scope = c.get("/api/settings?cam=gate").json()["effective"]["db_detector"]
        HistoryDB(tmp_path / "cameras" / "gate" / "parking_history.db", "Gate", detector=scope).apply(
            {"slot": 7, "state": 1, "time": 50.0, "initial": False})
        assert [e["slot_id"] for e in c.get("/api/events?cam=gate").json()] == [7]
        assert c.get("/api/events").json() == []
        store = LiveStore(tmp_path / "cameras" / "gate" / "parking_history.db")
        store.claim("gate-worker")
        store.beat("gate-worker", {"status": "streaming", "enabled": True})
        assert c.get("/api/status?cam=gate").json()["connected"] and not c.get("/api/status").json()["connected"]
        with c.websocket_connect("/ws/stream?cam=gate") as ws:
            assert ws.receive_json()["snapshot"]["status"] == "streaming"
        assert c.get("/api/status?cam=nope").status_code == 404
        # video: the fullscreen stream of the new camera, registered in go2rtc again if go2rtc restarted
        assert c.get("/api/video?cam=gate").json()["grid"] == "gate_sd"
        r = c.post("/api/webrtc?src=gate_hd", content=b"v=0 offer", headers={"Content-Type": "application/sdp"})
        assert r.status_code == 201 and calls[-1][1] == "/api/webrtc" and calls[-1][2] == {"src": "gate_hd"}
        # edit, remove
        same = c.patch("/api/cameras/gate", headers=AUTH, json={"rtsp_url": gate["rtsp_url"], "sort_order": 3}).json()
        assert same["sort_order"] == 3 and same["rtsp_url"] == gate["rtsp_url"]  # the masked URL kept the real one
        assert c.delete("/api/cameras/camera-1", headers=AUTH).status_code == 409  # the primary stays
        assert c.delete("/api/cameras/gate", headers=AUTH).json() == {"removed": "gate"}
        assert ("DELETE", "/api/streams", {"src": "gate_hd"}) in calls
        assert [x["id"] for x in c.get("/api/cameras").json()] == ["camera-1"]
        assert (tmp_path / "cameras" / "gate" / "parking_history.db").exists()  # its history stays on disk


def test_camera_display_name_is_editable_its_history_stays(client, env):
    c = client
    assert c.get("/api/cameras").json()[0]["display"] == "Camera 1"
    r = c.patch("/api/cameras/camera-1", headers=AUTH, json={"label": "Main Parking Area"})
    assert r.status_code == 200 and r.json()["display"] == "Main Parking Area" and r.json()["name"] == "Camera 1"
    assert c.patch("/api/cameras/camera-1", json={"label": "x"}).status_code == 401
    # the history key is unchanged: same settings camera, same scope
    assert c.get("/api/settings").json()["values"]["camera"] == "Camera 1"
    gate = c.post("/api/cameras", headers=AUTH, json={"name": "Gate", "label": "East Gate",
                                                      "rtsp_url": "rtsp://u:pw@cam2/101"}).json()
    assert gate["name"] == "Gate" and gate["display"] == "East Gate"


def test_history_and_analytics_fall_back_to_sqlite_without_postgres(client, env):
    """No PostgreSQL (cameras.json registry): History & Analytics come from the camera's SQLite file, labelled so."""
    scope = client.get("/api/settings").json()["effective"]["db_detector"]
    db = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=scope)
    now = time.time()
    db.apply({"slot": 1, "state": 1, "time": now - 3600, "initial": False})
    db.apply({"slot": 1, "state": 0, "time": now - 600, "initial": False})
    r = client.get("/api/sessions")
    assert r.headers["x-history-source"] == "sqlite" and r.json()[0]["duration_min"] == 50.0
    assert client.get("/api/summary").json()["source"] == "sqlite"
    a = client.get("/api/analytics?days=1").json()
    assert a["source"] == "sqlite" and a["totals"]["entries"] == 1 and a["totals"]["avg_min"] == 50.0
    assert len(a["per_hour"]) == 24 and [d["sessions"] for d in a["durations"]][1] == 1 and a["occupancy"] == []
    assert client.get("/api/analytics?days=1&scope=current").json()["totals"]["closed"] == 1
    assert client.get("/api/analytics?scope=bad").status_code == 422
    ov = client.get("/api/analytics/overview").json()
    assert ov["source"] == "sqlite" and ov["cameras"][0]["entries"] == 1 and ov["cameras"][0]["exits"] == 1
    assert client.get("/api/camera-events").json() == []
    sync = client.get("/api/history/sync").json()
    assert sync["backend"] == "sqlite" and sync["cameras"][0]["sqlite"] == {"sessions": 1, "events": 2}


def test_tokens_in_urls_are_masked_in_the_server_logs(client):
    import logging

    assert client is not None  # create_app installed the filter
    rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d', (
        "1.2.3.4:5", "GET", "/api/frame.jpg?raw=1&token=s3cret&cam=gate", "1.1", 200), None)
    for f in logging.getLogger("uvicorn.access").filters:
        f.filter(rec)
    assert "s3cret" not in rec.getMessage() and "token=***&cam=gate" in rec.getMessage()
    ws = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, '%s - "WebSocket %s" [accepted]',
                           ("1.2.3.4:5", "/ws/stream?token=s3cret"), None)
    for f in logging.getLogger("uvicorn.error").filters:
        f.filter(ws)
    assert ws.getMessage().endswith('"WebSocket /ws/stream?token=***" [accepted]')


def test_websocket_every_throttles_routine_updates_but_sends_changes_at_once(client, env):
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    counts = {"free": 3, "busy": 1, "unknown": 0}
    with client.websocket_connect("/ws/stream?every=60") as ws:
        assert ws.receive_json()["type"] == "state"  # the current state at once (Refresh data reconnects for this)
        store.publish("w1", {"status": "streaming", "seq": 1, "counts": counts}, b"\xff\xd8one")
        time.sleep(0.6)
        store.publish("w1", {"status": "streaming", "seq": 2, "counts": counts}, b"\xff\xd8two")  # routine: held
        time.sleep(0.6)
        store.publish("w1", {"status": "streaming", "seq": 3, "counts": {**counts, "free": 2, "busy": 2}}, b"\xff\xd8x")
        got = ws.receive_json()
        # seq 1 changed the status / counts (from nothing): sent; seq 2 only a new frame number: skipped; seq 3 a car
        # parked: sent at once, without waiting for the 60 s
        assert got["snapshot"]["seq"] == 1
        assert ws.receive_json()["snapshot"]["seq"] == 3


def test_display_thresholds_are_worker_options_with_defaults(client, env):
    values = client.get("/api/settings").json()["values"]
    assert (values["hidden_ground"], values["ground_share"], values["vehicle_iou"]) == (0.8, 0.4, 0.2)
    opts = client.get("/api/settings/options").json()
    assert opts["tuning_defaults"] == {"hidden_ground": 0.8, "ground_share": 0.4, "vehicle_iou": 0.2, "bay_share": 0.5,
                                       "pin_reach": 0.25}
    assert opts["ranges"]["hidden_ground"] == [0.5, 1.0] and opts["ranges"]["pin_reach"] == [0.0, 0.5]
    assert client.put("/api/settings", json={"hidden_ground": 0.7}).status_code == 401
    assert client.put("/api/settings", headers=AUTH, json={"hidden_ground": 0.2}).status_code == 422  # out of range
    r = client.put("/api/settings", headers=AUTH, json={"hidden_ground": 0.7})
    assert r.status_code == 200 and r.json()["values"]["hidden_ground"] == 0.7 and r.json()["version"] == 0
    r = client.put("/api/settings", headers=AUTH, json={"ground_share": 0.5}).json()
    assert (r["values"]["hidden_ground"], r["values"]["ground_share"], r["values"]["vehicle_iou"]) == (0.7, 0.5, 0.2)
    assert json.loads(LiveStore(env["CARPARK_HISTORY_DB"]).option("tuning")) == {
        "hidden_ground": 0.7, "ground_share": 0.5, "vehicle_iou": 0.2, "bay_share": 0.5, "pin_reach": 0.25}
    r = client.put("/api/settings", headers=AUTH, json={"vehicle_iou": 0.8, "bay_share": 0.8}).json()
    assert (r["values"]["vehicle_iou"], r["values"]["bay_share"]) == (0.8, 0.8)  # 80 % is stored as typed, not clamped to 60 %
    assert client.put("/api/settings", headers=AUTH, json={"pin_reach": 0.9}).status_code == 422  # out of range
    r = client.put("/api/settings", headers=AUTH, json={"pin_reach": 0.4}).json()
    assert r["values"]["pin_reach"] == 0.4 and json.loads(LiveStore(env["CARPARK_HISTORY_DB"]).option("tuning"))["pin_reach"] == 0.4
    assert client.put("/api/settings", headers=AUTH, json={"pin_reach": 0}).json()["values"]["pin_reach"] == 0.0  # 0 = strictly inside
    assert stored(env)[1] == 0  # the page-shared settings (Streamlit) untouched


def test_test_slots_preview_is_read_only(client, env):
    bay = {"id": 1, "name": "A1", "poly": [[0.1, 0.5], [0.3, 0.5], [0.3, 0.8], [0.1, 0.8]]}
    other = {"id": 2, "name": "A2", "poly": [[0.6, 0.5], [0.8, 0.5], [0.8, 0.8], [0.6, 0.8]]}
    body = {"slots": [bay, other]}
    # no worker and no video service in the test: no frame to grab on demand
    assert client.post("/api/fixed-slots/preview", json=body).status_code == 502
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    car = {"label": "car", "conf": 0.9, "src": "", "kind": "vehicle", "box": [0.12, 0.4, 0.28, 0.75]}
    store.publish("w1", {"status": "streaming", "seq": 1, "frame_ts": time.time() - 2, "detections": [car]}, b"\xff\xd8x")
    before_map, before_settings = client.get("/api/fixed-slots").json(), stored(env)
    r = client.post("/api/fixed-slots/preview", json=body)
    assert r.status_code == 200, r.text
    got = r.json()
    assert [(x["id"], x["state"]) for x in got["results"]] == [(1, "BUSY"), (2, "FREE")]
    assert got["rule"] == "ground" and got["vehicles"] == [car["box"]] and 1 <= got["frame_age"] < 10
    assert got["source"] == "worker" and got["image"] is None  # monitoring on: the worker's fresh detections
    one = client.post("/api/fixed-slots/preview", json={**body, "only": [2]}).json()
    assert [(x["id"], x["state"]) for x in one["results"]] == [(2, "FREE")]
    bad = {"slots": [{"id": 1, "poly": [[0.1, 0.1], [1.4, 0.1], [0.2, 0.3]]}]}
    assert client.post("/api/fixed-slots/preview", json=bad).status_code == 422
    # read-only: the saved map, the settings and their version are exactly as before
    assert client.get("/api/fixed-slots").json() == before_map and stored(env) == before_settings



def test_test_slots_work_with_monitoring_off(env, monkeypatch):
    # monitoring off (no worker detections): one frame is grabbed from go2rtc now and run through the camera's detector
    # in the API (slot_preview.detect_on_demand, stubbed here: no model in the tests); the frame comes back as "image"
    import cv2
    import httpx
    import numpy as np
    from carparking import slot_preview

    jpeg = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))[1].tobytes()
    asked = []

    def go2rtc(request: httpx.Request):
        asked.append((request.url.path, dict(request.url.params)))
        return httpx.Response(200, content=jpeg, headers={"Content-Type": "image/jpeg"})

    seen = []
    car = {"label": "car", "conf": 0.8, "src": "", "kind": "vehicle", "box": [0.62, 0.3, 0.78, 0.7]}
    monkeypatch.setattr(slot_preview, "detect_on_demand", lambda run, frame, root: seen.append(frame.shape) or [car])
    http = httpx.AsyncClient(transport=httpx.MockTransport(go2rtc))
    with TestClient(create_app(Config({**env, "CARPARK_GO2RTC_INTERNAL": "http://g"}), names_of=names_of, http=http)) as c:
        bay = {"id": 1, "name": "A1", "poly": [[0.1, 0.5], [0.3, 0.5], [0.3, 0.8], [0.1, 0.8]]}
        other = {"id": 2, "name": "A2", "poly": [[0.6, 0.5], [0.8, 0.5], [0.8, 0.8], [0.6, 0.8]]}
        before_map, before_settings = c.get("/api/fixed-slots").json(), stored(env)
        r = c.post("/api/fixed-slots/preview", json={"slots": [bay, other]})
        assert r.status_code == 200, r.text
        got = r.json()
        assert got["source"] == "on-demand" and [(x["id"], x["state"]) for x in got["results"]] == [(1, "FREE"), (2, "BUSY")]
        assert got["image"].startswith("data:image/jpeg;base64,") and got["frame_age"] < 5 and seen == [(90, 160, 3)]
        assert asked[0] == ("/api/frame.jpeg", {"src": "parking_native"})  # the camera's full-resolution stream
        assert c.post("/api/fixed-slots/preview", json={"slots": [bay, other], "only": [1]}).json()["results"][0]["state"] == "FREE"
        assert c.get("/api/fixed-slots").json() == before_map and stored(env) == before_settings  # read-only


def test_slot_editor_snapshot_is_captured_once_and_tested_without_any_stream(env, monkeypatch):
    # the editor's still: the camera's own HTTP snapshot (Hikvision ISAPI, digest auth from the RTSP URL), kept in
    # memory; Test slots detects on it once and reuses that; go2rtc is only the fallback, never touched here
    import cv2
    import httpx
    import numpy as np
    from carparking import slot_preview

    jpeg = cv2.imencode(".jpg", np.zeros((144, 256, 3), np.uint8))[1].tobytes()
    asked, camera_up = [], [True]

    def net(request: httpx.Request):
        asked.append((request.url.host, request.url.path))
        if request.url.path.startswith("/ISAPI/"):
            return httpx.Response(200 if camera_up[0] else 401, content=jpeg if camera_up[0] else b"",
                                  headers={"Content-Type": "image/jpeg"})
        return httpx.Response(200, content=b"")  # go2rtc answering with an empty body (its "wrong response on SETUP")

    detected = []
    car = {"label": "car", "conf": 0.8, "src": "", "kind": "vehicle", "box": [0.62, 0.3, 0.78, 0.7]}
    monkeypatch.setattr(slot_preview, "detect_on_demand", lambda run, frame, root: detected.append(frame.shape) or [car])
    http = httpx.AsyncClient(transport=httpx.MockTransport(net))
    with TestClient(create_app(Config({**env, "CARPARK_GO2RTC_INTERNAL": "http://g"}), names_of=names_of, http=http)) as c:
        r = c.post("/api/fixed-slots/snapshot")
        assert r.status_code == 200, r.text
        still = r.json()
        assert (still["source"], still["width"], still["height"]) == ("camera", 256, 144)
        assert still["image"].startswith("data:image/jpeg;base64,") and len(still["id"]) == 32
        assert asked == [("cam.local", "/ISAPI/Streaming/channels/101/picture")]  # no video stream asked
        bay = {"id": 1, "name": "A1", "poly": [[0.1, 0.5], [0.3, 0.5], [0.3, 0.8], [0.1, 0.8]]}
        other = {"id": 2, "name": "A2", "poly": [[0.6, 0.5], [0.8, 0.5], [0.8, 0.8], [0.6, 0.8]]}
        before_map, before_settings = c.get("/api/fixed-slots").json(), stored(env)
        got = c.post("/api/fixed-slots/preview", json={"slots": [bay, other], "snapshot": still["id"]}).json()
        assert got["source"] == "snapshot" and [(x["id"], x["state"]) for x in got["results"]] == [(1, "FREE"), (2, "BUSY")]
        assert got["frame_ts"] == still["ts"] and got["image"] is None
        one = c.post("/api/fixed-slots/preview", json={"slots": [bay, other], "only": [2], "snapshot": still["id"]}).json()
        assert [(x["id"], x["state"]) for x in one["results"]] == [(2, "BUSY")]
        assert detected == [(144, 256, 3)] and len(asked) == 1  # detected once per still; no further network
        assert c.post("/api/fixed-slots/preview", json={"slots": [bay], "snapshot": "f" * 32}).status_code == 410
        assert c.get("/api/fixed-slots").json() == before_map and stored(env) == before_settings  # read-only
        # the camera's snapshot fails and go2rtc gives empty frames: the worker's last frame, if any; else 502
        camera_up[0] = False
        r = c.post("/api/fixed-slots/snapshot")
        assert r.status_code == 502 and "Refresh snapshot" in r.json()["detail"]
        store = LiveStore(env["CARPARK_HISTORY_DB"])
        store.claim("w1")
        store.publish("w1", {"status": "streaming", "seq": 1, "frame_ts": time.time(), "slots": []}, jpeg, jpeg)
        r = c.post("/api/fixed-slots/snapshot")
        assert r.status_code == 200 and r.json()["source"] == "worker"
        ids = [c.post("/api/fixed-slots/snapshot").json()["id"] for _ in range(3)]  # only the newest 3 are kept
        assert c.post("/api/fixed-slots/preview", json={"slots": [bay], "snapshot": still["id"]}).status_code == 410
        assert c.post("/api/fixed-slots/preview", json={"slots": [bay], "snapshot": ids[-1]}).status_code == 200
        # Fixed mode: the detection starts with the snapshot (warm), Test slots uses that run, not a second one
        assert c.put("/api/settings", json={"slot_mode": "fixed"}, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        n = len(detected)
        warm = c.post("/api/fixed-slots/snapshot").json()["id"]
        for _ in range(50):
            if len(detected) > n:
                break
            time.sleep(0.05)
        assert len(detected) == n + 1  # detected before any Test
        assert c.post("/api/fixed-slots/preview", json={"slots": [bay, other], "snapshot": warm}).json()["source"] == "snapshot"
        assert len(detected) == n + 1


def test_auto_pins_round_trip_reach_the_worker_and_need_the_token(client, env):
    pin = {"id": 1, "name": "Far", "poly": [[0.9, 0.05], [0.95, 0.05], [0.95, 0.1], [0.9, 0.1]]}
    r = client.get("/api/auto-pins").json()
    assert r["expected"] is None and r["pins"] == [] and r["max_extrapolated"] == 3
    assert client.put("/api/auto-pins", json={"expected": 11, "pins": [pin]}).status_code == 401
    before = stored(env)
    r = client.put("/api/auto-pins", headers=AUTH, json={"expected": 11, "pins": [pin]})
    assert r.status_code == 200, r.text
    assert r.json()["expected"] == 11 and r.json()["pins"] == [pin]
    assert client.get("/api/auto-pins").json()["pins"] == [pin]
    assert json.loads(LiveStore(env["CARPARK_HISTORY_DB"]).option("auto_grid")) == {"expected": 11, "pins": [pin]}  # the worker's copy
    assert stored(env) == before  # no settings version: the worker never restarts for it
    assert client.put("/api/auto-pins", headers=AUTH, json={"expected": 0}).status_code == 422
    assert client.put("/api/auto-pins", headers=AUTH, json={"expected": 11, "pins": [{"id": 1, "poly": [[0, 0]]}]}).status_code == 422
    assert client.get("/api/auto-pins").json()["expected"] == 11  # a refused save changes nothing
    assert client.put("/api/auto-pins", headers=AUTH, json={}).json()["pins"] == []  # nothing sent clears both
    assert LiveStore(env["CARPARK_HISTORY_DB"]).option("auto_grid") == ""


def test_overflow_switch_sets_both_worker_options_and_needs_the_token(client, env):
    live = LiveStore(env["CARPARK_HISTORY_DB"])
    assert client.get("/api/overflow").json() == {"enabled": True, "auto_dynamic": True, "slot_split": True}  # the worker's defaults
    assert client.put("/api/overflow", json={"enabled": False}).status_code == 401
    before = stored(env)
    r = client.put("/api/overflow", headers=AUTH, json={"enabled": False})
    assert r.status_code == 200, r.text
    assert r.json() == {"enabled": False, "auto_dynamic": False, "slot_split": False}
    assert (live.option("auto_dynamic"), live.option("slot_split")) == ("0", "0")
    assert stored(env) == before  # no settings version: the worker never restarts for it
    live.set_option("slot_split", "1")  # set elsewhere (e.g. by hand): only both on is "enabled"
    assert client.get("/api/overflow").json() == {"enabled": False, "auto_dynamic": False, "slot_split": True}
    assert client.put("/api/overflow", headers=AUTH, json={"enabled": True}).json()["enabled"] is True
    assert (live.option("auto_dynamic"), live.option("slot_split")) == ("1", "1")
    assert client.put("/api/overflow", headers=AUTH, json={}).status_code == 422


def test_history_and_analytics_answers_are_never_cached(client, env):
    client.get("/api/settings")
    for path in ("/api/sessions", "/api/sessions/page", "/api/sessions/page?all=1", "/api/events", "/api/summary", "/api/analytics",
                 "/api/analytics/overview", "/api/history/sync", "/api/camera-events", "/api/sessions.csv", "/api/events.csv"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["cache-control"] == "no-store", path


def two_scopes(client, env):
    """The camera's settings scope with one old open session, and the worker's auto-grid scope with two."""
    client.get("/api/settings")
    base = client.get("/api/settings").json()["effective"]["db_detector"]
    old = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=f"{base} | auto v3")
    old.apply({"slot": 1, "state": 1, "time": 100.0, "initial": False})
    now = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=f"{base} | auto v10")
    for slot in (7, 9):
        now.apply({"slot": slot, "state": 1, "time": 200.0 + slot, "initial": False})
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    store.publish("w1", {"status": "streaming", "seq": 1, "scope": now.scope}, b"\xff\xd8x")
    return old, now


def test_sessions_page_all_lists_every_scope_of_the_camera_with_its_scope(client, env):
    old, now = two_scopes(client, env)
    page = client.get("/api/sessions/page").json()
    assert sorted(x["slot_id"] for x in page["items"]) == [7, 9] and all("scope" not in x for x in page["items"])
    page = client.get("/api/sessions/page?all=1").json()
    assert (page["total"], page["open"], page["pages"]) == (3, 3, 1)
    assert {(x["slot_id"], x["scope"].rsplit(" | ", 1)[1]) for x in page["items"]} == {(1, "auto v3"), (7, "auto v10"), (9, "auto v10")}
    small = client.get("/api/sessions/page?all=1&size=2&page=2").json()
    assert (small["total"], len(small["items"]), small["pages"]) == (3, 1, 2)
    assert client.get("/api/sessions/page?all=0").json()["total"] == 2


def test_close_stale_command_counts_first_then_closes_other_scopes_only(client, env):
    old, now = two_scopes(client, env)
    body = {"action": "close_stale", "dry_run": True}
    assert client.post("/api/commands", json=body).status_code == 401  # admin only
    r = client.post("/api/commands", headers=AUTH, json=body).json()
    assert r["done"] and r["dry_run"] and r["sessions"] == 1 and list(r["scopes"].values()) == [1]
    assert len(old.sessions(open_only=True)) == 1  # a dry run closes nothing
    r = client.post("/api/commands", headers=AUTH, json={"action": "close_stale"}).json()
    assert r["done"] and not r["dry_run"] and r["sessions"] == 1
    assert old.sessions(open_only=True) == [] and old.sessions()[0]["note"].startswith("closed by the admin")
    assert len(now.sessions(open_only=True)) == 2  # the scope in use is the worker's: untouched
    assert client.post("/api/commands", headers=AUTH, json={"action": "close_stale"}).json()["sessions"] == 0
    page = client.get("/api/sessions/page?all=1").json()
    assert page["total"] == 3 and page["open"] == 2  # the closed session is still listed


def test_history_sync_reports_the_workers_check_of_its_history(client, env):
    client.get("/api/settings")
    assert client.get("/api/history/sync").json()["cameras"][0]["live"] is None  # no worker: nothing to report
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    store.claim("w1")
    check = {"expected": 4, "open": 3, "mismatch_s": 42.0, "error": ""}
    store.publish("w1", {"status": "streaming", "seq": 1, "sessions": check}, b"\xff\xd8x")
    assert client.get("/api/history/sync").json()["cameras"][0]["live"] == check


def test_spillage_and_obstacle_filter_are_worker_options_off_by_default(client, env):
    values = client.get("/api/settings").json()["values"]
    assert values["spillage"] is False and values["obstacle_filter"] is False
    assert client.put("/api/settings", json={"spillage": True}).status_code == 401  # writes need the admin token
    assert client.put("/api/settings", headers=AUTH, json={"spillage": "maybe"}).status_code == 422
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    r = client.put("/api/settings", headers=AUTH, json={"spillage": True})
    assert r.status_code == 200 and r.json()["values"]["spillage"] is True and r.json()["values"]["obstacle_filter"] is False
    assert r.json()["version"] == 0 and stored(env)[1] == 0  # a worker option: the page-shared settings and their version are untouched
    assert store.option("spillage") == "1" and store.option("obstacle_filter", "0") == "0"
    r = client.put("/api/settings", headers=AUTH, json={"obstacle_filter": True, "spillage": False})
    assert (r.json()["values"]["spillage"], r.json()["values"]["obstacle_filter"]) == (False, True)
    assert store.option("spillage") == "0" and store.option("obstacle_filter") == "1"
    assert client.get("/api/settings").json()["values"]["obstacle_filter"] is True  # persisted: a fresh read says so
    r = client.put("/api/settings", headers=AUTH, json={"spillage": None})  # an explicit null changes nothing
    assert r.json()["values"]["obstacle_filter"] is True and r.json()["values"]["spillage"] is False
    r = client.put("/api/settings", headers=AUTH, json={"spillage": True, "hold": 20})  # mixed with a page setting: that one versions
    assert r.json()["version"] == 1 and r.json()["values"]["hold"] == 20 and r.json()["values"]["spillage"] is True
    assert stored(env)[1] == 1 and store.option("spillage") == "1" and store.option("obstacle_filter") == "1"
    assert "spillage" not in stored(env)[0]["ui"] and "spillage" not in stored(env)[0]["run"]  # never written into the shared settings


def test_websocket_carries_notices_to_the_ticker_and_events_default_stays_movements(client, env):
    client.put("/api/settings", headers=AUTH, json={"hold": 20})
    scope = client.get("/api/settings").json()["effective"]["db_detector"]
    LiveStore(env["CARPARK_HISTORY_DB"]).claim("w1")
    db = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=scope)
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "state"
        time.sleep(0.6)  # the hub has seen the history
        db.log_notice(3, "encroached", 300.0, "by A2, 41 % of the bay: free but not usable")
        ev = ws.receive_json()
        assert ev["type"] == "event" and ev["event"] == "encroached" and ev["slot_id"] == 3 and ev["session_id"] is None
    assert client.get("/api/events").json() == []  # a client that asks for movements sees none
    assert [e["event"] for e in client.get("/api/events?kinds=notices").json()] == ["encroached"]


def test_blocked_objects_is_a_worker_option_off_by_default(client, env):
    values = client.get("/api/settings").json()["values"]
    assert values["blocked_objects"] is False
    assert client.put("/api/settings", json={"blocked_objects": True}).status_code == 401
    assert client.put("/api/settings", headers=AUTH, json={"blocked_objects": "maybe"}).status_code == 422
    store = LiveStore(env["CARPARK_HISTORY_DB"])
    r = client.put("/api/settings", headers=AUTH, json={"blocked_objects": True})
    assert r.status_code == 200 and r.json()["values"]["blocked_objects"] is True and r.json()["values"]["spillage"] is False
    assert r.json()["version"] == 0 and stored(env)[1] == 0  # a worker option: the shared settings and their version are untouched
    assert store.option("blocked_objects") == "1" and store.option("spillage", "0") == "0"
    saved = stored(env)[0]
    assert saved is None or ("blocked_objects" not in saved["ui"] and "blocked_objects" not in saved["run"])  # never written into the shared settings
    r = client.put("/api/settings", headers=AUTH, json={"blocked_objects": False})
    assert r.json()["values"]["blocked_objects"] is False and store.option("blocked_objects") == "0"
    assert client.put("/api/settings", headers=AUTH, json={"blocked_objects": None}).json()["values"]["blocked_objects"] is False  # null changes nothing


def test_object_notices_ride_the_event_log_like_the_others(client, env):
    client.put("/api/settings", headers=AUTH, json={"hold": 20})
    scope = client.get("/api/settings").json()["effective"]["db_detector"]
    db = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=scope)
    db.log_notice(2, "object_blocked", 400.0, "by potted plant: free but not usable")
    db.log_notice(2, "object_cleared", 460.0, "object gone")
    assert client.get("/api/events").json() == []  # entries and exits: none, and no false "vehicle out"
    assert [e["event"] for e in client.get("/api/events?kinds=notices").json()] == ["object_cleared", "object_blocked"]
    assert all(e["session_id"] is None for e in client.get("/api/events?kinds=all").json())
