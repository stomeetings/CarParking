"""Camera registry (carparking/registry.py) and the multi-camera supervisor (carparking/supervisor.py).
The PostgreSQL backend runs the same checks when CARPARK_TEST_PG_DSN points at a scratch database (compose: the db
service); without it those cases are skipped (the image build has no database)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.fixed_slots import FixedSlotStore  # noqa: E402
from carparking.registry import (Camera, JsonRegistry, PgRegistry, runtime_paths, seed, slugify,  # noqa: E402
                                 streams_of, sync_primary_slots)
from carparking.supervisor import Supervisor  # noqa: E402

MAIN = "rtsp://admin:pw@192.168.20.10:554/Streaming/Channels/101"
SUB = "rtsp://admin:pw@192.168.20.10:554/Streaming/Channels/102"
BAY = {"id": 1, "name": "A1", "poly": [[0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.1, 0.5]]}


def backends(tmp_path):
    out = [JsonRegistry(tmp_path / "cameras.json")]
    dsn = os.environ.get("CARPARK_TEST_PG_DSN")
    if dsn:
        reg = PgRegistry(dsn)
        for c in reg.list():  # a scratch database: start empty (the primary one last: the FK cascade drops slots)
            if not c.primary:
                reg.delete(c.id)
        import psycopg
        with psycopg.connect(dsn, autocommit=True) as con:
            con.execute("DELETE FROM cameras")
        out.append(reg)
    return out


@pytest.fixture(params=["json", "pg"])
def registry(request, tmp_path):
    regs = backends(tmp_path)
    if request.param == "pg" and len(regs) < 2:
        pytest.skip("set CARPARK_TEST_PG_DSN to run the PostgreSQL backend")
    return regs[0] if request.param == "json" else regs[1]


def test_cameras_crud_with_checks(registry):
    cam = registry.create(Camera("front", "Front Parking 02", MAIN, SUB, primary=True, metadata={"site": "HQ"}))
    assert cam.created and registry.primary().id == "front"
    registry.create(Camera("gate", "Gate", MAIN.replace("101", "201"), sort_order=1))
    assert [c.id for c in registry.list()] == ["front", "gate"]  # the primary first, then by order
    for bad, why in ((Camera("Front!", "X", MAIN), "lowercase"), (Camera("x", "Gate", MAIN), "exists already"),
                     (Camera("gate", "Other", MAIN), "exists already"), (Camera("y", "Y", "http//x"), "rtsp"),
                     (Camera("z", "Z", MAIN, primary=True), "primary")):
        with pytest.raises(ValueError, match=why):
            registry.create(bad)
    up = registry.update("gate", rtsp_sub_url=SUB, metadata={"site": "gate"}, name="ignored", primary=True)
    assert up.rtsp_sub_url == SUB and up.metadata == {"site": "gate"} and up.name == "Gate" and not up.primary
    assert registry.update("gate", rtsp_url="rtsp://***@192.168.20.10:554/x").rtsp_url == MAIN.replace("101", "201")
    with pytest.raises(ValueError, match="primary"):
        registry.delete("front")
    assert registry.delete("gate") and not registry.delete("gate") and [c.id for c in registry.list()] == ["front"]
    assert "pw" not in str(cam.public()) and cam.public()["rtsp_url"].startswith("rtsp://***@")


def test_slot_coordinates_per_camera(registry):
    registry.create(Camera("front", "Front", MAIN, primary=True))
    registry.create(Camera("gate", "Gate", MAIN))
    assert registry.slots("gate") == {"updated": None, "slots": []}
    assert registry.set_slots("gate", [BAY]) == [BAY]
    assert registry.slots("gate")["slots"] == [BAY] and registry.slots("front")["slots"] == []  # isolated
    with pytest.raises(ValueError, match="slot 1"):
        registry.set_slots("gate", [{"id": 1, "name": "x", "poly": [[0, 0], [2, 0], [0, 1]]}])
    with pytest.raises(KeyError):
        registry.set_slots("nope", [BAY])
    registry.delete("gate")
    registry.create(Camera("gate", "Gate", MAIN))
    assert registry.slots("gate")["slots"] == []  # a removed camera's map went with it


def test_seed_runtime_paths_and_streams(tmp_path):
    fixed = tmp_path / "config" / "fixed_slots.json"
    FixedSlotStore(fixed).save("Camera 1", [BAY])
    reg = JsonRegistry(tmp_path / "cameras.json")
    cam = seed(reg, "Camera 1", MAIN, SUB, fixed)
    assert cam.id == "camera-1" and cam.primary and reg.slots("camera-1")["slots"] == [BAY]
    assert seed(reg, "Camera 1", MAIN, SUB, fixed) is None  # only into an empty registry
    rt = runtime_paths(cam, tmp_path, tmp_path / "parking_history.db", fixed)
    assert rt.db == tmp_path / "parking_history.db" and rt.fixed == fixed  # the primary keeps the original files
    gate = reg.create(Camera("gate", "Gate", MAIN))
    rt = runtime_paths(gate, tmp_path, tmp_path / "parking_history.db", fixed)
    assert rt.db == tmp_path / "cameras" / "gate" / "parking_history.db" and rt.fixed.parent.name == "gate"
    assert streams_of(cam)["hd"][0] == "parking_hd" and streams_of(cam)["sd"][1] == f"ffmpeg:{SUB}#video=h264"
    assert streams_of(gate)["hd"] == ("gate_hd", f"ffmpeg:{MAIN}#video=h264#width=1920#height=1080")
    assert streams_of(gate)["sd"] is None  # no sub-stream given
    assert slugify("Front Parking 02!") == "front-parking-02"


def test_primary_slots_follow_the_newer_side(tmp_path):
    fixed = tmp_path / "fixed_slots.json"
    reg = JsonRegistry(tmp_path / "cameras.json")
    cam = reg.create(Camera("camera-1", "Camera 1", MAIN, primary=True))
    reg.set_slots("camera-1", [BAY], updated="2026-01-01T00:00:00+00:00")
    assert sync_primary_slots(reg, cam, fixed) == "exported" and FixedSlotStore(fixed).load("Camera 1") == [BAY]
    assert sync_primary_slots(reg, cam, fixed) is None
    edited = {**BAY, "name": "A1-edited"}  # Streamlit's editor writes the file (with a newer timestamp)
    FixedSlotStore(fixed).save("Camera 1", [edited])
    assert sync_primary_slots(reg, cam, fixed) == "imported" and reg.slots("camera-1")["slots"] == [edited]


class FakeProc:
    def __init__(self, env):
        self.env, self.code, self.terminated = env, None, False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated, self.code = True, 0

    def wait(self, timeout=None):
        return self.code


def test_supervisor_runs_one_worker_per_camera(tmp_path):
    reg = JsonRegistry(tmp_path / "cameras.json")
    reg.create(Camera("camera-1", "Camera 1", MAIN, SUB, primary=True))
    started, t = [], [1000.0]

    def spawn(env):
        started.append(FakeProc(env))
        return started[-1]

    sup = Supervisor(reg, tmp_path, tmp_path, tmp_path / "parking_history.db", tmp_path / "config" / "f.json",
                     base_env={"CARPARK_TZ": "UTC"}, spawn=spawn, clock=lambda: t[0], log=lambda m: None)
    sup.step()
    assert len(started) == 1 and started[0].env["CARPARK_HISTORY_DB"] == str(tmp_path / "parking_history.db")
    assert started[0].env["CARPARK_RTSP_URL"] == MAIN and started[0].env["CARPARK_TZ"] == "UTC"
    reg.create(Camera("gate", "Gate", MAIN.replace("101", "201")))
    reg.set_slots("gate", [BAY])
    sup.step()
    gate = started[1]
    assert gate.env["CARPARK_CAMERA_ID"] == "gate" and (tmp_path / "cameras" / "gate").is_dir()
    assert FixedSlotStore(gate.env["CARPARK_FIXED_SLOTS"]).load("Gate") == [BAY]  # its worker sees its map
    sup.step()
    assert len(started) == 2  # nothing changed: nothing restarted
    reg.update("gate", rtsp_url=MAIN.replace("101", "301"))
    sup.step()
    assert gate.terminated and started[2].env["CARPARK_RTSP_URL"].endswith("301")  # new URL: new process
    started[2].code = 1  # a crash: restarted after the backoff, not in a tight loop
    t[0] += 1
    sup.step()
    assert len(started) == 3
    t[0] += 10
    sup.step()
    assert len(started) == 4 and sup.procs["gate"]["fails"] == 1
    reg.delete("gate")
    sup.step()
    assert started[3].terminated and set(sup.procs) == {"camera-1"}
    sup.stop_all()
    assert started[0].terminated and sup.procs == {}


def test_display_name_is_editable_the_history_key_is_not(registry):
    cam = registry.create(Camera("gate", "Gate", MAIN, primary=True))
    assert cam.display == "Gate" and cam.public()["display"] == "Gate"  # no label yet: the name
    up = registry.update("gate", label="  East Gate  ", name="Renamed")
    assert up.label == "East Gate" and up.display == "East Gate" and up.name == "Gate"  # name stays: history key
    assert registry.get("gate").public()["display"] == "East Gate"
    assert registry.update("gate", label="").display == "Gate"  # cleared: back to the name
    other = registry.create(Camera("yard", "Yard", MAIN, label="Main Parking Area"))
    assert other.display == "Main Parking Area"
