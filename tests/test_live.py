"""Background worker + live store (carparking/live.py): settings, lease, requests and the Mode 5 loop without a page,
on a fake camera, detector and clock."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.history import HistoryDB  # noqa: E402
from carparking.live import LiveStore, Worker, decode_jpeg, heartbeat_age, stream_url  # noqa: E402
from carparking.stream_health import FreezeDetector, StreamHealth  # noqa: E402

FRAME = np.zeros((1000, 2000, 3), np.uint8)


def no_freeze():
    """The fake camera repeats one picture for ever: these tests are not about the stream (test_stream_health.py is)."""
    return StreamHealth(freeze=FreezeDetector(freeze_s=0))
CAR = {"poly": [(200, 300), (600, 300), (600, 500), (200, 500)], "box": (200, 300, 600, 500), "label": "car",
       "conf": 0.8, "state": 1}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeReader:
    """A camera delivering a new frame on every latest() call, stamped with the clock."""

    def __init__(self, url, clock):
        self.url, self.clock, self.seq, self.status, self.stopped = url, clock, 0, "streaming", False

    def latest(self):
        self.seq += 1
        return FRAME, self.seq, self.clock()

    def stop(self):
        self.stopped = True


class FakeDetector:
    task = "detect"
    weights = Path("fake.pt")

    def __init__(self, script):
        self.script, self.conf, self.calls = script, None, 0

    def __call__(self, frame, sliced=False):
        out = self.script(self.calls) if callable(self.script) else self.script
        self.calls += 1
        return [dict(d) for d in out]


def run_cfg(**over):
    m5 = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "roi": None, "whole": None, "classes": None,
          "extra": [], "match": "ground", "match_iou": 0.15, "save_every": 0, "hold": 2, "exit_hold": 5,
          "occlusion": 0.6, "hidden_max": 30, "learn_hold": 3, "learn": True, "absent_free": True}
    run = {"camera": "cam", "stream": "main", "custom_url": "", "db_detector": "fake", "conf": 0.3,
           "detector": {"kind": "coco", "weights": ["fake.pt"], "whole": None}, "fixed_mode": False, "fixed": None,
           "m5": m5}
    run.update(over)
    return run


def make_worker(tmp_path, store, clock, script=(CAR,), owner="w1", **kw):
    readers, dets = [], []

    def reader(url):
        readers.append(FakeReader(url, clock))
        return readers[-1]

    def detector(spec, frame):
        dets.append(FakeDetector(script))
        return dets[-1], False, "fake detector", True

    w = Worker(store, tmp_path, tmp_path / "h.db", raw_dir=tmp_path / "raw", default_url="rtsp://u:p@cam/Channels/101",
               owner=owner, reader_factory=reader, detector_factory=detector, clock=clock, **{"health_factory": no_freeze, **kw})
    return w, readers, dets


def steps(worker, clock, n, dt=1.0):
    for _ in range(n):
        worker.step()
        clock.t += dt


def test_settings_are_versioned_and_only_changes_count(tmp_path):
    store = LiveStore(tmp_path / "h.db")
    assert store.config() == (None, 0)
    assert store.set_config({"enabled": True, "run": run_cfg()}) == 1
    assert store.set_config({"enabled": True, "run": run_cfg()}) == 1  # same settings (every page rerun): no change
    assert store.set_config({"enabled": False, "run": run_cfg()}) == 2
    assert store.config() == ({"enabled": False, "run": run_cfg()}, 2)


def test_one_worker_holds_the_lease_a_second_one_waits(tmp_path):
    store = LiveStore(tmp_path / "h.db")
    assert store.claim("a", lease=20, now=0) and store.claim("a", lease=20, now=5)
    assert not store.claim("b", lease=20, now=10)  # a beat 5 s ago
    assert store.beat("a", {"status": "x"}, now=12)
    store.publish("a", {"status": "streaming", "frames": 583}, b"jpeg", now=13)
    assert store.claim("b", lease=20, now=40)  # a silent for 27 s: b takes over
    assert store.state()["snapshot"] == {"status": "starting", "owner": "b"}  # never a's numbers under b's beat
    assert not store.beat("a", {"status": "late"}, now=41)  # a lost it: its writes are refused
    store.release("b")
    assert store.claim("a", lease=20, now=42)  # released: free at once
    assert heartbeat_age(store.state(), now=50) == 8


def test_worker_logs_entries_and_publishes_without_a_page(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    w, readers, dets = make_worker(tmp_path, store, clock)
    steps(w, clock, 10)
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    assert [(e["slot_id"], e["event"]) for e in db.events()] == [(1, "entry")]
    assert readers[0].url == "rtsp://u:p@cam/Channels/101"
    state = store.state()
    snap = state["snapshot"]
    assert state["owner"] == "w1" and heartbeat_age(state, clock.t) <= 1.0
    assert snap["enabled"] and snap["cfg_ver"] == 1 and snap["counts"]["busy"] == 1 and snap["logged"] == 1
    assert snap["scope"] == "cam | fake" and snap["url"] == "rtsp://***@cam/Channels/101"
    overlay = decode_jpeg(state["overlay"])
    assert overlay.shape[1] == 1920  # the 2000 px frame scaled to the published width (+ the status bar)
    assert decode_jpeg(state["raw"]) is not None


def test_every_frame_is_published_clean_for_the_page_unless_the_option_is_off(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    w, _, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 4)
    state = store.state()
    assert state["snapshot"]["clean_frames"] is True and state["raw_ts"] == state["heartbeat"]  # a clean frame with every overlay
    assert not np.array_equal(decode_jpeg(state["raw"]), decode_jpeg(state["overlay"]))  # (the overlay has the drawing in it)
    store.set_option("clean_frames", "0")
    w.raw_every = 1e9  # (the first frame of this session already took its raw frame)
    steps(w, clock, 3)
    state = store.state()
    assert state["snapshot"]["clean_frames"] is False and state["raw_ts"] != state["heartbeat"]  # as it always was


def test_monitoring_off_idles_and_settings_changes_restart_the_session(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    w, readers, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 3)
    assert readers == [] and store.state()["snapshot"]["status"].startswith("idle")  # no settings yet
    store.set_config({"enabled": True, "run": run_cfg()})
    steps(w, clock, 3)
    assert len(readers) == 1 and w.mon is not None
    store.set_config({"enabled": True, "run": run_cfg(stream="sub")})
    steps(w, clock, 2)
    assert readers[0].stopped and readers[1].url.endswith("Channels/102")
    store.set_config({"enabled": False, "run": run_cfg(stream="sub")})
    steps(w, clock, 2)
    assert readers[1].stopped and w.reader is None and not store.state()["snapshot"]["enabled"]


def test_fixed_map_edit_is_applied_in_place(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    bay = {"id": 1, "name": "A1", "poly": [[0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.1, 0.5]]}
    other = {"id": 2, "name": "A2", "poly": [[0.6, 0.3], [0.8, 0.3], [0.8, 0.5], [0.6, 0.5]]}
    store.set_config({"enabled": True, "run": run_cfg(fixed_mode=True, fixed=[bay], db_detector="fake | fixed")})
    w, readers, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 6)
    mon = w.mon
    assert mon.mode == "fixed" and mon.tracker.slots[1]["state"] == 1
    store.set_config({"enabled": True, "run": run_cfg(fixed_mode=True, fixed=[bay, other],
                                                      db_detector="fake | fixed")})
    steps(w, clock, 2)
    assert w.mon is mon and len(readers) == 1  # same session: slot 1 keeps BUSY, slot 2 added
    assert mon.tracker.slots[1]["state"] == 1 and 2 in mon.tracker.slots
    store.set_config({"enabled": True, "run": run_cfg(fixed_mode=True, fixed=None, db_detector="fake | fixed")})
    steps(w, clock, 2)
    assert w.reader is None  # fixed mode without a saved map: idle, never learns into the fixed scope


def test_page_requests_reset_and_snapshot_old_ones_are_not_replayed(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    store.command("reset")  # made while no worker ran: the page handled it itself
    w, _, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 8)
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    assert list(db.load_slots()) == [1]
    store.command("snapshot")
    store.command("reset")
    steps(w, clock, 1)
    assert db.load_slots() == {} and db.sessions(open_only=True) == []  # reset: map gone, sessions closed
    assert len(list((tmp_path / "raw").glob("cam_*.jpg"))) == 1
    steps(w, clock, 8)
    assert list(db.load_slots()) == [1]  # relearned from the camera (a reset map starts its IDs over)
    store.command("restart")
    w.step()
    assert w.exit


def test_two_workers_never_log_twice(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    a, ra, _ = make_worker(tmp_path, store, clock, owner="a")
    b, rb, _ = make_worker(tmp_path, store, clock, owner="b")
    for _ in range(12):
        a.step()
        b.step()  # standby: a holds the lease
        clock.t += 1
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    assert [(e["slot_id"], e["event"]) for e in db.events()] == [(1, "entry")] and rb == []
    a.loop(max_frames=1, sleep=lambda s: None)  # a stops: releases the lease
    b.step()  # takes over at once
    assert len(rb) == 1 and store.state()["owner"] == "b"


def test_errors_are_reported_and_retried_never_a_crash(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    w, _, _ = make_worker(tmp_path, store, clock)

    def broken(spec, frame):
        raise RuntimeError("weights missing")

    w.detector_factory = broken
    pauses = []

    def sleep(s):
        pauses.append(s)
        clock.t += s
        if len(pauses) >= 3:
            w.exit = True

    w.loop(sleep=sleep)
    assert 5.0 in pauses and store.state()["snapshot"]["status"] == "error: weights missing"
    assert store.state()["heartbeat"] == 0  # released on the way out


def test_stream_urls():
    assert stream_url("rtsp://u:p@cam:554/Streaming/Channels/101", "sub") == "rtsp://u:p@cam:554/Streaming/Channels/102"
    assert stream_url("rtsp://u:p@cam:554/Streaming/Channels/102", "main").endswith("/101")
    assert stream_url("", "main") == ""


def test_overlay_is_a_valid_jpeg(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    w, _, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 3)
    img = cv2.imdecode(np.frombuffer(store.state()["overlay"], np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.mean() > 0  # the slot / box drawing is on it


def test_a_resave_with_the_same_run_settings_never_restarts_tracking(tmp_path):
    # the Streamlit page and the FastAPI backend store the same "run" for the same values, but the page also keeps
    # widget values ("ui") the other UI doesn't send: a new version with an identical run must not reset the tracker
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "ui": {"hold5": 2}, "run": run_cfg()})
    w, readers, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 6)
    mon = w.mon
    assert mon.tracker.slots[1]["state"] == 1
    assert store.set_config({"enabled": True, "ui": {"hold5": 2, "model5": "auto", "refresh": 1}, "run": run_cfg()}) == 2
    steps(w, clock, 2)
    assert w.mon is mon and len(readers) == 1 and mon.tracker.slots[1]["state"] == 1
    assert store.state()["snapshot"]["cfg_ver"] == 2  # the new version is acknowledged


def test_snapshot_carries_slots_detections_and_area_for_client_overlays(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    run = run_cfg()
    run["m5"].update(top=0.1, right=0.4)
    store.set_config({"enabled": True, "run": run})
    w, _, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 6)
    snap = store.state(images=False)["snapshot"]
    assert snap["seq"] == 6 and snap["shape"] == [2000, 1000]
    assert snap["detections"] == [{"label": "car", "conf": 0.8, "src": "", "kind": "vehicle",
                                   "box": [0.1, 0.3, 0.3, 0.5]}]
    assert [(s["id"], s["state"]) for s in snap["slots"]] == [(1, "BUSY")]
    assert snap["slots"][0]["poly"][0] == [0.1, 0.3]
    assert snap["area"] == {"top": 0.1, "bottom": 0.0, "left": 0.0, "right": 0.4, "roi": None}


def test_auto_model_from_the_hardware():
    from carparking.live import auto_variant
    assert auto_variant({"cuda": True, "gpu": "RTX 3060", "cores": 4, "ram_gb": 8})[0] == "yolo11m"
    assert auto_variant({"cuda": False, "cores": 32, "ram_gb": 64}, min_cores=16, min_ram_gb=16)[0] == "yolo11m"
    variant, why = auto_variant({"cuda": False, "cores": 6, "ram_gb": 15.5}, min_cores=16, min_ram_gb=16)
    assert variant == "default" and "6 CPU cores" in why  # this machine: the light model
    assert auto_variant({"cuda": False, "cores": 16, "ram_gb": 8}, min_cores=16, min_ram_gb=16)[0] == "default"


def test_model_variant_switches_the_detector_live(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg(detector={"kind": "coco", "weights": ["yolov8s.pt"],
                                                               "whole": 1920})})
    specs, readers = [], []

    def detector(spec, frame):
        specs.append(spec["weights"][0])
        return FakeDetector((CAR,)), False, "fake", True

    def reader(url):
        readers.append(FakeReader(url, clock))
        return readers[-1]

    w = Worker(store, tmp_path, tmp_path / "h.db", default_url="rtsp://u:p@cam/Channels/101", owner="w",
               reader_factory=reader, detector_factory=detector, clock=clock, health_factory=no_freeze,
               hw={"cuda": True, "gpu": "test GPU", "cores": 8, "ram_gb": 32},
               weights_for=lambda v: tmp_path / "models" / "yolo11m.pt")
    steps(w, clock, 6)  # no option stored = auto; a GPU here -> yolo11m
    model = store.state()["snapshot"]["model"]
    assert specs == [str(tmp_path / "models" / "yolo11m.pt")] and model["auto"] and model["active"] == "yolo11m"
    assert "GPU" in model["reason"] and w.mon.tracker.slots[1]["state"] == 1
    store.set_option("model_variant", "default")  # the user picks the light model in the dashboard
    steps(w, clock, 3)
    assert specs[-1] == "yolov8s.pt" and len(readers) == 1  # new detector, same camera session: no restart
    model = store.state()["snapshot"]["model"]
    assert model == {**model, "setting": "default", "active": "default", "auto": False}
    steps(w, clock, 4)
    assert w.mon.tracker.slots[1]["state"] == 1  # the slot map came back from the database and tracking goes on
    store.set_option("model_variant", "default")
    steps(w, clock, 3)
    assert len(specs) == 2  # the same choice again: nothing rebuilt


def test_camera_models_and_the_hybrid_are_never_swapped(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_option("model_variant", "yolo11m")
    store.set_config({"enabled": True, "run": run_cfg(detector={"kind": "coco", "weights": ["models/cam_x.pt"],
                                                               "whole": None})})
    specs = []

    def detector(spec, frame):
        specs.append(spec["weights"][0])
        return FakeDetector(()), False, "fake", True

    w = Worker(store, tmp_path, tmp_path / "h.db", default_url="rtsp://u:p@cam/Channels/101", owner="w",
               reader_factory=lambda url: FakeReader(url, clock), detector_factory=detector, clock=clock, health_factory=no_freeze,
               hw={"cuda": False, "cores": 4, "ram_gb": 8}, weights_for=lambda v: tmp_path / "yolo11m.pt")
    steps(w, clock, 3)
    assert specs == ["models/cam_x.pt"] and store.state()["snapshot"]["model"]["swappable"] is False


def test_display_thresholds_from_the_dashboard_apply_live(tmp_path):
    # Settings > Display thresholds: stored as the worker option "tuning"; the running tracker takes them on the
    # next frame (no restart, same monitor); nothing stored = the built-in defaults
    from carparking.history import DISPLAY_TUNING
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    w, _, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 3)
    tracker = w.mon.tracker
    assert {k: getattr(tracker, k) for k in DISPLAY_TUNING} == {k: v[0] for k, v in DISPLAY_TUNING.items()}
    store.set_option("tuning", '{"hidden_ground": 0.65, "ground_share": 5, "vehicle_iou": "x", "pin_reach": 0.4}')
    steps(w, clock, 1)
    assert w.mon.tracker is tracker  # applied in place
    assert (tracker.hidden_ground, tracker.ground_share, tracker.vehicle_iou) == (0.65, 0.9, 0.2)  # clamped / default
    assert tracker.pin_reach == 0.4
    store.set_option("tuning", "not json")
    steps(w, clock, 1)
    assert tracker.hidden_ground == DISPLAY_TUNING["hidden_ground"][0]  # unreadable: the defaults again


# --- Phase 2: the worker's flagged read of the active calibration (carparking/geometry_live.py) -------------------------------
def geo_record():
    """A calibration record for the 2000 x 1000 test frame: camera 6 m up, tilted 32 deg down, a trapezoid of ground."""
    from carparking.geometry import CameraModel, to_image
    from carparking.geometry.pipeline import build_record

    cam = CameraModel(1500.0, 1000.0, 500.0, np.radians(32), 0.0, 6.0)
    g = np.array([(-4, 9), (4, 9), (12, 24), (-12, 24)], float)
    ring = np.vstack([a + (b - a) * np.linspace(0, 1, 60, endpoint=False)[:, None] for a, b in zip(g, np.roll(g, -1, axis=0))])
    poly = (to_image(cam, ring) / (2000, 1000)).tolist()
    up = (-np.array([0.0, np.cos(cam.pitch), np.sin(cam.pitch)])).tolist()
    return build_record(width=2000, height=1000, polygon=poly, f=1500.0, up=up, depth={"h": 6.0})


def test_without_the_feed_the_snapshot_is_exactly_what_it_always_was(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    w, _, _ = make_worker(tmp_path, store, clock)  # CARPARK_GEOMETRY_ENABLED off: no feed
    assert w.geometry is None
    steps(w, clock, 6)
    assert "geometry" not in store.state(images=False)["snapshot"]


def test_an_active_calibration_adds_metrics_live_and_uncalibrated_means_nothing_extra(tmp_path):
    from carparking.geometry_live import GeometryFeed

    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    feed = GeometryFeed(lambda: store.geometry("cam"), "cam", clock=clock, log=lambda m: None)
    w, _, _ = make_worker(tmp_path, store, clock, geometry=feed)
    steps(w, clock, 6)
    assert "geometry" not in store.state(images=False)["snapshot"]  # flag on, camera not calibrated: legacy behaviour
    g = store.geometry("cam")
    g.create(geo_record())
    g.activate(1)  # the Admin accepted a calibration on the setup page
    mon, frames = w.mon, w.n
    steps(w, clock, 8)  # more than one POLL later, the same running worker
    snap = store.state(images=False)["snapshot"]
    assert w.mon is mon and w.n > frames and snap["counts"]["busy"] == 1  # nothing restarted, tracking untouched
    geo = snap["geometry"]
    assert geo["version"] == 1 and geo["valid"] is True and geo["counts"]["capacity"] == len(geo["bays"])
    assert {"vehicles", "distances", "bays", "confidence", "area_m2"} <= set(geo)
    g.mark_stale("moved")  # the drift guard: the camera counts as uncalibrated again
    steps(w, clock, 8)
    assert "geometry" not in store.state(images=False)["snapshot"]


def test_a_broken_metrics_feed_never_costs_a_frame(tmp_path, capsys):
    class Broken:
        def metrics(self, *a, **k):
            raise RuntimeError("boom")

    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    w, _, _ = make_worker(tmp_path, store, clock, geometry=Broken())
    steps(w, clock, 6)
    snap = store.state(images=False)["snapshot"]
    assert snap["seq"] == 6 and snap["counts"]["busy"] == 1 and "geometry" not in snap  # frames and tracking carry on
    assert capsys.readouterr().out.count("geometry metrics failed (RuntimeError: boom)") == 1  # said once, not every frame


def test_a_high_calibration_turns_dynamic_mode_into_a_full_grid_of_free_bays(tmp_path):
    from carparking.geometry_live import GeometryFeed

    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    feed = GeometryFeed(lambda: store.geometry("cam"), "cam", clock=clock, log=lambda m: None)
    w, _, _ = make_worker(tmp_path, store, clock, geometry=feed)
    rec = geo_record()
    rec["calibration"]["confidence"] = "HIGH"
    store.geometry("cam").create(rec)
    store.geometry("cam").activate(1)
    steps(w, clock, 8)
    snap = store.state(images=False)["snapshot"]
    n = len(snap["slots"])
    assert n > len(rec["capacity"]["slots"])  # the whole surface tiled, not just the calibration's 6 m-aisle packing
    a = snap["auto_slots"]
    assert snap["mode"] == "fixed" and (a["version"], a["count"]) == (1, n) and len(snap["slots"]) == n > 1
    assert a["measured"] + a["extrapolated"] == n and a["manual"] == 0 and a["expected"] is None and "short" not in a
    assert a["dynamic"] == 0 and w.mon.tracker.surface is not None and w.mon.tracker.auto is not None  # vehicle-driven bays are on
    assert w.mon.tracker.split is True  # sub-bays belong to the auto grid
    store.set_option("slot_split", "0")
    steps(w, clock, 3)
    assert w.mon.tracker.split is False
    store.set_option("slot_split", "1")
    steps(w, clock, 3)
    assert w.mon.tracker.split is True
    steps(w, clock, 40)  # past the hold: empty bays settle to FREE, the car's bay (if any) to BUSY
    snap = store.state(images=False)["snapshot"]
    c = snap["counts"]
    assert c["slots"] == n and c["free"] + c["busy"] + c["unknown"] == n and c["free"] >= n - 2  # every bay exists, empty ones FREE
    assert w.db.detector.endswith("| auto v1")  # its own history scope
    store.set_option("auto_dynamic", "0")  # the vehicle-driven rules off: the grid alone
    steps(w, clock, 3)
    assert w.mon.tracker.auto is None and store.state(images=False)["snapshot"]["auto_slots"]["count"] == n
    store.set_option("auto_dynamic", "1")
    steps(w, clock, 3)
    assert w.mon.tracker.auto is not None
    store.set_option("auto_slots", "0")  # the switch off: back to the learned map
    steps(w, clock, 3)
    snap = store.state(images=False)["snapshot"]
    assert snap["mode"] == "dynamic" and "auto_slots" not in snap
    store.set_option("auto_slots", "1")
    store.geometry("cam").mark_stale("moved")
    steps(w, clock, 8)
    assert store.state(images=False)["snapshot"]["mode"] == "dynamic"  # calibration gone: dynamic as always


def test_the_auto_grid_continues_the_lane_one_bay_past_the_last_measured_tile():
    import numpy as np

    from carparking.geometry_live import ActiveGeometry
    g = ActiveGeometry({"ver": 1, "record": geo_record()})
    tiles = sorted(g.tiles(), key=lambda ts: float(ts[0][:, 1].mean()))
    assert len(tiles) >= 4 and [s for _, s in tiles].count("extrapolated") == 1 and tiles[-1][1] == "extrapolated"
    a, b, c = (t.mean(axis=0) for t, _ in tiles[-3:])
    assert np.allclose(c - b, b - a, atol=0.3)  # the last tile is one pitch past the previous one: the lane's own spacing


def test_an_expected_count_extends_the_lane_up_to_it_and_never_past_the_cap():
    from carparking.geometry_live import MAX_EXTRAPOLATED, ActiveGeometry
    g = ActiveGeometry({"ver": 1, "record": geo_record()})
    m = len(g.measured())
    src = lambda e: [s for _, s in g.tiles(e)]  # noqa: E731
    assert src(m).count("extrapolated") == 0 and len(src(m - 1)) == m  # the count is met: nothing added, nothing removed
    assert src(m + 2).count("extrapolated") == 2 and len(src(m + 2)) == m + 2
    assert src(m + 50).count("extrapolated") == MAX_EXTRAPOLATED  # the camera model is not trusted further than that
    assert src(None).count("extrapolated") == 1  # no count set: the one bay it always added


def test_an_extrapolated_bay_that_leaves_the_frame_is_dropped_with_everything_beyond_it():
    from carparking.geometry_live import ActiveGeometry, GeometryFeed
    g = ActiveGeometry({"ver": 1, "record": geo_record()})
    t0 = g.measured()[0]
    far = lambda dy: t0 + (0.0, dy)  # noqa: E731
    g.tiles = lambda expected=None: [(t0, "measured"), (far(2000.0), "extrapolated"), (far(2005.0), "extrapolated")]
    slots = GeometryFeed._grid_slots(g, 2000, 1000, None)
    assert [s["source"] for s in slots] == ["measured"]  # a measured bay is never dropped for being near the edge


def test_pins_replace_the_bays_they_overlap_and_the_others_keep_their_ids():
    from carparking.geometry_live import merge_pins
    sq = lambda x, y, s=0.1: [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]  # noqa: E731
    grid = [{"id": i, "name": f"B{i}", "poly": sq(0.1 * i, 0.5), "lift": 0.01, "source": "measured"} for i in (1, 2, 3)]
    out = merge_pins(grid, [{"id": 1, "name": "1", "poly": sq(0.205, 0.5)}])  # on bay 2
    assert [(s["id"], s["source"]) for s in out] == [(1, "measured"), (2, "manual"), (3, "measured")]
    assert out[1]["lift"] == 0.0 and out[1]["name"] == "B2" and out[0]["lift"] == 0.01  # a pin is the operator's word
    out = merge_pins(grid, [{"id": 1, "name": "Truck", "poly": sq(0.7, 0.2)}, {"id": 2, "name": "2", "poly": sq(0.8, 0.2)}])
    assert [(s["id"], s["name"]) for s in out[3:]] == [(4, "Truck"), (5, "B5")]  # no overlap: added after the last bay
    assert merge_pins(grid, [{"id": 1}, {"poly": "x"}]) == grid  # bad pins are skipped, never raised
    assert grid[1]["source"] == "measured"  # the input is untouched


def test_the_operators_count_and_pins_reach_the_grid_and_the_status(tmp_path):
    from carparking.geometry_live import GeometryFeed
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    feed = GeometryFeed(lambda: store.geometry("cam"), "cam", clock=clock, log=lambda m: None)
    w, _, _ = make_worker(tmp_path, store, clock, geometry=feed)
    rec = geo_record()
    rec["calibration"]["confidence"] = "HIGH"
    store.geometry("cam").create(rec)
    store.geometry("cam").activate(1)
    steps(w, clock, 4)
    a = store.state(images=False)["snapshot"]["auto_slots"]
    n = a["count"]
    store.set_option("auto_grid", json.dumps({"expected": n + 5, "pins": []}))  # more than the lane can be extended to
    steps(w, clock, 4)
    a = store.state(images=False)["snapshot"]["auto_slots"]
    assert a["expected"] == n + 5 and a["count"] > n and a["short"] == n + 5 - a["count"]  # says how many it could not place
    pin = {"id": 1, "name": "Far", "poly": [[0.9, 0.05], [0.95, 0.05], [0.95, 0.1], [0.9, 0.1]]}
    store.set_option("auto_grid", json.dumps({"expected": None, "pins": [pin]}))
    steps(w, clock, 4)
    snap = store.state(images=False)["snapshot"]
    a = snap["auto_slots"]
    assert a["manual"] == 1 and a["count"] == n + 1 and a["sources"][str(a["count"])] == "manual"  # (JSON keys are strings)
    assert any(s["name"] == "Far" for s in snap["slots"])
    store.set_option("auto_grid", "not json")  # garbage in the option: the plain grid, frames carry on
    steps(w, clock, 4)
    assert store.state(images=False)["snapshot"]["auto_slots"]["manual"] == 0


def test_a_drawn_fixed_map_never_makes_sub_bays_whatever_the_slot_split_option_says(tmp_path):
    big = {"id": 1, "name": "G1", "poly": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg(fixed_mode=True, fixed=[big], db_detector="fake | fixed")})
    w, _, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 3)
    mon = w.mon
    assert w.mon.tracker.split is False and w.mon.tracker.auto is None  # the option is on by default: still none on a Fixed map
    for value in ("0", "1"):
        store.set_option("slot_split", value)
        steps(w, clock, 3)
        assert w.mon.tracker.split is False and w.mon is mon  # (and the monitor is not rebuilt for it every frame)


def test_a_newer_auto_grid_version_closes_the_older_versions_open_sessions_and_nothing_else(tmp_path):
    from carparking.geometry_live import GeometryFeed
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    feed = GeometryFeed(lambda: store.geometry("cam"), "cam", clock=clock, log=lambda m: None)
    w, _, _ = make_worker(tmp_path, store, clock, geometry=feed)
    scopes = {name: HistoryDB(tmp_path / "h.db", "cam", detector=det) for name, det in (
        ("older", "fake | auto v1"), ("newer", "fake | auto v9"), ("other detector", "yolo | auto v1"), ("learned map", "fake"))}
    for db in scopes.values():
        db.apply({"slot": 1, "state": 1, "time": 1.0, "initial": False})
    rec = geo_record()
    rec["calibration"]["confidence"] = "HIGH"
    store.geometry("cam").create(rec)
    store.geometry("cam").create(rec)
    store.geometry("cam").activate(2)
    steps(w, clock, 4)
    assert w.db.detector == "fake | auto v2"
    open_ = {name: len(db.sessions(open_only=True)) for name, db in scopes.items()}
    assert open_ == {"older": 0, "newer": 1, "other detector": 1, "learned map": 1}
    assert scopes["older"].sessions()[0]["note"] == "grid replaced by v2"


def test_the_snapshot_carries_the_history_check_and_a_long_mismatch_is_logged_once_and_when_it_clears(tmp_path, capsys):
    import sqlite3
    big = {"id": 1, "name": "G1", "poly": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg(fixed_mode=True, fixed=[big], db_detector="fake | fixed")})
    w, _, _ = make_worker(tmp_path, store, clock)
    steps(w, clock, 8)
    snap = store.state(images=False)["snapshot"]
    assert snap["sessions"] == {"expected": 1, "open": 1, "mismatch_s": 0.0, "error": ""}
    w.db.con.execute("DELETE FROM history WHERE exit_ts IS NULL")  # the log loses its session ...
    w.db.con.commit()

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    real, w.db.sync_open = w.db.sync_open, locked  # ... and cannot be repaired
    steps(w, clock, 20)
    s = store.state(images=False)["snapshot"]["sessions"]
    assert (s["expected"], s["open"]) == (1, 0) and 15 <= s["mismatch_s"] < 30 and "database is locked" in s["error"]
    assert "history differs" not in capsys.readouterr().out  # not yet 30 s
    steps(w, clock, 20)
    s = store.state(images=False)["snapshot"]["sessions"]
    assert s["mismatch_s"] >= 30
    steps(w, clock, 5)
    assert capsys.readouterr().out.count("history differs from the live count") == 1  # once, not every frame
    w.db.sync_open = real
    steps(w, clock, 2)
    assert store.state(images=False)["snapshot"]["sessions"] == {"expected": 1, "open": 1, "mismatch_s": 0.0, "error": ""}
    assert "matches the live count again" in capsys.readouterr().out


# --- the heartbeat from a thread of its own (item 106) --------------------------------------------------------------------

def heartbeat(store):
    return store.state(images=False)["heartbeat"]


def test_pulse_writes_only_the_heartbeat_of_the_lease_holder(tmp_path):
    from carparking.live import Pulse
    store = LiveStore(tmp_path / "h.db")
    store.claim("w1", now=100.0)
    store.publish("w1", {"status": "streaming", "seq": 7}, b"\xff\xd8x", now=100.0)
    assert store.pulse("w1", 150.0) and not store.pulse("w2", 160.0)
    state = store.state(images=False)
    assert state["heartbeat"] == 150.0 and state["snapshot"]["seq"] == 7  # the snapshot is not touched
    clock = Clock(200.0)
    p = Pulse(tmp_path / "h.db", "w1", every=3600, clock=clock)  # (the thread sleeps; tick() is called by hand)
    try:
        assert not p.tick() and heartbeat(store) == 150.0  # not active: the worker does not hold the lease (yet)
        p.active = True
        p.progress = 190.0
        assert p.tick() and heartbeat(store) == 200.0
        clock.t = 190.0 + 181.0  # the loop has not made progress for over PULSE_STALL: a hung worker must look down
        assert not p.tick() and heartbeat(store) == 200.0
        p.progress = clock.t
        store.claim("w2", now=500.0, lease=10)  # another worker took over
        assert not p.tick() and store.state(images=False)["owner"] == "w2"
    finally:
        p.stop()


def test_a_slow_frame_does_not_make_the_worker_look_down(tmp_path):
    import threading
    import time
    store = LiveStore(tmp_path / "h.db")
    store.set_config({"enabled": True, "run": run_cfg()})
    gate, inside = threading.Event(), threading.Event()

    def slow(frame, sliced=False):
        inside.set()
        gate.wait(10)  # one inference that takes longer than the page's patience
        return [dict(CAR)]

    class Det(FakeDetector):
        def __call__(self, frame, sliced=False):
            return slow(frame, sliced)

    w = Worker(store, tmp_path, tmp_path / "h.db", default_url="rtsp://u:p@cam/Channels/101", owner="w1",
               reader_factory=lambda url: FakeReader(url, time.time), clock=time.time, health_factory=no_freeze,
               detector_factory=lambda spec, frame: (Det(()), False, "slow", True), pulse_every=0.05)
    t = threading.Thread(target=lambda: [w.step() for _ in range(2)], daemon=True)
    t.start()
    assert inside.wait(5)  # the worker is inside the detector now
    first = heartbeat(store)
    time.sleep(0.4)
    assert heartbeat(store) > first  # the loop is blocked, the heartbeat moved on
    gate.set()
    t.join(10)
    w.pulse.stop()
    assert not t.is_alive()


def test_the_pulse_follows_the_lease_and_stops_with_the_loop(tmp_path):
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg()})
    w, _, _ = make_worker(tmp_path, store, clock, pulse_every=3600)
    assert not w.pulse.active
    steps(w, clock, 2)
    assert w.pulse.active and w.pulse.progress == clock.t - 1.0
    other = LiveStore(tmp_path / "h.db")
    other.con.execute("UPDATE live_state SET owner = 'w2', heartbeat = ? WHERE id = 1", (clock.t,))  # someone else holds it
    w.step()
    assert not w.pulse.active  # standby: the pulse must not keep another worker's lease alive
    w.exit = True
    w.loop()
    assert not w.pulse._thread.is_alive()  # stopped in the loop's finally


def test_status_carries_encroachment_and_obstacle_rejections_and_follows_the_options_live(tmp_path, monkeypatch):
    """The worker's snapshot (what /api/status serves) with the two options switched on and off while it runs."""
    grey = np.random.RandomState(1).randint(90, 130, (1000, 2000, 3)).astype(np.uint8)
    monkeypatch.setitem(globals(), "FRAME", grey)
    bays = [{"id": i + 1, "name": f"A{i + 1}", "poly": [[x, 0.5], [x + 0.18, 0.5], [x + 0.18, 0.7], [x, 0.7]]}
            for i, x in enumerate((0.10, 0.30, 0.50))]
    wide = {"poly": [(220, 420), (744, 420), (744, 695), (220, 695)], "box": (220, 420, 744, 695), "label": "car", "conf": 0.8, "state": 1}
    cone = {"poly": [(1100, 610), (1140, 610), (1140, 690), (1100, 690)], "box": (1100, 610, 1140, 690), "label": "car", "conf": 0.6, "state": 1}
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run_cfg(fixed_mode=True, fixed=bays, db_detector="fake | fixed")})
    w, _, _ = make_worker(tmp_path, store, clock, script=(wide, cone))

    def snap():
        steps(w, clock, 12)
        return store.state()["snapshot"]

    store.set_option("spillage", "1")  # switched on before the first frame
    store.set_option("obstacle_filter", "1")
    on = snap()
    by = {s["id"]: s for s in on["slots"]}
    assert by[2]["avail"] == "ENCROACHED" and by[2]["display"] == "ENCROACHED" and by[2]["state"] == "FREE"
    assert by[2]["encroached_by"] == ["A1"] and by[2]["cover"] >= 0.3
    assert by[1]["avail"] is None and by[1]["display"] == "BUSY"
    assert on["obstacle"] == {"rejected": 1, "reasons": {"too_small": 1}}
    assert by[3]["state"] == "FREE"  # the cone-sized box was dropped before it could hold A3
    marked = {d["label"] + str(d["box"]): d.get("rejected") for d in on["detections"]}
    assert sorted(v for v in marked.values() if v) == ["too_small"] and sum(1 for v in marked.values() if not v) == 1  # only the cone is marked
    assert on["vehicles"] == 1  # the dropped box is not counted as a vehicle
    c = on["counts"]
    assert (c["busy"], c["encroached"], c["free"], c["available"]) == (1, 1, 1, 1) and c["slots"] == 3
    json.dumps(on)

    store.set_option("spillage", "0")  # both off again, while the worker runs: no restart
    store.set_option("obstacle_filter", "0")
    off = snap()
    by = {s["id"]: s for s in off["slots"]}
    assert "obstacle" not in off and "encroached" not in off["counts"] and all("avail" not in s for s in off["slots"])
    assert by[2]["state"] == "FREE" and by[3]["state"] == "BUSY"
    assert all("rejected" not in d for d in off["detections"]) and off["vehicles"] == 2  # off: both boxes are vehicles, as always  # as it always was: the cone-sized box makes A3 BUSY
    assert off["counts"]["free"] == 1 and off["counts"]["busy"] == 2

    store.set_option("obstacle_filter", "1")  # on again: a bay that is already BUSY is never judged (a parked car is not taken away)
    again = snap()
    assert {s["id"]: s["state"] for s in again["slots"]}[3] == "BUSY" and again["obstacle"]["rejected"] == 0


def test_status_carries_blocked_bays_and_follows_the_option_live(tmp_path, monkeypatch):
    """Blocked by object through the worker: with the class list of a real run (cars only) the pot is not even reported until the option is on."""
    grey = np.random.RandomState(1).randint(90, 130, (1000, 2000, 3)).astype(np.uint8)
    monkeypatch.setitem(globals(), "FRAME", grey)
    bays = [{"id": i + 1, "name": f"A{i + 1}", "poly": [[x, 0.5], [x + 0.18, 0.5], [x + 0.18, 0.7], [x, 0.7]]}
            for i, x in enumerate((0.10, 0.30, 0.50))]
    car = {"poly": [(220, 420), (560, 420), (560, 695), (220, 695)], "box": (220, 420, 560, 695), "label": "car", "conf": 0.8, "state": 1}
    pot = {"poly": [(900, 560), (980, 560), (980, 690), (900, 690)], "box": (900, 560, 980, 690), "label": "potted plant", "conf": 0.7, "state": None}
    run = run_cfg(fixed_mode=True, fixed=bays, db_detector="fake | fixed")
    run["m5"]["classes"] = ["car"]
    store, clock = LiveStore(tmp_path / "h.db"), Clock()
    store.set_config({"enabled": True, "run": run})
    w, _, _ = make_worker(tmp_path, store, clock, script=(car, pot))

    def snap():
        steps(w, clock, 12)
        return store.state()["snapshot"]

    off = snap()  # the default: nothing about it anywhere
    assert all("blocked_by" not in s for s in off["slots"]) and "blocked_objects" not in off["counts"] and off["others"] == 0
    assert off["counts"]["free"] == 2 and {s["id"]: s["state"] for s in off["slots"]} == {1: "BUSY", 2: "FREE", 3: "FREE"}

    store.set_option("blocked_objects", "1")  # on while the worker runs: no restart
    on = snap()
    by = {s["id"]: s for s in on["slots"]}
    assert by[2]["avail"] == "BLOCKED" and by[2]["display"] == "BLOCKED" and by[2]["state"] == "FREE" and by[2]["blocked_by"] == ["potted plant"]
    assert by[1]["avail"] is None and by[1]["state"] == "BUSY" and by[3]["avail"] is None
    c = on["counts"]
    assert (c["busy"], c["blocked_objects"], c["free"], c["available"], c["slots"]) == (1, 1, 1, 1, 3)
    assert on["others"] == 1 and on["vehicles"] == 1  # the pot is shown as another object, never a vehicle
    json.dumps(on)

    store.set_option("blocked_objects", "0")
    again = snap()
    assert all("blocked_by" not in s for s in again["slots"]) and "blocked_objects" not in again["counts"] and again["counts"]["free"] == 2
    assert again["others"] == 0  # the class list is what it was
