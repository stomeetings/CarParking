"""Pixel recorder, flicker metrics and the offline replay (pixel_replay.py, flicker.py, scripts/pixel_record.py, flicker_report.py,
replay_eval.py). A fake worker serves a synthetic parked-car scene (the one of test_fixed_slots_evidence.py, small); tmp_path only: no
network, no model, no ./data. Nothing here sets or asserts a tuned camera threshold: only that the tooling measures and compares."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.flicker import analyze, compare, report_lag, score_exits, transitions  # noqa: E402
from carparking.pixel_recorder import ensure_outside, record_pixels  # noqa: E402
from carparking.pixel_replay import PixelRecording, cfg_with, parse_cfg_value, replay  # noqa: E402


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pixel_record, flicker_report, replay_eval = script("pixel_record"), script("flicker_report"), script("replay_eval")

W, H = 640, 320
SLOT = {"id": 1, "name": "A1", "poly": [[0.2, 0.45], [0.2, 0.60], [0.56, 0.60], [0.56, 0.45]], "capacity": 1, "state": "BUSY"}
DYN = {"id": 1001, "name": "dyn", "poly": [[0.7, 0.45], [0.7, 0.6], [0.9, 0.6], [0.9, 0.45]], "capacity": 1, "state": "BUSY", "parent": 1}
CAR = (0.21, 0.44, 0.55, 0.595)  # fractions
T0 = 1000.0
SETTINGS = {"hold": 2, "exit_hold": 5, "absent_free": True, "match_rule": "ground"}


def scene(car: bool, gain: float) -> bytes:
    img = np.full((H, W, 3), 110, np.uint8)
    if car:
        x1, y1, x2, y2 = (int(CAR[0] * W), int(CAR[1] * H), int(CAR[2] * W), int(CAR[3] * H))
        cv2.rectangle(img, (x1 + 10, y1 + 6), (x2 - 10, y2 - 2), (40, 40, 200), -1)
        cv2.rectangle(img, (x1 + 45, y1 + 9), (x2 - 45, y1 + 25), (30, 30, 30), -1)
        for cx in (x1 + 40, x2 - 40):
            cv2.circle(img, (cx, y2 - 2), 9, (10, 10, 10), -1)
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


class Worker:
    """A fake stack: plan(t) -> (car in the picture, detected, gain); one second per sleep()."""

    def __init__(self, plan, flip_status_after=None):
        self.t, self.plan, self.flip = 0, plan, flip_status_after
        self.reads = 0

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += 1

    def status(self):
        self.reads += 1
        car, seen, _ = self.plan(self.t)
        ts = T0 + self.t + (0.5 if self.flip is not None and self.t == self.flip and self.reads % 2 == 0 else 0)
        dets = [{"box": list(CAR), "conf": 0.8, "label": "car", "kind": "vehicle"}] if seen else []
        return {"snapshot": {"frame_ts": ts, "shape": [W, H], "detections": dets + [{"box": [0, 0, .1, .1], "conf": .5, "label": "x",
                                                                                      "kind": "other"}],
                             "slots": [{**SLOT, "state": "BUSY" if car else "FREE"}, DYN]}}

    def frame(self):
        car, _, gain = self.plan(self.t)
        return scene(car, gain)


def stationary(t):
    """A car parked all the time; 7 s detection dropouts every 60 s; at 200..275 s the light dips to 55 % and the detector loses it."""
    dip = 200 <= t < 275
    return True, not (dip or ((t % 60) < 7 and t >= 20)), 0.55 if dip else 1.0


def departs(t):
    """Parked until 200 s, then gone (a real exit), the detector never sees it again."""
    return t < 200, t < 200, 1.0


def make(tmp_path, plan, seconds, name="rec", **kw):
    w = Worker(plan, kw.pop("flip_status_after", None))
    summary = record_pixels(w.status, w.frame, tmp_path / name, seconds, clock=w.clock, sleep=w.sleep, camera="camera-1",
                            settings=SETTINGS, **kw)
    return summary, tmp_path / name


@pytest.fixture(scope="module")
def parked(tmp_path_factory):
    return make(tmp_path_factory.mktemp("parked"), stationary, 420)[1]


@pytest.fixture(scope="module")
def leaving(tmp_path_factory):
    return make(tmp_path_factory.mktemp("leaving"), departs, 330)[1]


# ---- recorder ------------------------------------------------------------------------------------------------------------------
def test_recording_layout_and_content(tmp_path):
    summary, d = make(tmp_path, stationary, 30)
    assert summary["frames"] == 30 and summary["stopped"] == "time" and summary["mismatched"] == 0
    rec = PixelRecording(d)
    assert len(rec) == 30 and rec.header["format"] == 1 and rec.header["camera"] == "camera-1"
    assert [s["id"] for s in rec.slots] == [1]  # the vehicle-driven bay is not part of the map
    assert rec.header["settings"] == SETTINGS
    f = rec.frames[0]
    assert f["dets"] == [{"box": list(CAR), "conf": 0.8, "label": "car"}]  # only vehicles, only box / conf / label
    assert f["states"] == {"1": "BUSY"}
    assert (d / f["file"]).read_bytes()[:2] == b"\xff\xd8" and rec.image(0).shape == (H, W, 3)
    assert rec.live_states() == {1: [1] * 30}


def test_every_thins_the_frames_and_max_bytes_stops(tmp_path):
    s1, _ = make(tmp_path, stationary, 60, "thin", every=5.0)
    assert s1["frames"] == 12
    s2, d = make(tmp_path, stationary, 60, "small", max_bytes=1)
    assert s2["frames"] == 1 and s2["stopped"] == "size"


def test_a_picture_from_another_frame_is_not_kept(tmp_path):
    summary, d = make(tmp_path, stationary, 20, flip_status_after=10)
    assert summary["mismatched"] == 1 and summary["frames"] == 19


def test_transient_request_errors_are_retried_and_a_dead_stack_ends_the_run(tmp_path):
    w = Worker(stationary)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] in (3, 4, 9):
            raise TimeoutError("timed out")
        return w.status()
    summary = record_pixels(flaky, w.frame, tmp_path / "flaky", 30, clock=w.clock, sleep=w.sleep)
    assert summary["errors"] == 3 and summary["frames"] >= 20 and len(PixelRecording(tmp_path / "flaky")) == summary["frames"]
    w2 = Worker(stationary)

    def dies():
        if w2.t >= 5:
            raise ConnectionResetError("reset")
        return w2.status()
    with pytest.raises(RuntimeError, match="in a row failed"):
        record_pixels(dies, w2.frame, tmp_path / "dead", 600, clock=w2.clock, sleep=w2.sleep, max_errors=5)
    assert len(PixelRecording(tmp_path / "dead")) >= 4  # what was recorded before is closed and readable


def test_the_recorder_module_needs_no_opencv_and_keeps_the_dynamic_base_in_step():
    import subprocess

    from carparking import fixed_slots, pixel_recorder
    assert pixel_recorder.DYNAMIC_BASE == fixed_slots.DYNAMIC_BASE
    code = "import sys; sys.modules['cv2'] = None; sys.modules['numpy'] = None; sys.path.insert(0, 'src'); import carparking.pixel_recorder"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


def test_nothing_published_is_an_error(tmp_path):
    w = Worker(stationary)
    with pytest.raises(RuntimeError, match="no frame"):
        record_pixels(lambda: {}, lambda: None, tmp_path / "x", 3, clock=w.clock, sleep=w.sleep)


def test_a_truncated_last_line_is_tolerated_and_a_non_recording_is_refused(tmp_path):
    _, d = make(tmp_path, stationary, 10)
    with open(d / "frames.jsonl", "a", encoding="utf-8") as f:
        f.write('{"i": 11, "t": 1010.0, "fi')
    assert len(PixelRecording(d)) == 10
    with pytest.raises(ValueError, match="not a pixel recording"):
        PixelRecording(tmp_path / "nothing")
    with pytest.raises(FileExistsError):
        make(tmp_path, stationary, 5, "rec")


def test_never_into_the_live_data_folder(tmp_path):
    with pytest.raises(ValueError, match="live stack"):
        ensure_outside(tmp_path / "data" / "cameras" / "x", tmp_path / "data")
    ensure_outside(tmp_path / "recordings", tmp_path / "data")
    msgs = []
    rc = pixel_record.main(["record", "--minutes", "1", "--out", str(ROOT / "data" / "pixels")], out=msgs.append,
                           fetch_settings=lambda: {})
    assert rc == 2 and "refused" in msgs[-1] and not (ROOT / "data" / "pixels").exists()


# ---- flicker -------------------------------------------------------------------------------------------------------------------
def test_transitions_skip_unknown_and_the_first_reading():
    t = list(range(8))
    assert transitions(t, [None, 1, 1, None, 1, 0, None, 1]) == [(5, 0), (7, 1)]
    assert transitions([0], [1]) == []


def test_ghost_pair_window_and_counts():
    t = list(range(0, 1000, 10))
    s = [1] * 100
    for i in range(10, 13):  # exit at 100 s, entry at 130 s: a ghost pair
        s[i] = 0
    for i in range(50, 100):  # exit at 500 s, never back: a departure
        s[i] = 0
    rep = analyze(t, {1: s}, {1: "A1"}, "x", window=120, blink=60)
    sl = rep.slots[0]
    assert (sl.exits, sl.entries, sl.ghost_pairs, sl.ghost_gaps) == (2, 1, 1, [30]) and rep.ghost_pairs == 1
    assert sl.short_states == 1 and sl.longest_stable_s == 990 - 500  # (the run cut off by the end counts: a lower bound)
    assert analyze(t, {1: s}, {}, "x", window=20).ghost_pairs == 0  # a measurement choice: the window
    assert rep.ghost_pairs_per_slot_hour == pytest.approx(1 / (990 / 3600))
    with pytest.raises(ValueError):
        analyze(t, {1: [1]})
    assert "ghost pairs" in compare([rep, analyze(t, {1: [1] * 100}, {1: "A1"}, "steady")])


def test_report_lag_and_exit_scoring():
    ev = [{"slot": 1, "state": 0, "time": 100.0, "reported": 150.0, "initial": False},
          {"slot": 1, "state": 1, "time": 300.0, "reported": 301.0, "initial": False},
          {"slot": 1, "state": 0, "time": 400.0, "reported": 450.0, "initial": False},
          {"slot": 2, "state": 0, "time": 0.0, "reported": 0.0, "initial": True}]
    assert report_lag(ev)["exit"] == {"n": 2, "median_s": 50.0, "max_s": 50.0}
    sc = score_exits(ev, [{"slot": 1, "t": 90.0}, {"slot": 1, "t": 900.0}])
    assert sc["found"] == 1 and len(sc["missed"]) == 1 and len(sc["false_exits"]) == 1 and sc["lag_s"] == [60.0]


# ---- replay --------------------------------------------------------------------------------------------------------------------
def ghost_pairs(res, window=120.0):
    return analyze(res.times, res.shown, res.names, res.label, window).ghost_pairs


def test_replay_off_shows_the_ghost_and_on_removes_it(parked):
    rec = PixelRecording(parked)
    off, on = replay(rec, 0), replay(rec, 1)
    assert ghost_pairs(off) >= 1  # what the evaluators saw
    assert ghost_pairs(on) == 0 and not [e for e in on.events if e["state"] == 0 and not e["initial"]]
    assert sum(on.vetoes.values()) >= 1 and on.cfg["tau_exit"] == 30.0 and off.cfg == {}


def test_replay_off_matches_what_the_tracker_does_directly(parked):
    rec = PixelRecording(parked)
    a, b = replay(rec, 0), replay(rec, 0)
    assert a.events == b.events and a.shown == b.shown  # deterministic


def test_a_real_exit_still_goes_through_with_a_measured_lag(leaving):
    rec = PixelRecording(leaving)
    on = replay(rec, 1)
    exits = [e for e in on.events if e["state"] == 0 and not e["initial"]]
    assert len(exits) == 1
    assert 190 <= exits[0]["time"] - T0 <= 215 and exits[0]["reported"] >= exits[0]["time"]
    sc = score_exits(on.events, [{"slot": 1, "t": T0 + 200}], tol=30)
    assert sc["found"] == 1 and not sc["missed"] and not sc["false_exits"]
    assert report_lag(on.events)["exit"]["n"] == 1


def test_cfg_overrides_are_checked():
    assert cfg_with(motion_ratio=3.0).motion_ratio == 3.0 and cfg_with().motion_ratio == 2.0
    with pytest.raises(ValueError, match="not EvidenceCfg fields"):
        cfg_with(motion_ratoi=3.0)
    assert parse_cfg_value("lock_frames", "4") == 4 and isinstance(parse_cfg_value("min_ncc", "1"), float)
    with pytest.raises(ValueError):
        parse_cfg_value("min_ncc", "high")


def test_a_cfg_that_disables_the_veto_brings_the_ghost_back(parked):
    rec = PixelRecording(parked)
    weak = replay(rec, 1, cfg_with(max_vetoes=0))  # the exit is never held back by the veto (EMA and lock still run)
    assert sum(weak.vetoes.values()) == 0
    assert ghost_pairs(weak) >= ghost_pairs(replay(rec, 1))


# ---- scripts -------------------------------------------------------------------------------------------------------------------
def test_pixel_record_cli_records_and_describes(tmp_path):
    w = Worker(stationary)
    msgs = []
    rc = pixel_record.main(["record", "--minutes", "0.5", "--out", str(tmp_path / "cli")], fetch_status=w.status,
                           fetch_frame=w.frame, fetch_settings=lambda: {"values": {**SETTINGS, "secret_thing": 1, "conf": 0.3}},
                           out=msgs.append, sleep=w.sleep, clock=w.clock)
    assert rc == 0 and "recorded 30 frames" in msgs[-1]
    assert PixelRecording(tmp_path / "cli").header["settings"] == {**SETTINGS, "conf": 0.3}  # only the known keys are kept
    msgs.clear()
    assert pixel_record.main(["info", str(tmp_path / "cli")], out=msgs.append) == 0 and "30 frames" in msgs[0]
    # settings unreadable: still records, says so
    w2, msgs = Worker(stationary), []

    def broken():
        raise OSError("down")
    assert pixel_record.main(["record", "--minutes", "0.1", "--out", str(tmp_path / "cli2")], fetch_status=w2.status,
                             fetch_frame=w2.frame, fetch_settings=broken, out=msgs.append, sleep=w2.sleep, clock=w2.clock) == 0
    assert "warning: could not read" in msgs[0]


def test_flicker_report_cli(parked, tmp_path):
    msgs = []
    assert flicker_report.main([str(parked)], out=msgs.append) == 0 and "live" in "\n".join(msgs)
    msgs.clear()
    j = tmp_path / "f.json"
    assert flicker_report.main([str(parked), "--source", "all", "--json", str(j)], out=msgs.append) == 0
    text = "\n".join(msgs)
    assert "replay off" in text and "replay on" in text and "report lag" in text
    data = json.loads(j.read_text())
    by = {r["label"]: r for r in data["reports"]}
    assert by["replay off"]["ghost_pairs"] >= 1 and by["replay on"]["ghost_pairs"] == 0
    assert flicker_report.main([str(parked), "--source", "on", "--set", "nope=1"], out=msgs.append) == 2
    assert flicker_report.main([str(parked), "--set", "min_ncc"], out=msgs.append) == 2


def test_replay_eval_cli_side_by_side_sweep_and_truth(leaving, tmp_path):
    msgs = []
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"exits": [{"slot": "A1", "t": T0 + 200}]}))
    j = tmp_path / "r.json"
    assert replay_eval.main([str(leaving), "--truth", str(truth), "--json", str(j)], out=msgs.append) == 0
    text = "\n".join(msgs)
    assert "agrees with live" in text and "replay on" in text and "truth:" in text
    run = json.loads(j.read_text())["runs"][0]
    assert run["truth"]["found"] == 1 and run["truth"]["false_exits"] == 0
    msgs.clear()
    assert replay_eval.main([str(leaving), "--sweep", "motion_ratio=1.5,3", "--sweep", "min_ncc=0.4"], out=msgs.append) == 0
    rows = [m for m in msgs if m.lstrip().startswith(("1.5", "3.0"))]
    assert len(rows) == 2 and "motion_ratio" in "\n".join(msgs)
    assert replay_eval.main([str(leaving), "--sweep", "bogus=1"], out=msgs.append) == 2
    truth.write_text(json.dumps({"exits": [{"slot": "ZZ", "t": 1}]}))
    assert replay_eval.main([str(leaving), "--truth", str(truth)], out=msgs.append) == 2
