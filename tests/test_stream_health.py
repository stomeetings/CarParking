"""Stream health (item 105): a frozen or stale camera is not counted from. FreezeDetector / StreamHealth on their own, the reader's
jittered backoff, and the Worker end to end (slots UNKNOWN, open sessions kept, one log line each way)."""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.live import LiveStore, Worker  # noqa: E402
from carparking.rtsp import LatestFrameReader  # noqa: E402
from carparking.stream_health import (ACTIVE, FROZEN, STALE, FreezeDetector, StreamHealth,  # noqa: E402
                                      health_from_env, thumbnail)

from test_live import CAR, Clock, FakeDetector, run_cfg  # noqa: E402


def grey(v, shape=(1000, 2000)):  # (the size of test_live's FRAME: CAR's box lies inside it)
    return np.full((*shape, 3), v, np.uint8)


def test_the_thumbnail_is_64_by_36_grayscale():
    t = thumbnail(grey(100, (1080, 1920)))
    assert t.shape == (36, 64) and abs(float(t.mean()) - 100) < 0.01


def test_a_repeated_picture_is_frozen_after_20_seconds_not_before():
    fd = FreezeDetector()
    still = grey(90)
    flags = [fd.update(still, float(t)) for t in range(0, 31)]
    assert not any(flags[:20]) and all(flags[20:])  # frozen_for reaches 20 s at t = 20
    assert fd.diff == 0.0 and fd.frozen_for == 30.0


def test_a_scene_that_drifts_slowly_is_never_frozen_although_consecutive_frames_barely_differ():
    fd = FreezeDetector()
    # +0.2 grey levels per frame: each frame is within 0.5 of the previous one, but the anchor is left behind after a few frames
    assert not any(fd.update(grey(100 + 0.2 * i), float(i)) for i in range(120))
    assert fd.frozen_for < 6  # (8-bit pictures: the 0.2 steps show as a whole grey level every 5th frame)


def test_a_moving_scene_restarts_the_quiet_stretch_and_the_check_can_be_switched_off():
    fd = FreezeDetector()
    for t in range(15):
        fd.update(grey(90), float(t))
    assert not fd.update(grey(140), 15.0) and fd.frozen_for == 0.0  # something changed: a new anchor
    assert not FreezeDetector(freeze_s=0).update(grey(90), 99999.0)


def test_stale_means_no_new_frame_for_ten_seconds_and_active_again_with_one():
    h = StreamHealth(freeze=FreezeDetector(freeze_s=0))
    assert h.observe(None, 0, 0.0, 100.0) == ACTIVE  # nothing yet: the reader's own status says "connecting"
    assert h.observe(grey(50), 1, 100.0, 105.0) == ACTIVE
    assert h.observe(grey(50), 1, 100.0, 110.0) == ACTIVE  # 10 s old: not yet
    assert h.observe(grey(50), 1, 100.0, 110.5) == STALE
    info = h.info(120.5)
    assert info["state"] == STALE and info["age_s"] == 20.5 and info["for_s"] == 10.0
    assert h.observe(grey(80), 2, 121.0, 121.0) == ACTIVE and h.info(121.0)["for_s"] == 0.0


def test_a_picture_that_was_quiet_before_an_outage_is_not_frozen_the_moment_the_stream_returns():
    h = StreamHealth()
    for t in range(10):  # 10 s of a static night scene
        h.observe(grey(30), t + 1, float(t), float(t))
    assert h.observe(grey(30), 11, 300.0, 300.0) == ACTIVE  # back after 290 s: the anchor was dropped, not carried across the gap
    assert h.freeze.frozen_for == 0.0


def test_health_from_env_reads_the_thresholds_and_ignores_bad_values():
    h = health_from_env({"CARPARK_STALE_S": "4", "CARPARK_FREEZE_DIFF": "0.2", "CARPARK_FREEZE_S": "x"})
    assert (h.stale_s, h.freeze.diff_max, h.freeze.freeze_s) == (4.0, 0.2, 20.0)
    assert health_from_env({"CARPARK_FREEZE_S": "0"}).freeze.freeze_s == 0.0


def test_reconnect_waits_are_jittered_up_to_half_the_backoff_and_differ_between_cameras():
    def reader(seed):
        r = object.__new__(LatestFrameReader)  # (no thread, no capture)
        r._rng = random.Random(seed)
        return r
    waits = {seed: [reader(seed).delay(b) for b in (1.0, 2.0, 4.0, 30.0)] for seed in range(5)}
    for ws in waits.values():
        for w, b in zip(ws, (1.0, 2.0, 4.0, 30.0)):
            assert b <= w <= b * 1.5
    assert len({tuple(ws) for ws in waits.values()}) == 5  # not in lockstep


# --- the worker -----------------------------------------------------------------------------------------------------------

class ScriptedReader:
    """latest() returns what the test last set; `vary` makes every frame a different picture."""

    def __init__(self):
        self.seq, self.t, self.frame, self.status, self.stopped = 0, 0.0, None, "streaming", False

    def push(self, t, frame):
        self.seq, self.t, self.frame = self.seq + 1, t, frame

    def latest(self):
        return self.frame, self.seq, self.t

    def stop(self):
        self.stopped = True


def rig(tmp_path):
    big = {"id": 1, "name": "G1", "poly": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}
    store, clock, reader, dets = LiveStore(tmp_path / "h.db"), Clock(), ScriptedReader(), []
    store.set_config({"enabled": True, "run": run_cfg(fixed_mode=True, fixed=[big], db_detector="fake | fixed")})

    def detector(spec, frame):
        dets.append(FakeDetector((CAR,)))
        return dets[-1], False, "fake detector", True

    w = Worker(store, tmp_path, tmp_path / "h.db", raw_dir=tmp_path / "raw", default_url="rtsp://u:p@cam/Channels/101",
               owner="w1", reader_factory=lambda url: reader, detector_factory=detector, clock=clock)

    def tick(vary=True, dt=1.0, advance=True):
        """One second: a new frame (a new picture if vary, else the same one) unless advance=False, then a worker step."""
        if advance:
            reader.push(clock.t, grey((int(clock.t) * 37) % 200 + 20) if vary else grey(77))
        w.step()
        clock.t += dt
        return store.state(images=False)["snapshot"]

    return w, store, clock, reader, dets, tick


def open_sessions(w):
    return len(w.db.sessions(open_only=True))


def test_a_stale_stream_shows_unknown_leaves_the_sessions_open_and_recovers(tmp_path, capsys):
    w, store, clock, reader, dets, tick = rig(tmp_path)
    for _ in range(8):
        snap = tick()
    assert snap["stream"]["state"] == ACTIVE and snap["counts"]["busy"] == 1 and open_sessions(w) == 1
    calls = dets[0].calls
    for _ in range(15):  # the camera is gone: no new frame
        snap = tick(advance=False)
    assert snap["stream"]["state"] == STALE and snap["waiting"]
    assert snap["counts"] == {**snap["counts"], "busy": 0, "free": 0, "unknown": 1}
    assert [s["state"] for s in snap["slots"]] == ["UNKNOWN"] and snap["detections"] == []
    assert open_sessions(w) == 1 and dets[0].calls == calls  # nothing processed, nothing closed
    out = capsys.readouterr().out
    assert out.count("stream stale") == 1  # one line, not one per step
    snap = tick()  # the camera is back
    assert snap["stream"]["state"] == ACTIVE and snap["counts"]["busy"] == 1 and dets[0].calls == calls + 1
    assert capsys.readouterr().out.count("live again") == 1 and open_sessions(w) == 1


def test_a_frozen_stream_is_not_counted_from_and_the_history_is_left_alone(tmp_path, capsys):
    w, store, clock, reader, dets, tick = rig(tmp_path)
    for _ in range(8):
        tick()
    assert open_sessions(w) == 1
    calls = dets[0].calls
    for _ in range(30):  # frames keep arriving (the NVR repeats the last one), all the same picture
        snap = tick(vary=False)
    assert snap["stream"]["state"] == FROZEN and snap["stream"]["same_s"] >= 20
    assert snap["counts"]["unknown"] == 1 and snap["counts"]["busy"] == 0
    assert dets[0].calls <= calls + 21  # processing stopped once the picture had been the same for 20 s
    assert open_sessions(w) == 1 and "stream frozen" in capsys.readouterr().out
    snap = tick(vary=True)  # the picture moves again
    assert snap["stream"]["state"] == ACTIVE and snap["counts"]["busy"] == 1 and open_sessions(w) == 1


def test_the_stream_health_rides_every_snapshot_and_a_new_session_starts_clean(tmp_path):
    w, store, clock, reader, dets, tick = rig(tmp_path)
    snap = tick()
    assert set(snap["stream"]) == {"state", "age_s", "for_s", "diff", "same_s"}
    for _ in range(30):
        tick(vary=False)
    assert w.health.state == FROZEN
    store.set_config({"enabled": True, "run": run_cfg(camera="cam2", fixed_mode=True, db_detector="fake | fixed",
                                                      fixed=[{"id": 1, "name": "A", "poly": [[0, 0], [1, 0], [1, 1], [0, 1]]}])})
    tick()
    assert w.health.state == ACTIVE and w.health.freeze.frozen_for == 0.0
