"""The drift guard and the occupancy history on the worker's feed (carparking/geometry_live.py): a confirmed shift marks the active
version stale and the camera falls back; what cannot be judged never does; a calibration accepted meanwhile is never staled by a
stale check; no failure reaches the worker; history is sampled regularly and on change, pruned, and a dead store costs nothing."""
import json

import numpy as np
import pytest

from carparking.geometry import drift as drift_mod
from carparking.geometry_live import POLL, GeometryFeed, feed_from_env
from carparking.geometry_store import SAMPLE_KEEP_DAYS
from carparking.live import LiveStore
from test_geometry_drift import scene, warp
from test_geometry_live import CARS, H, W, Clock, box_of, det, record

DRIFT_EVERY = 30.0
BLACK = np.zeros((H, W, 3), np.uint8)


@pytest.fixture
def store(tmp_path):
    return LiveStore(tmp_path / "live.db")


def drift_feed(store, clock, with_reference=True, **kw):
    """An active calibration (with the stored reference of the textured scene, or without: a version made before the guard) and its feed."""
    rec = record()
    if with_reference:
        rec["reference"] = drift_mod.extract_reference(scene())
    g = store.geometry("cam-1")
    g.create(rec, now=clock())  # on the same (fake) clock as the feed: the event log orders by time
    g.activate(1, now=clock())
    logs = []
    return GeometryFeed(lambda: store.geometry("cam-1"), "cam-1", clock=clock, log=logs.append, drift_every=DRIFT_EVERY, drift_confirm=3, **kw), logs


def look(feed, clock, frame, dt=DRIFT_EVERY):
    clock.t += dt
    feed.observe(frame, [], clock())
    return feed.drift


def cars_dets(n=3):
    return [det(box_of(*c)) for c in CARS[:n]]


def test_a_confirmed_shift_marks_the_version_stale_and_the_camera_falls_back(store):
    clock = Clock()
    feed, logs = drift_feed(store, clock)
    still, moved = scene(), warp(scene(), 28, 0)
    assert feed.metrics(cars_dets(), [W, H], clock())["drift_guard"] is True
    assert look(feed, clock, still)["status"] == "ok" and look(feed, clock, still)["baseline"] == "calibration"
    json.dumps(feed.metrics(cars_dets(), [W, H], clock()))  # the whole metrics payload with the drift status in it
    assert look(feed, clock, moved)["status"] == "drift"  # strike 1: suspected, not yet acted on
    assert store.geometry("cam-1").active_version() == 1 and feed.metrics(cars_dets(), [W, H], clock()) is not None
    assert look(feed, clock, moved)["status"] == "drift"  # strike 2
    assert look(feed, clock, still)["status"] == "ok"  # it was a truck: the strikes are forgotten
    for _ in range(3):  # three in a row: the camera moved
        look(feed, clock, moved)
    g = store.geometry("cam-1")
    assert g.active_version() is None and g.get(1)["state"] == "stale"
    assert "drift: the background moved" in g.get(1)["note"] and "confirmed on 3 checks" in g.get(1)["note"]
    assert feed.drift["status"] == "stale"
    json.dumps(feed.drift)  # what is published must be JSON (a numpy float here once silently dropped the events)
    assert feed.metrics(cars_dets(), [W, H], clock()) is None  # graceful fallback: the legacy snapshot
    assert any("marked STALE" in m for m in logs)
    kinds = [e["kind"] for e in reversed(g.events(limit=50))]
    assert kinds == ["draft", "activated", "drift_suspected", "drift_cleared", "drift_suspected", "stale"]
    stale = g.events(limit=1)[0]
    assert stale["kind"] == "stale" and stale["ver"] == 1
    assert 25 <= float(stale["detail"]["reason"].split("moved ")[1].split(" px")[0]) <= 32
    clock.t += 600  # and it stays that way until a new calibration is accepted: nothing flaps
    assert feed.metrics(cars_dets(), [W, H], clock()) is None and look(feed, clock, still) is not None
    assert g.active_version() is None


def test_the_guard_looks_only_every_so_often(store, monkeypatch):
    clock, calls = Clock(), []
    feed, _ = drift_feed(store, clock)
    real = drift_mod.compare
    monkeypatch.setattr(drift_mod, "compare", lambda *a, **k: calls.append(1) or real(*a, **k))
    for _ in range(100):  # a frame a second for 100 s
        clock.t += 1.0
        feed.observe(scene(), [], clock())
    assert len(calls) == 4  # at 1, 31, 61 and 91 s: 30 s apart


def test_what_cannot_be_judged_never_stales_a_calibration(store):
    clock = Clock()
    feed, _ = drift_feed(store, clock)
    for _ in range(12):  # a night: black frames, then another scene entirely
        assert look(feed, clock, BLACK)["status"] == "inconclusive"
    for _ in range(6):
        assert look(feed, clock, scene(seed=99))["status"] == "inconclusive"
    assert store.geometry("cam-1").active_version() == 1  # still active, never an alarm
    assert [e["kind"] for e in store.geometry("cam-1").events(limit=10)] == ["activated", "draft"]


def test_strikes_do_not_carry_over_a_long_inconclusive_night(store):
    clock = Clock()
    feed, _ = drift_feed(store, clock)
    moved = warp(scene(), 28, 0)
    look(feed, clock, moved)
    look(feed, clock, moved)  # two strikes
    for _ in range(12):  # twelve inconclusive checks (6 minutes at 30 s): the strikes expire
        look(feed, clock, BLACK)
    look(feed, clock, moved)  # one strike again: not the third
    assert store.geometry("cam-1").active_version() == 1


def test_a_version_made_before_the_guard_gets_a_baseline_from_the_first_frame(store):
    clock = Clock()
    feed, logs = drift_feed(store, clock, with_reference=False)
    still, moved = scene(), warp(scene(), 30, 0)
    first = look(feed, clock, still)
    assert first["status"] == "ok" and first["baseline"] == "worker" and any("drift baseline" in m for m in logs)
    assert feed.metrics(cars_dets(1), [W, H], clock())["drift_guard"] is False  # the version itself carries no reference
    assert look(feed, clock, still)["status"] == "ok"
    for _ in range(3):
        look(feed, clock, moved)
    assert store.geometry("cam-1").active_version() is None  # the baseline works the same way


def test_a_bare_scene_gives_no_baseline_and_no_alarm(store):
    clock = Clock()
    feed, _ = drift_feed(store, clock, with_reference=False)
    flat = np.full((H, W, 3), 90, np.uint8)
    assert look(feed, clock, flat)["status"] == "inconclusive"
    assert look(feed, clock, flat)["status"] == "inconclusive" and store.geometry("cam-1").active_version() == 1


def test_a_calibration_accepted_while_a_check_was_running_is_not_staled(store):
    clock = Clock()
    feed, _ = drift_feed(store, clock)
    feed.refresh(clock())
    old = feed.active
    rec = record()
    rec["reference"] = drift_mod.extract_reference(scene())
    g = store.geometry("cam-1")
    g.create(rec)
    g.activate(2)  # the Admin accepted a new calibration just now
    feed._declare_drift(old, {"shift_px": 30.0, "shift_frac": 0.016}, clock())  # the check was about version 1
    assert g.active_version() == 2 and g.get(2)["state"] == "active"
    clock.t += POLL
    assert feed.refresh(clock()).version == 2 and feed.drift is None  # a fresh check for the new calibration


def test_the_guard_and_the_history_never_raise_into_the_worker(store):
    clock = Clock()
    feed, logs = drift_feed(store, clock)
    broken = {"on": False}
    good = feed.open_store

    def flaky():
        if broken["on"]:
            raise RuntimeError("database is restarting")
        return good()

    feed.open_store = flaky
    feed.metrics(cars_dets(), [W, H], clock())
    broken["on"] = True
    moved = warp(scene(), 28, 0)
    for _ in range(5):
        look(feed, clock, moved)  # the strikes reach the threshold, but the store is down: logged, retried
    feed.metrics(cars_dets(), [W, H], clock())
    failures = [m for m in logs if "failed" in m]
    assert failures and all("database is restarting" in m for m in failures)
    assert store.geometry("cam-1").active_version() == 1  # (after a minute without the store the feed gave the calibration up: quiet)
    broken["on"] = False
    for _ in range(3):  # the store is back: the calibration is loaded afresh and three confirmed checks mark it
        look(feed, clock, moved)
    assert store.geometry("cam-1").active_version() is None
    feed.observe(None, [], clock())  # no frame (a waiting beat): nothing happens
    GeometryFeed(lambda: None, "x", clock=clock, log=logs.append).observe(object(), [], clock())  # a garbage frame: swallowed


def test_occupancy_history_is_sampled_regularly_and_when_it_changes(store):
    clock = Clock(1_800_000_000.0)
    feed, _ = drift_feed(store, clock, sample_every=60.0)
    cars = cars_dets()
    g = store.geometry("cam-1")
    for _ in range(130):  # a frame a second for 130 s, three cars throughout: a sample a minute
        feed.metrics(cars, [W, H], clock())
        clock.t += 1.0
    rows = g.history(1_800_000_000, 1_800_000_200, 30)
    assert sum(r["samples"] for r in rows) == 3 and {r["occupied_max"] for r in rows} == {3}
    feed.metrics(cars[:1], [W, H], clock())  # a car left: one sample at once
    clock.t += 5.0
    feed.metrics(cars[:1], [W, H], clock())  # (not again within the minimum gap)
    clock.t += 6.0
    feed.metrics(cars[:2], [W, H], clock())  # another change after the gap: another sample
    latest = g.history(1_800_000_000, 1_800_000_300, 1)
    assert [r["occupied_max"] for r in latest][-2:] == [1, 2] and len(latest) == 5
    assert latest[-1]["ver"] == 1 and latest[-1]["capacity_max"] == len(feed.active.bays)


def test_history_rows_are_pruned_and_a_dead_store_costs_nothing(store):
    clock = Clock(1_800_000_000.0)
    feed, logs = drift_feed(store, clock, sample_every=30.0)
    g = store.geometry("cam-1")
    g.add_sample(1, {"capacity": 3, "occupied": 1}, now=clock() - (SAMPLE_KEEP_DAYS + 2) * 86400)
    cars = cars_dets()
    feed.metrics(cars, [W, H], clock())
    assert g.history(0, clock() + 1, 86400 * 400)[0]["samples"] == 1  # the old one was pruned with the first sample
    feed.open_store = lambda: (_ for _ in ()).throw(RuntimeError("down"))
    clock.t += 40.0  # a sample is due; the poll fails too, but the last good calibration is kept for a minute
    assert feed.metrics(cars, [W, H], clock()) is not None  # metrics still served from the loaded calibration
    assert any("storing a history sample failed" in m for m in logs)


def test_drift_settings_come_from_the_environment():
    on = {"CARPARK_GEOMETRY_ENABLED": "1", "CARPARK_CAMERA_ID": "c"}
    feed = feed_from_env({**on, "CARPARK_DRIFT_EVERY": "5", "CARPARK_DRIFT_CONFIRM": "2"}, object(), log=lambda m: None)
    assert (feed.drift_every, feed.drift_confirm) == (5.0, 2)
    bad = feed_from_env({**on, "CARPARK_DRIFT_EVERY": "soon"}, object(), log=lambda m: None)
    assert (bad.drift_every, bad.drift_confirm, bad.drift_guard) == (30.0, 10, False)  # a typo falls back to the defaults, never a crash
    logs = []
    off = feed_from_env(on, object(), log=logs.append)
    assert off.drift_guard is False and any("drift guard OFF by default" in m for m in logs)  # off unless asked for
    for value in ("0", "off", "False", " no "):
        assert feed_from_env({**on, "CARPARK_DRIFT_GUARD": value}, object(), log=lambda m: None).drift_guard is False
    for value in ("1", "on", "TRUE", "yes"):
        assert feed_from_env({**on, "CARPARK_DRIFT_GUARD": value}, object(), log=lambda m: None).drift_guard is True
    assert GeometryFeed(lambda: None, "c").drift_guard is True  # the class itself guards unless told otherwise (tests, library use)


def test_the_settings_switch_turns_the_guard_on_and_off_while_the_worker_runs(tmp_path):
    from test_live import Clock as WorkerClock
    from test_live import make_worker, run_cfg, steps

    class Spy:
        drift_guard = False  # what the environment said at start (off by default)

        def observe(self, frame, detections, now):
            pass

        def metrics(self, detections, shape, now):
            return None

    store, clock = LiveStore(tmp_path / "h.db"), WorkerClock()
    store.set_config({"enabled": True, "run": run_cfg()})
    spy = Spy()
    (tmp_path / "w").mkdir()
    w, _, _ = make_worker(tmp_path / "w", store, clock, geometry=spy)
    steps(w, clock, 1)
    assert spy.drift_guard is False  # no switch ever set: the environment's default stands
    store.set_option("drift_guard", "1")  # the dashboard's Settings: on
    steps(w, clock, 1)
    assert spy.drift_guard is True
    store.set_option("drift_guard", "0")
    steps(w, clock, 1)
    assert spy.drift_guard is False
    store.set_option("drift_guard", "")  # cleared: back to the environment's default
    spy.drift_guard = True
    steps(w, clock, 1)
    assert spy.drift_guard is False


def test_with_the_guard_off_a_moved_camera_is_never_staled_and_nothing_is_logged(store):
    clock = Clock()
    feed, logs = drift_feed(store, clock, drift_guard=False)
    moved = warp(scene(), 28, 0)
    assert feed.metrics(cars_dets(), [W, H], clock())["drift_guard"] is True  # the calibration has a reference: only the checks are off
    for _ in range(12):  # four times the default confirmation window of a moved picture
        assert look(feed, clock, moved) is None  # no check, nothing published
    g = store.geometry("cam-1")
    assert g.active_version() == 1 and g.get(1)["state"] == "active"
    assert [e["kind"] for e in g.events(limit=50)] == ["activated", "draft"]  # no drift_suspected, no stale
    assert feed.metrics(cars_dets(), [W, H], clock()) is not None and not any("STALE" in m for m in logs)
    assert feed.refresh(clock()).version == 1  # the active version is still read (an activation from the setup page is seen)


def test_the_worker_hands_every_frame_to_the_guard_and_a_failing_guard_costs_no_frame(tmp_path, capsys):
    from test_live import Clock as WorkerClock
    from test_live import make_worker, run_cfg, steps

    class Spy:
        def __init__(self, boom=False):
            self.seen, self.boom = [], boom

        def observe(self, frame, detections, now):
            self.seen.append((frame.shape, [d["kind"] for d in detections]))
            if self.boom:
                raise RuntimeError("guard exploded")

        def metrics(self, detections, shape, now):
            return None

    for boom in (False, True):
        store, clock = LiveStore(tmp_path / f"h{boom}.db"), WorkerClock()
        store.set_config({"enabled": True, "run": run_cfg()})
        spy = Spy(boom)
        (tmp_path / f"w{boom}").mkdir()
        w, _, _ = make_worker(tmp_path / f"w{boom}", store, clock, geometry=spy)
        steps(w, clock, 6)
        snap = store.state(images=False)["snapshot"]
        assert len(spy.seen) == 6 and spy.seen[0] == ((1000, 2000, 3), ["vehicle"])  # the full frame and its detections, every frame
        assert snap["seq"] == 6 and snap["counts"]["busy"] == 1 and "geometry" not in snap  # tracking and publishing untouched
    assert capsys.readouterr().out.count("geometry metrics failed (RuntimeError: guard exploded)") == 1


def test_after_a_camera_move_a_reanchored_calibration_takes_over_with_a_fresh_guard(store):
    from carparking.geometry.reanchor import reanchor_record
    from test_geometry_reanchor import truth, turned_picture
    from test_geometry_live import CAM
    clock = Clock()
    feed, _ = drift_feed(store, clock)
    still = scene()
    _, _, R = truth(CAM, 2.0, -1.0, 4.0)
    moved = turned_picture(still, R)
    for _ in range(3):  # the camera is bumped: the guard stales version 1 and the legacy view takes over
        look(feed, clock, moved)
    g = store.geometry("cam-1")
    assert g.active_version() is None and feed.metrics(cars_dets(), [W, H], clock()) is None
    rec, report = reanchor_record(g.get(1)["record"], moved)  # the Admin re-anchors from a fresh still and accepts
    rec["reanchor"]["parent"] = 1
    g.create(rec, now=clock())
    g.activate(2, now=clock())
    clock.t += POLL
    assert feed.refresh(clock()).version == 2 and feed.drift is None  # picked up with no restart, strikes forgotten
    assert feed.metrics(cars_dets(), [W, H], clock()) is not None
    assert look(feed, clock, moved)["status"] == "ok"  # judged against where the camera is now: no new alarm
    for _ in range(4):
        look(feed, clock, moved)
    assert g.active_version() == 2 and g.get(1)["state"] == "stale"
    kinds = [(e["kind"], e["ver"]) for e in reversed(g.events(limit=50))]
    assert ("stale", 1) in kinds and ("activated", 2) in kinds
    assert [e for e in g.events(limit=50) if e["kind"] == "activated"][0]["detail"]["parent"] == 1
