"""The worker's read of the active calibration and the metric data it publishes (carparking/geometry_live.py): off by default,
silent when uncalibrated, picks up an activation without a restart, survives a failing store, and the numbers (footprints,
distances, bay capacity) on a synthetic scene with known geometry."""
import json

import cv2
import numpy as np
import pytest

from carparking.geometry import CameraModel, to_image
from carparking.geometry.pipeline import build_record
from carparking.geometry_live import POLL, STALE_AFTER, GeometryFeed, feed_from_env
from carparking.geometry_store import geometry_enabled
from carparking.live import LiveStore

W, H = 1920, 1080
CAM = CameraModel(f=1500.0, cx=W / 2, cy=H / 2, pitch=np.radians(32), roll=0.0, h=6.0)
GROUND = [(-4, 9), (4, 9), (12, 24), (-12, 24)]
CARS = [(0.0, 11.0), (0.0, 13.6), (0.0, 16.2)]  # side by side along the wall, each pointing across it (long axis along X)
UP = (-np.array([0.0, np.cos(CAM.pitch), np.sin(CAM.pitch)])).tolist()


def dense(ground, n=80):
    g = np.array(ground, float)
    return np.vstack([a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None] for a, b in zip(g, np.roll(g, -1, axis=0))])


def box_of(x, y, half=2.25, height=1.45):
    """The pixel box of an upright car-sized billboard at ground point (x, y)."""
    base = to_image(CAM, [(x - half, y), (x + half, y)])
    top = to_image(CAM.with_height(CAM.h - height), [(x - half, y), (x + half, y)])
    pts = np.vstack([base, top])
    return (*pts.min(axis=0), *pts.max(axis=0))


def det(box, label="car", kind="vehicle", frame=(W, H)):
    return {"label": label, "conf": 0.9, "src": "", "kind": kind, "box": [round(b / s, 5) for b, s in zip(box, (frame[0], frame[1]) * 2)]}


def record(**kw):
    boxes = [box_of(x, y) for x, y in CARS]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(to_image(CAM, dense(GROUND))).astype(np.int32)], 255)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = (max(cs, key=cv2.contourArea).reshape(-1, 2) / (W, H)).tolist()
    return build_record(width=W, height=H, polygon=outline, f=CAM.f, up=UP, depth={"h": 6.0}, vehicles=boxes, obstacles=boxes, **kw)


@pytest.fixture
def store(tmp_path):
    return LiveStore(tmp_path / "live.db")


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def feed_of(store, clock, log=None):
    logs = [] if log is None else log
    return GeometryFeed(lambda: store.geometry("cam-1"), "cam-1", clock=clock, log=logs.append), logs


def test_off_by_default_and_needs_the_camera():
    for off in ({}, {"CARPARK_GEOMETRY_ENABLED": ""}, {"CARPARK_GEOMETRY_ENABLED": "0"}, {"CARPARK_GEOMETRY_ENABLED": "no"}):
        assert not geometry_enabled(off)
    for on in ("1", "true", "YES", " on "):
        assert geometry_enabled({"CARPARK_GEOMETRY_ENABLED": on})
    asked = []

    class Spy:  # the store must not be touched while the flag is off
        def geometry(self, cam):
            asked.append(cam)

    logs = []
    assert feed_from_env({}, Spy()) is None and feed_from_env({"CARPARK_CAMERA_ID": "cam-1"}, Spy()) is None and asked == []
    assert feed_from_env({"CARPARK_GEOMETRY_ENABLED": "1"}, Spy(), log=logs.append) is None  # no camera id: says so
    assert "CARPARK_CAMERA_ID" in logs[0]
    feed = feed_from_env({"CARPARK_GEOMETRY_ENABLED": "1", "CARPARK_CAMERA_ID": "cam-1"}, Spy(), log=logs.append)
    assert isinstance(feed, GeometryFeed) and feed.camera_id == "cam-1"


def test_uncalibrated_means_none_not_an_error(store):
    clock = Clock()
    feed, logs = feed_of(store, clock)
    dets = [det(box_of(*CARS[0]))]
    assert feed.metrics(dets, [W, H]) is None  # nothing in the store at all
    store.geometry("cam-1").create(record())  # a draft: not used
    clock.t += POLL
    assert feed.metrics(dets, [W, H]) is None and feed.errors == 0 and logs == []


def test_an_activation_is_picked_up_without_a_restart_and_the_store_is_not_hammered(store):
    clock, calls = Clock(), []
    g = store.geometry("cam-1")

    class Counting:
        def active_version(self):
            calls.append(1)
            return g.active_version()

        def get(self, ver):
            return g.get(ver)

    logs = []
    feed = GeometryFeed(lambda: Counting(), "cam-1", clock=clock, log=logs.append)
    dets = [det(box_of(*c)) for c in CARS]
    assert feed.metrics(dets, [W, H]) is None and len(calls) == 1
    g.create(record())
    g.create(record())
    g.activate(1)
    for _ in range(10):  # a frame a second: only one look per POLL seconds
        clock.t += 0.5
        feed.metrics(dets, [W, H])
    assert len(calls) == 2  # ten frames in five seconds: the first look, and one more when POLL had passed
    clock.t += POLL
    m = feed.metrics(dets, [W, H])
    assert m["version"] == 1 and m["valid"] and any("version 1 is active" in line for line in logs)
    g.activate(2)  # a newer calibration from the setup page: the same feed object serves it within a poll
    clock.t += POLL
    assert feed.metrics(dets, [W, H])["version"] == 2
    g.rollback()
    clock.t += POLL
    assert feed.metrics(dets, [W, H])["version"] == 1
    g.mark_stale("moved")  # the drift guard: uncalibrated again
    clock.t += POLL
    assert feed.metrics(dets, [W, H]) is None


def test_a_failing_store_never_raises_keeps_the_last_good_one_for_a_minute_and_recovers(store):
    clock = Clock()
    g = store.geometry("cam-1")
    g.create(record())
    g.activate(1)
    broken = [False]

    def open_store():
        if broken[0]:
            raise RuntimeError('relation "camera_geometry" does not exist')
        return g

    logs = []
    feed = GeometryFeed(open_store, "cam-1", clock=clock, log=logs.append)
    dets = [det(box_of(*c)) for c in CARS]
    assert feed.metrics(dets, [W, H])["version"] == 1
    broken[0] = True
    clock.t += POLL
    assert feed.metrics(dets, [W, H])["version"] == 1  # the last good calibration carries on
    for _ in range(5):
        clock.t += POLL
        feed.metrics(dets, [W, H])
    assert len([m for m in logs if "failed" in m]) == 1 and feed.errors >= 2  # one message, not one per poll
    clock.t += STALE_AFTER
    assert feed.metrics(dets, [W, H]) is None  # too long: uncalibrated
    broken[0] = False
    clock.t += POLL
    assert feed.metrics(dets, [W, H])["version"] == 1  # and back


def calibrated(store):
    g = store.geometry("cam-1")
    g.create(record())
    g.activate(1)
    clock = Clock()
    feed, _ = feed_of(store, clock)
    return feed


def test_footprints_distances_and_bay_capacity(store):
    feed = calibrated(store)
    person = det(box_of(4.0, 12.5, half=0.3, height=1.7), label="person", kind="other")
    dets = [det(box_of(*c)) for c in CARS] + [det(box_of(*CARS[0]))] + [person]  # one car seen twice
    m = feed.metrics(dets, [W, H])
    json.dumps(m)  # publishable
    assert m["valid"] and m["version"] == 1 and m["confidence"] in ("LOW", "MEDIUM", "HIGH") and m["area_m2"] > 0
    assert m["counts"]["vehicles"] == 3 and m["counts"]["measured"] == 3 and m["counts"]["beyond_range"] == 0  # the duplicate is one car
    assert len(m["vehicles"]) == 3
    for v in m["vehicles"]:  # footprints are image fractions, a quadrilateral, inside the picture
        poly = np.array(v["poly"])
        assert poly.shape == (4, 2) and 0 <= poly.min() and poly.max() <= 1
    # the bays: one per car (BUSY), the rest FREE; capacity = all bays
    counts = m["counts"]
    assert counts["occupied"] == 3 and counts["capacity"] == len(m["bays"]) >= 3 and counts["free"] == counts["capacity"] - 3
    assert sum(b["state"] == "BUSY" for b in m["bays"]) == 3 and all(b["car"] for b in m["bays"] if b["state"] == "BUSY")
    # distances: each car's nearest neighbour (~0.4 m gaps between side-by-side cars), and the person to the nearest car
    cars_d = [d for d in m["distances"] if d["kind"] == "vehicle"]
    assert len(cars_d) == 2 and all(0.2 <= d["m"] <= 0.7 for d in cars_d)
    obj = [d for d in m["distances"] if d["kind"] == "object"]
    assert len(obj) == 1 and obj[0]["label"] == "person" and 1.2 <= obj[0]["m"] <= 2.2
    for d in m["distances"]:
        assert np.array(d["p"]).shape == (2,) and 0 <= min(d["p"] + d["q"]) and max(d["p"] + d["q"]) <= 1


def test_cars_too_far_or_cut_by_the_edge_are_counted_not_measured(store):
    feed = calibrated(store)
    cut = {"label": "car", "conf": 0.9, "src": "", "kind": "vehicle", "box": [0.0, 0.5, 0.2, 0.7]}  # touches the left edge
    m = feed.metrics([det(box_of(*CARS[0])), cut], [W, H])
    assert m["counts"]["vehicles"] == 2 and m["counts"]["measured"] == 1 and m["counts"]["beyond_range"] == 1
    assert m["distances"] == []  # one measured car: nothing to measure against
    empty = feed.metrics([], [W, H])  # no cars: every bay FREE
    assert empty["counts"]["occupied"] == 0 and empty["counts"]["free"] == empty["counts"]["capacity"] > 0


def test_the_same_picture_at_another_resolution_gives_the_same_metrics_another_aspect_is_refused(store):
    feed = calibrated(store)
    dets = [det(box_of(*c)) for c in CARS]
    a = feed.metrics(dets, [W, H])
    b = feed.metrics(dets, [2 * W, 2 * H])  # the worker's frame may be larger than the still the calibration was made on
    assert b["counts"] == a["counts"] and b["distances"] == a["distances"]
    assert np.allclose([v["poly"] for v in a["vehicles"]], [v["poly"] for v in b["vehicles"]], atol=2e-3)
    wrong = feed.metrics(dets, [2000, 1000])  # 2:1 is not this camera's picture
    assert wrong["valid"] is False and "not the picture" in wrong["reason"] and "counts" not in wrong
    json.dumps(wrong)

