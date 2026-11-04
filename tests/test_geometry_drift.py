"""The drift guard's comparison (carparking/geometry/drift.py) on synthetic scenes with known shifts. The thresholds themselves were
tuned on real frames of the live camera (see ACTIVITY_LOG): shifts measured within ~1 px, brightness / blur / jitter quiet, and a
frame from another day with other light answers "inconclusive" (not "drift")."""
import json

import cv2
import numpy as np
import pytest

from carparking.geometry import drift

W, H = 1920, 1080


def scene(seed=1, w=W, h=H, texture=900):
    """A static background with plenty of corners: random rectangles, discs and lines on a grey ground."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 120, np.uint8)
    for _ in range(texture):
        x, y = int(rng.integers(0, w - 90)), int(rng.integers(0, h - 90))
        sx, sy = int(rng.integers(12, 80)), int(rng.integers(12, 80))
        color = tuple(int(c) for c in rng.integers(20, 235, 3))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(img, (x, y), (x + sx, y + sy), color, -1)
        elif kind == 1:
            cv2.circle(img, (x + sx // 2, y + sy // 2), sx // 2, color, -1)
        else:
            cv2.line(img, (x, y), (x + sx, y + sy), color, 3)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def warp(img, dx=0.0, dy=0.0, deg=0.0):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


@pytest.fixture(scope="module")
def base():
    img = scene()
    return img, drift.extract_reference(img)


def test_the_reference_is_a_small_storable_record(base):
    img, ref = base
    assert ref["size"] == [W, H] and ref["n"] >= 300 and len(ref["kp"]) == ref["n"]
    assert len(ref["desc"]) < 40_000  # base64 of N x 32 bytes: a few tens of KB with the version
    assert json.loads(json.dumps(ref)) == ref  # JSON all the way: it lives inside the geometry record
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in ref["kp"])
    top, bottom = [y for _, y in ref["kp"] if y < 0.10], [y for _, y in ref["kp"] if y > 0.92]
    assert top == [] and bottom == []  # the camera's date / time and caption strips are left out


@pytest.mark.parametrize("dx,dy,expected", [(0, 0, "ok"), (3, 2, "ok"), (7, 0, "ok"), (25, 0, "drift"), (0, 30, "drift"), (60, 40, "drift")])
def test_a_shift_is_measured_in_pixels(base, dx, dy, expected):
    img, ref = base
    r = drift.compare(ref, warp(img, dx, dy))
    assert r["status"] == expected and r["matches"] > 100
    # the 90th percentile of the matched points' movement: the shift itself (plus a pixel or three of match noise)
    assert abs(r["shift_px"] - np.hypot(dx, dy)) < 4.5 and r["shift_frac"] == pytest.approx(r["shift_px"] / W, abs=2e-4)
    json.dumps(r)  # plain Python numbers: it is stored in events and published in the snapshot


def test_a_rotation_is_caught_though_features_near_the_pivot_barely_move(base):
    img, ref = base
    assert drift.compare(ref, warp(img, deg=0.3))["status"] == "ok"
    assert drift.compare(ref, warp(img, deg=3.0))["status"] == "drift"


def test_light_noise_and_blur_are_not_movement(base):
    img, ref = base
    rng = np.random.default_rng(3)
    for frame in (np.clip(img * 0.55, 0, 255).astype(np.uint8),  # dusk
                  np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8),  # sensor noise
                  cv2.GaussianBlur(img, (0, 0), 3)):  # soft focus
        assert drift.compare(ref, frame)["status"] == "ok"


def test_vehicles_and_the_date_do_not_matter_when_masked(base):
    img, ref = base
    now = img.copy()
    cars = [(300, 400, 700, 650), (1100, 500, 1500, 760)]
    rng = np.random.default_rng(9)
    for x1, y1, x2, y2 in cars:  # cars that were not there when the reference was taken
        now[y1:y2, x1:x2] = rng.integers(0, 255, (y2 - y1, x2 - x1, 3), np.uint8)
    cv2.putText(now, "10-23-2026 Fri 10:10:27", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)  # the OSD clock
    assert drift.compare(ref, now, cars)["status"] == "ok"
    moved = warp(now, 22, 0)
    shifted_cars = [(x1 + 22, y1, x2 + 22, y2) for x1, y1, x2, y2 in cars]
    r = drift.compare(ref, moved, shifted_cars)
    assert r["status"] == "drift" and abs(r["shift_px"] - 22) < 4.5


def test_burned_in_text_cannot_vouch_for_a_camera_that_moved():
    # the caption "Front Parking 02" stays where it is when the camera moves: matches on it would say "no movement"
    a = scene(5)
    cv2.putText(a, "Front Parking 02", (900, 1010), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    ref = drift.extract_reference(a)
    b = warp(a, 40, 0)
    cv2.putText(b, "Front Parking 02", (900, 1010), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)  # drawn by the camera again
    assert drift.compare(ref, b)["status"] == "drift"


def test_what_cannot_be_judged_is_inconclusive_never_an_alarm(base):
    img, ref = base
    other = drift.compare(ref, scene(seed=77))  # another scene altogether (or the same camera at another time of day)
    assert other["status"] == "inconclusive" and other["shift_px"] is None and other["why"]
    assert drift.compare(ref, np.zeros_like(img))["status"] == "inconclusive"  # night
    assert drift.compare(ref, np.full_like(img, 255))["status"] == "inconclusive"  # glare
    assert drift.compare(ref, None)["status"] == "inconclusive"
    assert drift.compare(ref, cv2.resize(img, (1000, 1000)))["why"].startswith("the frame has another aspect")
    assert drift.compare(None, img)["status"] == "inconclusive"
    bare = drift.extract_reference(np.full_like(img, 100))  # a scene with nothing to hold on to
    assert bare["n"] == 0 and drift.compare(bare, img)["why"] == "no usable reference"


def test_matches_confined_to_a_corner_say_nothing_about_the_rest():
    a = np.full((H, W, 3), 110, np.uint8)
    patch = scene(8, 500, 260, 250)
    a[600:860, 1300:1800] = patch  # the only texture in the whole picture, in one corner
    ref = drift.extract_reference(a)
    assert ref["n"] >= drift.MIN_REFERENCE
    r = drift.compare(ref, warp(a, 30, 0))
    assert r["status"] == "inconclusive" and "cover too little" in r["why"]


def test_the_same_picture_at_another_resolution_is_the_same_picture(base):
    img, ref = base
    small = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)  # the worker's frame may differ from the still
    r = drift.compare(ref, small)
    assert r["status"] == "ok"
    assert drift.compare(ref, warp(small, 18, 0))["status"] == "drift"  # 18 px of 1280 = 27 px at 1920
