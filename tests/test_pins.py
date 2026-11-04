"""Ground pins (geometry/pins.py): a camera from the four clicked corners of a ground rectangle of known size. Synthetic cameras, exact corners and
clicks with pixel noise; no models, no network, no ./data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.geometry import CameraModel, to_ground, to_image  # noqa: E402
from carparking.geometry.calib import camera_from_geocalib, height_from_reference  # noqa: E402
from carparking.geometry.pins import PinsError, camera_from_rectangle  # noqa: E402

W, H = 1920, 1080
CAMS = {
    "angled": CameraModel(f=1500.0, cx=W / 2, cy=H / 2, pitch=np.radians(32), roll=np.radians(0.5), h=6.0),
    "low": CameraModel(f=2530.0, cx=W / 2, cy=H / 2, pitch=np.radians(14), roll=np.radians(-0.2), h=2.8),
    "steep": CameraModel(f=1100.0, cx=W / 2, cy=H / 2, pitch=np.radians(55), roll=np.radians(2.0), h=9.0),
    "rolled": CameraModel(f=1700.0, cx=W / 2, cy=H / 2, pitch=np.radians(28), roll=np.radians(6.0), h=5.0),
}


def corners(a, b, x0=-1.0, y0=9.0):
    """A rectangle on the ground: a metres along x, b metres along y, clicked near-left, near-right, far-right, far-left."""
    return [(x0, y0), (x0 + a, y0), (x0 + a, y0 + b), (x0, y0 + b)]


@pytest.mark.parametrize("name", list(CAMS))
def test_the_pose_is_recovered_from_four_exact_corners(name):
    cam = CAMS[name]
    px = to_image(cam, corners(2.5, 5.0, y0=8.0 if name != "low" else 9.0))
    got = camera_from_rectangle(px, 2.5, 5.0, (W, H))
    assert got["f"] == pytest.approx(cam.f, rel=0.02)
    assert got["h"] == pytest.approx(cam.h, rel=0.02)
    assert np.degrees(got["pitch"]) == pytest.approx(np.degrees(cam.pitch), abs=0.5)
    assert np.degrees(got["roll"]) == pytest.approx(np.degrees(cam.roll), abs=0.5)
    assert got["reprojection_px"] < 0.5


@pytest.mark.parametrize("name", ["angled", "low", "rolled"])
def test_the_recovered_camera_measures_other_ground_points_in_metres(name):
    cam = CAMS[name]
    px = to_image(cam, corners(2.5, 5.0, y0=9.0))
    got = camera_from_rectangle(px, 2.5, 5.0, (W, H))
    est = camera_from_geocalib(got["f"], W, H, got["up"]).with_height(got["h"])
    a, b = to_image(cam, [(-3.0, 14.0), (2.0, 18.0)])  # two ground points 6.4 m apart, away from the pins
    d = np.linalg.norm(np.diff(to_ground(est, [a, b]), axis=0))
    assert d == pytest.approx(np.hypot(5.0, 4.0), rel=0.04)
    # and the typed-distance route (how build_record takes it) gives the same height
    ref = height_from_reference(camera_from_geocalib(got["f"], W, H, got["up"]), px[0], px[1], 2.5)
    assert ref == pytest.approx(got["h"], rel=0.01)


def test_clicks_a_couple_of_pixels_off_still_give_a_usable_camera():
    cam = CAMS["angled"]
    rng = np.random.default_rng(5)
    errs = []
    for _ in range(25):
        px = to_image(cam, corners(2.5, 5.0, y0=8.0)) + rng.normal(0, 2.0, (4, 2))
        got = camera_from_rectangle(px, 2.5, 5.0, (W, H))
        errs.append((got["h"] / cam.h - 1, got["f"] / cam.f - 1))
    h_err, f_err = np.abs(np.array(errs)).T
    assert np.median(h_err) < 0.06 and np.median(f_err) < 0.12


def test_a_bigger_rectangle_is_steadier_than_a_small_one():
    cam = CAMS["angled"]
    rng = np.random.default_rng(1)
    def spread(a, b):
        out = []
        for _ in range(30):
            px = to_image(cam, corners(a, b, x0=-a / 2, y0=7.0)) + rng.normal(0, 2.0, (4, 2))
            out.append(camera_from_rectangle(px, a, b, (W, H))["h"] / cam.h - 1)
        return float(np.std(out))
    assert spread(6.0, 12.0) < spread(2.5, 5.0)


def test_the_corners_must_be_clicked_in_order_around_the_rectangle():
    px = to_image(CAMS["angled"], corners(2.5, 5.0, y0=8.0))
    crossed = [px[0], px[2], px[1], px[3]]
    with pytest.raises(PinsError, match="in order"):
        camera_from_rectangle(crossed, 2.5, 5.0, (W, H))


def test_refusals_say_what_to_fix():
    px = to_image(CAMS["angled"], corners(2.5, 5.0, y0=8.0))
    with pytest.raises(PinsError, match="positive"):
        camera_from_rectangle(px, 0, 5.0, (W, H))
    line = [(100, 500), (300, 500), (500, 500), (700, 500)]  # all on one line
    with pytest.raises(PinsError):
        camera_from_rectangle(line, 2.5, 5.0, (W, H))
    head_on = [(500, 300), (900, 300), (900, 700), (500, 700)]  # a square straight on: no perspective, no focal length
    with pytest.raises(PinsError, match="focal length|impossible"):
        camera_from_rectangle(head_on, 2.5, 2.5, (W, H))
