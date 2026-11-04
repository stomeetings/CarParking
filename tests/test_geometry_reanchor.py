"""Re-anchoring a calibration after the camera turned (carparking/geometry/reanchor.py). The pose math is checked against an
independent construction (the new camera built from known pitch / roll / yaw, the true rotation derived from it); the pipeline on
warped synthetic pictures with real feature matching; and every refusal."""
import copy
import json

import cv2
import numpy as np
import pytest

from carparking.geometry import CameraModel, drift, to_image
from carparking.geometry import reanchor as ra
from carparking.geometry.camera import distort_n
from test_geometry_drift import scene
from test_geometry_live import CAM, H, W, record

LENS = CameraModel(f=1900.0, cx=960.0, cy=540.0, pitch=np.radians(14), roll=np.radians(0.8), h=3.2, k1=-0.40)


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def truth(cam0, dpitch_deg, droll_deg, yaw_deg):
    """A camera that turned: pitch / roll changed and the heading yawed. -> (its own model, its axes in the OLD world frame, the
    rotation R with b_new = R b_old)."""
    new = CameraModel(cam0.f, cam0.cx, cam0.cy, cam0.pitch + np.radians(dpitch_deg), cam0.roll + np.radians(droll_deg), cam0.h, cam0.k1, cam0.k2)
    axes = rz(np.radians(yaw_deg)) @ new.R_cw()
    return new, axes, axes.T @ cam0.R_cw()


def true_pixels(cam0, axes, ground):
    """Where the true new camera sees ground points given in the OLD world frame (distortion included)."""
    c = np.c_[np.asarray(ground, float), np.full(len(ground), -cam0.h)] @ axes  # camera coordinates: axes^T p
    pn = c[:, :2] / c[:, 2:3]
    return distort_n(pn, cam0.k1, cam0.k2) * cam0.f + (cam0.cx, cam0.cy)


TURNS = [(0, 0, 0), (2, 0, 0), (-3, 1, 5), (4, -2, -12), (1.5, 0.7, 20), (-4, 2, -25), (0.5, 0, 0.5)]  # (d pitch, d roll, yaw) degrees


@pytest.mark.parametrize("cam0", [CAM, LENS], ids=["pinhole", "wide lens"])
@pytest.mark.parametrize("dp,dr,yaw", TURNS)
def test_the_new_pose_is_recovered_exactly_from_the_rotation(cam0, dp, dr, yaw):
    new, axes, R = truth(cam0, dp, dr, yaw)
    got, psi = ra.reanchor_camera(cam0, R)
    assert np.degrees(got.pitch) == pytest.approx(np.degrees(new.pitch), abs=1e-6)
    assert np.degrees(got.roll) == pytest.approx(np.degrees(new.roll), abs=1e-6)
    assert np.degrees(psi) == pytest.approx(yaw, abs=1e-6)
    assert (got.f, got.cx, got.cy, got.h, got.k1, got.k2) == (cam0.f, cam0.cx, cam0.cy, cam0.h, cam0.k1, cam0.k2)  # optics, height kept
    # the old ground, rotated into the new frame, lands exactly where the true camera sees it
    ground = np.array([(-3.0, 8.0), (2.5, 9.0), (0.0, 12.0), (-6.0, 15.0), (5.0, 20.0)])
    want = true_pixels(cam0, axes, ground)
    ours = to_image(got, ra.rotate_xy(ground, psi))
    ok = np.isfinite(want).all(axis=1)
    assert ok.sum() >= 3 and np.abs(ours[ok] - want[ok]).max() < 1e-6


def test_ground_points_round_trip_through_the_frame_change():
    pts = np.array([(1.0, 2.0), (-4.0, 7.0)])
    yaw = np.radians(17.0)
    back = ra.rotate_xy(ra.rotate_xy(pts, yaw), -yaw)
    assert np.allclose(back, pts) and np.allclose(ra.rotate_xy(pts, 0), pts)
    assert np.allclose(ra.rotate_xy([(0.0, 1.0)], np.radians(90)), [(1.0, 0.0)])  # the camera panned left by 90: ahead is now to its right


def test_kabsch_gives_a_proper_rotation():
    rng = np.random.default_rng(1)
    for _ in range(5):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        r = q * np.sign(np.linalg.det(q))
        b = rng.normal(size=(40, 3))
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        got = ra.kabsch(b, b @ r.T)
        assert np.allclose(got, r, atol=1e-9) and np.isclose(np.linalg.det(got), 1.0)


def matches(cam0, R, n=300, noise=0.4, outliers=0.2, seed=0):
    """Synthetic matched pixels of far background seen before and after a turn: the new ones with noise, some matches wrong."""
    rng = np.random.default_rng(seed)
    px = np.column_stack([rng.uniform(80, 1840, n), rng.uniform(100, 980, n)])
    old = cam0.distort(px)
    b = ra.bearings(cam0, old)
    new = cam0.distort(ra.project(cam0, b @ R.T))
    new = new + rng.normal(0, noise, new.shape)
    bad = rng.random(n) < outliers
    new[bad] = np.column_stack([rng.uniform(0, 1920, bad.sum()), rng.uniform(0, 1080, bad.sum())])
    return old, new, bad


@pytest.mark.parametrize("cam0", [CAM, LENS], ids=["pinhole", "wide lens"])
def test_the_rotation_is_estimated_from_noisy_matches_with_outliers(cam0):
    _, _, R = truth(cam0, -2.5, 1.0, 6.0)
    old, new, bad = matches(cam0, R)
    est = ra.estimate_rotation(cam0, old, new)
    err = np.degrees(np.arccos(np.clip((np.trace(est["R"].T @ R) - 1) / 2, -1, 1)))
    assert err < 0.05 and est["rms_px"] < 0.8 and abs(est["angle_deg"] - np.degrees(np.arccos((np.trace(R) - 1) / 2))) < 0.05
    assert est["inliers"] >= 0.75 * (~bad).sum() and est["spread"][0] > 0.7 and est["spread"][1] > 0.6
    nothing = ra.estimate_rotation(cam0, old[:5], new[:5])  # too few to say anything
    assert nothing["R"] is None


# --- the pipeline on pictures ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def parent():
    rec = record()
    rec["reference"] = drift.extract_reference(scene())
    return rec


def turned_picture(img, R):
    """The picture a pinhole camera takes after the turn R (b_new = R b_old): the old one under H = K R K^-1."""
    return cv2.warpPerspective(img, CAM.K() @ R @ np.linalg.inv(CAM.K()), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("dp,dr,yaw", [(2.0, -1.0, 4.0), (-3.0, 0.5, -9.0), (0.0, 0.0, 6.0)])
def test_a_calibration_follows_a_camera_that_turned(parent, dp, dr, yaw):
    new, axes, R = truth(CAM, dp, dr, yaw)
    before = copy.deepcopy(parent)
    rec, rep = ra.reanchor_record(parent, turned_picture(scene(), R))
    assert parent == before  # the stored calibration itself is never touched
    cal = rec["calibration"]
    assert np.degrees(cal["pitch"]) == pytest.approx(np.degrees(new.pitch), abs=0.15)
    assert np.degrees(cal["roll"]) == pytest.approx(np.degrees(new.roll), abs=0.15)
    assert rep["yaw_deg"] == pytest.approx(yaw, abs=0.2) and rep["rms_px"] < 1.5 and rep["inliers"] > 150
    assert rep["rotation_deg"] == pytest.approx(np.degrees(np.arccos((np.trace(R) - 1) / 2)), abs=0.15)
    # what the scale rests on is carried over: optics, height, scale sources, confidence
    for k in ("f", "cx", "cy", "k1", "k2", "h", "confidence", "scale_sources", "scale_uncertainty", "lens"):
        assert cal[k] == parent["calibration"][k]
    # H is the new camera's: a pixel of a packed bay maps (through H) onto that bay's ground position in the NEW frame
    H_new = np.array(cal["H"])
    slot = rec["capacity"]["slots"][0]
    px = np.array(slot["poly"]) * (W, H)
    g = (np.c_[px, np.ones(4)] @ H_new.T)
    assert np.allclose(g[:, :2] / g[:, 2:3], slot["poly_m"], atol=0.02)
    # and the bays are where the TRUE new camera sees the old ground (the old frame's metres, yawed)
    ground_old = np.array(parent["capacity"]["slots"][0]["poly_m"])
    assert np.abs(np.array(slot["poly"]) * (W, H) - true_pixels(CAM, axes, ground_old)).max() < 2.5
    assert len(rec["capacity"]["slots"]) == len(parent["capacity"]["slots"]) and rec["capacity"]["with_cars"] == parent["capacity"]["with_cars"]
    assert rec["capacity"]["orientation_deg"] == pytest.approx((parent["capacity"]["orientation_deg"] - yaw) % 180, abs=0.3)
    area_old = parent["bev"]["area_m2"]
    assert rec["bev"]["area_m2"] == area_old and len(rec["bev"]["surface_m"]) == len(parent["bev"]["surface_m"])  # same ground, new frame
    assert rec["reference"] != parent["reference"] and rec["reference"]["n"] >= drift.MIN_REFERENCE  # the new picture's background
    assert rec["reanchor"]["parent"] is None and rec["reanchor"]["rotation_deg"] == rep["rotation_deg"]
    assert any("re-anchored" in w for w in rec["warnings"])
    json.dumps(rec)  # storable


def test_the_new_reference_is_the_new_picture(parent):
    _, _, R = truth(CAM, 2.0, 0.0, 5.0)
    moved = turned_picture(scene(), R)
    rec, _ = ra.reanchor_record(parent, moved)
    assert drift.compare(rec["reference"], moved)["status"] == "ok"  # the drift guard now judges against where the camera IS
    assert drift.compare(parent["reference"], moved)["status"] == "drift"  # ... and the old reference would still cry drift


def test_a_barely_moved_camera_reanchors_to_almost_the_same_calibration(parent):
    rec, rep = ra.reanchor_record(parent, scene())
    assert rep["rotation_deg"] < 0.05 and abs(rep["yaw_deg"]) < 0.05
    assert rec["calibration"]["pitch"] == pytest.approx(parent["calibration"]["pitch"], abs=2e-3)
    assert np.allclose(np.array(rec["capacity"]["slots"][0]["poly"]), np.array(parent["capacity"]["slots"][0]["poly"]), atol=2e-3)


def test_vehicles_in_the_new_frame_are_not_background(parent):
    _, _, R = truth(CAM, 2.0, -1.0, 4.0)
    moved = turned_picture(scene(), R)
    cars = [(300, 400, 700, 650), (1100, 500, 1500, 760)]
    rng = np.random.default_rng(4)
    for x1, y1, x2, y2 in cars:  # new cars parked where there was background
        moved[y1:y2, x1:x2] = rng.integers(0, 255, (y2 - y1, x2 - x1, 3), np.uint8)
    rec, rep = ra.reanchor_record(parent, moved, cars)
    assert rep["rms_px"] < 1.5 and rep["yaw_deg"] == pytest.approx(4.0, abs=0.25)


# --- the refusals: each says why, and the way out is always the full setup ----------------------------------------------------
def test_a_calibration_without_a_reference_cannot_be_reanchored(parent):
    bare = {k: v for k, v in parent.items() if k != "reference"}
    with pytest.raises(ra.ReanchorError, match="no stored reference.*full setup"):
        ra.reanchor_record(bare, scene())
    flat = {**parent, "reference": drift.extract_reference(np.full((H, W, 3), 90, np.uint8))}
    with pytest.raises(ra.ReanchorError, match="no stored reference"):
        ra.reanchor_record(flat, scene())


def test_another_scene_or_a_dark_frame_cannot_be_matched(parent):
    for frame in (scene(seed=77), np.zeros((H, W, 3), np.uint8)):
        with pytest.raises(ra.ReanchorError, match="cannot be matched.*full setup"):
            ra.reanchor_record(parent, frame)


def test_a_shifted_or_zoomed_camera_is_not_a_turn(parent):
    zoomed = cv2.warpAffine(scene(), cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1.12), (W, H), borderMode=cv2.BORDER_REFLECT)
    with pytest.raises(ra.ReanchorError, match="full setup"):
        ra.reanchor_record(parent, zoomed)  # a lens change (or a move towards the scene): a turn does not explain it
    sheared = cv2.warpAffine(scene(), np.float32([[1, 0.06, 0], [0.05, 1, 0]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    with pytest.raises(ra.ReanchorError, match="full setup"):
        ra.reanchor_record(parent, sheared)


def test_too_big_a_turn_a_level_camera_and_a_lot_out_of_view_are_refused(parent, monkeypatch):
    _, _, big = truth(CAM, 0.0, 0.0, 40.0)
    with pytest.raises(ra.ReanchorError, match="full setup"):
        ra.reanchor_record(parent, turned_picture(scene(), big))  # 40 degrees of pan
    low = copy.deepcopy(parent)
    low["calibration"]["pitch"] = np.radians(8.0)  # a camera that looked 8 degrees down, now 1 degree
    _, _, drop = truth(CameraModel(CAM.f, CAM.cx, CAM.cy, np.radians(8.0), CAM.roll, CAM.h), -7.0, 0.0, 0.0)
    with pytest.raises(ra.ReanchorError, match="too level"):
        ra.reanchor_record(low, turned_picture(scene(), drop))
    _, _, small = truth(CAM, 1.0, 0.0, 2.0)
    monkeypatch.setattr(ra, "MIN_BAYS_VISIBLE", 1.1)  # no bay can be "enough in view"
    with pytest.raises(ra.ReanchorError, match="out of the new view"):
        ra.reanchor_record(parent, turned_picture(scene(), small))


def test_a_degenerate_turn_is_caught_not_applied():
    with pytest.raises(ra.ReanchorError, match="internal"):
        ra.reanchor_camera(CAM, 2.0 * np.eye(3))  # not a rotation at all (a scaled matrix): refused, never applied
