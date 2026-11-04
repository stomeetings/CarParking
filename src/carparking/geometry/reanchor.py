"""Re-anchoring (Phase 2): the camera was bumped or re-aimed, so a calibration that was right for its old picture is wrong for the
new one. Instead of redoing the setup (SAM, depth, a typed distance), solve for how the camera TURNED and carry the calibration
over to the new pose.

Why that is enough: a bump changes the camera's orientation, not its position (a few centimetres against objects 5-30 m away).
What the calibration measured with effort stays true: the focal length and lens distortion (the optics did not change), the
camera height, and so the metric scale, including one anchored by a typed distance. Only the rotation R is unknown, and the
calibration's own reference (the static background of its still, stored with it for the drift guard) plus the new frame give it:

  1. the reference's features are matched to the new frame's (carparking.geometry.drift.matched), vehicles masked;
  2. both sets of pixels are undistorted with the calibration's lens and turned into bearing vectors b = K^-1 (u, v, 1);
     a pure rotation means b_new = R b_old for the background: RANSAC (on the induced homography) keeps the matches that agree, and
     R is the least-squares rotation of the kept bearings (Kabsch / SVD), refined on the points it explains;
  3. the new pitch and roll come from where "up" went: up_new = R up_old, read exactly as GeoCalib's up vector is
     (camera_from_geocalib); what is left over is a YAW about the vertical, which turns the ground frame (its Y axis is the
     camera's heading), so every stored ground position (bays, surface) is rotated by -yaw into the new frame;
  4. the homography H, the bays' image polygons, the surface outline and the drift reference are rebuilt for the new view.

Honest refusals (ReanchorError, with the reason): too few agreeing matches (night, other light, a changed scene), matches that
cover too little of the picture, a residual that a rotation does not explain (the camera was SHIFTED, not turned, or the lens
changed), a turn beyond MAX_ROTATION_DEG, a camera that now looks level, a lot that is mostly out of view. In each case the full
setup is the way. numpy + cv2 only: no model, no setup service."""
from __future__ import annotations

import copy

import cv2
import numpy as np

from . import drift
from .camera import CameraModel, to_image
from .pipeline import MIN_PITCH

MIN_INLIERS = 25
MIN_INLIER_SHARE = 0.40
RANSAC_PX = 3.0  # at a 1920 px wide picture; scaled with the width
MAX_RMS_PX = 2.0  # likewise: what a pure rotation must explain
MAX_ROTATION_DEG = 25.0
MIN_BAYS_VISIBLE = 0.5  # share of the packed bays that must still be in view


class ReanchorError(ValueError):
    """The calibration cannot be carried over; the message says why and what to do."""


def camera_of(cal: dict) -> CameraModel:
    """The calibrated camera of a stored calibration (at the record's own picture size)."""
    return CameraModel(cal["f"], cal["cx"], cal["cy"], cal["pitch"], cal["roll"], cal["h"], cal.get("k1", 0.0), cal.get("k2", 0.0))


def bearings(cam: CameraModel, px) -> np.ndarray:
    """Pixels (as the camera sees them) -> unit bearing vectors in the camera frame (x right, y down, z forward); NaN where
    the lens model cannot place the pixel."""
    ideal = cam.undistort(px)
    b = np.column_stack([(ideal[:, 0] - cam.cx) / cam.f, (ideal[:, 1] - cam.cy) / cam.f, np.ones(len(ideal))])
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def project(cam: CameraModel, b: np.ndarray) -> np.ndarray:
    """Bearing vectors -> IDEAL pixels (no lens); NaN for bearings not in front of the camera."""
    z = b[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.column_stack([cam.f * b[:, 0] / z + cam.cx, cam.f * b[:, 1] / z + cam.cy])
    out[z <= 1e-3] = np.nan
    return out


def kabsch(b_old: np.ndarray, b_new: np.ndarray) -> np.ndarray:
    """The rotation R that best maps the old bearings onto the new ones (b_new ~ R b_old), least squares, det +1."""
    u, _, vt = np.linalg.svd(b_new.T @ b_old)
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def estimate_rotation(cam: CameraModel, a_px, b_px) -> dict:
    """The camera's rotation from matched pixels (old picture a, new picture b, both at the calibration's picture size and as
    the camera sees them, lens included) -> {"R", "matches", "inliers", "rms_px", "angle_deg", "spread" (inlier extent as shares
    of the picture)}. RANSAC picks the matches that agree on one movement; R is refitted to them and to everything it explains
    within the tolerance; rms_px is the residual of the kept points under a rotation only."""
    a_px, b_px = np.asarray(a_px, float), np.asarray(b_px, float)
    w = 2.0 * cam.cx
    scale = w / 1920.0
    ia, ib = cam.undistort(a_px), cam.undistort(b_px)
    ok = np.isfinite(ia).all(axis=1) & np.isfinite(ib).all(axis=1)
    out = {"R": None, "matches": int(ok.sum()), "inliers": 0, "rms_px": None, "angle_deg": None, "spread": (0.0, 0.0)}
    if ok.sum() < 8:
        return out
    ba, bb = bearings(cam, a_px[ok]), bearings(cam, b_px[ok])
    _, inl = cv2.findHomography(ia[ok].astype(np.float32), ib[ok].astype(np.float32), cv2.RANSAC, RANSAC_PX * scale)
    if inl is None:
        return out
    keep = inl.ravel().astype(bool)
    if keep.sum() < 4:
        return out
    R = kabsch(ba[keep], bb[keep])
    for _ in range(3):  # refit on everything the rotation explains
        moved = project(cam, ba @ R.T)
        res = np.linalg.norm(moved - ib[ok], axis=1)
        keep = np.nan_to_num(res, nan=1e9) < RANSAC_PX * scale * 0.67
        if keep.sum() < 4:
            return out
        R = kabsch(ba[keep], bb[keep])
    res = np.linalg.norm(project(cam, ba[keep] @ R.T) - ib[ok][keep], axis=1)
    shares = a_px[ok][keep] / np.array([w, 2.0 * cam.cy])
    out.update(R=R, inliers=int(keep.sum()), rms_px=float(np.sqrt(np.mean(res ** 2)) / scale),
               angle_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
               spread=(float(np.ptp(shares[:, 0])), float(np.ptp(shares[:, 1]))))
    return out


def reanchor_camera(cam: CameraModel, R: np.ndarray) -> tuple[CameraModel, float]:
    """The camera after it turned by R (b_new = R b_old): the same optics and height at the new pitch and roll, and the YAW (radians)
    between the old ground frame and the new one: a ground point (x, y) of the old frame is Rot(-yaw) (x, y) in the new frame.

    pitch / roll: the world's up vector in camera coordinates is the third row of R_cw (camera axes in world); it turns with the
    camera, and (pitch, roll) are read off it as camera_from_geocalib reads GeoCalib's. yaw: the new camera's axes in the OLD world
    frame are R_cw R^T; the new model (yaw 0, new pitch / roll) has its own R_cw'; they differ by a rotation about Z."""
    up_old = cam.R_cw()[2]
    down = -(R @ up_old)
    down = down / np.linalg.norm(down)
    new = CameraModel(cam.f, cam.cx, cam.cy, float(np.arcsin(np.clip(down[2], -1, 1))), float(np.arctan2(down[0], down[1])),
                      cam.h, cam.k1, cam.k2)
    rz = (cam.R_cw() @ R.T) @ new.R_cw().T
    if abs(rz[2, 2] - 1.0) > 1e-2:
        raise ReanchorError("internal: the turn does not keep the vertical (a degenerate estimate)")
    return new, float(np.arctan2(rz[1, 0], rz[0, 0]))


def rotate_xy(pts, yaw: float) -> np.ndarray:
    """Ground points of the old frame in the new one: rotated by -yaw about the camera's foot."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(pts, float) @ np.array([[c, -s], [s, c]])  # row vectors: p' = p Rot(+yaw)^T = Rot(-yaw) p


def map_pixels(cam: CameraModel, R: np.ndarray, px) -> np.ndarray:
    """Where pixels of the old picture are in the new one (a pure rotation: distant things). NaN where they leave the view."""
    b = bearings(cam, px)
    return cam.distort(project(cam, b @ R.T))


def _deg(x: float) -> float:
    return round(float(np.degrees(x)), 3)


def reanchor_record(parent: dict, frame: np.ndarray, boxes=()) -> tuple[dict, dict]:
    """Carry a calibration record over to the camera's new pose -> (the new record, a report). parent: a stored record (with its
    `reference`); frame: the new BGR frame of the same camera; boxes: the vehicles' / people's pixel boxes in it (masked: they are
    not background). The new record keeps the optics, the height, the scale and its confidence; rotates the ground data into the new
    frame, rebuilds H and the image polygons, and takes the NEW frame's background as its drift reference. Raises ReanchorError."""
    ref = parent.get("reference")
    if not ref or ref.get("n", 0) < drift.MIN_REFERENCE:
        raise ReanchorError("this calibration has no stored reference (it was made before the drift guard existed): "
                            "do the full setup")
    cal, (w, h) = parent["calibration"], (parent["image"]["width"], parent["image"]["height"])
    cam = camera_of(cal)
    a, b, _, n, why = drift.matched(ref, frame, boxes)
    if why:
        raise ReanchorError(f"the new frame cannot be matched to the calibration's reference ({why}): night, other light or too "
                            "different a view: try again in the light of the original setup, or do the full setup")
    size = np.array([w, h], float)
    est = estimate_rotation(cam, a * size, b * size)
    if est["R"] is None or est["inliers"] < MIN_INLIERS or est["inliers"] < MIN_INLIER_SHARE * est["matches"]:
        raise ReanchorError(f"only {est['inliers']} of {est['matches']} matches agree on one movement (need {MIN_INLIERS}): the "
                            "scene changed too much, or the camera did more than turn: do the full setup")
    if est["spread"][0] < drift.MIN_SPREAD[0] or est["spread"][1] < drift.MIN_SPREAD[1]:
        raise ReanchorError("the agreeing matches cover too little of the picture to fix the camera's turn: do the full setup")
    if est["rms_px"] > MAX_RMS_PX:
        raise ReanchorError(f"a turn does not explain the movement (residual {est['rms_px']:.1f} px): the camera was probably "
                            "shifted or its lens changed: do the full setup")
    if est["angle_deg"] > MAX_ROTATION_DEG:
        raise ReanchorError(f"the camera turned {est['angle_deg']:.0f} degrees: too far to carry the calibration over safely: "
                            "do the full setup")
    new, yaw = reanchor_camera(cam, est["R"])
    if new.pitch < MIN_PITCH:
        raise ReanchorError(f"the camera now looks {np.degrees(new.pitch):.1f} degrees below the horizon: too level for ground "
                            "measurements: aim it back down, or do the full setup")

    rec = copy.deepcopy(parent)
    rec.pop("reanchor", None)
    rec["calibration"].update(pitch=round(float(new.pitch), 5), roll=round(float(new.roll), 5), H=new.homography().tolist())
    if rec.get("surface", {}).get("polygon"):  # the outline of the lot in the new picture
        moved = map_pixels(cam, est["R"], np.asarray(rec["surface"]["polygon"], float) * size) / size
        rec["surface"]["polygon"] = np.round(moved[np.isfinite(moved).all(axis=1)], 5).tolist()
    if rec.get("bev", {}).get("surface_m"):
        rec["bev"]["surface_m"] = np.round(rotate_xy(rec["bev"]["surface_m"], yaw), 3).tolist()
    slots, inside = [], 0
    for s in rec["capacity"]["slots"]:
        poly_m = rotate_xy(s["poly_m"], yaw)
        px = to_image(new, poly_m) / size
        if not np.isfinite(px).all():
            continue  # behind the new camera: not part of this view
        inside += bool(((px > -0.02) & (px < 1.02)).all())
        slots.append({**s, "poly_m": np.round(poly_m, 3).tolist(), "poly": np.round(px, 5).tolist()})
    if not slots or inside < MIN_BAYS_VISIBLE * len(rec["capacity"]["slots"]):
        raise ReanchorError("the lot is mostly out of the new view: aim the camera back at it, or do the full setup")
    rec["capacity"].update(slots=slots, geometric_capacity=len(slots), with_cars=sum(bool(s.get("car")) for s in slots),
                           orientation_deg=round(float((rec["capacity"].get("orientation_deg", 0.0) - np.degrees(yaw)) % 180.0), 1))
    fresh = drift.extract_reference(frame, boxes)
    if fresh["n"] < drift.MIN_REFERENCE:
        raise ReanchorError("the new frame has too few stable features to become the drift guard's new reference: try another frame")
    rec["reference"] = fresh
    report = {"matches": est["matches"], "inliers": est["inliers"], "rms_px": round(est["rms_px"], 2), "rotation_deg": round(est["angle_deg"], 3),
              "yaw_deg": _deg(yaw), "pitch_before_deg": _deg(cal["pitch"]), "pitch_after_deg": _deg(new.pitch),
              "roll_before_deg": _deg(cal["roll"]), "roll_after_deg": _deg(new.roll), "bays": len(slots), "bays_in_view": inside}
    rec["reanchor"] = {**report, "method": "feature matching + rotation (Kabsch)", "parent": None}  # the API fills in the version
    rec["warnings"] = [*parent.get("warnings", []), f"re-anchored after a camera move: turned {report['rotation_deg']:.1f} deg "
                       f"(pitch {report['pitch_before_deg']:.1f} -> {report['pitch_after_deg']:.1f}, roll {report['roll_before_deg']:.1f} -> "
                       f"{report['roll_after_deg']:.1f}, yaw {report['yaw_deg']:.1f}); the scale is the original one"]
    return rec, report
