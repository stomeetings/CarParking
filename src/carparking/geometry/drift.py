"""Drift guard (Phase 2): has the camera moved since it was calibrated? The ground metres are only true for the picture the
calibration was made on, so a bumped or re-aimed camera must not keep producing confident distances.

How: the calibration stores a reference of the STATIC background of its still (ORB features: where, and what they look like),
with vehicles and the on-screen date / time left out. Later the worker matches the current frame's features (again without
the detected vehicles) to it; RANSAC keeps the matches that agree on one movement, and the median displacement of THOSE matched
points, as a share of the image width, is the drift. It is measured where the features are, never extrapolated from a fitted
transform: on the live camera a handful of matches in one thin strip fitted a transform that claimed 27 px of drift for two
frames whose matched points had moved 0.1 px. The matches must also cover the picture (a few ones in a corner say nothing about
the rest). numpy + cv2 only; no model.

Three outcomes, and only one of them is ever acted on:
  ok            the background is where it was (within DRIFT_FRAC of the width: a few pixels of jitter are normal)
  drift         the background is found and has moved by more than that
  inconclusive  too few matches to say (night, glare, rain, a different scene, an occluded view, a blurred frame) - never an
                alarm: no match is not movement. The guard then simply cannot vouch for the calibration for a while."""
from __future__ import annotations

import base64

import cv2
import numpy as np

WORK_W = 960  # the guard works on a frame scaled to this width (ORB at 1080p buys nothing here, and costs time)
N_FEATURES = 600
OSD_TOP = 0.10  # share of the image height masked off at the top: the camera's on-screen date / time changes every second
OSD_BOTTOM = 0.08  # ... and at the bottom: the burned-in camera name stays put when the camera moves, so it would vouch for a
# camera that has not (matches on it say "no movement" however far the picture has moved)
RATIO = 0.75
SHIFT_PERCENTILE = 90  # the displacement that counts: features near a rotation's pivot barely move (a 2 deg roll: median 10 px,
# 90th percentile ~25 px), so the median would call a rotated camera fine; a pure shift moves every feature alike  # Lowe's ratio test on the two nearest descriptors
MIN_REFERENCE = 40  # fewer features than this in the calibration still: the scene is too bare for the guard
MIN_MATCHES = 15
MIN_INLIERS = 12
MIN_INLIER_SHARE = 0.35
MIN_SPREAD = (0.30, 0.25)  # the inliers must span at least this share of the picture's width / height
DRIFT_FRAC = 0.006  # median anchor displacement / image width above which the camera counts as moved (~11 px at 1920)


def _work(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """The frame as a grayscale picture WORK_W wide, and the scale from frame pixels."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    s = WORK_W / w
    return (cv2.resize(gray, (WORK_W, max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s != 1 else gray), s


def _mask(shape, scale: float, boxes) -> np.ndarray:
    """255 where features may be taken: everything except the OSD strip and the (pixel) boxes of the vehicles."""
    m = np.full(shape, 255, np.uint8)
    m[: int(OSD_TOP * shape[0])] = 0
    m[int((1 - OSD_BOTTOM) * shape[0]):] = 0
    for x1, y1, x2, y2 in boxes:
        pad = 6  # a little beyond the box: features on a car's edge belong to the car
        cv2.rectangle(m, (int(x1 * scale) - pad, int(y1 * scale) - pad), (int(x2 * scale) + pad, int(y2 * scale) + pad), 0, -1)
    return m


def _features(frame: np.ndarray, boxes=()) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(keypoints as shares of the picture Nx2, descriptors Nx32, (width, height) of the frame)."""
    gray, s = _work(frame)
    orb = cv2.ORB_create(nfeatures=N_FEATURES, scaleFactor=1.2, nlevels=6, edgeThreshold=15, fastThreshold=12)
    kps, desc = orb.detectAndCompute(gray, _mask(gray.shape, s, boxes))
    if desc is None or not len(kps):
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8), (frame.shape[1], frame.shape[0])
    wh = np.array([gray.shape[1], gray.shape[0]], np.float32)
    return np.array([k.pt for k in kps], np.float32) / wh, desc, (frame.shape[1], frame.shape[0])


def extract_reference(frame: np.ndarray, boxes=()) -> dict:
    """The storable reference of a still: {"size": [w, h], "n", "kp" (shares, 4 decimals), "desc" (base64 of the N x 32 uint8
    descriptors)}. boxes: the vehicles' pixel boxes (x1, y1, x2, y2) in the still: their features are left out."""
    kp, desc, size = _features(frame, boxes)
    return {"size": list(size), "n": int(len(kp)), "kp": np.round(kp, 4).tolist(),
            "desc": base64.b64encode(np.ascontiguousarray(desc).tobytes()).decode()}


def _load(ref: dict) -> tuple[np.ndarray, np.ndarray]:
    kp = np.array(ref["kp"], np.float32).reshape(-1, 2)
    desc = np.frombuffer(base64.b64decode(ref["desc"]), np.uint8).reshape(-1, 32)
    return kp, desc


def matched(ref: dict, frame: np.ndarray, boxes=()):
    """Match the frame's background features to the reference -> (a, b, size, n, why): the matched keypoints as shares of the
    picture (a: where they were in the reference, b: where they are now, both Nx2), the frame's (width, height), the number of
    matches, and why (None when matching went through; otherwise a reason in words and a / b are None). Shared by the drift
    check (compare) and the re-anchoring of a calibration (geometry/reanchor.py)."""
    if frame is None or not getattr(frame, "size", 0) or not ref or ref.get("n", 0) < MIN_REFERENCE:
        return None, None, None, 0, "no usable reference" if not ref or ref.get("n", 0) < MIN_REFERENCE else "no frame"
    if abs(frame.shape[1] / frame.shape[0] - ref["size"][0] / ref["size"][1]) > 0.02:
        return None, None, None, 0, "the frame has another aspect ratio than the reference"
    kp_ref, d_ref = _load(ref)
    kp_cur, d_cur, size = _features(frame, boxes)
    if len(d_cur) < MIN_MATCHES:
        return None, None, size, 0, f"only {len(d_cur)} features in the frame"
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d_ref, d_cur, k=2)
    good = [m for p in pairs if len(p) == 2 for m, n in [p] if m.distance < RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return None, None, size, len(good), f"only {len(good)} matches"
    return kp_ref[[m.queryIdx for m in good]], kp_cur[[m.trainIdx for m in good]], size, len(good), None


def compare(ref: dict, frame: np.ndarray, boxes=(), frac: float = DRIFT_FRAC) -> dict:
    """Where is the background now relative to the reference? -> {"status": ok / drift / inconclusive, "shift_px" (the 90th
    percentile displacement of the matched points, in frame pixels), "shift_frac" (of the width), "matches", "inliers", "why" (inconclusive only)}."""
    out = {"status": "inconclusive", "shift_px": None, "shift_frac": None, "matches": 0, "inliers": 0}
    a_sh, b_sh, size, n, why = matched(ref, frame, boxes)
    out["matches"] = n
    if why:
        return {**out, "why": why}
    wh = np.array([WORK_W, size[1] * WORK_W / size[0]], np.float32)
    a, b = a_sh * wh, b_sh * wh
    good = a  # (only its length is used below)
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    n_in = 0 if inl is None else int(inl.sum())
    out["inliers"] = n_in
    if H is None or n_in < MIN_INLIERS or n_in < MIN_INLIER_SHARE * len(good):
        return {**out, "why": "the matches do not agree on one movement"}
    if not (0.5 < abs(np.linalg.det(H[:2, :2])) < 2.0) or abs(H[2, 0]) > 1e-3 or abs(H[2, 1]) > 1e-3:
        return {**out, "why": "implausible transform"}  # a tilted / scaled world, not a camera nudge: do not trust it
    keep = inl.ravel().astype(bool)
    ia = a[keep]
    spread = (np.ptp(ia[:, 0]) / wh[0], np.ptp(ia[:, 1]) / wh[1])
    if spread[0] < MIN_SPREAD[0] or spread[1] < MIN_SPREAD[1]:
        return {**out, "why": "the matches cover too little of the picture"}
    shift = float(np.percentile(np.linalg.norm(b[keep] - ia, axis=1), SHIFT_PERCENTILE))  # at the matched points, not extrapolated
    return {**out, "status": "drift" if shift / float(wh[0]) > frac else "ok",
            "shift_px": round(float(shift * size[0] / wh[0]), 1), "shift_frac": round(shift / float(wh[0]), 5)}  # plain floats: JSON
