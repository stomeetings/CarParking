"""Ground pins: a camera from four clicked corners of a ground rectangle of known size (a bay, a marked box on the asphalt).

The four corners and their metres fix the plane-to-picture homography H exactly. With square pixels and the principal point at the image centre (the
assumption GeoCalib makes too) H = K [r1 r2 t] has one unknown, the focal length f, which the two constraints on the rotation's columns give
(r1 . r2 = 0 and |r1| = |r2|). Then the pose follows, so the result is what a GeoCalib run plus a typed reference distance would give, from four
clicks, in milliseconds, without any model: {"f", "up" (the camera-frame up vector, as GeoCalib's `gravity`), "h" (metres above the plane),
"pitch", "roll"}. It feeds geometry.pipeline.build_record unchanged (f, up, and a reference distance taken from the first side).
No lens distortion is estimated: a wide-angle lens needs the GeoCalib path. numpy / cv2 only."""
from __future__ import annotations

import numpy as np

from .camera import to_image


class PinsError(ValueError):
    """The four pins cannot be a rectangle seen by a camera (collinear, crossed, behind the camera ...): the message says what to fix."""


def _order_error(px: np.ndarray) -> bool:
    """True when the clicked polygon crosses itself (the corners were not clicked in order around the rectangle)."""
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    s = [np.sign(side(px[i], px[(i + 1) % 4], px[(i + 2) % 4])) for i in range(4)]
    return len(set(s)) != 1 or 0 in s


def camera_from_rectangle(pins_px, a_m: float, b_m: float, size) -> dict:
    """pins_px: 4 corners in pixels, clicked in order around the rectangle (1 -> 2 is a_m metres long, 2 -> 3 is b_m); size: (width, height) of the
    picture. -> {"f", "up" (list), "h", "pitch" (rad), "roll" (rad), "reprojection_px" (largest corner error), "hfov_deg"}. Raises PinsError."""
    import cv2

    w, h = float(size[0]), float(size[1])
    px = np.asarray(pins_px, float).reshape(4, 2)
    if a_m <= 0 or b_m <= 0:
        raise PinsError("the rectangle's sides must be positive lengths in metres")
    if _order_error(px):
        raise PinsError("the four pins do not form a rectangle's outline: click the corners in order around it (no crossing lines)")
    world = np.float64([[0, 0], [a_m, 0], [a_m, b_m], [0, b_m]])
    Hm = cv2.findHomography(world, px - (w / 2, h / 2), 0)[0]
    if Hm is None or not np.isfinite(Hm).all():
        raise PinsError("the pins give no consistent view of a rectangle: check that they mark four distinct corners")
    h1, h2, h3 = Hm[:, 0], Hm[:, 1], Hm[:, 2]
    # s = 1 / f^2 from the two constraints on the rotation's columns, each of the form a s + b = 0:
    #   r1 . r2 = 0      (h1x h2x + h1y h2y) s + h1z h2z = 0
    #   |r1| = |r2|      (h1x^2 + h1y^2 - h2x^2 - h2y^2) s + (h1z^2 - h2z^2) = 0
    # Either alone is degenerate in the commonest layout (a rectangle side parallel to the picture's rows makes h1z 0: the first says 0 = 0), so
    # both are used, in one least squares
    eqs = [((h1[0] * h2[0] + h1[1] * h2[1]), h1[2] * h2[2]),
           ((h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2), h1[2] ** 2 - h2[2] ** 2)]
    # (not rescaled per equation: a degenerate one is all noise, and its small terms are what keeps it from outvoting the good one)
    den = sum(a * a for a, _ in eqs)
    s_ = -sum(a * b for a, b in eqs) / den if den > 1e-12 else -1.0
    f2 = 1.0 / s_ if np.isfinite(s_) and s_ > 1e-12 else -1.0
    if not np.isfinite(f2) or f2 <= 0:
        raise PinsError("the pins do not determine a focal length (the rectangle is seen head-on or edge-on): pick a rectangle with a clear "
                        "perspective, or a bigger one")
    f = float(np.sqrt(f2))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    g1, g2, g3 = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = 2.0 / (np.linalg.norm(g1) + np.linalg.norm(g2))
    r1, r2, t = g1 * lam, g2 * lam, g3 * lam
    if t[2] < 0:  # the homography is defined up to sign: the plane must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r1, r2 = r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)  # the nearest true rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    C = -R.T @ t  # the camera centre in the rectangle's frame
    sign = 1.0 if C[2] > 0 else -1.0
    height = float(abs(C[2]))
    up = R[:, 2] * sign  # the plane's normal towards the camera, in camera coordinates (x right, y down, z forward)
    down = -up
    pitch, roll = float(np.arcsin(np.clip(down[2], -1, 1))), float(np.arctan2(down[0], down[1]))
    if height < 0.3 or height > 200 or pitch <= 0:
        raise PinsError("the pins put the camera at an impossible place (" + (f"{height:.1f} m up" if pitch > 0 else "looking up") +
                        "): check the corners and the sides' lengths")
    # how well the recovered camera explains the clicks
    P = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]) @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    q = (P @ np.c_[world, np.ones(4)].T).T
    err = float(np.max(np.linalg.norm(q[:, :2] / q[:, 2:3] - px, axis=1)))
    hfov = float(np.degrees(2 * np.arctan(w / 2 / f)))
    return {"f": f, "up": up.tolist(), "h": height, "pitch": pitch, "roll": roll, "reprojection_px": round(err, 2), "hfov_deg": round(hfov, 1)}


def pins_from_camera(cam, world_xy) -> np.ndarray:
    """Test helper / UI hint: where ground points of a CameraModel land in the picture (pixels)."""
    return to_image(cam, np.asarray(world_xy, float))
