"""Pinhole camera with radial lens distortion over a flat ground plane: the map between image pixels and ground metres
(Phase 2, see docs/PHASE2_GEOMETRY_PLAN.md).

World frame: X to the right, Y forward, Z up, the origin on the ground under the camera, the camera at height h. Camera
frame: x right, y down, z forward. The camera looks down by `pitch` (radians, > 0 = towards the ground) and is rolled by
`roll` around its optical axis. Metric scale is linear in h: ground distances at h = 1 times h are the distances at h.

Lens distortion follows GeoCalib's radial model exactly (geocalib/camera.py, "simple_radial" with k2 = 0 and "radial"):
a point at normalised coordinates p = (x / z, y / z) lands in the image at  f * p * (1 + k1 r^2 + k2 r^4) + c  with
r^2 = |p|^2. So a pixel is first normalised, then UNDISTORTED, and only then does the pinhole homography apply. GeoCalib's
own undistort() is a first-order approximation that is poor for strong barrel distortion (k1 around -0.5); here the inverse
is exact (bisection on the monotone branch of the forward model).

numpy only: the worker needs nothing but a CameraModel rebuilt from the stored record, or the 3x3 `homography()` of a
camera without distortion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BISECT = 48  # halvings of the bracket: far below a millionth of a pixel
VALID_TOL = 1e-6  # normalised units: the inverse must reproduce the distorted radius this closely


def distort_n(pn, k1: float, k2: float = 0.0) -> np.ndarray:
    """Normalised ideal coordinates -> normalised distorted coordinates (GeoCalib's forward model)."""
    p = np.asarray(pn, float)
    r2 = (p ** 2).sum(axis=-1, keepdims=True)
    return p * (1.0 + k1 * r2 + k2 * r2 ** 2)


def _monotone_limit(k1: float, k2: float) -> float:
    """Radius up to which the forward model r (1 + k1 r^2 + k2 r^4) increases (beyond it the distortion folds back and
    the inverse does not exist)."""
    a, b = 5.0 * k2, 3.0 * k1  # derivative 1 + b x + a x^2 with x = r^2
    roots = []
    if abs(a) < 1e-12:
        if b < 0:
            roots = [-1.0 / b]
    else:
        disc = b * b - 4.0 * a
        if disc >= 0:
            roots = [(-b + s * np.sqrt(disc)) / (2.0 * a) for s in (1.0, -1.0)]
    roots = [x for x in roots if x > 0]
    return float(np.sqrt(min(roots))) if roots else 10.0


def undistort_n(pd, k1: float, k2: float = 0.0) -> np.ndarray:
    """Normalised distorted coordinates -> normalised ideal coordinates, the exact inverse of distort_n. Points beyond
    the range the distortion model can represent (the radius the forward model folds back at) come back NaN."""
    p = np.asarray(pd, float)
    if k1 == 0.0 and k2 == 0.0:
        return p.copy()
    rd = np.linalg.norm(p, axis=-1)
    hi = _monotone_limit(k1, k2)
    lo_r, hi_r = np.zeros_like(rd), np.full_like(rd, hi)
    for _ in range(BISECT):
        mid = (lo_r + hi_r) / 2.0
        too_far = mid * (1.0 + k1 * mid ** 2 + k2 * mid ** 4) > rd
        hi_r = np.where(too_far, mid, hi_r)
        lo_r = np.where(too_far, lo_r, mid)
    ru = (lo_r + hi_r) / 2.0
    scale = np.divide(ru, rd, out=np.ones_like(rd), where=rd > 0)
    out = p * scale[..., None]
    reach = hi * (1.0 + k1 * hi ** 2 + k2 * hi ** 4)
    out[rd > reach - VALID_TOL] = np.nan
    return out


@dataclass(frozen=True)
class CameraModel:
    f: float  # focal length, pixels
    cx: float
    cy: float
    pitch: float  # radians, camera tilted down
    roll: float = 0.0  # radians
    h: float = 1.0  # camera height above the ground, metres
    k1: float = 0.0  # radial distortion (GeoCalib convention, see the module doc)
    k2: float = 0.0

    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], float)

    def R_cw(self) -> np.ndarray:
        """Columns = the camera's x, y, z axes in world coordinates."""
        sp, cp, sr, cr = np.sin(self.pitch), np.cos(self.pitch), np.sin(self.roll), np.cos(self.roll)
        x, z = np.array([1.0, 0, 0]), np.array([0, cp, -sp])
        y = np.cross(z, x)
        x, y = x * cr + y * sr, -x * sr + y * cr
        return np.stack([x, y, z], axis=1)

    def with_height(self, h: float) -> "CameraModel":
        return CameraModel(self.f, self.cx, self.cy, self.pitch, self.roll, h, self.k1, self.k2)

    @property
    def distorted(self) -> bool:
        return self.k1 != 0.0 or self.k2 != 0.0

    def undistort(self, px) -> np.ndarray:
        """Nx2 image pixels (as the camera sees them) -> Nx2 pixels of the ideal pinhole image (NaN where the model
        can't represent the point)."""
        p = np.asarray(px, float).reshape(-1, 2)
        if not self.distorted:
            return p.copy()
        c = np.array([self.cx, self.cy])
        return undistort_n((p - c) / self.f, self.k1, self.k2) * self.f + c

    def distort(self, px_ideal) -> np.ndarray:
        """Nx2 pixels of the ideal pinhole image -> Nx2 image pixels."""
        p = np.asarray(px_ideal, float).reshape(-1, 2)
        if not self.distorted:
            return p.copy()
        c = np.array([self.cx, self.cy])
        return distort_n((p - c) / self.f, self.k1, self.k2) * self.f + c

    def homography(self) -> np.ndarray:
        """3x3 matrix H of the IDEAL pinhole image: pixel (u, v, 1) -> ground (X, Y, 1) after dividing by the last
        component, in metres. For a distorted camera undistort the pixels first (to_ground does)."""
        R_wc = self.R_cw().T
        g2i = self.K() @ np.stack([R_wc[:, 0], R_wc[:, 1], R_wc @ np.array([0, 0, -self.h])], axis=1)
        return np.linalg.inv(g2i)

    def horizon_v(self, u: float | None = None) -> float:
        """Row of the horizon in the IDEAL pinhole image at column u (default the principal point)."""
        return self._horizon_at(self.cx if u is None else u)

    def horizon_px(self, us) -> np.ndarray:
        """The horizon as it appears in the distorted image: Nx2 pixels (a curve for a distorted lens), one per ideal
        column in `us`. Points that fall outside the distortion model's range are NaN."""
        us = np.asarray(us, float)
        return self.distort(np.stack([us, [self._horizon_at(u) for u in us]], axis=1))

    def _horizon_at(self, u: float) -> float:
        # the horizon is the image of the ground's line at infinity: points with zero world-Z ray component
        R = self.R_cw()
        n = R[2]  # world Z of the camera's x, y, z axes
        # ray (u - cx, v - cy, f) in camera coordinates has world Z = n . ray = 0
        return self.cy - (n[0] * (u - self.cx) + n[2] * self.f) / n[1]


def _split(m):
    """A camera (distortion-aware) or a bare 3x3 homography of a pinhole image -> (undistort function, H)."""
    if hasattr(m, "undistort") and hasattr(m, "homography"):  # a camera; duck-typed, not isinstance: Streamlit's test runner
        return m.undistort, m.homography()  # purges modules, and a second copy of this class would fail the check
    return (lambda p: np.asarray(p, float).reshape(-1, 2)), np.asarray(m, float)


def to_ground(m, px) -> np.ndarray:
    """Nx2 image pixels -> Nx2 ground metres. m: a CameraModel (the pixels are undistorted first) or a bare 3x3
    homography (pinhole). Rows at or above the horizon, or outside the lens model's range, are NaN."""
    undistort, H = _split(m)
    p = undistort(px)
    q = np.c_[p, np.ones(len(p))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        g = q[:, :2] / q[:, 2:3]
    # H @ (u, v, 1) = (X, Y, 1) / depth, so the last component is positive exactly when the ray hits the ground ahead
    g[~(q[:, 2] > 0)] = np.nan
    return g


def to_image(cam: CameraModel, ground) -> np.ndarray:
    """Nx2 ground metres (X, Y) -> Nx2 image pixels (distorted if the camera is); NaN for points behind the camera."""
    g = np.asarray(ground, float).reshape(-1, 2)
    R_wc = cam.R_cw().T
    c = (np.c_[g, np.zeros(len(g))] - np.array([0, 0, cam.h])) @ R_wc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pn = c[:, :2] / c[:, 2:3]
    pn[~(c[:, 2] > 1e-9)] = np.nan
    return distort_n(pn, cam.k1, cam.k2) * cam.f + (cam.cx, cam.cy)


def ground_resolution(m, px, step: float = 1.0) -> np.ndarray:
    """Metres on the ground that one pixel covers at each point (the larger of the u and v steps): how much a +-1 px
    error in the image is worth there. Grows towards the horizon. m: CameraModel or 3x3 homography."""
    p = np.asarray(px, float).reshape(-1, 2)
    g0 = to_ground(m, p)
    du, dv = to_ground(m, p + (step, 0)), to_ground(m, p + (0, step))
    return np.maximum(np.linalg.norm(du - g0, axis=1), np.linalg.norm(dv - g0, axis=1)) / step
