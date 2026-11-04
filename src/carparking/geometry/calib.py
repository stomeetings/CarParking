"""Metric scale for a camera whose orientation is known (Phase 2): the camera height h from parked cars, and the fusion of
several independent estimates into one value with a confidence. The orientation itself (focal length, pitch, roll) comes
from GeoCalib in the setup service; this module is the part that needs only numpy, so the worker image can import it.

Metric scale is the one thing a single image cannot give. Every ground distance at h = 1 times h is the distance at h, so
any known real-world size fixes h. Cars are the size reference that is always in view."""
from __future__ import annotations

import numpy as np

from .camera import CameraModel

CAR_HEIGHT = 1.45  # metres, roof height of a typical passenger car (the box top is the roof)
AGREE = 1.15  # two scale estimates agree when max / min <= AGREE


def height_from_box(cam: CameraModel, box, car_height: float = CAR_HEIGHT) -> float | None:
    """Camera height implied by one vehicle box (x1, y1, x2, y2 pixels), or None if the box is unusable.

    The bottom edge's centre is the car's contact point P; the box top is its roof. At h = 1 the pixel row of the roof
    fixes how tall (z1, in h = 1 units) a vertical line at P must be to reach it. A real car is `car_height` metres tall,
    so h = car_height / z1. Only the camera's orientation, focal length and lens distortion are used (cam.h is
    ignored); the box's two ends are undistorted first, so a wide-angle lens does not skew the height."""
    x1, y1, x2, y2 = box
    unit = cam.with_height(1.0)
    xc = (x1 + x2) / 2
    (ub, vb), (ut, vt) = unit.undistort([(xc, y2), (xc, y1)])  # the lens distortion first: ideal pinhole pixels
    if np.isnan([ub, vb, ut, vt]).any():
        return None
    from .camera import to_ground
    P = to_ground(unit.homography(), [(ub, vb)])[0]
    if np.isnan(P).any():
        return None
    R_wc = unit.R_cw().T
    a = R_wc @ np.array([P[0], P[1], -1.0])  # camera coordinates of the contact point's base relative to the camera
    b = R_wc[:, 2]  # one unit of height, in camera coordinates
    q = (vt - unit.cy) / unit.f
    den = b[1] - q * b[2]
    if abs(den) < 1e-9:
        return None
    z1 = (q * a[2] - a[1]) / den
    return float(car_height / z1) if z1 > 1e-6 else None


def height_from_vehicles(cam: CameraModel, boxes, car_height: float = CAR_HEIGHT, min_cars: int = 3) -> dict | None:
    """Median camera height from many vehicle boxes, MAD outliers dropped -> {"h", "n", "spread"} (spread = relative
    MAD), or None with fewer than min_cars usable boxes. Run it on unoccluded, confident, not frame-cut boxes."""
    hs = np.array([h for h in (height_from_box(cam, b, car_height) for b in boxes) if h is not None and 0.5 < h < 60])
    if len(hs) < min_cars:
        return None
    med = float(np.median(hs))
    mad = float(np.median(np.abs(hs - med))) or 1e-9
    kept = hs[np.abs(hs - med) <= 3.5 * 1.4826 * mad] if mad > 1e-6 else hs
    if len(kept) < min_cars:
        return None
    med = float(np.median(kept))
    return {"h": med, "n": int(len(kept)), "spread": float(np.median(np.abs(kept - med)) / med)}


PRIOR_ERROR = 0.25  # relative error of a prior-based scale (vehicle sizes, monocular depth), see fuse_scales
REFERENCE_ERROR = 0.05  # a typed real-world distance: limited by the Admin's two clicks


def fuse_scales(estimates: dict[str, float | None]) -> dict:
    """Combine the camera-height estimates of the available sources ("vehicles", "depth", "manual") -> {"h",
    "confidence", "uncertainty" (relative), "sources"}.

    The confidence says how far the metres can be trusted, and it is what was MEASURED against known car wheelbases on
    the live camera (Honda HR-V 2.61 m, Suzuki Celerio 2.425 m, 2026-10-23): the vehicle-size prior and the metric depth
    agreed with each other within 8-12 % yet both put the wheelbases 17-21 % short, and the vehicle estimate alone
    scatters +-15-19 % between cars and differs by 20 % between two stills of the same camera. So agreement of two
    priors is evidence of consistency, not of accuracy:
      HIGH    a typed reference distance (the Admin's real-world measurement, +-5 %); it wins over the estimates
      MEDIUM  two or more priors that agree within AGREE: plausible, expect +-25 %
      LOW     one prior alone, or priors that disagree (the median is still returned, the UI shows the badge)
      NONE    nothing"""
    got = {k: float(v) for k, v in estimates.items() if v is not None and v > 0}
    if not got:
        return {"h": None, "confidence": "NONE", "uncertainty": None, "sources": {}}
    if "manual" in got:
        return {"h": got["manual"], "confidence": "HIGH", "uncertainty": REFERENCE_ERROR, "sources": got}
    vals = list(got.values())
    h = float(np.median(vals))
    if len(vals) >= 2 and max(vals) / min(vals) <= AGREE:
        return {"h": h, "confidence": "MEDIUM", "uncertainty": PRIOR_ERROR, "sources": got}
    return {"h": h, "confidence": "LOW", "uncertainty": PRIOR_ERROR * (1.4 if len(vals) >= 2 else 1.2), "sources": got}


def camera_from_geocalib(f: float, width: int, height: int, up, k1: float = 0.0, k2: float = 0.0) -> CameraModel:
    """GeoCalib's output -> CameraModel (h = 1). f: focal length in pixels at the image's own size; up: its `gravity`
    vector, which is the direction UP in the camera frame (x right, y down, z forward: a level camera gives (0, -1, 0),
    one looking down has a negative z). The principal point is the image centre (GeoCalib assumes it too).
    pitch (down) = asin(down_z), roll = atan2(down_x, down_y) with down = -up. k1, k2: GeoCalib's radial distortion
    (its "simple_radial" model has k1 only, "radial" both), in the convention of geometry/camera.py."""
    d = -np.asarray(up, float)
    d = d / np.linalg.norm(d)
    return CameraModel(float(f), width / 2.0, height / 2.0, float(np.arcsin(np.clip(d[2], -1, 1))),
                       float(np.arctan2(d[0], d[1])), 1.0, float(k1), float(k2))


def height_from_depth(points, up, near: float = 2.0, far: float = 40.0) -> dict | None:
    """Camera height from a metric depth map's ground pixels. points: Nx3 camera-frame metres (x right, y down,
    z forward) of pixels known to be ground (the SAM mask); up: the camera-frame up vector. The distance of each point
    below the camera is its component along `down`; the median of the points between `near` and `far` metres away (the
    depth model is least reliable at the extremes) is h -> {"h", "n", "spread"} or None with fewer than 200 points."""
    p = np.asarray(points, float).reshape(-1, 3)
    down = -np.asarray(up, float)
    down = down / np.linalg.norm(down)
    r = np.linalg.norm(p, axis=1)
    p = p[(r >= near) & (r <= far) & np.isfinite(r)]
    if len(p) < 200:
        return None
    h = p @ down
    h = h[h > 0]
    if len(h) < 200:
        return None
    med = float(np.median(h))
    mad = float(np.median(np.abs(h - med)))
    kept = h[np.abs(h - med) <= 3.5 * 1.4826 * mad] if mad > 1e-6 else h
    med = float(np.median(kept))
    return {"h": med, "n": int(len(kept)), "spread": float(np.median(np.abs(kept - med)) / med)}


def height_from_reference(cam: CameraModel, a_px, b_px, metres: float) -> float | None:
    """Camera height from a known distance on the ground: two clicked pixels `metres` apart. At h = 1 they are d1
    apart; the real distance is h * d1."""
    from .camera import to_ground
    g = to_ground(cam.with_height(1.0), [a_px, b_px])  # the camera: its lens distortion is undone first
    d1 = float(np.linalg.norm(g[1] - g[0]))
    return float(metres / d1) if np.isfinite(d1) and d1 > 1e-9 and metres > 0 else None


def height_from_depth_map(dmap, mask, f: float, up, max_points: int = 40_000, k1: float = 0.0,
                          k2: float = 0.0) -> dict | None:
    """Camera height from a metric depth map (HxW, metres along the optical axis) and the ground mask (HxW, > 127 =
    ground): the mask's pixels (a seeded sample of at most max_points) are undistorted (k1, k2), back-projected with
    focal length f and the principal point at the image centre, then height_from_depth. None when the mask is empty or
    the points are unusable."""
    dmap = np.asarray(dmap)
    h, w = dmap.shape
    vs, us = np.nonzero(np.asarray(mask) > 127)
    if len(us) == 0:
        return None
    pick = np.random.default_rng(0).choice(len(us), size=min(max_points, len(us)), replace=False)
    d = dmap[vs[pick], us[pick]].astype(np.float64)
    ideal = CameraModel(f, w / 2, h / 2, 0.0, 0.0, 1.0, k1, k2).undistort(np.stack([us[pick], vs[pick]], axis=1))
    ok = np.isfinite(d) & (d > 0) & ~np.isnan(ideal).any(axis=1)
    pts = np.stack([(ideal[:, 0] - w / 2) / f * d, (ideal[:, 1] - h / 2) / f * d, d], axis=1)[ok]
    return height_from_depth(pts, up)
