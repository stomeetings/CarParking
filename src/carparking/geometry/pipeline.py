"""Setup pipeline (Phase 2): the model outputs of the setup service (GeoCalib: focal + up vector; depth: camera height
from the mask's ground pixels) + the surface polygon + the vehicle boxes on the still -> one geometry record for the
versioned store (carparking/geometry_store.py): calibration, bird's-eye surface, area, packed bays.

numpy + cv2 only and no model inside: the api runs it, the setup service never needs the database or YOLO."""
from __future__ import annotations

import numpy as np

import cv2

from .bev import MAX_RANGE, MAX_RES, add_footprints, car_footprint, explain_surface, surface_grid, surface_polygon
from .calib import (camera_from_geocalib, fuse_scales, height_from_reference, height_from_vehicles)
from .camera import ground_resolution, to_image
from .capacity import AISLE, BAY, bay_angle_from_cars, dominant_angle, pack_bays

SCHEMA = 1
H_RANGE = (0.5, 60.0)  # metres: a camera height outside this is a failed estimate, not a calibration
MIN_PITCH = np.radians(3.0)  # a camera that looks (nearly) level has the lot at the horizon: no usable ground metres
MIN_SURFACE_M2 = 20.0
CAR_AGREEMENT = 0.6  # the parked cars give the bays' direction when their axes agree at least this much (0-1)
CAR_MAX_RES = 0.06  # m per pixel: a car's footprint is only trusted where one pixel covers at most this much ground. Live
# camera-1 (3.2 m high, 14 deg tilt): a 15 px error in a far car's box edge is 4 m on the ground at 25 m; 0.06 is ~18 m
HFOV_PLAUSIBLE = (35.0, 130.0)  # degrees: a field of view outside this is not a believable CCTV lens


class SetupError(ValueError):
    """The inputs can't give a calibration (the API reports the message to the Admin)."""


def usable_vehicle_boxes(dets: list[dict], width: int, height: int, min_conf: float = 0.4) -> list[tuple]:
    """Boxes good for the height estimate: confident vehicles whole in the image (not cut by its edge) and not tiny."""
    out = []
    for d in dets:
        if d.get("kind", "vehicle") != "vehicle" or float(d.get("conf", 1.0)) < min_conf:
            continue
        x1, y1, x2, y2 = d["box"]
        if x1 <= 0.01 * width or x2 >= 0.99 * width or y2 >= 0.99 * height or y1 <= 0.01 * height:
            continue
        if (y2 - y1) < 0.03 * height or (x2 - x1) < 0.03 * width:
            continue
        out.append((x1, y1, x2, y2))
    return out


def build_record(*, width: int, height: int, polygon, f: float, up, depth: dict | None = None, vehicles=(),
                 reference: dict | None = None, bay=BAY, aisle: float = AISLE, meta: dict | None = None,
                 pins: dict | None = None, k1: float = 0.0, k2: float = 0.0, lens: str = "pinhole",
                 obstacles=None, height_hint: float | None = None, row_outline=None, min_surface_m2: float = MIN_SURFACE_M2) -> dict:
    """The geometry record (schema 1) for one camera still.

    polygon: the parking surface, [[x, y]...] normalised 0-1. f, up: GeoCalib (see camera_from_geocalib). depth:
    {"h": ...} from height_from_depth or None. vehicles: pixel boxes (usable_vehicle_boxes). reference:
    {"a": [x, y], "b": [x, y], "metres": m} normalised points, the Admin's typed distance (wins over the estimates).
    k1, k2, lens: GeoCalib's radial distortion and the model that produced it ("pinhole", "simple_radial", "radial"):
    every pixel is undistorted before it is placed on the ground (geometry/camera.py).
    obstacles: every vehicle box (pixels) seen in the still, whole or cut by the edge, or None. Their bodies are cut out of
    the surface (the rays through a car's pixels hit the ground metres behind it), each whole one's footprint is added
    (cars park on parking surface) and the bays are oriented along the cars. None: the polygon is the surface as it is.
    meta: what to keep about how the polygon was made (clicks, score, model, edited).
    height_hint: the camera height (m) of an earlier, typed-reference calibration of the SAME physical camera: used as the
    manual scale when no reference is typed now (a re-measure of the lens and tilt on a fresh still keeps the metres).
    row_outline: [[x, y]...] normalised, an outline drawn along ONE row or block (a sample-test zone): the bays are oriented across its longer side
    on the ground, whatever the vehicles say. min_surface_m2: the least measurable ground of a parking lot (a sample-test zone, whose bays are laid
    from its outline and cars, passes 0: one row drawn over its parked cars leaves little open ground). Raises SetupError."""
    poly_n = np.asarray(polygon, float)
    if poly_n.ndim != 2 or poly_n.shape[1] != 2 or len(poly_n) < 3 or poly_n.min() < 0 or poly_n.max() > 1:
        raise SetupError("polygon: at least 3 points with x, y between 0 and 1")
    cam = camera_from_geocalib(f, width, height, up, k1, k2)
    warnings = []
    if cam.pitch < MIN_PITCH:
        raise SetupError(f"the camera looks {np.degrees(cam.pitch):.1f} deg below the horizon: too level for ground "
                         "measurements (GeoCalib may have failed on this image; try another still)")

    edge = cam.undistort([(0.0, cam.cy), (float(width), cam.cy)])  # the ideal image's half width, lens undone
    hfov = None
    if np.isnan(edge).any():
        warnings.append("the lens distortion model does not reach the image edge: the outer parts are unreliable")
    else:
        hfov = float(np.degrees(2 * np.arctan(np.abs(edge[:, 0] - cam.cx).mean() / cam.f)))
        if not HFOV_PLAUSIBLE[0] <= hfov <= HFOV_PLAUSIBLE[1]:
            warnings.append(f"the focal length gives a {hfov:.0f} deg field of view, unusual for a CCTV camera "
                            f"({HFOV_PLAUSIBLE[0]:.0f}-{HFOV_PLAUSIBLE[1]:.0f}): GeoCalib may be wrong here; do not "
                            "trust the metres without a typed reference distance")

    ref_h = None
    if reference:
        a, b = (np.asarray(reference[k], float) * (width, height) for k in ("a", "b"))
        ref_h = height_from_reference(cam, a, b, float(reference["metres"]))
    veh = height_from_vehicles(cam, list(vehicles)) if len(vehicles) else None
    sources = {"vehicles": veh and veh["h"], "depth": depth and depth.get("h"), "manual": ref_h or height_hint}
    fused = fuse_scales(sources)
    if fused["h"] is None:
        raise SetupError("no scale source: fewer than 3 whole vehicles in the still and no depth estimate; type a "
                         "reference distance between two ground points")
    h = fused["h"]
    if not H_RANGE[0] <= h <= H_RANGE[1]:
        raise SetupError(f"camera height {h:.1f} m is outside {H_RANGE}: the estimates are unusable; type a reference "
                         "distance")
    if fused["confidence"] == "LOW":
        what = ("the scale estimates disagree" if len(fused["sources"]) > 1 else "the scale rests on one prior")
        warnings.append(what + " (" + ", ".join(f"{k} {v:.1f} m" for k, v in fused["sources"].items()) +
                        f"): expect about +-{fused['uncertainty']:.0%}; type a reference distance for accurate metres")
    elif fused["confidence"] == "MEDIUM":
        warnings.append(f"the scale comes from priors (vehicle sizes, monocular depth): expect about "
                        f"+-{fused['uncertainty']:.0%}; type a reference distance (e.g. a known wheelbase or bay width) "
                        "for accurate metres")
    if veh and veh["spread"] > 0.2:
        warnings.append(f"the cars give a spread of {veh['spread']:.0%} in height: the ground may not be flat")
    cam = cam.with_height(h)
    H = cam.homography()  # of the ideal (undistorted) image; to_ground / surface_polygon take the camera itself

    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.round(poly_n * (width, height)).astype(np.int32)], 255)
    polygon_mask = mask.copy()  # the outline as drawn, before the vehicles are cut out of it
    footprints, cars, too_far = [], [], 0
    for x1, y1, x2, y2 in obstacles or ():
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)  # a vehicle's body is not ground
        if x1 > 0.01 * width and x2 < 0.99 * width and y2 < 0.99 * height and y1 > 0.01 * height:  # whole in the image
            if ground_resolution(cam, [((x1 + x2) / 2, y2)])[0] > CAR_MAX_RES:
                too_far += 1  # one pixel is worth too much ground here for the car's position to mean anything
                continue
            fp = car_footprint(cam, (x1, y1, x2, y2))
            if fp is not None:
                footprints.append(fp[0])
                cars.append((fp[1], fp[2]))
    if too_far:
        warnings.append(f"{too_far} parked car(s) are too far from the camera for their position to be measured "
                        f"(over {CAR_MAX_RES * 100:.0f} cm of ground per pixel): no bay was placed on them")
    grid = surface_grid(cam, mask, include=footprints)
    if grid is None and obstacles:
        # the outline was drawn over the parked cars: cutting every car's box out left only slivers. The cars' own footprints are laid back below, so measure the
        # outline as drawn instead of giving up
        grid = surface_grid(cam, polygon_mask, include=footprints)
        if grid is not None:
            warnings.append("the outline lies mostly over parked vehicles, so cutting them out left no ground: the outline was measured as drawn "
                            "(draw it around the open ground, or let SAM outline it, for a tighter result)")
    if grid is None:
        why = explain_surface(cam, polygon_mask)
        raise SetupError("the surface polygon has no measurable ground (above the horizon, out of range, or only slivers thinner than 1.2 m / pieces under 3 m2): "
                         f"of the outline's pixels {why['unprojectable']:.0%} are at or above the horizon, {why['too_far']:.0%} are over {MAX_RANGE:.0f} m away, "
                         f"{why['too_coarse']:.0%} are too coarse (one pixel over {MAX_RES * 100:.0f} cm), {why['usable']:.0%} are measurable. "
                         + ("Mostly above the horizon or far away means the scale or tilt is off: check the typed distance and the two ground points. "
                            if why["usable"] < 0.3 else "The measurable part is cut into slivers: draw the outline around the open ground, not over the cars. "))
    add_footprints(grid, footprints)
    area_m2 = grid.area
    if area_m2 < min_surface_m2:
        raise SetupError(f"the measurable surface is only {area_m2:.1f} m2: too small to be a parking lot")
    surf = grid.polygon()
    from_cars = bay_angle_from_cars(cars)
    row = None
    if row_outline is not None:
        g = surface_polygon(cam, np.asarray(row_outline, float) * (width, height))
        row = (dominant_angle(g) + 90) % 180 if len(g) >= 3 and cv2.contourArea(np.float32(g)) > 1.0 else None
    if row is not None:
        angle, source = row, "zone outline"
    elif from_cars and len(cars) >= 2 and from_cars[1] >= CAR_AGREEMENT:
        angle, source = from_cars[0], "vehicles"
    elif surf is not None:
        angle, source = (dominant_angle(surf) + 90) % 180, "polygon"
    else:
        angle, source = 90.0, "default"
    bays, n_cars = pack_bays(grid, angle, bay, aisle, anchors=[fp.mean(axis=0) for fp in footprints])
    slots = []
    for i, b in enumerate(bays, 1):
        px = to_image(cam, b) / (width, height)
        if np.isfinite(px).all():
            slots.append({"id": i, "poly": np.round(px, 5).tolist(), "poly_m": np.round(b, 3).tolist(),
                          "car": i <= n_cars})  # True: a car stood there in the still
    return {
        "schema": SCHEMA,
        "image": {"width": width, "height": height},
        "surface": {"polygon": np.round(poly_n, 5).tolist(), **(meta or {})},
        "calibration": {"f": round(float(cam.f), 2), "k1": round(float(k1), 5), "k2": round(float(k2), 5), "lens": lens,
                        "hfov_deg": None if hfov is None else round(hfov, 1), "cx": cam.cx, "cy": cam.cy, "pitch": round(float(cam.pitch), 5),
                        "roll": round(float(cam.roll), 5), "h": round(float(h), 3),
                        "H": H.tolist(), "confidence": fused["confidence"],
                        "scale_sources": {k: round(v, 3) for k, v in fused["sources"].items()},
                        "scale_uncertainty": fused["uncertainty"],
                        "vehicles_used": veh["n"] if veh else 0,
                        "vehicles_spread": round(veh["spread"], 3) if veh else None,
                        "depth_spread": round(depth["spread"], 3) if depth and "spread" in depth else None, "pins": pins or {}},
        "bev": {"surface_m": [] if surf is None else np.round(surf, 3).tolist(), "area_m2": round(area_m2, 1),
                "cell_m": grid.cell, "footprints": len(footprints), "cars_too_far": too_far},
        "capacity": {"bay": list(bay), "aisle": aisle, "orientation_deg": round(float(angle), 1),
                     "orientation_source": source, "geometric_capacity": len(slots),
                     "with_cars": sum(s["car"] for s in slots), "slots": slots},
        "warnings": warnings,
    }
