"""Phase 2 metric geometry: pixels <-> ground metres, parking surface area, capacity packing, distances.
See docs/PHASE2_GEOMETRY_PLAN.md. numpy + cv2 only; SAM / GeoCalib live in the isolated setup service."""
from .bev import area, footprint, nearest_pairs, poly_distance, surface_polygon
from .calib import fuse_scales, height_from_box, height_from_vehicles
from .camera import CameraModel, ground_resolution, to_ground, to_image
from .capacity import dominant_angle, pack_capacity, to_pixel_polys

__all__ = ["CameraModel", "to_ground", "to_image", "ground_resolution", "surface_polygon", "area", "footprint",
           "poly_distance", "nearest_pairs", "fuse_scales", "height_from_box", "height_from_vehicles",
           "dominant_angle", "pack_capacity", "to_pixel_polys"]
