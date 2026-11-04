"""Frame sources. The engine only sees (frame_id, BGR image) pairs; where they come from is this file's job.

Phase 2 adds a webcam generator here with the same output shape.
"""
from __future__ import annotations

import cv2


def iter_images(paths):
    """Yield (path_str, BGR image) for each image file; fail loudly on unreadable files."""
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise IOError(f"cannot read image {p}")
        yield str(p), img
