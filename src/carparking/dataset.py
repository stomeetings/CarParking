"""Camera dataset helpers: save raw RTSP frames for annotation (scripts/collect_dataset.py, Mode 5 snapshots)."""
from __future__ import annotations

import re
import time
from datetime import datetime, tzinfo
from pathlib import Path

import cv2
import numpy as np


def slug(name: str) -> str:
    """'Front Parking 02' -> 'front_parking_02' (file names, model names)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "camera"


def save_frame(frame, out_dir, camera: str, t: float | None = None, tz: tzinfo | None = None) -> Path:
    """Full-resolution raw frame (no overlay) -> out_dir/<camera>_YYYYmmdd_HHMMSS_mmm.jpg (JPEG q95: ~1 MB at 2560 px,
    no visible loss for labelling). Time-sorted names, so a split by capture time is a split by file name."""
    t = time.time() if t is None else t
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(t, tz)
    path = out / f"{slug(camera)}_{stamp:%Y%m%d_%H%M%S}_{stamp.microsecond // 1000:03d}.jpg"
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"could not write {path}")
    return path


def change(a, b) -> float:
    """Mean absolute grey difference (0..255) of two frames at 64x36: ~0-2 for the same static scene with sensor
    noise, more once something moved or the light changed. Used to skip near-duplicate frames."""
    small = lambda f: cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)  # noqa: E731
    return float(np.abs(small(a).astype(np.int16) - small(b).astype(np.int16)).mean())
