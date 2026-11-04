"""Ground truth: per-bay occupancy for a CNR-EXT full frame, joined from CNRPark+EXT.csv.

Full frames ship without labels. Labels exist per *patch* (one crop of one bay at one moment) in
CNRPark+EXT.csv. We join them back on (camera, date, time, slot_id), keyed from file names:
  patch : .../S_2015-11-12_07.09_C01_184.jpg  -> weather S, date, 07:09, camera 1, slot 184
  frame : .../camera1/2015-11-12_0709.jpg     -> camera 1, date, 07:09
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import PurePath

PATCH_RE = re.compile(r"^[A-Z]_(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})_C(\d{2})_(\d+)\.jpg$")
FRAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.jpg$")
CAMERA_RE = re.compile(r"^camera(\d+)$")

FREE, OCCUPIED = 0, 1


def frame_key(path) -> tuple[int, str, str]:
    """(camera, 'YYYY-MM-DD', 'HHMM') from a full-frame path like .../camera1/2015-11-12_0709.jpg."""
    p = PurePath(path)
    m, c = FRAME_RE.match(p.name), CAMERA_RE.match(p.parent.name)
    if not (m and c):
        raise ValueError(f"not a CNR-EXT full-frame path: {path}")
    return int(c.group(1)), m.group(1), m.group(2) + m.group(3)


def load_labels(csv_path) -> dict[tuple[int, str, str], dict[int, int]]:
    """CNRPark+EXT.csv -> {(camera, date, HHMM): {slot_id: 0 free / 1 occupied}} for CNR-EXT rows only."""
    labels = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            url = row["image_url"]
            if "CNR-EXT" not in url:  # older CNRPark subset uses other cameras/naming
                continue
            m = PATCH_RE.match(PurePath(url).name)
            if not m:
                raise ValueError(f"unexpected CNR-EXT patch name: {url}")
            date, hh, mm, cam, slot = m.groups()
            occ = int(row["occupancy"])
            if occ not in (FREE, OCCUPIED):
                raise ValueError(f"bad occupancy {occ!r} for {url}")
            labels[(int(cam), date, hh + mm)][int(slot)] = occ
    return dict(labels)
