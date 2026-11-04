"""Collect raw frames from an RTSP camera for fine-tuning: one full-resolution JPEG every N seconds.

Runs unattended (hours or days) next to the app; near-duplicate frames (nothing moved, same light) are skipped, so
leaving it on overnight does not fill the disk with copies of an empty lot. Ctrl+C stops it.

Usage (inside the container, files land in the bind-mounted project folder):
  docker exec -it carparking python scripts/collect_dataset.py --camera "Front Parking 02" --every 60
  docker exec -d carparking python scripts/collect_dataset.py --camera "Front Parking 02" --every 120 --hours 72
  python scripts/collect_dataset.py --stream sub --every 10 --count 50        # quick test
Then: python scripts/auto_label.py (pre-labels), review the labels, yolo detect train (see auto_label.py).
URL: --url, else $CARPARK_RTSP_URL, else CARPARK_RTSP_URL in the project's .env. Times in $CARPARK_TZ (default Asia/Karachi).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.dataset import change, save_frame  # noqa: E402
from carparking.rtsp import LatestFrameReader, env_setting, mask_url  # noqa: E402

DEFAULT_URL = env_setting("CARPARK_RTSP_URL", ROOT / ".env")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--stream", choices=["main", "sub"], default="main",
                    help="Hikvision channel ending 01 (full resolution, use this for training) or 02")
    ap.add_argument("--camera", default="Front Parking 02", help="file name prefix")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset_raw")
    ap.add_argument("--every", type=float, default=60, help="seconds between saved frames")
    ap.add_argument("--min-change", type=float, default=1.5,
                    help="skip a frame if it differs from the last saved one by less than this (mean grey level "
                         "difference 0..255 at 64x36). 0 = save every frame")
    ap.add_argument("--count", type=int, default=0, help="stop after this many frames (0 = no limit)")
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = no limit)")
    a = ap.parse_args()

    if not a.url:
        ap.error("no camera URL: pass --url or set CARPARK_RTSP_URL (environment or .env)")
    url = re.sub(r"(Channels/\d*?)0?[12]$", lambda m: m.group(1) + ("02" if a.stream == "sub" else "01"), a.url)
    tz = ZoneInfo(os.environ.get("CARPARK_TZ", "Asia/Karachi"))
    reader = LatestFrameReader(url)
    print(f"{mask_url(url)} -> {a.out}  (every {a.every:g} s, min change {a.min_change:g}; Ctrl+C to stop)")
    stop_at = time.time() + a.hours * 3600 if a.hours else float("inf")
    last, saved, skipped, next_t, status = None, 0, 0, 0.0, ""
    try:
        while time.time() < stop_at and not (a.count and saved >= a.count):
            frame, _, t = reader.latest()
            if reader.status != status:
                status = reader.status
                print(f"[{time.strftime('%H:%M:%S')}] {status}", flush=True)
            if frame is None or time.time() < next_t or time.time() - t > 5:  # nothing yet, not due, or stale
                time.sleep(0.2)
                continue
            next_t = time.time() + a.every
            if last is not None and change(frame, last) < a.min_change:
                skipped += 1
                continue
            path = save_frame(frame, a.out, a.camera, t, tz)
            last, saved = frame, saved + 1
            print(f"[{time.strftime('%H:%M:%S')}] #{saved} {path.name} ({frame.shape[1]}x{frame.shape[0]}), "
                  f"{skipped} near-duplicates skipped", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
    print(f"done: {saved} frames saved to {a.out}, {skipped} near-duplicates skipped")


if __name__ == "__main__":
    main()
