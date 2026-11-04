"""Pixel recorder (post-demo item 4): saves the raw frames the running worker publishes, with its detections and slot states, so the
evidence layer's veto thresholds can be tuned offline on real footage (scripts/replay_eval.py, scripts/flicker_report.py).

Usage:
  python scripts/pixel_record.py record --cam camera-1 --minutes 120 [--every 1.0] [--max-gb 4] [--out DIR]
  python scripts/pixel_record.py info DIR

Read only towards the stack: it only asks GET /api/status, /api/settings and /api/frame.jpg?raw=true. Everything is written to DIR
(default recordings/<camera>_pixels/<timestamp> in the project folder), never into ./data (refused). One frame is kept per --every
seconds of capture time; the recording stops after --minutes or at --max-gb of pictures. Leave it running across a change of light
(morning to afternoon, a cloud front, dusk) - the veto is judged on exactly those changes. The API token is read from the environment
variable named by --token-env (default CARPARK_VIEW_TOKEN, else CARPARK_API_TOKEN), sent only as a header, never printed or saved."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.pixel_recorder import SETTING_TO_TRACKER, PixelRecording, ensure_outside, record_pixels  # noqa: E402

SETTINGS_KEPT = (*SETTING_TO_TRACKER, "match_rule", "match_iou", "conf", "slot_mode")


def api_get(url: str, path: str, cam: str, token: str, raw: bool = False):
    sep = "&" if "?" in path else "?"
    full = f"{url.rstrip('/')}{path}" + (f"{sep}cam={cam}" if cam else "")
    req = urllib.request.Request(full, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read() if raw else json.load(r)


def main(argv=None, fetch_status=None, fetch_frame=None, fetch_settings=None, out=print, sleep=time.sleep, clock=time.monotonic) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("record")
    rc.add_argument("--url", default="http://localhost:8080")
    rc.add_argument("--cam", default="camera-1")
    rc.add_argument("--minutes", type=float, default=60.0)
    rc.add_argument("--every", type=float, default=1.0, help="seconds of capture time between saved frames")
    rc.add_argument("--poll", type=float, default=0.5)
    rc.add_argument("--max-gb", type=float, default=4.0, help="stop when the pictures take this much (0 = no limit)")
    rc.add_argument("--out", help="recording directory (default recordings/<camera>_pixels/<timestamp>)")
    rc.add_argument("--token-env", default="CARPARK_VIEW_TOKEN")
    inf = sub.add_parser("info")
    inf.add_argument("dir")
    args = ap.parse_args(argv)

    if args.cmd == "info":
        rec = PixelRecording(args.dir)
        h = rec.header
        out(f"{args.dir}: camera {h.get('camera')!r}, started {h.get('started')}, {len(rec)} frames over {rec.duration / 60:.1f} min, "
            f"{len(rec.slots)} slots, picture {h.get('size')}, settings {sorted(h.get('settings', {}))}")
        return 0

    token = os.environ.get(args.token_env) or os.environ.get("CARPARK_API_TOKEN", "")
    fetch_status = fetch_status or (lambda: api_get(args.url, "/api/status", args.cam, token))
    fetch_frame = fetch_frame or (lambda: api_get(args.url, "/api/frame.jpg?raw=true", args.cam, token, raw=True))
    fetch_settings = fetch_settings or (lambda: api_get(args.url, "/api/settings", args.cam, token))
    try:  # the tracker settings the replay needs; a recording without them replays with the tracker's own defaults
        values = (fetch_settings() or {}).get("values", {})
        settings = {k: values[k] for k in SETTINGS_KEPT if k in values}
    except Exception as e:  # noqa: BLE001 - any failure here only costs the settings
        settings = {}
        out(f"warning: could not read the worker's settings ({type(e).__name__}); the replay will use the tracker's defaults")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cam_dir = args.cam.replace("-", "").replace("_", "") + "_pixels"
    try:
        target = ensure_outside(args.out or ROOT / "recordings" / cam_dir / stamp, ROOT / "data")
    except ValueError as e:
        out(f"refused: {e}")
        return 2
    last = [0.0]

    def progress(n, size):
        if clock() - last[0] >= 30:
            last[0] = clock()
            out(f"  {n} frames, {size / 1e6:.0f} MB")

    summary = record_pixels(fetch_status, fetch_frame, target, args.minutes * 60, args.every,
                            int(args.max_gb * 1e9) if args.max_gb else None, args.poll, args.cam, settings, clock, sleep, progress)
    out(f"recorded {summary['frames']} frames ({summary['bytes'] / 1e6:.0f} MB, stopped on {summary['stopped']}) -> {summary['dir']}")
    if summary["errors"]:
        out(f"  {summary['errors']} requests failed and were retried")
    if summary["mismatched"]:
        out(f"  {summary['mismatched']} frames skipped: the picture and the status were not the same frame")
    if summary["slot_map_changed"]:
        out(f"  warning: the slot map changed in {summary['slot_map_changed']} frames; the replay uses the first map")
    return 0


if __name__ == "__main__":
    sys.exit(main())
