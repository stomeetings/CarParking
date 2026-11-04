"""Offline tracking evaluation (item 104): would ByteTrack-style IDs steady a capacity pin's vehicle count? Read only: the live
system is only ever asked for its status; nothing is written to it, to ./data or to the history.

Usage:
  python scripts/track_eval.py selftest [--minutes 10] [--seed 0]     a synthetic scene with ground truth through the trackers
  python scripts/track_eval.py record --cam camera-1 --minutes 30 --out rec.json
                                      sample the running worker's own detections (GET /api/status) at its real rate
  python scripts/track_eval.py eval rec.json [--tracker bytetrack iou ultralytics] [--json report.json]
                                      replay a recording; with ground truth ("gt" per frame) it also scores IDs and counts

The API token is read from the environment variable named by --token-env (default CARPARK_VIEW_TOKEN, else CARPARK_API_TOKEN) and
is only sent as a header; it is never printed. The ultralytics tracker needs `lap` and `scipy` (not in the runtime image); when they are
missing it is reported as unavailable and nothing is installed."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.track_eval import (TRACKERS, format_report, load_recording, record_from_api, run_all,  # noqa: E402
                                   save_recording, synthesize)


def api_fetch(url: str, cam: str, token: str):
    """A fetch() for record_from_api: the status JSON of one camera, the token as a Bearer header only."""
    full = f"{url.rstrip('/')}/api/status" + (f"?cam={cam}" if cam else "")
    req = urllib.request.Request(full, headers={"Authorization": f"Bearer {token}"} if token else {})

    def fetch():
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    return fetch


def main(argv=None, fetch=None, out=print, sleep=time.sleep, clock=time.monotonic) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest")
    st.add_argument("--minutes", type=float, default=10.0)
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--fps", type=float, default=1.0)
    rc = sub.add_parser("record")
    rc.add_argument("--url", default="http://localhost:8080")
    rc.add_argument("--cam", default="")
    rc.add_argument("--minutes", type=float, default=30.0)
    rc.add_argument("--poll", type=float, default=0.5)
    rc.add_argument("--out", required=True)
    rc.add_argument("--token-env", default="CARPARK_VIEW_TOKEN")
    ev = sub.add_parser("eval")
    ev.add_argument("recording")
    ev.add_argument("--tracker", nargs="+", default=["bytetrack", "iou"], choices=list(TRACKERS))
    ev.add_argument("--conf-min", type=float, default=0.3)
    ev.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        rec = synthesize(args.seed, args.minutes, args.fps)
        out(f"synthetic scene: {len(rec.frames)} frames, {rec.duration / 60:.1f} min, {len(rec.slots)} slots")
        reports = run_all(rec, ("bytetrack", "iou"))
    elif args.cmd == "record":
        token = os.environ.get(args.token_env) or os.environ.get("CARPARK_API_TOKEN", "")
        rec = record_from_api(fetch or api_fetch(args.url, args.cam, token), args.minutes * 60, args.poll, clock, sleep, args.cam)
        save_recording(rec, args.out)
        out(f"recorded {len(rec.frames)} frames over {rec.duration / 60:.1f} min, {len(rec.slots)} slots -> {args.out}")
        return 0
    else:
        rec = load_recording(args.recording)
        reports = run_all(rec, args.tracker, args.conf_min)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    out(format_report(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
