"""Flicker report (post-demo item 4): ghost exit/entry pairs and state stability of a pixel recording (scripts/pixel_record.py).

Usage:
  python scripts/flicker_report.py DIR                     the states the running worker showed while it was recorded (no replay; fast)
  python scripts/flicker_report.py DIR --source all        live, replay with the evidence layer off, replay with it on, side by side
  python scripts/flicker_report.py DIR --source on --set motion_ratio=2.5 --set min_ncc=0.6
                                                            replay with the layer on and these EvidenceCfg values (others: the defaults)
  options: --window S (a ghost pair = exit then entry of one slot within S seconds, default 120), --blink S (a state shorter than this
           between two changes is a short state, default 60), --json FILE

The window and blink are measurement choices, not tracker settings. Nothing is written except --json FILE; the recording is only read."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.flicker import analyze, compare, report_lag  # noqa: E402
from carparking.pixel_replay import PixelRecording, cfg_with, parse_cfg_value, replay  # noqa: E402


def parse_sets(pairs):
    out = {}
    for p in pairs or []:
        name, sep, text = p.partition("=")
        if not sep:
            raise ValueError(f"--set {p!r}: use name=value")
        out[name.strip()] = parse_cfg_value(name.strip(), text.strip())
    return out


def main(argv=None, out=print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--source", choices=["live", "off", "on", "all"], default="live")
    ap.add_argument("--set", action="append", dest="sets", metavar="NAME=VALUE")
    ap.add_argument("--window", type=float, default=120.0)
    ap.add_argument("--blink", type=float, default=60.0)
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)
    try:
        rec = PixelRecording(args.dir)
        cfg = cfg_with(**parse_sets(args.sets))
    except ValueError as e:
        out(f"error: {e}")
        return 2
    names = {s["id"]: s["name"] for s in rec.slots}
    out(f"{args.dir}: {len(rec)} frames over {rec.duration / 60:.1f} min, {len(rec.slots)} slots")
    reports, lags = [], {}
    if args.source in ("live", "all"):
        reports.append(analyze([f["t"] for f in rec.frames], rec.live_states(), names, "live", args.window, args.blink))
    for src in ("off", "on"):
        if args.source in (src, "all"):
            res = replay(rec, 1 if src == "on" else 0, cfg if src == "on" else None, label=f"replay {src}")
            reports.append(analyze(res.times, res.shown, res.names, res.label, args.window, args.blink))
            lags[res.label] = report_lag(res.events)
    out(compare(reports))
    for label, lag in lags.items():
        out(f"report lag ({label}): exits {lag['exit']}, entries {lag['entry']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"dir": args.dir, "reports": [r.summary() for r in reports], "lag": lags}, indent=1),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
