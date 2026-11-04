"""Offline replay of a pixel recording through FixedSlotTracker, evidence layer off and on side by side (post-demo item 4).

Usage:
  python scripts/replay_eval.py DIR                         live (as recorded) | replay off | replay on (EvidenceCfg defaults)
  python scripts/replay_eval.py DIR --set motion_floor=4    ... with these EvidenceCfg values for the "on" column
  python scripts/replay_eval.py DIR --sweep motion_ratio=1.5,2,3 [--sweep min_ncc=0.5,0.6]
                                                            one "on" replay per combination, one table row each, to derive a value
                                                            empirically; NO value is preferred or built in here
  python scripts/replay_eval.py DIR --obstacle              adds a column with the non-vehicle filter (obstacle_filter) on, evidence layer off: the
                                                            detections it drops, by reason, and whether anything else changed
  options: --truth FILE   the real departures seen in the footage: {"exits": [{"slot": id or name, "t": epoch s | "t_rel": s after the
                          first frame}]}; adds missed / false exits and their lag (false exits are only meaningful if the file lists
                          every real departure of the recording)
           --window S --blink S   the flicker measurement (see flicker_report.py)   --json FILE   everything as data

How to read it: a good setting has no more ghost pairs than the layer needs to remove (ideally 0 on footage with parked cars only),
keeps every real exit (missed = 0 with a truth file) and reports it with a small lag. The replay cannot use the worker's zoom re-look (the
veto's 'relook' reason), so it is slightly less protective than the live layer would be; the 'replay off' column should agree with 'live'
(the agreement line shows by how much; a low figure means the offline tracker setup differs from the worker's).
The recording is only read; nothing else is written except --json FILE."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.flicker import analyze, compare, report_lag, score_exits  # noqa: E402
from carparking.pixel_replay import PixelRecording, cfg_with, parse_cfg_value, replay  # noqa: E402


def pairs_of(items, many=False):
    out = {}
    for p in items or []:
        name, sep, text = p.partition("=")
        if not sep:
            raise ValueError(f"{p!r}: use " + ("name=v1,v2,..." if many else "name=value"))
        name = name.strip()
        out[name] = [parse_cfg_value(name, t.strip()) for t in text.split(",")] if many else parse_cfg_value(name, text.strip())
    return out


def load_truth(path, rec) -> list[dict]:
    ids = {s["id"]: s["id"] for s in rec.slots} | {s["name"]: s["id"] for s in rec.slots}
    out = []
    for e in json.loads(Path(path).read_text(encoding="utf-8")).get("exits", []):
        if e["slot"] not in ids:
            raise ValueError(f"truth: unknown slot {e['slot']!r}")
        out.append({"slot": ids[e["slot"]], "t": e["t"] if "t" in e else rec.frames[0]["t"] + e["t_rel"]})
    return out


def agreement(live: dict, shown: dict) -> float:
    n = same = 0
    for k, series in shown.items():
        for a, b in zip(live[k], series):
            n += 1
            same += a == b
    return same / n if n else 1.0


def main(argv=None, out=print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--set", action="append", dest="sets", metavar="NAME=VALUE")
    ap.add_argument("--sweep", action="append", metavar="NAME=V1,V2,...")
    ap.add_argument("--truth")
    ap.add_argument("--obstacle", action="store_true", help="add a replay with the non-vehicle filter on")
    ap.add_argument("--window", type=float, default=120.0)
    ap.add_argument("--blink", type=float, default=60.0)
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)
    try:
        rec = PixelRecording(args.dir)
        base = pairs_of(args.sets)
        sweep = pairs_of(args.sweep, many=True)
        cfg = cfg_with(**base)
        truth = load_truth(args.truth, rec) if args.truth else None
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        out(f"error: {e}")
        return 2
    names = {s["id"]: s["name"] for s in rec.slots}
    out(f"{args.dir}: {len(rec)} frames over {rec.duration / 60:.1f} min, {len(rec.slots)} slots")
    live = rec.live_states()
    times = [f["t"] for f in rec.frames]
    reports, data = [analyze(times, live, names, "live", args.window, args.blink)], {"runs": []}
    off = replay(rec, 0, label="replay off")
    reports.append(analyze(off.times, off.shown, off.names, off.label, args.window, args.blink))
    out(f"replay off agrees with live on {agreement(live, off.shown):.1%} of slot-frames")
    if args.obstacle:
        obs = replay(rec, 0, tracker_kwargs={"obstacle_filter": 1}, label="replay obstacle")
        reports.append(analyze(obs.times, obs.shown, obs.names, obs.label, args.window, args.blink))
        data["obstacle"] = {"rejected": obs.rejected, "same_events_as_off": obs.events == off.events,
                            "same_states_as_off": obs.shown == off.shown}
        out(f"obstacle filter: dropped {sum(obs.rejected.values())} detections {obs.rejected or ''}; events {'identical to' if obs.events == off.events else 'DIFFER from'} "
            f"replay off, shown states {'identical' if obs.shown == off.shown else 'DIFFER'}")

    def run_on(label, c):
        res = replay(rec, 1, c, label=label)
        rep = analyze(res.times, res.shown, res.names, label, args.window, args.blink)
        row = {"label": label, "cfg_overrides": {k: getattr(c, k) for k in {**base, **{k: 0 for k in sweep}}},
               "flicker": rep.summary(), "lag": report_lag(res.events), "vetoes": sum(res.vetoes.values())}
        if truth is not None:
            sc = score_exits(res.events, truth)
            row["truth"] = {"found": sc["found"], "missed": len(sc["missed"]), "false_exits": len(sc["false_exits"]),
                            "lag_s": sc["lag_s"]}
        return rep, row

    if not sweep:
        rep, row = run_on("replay on", cfg)
        reports.append(rep)
        data["runs"].append(row)
        out(compare(reports))
        out(f"evidence layer: {row['vetoes']} exit probes vetoed; report lag exits {row['lag']['exit']}, entries {row['lag']['entry']}")
        out(f"  off:           report lag exits {report_lag(off.events)['exit']}")
        if truth is not None:
            out(f"truth: {row['truth']}  (off: {score_exits(off.events, truth)['found']} found, "
                f"{len(score_exits(off.events, truth)['false_exits'])} false)")
    else:
        out(compare(reports))
        out("")
        keys = list(sweep)
        head = "".join(f"{k:>14}" for k in keys) + f"{'ghost pairs':>13}{'exits':>7}{'vetoes':>8}{'lag med s':>11}{'lag max s':>11}"
        head += f"{'missed':>8}{'false':>7}" if truth is not None else ""
        out(head)
        for combo in itertools.product(*(sweep[k] for k in keys)):
            c = cfg_with(cfg, **dict(zip(keys, combo)))
            rep, row = run_on(" ".join(f"{k}={v}" for k, v in zip(keys, combo)), c)
            data["runs"].append(row)
            lag = row["lag"]["exit"]
            line = "".join(f"{v:>14}" for v in combo) + f"{rep.ghost_pairs:>13}{rep.exits:>7}{row['vetoes']:>8}"
            line += f"{'-' if lag['median_s'] is None else lag['median_s']:>11}{'-' if lag['max_s'] is None else lag['max_s']:>11}"
            line += f"{row['truth']['missed']:>8}{row['truth']['false_exits']:>7}" if truth is not None else ""
            out(line)
    data["off"] = {"flicker": reports[1].summary(), "lag": report_lag(off.events)}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(data, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
