"""History sync: every registered camera's SQLite history -> PostgreSQL (carparking/history_pg.py).

Runs as the compose service "sync". Every CARPARK_SYNC_EVERY seconds (2) it copies new parking events and sessions
(and the exits of sessions that were open), records each camera's state (camera_state, camera_events) and, once a
minute, an occupancy sample. The first run migrates the existing history. Hourly: the retention policy
(CARPARK_RETENTION_DAYS, default 365, for sessions / events / camera events; CARPARK_SAMPLE_RETENTION_DAYS, default 90,
for occupancy samples; 0 = keep forever). Only PostgreSQL rows are ever deleted; the SQLite files stay complete.

Usage:
  python scripts/sync.py                 run (docker compose service "sync")
  python scripts/sync.py --once          one round, print what was copied
  python scripts/sync.py --verify        per camera: rows in SQLite vs PostgreSQL (exit 1 if a copy is behind)
  python scripts/sync.py --check         health: exit 0 if every camera was synced within --max-age seconds
  python scripts/sync.py --rebuild ID    forget one camera's copy (its SQLite file was replaced), copy it again
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.history_pg import HistorySync, PgHistory, sqlite_counts  # noqa: E402
from carparking.registry import PgRegistry, open_registry, runtime_paths  # noqa: E402

HISTORY_DB = Path(os.environ.get("CARPARK_HISTORY_DB", ROOT / "parking_history.db"))
FIXED_SLOTS = Path(os.environ.get("CARPARK_FIXED_SLOTS", ROOT / "config" / "fixed_slots.json"))
DATASET_RAW = Path(os.environ.get("CARPARK_DATASET_RAW", ROOT / "dataset_raw"))
DATA_DIR = HISTORY_DB.parent


def pairs(registry) -> list[tuple[str, Path]]:
    return [(c.id, runtime_paths(c, DATA_DIR, HISTORY_DB, FIXED_SLOTS, DATASET_RAW).db) for c in registry.list()]


def number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--max-age", type=float, default=60.0)
    ap.add_argument("--rebuild", metavar="CAMERA_ID")
    a = ap.parse_args()
    dsn = os.environ.get("CARPARK_DATABASE_URL")
    if not dsn:
        print("sync: CARPARK_DATABASE_URL is not set: nothing to sync (history stays in SQLite only)")
        return 0 if a.check else 1
    registry = open_registry()
    assert isinstance(registry, PgRegistry)  # also runs the migrations (003_history: the history tables)
    log = lambda m: print(m, flush=True)  # noqa: E731

    if a.check or a.verify:
        status = PgHistory(dsn).sync_status()
        bad = []
        for cam_id, path in pairs(registry):
            st = status.get(cam_id)
            if a.verify:
                src = sqlite_counts(path)
                copied = {"sessions": st["sessions_id"], "events": st["events_id"]} if st else {"sessions": 0, "events": 0}
                rows = st["rows"] if st else {"sessions": 0, "events": 0, "samples": 0}
                behind = src["sessions"] > copied["sessions"] or src["events"] > copied["events"]
                print(f"{cam_id}: SQLite sessions {src['sessions']} / events {src['events']} (highest ids)  |  "
                      f"PostgreSQL copied up to {copied['sessions']} / {copied['events']}, rows {rows['sessions']} / "
                      f"{rows['events']}, occupancy samples {rows['samples']}  |  "
                      f"{'BEHIND' if behind else 'in sync'}{'  error: ' + st['error'] if st and st['error'] else ''}")
                if behind:
                    bad.append(cam_id)
            elif st is None or st["age_s"] is None or st["age_s"] > a.max_age:
                age = "never" if not st or st["age_s"] is None else f"{st['age_s']:.0f} s"
                bad.append(f"{cam_id} ({age})")
        if a.check:
            print("all cameras synced" if not bad else "stale: " + ", ".join(bad))
        return 0 if not bad else 1

    sync = HistorySync(dsn, retention_days=number("CARPARK_RETENTION_DAYS", 365),
                       sample_days=number("CARPARK_SAMPLE_RETENTION_DAYS", 90), log=log)
    if a.rebuild:
        sync.rebuild(a.rebuild)
        print(f"sync: {a.rebuild}: copy forgotten; copying again")
        a.once = True
    if a.once:
        for cam_id, done in sync.step(pairs(registry)).items():
            print(f"{cam_id}: {done}")
        return 0

    every = number("CARPARK_SYNC_EVERY", 2.0)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    log(f"sync: every {every:g} s, {len(registry.list())} camera(s), retention {sync.retention_days:g} days "
        f"(occupancy samples {sync.sample_days:g} days)")
    first = True
    while not stop:
        try:
            done = sync.step(pairs(registry))
            for cam_id, d in done.items():
                if "error" in d:
                    log(f"sync: {cam_id}: {d['error']}")
                elif first or d.get("events") or d.get("sessions") or d.get("closed"):
                    log(f"sync: {cam_id}: copied {d}")
            first = False
        except Exception as e:  # PostgreSQL unreachable: the SQLite files keep everything, the next round catches up
            log(f"sync: {e}")
        t = time.monotonic()
        while not stop and time.monotonic() - t < every:
            time.sleep(0.2)
    sync.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
