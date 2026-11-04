"""Headless Mode 5 worker: camera -> ParkingMonitor -> entry/exit history, independent of any browser tab.

Runs the settings saved by the Streamlit page (Mode 5 sidebar, table live_config of the history database) and
publishes a heartbeat, a status snapshot and the overlay image for the page (table live_state). See
carparking/live.py. One worker at a time: a second one waits on standby until the first stops.

Usage:
  python scripts/worker.py                 run until stopped (or until the page asks for a restart)
  python scripts/worker.py --check         health check: exit 0 if a worker's heartbeat is fresh (docker compose)
  python scripts/worker.py --max-frames 5  stop after 5 processed frames (smoke tests)
Environment: CARPARK_HISTORY_DB, CARPARK_DATASET_RAW, CARPARK_TZ, CARPARK_RTSP_URL (else the project's .env),
CARPARK_MODELS_DIR (downloaded weights, default next to the database), CARPARK_YOLO11M_MIN_CORES / _MIN_RAM_GB.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.geometry_store import geometry_enabled  # noqa: E402
from carparking.live import PULSE_EVERY, LiveStore, Worker, heartbeat_age  # noqa: E402
from carparking.rtsp import env_setting  # noqa: E402
from carparking.stream_health import health_from_env  # noqa: E402

HISTORY_DB = Path(os.environ.get("CARPARK_HISTORY_DB", ROOT / "parking_history.db"))
DATASET_RAW = Path(os.environ.get("CARPARK_DATASET_RAW", ROOT / "dataset_raw"))
# downloaded weights (yolo11m.pt): next to the database, so under compose on the data volume (kept across rebuilds)
MODELS_DIR = Path(os.environ.get("CARPARK_MODELS_DIR", HISTORY_DB.parent / "models"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 0 if a worker's heartbeat is younger than --max-age")
    ap.add_argument("--max-age", type=float, default=60.0, help="seconds (--check)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many processed frames (0 = no limit)")
    a = ap.parse_args()

    store = LiveStore(HISTORY_DB)
    if a.check:
        age = heartbeat_age(store.state(images=False))
        print("no worker has run" if age is None else f"last heartbeat {age:.1f} s ago")
        return 0 if age is not None and age < a.max_age else 1

    try:
        tz = ZoneInfo(os.environ.get("CARPARK_TZ", "Asia/Karachi"))
    except Exception:  # unknown name or no tzdata: the process clock
        tz = None
    geometry = None
    if geometry_enabled(os.environ):  # Phase 2, opt-in: only then is any of it imported or read
        from carparking.geometry_live import feed_from_env
        geometry = feed_from_env(os.environ, store, log=lambda m: print(m, flush=True))
    worker = Worker(store, ROOT, HISTORY_DB, tz, DATASET_RAW, env_setting("CARPARK_RTSP_URL", ROOT / ".env"),
                    models_dir=MODELS_DIR, geometry=geometry, health_factory=lambda: health_from_env(os.environ),
                    pulse_every=float(os.environ.get("CARPARK_PULSE_S", PULSE_EVERY)))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # docker stop: release the lease on the way out
    print(f"worker {worker.owner}: history {HISTORY_DB}, waiting for settings from the page", flush=True)
    print(f"worker {worker.owner}: hardware {worker.hw} -> auto model: {worker.auto_choice} ({worker.auto_reason})",
          flush=True)
    t0 = time.time()
    try:
        worker.loop(a.max_frames)
    except KeyboardInterrupt:
        pass
    print(f"worker {worker.owner}: stopped after {time.time() - t0:.0f} s"
          + (" (restart requested)" if worker.exit else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
