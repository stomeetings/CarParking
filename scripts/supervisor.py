"""Multi-camera worker supervisor: one scripts/worker.py process per registered camera (carparking/supervisor.py).

The registry (cameras + slot coordinates) is PostgreSQL when CARPARK_DATABASE_URL is set (compose service "db"),
else cameras.json next to the history database. On the first start with an empty registry, the existing single
camera (CARPARK_RTSP_URL / CARPARK_RTSP_SUB_URL, its name from the current settings) becomes the primary camera, so
nothing changes for it.

Usage:
  python scripts/supervisor.py           run (docker compose service "worker")
  python scripts/supervisor.py --check   health: exit 0 if every camera's worker has a fresh heartbeat
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

from carparking.live import LiveStore, heartbeat_age  # noqa: E402
from carparking.registry import open_registry, runtime_paths, seed  # noqa: E402
from carparking.rtsp import env_setting  # noqa: E402
from carparking.supervisor import Supervisor  # noqa: E402

HISTORY_DB = Path(os.environ.get("CARPARK_HISTORY_DB", ROOT / "parking_history.db"))
FIXED_SLOTS = Path(os.environ.get("CARPARK_FIXED_SLOTS", ROOT / "config" / "fixed_slots.json"))
DATASET_RAW = Path(os.environ.get("CARPARK_DATASET_RAW", ROOT / "dataset_raw"))
DATA_DIR = HISTORY_DB.parent
MODELS_DIR = Path(os.environ.get("CARPARK_MODELS_DIR", DATA_DIR / "models"))


def primary_name() -> str:
    cfg, _ = LiveStore(HISTORY_DB).config()
    return ((cfg or {}).get("run") or {}).get("camera") or "Camera 1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--max-age", type=float, default=60.0)
    a = ap.parse_args()
    registry = open_registry(data_dir=DATA_DIR)

    if a.check:
        stale = []
        for cam in registry.list():
            age = heartbeat_age(LiveStore(runtime_paths(cam, DATA_DIR, HISTORY_DB, FIXED_SLOTS, DATASET_RAW).db)
                                .state(images=False))
            if age is None or age > a.max_age:
                stale.append(f"{cam.id} ({'never' if age is None else f'{age:.0f} s'})")
        print("all workers beating" if not stale else "stale: " + ", ".join(stale))
        return 0 if not stale else 1

    made = seed(registry, primary_name(), env_setting("CARPARK_RTSP_URL", ROOT / ".env"),
                env_setting("CARPARK_RTSP_SUB_URL", ROOT / ".env"), FIXED_SLOTS)
    if made:
        print(f"supervisor: registry seeded with the primary camera {made.id} ({made.name})", flush=True)
    try:
        tz = ZoneInfo(os.environ.get("CARPARK_TZ", "Asia/Karachi"))
    except Exception:
        tz = None
    sup = Supervisor(registry, ROOT, DATA_DIR, HISTORY_DB, FIXED_SLOTS, DATASET_RAW, MODELS_DIR, dict(os.environ),
                     tz=tz, log=lambda m: print(m, flush=True))
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    print(f"supervisor: registry {type(registry).__name__}, {len(registry.list())} camera(s)", flush=True)
    try:
        while not stop:
            try:
                sup.step()
            except Exception as e:  # the database briefly unreachable: keep the running workers, retry
                print(f"supervisor: {e}", flush=True)
            for _ in range(10):
                if stop:
                    break
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        sup.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
