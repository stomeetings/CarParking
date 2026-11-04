"""One worker process per registered camera (scripts/supervisor.py runs this; the compose service "worker").

Every few seconds (step): read the registry; for each camera make sure a scripts/worker.py process runs with that
camera's environment (its own SQLite file, fixed-slot file, raw-frame folder, RTSP URLs: carparking.registry
runtime_paths); restart it when that environment changes (a new URL), and after a crash (with backoff); stop the
workers of removed cameras. The worker itself is unchanged: one camera, its settings (monitoring on/off, Fixed /
Dynamic, detector, timers) in its own database, idle while monitoring is off (no model loaded, no camera opened).
Slot maps: the registry is the source; each camera's fixed-slot file is kept in step for its worker (the primary
camera's file is also the Streamlit editor's, so there the newer side wins: sync_primary_slots).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from carparking.fixed_slots import FixedSlotStore
from carparking.registry import Registry, mirror_slots, runtime_paths, sync_primary_slots


class Supervisor:
    """registry: carparking.registry.Registry. base_env: the environment every worker inherits. spawn(env) -> a
    process-like object (poll() -> None while running, terminate(), wait(timeout)); default: scripts/worker.py."""

    def __init__(self, registry: Registry, root, data_dir, primary_db, primary_fixed, primary_raw=None,
                 models_dir=None, base_env=None, spawn=None, clock=time.time, tz=None, backoff=(5, 60), log=print):
        self.registry, self.root, self.data_dir = registry, Path(root), Path(data_dir)
        self.primary = (Path(primary_db), Path(primary_fixed), Path(primary_raw) if primary_raw else None)
        self.models_dir = Path(models_dir) if models_dir else self.data_dir / "models"
        self.base_env = dict(base_env or {})
        self.spawn = spawn or self._spawn
        self.clock, self.tz, self.backoff, self.log = clock, tz, backoff, log
        self.procs: dict[str, dict] = {}  # cam id -> {"proc", "env", "started", "fails"}

    def _spawn(self, env: dict):
        return subprocess.Popen([sys.executable, str(self.root / "scripts" / "worker.py")], env=env, cwd=str(self.root))

    def env_for(self, cam) -> dict:
        rt = runtime_paths(cam, self.data_dir, *self.primary)
        return {**self.base_env, "CARPARK_HISTORY_DB": str(rt.db), "CARPARK_FIXED_SLOTS": str(rt.fixed),
                "CARPARK_DATASET_RAW": str(rt.raw), "CARPARK_RTSP_URL": cam.rtsp_url,
                "CARPARK_RTSP_SUB_URL": cam.rtsp_sub_url, "CARPARK_MODELS_DIR": str(self.models_dir),
                "CARPARK_CAMERA_ID": cam.id, "CARPARK_CAMERA_NAME": cam.name}

    def sync_slots(self, cam) -> str | None:
        rt = runtime_paths(cam, self.data_dir, *self.primary)
        if cam.primary:
            return sync_primary_slots(self.registry, cam, rt.fixed, self.tz)
        if FixedSlotStore(rt.fixed).load(cam.name) != self.registry.slots(cam.id)["slots"]:
            mirror_slots(self.registry, cam, rt.fixed, self.tz)
            return "exported"
        return None

    def _stop(self, cam_id: str, why: str):
        entry = self.procs.pop(cam_id, None)
        if entry is None:
            return
        proc = entry["proc"]
        if proc.poll() is None:
            proc.terminate()  # the worker releases its lease on SIGTERM
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        self.log(f"supervisor: stopped worker {cam_id} ({why})")

    def step(self):
        now = self.clock()
        cams = self.registry.list()
        for cam in cams:
            try:
                if (what := self.sync_slots(cam)) is not None:
                    self.log(f"supervisor: {cam.id} slot map {what}")
            except Exception as e:  # a broken file must not stop the other cameras
                self.log(f"supervisor: {cam.id} slot sync failed: {e}")
            env = self.env_for(cam)
            entry = self.procs.get(cam.id)
            if entry is not None and entry["env"] != env:
                self._stop(cam.id, "its camera settings changed")
                entry = None
            if entry is not None and entry["proc"].poll() is not None:
                code = entry["proc"].poll()
                fails = entry["fails"] + (0 if code == 0 else 1)  # 0 = a requested restart
                wait = min(self.backoff[0] * 2 ** max(fails - 1, 0), self.backoff[1]) if fails else 0
                if now - entry["started"] < wait:
                    continue
                self.log(f"supervisor: worker {cam.id} exited ({code}), restarting")
                self.procs.pop(cam.id)
                entry = {"fails": fails}
            if entry is None or "proc" not in entry:
                Path(env["CARPARK_HISTORY_DB"]).parent.mkdir(parents=True, exist_ok=True)
                proc = self.spawn(env)
                self.procs[cam.id] = {"proc": proc, "env": env, "started": now, "fails": (entry or {}).get("fails", 0)}
                self.log(f"supervisor: started worker {cam.id} ({cam.name})")
        for cam_id in set(self.procs) - {c.id for c in cams}:
            self._stop(cam_id, "camera removed")

    def stop_all(self):
        for cam_id in list(self.procs):
            self._stop(cam_id, "supervisor stopping")
