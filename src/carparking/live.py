"""Mode 5 as a background service: a headless worker runs the camera -> ParkingMonitor loop; the Streamlit page is a
viewer that reads what the worker publishes and writes the settings the worker runs with.

Everything goes through the history database (one SQLite file, shared by both processes), in three tables:
  * live_config    one row: the Mode 5 settings as JSON + a version number (the page writes, the worker polls)
  * live_state     one row: the worker's heartbeat and lease owner, a status snapshot (JSON), the latest overlay
                   image (JPEG) and a raw frame (JPEG, for the fixed slot editor)
  * live_commands  page -> worker requests: "reset" (slot map), "snapshot" (save a raw frame), "restart"
One worker at a time: claim() is a lease on live_state (heartbeat younger than `lease` seconds = taken); a second
worker waits on standby and takes over when the first one stops beating. So two browser tabs, or a page that starts
a worker while another runs, never log an entry twice.

Worker.step() is one pass of the loop (tests drive it with a fake reader and detector); scripts/worker.py runs it.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from carparking.dataset import save_frame
from carparking.fixed_slots import DYNAMIC_BASE, clean_expected, validate_slots
from carparking.history import HistoryDB, tuning_values
from carparking.monitor import ParkingArea, ParkingMonitor
from carparking.rtsp import LatestFrameReader, mask_url
from carparking.stream_health import ACTIVE, StreamHealth
from carparking.viz import draw

LEASE = 20.0  # seconds without a heartbeat after which another worker may take over
STALE = 15.0  # seconds without a heartbeat after which the page shows "worker down"
PULSE_EVERY = 3.0  # seconds between the background heartbeats (Pulse); well under STALE and LEASE
PULSE_STALL = 180.0  # the background heartbeat stops when the worker loop has made no progress for this long (a hung worker must look down)
MISMATCH_LOG_S = 30.0  # the history differing from the live count for this long is logged (and the History page warns)
VIEW_W = 1920  # overlay / raw frame width published to the pages (detection always uses the full frame)


def stream_url(base: str, stream: str) -> str:
    """Hikvision channel path .../Channels/<camera><stream>: 101 = camera 1 main, 102 = camera 1 sub-stream."""
    return re.sub(r"(Channels/\d*?)0?[12]$", lambda m: m.group(1) + ("02" if stream == "sub" else "01"), base)


def code_version(root) -> float:
    """Newest modification time of the worker's code (src/carparking, scripts/worker.py): the page compares it with
    the running worker's to tell that the worker still runs older code."""
    root = Path(root)
    files = list((root / "src" / "carparking").glob("*.py")) + [root / "scripts" / "worker.py"]
    return max((p.stat().st_mtime for p in files if p.exists()), default=0.0)


def _jpeg(img, width: int = VIEW_W, quality: int = 85) -> bytes:
    if img.shape[1] > width:  # 1920 px: sharp on the dashboard; a 2560 px image per frame is too heavy
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def decode_jpeg(data: bytes | None):
    return None if not data else cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class LiveStore:
    """The three live_* tables in the history database (see the module doc). Autocommit connection: each call is its
    own transaction, claim() an IMMEDIATE one, so two processes can't both take the lease."""

    def __init__(self, path, timeout: float = 10.0):
        self.path = path
        self.con = sqlite3.connect(str(path), timeout=timeout, isolation_level=None, check_same_thread=False)
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS live_config (id INTEGER PRIMARY KEY CHECK (id = 1), ver INTEGER, cfg TEXT,
                ts REAL);
            CREATE TABLE IF NOT EXISTS live_state (id INTEGER PRIMARY KEY CHECK (id = 1), owner TEXT, heartbeat REAL,
                snapshot TEXT, overlay BLOB, raw BLOB, raw_ts REAL);
            CREATE TABLE IF NOT EXISTS live_options (key TEXT PRIMARY KEY, value TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS live_commands (id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT, ts REAL,
                done REAL);
        """)

    def close(self):
        self.con.close()

    def geometry(self, camera_id: str):
        """Phase 2: this camera's versioned geometry (carparking/geometry_store.py), in the same database. The table is
        created on first use, so a store nobody asks for is exactly as before; no active version = uncalibrated."""
        from carparking.geometry_store import sqlite_geometry
        return sqlite_geometry(self.con, camera_id)

    # settings: written by the page, polled by the worker
    def config(self) -> tuple[dict | None, int]:
        row = self.con.execute("SELECT cfg, ver FROM live_config WHERE id = 1").fetchone()
        return (None, 0) if row is None else (json.loads(row[0]), row[1])

    def set_config(self, cfg: dict) -> int:
        """Store cfg as the new version (only if it differs from the stored one) -> the current version."""
        old, ver = self.config()
        if old == cfg:
            return ver
        self.con.execute("INSERT INTO live_config (id, ver, cfg, ts) VALUES (1, ?, ?, ?) ON CONFLICT (id) DO UPDATE "
                         "SET ver = excluded.ver, cfg = excluded.cfg, ts = excluded.ts",
                         (ver + 1, json.dumps(cfg), time.time()))
        return ver + 1

    # worker lease, heartbeat and what it publishes
    def claim(self, owner: str, lease: float = LEASE, now: float | None = None) -> bool:
        """Take or keep the worker lease: True if owner holds it now (free, its own, or the holder's heartbeat is older
        than lease seconds)."""
        now = time.time() if now is None else now
        cur = self.con.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            row = cur.execute("SELECT owner, heartbeat FROM live_state WHERE id = 1").fetchone()
            if row is not None and row[0] not in (owner, None) and now - (row[1] or 0) < lease:
                cur.execute("ROLLBACK")
                return False
            cur.execute("INSERT INTO live_state (id, owner, heartbeat) VALUES (1, ?, ?) ON CONFLICT (id) DO UPDATE "
                        "SET owner = excluded.owner, heartbeat = excluded.heartbeat", (owner, now))
            if row is not None and row[0] != owner:  # a takeover: the previous worker's numbers are not ours
                cur.execute("UPDATE live_state SET snapshot = ? WHERE id = 1",
                            (json.dumps({"status": "starting", "owner": owner}),))
            cur.execute("COMMIT")
            return True
        except BaseException:
            cur.execute("ROLLBACK")
            raise

    def beat(self, owner: str, snapshot: dict, now: float | None = None) -> bool:
        """Heartbeat + status snapshot (images kept). False if owner lost the lease."""
        now = time.time() if now is None else now
        return self.con.execute("UPDATE live_state SET heartbeat = ?, snapshot = ? WHERE id = 1 AND owner = ?",
                                (now, json.dumps(snapshot), owner)).rowcount == 1

    def pulse(self, owner: str, now: float | None = None) -> bool:
        """Heartbeat only (snapshot and images untouched), and only while owner still holds the lease: False if it does not."""
        now = time.time() if now is None else now
        return self.con.execute("UPDATE live_state SET heartbeat = ? WHERE id = 1 AND owner = ?", (now, owner)).rowcount == 1

    def publish(self, owner: str, snapshot: dict, overlay: bytes, raw: bytes | None = None,
                now: float | None = None) -> bool:
        """Heartbeat + snapshot + the frame's overlay image (and a raw frame if given). False if owner lost the
        lease."""
        now = time.time() if now is None else now
        if raw is None:
            q = ("UPDATE live_state SET heartbeat = ?, snapshot = ?, overlay = ? WHERE id = 1 AND owner = ?",
                 (now, json.dumps(snapshot), overlay, owner))
        else:
            q = ("UPDATE live_state SET heartbeat = ?, snapshot = ?, overlay = ?, raw = ?, raw_ts = ? WHERE id = 1 "
                 "AND owner = ?", (now, json.dumps(snapshot), overlay, raw, now, owner))
        return self.con.execute(*q).rowcount == 1

    def release(self, owner: str):
        """A worker that stops cleanly: the lease is free at once (the page shows it down, a standby takes over)."""
        self.con.execute("UPDATE live_state SET heartbeat = 0 WHERE id = 1 AND owner = ?", (owner,))

    def state(self, images: bool = True) -> dict:
        """{"owner", "heartbeat", "snapshot" (dict), "overlay", "raw" (JPEG bytes or None), "raw_ts"}; {} if no worker
        ever ran."""
        cols = "owner, heartbeat, snapshot, raw_ts" + (", overlay, raw" if images else "")
        row = self.con.execute(f"SELECT {cols} FROM live_state WHERE id = 1").fetchone()
        if row is None:
            return {}
        out = {"owner": row[0], "heartbeat": row[1] or 0.0, "snapshot": json.loads(row[2]) if row[2] else {},
               "raw_ts": row[3]}
        if images:
            out["overlay"], out["raw"] = row[4], row[5]
        return out

    # page -> worker requests
    def command(self, action: str) -> int:
        return self.con.execute("INSERT INTO live_commands (action, ts) VALUES (?, ?)",
                                (action, time.time())).lastrowid

    def commands(self, after: int) -> list[tuple[int, str]]:
        """Requests newer than id `after`, oldest first; marked done."""
        rows = self.con.execute("SELECT id, action FROM live_commands WHERE id > ? ORDER BY id", (after,)).fetchall()
        if rows:
            self.con.execute("UPDATE live_commands SET done = ? WHERE id > ? AND id <= ?",
                             (time.time(), after, rows[-1][0]))
        return rows

    # worker options outside the shared settings: the Streamlit page rewrites live_config whole, and these are set
    # from the FastAPI side only (e.g. "model_variant"), so they live in their own table it never touches
    def option(self, key: str, default: str = "") -> str:
        row = self.con.execute("SELECT value FROM live_options WHERE key = ?", (key,)).fetchone()
        return default if row is None else row[0]

    def set_option(self, key: str, value: str):
        self.con.execute("INSERT INTO live_options (key, value, ts) VALUES (?, ?, ?) ON CONFLICT (key) DO UPDATE SET "
                         "value = excluded.value, ts = excluded.ts", (key, value, time.time()))

    def last_command(self) -> int:
        return self.con.execute("SELECT COALESCE(MAX(id), 0) FROM live_commands").fetchone()[0]


# --- Mode 5 settings: one builder for every UI ---------------------------------------------------------------------
# The (removed) Streamlit page used to store {"enabled", "ui": its widget values, "run": what they
# mean for the worker}. build_settings() computes the same "run" from the same widget values, so any other UI (the
# FastAPI backend) writes settings identical to the page's: the page, re-seeded from them, recomputes the same dict
# and writes nothing, and the worker never restarts its session for a mere re-save.
# (the page and its parity test are gone: tests/test_api.py pins what the API stores).
CHOICES = {"🤖 Auto (layout analyser)": "auto", "YOLOv8 Axis-Aligned (Baseline)": "baseline",
           "YOLO-OBB (Oriented / Advanced)": "obb"}
STREAMS = ["Main stream (101, full resolution)", "Sub-stream (102, lower latency)", "Custom URL"]
SLOT_MODES = ["Dynamic (Auto)", "Fixed (Pre-defined)"]
SIZES = ["Whole frame 1280 px", "Whole frame 1920 px", "SAHI 640 px tiles"]
MATCH_RULE_NAMES = ["ground", "center", "iou"]  # fixed_slots.MATCH_RULES, in its order
CAM_PREFIX, COCO_PREFIX = "📷 Camera model · ", "🚗 COCO vehicles · "
BAY_SRC = "🅿️ Bay model · PKLot (Model selector above)"
CUT_DEFAULTS = {"top": 0, "bottom": 0, "left": 0, "right": 40}  # % of the frame; right: Front Parking 02's road
UI_DEFAULTS = {"conf5": 0.30, "camera5": "Camera 1", "url5": "", "match5": "ground", "matchiou5": 0.15, "save5": 0,
               "hold5": 15, "exit5": 30, "exitn5": 15, "occl5": 0.6, "hmax5": 30, "lhold5": 20, "learn5": True,
               "enabled5": False}


def bay_weights(root) -> dict[str, Path]:
    """The PKLot bay models of the page's Model selector."""
    root = Path(root)
    fair = root / "models" / "parking_yolo_obb.pt"
    return {"baseline": root / "models" / "parking_yolo.pt",
            "obb": fair if fair.exists() else root / "models" / "parking_yolo_obb_holdout-PUCPR.pt"}


def detector_options(root, camera: str) -> tuple[list[str], dict, dict]:
    """The page's Mode 5 detector list for this camera name -> (labels in order, {label: weights} of the single
    models, {label: (camera model, COCO model)} of the hybrid). Order: stock COCO (yolov8s, yolov8n), this camera's own
    fine-tune, the hybrid of it + COCO, other cameras' models, the bay model."""
    from carparking.dataset import slug

    root = Path(root)
    coco = {**{f"{CAM_PREFIX}{p.name}": p for p in sorted((root / "models").glob("cam_*.pt"))},
            **{f"{COCO_PREFIX}{n}": root / n for n in ("yolov8s.pt", "yolov8n.pt") if (root / n).exists()}}
    own_weights = root / "models" / f"cam_{slug(camera)}.pt"
    own = f"{CAM_PREFIX}{own_weights.name}"
    coco_keys = [k for k in coco if k.startswith(COCO_PREFIX)]
    coco_weights = coco[coco_keys[0]] if coco_keys else None
    hybrid = {}
    if own in coco and coco_weights:
        hybrid[f"🔀 Hybrid Dual Detector · {own_weights.name} + {coco_weights.name}"] = (own_weights, coco_weights)
    cams = sorted((k for k in coco if k.startswith(CAM_PREFIX)), key=lambda k: k != own)
    return coco_keys + cams[:1] + list(hybrid) + cams[1:] + [BAY_SRC], coco, hybrid


_NAMES: dict = {}


def class_names(weights) -> tuple[list[str], list[str]]:
    """(vehicle classes, other classes) of one or more weights files, in model order (union for a hybrid), as the
    page's target-class lists. Loads each model once per process."""
    from carparking.yolo_engine import state_of

    kinds, others = {}, {}
    for w in weights:
        key = str(Path(w).resolve())
        if key not in _NAMES:
            from ultralytics import YOLO
            _NAMES[key] = list(YOLO(key).names.values())
        for n in _NAMES[key]:
            (kinds if state_of(n) is not None else others).setdefault(n, None)
    return list(kinds), list(others)


def _pick(ui: dict, key: str, default, options=None):
    """A widget's value as the page computes it (init5): the stored one, else the default; a value no longer among
    options falls back to the default (lists: filtered)."""
    v = ui.get(key, default)
    if options is not None:
        if isinstance(v, list):
            return [x for x in v if x in options]
        if v not in options:
            return default
    return v


def build_settings(ui: dict, root, db_path, rtsp_url: str = "", fixed_path=None, names_of=class_names) -> dict:
    """The Mode 5 settings {"enabled", "ui", "run"} for these widget values (the page's keys: stream5, camera5, vsrc5,
    conf5, hold5, ...; missing = the page's default), exactly as the Streamlit page stores them. The parking-area cuts
    and polygon come from the camera's saved settings (HistoryDB.setting), the fixed slots from fixed_path
    (FixedSlotStore), as on the page. names_of(weights) -> (vehicle classes, other classes)."""
    from carparking.fixed_slots import FixedSlotStore
    from carparking.monitor import parse_roi

    root = Path(root)
    choice = CHOICES[_pick(ui, "model5", list(CHOICES)[0], list(CHOICES))]
    conf = _pick(ui, "conf5", UI_DEFAULTS["conf5"])
    stream = _pick(ui, "stream5", STREAMS[0], STREAMS)
    url = _pick(ui, "url5", "") if stream == "Custom URL" else stream_url(rtsp_url, "sub" if stream.startswith("Sub")
                                                                            else "main")
    camera = str(_pick(ui, "camera5", "Camera 1")).strip() or "Camera 1"
    detectors, coco, hybrid = detector_options(root, camera)
    vsrc = _pick(ui, "vsrc5", detectors[0], detectors)
    fixed_mode = _pick(ui, "slotmode5", SLOT_MODES[0], SLOT_MODES) == SLOT_MODES[1]
    if vsrc in hybrid:
        own_weights, coco_weights = hybrid[vsrc]
        detector_id = f"hybrid_{own_weights.stem}+{coco_weights.stem}"
    else:
        detector_id = coco[vsrc].stem if vsrc in coco else f"bay_{choice}"
    db_detector = detector_id + (" | fixed" if fixed_mode else "")
    m5 = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "roi": None, "whole": None, "classes": None,
          "extra": [], "match": "ground", "match_iou": 0.15}
    if fixed_mode:
        m5["match"] = _pick(ui, "match5", "ground", MATCH_RULE_NAMES)
        if m5["match"] == "iou":
            m5["match_iou"] = _pick(ui, "matchiou5", 0.15)
    if vsrc != BAY_SRC:
        big = vsrc in coco and coco[vsrc].name == "yolov8s.pt"
        size = _pick(ui, "size5", SIZES[int(big)], SIZES)
        m5["whole"] = {SIZES[0]: 1280, SIZES[1]: 1920}.get(size)
        kinds, others = names_of(list(hybrid[vsrc]) if vsrc in hybrid else [coco[vsrc]])
        m5["classes"] = _pick(ui, "classes5", kinds, kinds)
        if others:
            m5["extra"] = _pick(ui, "extra5", [n for n in ("person", "dog", "cat") if n in others]
                                if vsrc in hybrid else [], others)
    db = HistoryDB(db_path, camera, detector=db_detector)
    try:
        try:
            roi = parse_roi(db.setting("roi"))
            m5["roi"] = None if roi is None else roi.tolist()
        except ValueError:  # the page shows the error and keeps no polygon
            pass
        for side, default in CUT_DEFAULTS.items():
            saved = db.setting(f"cut_{side}")
            m5[side] = (int(saved) if saved.isdigit() else default) / 100
    finally:
        db.close()
    m5["save_every"] = _pick(ui, "save5", 0)
    for key, k in (("hold", "hold5"), ("exit_hold", "exit5"), ("exit_frames", "exitn5"), ("occlusion", "occl5"),
                   ("hidden_max", "hmax5")):
        m5[key] = _pick(ui, k, UI_DEFAULTS[k])
    m5["learn_hold"], m5["learn"] = 20, False
    if not fixed_mode:
        m5["learn_hold"], m5["learn"] = _pick(ui, "lhold5", 20), _pick(ui, "learn5", True)
    m5["absent_free"] = _pick(ui, "absent5", vsrc != BAY_SRC)
    fixed = None
    if fixed_mode:
        try:
            fixed = FixedSlotStore(fixed_path).load(camera) if fixed_path is not None else []
        except ValueError:  # hand-edited file with a bad slot: the page shows the error, no slots
            fixed = []
    rel = lambda p: Path(p).resolve().relative_to(root.resolve()).as_posix()  # noqa: E731
    spec = ({"kind": "hybrid", "weights": [rel(p) for p in hybrid[vsrc]], "whole": m5["whole"]} if vsrc in hybrid
            else {"kind": "coco", "weights": [rel(coco[vsrc])], "whole": m5["whole"]} if vsrc in coco
            else {"kind": "bay", "choice": choice, "bay": {k: rel(p) for k, p in bay_weights(root).items()}})
    run = {"camera": camera, "stream": "custom" if stream == "Custom URL" else "sub" if stream.startswith("Sub")
           else "main", "custom_url": url if stream == "Custom URL" else "", "db_detector": db_detector, "conf": conf,
           "detector": spec, "fixed_mode": fixed_mode, "fixed": fixed if fixed_mode else None, "m5": m5}
    return {"enabled": bool(_pick(ui, "enabled5", False)), "ui": dict(ui), "run": run}


def heartbeat_age(state: dict, now: float | None = None) -> float | None:
    """Seconds since the worker's last heartbeat, None if no worker ever ran."""
    if not state:
        return None
    return (time.time() if now is None else now) - state["heartbeat"]


# --- model variant: the stock COCO detector's weights, picked for the hardware ------------------------------------
# "default" = the page's choice as it is (yolov8s / yolov8n: ~1.5 s per 1920 px frame on a 6-core CPU); "yolo11m" =
# Ultralytics YOLO11 medium in its place (more accurate on small, far and half-hidden cars; ~2.4x the compute of
# yolov8s, so a GPU or a big CPU). "auto" = picked at worker start from the hardware. Only the stock COCO detector
# is swapped (same classes, same kind of boxes): the slot map and the history scope stay, so Streamlit, the history
# and the learned slots continue. Camera fine-tunes, the hybrid and the bay model are never swapped.
MODEL_VARIANTS = ("auto", "default", "yolo11m")
YOLO11M = "yolo11m.pt"


def hardware() -> dict:
    """{"cuda", "gpu" (name or None), "cores", "ram_gb"} of this machine (container: what it is given)."""
    out = {"cuda": False, "gpu": None, "cores": os.cpu_count() or 1, "ram_gb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            out["cuda"], out["gpu"] = True, torch.cuda.get_device_name(0)
    except Exception:  # no torch / a broken driver: CPU
        pass
    try:
        import psutil
        out["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    return out


def auto_variant(hw: dict, min_cores: int | None = None, min_ram_gb: float | None = None) -> tuple[str, str]:
    """The model the hardware can run in time -> (variant, reason). yolo11m with a CUDA GPU, or on a CPU with >=
    min_cores logical cores and >= min_ram_gb RAM (env CARPARK_YOLO11M_MIN_CORES / _MIN_RAM_GB, default 16 / 16);
    else the light default."""
    min_cores = int(os.environ.get("CARPARK_YOLO11M_MIN_CORES", 16)) if min_cores is None else min_cores
    min_ram_gb = float(os.environ.get("CARPARK_YOLO11M_MIN_RAM_GB", 16)) if min_ram_gb is None else min_ram_gb
    if hw.get("cuda"):
        return "yolo11m", f"CUDA GPU ({hw.get('gpu')})"
    spec = f"{hw.get('cores')} CPU cores, {hw.get('ram_gb')} GB RAM"
    if hw.get("cores", 0) >= min_cores and hw.get("ram_gb", 0) >= min_ram_gb:
        return "yolo11m", f"no GPU, but {spec}"
    return "default", f"no CUDA GPU, {spec} (yolo11m needs a GPU or >= {min_cores} cores + {min_ram_gb:g} GB)"


def yolo11m_weights(models_dir) -> Path:
    """models_dir/yolo11m.pt, downloaded from the Ultralytics release on first use (it is kept: models_dir is on
    the data volume under compose, so a rebuild doesn't download it again)."""
    path = Path(models_dir) / YOLO11M
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        from ultralytics.utils.downloads import attempt_download_asset
        got = Path(attempt_download_asset(str(path)))
        if not got.exists():
            raise FileNotFoundError(f"{YOLO11M} could not be downloaded to {path.parent}")
        path = got
    return path


def detector_label(det) -> str:
    """Name of the running detector(s) for the status line."""
    if det.task == "hybrid":  # not isinstance: a cached detector may predate a module reload
        return "🔀 Hybrid: " + " + ".join(m.weights.name for m in det.members) + " running together"
    return f"detector {det.weights.name}"


NEAR_IMGSZ = 640  # whole-frame detectors above this size also run a near pass at it (YoloDetector near_imgsz)


def build_detector(spec: dict, frame, root, cache: dict):
    """The Mode 5 detector from the page's choice -> (detector, sliced, status text, auto?). spec: {"kind": "coco" |
    "hybrid" | "bay", "weights": [paths relative to root], "whole": whole-frame imgsz or None, "choice" and "bay"
    (model -> path) for the bay model}. cache: {(path, whole): YoloDetector}, kept across rebuilds."""
    from carparking.view import badge, route
    from carparking.yolo_engine import HybridDetector, YoloDetector

    root = Path(root)

    def yolo(rel, whole):
        near = NEAR_IMGSZ if whole and whole > NEAR_IMGSZ else None  # cars close to the lens (YoloDetector doc)
        key = (rel, whole, near)
        if key not in cache:
            cache[key] = YoloDetector(root / rel, whole_imgsz=whole, near_imgsz=near)
        return cache[key]

    whole = spec.get("whole")
    if spec["kind"] == "hybrid":
        det = HybridDetector([yolo(spec["weights"][0], whole), yolo(spec["weights"][1], whole)], ["cam", "coco"])
        return det, False, f"🔀 Vehicles + objects from both models, merged: {det.describe()}", True
    if spec["kind"] == "coco":
        det = yolo(spec["weights"][0], whole)
        kinds = "/".join(det.names[i] for i in det.classes)
        return (det, False, f"🚗 Vehicles: {Path(spec['weights'][0]).name}, classes {det.classes} ({kinds}) "
                            f"[{det.describe()}]", True)
    bay = {k: p for k, p in spec["bay"].items() if (root / p).exists()}
    r = route(frame, spec.get("choice", "auto"), tuple(bay))
    det = yolo(bay[r["model"]], None)
    return det, r["sliced"], badge(r, det.describe(r["sliced"])), r["auto"]


def build_monitor(run: dict, det, sliced: bool, db: HistoryDB) -> ParkingMonitor:
    """ParkingMonitor with the page's Mode 5 settings (run["m5"]), as the page's own loop used to build it."""
    m5 = run["m5"]
    return ParkingMonitor(det, db, ParkingArea.from_settings(m5),
                          targets=None if m5["classes"] is None else m5["classes"] + m5["extra"],
                          sliced=sliced, conf=run["conf"], hold=m5["hold"], learn_hold=m5["learn_hold"],
                          absent_free=m5["absent_free"], learn=m5["learn"], exit_hold=m5["exit_hold"],
                          occlusion_cover=m5["occlusion"], hidden_max=m5["hidden_max"],
                          exit_frames=m5.get("exit_frames", 0), fixed=run.get("fixed"), match_rule=m5["match"], match_iou=m5["match_iou"],
                          **({"auto": run["auto"]} if run.get("auto") is not None else {}),
                          **({"split": run["split"]} if run.get("fixed_mode") and "split" in run else {}))


def _box_key(d) -> tuple:
    """A detection's box as whole pixels: how FixedSlotTracker.rejected names the boxes it dropped."""
    return tuple(round(float(v)) for v in d["box"])


def _rejected_boxes(tracker) -> dict:
    """{box in pixels: reason} of the detections the non-vehicle filter dropped in the last frame ({} while it is off)."""
    if not getattr(tracker, "obstacle_filter", 0):
        return {}
    return {tuple(r["box"]): r["reason"] for r in getattr(tracker, "rejected", ())}


def _kept(snapshot: dict) -> list:
    """The snapshot's detections without the ones the non-vehicle filter dropped (the metric view measures vehicles)."""
    return [d for d in snapshot.get("detections") or [] if not d.get("rejected")]


def _dashed_poly(img, poly, colour, dash: int = 14, thickness: int = 2):
    """A dashed closed polygon (cv2 has no dashed lines)."""
    pts = np.array(poly, np.float32)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n = max(1, int(np.linalg.norm(b - a) // dash))
        for i in range(0, n, 2):
            p, q = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
            cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), colour, thickness, cv2.LINE_AA)


def render_overlay(frame, res, tracker):
    """The Mode 5 picture: parking area (outside dimmed, cyan edge), detections (yellow vehicles, magenta others, with
    class, confidence and source), the slot map (FREE green-filled, BUSY red, UNKNOWN grey) and its HIDDEN / phantom /
    DUPLICATE / fixed-slot-name labels. BGR, full frame size."""
    mask, dets, others = res.mask, res.vehicles, res.others
    img = frame.copy()
    img[mask == 0] = (img[mask == 0] * 0.35).astype(np.uint8)
    edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, edges, -1, (255, 255, 0), 3)
    dropped = _rejected_boxes(tracker)  # non-vehicle filter on: the boxes it did not let through are grey and dashed, with the reason
    for d in [d for d in dets if _box_key(d) in dropped]:
        _dashed_poly(img, d["poly"], (150, 150, 150))
        cv2.putText(img, f"ignored: {dropped[_box_key(d)]}", (d["box"][0], max(d["box"][1] - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (150, 150, 150), 2, cv2.LINE_AA)
    dets = [d for d in dets if _box_key(d) not in dropped]
    for group, colour in ((dets, (0, 220, 255)), (others, (255, 0, 255))):
        if group:  # cv2.polylines rejects an empty list
            cv2.polylines(img, [np.array(d["poly"], np.int32) for d in group], True, colour, 2)
        for d in group:
            src = f" [{d['src']}]" if d.get("src") else ""
            cv2.putText(img, f"{d['label']} {d['conf']:.2f}{src}", (d["box"][0], max(d["box"][1] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    bays, states = tracker.view(frame.shape)
    labels = [(k, "HIDDEN (kept BUSY)" if states[k] == 1 else "HIDDEN", (0, 140, 255)) for k in tracker.hidden]
    labels += [(k, "NO ROOM (phantom)", (160, 160, 160)) for k in tracker.phantoms]
    labels += [(k, "DUPLICATE" + (f" of {o}" if o is not None else ""), (160, 160, 160))
               for k, o in getattr(tracker, "covered", {}).items()]
    for k, text, colour in labels:
        x, y = bays[k].min(0).astype(int)
        cv2.putText(img, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    img = draw(img, bays, states, "RTSP", labels=True)  # every slot, FREE ones green-filled
    for k, name in getattr(tracker, "names", {}).items():  # fixed slots: their names (draw shows the IDs)
        if name != str(k):
            x, y = bays[k].min(0).astype(int)
            cv2.putText(img, name, (x + 8, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    return img


class Pulse:
    """The worker's heartbeat from a thread of its own. step() only writes one between frames, so one slow inference (a CPU
    without a GPU: 5-60 s a frame, longer while the host is busy) made the page say "worker down" and let a second worker take the
    lease. The thread has its own connection to the store and writes the heartbeat alone, every `every` seconds, while the
    worker holds the lease (`active`) and its loop is making progress: the loop stamps `progress` at the start and end of every
    step, and after `max_stall` seconds without a stamp the pulse stops, so a hung worker (not a slow one) still shows as down."""

    def __init__(self, path, owner: str, every: float = PULSE_EVERY, max_stall: float = PULSE_STALL, clock=time.time):
        self.owner, self.every, self.max_stall, self.clock = owner, every, max_stall, clock
        self.store, self.active, self.progress = LiveStore(path), False, clock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="worker-pulse", daemon=True)
        self._thread.start()

    def tick(self) -> bool:
        """One heartbeat if one is due -> whether it was written."""
        now = self.clock()
        return self.active and now - self.progress <= self.max_stall and self.store.pulse(self.owner, now)

    def _run(self):
        while not self._stop.wait(self.every):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a locked file: the next beat tries again; the pulse never takes the worker down
                pass

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self.store.close()


class Worker:
    """The Mode 5 loop without a page. Each step(): hold the lease (else standby), read the settings (a new version
    restarts the session: camera, detector, monitor; a fixed map edit is applied in place), run the page's
    requests, and process the newest frame: ParkingMonitor (entries / exits into the history), raw frames for the
    dataset, the overlay and a status snapshot for the page.

    store: LiveStore. root: project folder (weights paths in the settings are relative to it). db_path: the history
    database. raw_dir: dataset_raw. default_url: the camera for stream "main" / "sub" ($CARPARK_RTSP_URL).
    reader_factory(url) -> object with latest() -> (frame, seq, capture time), .status, stop() (LatestFrameReader).
    detector_factory(spec, frame) -> (detector, sliced, text, auto) (default build_detector).
    Model variant (option "model_variant": auto / default / yolo11m, set from the FastAPI side): read every step; a
    change rebuilds the detector and the monitor from the next frame (the slot map is reloaded from the database), no
    restart. hw: the hardware for "auto" (default: hardware()); weights_for(variant) -> the weights path of a variant
    (default: yolo11m downloaded into models_dir on first use).
    geometry: Phase 2, None unless CARPARK_GEOMETRY_ENABLED (scripts/worker.py): a carparking.geometry_live.GeometryFeed.
    With it, a camera that has an ACTIVE calibration also publishes "geometry" (footprints, distances, bay capacity) in
    its snapshot; without an active calibration, or on any error in it, nothing changes (never a failed frame).
    health_factory() -> carparking.stream_health.StreamHealth: one per session. A stale (no new frame) or frozen (the same picture)
    stream is not processed: every slot shows UNKNOWN, the tracker and the open sessions stay as they were, and the snapshot's
    "stream" says why (item 105).
    pulse_every: > 0 starts a Pulse thread that keeps the heartbeat alive during a slow frame (scripts/worker.py: CARPARK_PULSE_S).
    Off by default so a test worker starts no thread."""

    def __init__(self, store: LiveStore, root, db_path, tz=None, raw_dir=None, default_url: str = "",
                 owner: str | None = None, reader_factory=LatestFrameReader, detector_factory=None,
                 lease: float = LEASE, raw_every: float = 5.0, clock=time.time, models_dir=None, hw: dict | None = None,
                 weights_for=None, geometry=None, health_factory=StreamHealth, pulse_every: float = 0.0,
                 pulse_stall: float = PULSE_STALL):
        self.store, self.root, self.db_path, self.tz = store, Path(root), db_path, tz
        self.raw_dir, self.default_url = raw_dir, default_url
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}"
        self.reader_factory, self.lease, self.raw_every, self.clock = reader_factory, lease, raw_every, clock
        self._detectors: dict = {}
        self.detector_factory = detector_factory or (lambda spec, frame: build_detector(spec, frame, self.root,
                                                                                         self._detectors))
        self.code = code_version(self.root)
        self.hw = hardware() if hw is None else hw
        self.auto_choice, self.auto_reason = auto_variant(self.hw)  # what "auto" means on this machine
        models_dir = Path(models_dir) if models_dir is not None else self.root / "models"
        self.weights_for = weights_for or (lambda variant: yolo11m_weights(models_dir))
        self.variant = None  # the variant the current detector was built with
        self.tuning: dict | None = None  # display thresholds applied to the current tracker (option "tuning")
        self.started = self.clock()
        self.cmd_after = store.last_command()  # requests made while no worker ran are not replayed
        self.ver, self.run, self.exit = None, None, False
        self.reader = self.db = self.mon = self.det = None
        self.holding = False
        self.geometry, self._geometry_error = geometry, ""
        self.health_factory, self.health, self._stream_logged = health_factory, None, ACTIVE
        self.pulse = Pulse(store.path, self.owner, pulse_every, pulse_stall, self.clock) if pulse_every > 0 else None
        # (calibration version, slot count, operator input as stored) of the auto grid the monitor runs on, else None: any change
        # of it rebuilds the monitor
        self.auto_key: tuple | None = None
        self._grid_raw: str | None = None  # option "auto_grid" as last read, and what it means: (expected bays, pinned bays)
        self.grid_cfg: tuple = (None, [])
        self._mismatch_logged = False  # (see _watch_sessions)
        self.split, self.split_wanted = None, True  # sub-bays on (the monitor's), and what the option "slot_split" says now
        self.auto_surface: list | None = None  # the measured ground (image fractions) the vehicle-driven bays stand on
        self.auto_info: dict = {}  # the grid's make-up (bays by source, expected count) for the status

    # session = one camera + settings version
    def _url(self, run: dict) -> str:
        return run.get("custom_url", "") if run.get("stream") == "custom" else stream_url(self.default_url,
                                                                                            run.get("stream", "main"))

    def stop_session(self):
        if self.reader is not None:
            self.reader.stop()
        self.reader = self.mon = self.det = None
        self.run = self.auto_key = self.split = None

    def _start_session(self, run: dict):
        self.stop_session()
        if self.db is not None:
            self.db.close()
        self.run, self.url = run, self._url(run)
        self.db = HistoryDB(self.db_path, run["camera"], self.tz, detector=run["db_detector"])
        self.reader = self.reader_factory(self.url)
        self.health, self._stream_logged = self.health_factory(), ACTIVE
        self.last_seq, self.n, self.t0 = 0, 0, self.clock()
        self.last_save = self.last_raw = 0.0
        self.saved, self.last_saved, self.logged, self.snap = 0, "", 0, False
        self.det_text, self.det_auto, self.det_error = "", True, ""

    def _apply(self, cfg: dict | None, ver: int):
        self.ver = ver
        run = (cfg or {}).get("run")
        if not (cfg or {}).get("enabled") or not run or not self._url(run) or (run.get("fixed_mode") and
                                                                               not run.get("fixed")):
            self.stop_session()  # off, no camera, or fixed mode without a saved map (never learn into its scope)
            return
        if self.run is not None and run == self.run:
            return  # a new version with the same run settings (another UI re-saved, the on switch unchanged): no reset
        if (self.run is not None and self.mon is not None and self.mon.mode == "fixed" and run.get("fixed")
                and {**run, "fixed": None} == {**self.run, "fixed": None}):
            self.mon.update_fixed(run["fixed"], self.clock())  # only the fixed map changed: keep the slot states
            self.run = run
            return
        self._start_session(run)

    def _tune(self):
        """Display thresholds (option "tuning", JSON, set from the dashboard's Settings; history.DISPLAY_TUNING
        defaults): applied to the running tracker whenever they change, also to a new monitor. Dynamic maps only
        (a fixed map has no shown-state rules)."""
        try:
            raw = json.loads(self.store.option("tuning", "{}") or "{}")
        except ValueError:
            raw = {}
        values = tuning_values(raw)
        tracker = getattr(self.mon, "tracker", None)
        if tracker is not None and hasattr(tracker, "tune") and getattr(tracker, "_tuned", None) != values:
            tracker._tuned = tracker.tune(values)
            self.tuning = values

    def _evidence(self):
        """The evidence layer (evidence.py; post-demo item 4) follows the option "evidence_filter" ("0" = off, the default; anything
        else = on), applied to the running tracker whenever it changes, also to a new monitor. Fixed maps and auto grids only (the
        learned map has no such tracker)."""
        tracker = getattr(self.mon, "tracker", None)
        if tracker is not None and hasattr(tracker, "set_evidence"):
            tracker.set_evidence(self.store.option("evidence_filter", "0") not in ("", "0"))

    def _spillage(self):
        """The spillage layer (spillage.py; post-demo item 1) follows the option "spillage" ("0" = off, the default), like _evidence().
        Fixed maps and auto grids only."""
        tracker = getattr(self.mon, "tracker", None)
        if tracker is not None and hasattr(tracker, "set_spillage"):
            tracker.set_spillage(self.store.option("spillage", "0") not in ("", "0"))

    def _blocked(self):
        """Blocked by object (blocked.py; post-demo item 5) follows the option "blocked_objects" ("0" = off, the default), like _spillage():
        the tracker's layer on or off, and the object classes join / leave the monitor's target classes (the detector must report them)."""
        on = self.store.option("blocked_objects", "0") not in ("", "0")
        tracker = getattr(self.mon, "tracker", None)
        if tracker is not None and hasattr(tracker, "set_blocked"):
            tracker.set_blocked(on)
            if hasattr(self.mon, "set_blocked"):
                self.mon.set_blocked(on and bool(getattr(tracker, "blocked", 0)), tracker.blocked_cfg.labels)

    def _ground(self, frame, now):
        """The calibrated camera for the spillage ground metric (spillage.GroundMeasure): only while the layer is on and the camera has an
        active calibration; otherwise None and the tracker measures in pixels, exactly as before."""
        tracker = getattr(self.mon, "tracker", None)
        if tracker is None or not hasattr(tracker, "ground_cam"):
            return
        cam = None
        mode = (self.store.option("spillage_ground", "auto") or "auto").strip().lower()  # "off" = pixels only, "union" = ground and pixels
        if mode != "off" and getattr(tracker, "spillage", 0) and self.geometry is not None and hasattr(self.geometry, "camera"):
            cam = self.geometry.camera([frame.shape[1], frame.shape[0]], now)
        tracker.ground_cam, tracker.ground_mode = cam, ("union" if mode == "union" else "auto")

    def _obstacle_filter(self):
        """The non-vehicle filter (obstacle_filter.py; post-demo item 2) follows the option "obstacle_filter" ("0" = off, the default),
        like _evidence(). Fixed maps and auto grids only."""
        tracker = getattr(self.mon, "tracker", None)
        if tracker is not None and hasattr(tracker, "set_obstacle_filter"):
            tracker.set_obstacle_filter(self.store.option("obstacle_filter", "0") not in ("", "0"))

    def _wanted_variant(self) -> tuple[str, str]:
        """(the stored setting, the variant it means here)."""
        setting = self.store.option("model_variant", "auto")
        setting = setting if setting in MODEL_VARIANTS else "auto"
        return setting, self.auto_choice if setting == "auto" else setting

    def _spec(self, variant: str) -> dict:
        """The page's detector spec, with yolo11m in place of a stock COCO model if that variant is wanted."""
        spec = self.run["detector"]
        if variant != "yolo11m" or spec["kind"] != "coco" or Path(spec["weights"][0]).name.startswith("cam_"):
            return spec
        return {**spec, "weights": [str(self.weights_for(variant))]}

    def _model(self) -> dict:
        setting, wanted = self._wanted_variant()
        spec = self.run["detector"] if self.run else {}
        swappable = spec.get("kind") == "coco" and not Path(spec.get("weights", ["x"])[0]).name.startswith("cam_")
        return {"setting": setting, "active": self.variant or wanted, "auto": setting == "auto",
                "auto_choice": self.auto_choice, "reason": self.auto_reason, "swappable": swappable,
                "weights": self.det.weights.name if self.det is not None and hasattr(self.det, "weights") else None,
                "hardware": self.hw}

    def _snapshot(self, status: str, **extra) -> dict:
        return {"status": status, "enabled": self.run is not None, "cfg_ver": self.ver, "code": self.code,
                "started": self.started, "owner": self.owner, "model": self._model(), **extra}

    def step(self) -> float:
        """One pass -> seconds to sleep before the next."""
        now = self.clock()
        if self.pulse is not None:
            self.pulse.progress = now
        if not self.store.claim(self.owner, self.lease, now):
            if self.holding or self.reader is not None:
                self.stop_session()
            self.holding = False
            if self.pulse is not None:
                self.pulse.active = False
            return 5.0  # standby: another worker holds the lease
        self.holding = True
        if self.pulse is not None:
            self.pulse.active = True
        cfg, ver = self.store.config()
        if ver != self.ver:
            self._apply(cfg, ver)
        for _, action in self.store.commands(self.cmd_after):
            self.cmd_after = max(self.cmd_after, _)
            if action == "restart":
                self.exit = True
            elif action == "reset" and (cfg or {}).get("run"):
                db = self.db or HistoryDB(self.db_path, cfg["run"]["camera"], self.tz,
                                          detector=cfg["run"]["db_detector"])
                db.reset(now)
                self.mon = None  # rebuilt on the next frame from the (now empty) map
            elif action == "snapshot":
                self.snap = True
        if self.exit:
            return 0.0
        if self.reader is None:
            self.store.beat(self.owner, self._snapshot("idle: monitoring is off"), now)
            return 1.0
        frame, seq, t = self.reader.latest()
        stream = self.health.observe(frame, seq, t, now)
        if self.mon is not None:
            self.mon.stream = stream  # a stale / frozen stream shows every slot UNKNOWN (ParkingMonitor.state)
        self._watch_stream(stream, now)
        if frame is None or seq == self.last_seq or stream != ACTIVE:  # nothing new, or nothing true: report, never block
            age = None if frame is None else now - t
            self.last_seq = seq if frame is not None else self.last_seq  # (a frozen picture is not processed later either)
            self.store.beat(self.owner, self._snapshot(self.reader.status, url=mask_url(self.url), frame_age=age, waiting=True,
                                                       **self._stats(), **self._unknown_view()), now)
            return 0.1
        self.last_seq = seq
        if self.raw_dir is not None and (self.snap or (self.run["m5"].get("save_every") and
                                                       t - self.last_save >= self.run["m5"]["save_every"])):
            path = save_frame(frame, self.raw_dir, self.db.camera, t, self.tz)  # raw full-res, no overlay
            self.last_save, self.saved, self.last_saved, self.snap = t, self.saved + 1, path.name, False
        _, wanted = self._wanted_variant()
        if self.mon is not None and wanted != self.variant:
            self.mon = None  # the model variant changed: a new detector and monitor from this frame, no restart
        self.split_wanted = self.store.option("slot_split", "1") != "0"  # sub-bays for vehicles sharing a slot (fixed maps, auto grids)
        auto = self._auto_slots(frame, now)
        if self.mon is not None and self.split is not None and (self.split_wanted and auto is not None) != self.split:
            self.mon = None  # (sub-bays belong to the auto grid: on a drawn Fixed map the option changes nothing)
        if self.mon is not None and (auto and auto[0]) != self.auto_key:
            self.mon = None  # the auto grid appeared, changed (new calibration) or went (stale, not HIGH): a new monitor
        if self.mon is None:  # the detector is picked on the first frame (the bay model routes on it)
            # a heartbeat first: loading (or, the first time, downloading yolo11m.pt) can take a while
            self.store.beat(self.owner, self._snapshot(f"loading the {wanted} model"), now)
            self.variant = wanted
            self.det, sliced, self.det_text, self.det_auto = self.detector_factory(self._spec(wanted), frame)
            run = self._auto_run(auto)
            self.mon = build_monitor(run, self.det, sliced, self.db)
            self.mon.start(frame.shape)
            if self.mon.ignored:  # learned before the parking-area polygon was set: not drawn, tracked or logged
                self.det_text += f" · {self.mon.ignored} saved slot(s) outside the parking area ignored"
        self._tune()
        self._evidence()
        self._spillage()
        self._blocked()
        self._ground(frame, now)
        self._obstacle_filter()
        res = self.mon.process(frame, t)  # capture time, not the time inference finished
        self.logged += sum(e["logged"] for e in res.events)
        self._watch_sessions()
        self.n += 1
        raw = None
        # the React page draws every layer itself (Slots, Boxes, Area, Metrics: the viewer's toggles), so its frame feed is the
        # clean frame, published with every overlay image (option "clean_frames", "0" = off: the baked overlay, as it was)
        clean = self.store.option("clean_frames", "1") != "0"
        if clean or self.snap or now - self.last_raw >= self.raw_every:  # (and for the fixed slot editor on the page)
            raw, self.last_raw = _jpeg(frame), now
        snapshot = self._snapshot(self.reader.status, url=mask_url(self.url), frame_ts=t, frame_age=now - t,
                                  waiting=False, shape=[frame.shape[1], frame.shape[0]], seq=self.n, clean_frames=clean,
                                  stream=self.health.info(now), **self._stats(res), **self._vectors(frame, res))
        if self.geometry is not None:
            metric = self._metrics(snapshot, now, frame)
            if metric is not None:
                snapshot["geometry"] = metric
        self.store.publish(self.owner, snapshot, _jpeg(render_overlay(frame, res, self.mon.tracker)), raw, now)
        return 0.0

    def _watch_stream(self, state: str, now: float):
        """One log line when the stream turns stale or frozen, one when it is live again (never one per frame)."""
        if state == self._stream_logged:
            return
        info = self.health.info(now)
        if state == ACTIVE:
            print(f"live: camera {self.db.camera}: the stream is live again", flush=True)
        else:
            print(f"live: camera {self.db.camera}: stream {state} (last frame {info['age_s']:.0f} s ago, same picture for "
                  f"{info['same_s']:.0f} s): slots shown UNKNOWN, open sessions kept", flush=True)
        self._stream_logged = state

    def _unknown_view(self) -> dict:
        """While the stream is not active: the slots as the monitor shows them (all UNKNOWN), no detections, and the stream's health.
        {} when the stream is fine and nothing is waiting on a first monitor."""
        out = {"stream": self.health.info(self.clock())}
        if self.mon is not None and self.mon.stream != ACTIVE and self.run is not None:
            m5 = self.run["m5"]
            out.update(slots=self.mon.state()["slots"], detections=[],
                       area={k: m5.get(k, 0.0) for k in ("top", "bottom", "left", "right")} | {"roi": m5.get("roi")})
        return out

    def _auto_slots(self, frame, now: float) -> tuple[tuple, list[dict]] | None:
        """Dynamic mode with a HIGH-confidence active calibration (Phase 2): the packed bays of the whole measured surface
        as the slot map, so every bay exists as FREE from the start and a vehicle only flips it BUSY. -> (key, slots) or
        None (fixed mode, no feed, not HIGH, option "auto_slots" = 0, or any error: the learned map as always)."""
        feed = getattr(self.geometry, "auto_slots", None)
        if (feed is None or self.run is None or self.run.get("fixed_mode")
                or self.store.option("auto_slots", "1") == "0"):
            return None
        raw = self.store.option("auto_grid", "")
        if raw != self._grid_raw:  # the operator's input (PUT /api/auto-pins): the lot's bay count and pinned bays
            self._grid_raw = raw
            try:
                cfg = json.loads(raw) if raw else {}
                self.grid_cfg = (clean_expected(cfg.get("expected")), validate_slots(cfg.get("pins") or []))
            except (ValueError, AttributeError, TypeError):
                self.grid_cfg = (None, [])  # a bad value is the grid without it, never a lost frame
        expected, pins = self.grid_cfg
        try:
            got = feed([frame.shape[1], frame.shape[0]], now, expected=expected, pins=pins)
        except Exception:  # noqa: BLE001
            return None
        if got is None:
            return None
        dynamic = self.store.option("auto_dynamic", "1") != "0"  # the switch for vehicle-driven bays and footprint matching
        self.auto_surface = None
        if dynamic:
            try:
                self.auto_surface = getattr(self.geometry, "auto_surface")([frame.shape[1], frame.shape[0]], now)
            except Exception:  # noqa: BLE001 - footprint matching still works, no new bays
                self.auto_surface = None
        sources = {s["id"]: s.get("source", "measured") for s in got[1]}
        self.auto_info = {"expected": expected, "sources": sources,
                          **{k: sum(v == k for v in sources.values()) for k in ("measured", "extrapolated", "manual")}}
        return (got[0], len(got[1]), self._grid_raw, dynamic), got[1]

    def _auto_run(self, auto) -> dict:
        """self.run, or - with an auto grid - the same run as a fixed map: the grid's slots, matched on the box's foot point,
        nothing learned, and a history scope of their own ("| auto v<N>") so the grid's ids never mix with learned ones.
        Also (re)opens self.db for that scope."""
        run = self.run
        if auto is not None:
            key, slots = auto
            run = {**run, "fixed_mode": True, "fixed": slots, "db_detector": f"{run['db_detector']} | auto v{key[0]}",
                   "auto": {"surface": self.auto_surface} if key[3] else None,
                   "m5": {**run["m5"], "match": "foot", "learn": False, "learn_hold": 0}}
        if run.get("fixed_mode"):
            # sub-bays only on the auto grid (Dynamic mode): a drawn Fixed map holds exactly the operator's slots and capacities
            run = {**run, "split": self.split_wanted and auto is not None}
            self.split = run["split"]
        if run["db_detector"] != self.db.detector:
            self.db.close()
            self.db = HistoryDB(self.db_path, run["camera"], self.tz, detector=run["db_detector"])
            self._close_older_grids(run["db_detector"])
        self.auto_key = None if auto is None else auto[0]
        return run

    def _session_report(self) -> dict | None:
        """The history against the live count for the snapshot: {"expected" (vehicles counted BUSY), "open" (open sessions), "mismatch_s"
        (how long they have differed, 0 = equal), "error" (a failed write, else "")} - ParkingMonitor._check_sessions."""
        chk = getattr(self.mon, "session_check", None)
        if chk is None:
            return None
        since = chk["since"]
        return {"expected": chk["expected"], "open": chk["open"], "mismatch_s": 0.0 if since is None else round(max(chk["at"] - since, 0.0), 1),
                "error": self.mon.session_error}

    def _watch_sessions(self):
        """One log line when the history has differed from the live count for MISMATCH_LOG_S, one when it is equal again."""
        rep = self._session_report()
        if rep is None:
            return
        if rep["mismatch_s"] >= MISMATCH_LOG_S and not self._mismatch_logged:
            self._mismatch_logged = True
            print(f"live: history differs from the live count for {rep['mismatch_s']:.0f} s: {rep['open']} open session(s), "
                  f"{rep['expected']} vehicle(s) counted" + (f" ({rep['error']})" if rep["error"] else ""), flush=True)
        elif rep["mismatch_s"] == 0.0 and self._mismatch_logged:
            self._mismatch_logged = False
            print("live: the history matches the live count again", flush=True)

    def _close_older_grids(self, detector: str):
        """A new auto grid version (scope "<detector> | auto vN") replaces the older ones of this camera: their worker is gone, so
        sessions still open in them would stay "parked" for ever. They are closed (never deleted) with the note "grid replaced
        by vN"; a newer version, another detector or the learned map are left alone."""
        m = re.search(r" \| auto v(\d+)$", detector)
        if m is None:
            return
        new, pre = int(m.group(1)), f"{self.db.camera} | {detector[:m.start()]} | auto v"
        older = lambda scope: scope.startswith(pre) and scope[len(pre):].isdigit() and int(scope[len(pre):]) < new  # noqa: E731
        try:
            closed = self.db.close_stale(f"grid replaced by v{new}", self.clock(), only=older)
        except sqlite3.Error as e:  # (a locked file: the next start tries again; never a lost frame)
            print(f"live: closing the older grids' sessions failed: {e}", flush=True)
            return
        if closed:
            print(f"live: grid v{new}: closed {sum(closed.values())} open session(s) of {', '.join(sorted(closed))}", flush=True)

    def _metrics(self, snapshot: dict, now: float, frame=None) -> dict | None:
        """The metric view of this frame from the camera's active calibration (Phase 2), or None: uncalibrated, or any
        error here (reported once per distinct message): the frame is published without it, as it always was. The drift
        guard looks at the frame first (it may mark the calibration stale, which makes this frame the first without metrics)."""
        try:
            observe = getattr(self.geometry, "observe", None)
            if observe is not None:
                observe(frame, _kept(snapshot), now)
            return self.geometry.metrics(_kept(snapshot), snapshot["shape"], now)
        except Exception as e:  # noqa: BLE001 - an optional extra must never cost a frame
            msg = f"{type(e).__name__}: {e}"[:200]
            if msg != self._geometry_error:
                self._geometry_error = msg
                print(f"worker {self.owner}: geometry metrics failed ({msg}); frames go out without them", flush=True)
            return None

    def _vectors(self, frame, res) -> dict:
        """The frame's content as data, for overlays drawn by a client (SVG over the video, with tooltips): slots
        ({"id", "name", "state" FREE/BUSY/UNKNOWN, "hidden", "phantom", "duplicate_of", "poly" fractions}),
        detections ({"label", "conf", "src", "kind" vehicle/other, "box" [x1, y1, x2, y2] fractions}) and the
        parking area (cuts as shares of the frame, polygon or None)."""
        h, w = frame.shape[:2]
        box = lambda d: [round(d["box"][0] / w, 4), round(d["box"][1] / h, 4), round(d["box"][2] / w, 4),  # noqa: E731
                         round(d["box"][3] / h, 4)]
        dropped = _rejected_boxes(self.mon.tracker)  # non-vehicle filter on: box -> reason, for boxes it did not let through
        dets = [{"label": d["label"], "conf": round(d["conf"], 3), "src": d.get("src") or "", "kind": kind,
                 "box": box(d), **({"rejected": dropped[_box_key(d)]} if kind == "vehicle" and _box_key(d) in dropped else {})}
                for kind, group in (("vehicle", res.vehicles), ("other", res.others)) for d in group]
        m5 = self.run["m5"]
        area = {k: m5.get(k, 0.0) for k in ("top", "bottom", "left", "right")} | {"roi": m5.get("roi")}
        return {"slots": self.mon.state()["slots"], "detections": dets, "area": area}

    def _stats(self, res=None) -> dict:
        """The numbers behind the page's status line and metrics."""
        if self.mon is None:
            return {"camera": self.run["camera"] if self.run else None}
        tr, st = self.mon.tracker, self.mon.state()
        out = {"camera": self.db.camera, "scope": self.db.scope, "mode": self.mon.mode, "counts": st["counts"],
               "detector": detector_label(self.det), "det_text": self.det_text, "det_auto": self.det_auto,
               "rate": self.n / max(self.clock() - self.t0, 1e-6), "frames": self.n, "logged": self.logged,
               "saved": self.saved, "last_saved": self.last_saved, "learning": len(tr.candidates),
               "dropped": getattr(tr, "dropped", 0), "raw_counts": getattr(self.det, "counts", {})}
        vehicle_bays = [k for k in getattr(tr, "slots", ()) if k >= DYNAMIC_BASE]  # bays made for vehicles (FixedSlotTracker)
        if vehicle_bays:
            out["vehicle_bays"] = len(vehicle_bays)
        if "rejected" in st:  # the non-vehicle filter is on: what it dropped this frame, for the page ({"rejected": n, "reasons": {...}})
            reasons: dict[str, int] = {}
            for r in st["rejected"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            out["obstacle"] = {"rejected": len(st["rejected"]), "reasons": reasons}
        if self.auto_key is not None:
            info, count = self.auto_info, self.auto_key[1]
            parent = getattr(tr, "parent", {})
            out["auto_slots"] = {"version": self.auto_key[0], "count": count, **info, "dynamic": len(vehicle_bays),
                                 "sub": sum(k in parent for k in vehicle_bays),
                                 "sources": {**info.get("sources", {}), **{k: "sub" if k in parent else "dynamic" for k in vehicle_bays}}}
            if info.get("expected") and count < info["expected"]:  # bays the grid could not place: the operator pins them
                out["auto_slots"]["short"] = info["expected"] - count
        if res is not None:
            dropped = _rejected_boxes(tr)
            out.update(vehicles=sum(_box_key(d) not in dropped for d in res.vehicles), others=len(res.others))  # (dropped boxes are not vehicles)
            if (rep := self._session_report()) is not None:
                out["sessions"] = rep
        return out

    def loop(self, max_frames: int = 0, sleep=time.sleep):
        """Run until a "restart" request (exit, so the supervisor starts fresh code) or max_frames frames (tests).
        An error in a pass is reported on the page and the session restarted after a pause; never a crash."""
        try:
            while not self.exit and not (max_frames and self.mon is not None and self.n >= max_frames):
                try:
                    pause = self.step()
                except Exception as e:  # e.g. weights that fail to load, a broken settings row
                    traceback.print_exc()
                    self.stop_session()
                    self.ver = None  # re-apply the settings after the pause
                    if self.holding:
                        self.store.beat(self.owner, self._snapshot(f"error: {e}"), self.clock())
                    pause = 5.0
                if self.pulse is not None:
                    self.pulse.progress = self.clock()
                if pause:
                    sleep(pause)
        finally:
            if self.pulse is not None:
                self.pulse.stop()  # (before release(): a late pulse must not undo it)
            self.stop_session()
            if self.holding:
                self.store.release(self.owner)
