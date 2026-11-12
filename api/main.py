"""FastAPI backend for the parking monitor: REST + one WebSocket over the history database, and the React SPA.

It never runs a detector or opens the camera: the headless worker (scripts/worker.py, carparking/live.py) does, and
publishes its heartbeat, status, slot map, detections and images to the database; this process only reads that, and
writes settings / requests the worker picks up. Settings go through
carparking.live.build_settings, so they are exactly what the Streamlit page stores for the same values: either UI can
be used, and a re-save never restarts the worker's tracking.

  GET  /api/health                  liveness
  GET  /api/status                  worker connection, heartbeat age, status snapshot (counts, slots, detections, area)
  WS   /ws/stream?every=20           live state; every = seconds between routine updates (changes go at once)
  GET  /api/frame.jpg[?raw=1]       the worker's latest overlay image (raw=1: the clean frame, every few seconds)
  GET  /api/settings                current settings as named values (+ version); camera passwords masked
  GET  /api/settings/options        choices for every setting (detectors, classes of the chosen one, ranges)
  PUT  /api/settings          🔒    change some values ({"version": n} optional: 409 if someone saved in between)
  POST /api/commands          🔒    {"action": "reset" | "snapshot" | "restart"}
  GET  /api/sessions, /api/events?since_id=&kinds=moves|notices|all, /api/summary     the history of the running scope
  GET  /api/analytics?days=&scope=all|current      arrivals / departures per hour and day, stay lengths, occupancy
  GET  /api/analytics/overview?days=               every camera: entries, exits, open sessions, state now
  GET  /api/camera-events                          worker online / offline, monitoring on / off (PostgreSQL)
  GET  /api/history/sync                           where the history is served from, how far the copy is
       History & Analytics come from PostgreSQL (the sync service's copy, carparking/history_pg.py) while that copy
       is fresh, else from the camera's SQLite file; header X-History-Source / field "source" says which.
  GET  /api/video                   the camera's video streams; POST /api/webrtc?src= : WebRTC signalling (go2rtc)
  GET  /api/sessions.csv, /api/events.csv                     downloads
  GET  /api/auth                    what the presented token allows
  GET  /api/cameras                 every camera (URLs masked) with streams and live status; POST 🔒 adds one,
       PATCH/DELETE /api/cameras/{id} 🔒 edit / remove (registry: PostgreSQL or cameras.json)
  Every camera endpoint takes ?cam=<id> (default: the primary camera); WS /ws/stream?cam=<id> too.
  POST /api/fixed-slots/snapshot    Slot editor: capture one still of the camera now (its own HTTP snapshot, else go2rtc,
                                    else the worker's last frame), kept in memory; Test slots runs on it
  POST /api/fixed-slots/preview     Test slots (read-only): drawn slots vs the vehicles of the current frame (the worker's,
                                    or a frame grabbed and detected on demand while monitoring is off) -> FREE / BUSY
  GET  /api/fixed-slots?camera=     a camera's fixed slot map (+ mode, editable); PUT 🔒 replaces it (the Streamlit
                                    editor's file); 409 while the camera is in Dynamic mode (Fixed mode only)
  GET  /api/auto-pins?cam=          the auto grid's operator input (Dynamic mode on a HIGH calibration): expected bay count +
                                    bays pinned on the image; PUT 🔒 replaces both (a worker option, applied from the next frame)
  WS   /ws/stream[?frames=1]        {"type": "state"} on every worker change, {"type": "event"} per logged entry/exit,
                                    and (frames=1) the frame JPEG as a binary message per new frame (the clean frame when the
                                  worker makes one, snapshot.clean_frames; else its baked overlay)
🔒 = admin token: header "Authorization: Bearer <CARPARK_API_TOKEN>" (or X-API-Token). Without CARPARK_API_TOKEN set,
writes are refused (read-only API). CARPARK_VIEW_TOKEN set: every read (camera image, video, status, history, the
WebSocket) also needs a token (that one or the admin one; ?token= where headers can't be sent).
Run: uvicorn --factory api.main:create_app --host 0.0.0.0 --port 8000  (one process: one broadcaster).
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

import cv2
import httpx
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.geometry_live import DRIFT_GUARD_OFF, drift_guard_wanted  # noqa: E402
from carparking.geometry_store import geometry_enabled  # noqa: E402
from carparking.history import DISPLAY_TUNING, MOVES, NOTICES, HistoryDB, tuning_values  # noqa: E402
from carparking.history_pg import PgHistory, Scope, SqliteHistory, sqlite_counts  # noqa: E402
from carparking.live import (BAY_SRC, CHOICES, MATCH_RULE_NAMES, MODEL_VARIANTS, SIZES, SLOT_MODES, STALE,  # noqa: E402
                             STREAMS, LiveStore, build_settings, class_names, detector_options, heartbeat_age)
from carparking.fixed_slots import MAX_EXPECTED, MAX_EXTRAPOLATED, FixedSlotStore  # noqa: E402
from carparking import slot_preview  # noqa: E402
from carparking.monitor import parse_roi  # noqa: E402
from carparking.registry import (Camera, PgRegistry, mirror_slots, open_registry, runtime_paths, seed, slugify,  # noqa: E402
                                 streams_of, sync_primary_slots)
from carparking.rtsp import env_setting, mask_url  # noqa: E402
from api.geometry_routes import register_setup  # noqa: E402
from api.sample_sessions import register_sample_sessions  # noqa: E402

POLL = 0.25  # seconds between the broadcaster's looks at the database
PREVIEW_FRESH = 15.0  # Test slots: the worker's detections are used while its frame is at most this old (s)
SNAPSHOT_KEEP = 3  # slot-editor snapshots kept in memory per camera (the newest ones)
SNAPSHOT_RAW_AGE = 600.0  # the worker's last published frame is a snapshot source while at most this old (s)


def state_key(msg: dict):
    """What makes a state message worth sending at once on a throttled stream (?every=): the connection, the settings
    version, monitoring on / off, the worker status, the counts and every slot's shown state. Detections, ages and
    frame numbers change every second and wait for the interval."""
    snap = msg.get("snapshot") or {}
    return (msg.get("connected"), msg.get("settings_version"), snap.get("enabled"), snap.get("status"),
            tuple(sorted((snap.get("counts") or {}).items())),
            tuple((x.get("id"), x.get("state"), x.get("hidden"), x.get("blocked")) for x in snap.get("slots") or []),
            # Phase 2: a new calibration or a change of the bay occupancy is worth sending at once
            (snap.get("geometry") or {}).get("version"), tuple(sorted(((snap.get("geometry") or {}).get("counts") or {}).items())))
TOKEN_IN_URL = re.compile(r"([?&]token=)[^&\s\"]*")


class MaskTokens(logging.Filter):
    """<img>, <video> and the WebSocket carry the viewer token as ?token= (no headers there): uvicorn's access and
    WebSocket log lines would print it. Masked before any handler sees the record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(TOKEN_IN_URL.sub(r"\1***", a) if isinstance(a, str) else a for a in record.args)
        if isinstance(record.msg, str) and "token=" in record.msg:
            record.msg = TOKEN_IN_URL.sub(r"\1***", record.msg)
        return True


def mask_token_logs():
    for name in ("uvicorn.access", "uvicorn.error"):
        log = logging.getLogger(name)
        if not any(isinstance(f, MaskTokens) for f in log.filters):
            log.addFilter(MaskTokens())


class Config:
    """Paths and secrets from the environment, secrets also from the project's .env (the bind-mounted dev setup).
    Tests pass their own dict: then only that dict counts, never the real .env."""

    def __init__(self, env=None):
        dotenv = ROOT / ".env" if env is None else None
        env = os.environ if env is None else env
        secret = lambda name: env.get(name) or (env_setting(name, dotenv) if dotenv else "")  # noqa: E731
        self.env = env  # the camera registry: CARPARK_DATABASE_URL (PostgreSQL) or cameras.json in data_dir
        self.db = Path(env.get("CARPARK_HISTORY_DB", ROOT / "parking_history.db"))  # the primary camera's
        self.data_dir = self.db.parent
        self.fixed = Path(env.get("CARPARK_FIXED_SLOTS", ROOT / "config" / "fixed_slots.json"))
        self.raw = Path(env.get("CARPARK_DATASET_RAW", self.data_dir / "dataset_raw"))
        self.rtsp = secret("CARPARK_RTSP_URL")  # the primary camera, seeded into an empty registry
        self.rtsp_sub = secret("CARPARK_RTSP_SUB_URL")
        self.token = secret("CARPARK_API_TOKEN")
        self.spa = Path(env.get("CARPARK_SPA_DIR", ROOT / "frontend" / "dist"))
        # viewing (camera image, video, status, history): open without it; with it, this token or the admin one
        self.view_token = secret("CARPARK_VIEW_TOKEN")
        # go2rtc as this process reaches it (compose: the service, never published): WebRTC signalling is proxied
        self.go2rtc = env.get("CARPARK_GO2RTC_INTERNAL", "http://go2rtc:1984")
        # Phase 2 camera setup: off unless switched on; the on-demand `setup` service (SAM / GeoCalib / depth) it calls
        self.geometry = geometry_enabled(env)
        self.setup_url = env.get("CARPARK_SETUP_URL", "http://setup:8100")
        try:
            self.tz = ZoneInfo(env.get("CARPARK_TZ", "Asia/Karachi"))
        except Exception:
            self.tz = None


@contextlib.contextmanager
def opened(cls, *args, **kw):
    """A short-lived connection per use: requests run on a thread pool, and one SQLite connection must not be used
    by two threads at once."""
    obj = cls(*args, **kw)
    try:
        yield obj
    finally:
        obj.close()


# --- settings: named values <-> the Streamlit page's widget keys ---------------------------------------------------
STREAM_NAMES = {"main": STREAMS[0], "sub": STREAMS[1], "custom": STREAMS[2]}
SIZE_NAMES = {"1280": SIZES[0], "1920": SIZES[1], "sahi": SIZES[2]}
MODE_NAMES = {"dynamic": SLOT_MODES[0], "fixed": SLOT_MODES[1]}
# Slot maps are drawn only in Fixed mode: in Dynamic mode the worker finds the slots itself (read-only editor)
DYNAMIC_LOCKED = ("Dynamic Mode automatically detects parking spaces. Switch to Fixed Mode to manually draw or edit "
                  "slots.")
CHOICE_NAMES = {v: k for k, v in CHOICES.items()}  # auto / baseline / obb -> the page's label
FIELDS = {  # named value -> (the page's widget key, named -> page value); identity unless mapped
    "enabled": ("enabled5", None), "camera": ("camera5", None), "custom_url": ("url5", None),
    "stream": ("stream5", STREAM_NAMES), "detector": ("vsrc5", None), "bay_model": ("model5", CHOICE_NAMES),
    "inference": ("size5", SIZE_NAMES), "conf": ("conf5", None), "classes": ("classes5", None),
    "extra": ("extra5", None), "slot_mode": ("slotmode5", MODE_NAMES), "match_rule": ("match5", None),
    "match_iou": ("matchiou5", None), "save_every": ("save5", None), "hold": ("hold5", None),
    "exit_hold": ("exit5", None), "exit_frames": ("exitn5", None), "occlusion": ("occl5", None),
    "hidden_max": ("hmax5", None), "learn_hold": ("lhold5", None), "learn": ("learn5", None),
    "absent_free": ("absent5", None),
}
AREA = ("cut_top", "cut_bottom", "cut_left", "cut_right", "roi")  # per camera (HistoryDB settings), like the page


class SettingsPatch(BaseModel):
    """Values to change (all optional; ranges as the Streamlit sliders). version: the one you read (409 if stale)."""
    version: int | None = None
    enabled: bool | None = None
    camera: str | None = Field(None, min_length=1, max_length=80)
    stream: str | None = Field(None, pattern="^(main|sub|custom)$")
    custom_url: str | None = Field(None, max_length=500)
    detector: str | None = None
    bay_model: str | None = Field(None, pattern="^(auto|baseline|obb)$")
    inference: str | None = Field(None, pattern="^(1280|1920|sahi)$")
    conf: float | None = Field(None, ge=0.05, le=0.50)
    classes: list[str] | None = None
    extra: list[str] | None = None
    slot_mode: str | None = Field(None, pattern="^(dynamic|fixed)$")
    match_rule: str | None = Field(None, pattern="^(ground|center|iou)$")
    match_iou: float | None = Field(None, ge=0.05, le=0.8)
    save_every: int | None = Field(None, ge=0, le=600)
    hold: int | None = Field(None, ge=0, le=120)
    exit_hold: int | None = Field(None, ge=0, le=600)
    exit_frames: int | None = Field(None, ge=1, le=60)
    occlusion: float | None = Field(None, ge=0.1, le=0.9)
    hidden_max: int | None = Field(None, ge=10, le=1800)
    learn_hold: int | None = Field(None, ge=0, le=300)
    learn: bool | None = None
    absent_free: bool | None = None
    cut_top: int | None = Field(None, ge=0, le=60)
    cut_bottom: int | None = Field(None, ge=0, le=60)
    cut_left: int | None = Field(None, ge=0, le=60)
    cut_right: int | None = Field(None, ge=0, le=60)
    roi: str | None = Field(None, max_length=2000)
    # the stock COCO detector's weights (auto = picked from the worker's hardware); a worker option, not part of the
    # page-shared settings (the Streamlit page rewrites those whole), so it doesn't change their version
    model_variant: str | None = Field(None, pattern="^(auto|default|yolo11m)$")
    # the spillage layer (spillage.py: a free bay a neighbour's vehicle stands on is ENCROACHED, not counted free) and the non-vehicle
    # filter (obstacle_filter.py: cones, people, shadows do not make a bay BUSY): worker options like model_variant, both off by
    # default, applied to the running tracker on the worker's next frame (no restart, no settings version)
    spillage: bool | None = None
    obstacle_filter: bool | None = None
    # blocked by object (blocked.py): a free bay with a potted plant, vase, box ... in it is BLOCKED (free for the history, not counted free):
    # a worker option too, off by default, applied on the worker's next frame
    blocked_objects: bool | None = None
    # the camera drift guard (geometry_live: the worker checks the static background against the active calibration and stales it on a
    # confirmed move): a worker option, off by default (CARPARK_DRIFT_GUARD sets the default), applied on the worker's next frame; the
    # active calibration version is never touched by this switch
    drift_guard: bool | None = None
    # display thresholds of the shown state (history.DISPLAY_TUNING; a worker option like model_variant, not part of
    # the page-shared settings): what counts as FREE / BUSY / UNKNOWN, never the slot map or the history
    hidden_ground: float | None = Field(None, ge=DISPLAY_TUNING["hidden_ground"][1], le=DISPLAY_TUNING["hidden_ground"][2])
    ground_share: float | None = Field(None, ge=DISPLAY_TUNING["ground_share"][1], le=DISPLAY_TUNING["ground_share"][2])
    vehicle_iou: float | None = Field(None, ge=DISPLAY_TUNING["vehicle_iou"][1], le=DISPLAY_TUNING["vehicle_iou"][2])
    bay_share: float | None = Field(None, ge=DISPLAY_TUNING["bay_share"][1], le=DISPLAY_TUNING["bay_share"][2])
    pin_reach: float | None = Field(None, ge=DISPLAY_TUNING["pin_reach"][1], le=DISPLAY_TUNING["pin_reach"][2])


def stored_tuning(store) -> dict[str, float]:
    """The display thresholds a camera's worker uses: the stored option "tuning", else the defaults."""
    try:
        raw = json.loads(store.option("tuning", "{}") or "{}")
    except ValueError:
        raw = {}
    return tuning_values(raw)


def _ranges() -> dict[str, list]:
    """{field: [min, max]} of the numeric settings, from SettingsPatch (for sliders in a client)."""
    out = {}
    for name, f in SettingsPatch.model_fields.items():
        ge = next((m.ge for m in f.metadata if hasattr(m, "ge")), None)
        le = next((m.le for m in f.metadata if hasattr(m, "le")), None)
        if ge is not None and le is not None:
            out[name] = [ge, le]
    return out


class FixedSlotsIn(BaseModel):
    """A camera's fixed slot map: [{"id", "name", "poly": [[x, y], ...] as shares of the frame (0..1)}]. The camera:
    cam (registry id, or ?cam=), else camera (name)."""
    camera: str | None = Field(None, max_length=80)
    cam: str | None = Field(None, max_length=40)
    slots: list[dict] = Field(max_length=500)


class AutoPinsIn(BaseModel):
    """The auto grid's operator input (Dynamic mode on a HIGH calibration): expected = how many bays the lot has (null: not
    set), pins = bays drawn on the image where the calibration cannot see: [{"id", "name", "poly" (frame fractions)}]."""
    expected: int | None = Field(None, ge=1, le=MAX_EXPECTED)
    pins: list[dict] = Field(default_factory=list, max_length=100)


class OverflowIn(BaseModel):
    """The auto grid's vehicle-driven bays and sub-bays as one switch: enabled sets both worker options (auto_dynamic, slot_split)."""
    enabled: bool


class SlotPreviewIn(BaseModel):
    """Test slots (read-only): the slots as drawn in the editor (saved or not) and, optionally, which ids to test.
    snapshot: test against this editor snapshot (POST /api/fixed-slots/snapshot) instead of the current frame."""
    slots: list[dict] = Field(max_length=500)
    only: list[int] | None = Field(None, max_length=500)
    snapshot: str | None = Field(None, pattern="^[0-9a-f]{32}$")


class CameraIn(BaseModel):
    """A new camera. name: its history key (fixed); label: the display name (editable later; default: name);
    id: the URL key (default: from name)."""
    name: str = Field(min_length=1, max_length=80)
    label: str | None = Field(None, max_length=80)
    id: str | None = Field(None, pattern="^[a-z0-9][a-z0-9-]{0,39}$")
    rtsp_url: str = Field(min_length=1, max_length=500)
    rtsp_sub_url: str | None = Field(None, max_length=500)
    sort_order: int | None = Field(None, ge=0, le=999)
    metadata: dict | None = None


class CameraPatch(BaseModel):
    label: str | None = Field(None, max_length=80)  # the display name ("East Gate"); the name / history key stays
    rtsp_url: str | None = Field(None, min_length=1, max_length=500)
    rtsp_sub_url: str | None = Field(None, max_length=500)
    sort_order: int | None = Field(None, ge=0, le=999)
    metadata: dict | None = None


@dataclass
class CamCtx:
    """One camera as the endpoints use it: its registry entry (None: no registry, the environment's camera) and its
    runtime files (carparking.registry.runtime_paths)."""
    camera: Camera | None
    db: Path
    fixed: Path
    raw: Path
    rtsp: str
    tz: object = None
    geometry: bool = False  # CARPARK_GEOMETRY_ENABLED: whether this camera's stream may carry Phase 2 metrics


PRIMARY_STREAMS = {"parking_hd", "parking", "parking_native"}  # go2rtc.yaml (the primary camera)


class Command(BaseModel):
    action: str = Field(pattern="^(reset|snapshot|restart|close_stale)$")
    # close_stale only: count what would be closed and change nothing (the page asks first)
    dry_run: bool = False


# every history / analytics answer: always fresh (the page polls it, a browser or proxy must never serve an earlier one)
NO_STORE = {"Cache-Control": "no-store"}


def live_detector(run: dict, snap: dict | None) -> str:
    """The history detector the worker writes with. The worker's snapshot names its scope ("<camera> | <detector>"), which carries
    " | auto vN" while the auto grid runs (Worker._auto_run); the stored settings' db_detector does not. Reading the history by the
    settings alone showed an older scope's sessions (camera-1 on 2026-10-26: 25 October rows, 3 open, while the worker wrote to
    auto v10). No snapshot yet / another camera's name: the settings' own."""
    scope, prefix = (snap or {}).get("scope") or "", f"{run['camera']} | "
    return scope[len(prefix):] if scope.startswith(prefix) and len(scope) > len(prefix) else run["db_detector"]


class Hub:
    """One poller for every WebSocket client: looks at the worker's published state every POLL seconds and fans out
    what changed (a slow client only drops its own oldest messages, never stalls the others). All its database use
    goes through one connection under a lock."""

    def __init__(self, cfg: Config, queue_size: int = 20):
        self.cfg, self.queue_size = cfg, queue_size
        self.store, self.lock = LiveStore(cfg.db), threading.Lock()
        self.clients: dict[asyncio.Queue, bool] = {}
        self.last_beat = self.last_seq = self.last_event = None
        self.primed = False  # the first look only records: every new client got the current state on connect

    def join(self, frames: bool) -> asyncio.Queue:
        q = asyncio.Queue(self.queue_size)
        self.clients[q] = frames
        return q

    def leave(self, q):
        self.clients.pop(q, None)

    def status_message(self) -> dict:
        with self.lock:
            state = self.store.state(images=False)
            _, ver = self.store.config()
        age = heartbeat_age(state)
        snap = state.get("snapshot") or {}
        if "geometry" in snap and not self.cfg.geometry:  # (cfg here is the camera's CamCtx)  # Phase 2 is off here: never pass its data on, whatever a worker did
            snap = {k: v for k, v in snap.items() if k != "geometry"}
        return {"type": "state", "connected": age is not None and age <= STALE, "heartbeat_age": age,
                "settings_version": ver, "snapshot": snap, "time": time.time()}

    def _push(self, msg, frames_only: bool = False):
        for q, wants_frames in list(self.clients.items()):
            if frames_only and not wants_frames:
                continue
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(msg)

    def _new_events(self, run: dict | None, snap: dict | None = None) -> list[dict]:
        if not run:
            return []
        with opened(HistoryDB, self.cfg.db, run["camera"], self.cfg.tz, detector=live_detector(run, snap)) as db:
            if self.last_event is None:  # start after the newest event: clients fetch the past via /api/events
                newest = db.events(limit=1, kinds=None)
                self.last_event = newest[0]["id"] if newest else 0
                return []
            new = list(reversed(db.events(limit=100, since_id=self.last_event, kinds=None)))  # notices too: the ticker files them apart
        if new:
            self.last_event = new[-1]["id"]
        return new

    def tick(self) -> list:
        """One look (sync, run in a thread) -> messages to send (dict = JSON, bytes = a frame for frames=1)."""
        out = []
        with self.lock:
            state = self.store.state(images=False)
            cfg, _ = self.store.config()
        beat, seq = state.get("heartbeat"), (state.get("snapshot") or {}).get("seq")
        if not self.primed:
            self.primed, self.last_beat, self.last_seq = True, beat, seq
        elif beat != self.last_beat:
            self.last_beat = beat
            out.append(self.status_message())
            if seq is not None and seq != self.last_seq:
                self.last_seq = seq
                with self.lock:
                    st = self.store.state()
                # the page draws the layers itself: the clean frame published with this snapshot, if the worker makes one
                # (snapshot.clean_frames, option clean_frames), else the baked overlay
                clean = (st.get("snapshot") or {}).get("clean_frames") and st.get("raw")
                img = st["raw"] if clean else st.get("overlay")
                if img:
                    out.append(img)
        return out + [{"type": "event", **e} for e in self._new_events((cfg or {}).get("run"), state.get("snapshot"))]

    async def run(self):
        while True:
            try:
                if self.clients:
                    for msg in await asyncio.to_thread(self.tick):
                        self._push(msg, frames_only=isinstance(msg, bytes))
            except Exception as e:  # a locked or briefly missing database: try again on the next tick
                print(f"broadcaster: {e}", flush=True)
            await asyncio.sleep(POLL)


def _given(authorization: str, x_api_token: str, token: str) -> str:
    """The token a request carries: X-API-Token, Authorization: Bearer, or ?token= (an <img> or a WebSocket can't
    send headers)."""
    return x_api_token or authorization.removeprefix("Bearer ").strip() or token


def _matches(given: str, *tokens: str) -> bool:
    return any(t and secrets.compare_digest(given.encode(), t.encode()) for t in tokens)


def create_app(cfg: Config | None = None, names_of=class_names, http: httpx.AsyncClient | None = None,
               setup_http: httpx.AsyncClient | None = None) -> FastAPI:
    mask_token_logs()
    """The app. names_of(weights) -> (vehicle classes, other classes) (tests pass a stub: no model loading).
    http: the client for go2rtc, setup_http: the one for the setup service (tests pass ones with a mock transport)."""
    cfg = cfg or Config()
    registry = open_registry(cfg.env, data_dir=cfg.data_dir)
    with opened(LiveStore, cfg.db) as store:  # also creates the live_* tables in a fresh database
        stored, _ = store.config()
    seed(registry, ((stored or {}).get("run") or {}).get("camera") or "Camera 1", cfg.rtsp, cfg.rtsp_sub, cfg.fixed)
    hubs: dict[str, Hub] = {}
    hub_tasks: dict[str, asyncio.Task] = {}
    pg = PgHistory(registry.dsn) if isinstance(registry, PgRegistry) else None  # the central history (Step 4)

    @contextlib.asynccontextmanager
    async def lifespan(_):
        try:
            yield
        finally:
            for task in hub_tasks.values():
                task.cancel()
            for task in hub_tasks.values():
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="Parking monitor API", lifespan=lifespan)
    app.state.cfg, app.state.registry = cfg, registry

    def require_token(authorization: str = Header(""), x_api_token: str = Header(""), token: str = ""):
        if not cfg.token:
            raise HTTPException(503, "writes are disabled: set CARPARK_API_TOKEN")
        if not _matches(_given(authorization, x_api_token, token), cfg.token):
            raise HTTPException(401, "missing or wrong API token")

    def require_view(authorization: str = Header(""), x_api_token: str = Header(""), token: str = ""):
        if cfg.view_token and not _matches(_given(authorization, x_api_token, token), cfg.view_token, cfg.token):
            raise HTTPException(401, "missing or wrong viewer token")

    view = [Depends(require_view)]
    admin = [Depends(require_token)]

    # --- which camera ---------------------------------------------------------------------------------------------
    def ctx_of(camera: Camera | None) -> CamCtx:
        if camera is None:  # no registry entry (a bare test setup): the environment's single camera
            return CamCtx(None, cfg.db, cfg.fixed, cfg.raw, cfg.rtsp, cfg.tz, cfg.geometry)
        rt = runtime_paths(camera, cfg.data_dir, cfg.db, cfg.fixed, cfg.raw)
        return CamCtx(camera, rt.db, rt.fixed, rt.raw, camera.rtsp_url, cfg.tz, cfg.geometry)

    def ctx(cam: str | None = None) -> CamCtx:
        """?cam=<id> (default: the primary camera, which every endpoint used before cameras existed)."""
        camera = registry.get(cam) if cam else registry.primary()
        if cam and camera is None:
            raise HTTPException(404, f"no camera {cam!r}")
        return ctx_of(camera)

    def hub_of(c: CamCtx) -> Hub:
        key = c.camera.id if c.camera else ""
        if key not in hubs:
            hubs[key] = Hub(c)
        return hubs[key]

    def current(c: CamCtx) -> tuple[dict, int]:
        """The camera's stored settings (or the defaults the page would start from) and their version."""
        with opened(LiveStore, c.db) as store:
            stored, ver = store.config()
        if stored and stored.get("run"):
            return stored, ver
        ui = dict((stored or {}).get("ui", {}))
        if c.camera is not None and not c.camera.primary:
            ui["camera5"] = c.camera.name  # a registry camera's history scope is its name
        return build_settings(ui, ROOT, c.db, c.rtsp, c.fixed, names_of), ver

    def history_detector(c: CamCtx) -> tuple[dict, str]:
        """(run, the detector of the scope the worker is writing history to now)."""
        run = current(c)[0]["run"]
        with opened(LiveStore, c.db) as store:
            snap = store.state(images=False).get("snapshot")
        return run, live_detector(run, snap)

    def history_db(c: CamCtx) -> HistoryDB:
        run, detector = history_detector(c)
        return HistoryDB(c.db, run["camera"], c.tz, detector=detector)

    def named(settings: dict, ver: int, c: CamCtx) -> dict:
        """Settings as the values a client edits (what the worker runs with), + version and the run itself."""
        ui, run = settings.get("ui", {}), settings["run"]
        m5 = run["m5"]
        detectors = detector_options(ROOT, run["camera"])[0]
        values = {
            "enabled": settings["enabled"], "camera": run["camera"], "stream": run["stream"],
            "custom_url": mask_url(run["custom_url"]) if run["custom_url"] else "",
            "detector": ui.get("vsrc5") if ui.get("vsrc5") in detectors else detectors[0],
            "bay_model": CHOICES.get(ui.get("model5"), "auto"),
            "inference": {1280: "1280", 1920: "1920"}.get(m5["whole"], "sahi"), "conf": run["conf"],
            "classes": m5["classes"] or [], "extra": m5["extra"], "slot_mode": "fixed" if run["fixed_mode"] else
            "dynamic", "match_rule": ui.get("match5", m5["match"]), "match_iou": ui.get("matchiou5", m5["match_iou"]),
            "save_every": m5["save_every"], "hold": m5["hold"], "exit_hold": m5["exit_hold"],
            "exit_frames": m5["exit_frames"], "occlusion": m5["occlusion"], "hidden_max": m5["hidden_max"],
            "learn_hold": ui.get("lhold5", m5["learn_hold"]), "learn": ui.get("learn5", m5["learn"]),
            "absent_free": m5["absent_free"],
            **{f"cut_{s}": round(m5[s] * 100) for s in ("top", "bottom", "left", "right")},
        }
        with opened(HistoryDB, c.db, run["camera"]) as db:
            values["roi"] = db.setting("roi")  # the text as typed (the parking area lives per camera, as on the page)
        with opened(LiveStore, c.db) as store:
            values["model_variant"] = store.option("model_variant", "auto")
            values.update(stored_tuning(store))
            values.update({k: store.option(k, "0") not in ("", "0") for k in ("spillage", "obstacle_filter", "blocked_objects")})
            guard = (store.option("drift_guard", "") or "").strip().lower()  # unset: the deployment's default (CARPARK_DRIFT_GUARD, off)
            values["drift_guard"] = drift_guard_wanted(cfg.env) if guard == "" else guard not in DRIFT_GUARD_OFF
        effective = {**run, "custom_url": values["custom_url"]}
        return {"version": ver, "values": values, "effective": effective, "fixed_slots": run["fixed"],
                "cam": c.camera.id if c.camera else None}

    def write_settings(c: CamCtx, ui: dict) -> tuple[dict, int]:
        new = build_settings(ui, ROOT, c.db, c.rtsp, c.fixed, names_of)
        with opened(LiveStore, c.db) as store:
            return new, store.set_config(new)

    # --- endpoints (every camera endpoint takes ?cam=<id>; none = the primary camera) ------------------------------
    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/auth")
    def auth(authorization: str = Header(""), x_api_token: str = Header(""), token: str = ""):
        """What this client may do: {"view", "admin", "view_required", "writes_enabled"}."""
        given = _given(authorization, x_api_token, token)
        is_admin = bool(cfg.token) and _matches(given, cfg.token)
        return {"view": not cfg.view_token or is_admin or _matches(given, cfg.view_token), "admin": is_admin,
                "view_required": bool(cfg.view_token), "writes_enabled": bool(cfg.token)}

    @app.get("/api/status", dependencies=view)
    def status(cam: str | None = None):
        return hub_of(ctx(cam)).status_message()

    @app.get("/api/frame.jpg", dependencies=view)
    def frame(raw: bool = False, cam: str | None = None):
        with opened(LiveStore, ctx(cam).db) as store:
            img = store.state().get("raw" if raw else "overlay")
        if not img:
            raise HTTPException(404, "no frame published yet")
        return Response(img, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/settings", dependencies=view)
    def get_settings(cam: str | None = None):
        c = ctx(cam)
        return named(*current(c), c)

    @app.get("/api/settings/options", dependencies=view)
    def options(camera: str | None = None, detector: str | None = None, cam: str | None = None):
        settings, _ = current(ctx(cam))
        camera = camera or settings["run"]["camera"]
        detectors, coco, hybrid = detector_options(ROOT, camera)
        chosen = detector or settings.get("ui", {}).get("vsrc5", detectors[0])
        chosen = chosen if chosen in detectors else detectors[0]
        kinds, others = (([], []) if chosen == BAY_SRC else
                         names_of(list(hybrid[chosen]) if chosen in hybrid else [coco[chosen]]))
        return {"stream": list(STREAM_NAMES), "detector": detectors, "bay_model": list(CHOICE_NAMES),
                "inference": list(SIZE_NAMES), "slot_mode": list(MODE_NAMES), "match_rule": MATCH_RULE_NAMES,
                "classes": kinds, "extra": others, "detector_selected": chosen, "ranges": _ranges(),
                "model_variant": list(MODEL_VARIANTS),
                "tuning_defaults": {k: v[0] for k, v in DISPLAY_TUNING.items()}}

    @app.put("/api/settings", dependencies=admin)
    def put_settings(patch: SettingsPatch, cam: str | None = None):
        c = ctx(cam)
        settings, ver = current(c)
        if patch.version is not None and patch.version != ver:
            raise HTTPException(409, f"settings changed meanwhile (now version {ver}): reload and apply again")
        changes = patch.model_dump(exclude_unset=True, exclude={"version"})
        variant = changes.pop("model_variant", None)
        if variant is not None:  # the worker reads it every second and swaps the detector on the next frame
            with opened(LiveStore, c.db) as store:
                store.set_option("model_variant", variant)
        switches = {k: changes.pop(k) for k in ("spillage", "obstacle_filter", "blocked_objects", "drift_guard") if changes.get(k) is not None}
        for k in ("spillage", "obstacle_filter", "blocked_objects", "drift_guard"):
            changes.pop(k, None)
        if switches:
            with opened(LiveStore, c.db) as store:
                for k, on in switches.items():
                    store.set_option(k, "1" if on else "0")
        tuning = {k: changes.pop(k) for k in list(changes) if k in DISPLAY_TUNING}
        if tuning:  # the worker applies them to the running tracker on its next frame (no restart, no new version)
            with opened(LiveStore, c.db) as store:
                store.set_option("tuning", json.dumps({**stored_tuning(store), **tuning}))
        if "camera" in changes and c.camera is not None and changes["camera"] != current(c)[0]["run"]["camera"]:
            raise HTTPException(422, "camera: a camera's name is its history key; add a new camera instead")
        changes.pop("camera", None)
        if not changes:
            return named(settings, ver, c)
        ui = dict(settings.get("ui", {}))
        for name, value in changes.items():
            if name not in FIELDS or (name == "custom_url" and value and "***" in value):
                continue  # area values below; a masked URL sent back unchanged keeps the stored one
            key, mapping = FIELDS[name]
            if mapping is not None:
                if value not in mapping:
                    raise HTTPException(422, f"{name}: unknown value {value!r}")
                value = mapping[value]
            ui[key] = value
        if c.camera is not None and not c.camera.primary:
            ui["camera5"] = c.camera.name
        camera = (ui.get("camera5") or "Camera 1").strip() or "Camera 1"
        if "detector" in changes and changes["detector"] not in detector_options(ROOT, camera)[0]:
            raise HTTPException(422, f"detector: not available for camera {camera!r}")
        if changes.get("roi"):
            try:
                parse_roi(changes["roi"])
            except ValueError as e:
                raise HTTPException(422, f"roi: {e}") from None
        area = {k: v for k, v in changes.items() if k in AREA}
        if area:
            with opened(HistoryDB, c.db, camera) as db:
                for k, v in area.items():
                    db.set_setting(k, (v or "") if k == "roi" else str(v))
        new, ver = write_settings(c, ui)
        return named(new, ver, c)

    # --- fixed slot maps: in the registry (PostgreSQL / cameras.json), mirrored into each camera's slot file -----
    def registry_camera(cam: str | None, name: str | None) -> Camera | None:
        if cam:
            camera = registry.get(cam)
            if camera is None:
                raise HTTPException(404, f"no camera {cam!r}")
            return camera
        if name:
            return next((x for x in registry.list() if x.name == name), None)
        return registry.primary()

    @app.get("/api/fixed-slots", dependencies=view)
    def get_fixed_slots(camera: str | None = None, cam: str | None = None):
        """A camera's fixed slot map: {"camera" (name), "cam" (id), "updated", "slots", "cameras" (all names with a
        map), "in_use" (its settings run it in fixed mode)}. ?cam=<id>, or ?camera=<name> (also names outside the
        registry: the slot file only)."""
        reg_cam = registry_camera(cam, camera)
        if reg_cam is None:  # a name that is not a registered camera: the shared slot file, as before
            settings, _ = current(ctx(None))
            out = FixedSlotStore(cfg.fixed).get(camera)
            out.update(cameras=FixedSlotStore(cfg.fixed).cameras(), cam=None,
                       in_use=settings["run"]["camera"] == camera and bool(settings["run"]["fixed_mode"]))
            return out
        c = ctx_of(reg_cam)
        if reg_cam.primary:
            sync_primary_slots(registry, reg_cam, c.fixed, cfg.tz)  # the Streamlit editor may have changed the file
        entry = registry.slots(reg_cam.id)
        run = current(c)[0]["run"]
        names = sorted({x.name for x in registry.list()} | set(FixedSlotStore(cfg.fixed).cameras()))
        mode = "fixed" if run["fixed_mode"] else "dynamic"
        return {"camera": reg_cam.name, "cam": reg_cam.id, "updated": entry["updated"], "slots": entry["slots"],
                "cameras": names, "in_use": run["camera"] == reg_cam.name and bool(run["fixed_mode"]),
                "mode": mode, "editable": mode == "fixed", "locked": None if mode == "fixed" else DYNAMIC_LOCKED}

    @app.put("/api/fixed-slots", dependencies=admin)
    def put_fixed_slots(body: FixedSlotsIn, cam: str | None = None):
        """Replace a camera's fixed slot map (same validation as the Streamlit editor). Stored in the registry and
        written to the camera's slot file; if the camera runs in fixed mode its settings are rebuilt, so the worker
        applies the map within a second (slots that stay keep their state; a removed slot's session is closed).
        409 while the camera is in Dynamic mode: slots are drawn / edited only in Fixed mode."""
        name = (body.camera or "").strip()
        reg_cam = registry_camera(cam or body.cam, name or None)
        if reg_cam is not None and not current(ctx_of(reg_cam))[0]["run"]["fixed_mode"]:
            raise HTTPException(409, DYNAMIC_LOCKED)
        try:
            if reg_cam is None:  # not a registered camera: the shared slot file only (as before cameras existed)
                if not name:
                    raise HTTPException(422, "camera: required")
                saved = FixedSlotStore(cfg.fixed).save(name, body.slots, cfg.tz)
                return {"camera": name, "cam": None, "slots": saved, "settings_version": current(ctx(None))[1]}
            saved, ver = save_fixed(reg_cam, body.slots)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"camera": reg_cam.name, "cam": reg_cam.id, "slots": saved, "settings_version": ver}

    def save_fixed(reg_cam: Camera, slots: list[dict]) -> tuple[list[dict], int]:
        """Store a registered camera's Fixed map and hand it to its worker -> (the stored slots, the settings version). ValueError
        for a bad slot. Shared by the slot editor's save and by Accept on a slot suggestion."""
        saved = registry.set_slots(reg_cam.id, slots)
        c = ctx_of(reg_cam)
        mirror_slots(registry, reg_cam, c.fixed, cfg.tz)
        settings, ver = current(c)
        if settings["run"]["camera"] == reg_cam.name:
            _, ver = write_settings(c, settings.get("ui", {}))  # unchanged unless the map changed what the worker runs
        return saved, ver

    # --- the auto grid's operator input: expected bay count + pinned bays (any slot mode; used in Dynamic mode) -----------
    def grid_target(cam: str | None) -> tuple[CamCtx, str]:
        c = ctx(cam)
        return c, (c.camera.name if c.camera else current(c)[0]["run"]["camera"])

    @app.get("/api/auto-pins", dependencies=view)
    def get_auto_pins(cam: str | None = None):
        """{"camera", "cam", "updated", "expected", "pins", "max_extrapolated"}: how many bays the camera's lot has and the
        bays pinned on the image (Dynamic mode on a HIGH calibration). The grid itself is in the status (slots,
        auto_slots.sources)."""
        c, name = grid_target(cam)
        return {**FixedSlotStore(c.fixed).get_auto_grid(name), "cam": c.camera.id if c.camera else None,
                "max_extrapolated": MAX_EXTRAPOLATED}

    @app.put("/api/auto-pins", dependencies=admin)
    def put_auto_pins(body: AutoPinsIn, cam: str | None = None):
        """Replace the camera's expected bay count and pinned bays (same slot validation as the editor). Stored with the
        camera's slot file and handed to its worker as an option: the grid is rebuilt from the next frame, nothing restarts
        and no settings version changes. Sending no count and no pins clears both."""
        c, name = grid_target(cam)
        try:
            saved = FixedSlotStore(c.fixed).save_auto_grid(name, body.expected, body.pins, cfg.tz)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        with opened(LiveStore, c.db) as store:
            store.set_option("auto_grid", json.dumps({"expected": saved["expected"], "pins": saved["pins"]})
                             if saved["expected"] or saved["pins"] else "")
        return {**saved, "cam": c.camera.id if c.camera else None, "max_extrapolated": MAX_EXTRAPOLATED}

    # --- dynamic overflow & sub-bays: the worker options auto_dynamic + slot_split as one switch (auto grid only) ----------
    def overflow_state(store: LiveStore) -> dict:
        # the worker's own defaults ("1" while unset, Worker.step / Worker._auto_slots): on unless the option says "0"
        flags = {k: store.option(k, "1") != "0" for k in ("auto_dynamic", "slot_split")}
        return {"enabled": all(flags.values()), **flags}

    @app.get("/api/overflow", dependencies=view)
    def get_overflow(cam: str | None = None):
        """{"enabled" (both on), "auto_dynamic", "slot_split"}: what the camera's worker runs with now."""
        with opened(LiveStore, ctx(cam).db) as store:
            return overflow_state(store)

    @app.put("/api/overflow", dependencies=admin)
    def put_overflow(body: OverflowIn, cam: str | None = None):
        """Set both options at once; the worker rebuilds its monitor on the next frame, no restart, no settings version."""
        with opened(LiveStore, ctx(cam).db) as store:
            for key in ("auto_dynamic", "slot_split"):
                store.set_option(key, "1" if body.enabled else "0")
            return overflow_state(store)

    # slot-editor snapshots: {id: {"cam", "jpeg", "frame", "ts", "source", "dets", "ver", "task" (settings version,
    # the running detection) or None}}, SNAPSHOT_KEEP per camera,
    # in memory only (gone after a restart: the editor takes a new one)
    snapshots: dict[str, dict] = {}

    async def capture_still(c: CamCtx) -> tuple[str, dict]:
        """One frame of the camera captured now and kept in `snapshots` (SNAPSHOT_KEEP per camera) -> (id, still).
        Sources, first that answers: the camera's own HTTP snapshot, go2rtc, the worker's last frame. 502 when none."""
        camera = c.camera or registry.primary()
        jpeg, source, ts = None, None, time.time()
        if camera is not None:
            jpeg, source = await camera_still(camera), "camera"
            if jpeg is None:
                jpeg, source = await grab_frame(camera), "go2rtc"
        if jpeg is None:
            with opened(LiveStore, c.db) as store:
                state = store.state()
            frame_ts = (state.get("snapshot") or {}).get("frame_ts")
            if state.get("raw") and frame_ts is not None and time.time() - frame_ts <= SNAPSHOT_RAW_AGE:
                jpeg, source, ts = state["raw"], "worker", frame_ts
        frame = None if jpeg is None else cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(502, "couldn't get a still from the camera (its snapshot, the video service and the "
                                     "worker's last frame all failed); check the camera's network, then Refresh snapshot")
        key = c.camera.id if c.camera else ""
        mine = sorted((k for k, v in snapshots.items() if v["cam"] == key), key=lambda k: snapshots[k]["ts"])
        for old in mine[:max(0, len(mine) - SNAPSHOT_KEEP + 1)]:
            snapshots.pop(old, None)
        sid = secrets.token_hex(16)
        snapshots[sid] = still = {"cam": key, "jpeg": jpeg, "frame": frame, "ts": ts, "source": source, "dets": None,
                                  "ver": None, "task": None}
        return sid, still

    @app.post("/api/fixed-slots/snapshot", dependencies=view)
    async def take_snapshot(cam: str | None = None):
        """The slot editor's still (read-only, viewer token): one frame of the camera captured now and kept here, so
        drawing and Test slots work on a frozen image, whatever the video stream does. Sources, first that answers:
        the camera's own HTTP snapshot (Hikvision ISAPI, no video stream involved), go2rtc's snapshot of the native /
        HD stream, the worker's last published frame (<= SNAPSHOT_RAW_AGE old). -> {"id", "ts", "source" (camera /
        go2rtc / worker), "width", "height", "image" (data URL)}. 502 when none answers. In Fixed mode the vehicle
        detection on it starts at once in the background, so Test slots answers without waiting for the model."""
        c = ctx(cam)
        sid, still = await capture_still(c)
        jpeg, source, ts = still["jpeg"], still["source"], still["ts"]
        settings, ver = current(c)
        if settings["run"].get("fixed_mode"):  # Fixed mode tests slots: detect now, so the first Test is (near) instant
            start_detection(still, settings["run"], ver)
        return {"id": sid, "ts": ts, "source": source, "width": still["frame"].shape[1],
                "height": still["frame"].shape[0], "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}

    def start_detection(still: dict, run: dict, ver: int):
        """Detect the vehicles on a snapshot in a thread (the camera's own detector, settings version ver), once;
        Test slots awaits the running job instead of starting its own."""
        if still["task"] is None or still["task"][0] != ver:
            task = asyncio.ensure_future(asyncio.to_thread(slot_preview.detect_on_demand, run, still["frame"], ROOT))
            task.add_done_callback(lambda t: t.cancelled() or t.exception())  # a failure is reported by Test slots
            still["task"] = (ver, task)

    async def camera_still(camera: Camera) -> bytes | None:
        """The camera's own JPEG snapshot over HTTP (Hikvision ISAPI /ISAPI/Streaming/channels/<ch>/picture, digest
        auth with the RTSP URL's credentials, the channel from its path; ~0.7 s, independent of any video stream).
        None for another make or when it doesn't answer. The URL and credentials are never logged."""
        u = urlsplit(camera.rtsp_url or "")
        if not u.hostname:
            return None
        m = re.search(r"/channels/(\d+)", u.path or "", re.I)
        url = f"http://{u.hostname}/ISAPI/Streaming/channels/{m.group(1) if m else '101'}/picture"
        try:  # the app's HTTP client (go2rtc's): one place to mock in tests
            r = await go2rtc.get(url, auth=httpx.DigestAuth(unquote(u.username or ""), unquote(u.password or "")),
                                 timeout=6)
        except (httpx.HTTPError, ValueError):
            return None
        return r.content if r.status_code == 200 and r.content[:2] == b"\xff\xd8" else None

    @app.post("/api/fixed-slots/preview", dependencies=view)
    async def preview_fixed_slots(body: SlotPreviewIn, cam: str | None = None):
        """Test slots (read-only, the viewer token is enough): the drawn slots against the vehicles in a frame, with
        Fixed mode's matching under this camera's match rule (carparking.slot_preview).
        With "snapshot" (the editor's still, POST /api/fixed-slots/snapshot): the vehicles detected on that still by
        the camera's own detector, once per still and settings version, then reused ("source": "snapshot"); no
        video stream or worker involved. 410 when the still is gone (the API restarted): take a new one.
        Without: while the worker publishes fresh detections they are used ("source": "worker"); otherwise one frame
        is grabbed from the camera now (go2rtc's snapshot of its native stream) and run through the camera's own
        detector here ("source": "on-demand", the frame returned as "image" so the editor shows what was tested).
        Stores nothing. -> {"results": [{"id", "name", "state" FREE/BUSY, "score", "vehicle"}], "vehicles": [boxes],
        "frame_ts", "frame_age", "rule", "match_iou", "source", "image" (data URL or None)}. 422 for a bad slot (the
        same check as a save); 502 when no frame can be had from the camera."""
        c = ctx(cam)
        settings, ver = current(c)
        run = settings["run"]
        m5 = run.get("m5") or {}
        rule, min_iou = m5.get("match") or "ground", float(m5.get("match_iou") or 0.15)
        try:
            slot_preview.preview_slots(body.slots, [], rule, min_iou, body.only)  # a bad slot: 422 before any work
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        image = None
        if body.snapshot is not None:
            still = snapshots.get(body.snapshot)
            if still is None or still["cam"] != (c.camera.id if c.camera else ""):
                raise HTTPException(410, "this snapshot is no longer kept (the server restarted): Refresh snapshot")
            if still["dets"] is None or still["ver"] != ver:
                start_detection(still, run, ver)  # already running since the snapshot was taken (Fixed mode), or now
                try:
                    still["dets"] = await asyncio.shield(still["task"][1])
                except Exception as e:  # a model that fails to load / run: say so, the editor shows it
                    still["task"] = None  # the next Test tries again
                    raise HTTPException(500, f"detection failed: {e}") from None
                still["ver"] = ver
            dets, frame_ts, source = still["dets"], still["ts"], "snapshot"
        else:
            with opened(LiveStore, c.db) as store:
                state = store.state(images=False)
            snap = state.get("snapshot") or {}
            age, frame_ts = heartbeat_age(state), snap.get("frame_ts")
            if ("detections" in snap and age is not None and age <= STALE and frame_ts is not None
                    and time.time() - frame_ts <= PREVIEW_FRESH):
                dets, source = snap["detections"], "worker"
            else:
                camera = c.camera or registry.primary()
                jpeg = await grab_frame(camera) if camera is not None else None
                frame = None if jpeg is None else cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise HTTPException(502, "couldn't get a frame from the camera (video service or camera unreachable)")
                try:
                    dets = await asyncio.to_thread(slot_preview.detect_on_demand, run, frame, ROOT)
                except Exception as e:  # a model that fails to load / run: say so, the editor shows it
                    raise HTTPException(500, f"detection failed: {e}") from None
                frame_ts, source = time.time(), "on-demand"
                image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        results = slot_preview.preview_slots(body.slots, dets, rule, min_iou, body.only)
        return {"results": results, "rule": rule, "match_iou": min_iou, "frame_ts": frame_ts, "source": source,
                "frame_age": None if frame_ts is None else round(time.time() - frame_ts, 1), "image": image,
                "vehicles": [d["box"] for d in dets if d.get("kind", "vehicle") == "vehicle"]}

    async def grab_frame(camera: Camera) -> bytes | None:
        """One JPEG of the camera now, from go2rtc (its snapshot of the native stream, else the HD one); None if
        neither answers."""
        if not camera.primary:
            await register_streams(camera)  # go2rtc restarted: the runtime streams are back first
        for kind in ("native", "hd"):
            stream = streams_of(camera).get(kind)
            if not stream:
                continue
            try:
                r = await go2rtc.get(f"{cfg.go2rtc}/api/frame.jpeg", params={"src": stream[0]}, timeout=20)
            except httpx.HTTPError:
                continue
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                return r.content
        return None

    @app.post("/api/commands", dependencies=admin)
    def command(body: Command, cam: str | None = None):
        c = ctx(cam)
        if body.action == "close_stale":  # open sessions of the camera's other scopes (an earlier detector / grid version):
            # nothing will ever close them. The running worker only writes its own scope, so this is safe beside it.
            with contextlib.closing(history_db(c)) as db:
                found = db.close_stale("closed by the admin: not the scope in use", dry_run=body.dry_run)
            return {"done": True, "action": "close_stale", "dry_run": body.dry_run, "sessions": sum(found.values()),
                    "scopes": found}
        with opened(LiveStore, c.db) as store:
            if (heartbeat_age(store.state(images=False)) or STALE + 1) <= STALE:
                return {"queued": store.command(body.action), "action": body.action}
        if body.action == "reset":  # no worker holds the map in memory: reset the stored one directly, like the page
            with contextlib.closing(history_db(c)) as db:
                db.reset()
            return {"done": True, "action": "reset"}
        raise HTTPException(409, f"no worker is running: {body.action} needs one")

    # --- history & analytics: PostgreSQL while its copy is fresh, else the camera's SQLite file ------------------
    def scope_of(c: CamCtx) -> Scope:
        run, detector = history_detector(c)
        return Scope(c.camera.id if c.camera else "", run["camera"], detector, c.tz)

    def central(c: CamCtx) -> PgHistory | None:
        return pg if pg is not None and c.camera is not None and pg.fresh(c.camera.id) else None

    def served(c: CamCtx, from_pg, from_sqlite):
        """(result, source): PostgreSQL if fresh and answering, else SQLite (the same data, at most seconds newer)."""
        store = central(c)
        if store is not None:
            try:
                return from_pg(store, scope_of(c)), "postgres"
            except Exception as e:  # PostgreSQL went away mid-request: the SQLite file has it all
                print(f"api: history from PostgreSQL failed ({e}); using SQLite", flush=True)
        return from_sqlite(), "sqlite"

    def json_from(result, source: str):
        return JSONResponse(result, headers={"X-History-Source": source, **NO_STORE})

    @app.get("/api/sessions", dependencies=view)
    def sessions(limit: int = Query(50, ge=1, le=1000), open_only: bool = False, cam: str | None = None):
        c = ctx(cam)

        def lite():
            with contextlib.closing(history_db(c)) as db:
                return db.sessions(limit, open_only)
        return json_from(*served(c, lambda st, s: st.sessions(s, limit, open_only), lite))

    @app.get("/api/sessions/page", dependencies=view)
    def sessions_page(page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=200), all_: bool = Query(False, alias="all"),
                      cam: str | None = None):
        """The History table, one page at a time: the open sessions (still parked: the live BUSY slots) first, then
        newest first. -> {"items", "total", "open", "page", "size", "pages"}. ?all=1: every scope of the camera (each detector /
        auto grid version it ran with), each item with its "scope"."""
        c = ctx(cam)
        offset = (page - 1) * size

        def lite():
            with contextlib.closing(history_db(c)) as db:
                return db.sessions_page(offset, size, all_)
        out, source = served(c, lambda st, s: st.sessions_page(s, offset, size, all_), lite)
        return json_from({**out, "page": page, "size": size, "pages": max(1, -(-out["total"] // size))}, source)

    KINDS = {"moves": MOVES, "notices": NOTICES, "all": None}  # /api/events?kinds=

    @app.get("/api/events", dependencies=view)
    def events(since_id: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000), cam: str | None = None,
               kinds: str = Query("moves", pattern="^(moves|notices|all)$")):
        """The event log, newest first. kinds=moves (the default, what older clients expect): entries and exits, from PostgreSQL when it
        is fresh. notices: the encroached / encroach_cleared / rejected audit notices; all: both. Notices live in the camera's SQLite
        file only (PostgreSQL mirrors vehicle movements), so those kinds are always read from it."""
        c = ctx(cam)

        def lite():
            with contextlib.closing(history_db(c)) as db:
                return db.events(limit, since_id, KINDS[kinds])
        if kinds != "moves":
            return json_from(lite(), "sqlite")
        return json_from(*served(c, lambda st, s: st.events(s, limit, since_id), lite))

    @app.get("/api/summary", dependencies=view)
    def summary(cam: str | None = None):
        c = ctx(cam)

        def lite():
            with contextlib.closing(history_db(c)) as db:
                return db.summary()
        out, source = served(c, lambda st, s: st.summary(s), lite)
        return json_from({**out, "source": source}, source)

    @app.get("/api/sessions.csv", dependencies=view)
    def sessions_csv(cam: str | None = None):
        c = ctx(cam)

        def lite():
            with contextlib.closing(history_db(c)) as db:
                return db.csv()
        text, source = served(c, lambda st, s: st.sessions_csv(s), lite)
        return Response(text, media_type="text/csv", headers={
            "Content-Disposition": "attachment; filename=parking_sessions.csv", "X-History-Source": source, **NO_STORE})

    @app.get("/api/events.csv", dependencies=view)
    def events_csv(cam: str | None = None, kinds: str = Query("moves", pattern="^(moves|notices|all)$")):
        c = ctx(cam)

        def lite():
            with contextlib.closing(history_db(c)) as db:
                return db.events_csv(KINDS[kinds])
        text, source = (lite(), "sqlite") if kinds != "moves" else served(c, lambda st, s: st.events_csv(s), lite)
        return Response(text, media_type="text/csv", headers={
            "Content-Disposition": "attachment; filename=parking_events.csv", "X-History-Source": source, **NO_STORE})

    @app.get("/api/analytics", dependencies=view)
    def analytics(days: float = Query(7, gt=0, le=3650), scope: str = Query("all", pattern="^(all|current)$"),
                  cam: str | None = None):
        """One camera over the last `days`: totals (entries, exits, sessions, average / median stay, busiest hour,
        peak occupancy), per hour of day and per day (camera's time zone), stay lengths, occupancy over time.
        scope=all: the camera's whole history (every detector / slot map it ran with); current: the running one."""
        c = ctx(cam)
        every = scope == "all"
        out, source = served(c, lambda st, s: st.analytics(s, days, every),
                             lambda: SqliteHistory(c.db).analytics(scope_of(c), days, every))
        return JSONResponse({**out, "source": source, "scope": scope, "cam": c.camera.id if c.camera else None}, headers=NO_STORE)

    @app.get("/api/analytics/overview", dependencies=view)
    def overview(days: float = Query(1, gt=0, le=3650)):
        """Every camera at once: entries / exits in the window, open sessions, its state now."""
        cams = registry.list()
        ctxs = {cam.id: ctx_of(cam) for cam in cams}
        rows, source = None, "sqlite"
        if pg is not None and cams and all(pg.fresh(cid) for cid in ctxs):
            try:
                rows, source = pg.overview(list(ctxs), cfg.tz, days), "postgres"
            except Exception as e:
                print(f"api: overview from PostgreSQL failed ({e}); using SQLite", flush=True)
        out = []
        for cam in cams:
            c = ctxs[cam.id]
            st = hub_of(c).status_message()
            snap = st["snapshot"] or {}
            counts = snap.get("counts") or {}
            if rows is not None:
                r = rows[cam.id]
            else:
                t = SqliteHistory(c.db).analytics(scope_of(c), days)["totals"]
                r = {"entries": t["entries"], "exits": t["exits"], "open": t["open"]}
            out.append({"id": cam.id, "display": cam.display, "entries": r["entries"], "exits": r["exits"],
                        "open": r["open"], "online": st["connected"], "monitoring": bool(snap.get("enabled")),
                        "free": counts.get("free"), "busy": counts.get("busy"), "total": counts.get("slots")})
        return JSONResponse({"days": days, "source": source, "cameras": out}, headers=NO_STORE)

    @app.get("/api/camera-events", dependencies=view)
    def camera_events(limit: int = Query(100, ge=1, le=1000), cam: str | None = None):
        """Worker online / offline, monitoring on / off, slot map changes (recorded by the sync: PostgreSQL only)."""
        c = ctx(cam)
        if pg is None or c.camera is None:
            return json_from([], "sqlite")
        try:
            return json_from(pg.camera_events(c.camera.id, c.tz, limit), "postgres")
        except Exception:
            return json_from([], "sqlite")

    @app.get("/api/history/sync", dependencies=view)
    def history_sync():
        """Where History & Analytics come from and how far the PostgreSQL copy is, per camera, and "live": the running worker's
        {"expected" (vehicles counted BUSY), "open" (open sessions), "mismatch_s" (how long they have differed), "error"}."""
        status, error = {}, None
        if pg is not None:
            try:
                status = pg.sync_status()
            except Exception as e:
                error = str(e)
        out = []
        for cam in registry.list():
            c = ctx_of(cam)
            st = status.get(cam.id)
            src = sqlite_counts(c.db)
            behind = {k: max(0, src[k] - (st[f"{k}_id"] if st else 0)) for k in ("sessions", "events")}
            live = hub_of(c).status_message()  # the worker's own check of its history against the live count (None: no worker)
            out.append({"id": cam.id, "display": cam.display, "sqlite": src, "behind": behind,
                        "live": (live["snapshot"] or {}).get("sessions") if live["connected"] else None,
                        "rows": st["rows"] if st else None, "age_s": st["age_s"] if st else None,
                        "error": st["error"] if st else "", "source": "postgres" if central(c) else "sqlite"})
        return JSONResponse({"backend": "postgres" if pg is not None else "sqlite", "error": error, "cameras": out}, headers=NO_STORE)

    # --- cameras ------------------------------------------------------------------------------------------------
    def camera_out(camera: Camera) -> dict:
        c = ctx_of(camera)
        st = hub_of(c).status_message()
        snap = st["snapshot"]
        with opened(LiveStore, c.db) as store:
            stored, _ = store.config()
        run = (stored or {}).get("run") or {}
        streams = {k: (v[0] if v else None) for k, v in streams_of(camera).items()}
        return {**camera.public(), "streams": streams,
                "status": {"connected": st["connected"], "heartbeat_age": st["heartbeat_age"],
                           "enabled": bool((stored or {}).get("enabled")), "running": bool(snap.get("enabled")),
                           "slot_mode": "fixed" if run.get("fixed_mode") else "dynamic",
                           "counts": snap.get("counts"), "state": snap.get("status")}}

    @app.get("/api/cameras", dependencies=view)
    def cameras():
        """Every camera with its streams and live status (URLs masked)."""
        return [camera_out(x) for x in registry.list()]

    @app.post("/api/cameras", dependencies=admin, status_code=201)
    async def add_camera(body: CameraIn):
        """A new camera: registered, its own database initialised (monitoring off, Dynamic mode, the defaults), its
        video streams set up; the worker supervisor starts its worker within seconds."""
        try:
            camera = registry.create(Camera(body.id or slugify(body.name), body.name.strip(), body.rtsp_url,
                                            body.rtsp_sub_url or "", sort_order=body.sort_order or 0,
                                            metadata=body.metadata or {}, label=(body.label or "").strip()))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        c = ctx_of(camera)
        c.db.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(write_settings, c, {"camera5": camera.name, "enabled5": False})
        await register_streams(camera)
        return await asyncio.to_thread(camera_out, camera)

    @app.patch("/api/cameras/{cam_id}", dependencies=admin)
    async def edit_camera(cam_id: str, body: CameraPatch):
        """Display name (label), URLs (a masked one sent back keeps the stored one), order, metadata. Its worker
        restarts with a new URL. The name (history key) and id never change."""
        try:
            camera = registry.update(cam_id, **body.model_dump(exclude_unset=True))
        except KeyError:
            raise HTTPException(404, f"no camera {cam_id!r}") from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        await register_streams(camera, force=True)
        return await asyncio.to_thread(camera_out, camera)

    @app.delete("/api/cameras/{cam_id}", dependencies=admin)
    async def remove_camera(cam_id: str):
        """Unregister a camera (its worker stops; its database and history files are kept on disk)."""
        camera = registry.get(cam_id)
        try:
            if camera is None or not registry.delete(cam_id):
                raise HTTPException(404, f"no camera {cam_id!r}")
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        for kind in streams_of(camera).values():
            if kind:
                with contextlib.suppress(httpx.HTTPError):
                    await go2rtc.delete(f"{cfg.go2rtc}/api/streams", params={"src": kind[0]})
                registered.discard(kind[0])
        return {"removed": cam_id}

    # --- video: go2rtc streams per camera, WebRTC signalling proxied ---------------------------------------------
    # 30 s: a stream nobody watched starts cold (go2rtc launches ffmpeg on the HEVC camera, several seconds)
    go2rtc = http or httpx.AsyncClient(timeout=30)
    registered: set[str] = set()  # streams this process has registered in go2rtc (the primary's are in go2rtc.yaml)

    async def register_streams(camera: Camera, force: bool = False):
        if camera.primary:
            return
        for kind in streams_of(camera).values():
            if kind and (force or kind[0] not in registered):
                name, src = kind
                try:
                    r = await go2rtc.put(f"{cfg.go2rtc}/api/streams", params={"name": name, "src": src})
                    if r.status_code < 300:
                        registered.add(name)
                except httpx.HTTPError:
                    pass  # go2rtc not up yet: registered on the first play instead

    def stream_owner(src: str) -> Camera | None:
        return next((x for x in registry.list() if any(k and k[0] == src for k in streams_of(x).values())), None)

    @app.get("/api/video", dependencies=view)
    def video(cam: str | None = None):
        """The camera's video: WebRTC through POST /api/webrtc?src= (go2rtc itself is never published). hd: 1080p
        H.264 (fullscreen), sd: the sub-stream as H.264 (grid tiles; null without a sub-stream), native: as sent."""
        c = ctx(cam)
        camera = c.camera or Camera("camera", "Camera", c.rtsp, primary=True)
        streams = {k: (v[0] if v else None) for k, v in streams_of(camera).items()}
        return {"webrtc": "/api/webrtc", "streams": streams, "default": streams["hd"], "grid": streams["sd"] or streams["hd"]}

    @app.post("/api/webrtc", dependencies=view)
    async def webrtc(request: Request, src: str = "parking_hd"):
        """WebRTC signalling for the browser's player: its SDP offer -> go2rtc -> the SDP answer (WHEP style, ICE
        candidates inside the SDP). The media then flows browser <-> go2rtc on port 8555 directly."""
        owner = stream_owner(src) if src not in PRIMARY_STREAMS else None
        if src not in PRIMARY_STREAMS and owner is None:
            raise HTTPException(404, f"unknown stream {src!r}")
        body = await request.body()
        headers = {"Content-Type": request.headers.get("content-type", "application/sdp")}
        for attempt in range(2):
            if owner is not None:
                await register_streams(owner, force=attempt > 0)  # go2rtc restarted: its runtime streams are gone
            try:
                r = await go2rtc.post(f"{cfg.go2rtc}/api/webrtc", params={"src": src}, content=body, headers=headers)
            except httpx.HTTPError as e:
                raise HTTPException(502, f"video service unreachable: {e}") from None
            if r.status_code < 400 or owner is None:
                break
        return Response(r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/sdp"))

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket, frames: bool = False, token: str = "", cam: str | None = None,
                     every: float = Query(0, ge=0, le=300)):
        """Live state for the dashboard. every: seconds between routine state messages for this client (0 = each
        worker heartbeat, about 1 s); a change of the counts, a slot's state, the connection, monitoring or the
        settings is sent at once anyway, and so are entries / exits and frames (frames=1)."""
        await ws.accept()  # accept first: a close before it reaches a browser only as a bare 403 handshake error
        if cfg.view_token and not _matches(token, cfg.view_token, cfg.token):
            await ws.close(code=4401)  # the client asks for a token and reconnects with ?token=
            return
        try:
            c = ctx(cam)
        except HTTPException:
            await ws.close(code=4404)
            return
        hub, key = hub_of(c), c.camera.id if c.camera else ""
        if key not in hub_tasks or hub_tasks[key].done():
            hub_tasks[key] = asyncio.create_task(hub.run())
        q = hub.join(frames)

        async def pump():
            first = await asyncio.to_thread(hub.status_message)
            await ws.send_json(first)
            last_key, last_at = state_key(first), time.monotonic()
            while True:
                msg = await q.get()
                if isinstance(msg, bytes):
                    await ws.send_bytes(msg)
                    continue
                if every and msg.get("type") == "state":
                    key = state_key(msg)
                    if key == last_key and time.monotonic() - last_at < every:
                        continue  # nothing that matters changed: the next one after the interval goes
                    last_key, last_at = key, time.monotonic()
                await ws.send_json(msg)

        sender = asyncio.create_task(pump())
        try:
            while True:  # notices a closed socket even when nothing is being sent
                await ws.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sender.cancel()
            hub.leave(q)

    # Phase 2 camera setup endpoints: 503 unless CARPARK_GEOMETRY_ENABLED; the models are in the `setup` service
    setup_client = setup_http or httpx.AsyncClient(timeout=30)
    register_setup(app, cfg=cfg, ctx=ctx, current=current, snapshots=snapshots, capture_still=capture_still,
                   start_detection=start_detection, admin=admin, setup_http=setup_client,
                   root=ROOT, view=view, registry=registry, save_fixed=save_fixed)
    # a random picture as its own throw-away camera (calibrate from pins or GeoCalib, review the grid, run the stack): in memory only
    register_sample_sessions(app, cfg=cfg, ctx=ctx, current=current, admin=admin, root=ROOT, setup_http=setup_client)
    mount_spa(app, cfg.spa)
    return app


def mount_spa(app: FastAPI, dist: Path):
    """The built React SPA (frontend/dist) at /: its files, and index.html for every other non-API path (client-side
    routes). Not built yet: a short page saying so."""
    index = dist / "index.html"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith(("api/", "ws/")):
            raise HTTPException(404)
        f = (dist / path).resolve()
        if path and f.is_file() and f.is_relative_to(dist.resolve()):
            return FileResponse(f)
        if index.is_file():  # never cached: a new build reaches every browser on the next load (the assets are hashed)
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Parking monitor API</h1><p>The web app is not built yet (frontend/dist). "
                            "API docs: <a href='/docs'>/docs</a>.</p>")
