"""Camera registry: which cameras exist, how to reach them, and their fixed slot coordinates.

Backends (open_registry): PostgreSQL when CARPARK_DATABASE_URL is set (the compose service "db"), else a cameras.json
file (tests, or a host without the database). Both store the same thing:
  cameras        id (slug, the key in URLs), name (the history scope and slot-map key, never renamed), label (the
                 display name, editable any time; empty = name), the RTSP
                 main / sub-stream URLs (secrets: never returned by the API), primary (the camera the Streamlit page
                 shows; exactly one), sort order, free metadata (location, model, notes ...)
  camera_slots   each camera's fixed slot map: [{"id", "name", "poly": [[x, y], ...] shares of the frame}]
What a camera does at run time (its settings incl. monitoring on/off and Fixed / Dynamic mode, its heartbeat and
images, its entry/exit history) stays in its own SQLite file (runtime_paths), one per camera, so cameras are isolated
and the existing worker runs unchanged, one process per camera (scripts/supervisor.py). The primary camera keeps the
original files (parking_history.db, config/fixed_slots.json): the Streamlit page and the history continue as they were.
The PostgreSQL schema is plain SQL (Supabase is PostgreSQL), so a later Supabase install can use it as is.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from carparking.geometry_store import HISTORY_MIGRATION as GEOMETRY_HISTORY_MIGRATION
from carparking.geometry_store import MIGRATION as GEOMETRY_MIGRATION
from carparking.history_pg import MIGRATION as HISTORY_MIGRATION
from carparking.fixed_slots import FixedSlotStore, validate_slots
from carparking.rtsp import mask_url

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

MIGRATIONS = [
    ("001_cameras", """
        CREATE TABLE IF NOT EXISTS cameras (
            id           TEXT PRIMARY KEY CHECK (id ~ '^[a-z0-9][a-z0-9-]{0,39}$'),
            name         TEXT NOT NULL UNIQUE,
            rtsp_url     TEXT NOT NULL,
            rtsp_sub_url TEXT NOT NULL DEFAULT '',
            is_primary   BOOLEAN NOT NULL DEFAULT FALSE,
            sort_order   INTEGER NOT NULL DEFAULT 0,
            metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE UNIQUE INDEX IF NOT EXISTS cameras_one_primary ON cameras (is_primary) WHERE is_primary;
        CREATE TABLE IF NOT EXISTS camera_slots (
            camera_id  TEXT PRIMARY KEY REFERENCES cameras (id) ON DELETE CASCADE,
            slots      JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),
    # the display name: editable any time ("East Gate"); name stays the camera's history key
    ("002_camera_label", "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS label TEXT NOT NULL DEFAULT ''"),
    # Step 4: the history of every camera, copied from its SQLite file (carparking/history_pg.py)
    HISTORY_MIGRATION,
    # Phase 2: versioned geometry per camera (carparking/geometry_store.py); empty until a setup is accepted
    GEOMETRY_MIGRATION,
    # Phase 2: calibration lifecycle events + occupancy samples (carparking/geometry_store.py), empty until the flag is on
    GEOMETRY_HISTORY_MIGRATION,
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40].strip("-")
    return s or "camera"


@dataclass
class Camera:
    id: str
    name: str
    rtsp_url: str
    rtsp_sub_url: str = ""
    primary: bool = False
    sort_order: int = 0
    metadata: dict = field(default_factory=dict)
    created: str = ""
    updated: str = ""
    label: str = ""  # display name, editable; empty = name

    @property
    def display(self) -> str:
        return self.label.strip() or self.name

    def public(self) -> dict:
        """For the API: the URLs masked (rtsp://***@host/...), never the passwords; + display (label or name)."""
        d = asdict(self)
        d["display"] = self.display
        d["rtsp_url"], d["rtsp_sub_url"] = mask_url(self.rtsp_url), mask_url(self.rtsp_sub_url) if self.rtsp_sub_url else ""
        return d


EDITABLE = ("label", "rtsp_url", "rtsp_sub_url", "sort_order", "metadata")  # id, name, primary never change


class Registry:
    """Common checks; the backends store."""

    def list(self) -> list[Camera]:
        raise NotImplementedError

    def get(self, cam_id: str) -> Camera | None:
        return next((c for c in self.list() if c.id == cam_id), None)

    def primary(self) -> Camera | None:
        return next((c for c in self.list() if c.primary), None)

    def _check_new(self, cam: Camera):
        if not SLUG.match(cam.id):
            raise ValueError(f"id {cam.id!r}: lowercase letters, digits and '-' only (max 40)")
        if not cam.name.strip():
            raise ValueError("name: required")
        others = self.list()
        if any(c.id == cam.id for c in others):
            raise ValueError(f"id {cam.id!r} exists already")
        if any(c.name.strip().lower() == cam.name.strip().lower() for c in others):
            raise ValueError(f"name {cam.name!r} exists already")
        if cam.primary and any(c.primary for c in others):
            raise ValueError("there is a primary camera already")
        check_url(cam.rtsp_url, "rtsp_url")
        if cam.rtsp_sub_url:
            check_url(cam.rtsp_sub_url, "rtsp_sub_url")

    def create(self, cam: Camera) -> Camera:
        raise NotImplementedError

    def update(self, cam_id: str, **changes) -> Camera:
        raise NotImplementedError

    def delete(self, cam_id: str) -> bool:
        raise NotImplementedError

    def slots(self, cam_id: str) -> dict:
        """{"updated": iso or None, "slots": [...]}"""
        raise NotImplementedError

    def set_slots(self, cam_id: str, slots: list[dict]) -> list[dict]:
        raise NotImplementedError

    def close(self):
        pass


def check_url(url: str, what: str):
    if not (url.startswith(("rtsp://", "rtsps://", "http://", "https://")) or url.isdigit()):
        raise ValueError(f"{what}: an rtsp:// URL (or a webcam index) is required")


def _changes(cam: Camera, changes: dict) -> dict:
    out = {k: v for k, v in changes.items() if k in EDITABLE and v is not None}
    if "label" in out:
        out["label"] = str(out["label"]).strip()[:80]
    for k in ("rtsp_url", "rtsp_sub_url"):  # a masked URL sent back unchanged keeps the stored one
        if k in out and "***" in out[k]:
            del out[k]
    if "rtsp_url" in out:
        check_url(out["rtsp_url"], "rtsp_url")
    if out.get("rtsp_sub_url"):
        check_url(out["rtsp_sub_url"], "rtsp_sub_url")
    return out


class JsonRegistry(Registry):
    """cameras.json: {"version": 1, "cameras": [...], "slots": {id: {"updated", "slots"}}}; atomic writes."""

    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "cameras": [], "slots": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.setdefault("cameras", [])
        data.setdefault("slots", {})
        return data

    def _write(self, data: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def list(self) -> list[Camera]:
        cams = [Camera(**c) for c in self._read()["cameras"]]
        return sorted(cams, key=lambda c: (not c.primary, c.sort_order, c.name.lower()))

    def create(self, cam: Camera) -> Camera:
        with self.lock:
            self._check_new(cam)
            cam.created = cam.updated = now_iso()
            data = self._read()
            data["cameras"].append(asdict(cam))
            self._write(data)
        return cam

    def update(self, cam_id: str, **changes) -> Camera:
        with self.lock:
            data = self._read()
            for c in data["cameras"]:
                if c["id"] == cam_id:
                    c.update(_changes(Camera(**c), changes), updated=now_iso())
                    self._write(data)
                    return Camera(**c)
        raise KeyError(cam_id)

    def delete(self, cam_id: str) -> bool:
        with self.lock:
            data = self._read()
            cam = next((c for c in data["cameras"] if c["id"] == cam_id), None)
            if cam is None:
                return False
            if cam.get("primary"):
                raise ValueError("the primary camera can't be removed (the Streamlit page and its history use it)")
            data["cameras"] = [c for c in data["cameras"] if c["id"] != cam_id]
            data["slots"].pop(cam_id, None)
            self._write(data)
            return True

    def slots(self, cam_id: str) -> dict:
        entry = self._read()["slots"].get(cam_id, {})
        return {"updated": entry.get("updated"), "slots": validate_slots(entry.get("slots", []))}

    def set_slots(self, cam_id: str, slots: list[dict], updated: str | None = None) -> list[dict]:
        clean = validate_slots(slots)
        with self.lock:
            data = self._read()
            if all(c["id"] != cam_id for c in data["cameras"]):
                raise KeyError(cam_id)
            data["slots"][cam_id] = {"updated": updated or now_iso(), "slots": clean}
            self._write(data)
        return clean


class PgRegistry(Registry):
    """PostgreSQL (psycopg 3). A short connection per call: the API runs requests on a thread pool, and the
    supervisor polls; both are light. Migrations (MIGRATIONS) run once, tracked in schema_migrations."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.migrate()

    def _conn(self):
        import psycopg
        return psycopg.connect(self.dsn, autocommit=True)

    def migrate(self):
        # the API and the supervisor start together: concurrent CREATE TABLE IF NOT EXISTS can still collide in
        # PostgreSQL's catalog (UniqueViolation on pg_type), so one process migrates at a time (advisory lock)
        with self._conn() as con:
            con.execute("SELECT pg_advisory_lock(727274)")
            try:
                con.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, "
                            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
                done = {r[0] for r in con.execute("SELECT version FROM schema_migrations")}
                for version, sql in MIGRATIONS:
                    if version not in done:
                        with con.transaction():
                            con.execute(sql)
                            con.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            finally:
                con.execute("SELECT pg_advisory_unlock(727274)")

    @staticmethod
    def _cam(row) -> Camera:
        cid, name, url, sub, primary, order, meta, created, updated, label = row
        return Camera(cid, name, url, sub, primary, order, meta or {}, created.isoformat(timespec="seconds"),
                      updated.isoformat(timespec="seconds"), label)

    _COLS = "id, name, rtsp_url, rtsp_sub_url, is_primary, sort_order, metadata, created_at, updated_at, label"

    def list(self) -> list[Camera]:
        with self._conn() as con:
            rows = con.execute(f"SELECT {self._COLS} FROM cameras ORDER BY is_primary DESC, sort_order, lower(name)")
            return [self._cam(r) for r in rows]

    def create(self, cam: Camera) -> Camera:
        from psycopg.types.json import Jsonb
        self._check_new(cam)
        with self._conn() as con:
            row = con.execute(f"INSERT INTO cameras (id, name, rtsp_url, rtsp_sub_url, is_primary, sort_order, metadata, "
                              f"label) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING {self._COLS}",
                              (cam.id, cam.name.strip(), cam.rtsp_url, cam.rtsp_sub_url, cam.primary, cam.sort_order,
                               Jsonb(cam.metadata), cam.label.strip())).fetchone()
        return self._cam(row)

    def update(self, cam_id: str, **changes) -> Camera:
        from psycopg.types.json import Jsonb
        cam = self.get(cam_id)
        if cam is None:
            raise KeyError(cam_id)
        out = _changes(cam, changes)
        if out:
            sets = ", ".join(f"{k} = %s" for k in out)
            vals = [Jsonb(v) if k == "metadata" else v for k, v in out.items()]
            with self._conn() as con:
                con.execute(f"UPDATE cameras SET {sets}, updated_at = now() WHERE id = %s", (*vals, cam_id))
        return self.get(cam_id)

    def delete(self, cam_id: str) -> bool:
        cam = self.get(cam_id)
        if cam is None:
            return False
        if cam.primary:
            raise ValueError("the primary camera can't be removed (the Streamlit page and its history use it)")
        with self._conn() as con:
            con.execute("DELETE FROM cameras WHERE id = %s", (cam_id,))
        return True

    def slots(self, cam_id: str) -> dict:
        with self._conn() as con:
            row = con.execute("SELECT slots, updated_at FROM camera_slots WHERE camera_id = %s", (cam_id,)).fetchone()
        if row is None:
            return {"updated": None, "slots": []}
        return {"updated": row[1].isoformat(timespec="seconds"), "slots": validate_slots(row[0])}

    def set_slots(self, cam_id: str, slots: list[dict], updated: str | None = None) -> list[dict]:
        from psycopg.types.json import Jsonb
        clean = validate_slots(slots)
        if self.get(cam_id) is None:
            raise KeyError(cam_id)
        with self._conn() as con:
            con.execute("INSERT INTO camera_slots (camera_id, slots, updated_at) VALUES (%s, %s, COALESCE(%s, now())) "
                        "ON CONFLICT (camera_id) DO UPDATE SET slots = excluded.slots, updated_at = excluded.updated_at",
                        (cam_id, Jsonb(clean), updated))
        return clean


def open_registry(env=None, data_dir=None) -> Registry:
    """PostgreSQL if CARPARK_DATABASE_URL is set, else CARPARK_CAMERAS_FILE (default data_dir/cameras.json)."""
    env = os.environ if env is None else env
    if env.get("CARPARK_DATABASE_URL"):
        return PgRegistry(env["CARPARK_DATABASE_URL"])
    path = env.get("CARPARK_CAMERAS_FILE") or Path(data_dir or ".") / "cameras.json"
    return JsonRegistry(path)


@dataclass
class Runtime:
    """Where one camera lives at run time: its SQLite file (settings, heartbeat, images, history), its fixed-slot
    file (what the worker's settings read), its raw-frame folder."""
    db: Path
    fixed: Path
    raw: Path


def runtime_paths(cam: Camera, data_dir, primary_db, primary_fixed, primary_raw=None) -> Runtime:
    data_dir = Path(data_dir)
    if cam.primary:  # the original files: Streamlit and the existing history keep working
        return Runtime(Path(primary_db), Path(primary_fixed), Path(primary_raw or data_dir / "dataset_raw"))
    base = data_dir / "cameras" / cam.id
    return Runtime(base / "parking_history.db", base / "fixed_slots.json", base / "dataset_raw")


def streams_of(cam: Camera) -> dict[str, tuple[str, str] | None]:
    """go2rtc streams of a camera: {"hd", "sd", "native"} -> (name, source) or None. hd: the main stream as 1080p
    H.264 (fullscreen); sd: the sub-stream as H.264 (grid tiles: cheap); native: as sent. The primary camera keeps the
    names of go2rtc.yaml."""
    hd = f"ffmpeg:{cam.rtsp_url}#video=h264#width=1920#height=1080"
    sd = f"ffmpeg:{cam.rtsp_sub_url}#video=h264" if cam.rtsp_sub_url else None
    names = ({"hd": "parking_hd", "sd": "parking", "native": "parking_native"} if cam.primary else
             {"hd": f"{cam.id}_hd", "sd": f"{cam.id}_sd", "native": f"{cam.id}_native"})
    return {"hd": (names["hd"], hd), "sd": (names["sd"], sd) if sd else None, "native": (names["native"], cam.rtsp_url)}


def mirror_slots(registry: Registry, cam: Camera, fixed_path, tz=None) -> list[dict]:
    """Write the camera's slot map from the registry into its fixed-slot file (what its worker's settings read)."""
    slots = registry.slots(cam.id)["slots"]
    return FixedSlotStore(fixed_path).save(cam.name, slots, tz)


def sync_primary_slots(registry: Registry, cam: Camera, fixed_path, tz=None) -> str | None:
    """The primary camera's map can also be edited in the Streamlit editor (config/fixed_slots.json): whichever side
    changed last wins -> "imported" (file -> registry), "exported" (registry -> file) or None (in step)."""
    store = FixedSlotStore(fixed_path)
    file = store.get(cam.name)
    reg = registry.slots(cam.id)
    if file["slots"] == reg["slots"]:
        return None
    file_t, reg_t = _ts(file["updated"]), _ts(reg["updated"])
    if file["updated"] and (reg["updated"] is None or file_t > reg_t):
        registry.set_slots(cam.id, file["slots"], file["updated"])
        return "imported"
    store.save(cam.name, reg["slots"], tz)
    return "exported"


def _ts(iso: str | None) -> float:
    if not iso:
        return 0.0
    d = datetime.fromisoformat(iso)
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


def seed(registry: Registry, name: str, rtsp_url: str, rtsp_sub_url: str = "", fixed_path=None) -> Camera | None:
    """First start with an empty registry: the existing single camera becomes the primary one (its slots imported
    from the fixed-slot file). -> the new camera, or None if the registry already has cameras."""
    if registry.list() or not rtsp_url:
        return None
    try:
        cam = registry.create(Camera(slugify(name), name, rtsp_url, rtsp_sub_url, primary=True,
                                     metadata={"seeded_from": ".env"}))
    except Exception:  # the API and the supervisor start together: the other one seeded it first
        if registry.list():
            return None
        raise
    if fixed_path is not None:
        entry = FixedSlotStore(fixed_path).get(name)
        if entry["slots"]:
            registry.set_slots(cam.id, entry["slots"], entry["updated"])
    return cam
