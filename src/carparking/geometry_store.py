"""Versioned geometry records per camera (Phase 2, docs/PHASE2_GEOMETRY_PLAN.md): the SAM surface polygon, the camera
calibration (K, pitch, roll, h, H), the BEV surface and the packed slots, as one JSON record per version.

Lifecycle of a version (state):
    draft       written by the setup job, waiting for the Admin's Accept
    active      the one the worker uses (at most one per camera: a partial unique index enforces it in the database)
    superseded  was active, replaced by a newer activation; kept (the last KEEP_SUPERSEDED) for rollback
    stale       was active when the drift guard saw the camera move; not used, not re-activatable (re-run the setup)
Phase 2 history lives in the same backends: calibration lifecycle EVENTS (draft, activated, stale, discarded: written in the
same transaction as the change, so the log can never disagree with the data) and occupancy SAMPLES (one row per camera per
second at most, written by the worker; bucketed in SQL over the (camera, time) key for fast range queries).
No active version = uncalibrated: active() returns None and the worker behaves exactly as before. Nothing here is
created until a store is opened (LiveStore.geometry() / open_geometry), so existing databases are not touched.

One class, two backends: SQLite (the camera's own history database, next to its live_* tables) and PostgreSQL (table
camera_geometry from MIGRATION, run by PgRegistry). numpy-free on purpose: importing this pulls in nothing heavy."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager

STATES = ("draft", "active", "stale", "superseded")
KEEP_SUPERSEDED = 5

_COLUMNS = "camera_id, ver, state, record, note, created_by, created_at, activated_at"
_TABLE = """CREATE TABLE IF NOT EXISTS camera_geometry (
    camera_id    TEXT NOT NULL{fk},
    ver          INTEGER NOT NULL,
    state        TEXT NOT NULL CHECK (state IN ('draft', 'active', 'stale', 'superseded')),
    record       {json} NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    created_by   TEXT NOT NULL DEFAULT '',
    created_at   {real} NOT NULL,
    activated_at {real},
    PRIMARY KEY (camera_id, ver)
);
CREATE UNIQUE INDEX IF NOT EXISTS camera_geometry_one_active ON camera_geometry (camera_id) WHERE state = 'active';"""

_HISTORY = """CREATE TABLE IF NOT EXISTS geometry_samples (
    camera_id TEXT NOT NULL{fk},
    t         {bigint} NOT NULL,
    ver       INTEGER NOT NULL,
    capacity  INTEGER NOT NULL,
    occupied  INTEGER NOT NULL,
    vehicles  INTEGER NOT NULL,
    measured  INTEGER NOT NULL,
    PRIMARY KEY (camera_id, t)
);
CREATE TABLE IF NOT EXISTS geometry_events (
    id        {pk},
    camera_id TEXT NOT NULL{fk},
    t         {real} NOT NULL,
    kind      TEXT NOT NULL,
    ver       INTEGER,
    detail    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS geometry_events_camera_t ON geometry_events (camera_id, t);"""
_PG = dict(fk=" REFERENCES cameras (id) ON DELETE CASCADE", bigint="BIGINT", real="DOUBLE PRECISION", pk="BIGSERIAL PRIMARY KEY")
_LITE = dict(fk="", bigint="INTEGER", real="REAL", pk="INTEGER PRIMARY KEY AUTOINCREMENT")

# PostgreSQL: run by PgRegistry.migrate (registry.MIGRATIONS); the rows go with their camera
MIGRATION = ("004_camera_geometry", _TABLE.format(fk=" REFERENCES cameras (id) ON DELETE CASCADE", json="JSONB",
                                                    real="DOUBLE PRECISION"))
HISTORY_MIGRATION = ("005_geometry_history", _HISTORY.format(**_PG))
SAMPLE_KEEP_DAYS, EVENT_KEEP_DAYS = 90, 365  # what prune_history() keeps

_CREDENTIALS = re.compile(r"\w+://[^/\s]*@")  # rtsp://user:pass@host


def geometry_enabled(env) -> bool:
    """CARPARK_GEOMETRY_ENABLED (1 / true / yes / on): the one switch for everything Phase 2 does at runtime: the setup
    endpoints, the worker's read of the active calibration and the dashboard's metric layer. Off (the default) = none of it
    runs, nothing is read, nothing is imported. API and worker use this same function, so they can never disagree."""
    return str(env.get("CARPARK_GEOMETRY_ENABLED", "")).strip().lower() in ("1", "true", "yes", "on")


class GeometryError(ValueError):
    """A refused transition or an unknown version (the API maps it to 409 / 404)."""


def _clean(value, path="record"):
    """The record must be JSON data without camera credentials (a URL with user:pass@)."""
    if isinstance(value, str) and _CREDENTIALS.search(value):
        raise GeometryError(f"{path} holds a URL with credentials; geometry records never carry them")
    if isinstance(value, dict):
        for k, v in value.items():
            _clean(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _clean(v, f"{path}[{i}]")


class _Sqlite:
    ph, ins_record, sel_record = "?", "?", "record"

    def __init__(self, con: sqlite3.Connection):
        self.con, self.lock, self.ready = con, threading.RLock(), False

    def setup(self):
        if not self.ready:  # created on first use, never when the store is only constructed
            with self.lock:
                self.con.executescript(_TABLE.format(fk="", json="TEXT", real="REAL") + "\n" + _HISTORY.format(**_LITE))
            self.ready = True

    @contextmanager
    def tx(self):
        self.setup()
        with self.lock:
            self.con.execute("BEGIN IMMEDIATE")
            try:
                yield lambda sql, args=(): self.con.execute(sql, args).fetchall()
            except BaseException:
                self.con.execute("ROLLBACK")
                raise
            self.con.execute("COMMIT")


class _Pg:
    ph, ins_record, sel_record = "%s", "CAST(%s AS jsonb)", "record::text"

    def __init__(self, dsn: str):
        self.dsn = dsn

    def setup(self):
        pass  # PgRegistry's migration creates the table

    @contextmanager
    def tx(self):
        import psycopg
        with psycopg.connect(self.dsn, autocommit=True) as con, con.transaction():
            def run(sql, args=()):
                cur = con.execute(sql, args)
                return cur.fetchall() if cur.description else []
            yield run


class GeometryStore:
    def __init__(self, backend, camera_id: str):
        self.b, self.camera = backend, camera_id

    def _q(self, sql: str) -> str:
        return sql.replace("?", self.b.ph)

    def _row(self, r) -> dict:
        cam, ver, state, record, note, by, created, activated = r
        return {"camera_id": cam, "ver": ver, "state": state, "record": json.loads(record), "note": note,
                "created_by": by, "created_at": created, "activated_at": activated}

    def _select(self, run, where: str, args=()) -> list[dict]:
        cols = _COLUMNS.replace("record,", f"{self.b.sel_record},")
        return [self._row(r) for r in run(self._q(f"SELECT {cols} FROM camera_geometry WHERE camera_id = ? {where}"),
                                          (self.camera, *args))]

    # -- reads ---------------------------------------------------------------------------------------------------
    def active(self) -> dict | None:
        """The active version, or None: uncalibrated (the default; the worker then does nothing new)."""
        with self.b.tx() as run:
            rows = self._select(run, "AND state = 'active'")
        return rows[0] if rows else None

    def active_version(self) -> int | None:
        """The active version's number, or None: one small query, for a process (the worker, later) that checks every
        so often whether the calibration changed. No cache here: an activation is visible to the next call."""
        with self.b.tx() as run:
            rows = run(self._q("SELECT ver FROM camera_geometry WHERE camera_id = ? AND state = 'active'"), (self.camera,))
        return rows[0][0] if rows else None

    def get(self, ver: int) -> dict | None:
        with self.b.tx() as run:
            rows = self._select(run, "AND ver = ?", (ver,))
        return rows[0] if rows else None

    def list(self, state: str | None = None) -> list[dict]:
        """All versions of the camera, newest first (optionally one state)."""
        if state is not None and state not in STATES:
            raise GeometryError(f"unknown state {state!r}")
        with self.b.tx() as run:
            rows = self._select(run, "AND state = ?" if state else "", (state,) if state else ())
        return sorted(rows, key=lambda r: -r["ver"])

    # -- writes --------------------------------------------------------------------------------------------------
    def create(self, record: dict, created_by: str = "", note: str = "", now: float | None = None) -> int:
        """Store a new draft -> its version number (above every earlier one of the camera)."""
        if not isinstance(record, dict):
            raise GeometryError("record must be an object")
        _clean(record)
        text = json.dumps(record)
        with self.b.tx() as run:
            ver = run(self._q("SELECT COALESCE(MAX(ver), 0) FROM camera_geometry WHERE camera_id = ?"),
                      (self.camera,))[0][0] + 1
            at = time.time() if now is None else now
            run(self._q(f"INSERT INTO camera_geometry (camera_id, ver, state, record, note, created_by, created_at) "
                        f"VALUES (?, ?, 'draft', {self.b.ins_record}, ?, ?, ?)"),
                (self.camera, ver, text, note, created_by, at))
            detail = {"note": note, "created_by": created_by}
            if (record.get("reanchor") or {}).get("parent"):
                detail["parent"] = record["reanchor"]["parent"]  # carried over from this version (geometry/reanchor.py)
            if (record.get("suggest") or {}).get("parent"):
                detail.update(parent=record["suggest"]["parent"], suggest=record["suggest"].get("mode"))  # a slot suggestion
            self._event(run, "draft", ver, detail, at)
        return ver

    def update_draft(self, ver: int, record: dict) -> bool:
        """Replace a draft's record (the Admin fine-tuned a suggestion before Accept) -> False for an unknown version. Only
        drafts: an active or earlier version is history (GeometryError)."""
        if not isinstance(record, dict):
            raise GeometryError("record must be an object")
        _clean(record)
        with self.b.tx() as run:
            row = self._select(run, "AND ver = ?", (ver,))
            if not row:
                return False
            if row[0]["state"] != "draft":
                raise GeometryError(f"version {ver} is {row[0]['state']}, only drafts can be edited")
            run(self._q(f"UPDATE camera_geometry SET record = {self.b.ins_record} WHERE camera_id = ? AND ver = ?"),
                (json.dumps(record), self.camera, ver))
        return True

    def activate(self, ver: int, now: float | None = None) -> dict:
        """Accept a draft, or roll back to a superseded version: it becomes active, the current active one superseded
        (the oldest superseded ones beyond KEEP_SUPERSEDED are dropped). A stale version can't be re-activated."""
        now = time.time() if now is None else now
        with self.b.tx() as run:
            rows = self._select(run, "AND ver = ?", (ver,))
            if not rows:
                raise GeometryError(f"camera {self.camera!r} has no geometry version {ver}")
            state = rows[0]["state"]
            if state == "stale":
                raise GeometryError(f"version {ver} is stale (the camera moved): run the setup again")
            if state != "active":
                before = run(self._q("SELECT ver FROM camera_geometry WHERE camera_id = ? AND state = 'active'"), (self.camera,))
                run(self._q("UPDATE camera_geometry SET state = 'superseded' WHERE camera_id = ? AND state = 'active'"),
                    (self.camera,))
                run(self._q("UPDATE camera_geometry SET state = 'active', activated_at = ? WHERE camera_id = ? AND ver = ?"),
                    (now, self.camera, ver))
                detail = {"previous": before[0][0] if before else None, "rollback": state == "superseded"}
                parent = (rows[0]["record"].get("reanchor") or {}).get("parent")
                if parent:
                    detail["parent"] = parent  # re-anchored from this version: the calibration continues through the camera's move
                suggest = rows[0]["record"].get("suggest") or {}
                if suggest.get("parent"):
                    detail.update(parent=suggest["parent"], suggest=suggest.get("mode"))  # an accepted slot suggestion
                self._event(run, "activated", ver, detail, now)
                old = run(self._q("SELECT ver FROM camera_geometry WHERE camera_id = ? AND state = 'superseded' "
                                  "ORDER BY ver DESC"), (self.camera,))
                for (v,) in old[KEEP_SUPERSEDED:]:
                    run(self._q("DELETE FROM camera_geometry WHERE camera_id = ? AND ver = ?"), (self.camera, v))
            return self._select(run, "AND ver = ?", (ver,))[0]

    def rollback(self, now: float | None = None) -> dict:
        """Re-activate the version that was active before the current one (the superseded one activated last)."""
        with self.b.tx() as run:
            prev = run(self._q("SELECT ver FROM camera_geometry WHERE camera_id = ? AND state = 'superseded' "
                               "ORDER BY activated_at DESC, ver DESC"), (self.camera,))
        if not prev:
            raise GeometryError(f"camera {self.camera!r} has no earlier version to roll back to")
        return self.activate(prev[0][0], now)

    def mark_stale(self, reason: str = "", now: float | None = None, ver: int | None = None) -> int | None:
        """The drift guard saw the camera move: the active version becomes stale (metrics stop; uncalibrated again) -> its
        version, or None if none was active. ver: only if THAT version is the active one (a check that began before the
        Admin accepted a newer calibration must not stale the new one)."""
        with self.b.tx() as run:
            rows = self._select(run, "AND state = 'active'")
            if not rows or (ver is not None and rows[0]["ver"] != ver):
                return None
            note = " | ".join(x for x in (rows[0]["note"], f"stale: {reason}" if reason else "") if x)
            run(self._q("UPDATE camera_geometry SET state = 'stale', note = ? WHERE camera_id = ? AND state = 'active'"),
                (note, self.camera))
            self._event(run, "stale", rows[0]["ver"], {"reason": reason}, time.time() if now is None else now)
            return rows[0]["ver"]

    def discard_drafts(self) -> list[int]:
        """Delete every draft of the camera (never an active, superseded or stale version) -> their numbers."""
        with self.b.tx() as run:
            gone = [r["ver"] for r in self._select(run, "AND state = 'draft'")]
            for v in gone:
                run(self._q("DELETE FROM camera_geometry WHERE camera_id = ? AND ver = ?"), (self.camera, v))
            if gone:
                self._event(run, "discarded", None, {"versions": sorted(gone)}, time.time())
        return sorted(gone)

    def delete_draft(self, ver: int) -> bool:
        """Discard a draft (the Admin rejected it). Only drafts: an active or earlier version is history."""
        with self.b.tx() as run:
            row = self._select(run, "AND ver = ?", (ver,))
            if not row:
                return False
            if row[0]["state"] != "draft":
                raise GeometryError(f"version {ver} is {row[0]['state']}, only drafts can be deleted")
            run(self._q("DELETE FROM camera_geometry WHERE camera_id = ? AND ver = ?"), (self.camera, ver))
            self._event(run, "discarded", ver, {"versions": [ver]}, time.time())
            return True


    # -- history: lifecycle events and occupancy samples ---------------------------------------------------------------
    def _event(self, run, kind: str, ver: int | None, detail: dict | None, at: float):
        run(self._q("INSERT INTO geometry_events (camera_id, t, kind, ver, detail) VALUES (?, ?, ?, ?, ?)"),
            (self.camera, at, kind, ver, json.dumps(detail or {})))

    def log_event(self, kind: str, ver: int | None = None, detail: dict | None = None, now: float | None = None):
        """Record something that happened to the calibration outside a state change (e.g. the drift guard's suspicion)."""
        _clean(detail or {}, "detail")
        with self.b.tx() as run:
            self._event(run, kind, ver, detail, time.time() if now is None else now)

    def events(self, limit: int = 50, since: float | None = None) -> list[dict]:
        """The camera's calibration lifecycle, newest first: [{"id", "t", "kind" (draft / activated / stale / discarded /
        ...), "ver", "detail"}]."""
        with self.b.tx() as run:
            rows = run(self._q("SELECT id, t, kind, ver, detail FROM geometry_events WHERE camera_id = ? "
                               + ("AND t > ? " if since is not None else "") + "ORDER BY t DESC, id DESC LIMIT ?"),
                       (self.camera, *(() if since is None else (since,)), int(limit)))
        return [{"id": r[0], "t": r[1], "kind": r[2], "ver": r[3], "detail": json.loads(r[4] or "{}")} for r in rows]

    def add_sample(self, ver: int, counts: dict, now: float | None = None):
        """One occupancy sample ({"capacity", "occupied", "vehicles", "measured"} as the worker's metrics report them), keyed
        by the whole second: a second sample in the same second is ignored."""
        t = int(time.time() if now is None else now)
        with self.b.tx() as run:
            run(self._q("INSERT INTO geometry_samples (camera_id, t, ver, capacity, occupied, vehicles, measured) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (camera_id, t) DO NOTHING"),
                (self.camera, t, ver, int(counts["capacity"]), int(counts["occupied"]), int(counts.get("vehicles", 0)),
                 int(counts.get("measured", 0))))

    def history(self, since: float, until: float, bucket: int) -> list[dict]:
        """Occupancy over time, bucketed in the database: one row per `bucket` seconds that has samples (oldest first):
        {"t" (bucket start), "samples", "occupied_avg" / "_max" / "_min", "capacity_avg" / "_max", "vehicles_avg", "ver"
        (the newest calibration in it), "first" / "last" (sample times)}. One indexed range scan on (camera, time): the
        cost follows the number of buckets asked for, not the length of the history."""
        bucket = max(1, int(bucket))
        with self.b.tx() as run:
            rows = run(self._q("SELECT t / ?, COUNT(*), AVG(occupied), MAX(occupied), MIN(occupied), AVG(capacity), MAX(capacity), "
                               "AVG(vehicles), MAX(ver), MIN(t), MAX(t) FROM geometry_samples "
                               "WHERE camera_id = ? AND t >= ? AND t < ? GROUP BY 1 ORDER BY 1"),
                       (bucket, self.camera, int(since), int(until)))
        return [{"t": int(r[0]) * bucket, "samples": r[1], "occupied_avg": round(float(r[2]), 2), "occupied_max": r[3],
                 "occupied_min": r[4], "capacity_avg": round(float(r[5]), 2), "capacity_max": r[6],
                 "vehicles_avg": round(float(r[7]), 2), "ver": r[8], "first": r[9], "last": r[10]} for r in rows]

    def prune_history(self, now: float | None = None) -> int:
        """Drop samples older than SAMPLE_KEEP_DAYS and events older than EVENT_KEEP_DAYS -> the samples removed."""
        now = time.time() if now is None else now
        with self.b.tx() as run:
            old = run(self._q("SELECT COUNT(*) FROM geometry_samples WHERE camera_id = ? AND t < ?"),
                      (self.camera, int(now - SAMPLE_KEEP_DAYS * 86400)))[0][0]
            run(self._q("DELETE FROM geometry_samples WHERE camera_id = ? AND t < ?"), (self.camera, int(now - SAMPLE_KEEP_DAYS * 86400)))
            run(self._q("DELETE FROM geometry_events WHERE camera_id = ? AND t < ?"), (self.camera, now - EVENT_KEEP_DAYS * 86400))
        return int(old)


def sqlite_geometry(con: sqlite3.Connection, camera_id: str) -> GeometryStore:
    """The store in a camera's history database (what LiveStore.geometry() returns)."""
    return GeometryStore(_Sqlite(con), camera_id)


def pg_geometry(dsn: str, camera_id: str) -> GeometryStore:
    """The store in PostgreSQL (the camera must exist in the registry; PgRegistry(dsn) has run the migration)."""
    return GeometryStore(_Pg(dsn), camera_id)


def open_geometry(live_store, camera_id: str, env=None) -> GeometryStore:
    """PostgreSQL when CARPARK_DATABASE_URL is set (the same rule as open_registry), else the camera's SQLite file."""
    import os
    env = os.environ if env is None else env
    dsn = env.get("CARPARK_DATABASE_URL")
    return pg_geometry(dsn, camera_id) if dsn else live_store.geometry(camera_id)
