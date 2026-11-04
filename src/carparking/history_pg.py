"""Step 4: every camera's history in PostgreSQL: central, indexed, kept by a retention policy.

The workers keep writing their own SQLite file (carparking/history.py: the source of truth, also what the Streamlit
page reads); HistorySync copies it into PostgreSQL (compose service "sync", scripts/sync.py):
  * parking_sessions   one row per parking session (entry -> exit), updated when an open session closes
  * parking_events     the append-only event log (entry / exit per slot)
  * occupancy_samples  FREE / BUSY / UNKNOWN / TOTAL once a minute per camera (occupancy over the day)
  * camera_state       each camera's state now (worker online, monitoring, counts, heartbeat)
  * camera_events      camera state changes: worker online / offline, monitoring on / off, slot map changed
  * history_sync       per camera: how far the copy got (the last SQLite ids) and when
Rows keep their SQLite ids (source_id), so the API serves the same ids from either database and a client polling
with since_id never sees a jump. The copy is incremental and idempotent (ON CONFLICT): a sync that was down, or
PostgreSQL being unreachable, only delays it; the next run catches up. The first run is the migration of the
existing history. The retention policy deletes old rows in PostgreSQL only (the SQLite files keep everything).

PgHistory serves History & Analytics from PostgreSQL (SQL aggregation, per camera time zone); SqliteHistory is the
same interface over a camera's SQLite file (fallback: no PostgreSQL, or its copy is not fresh). summarize() is the
analytics in Python, used by the fallback; the PostgreSQL queries give the same numbers (tests compare them).
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import statistics
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path

STALE = 15.0  # seconds without a heartbeat: the worker counts as offline (as live.STALE)
BATCH = 5000  # rows per copy round
SAMPLE_EVERY = 60.0  # seconds between occupancy samples
DURATION_EDGES = [("< 15 min", 15), ("15–60 min", 60), ("1–2 h", 120), ("2–4 h", 240), ("4–8 h", 480),
                  ("> 8 h", None)]

MIGRATION = ("003_history", """
    CREATE TABLE IF NOT EXISTS parking_sessions (
        camera_id    TEXT NOT NULL,
        source_id    BIGINT NOT NULL,              -- history.id in the camera's SQLite file
        scope        TEXT NOT NULL,                -- "Camera 1 | yolov8s": camera name | detector (| fixed)
        slot_id      INTEGER NOT NULL,
        entry_at     TIMESTAMPTZ NOT NULL,
        exit_at      TIMESTAMPTZ,                  -- NULL: still parked
        duration_min DOUBLE PRECISION,
        status       TEXT NOT NULL,                -- Occupied / Available
        note         TEXT NOT NULL DEFAULT '',
        synced_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (camera_id, source_id)
    );
    CREATE INDEX IF NOT EXISTS parking_sessions_scope ON parking_sessions (camera_id, scope, source_id DESC);
    CREATE INDEX IF NOT EXISTS parking_sessions_entry ON parking_sessions (camera_id, entry_at);
    CREATE INDEX IF NOT EXISTS parking_sessions_open ON parking_sessions (camera_id) WHERE exit_at IS NULL;
    CREATE INDEX IF NOT EXISTS parking_sessions_exit ON parking_sessions (exit_at);   -- retention

    CREATE TABLE IF NOT EXISTS parking_events (
        camera_id  TEXT NOT NULL,
        source_id  BIGINT NOT NULL,                -- events.id in the camera's SQLite file
        scope      TEXT NOT NULL,
        slot_id    INTEGER NOT NULL,
        event      TEXT NOT NULL CHECK (event IN ('entry', 'exit')),
        at         TIMESTAMPTZ NOT NULL,
        session_id BIGINT,                         -- parking_sessions.source_id
        note       TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (camera_id, source_id)
    );
    CREATE INDEX IF NOT EXISTS parking_events_scope ON parking_events (camera_id, scope, source_id DESC);
    CREATE INDEX IF NOT EXISTS parking_events_at ON parking_events (camera_id, at);
    CREATE INDEX IF NOT EXISTS parking_events_age ON parking_events USING brin (at);   -- retention

    CREATE TABLE IF NOT EXISTS occupancy_samples (
        camera_id TEXT NOT NULL,
        at        TIMESTAMPTZ NOT NULL,            -- the minute
        scope     TEXT NOT NULL DEFAULT '',
        free      INTEGER NOT NULL,
        busy      INTEGER NOT NULL,
        unknown   INTEGER NOT NULL,
        total     INTEGER NOT NULL,
        PRIMARY KEY (camera_id, at)
    );
    CREATE INDEX IF NOT EXISTS occupancy_samples_age ON occupancy_samples USING brin (at);

    CREATE TABLE IF NOT EXISTS camera_state (
        camera_id    TEXT PRIMARY KEY,
        scope        TEXT NOT NULL DEFAULT '',
        online       BOOLEAN NOT NULL,
        monitoring   BOOLEAN NOT NULL,
        status       TEXT NOT NULL DEFAULT '',
        free         INTEGER,
        busy         INTEGER,
        unknown      INTEGER,
        total        INTEGER,
        heartbeat_at TIMESTAMPTZ,
        updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS camera_events (
        id        BIGSERIAL PRIMARY KEY,
        camera_id TEXT NOT NULL,
        at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        kind      TEXT NOT NULL,                   -- worker_online / worker_offline / monitoring_on / ... / first_seen
        detail    TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS camera_events_cam ON camera_events (camera_id, at DESC);

    CREATE TABLE IF NOT EXISTS history_sync (
        camera_id   TEXT PRIMARY KEY,
        db_path     TEXT NOT NULL DEFAULT '',
        sessions_id BIGINT NOT NULL DEFAULT 0,     -- the last SQLite history.id copied
        events_id   BIGINT NOT NULL DEFAULT 0,     -- the last SQLite events.id copied
        synced_at   TIMESTAMPTZ,
        error       TEXT NOT NULL DEFAULT ''
    );
""")


def scope_of(camera: str, detector: str) -> str:
    """The SQLite scope key (HistoryDB.scope)."""
    return f"{camera} | {detector}" if detector else camera


def iso(t: float | None, tz: tzinfo | None) -> str | None:
    """Epoch seconds -> ISO 8601 with UTC offset in the camera's time zone (HistoryDB.iso)."""
    if t is None:
        return None
    dt = datetime.fromtimestamp(t, tz)
    return (dt if tz else dt.astimezone()).isoformat(timespec="seconds")


def tz_name(tz: tzinfo | None) -> str:
    return getattr(tz, "key", None) or "UTC"


def bucket_of(days: float) -> tuple[str, int]:
    """Occupancy chart resolution for a window: (label, seconds)."""
    if days <= 1:
        return "5 min", 300
    if days <= 7:
        return "1 h", 3600
    return "1 day", 86400


def _duration_bucket(minutes: float) -> int:
    for i, (_, hi) in enumerate(DURATION_EDGES):
        if hi is None or minutes < hi:
            return i
    return len(DURATION_EDGES) - 1


def summarize(events, sessions, samples, tz: tzinfo | None, days: float, now: float | None = None) -> dict:
    """Analytics of one camera over the last `days` (Python; the PostgreSQL queries give the same numbers).
    events: (ts, "entry"/"exit"); sessions: (entry_ts, exit_ts, duration_min); samples: (ts, free, busy, unknown,
    total). Hours and days are the camera's local time."""
    now = time.time() if now is None else now
    since = now - days * 86400
    local = lambda t: datetime.fromtimestamp(t, tz)  # noqa: E731
    ev = [(local(t), kind) for t, kind in events if t >= since and kind in ("entry", "exit")]  # (notices are no movements)
    hours = [{"hour": f"{h:02d}", "entries": 0, "exits": 0} for h in range(24)]
    per_day = {d: {"day": d, "entries": 0, "exits": 0} for d in _days(local(since).date(), local(now).date())}
    for dt, kind in ev:
        key = "entries" if kind == "entry" else "exits"
        hours[dt.hour][key] += 1
        if dt.date().isoformat() in per_day:
            per_day[dt.date().isoformat()][key] += 1
    window = [(t_in, t_out, d) for t_in, t_out, d in sessions if t_in >= since]
    closed = [d for _, t_out, d in window if t_out is not None and d is not None]
    lengths = [{"bucket": b, "sessions": 0} for b, _ in DURATION_EDGES]
    for d in closed:
        lengths[_duration_bucket(d)]["sessions"] += 1
    label, step = bucket_of(days)
    occ: dict[datetime, list] = {}
    for t, free, busy, unknown, total in samples:
        if t < since:
            continue
        dt = local(t).replace(tzinfo=None)
        start = _floor(dt, step)
        occ.setdefault(start, []).append((free, busy, unknown, total))
    occupancy = [{"t": k.isoformat(timespec="minutes"), "free": _avg(v, 0), "busy": _avg(v, 1),
                  "unknown": _avg(v, 2), "total": max(r[3] for r in v)} for k, v in sorted(occ.items())]
    entries = sum(h["entries"] for h in hours)
    busiest = max(hours, key=lambda h: h["entries"]) if entries else None
    return {
        "days": days, "since": iso(since, tz), "bucket": label,
        "totals": {"entries": entries, "exits": sum(h["exits"] for h in hours), "sessions": len(window),
                   "closed": len(closed), "open": sum(1 for _, t_out, _ in window if t_out is None),
                   "avg_min": round(statistics.fmean(closed), 1) if closed else None,
                   "median_min": round(statistics.median(closed), 1) if closed else None,
                   "busiest_hour": busiest["hour"] if busiest else None,
                   "peak_busy": max((s[2] for s in samples if s[0] >= since), default=None)},
        "per_hour": hours, "per_day": list(per_day.values()), "durations": lengths, "occupancy": occupancy,
    }


def _days(first: date, last: date) -> list[str]:
    n = (last - first).days
    return [(first + timedelta(days=i)).isoformat() for i in range(n + 1)]


def _floor(dt: datetime, step: int) -> datetime:
    """Start of the step-second bucket of a local (naive) time, aligned to local midnight."""
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(seconds=int((dt - midnight).total_seconds() // step * step))


def _avg(rows, i) -> float:
    return round(sum(r[i] for r in rows) / len(rows), 1)


def _ro(path: Path) -> sqlite3.Connection:
    """The camera's SQLite file, read only (the sync never writes it: the worker and the page own it)."""
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, timeout=5)


def _tables(src: sqlite3.Connection) -> set[str]:
    return {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


@dataclass
class Scope:
    """Whose history: the camera (registry id), its history scope (camera name + detector) and time zone."""
    camera_id: str
    camera: str
    detector: str
    tz: tzinfo | None = None

    @property
    def key(self) -> str:
        return scope_of(self.camera, self.detector)


SESSION_CSV = ["id", "camera", "slot_id", "entry_time", "exit_time", "duration_min", "status", "note", "entry_ts",
               "exit_ts"]
EVENT_CSV = ["id", "slot_id", "event", "time", "ts", "session_id", "note"]


def _csv(header, rows) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return out.getvalue()


def _local(t: float | None, tz) -> str:
    return "" if t is None else datetime.fromtimestamp(t, tz).strftime("%Y-%m-%d %H:%M:%S")


# --- the copy: SQLite -> PostgreSQL ---------------------------------------------------------------------------------

class HistorySync:
    """Copies each camera's SQLite history into PostgreSQL. step(pairs) with pairs = [(camera_id, sqlite path)];
    one PostgreSQL connection, reopened after an error. Never writes a SQLite file."""

    def __init__(self, dsn: str, retention_days: float = 365, sample_days: float = 90, log=print,
                 clock=time.time, sample_every: float = SAMPLE_EVERY):
        self.dsn, self.retention_days, self.sample_days, self.log = dsn, retention_days, sample_days, log
        self.clock, self.sample_every = clock, sample_every
        self.con = None
        self.last_sample: dict[str, float] = {}
        self.last_retention = 0.0

    def _conn(self):
        if self.con is None or self.con.closed:
            import psycopg
            self.con = psycopg.connect(self.dsn, autocommit=True)
        return self.con

    def close(self):
        if self.con is not None:
            self.con.close()

    def step(self, pairs) -> dict[str, dict]:
        """One round over every camera -> {camera_id: {"events", "sessions", "closed"} copied | {"error"}}."""
        out = {}
        try:
            con = self._conn()
        except Exception as e:
            self.con = None
            raise RuntimeError(f"PostgreSQL unreachable: {e}") from e
        for cam_id, path in pairs:
            try:
                out[cam_id] = self.camera(con, cam_id, Path(path))
            except Exception as e:  # one camera's problem never stops the others
                out[cam_id] = {"error": str(e)}
                if con.closed or con.broken:
                    self.con = None
                    break
                try:
                    con.execute("INSERT INTO history_sync (camera_id, error) VALUES (%s, %s) ON CONFLICT (camera_id) "
                                "DO UPDATE SET error = EXCLUDED.error", (cam_id, str(e)[:500]))
                except Exception:
                    self.con = None
                    break
        if self.con is not None and self.clock() - self.last_retention >= 3600:
            self.last_retention = self.clock()
            removed = self.retention(con)
            if any(removed.values()):
                self.log(f"sync: retention removed {removed}")
        return out

    def camera(self, con, cam_id: str, path: Path) -> dict:
        if not path.exists():
            return {"skipped": "no database yet"}
        src = _ro(path)
        try:
            tables = _tables(src)
            done = {"events": 0, "sessions": 0, "closed": 0}
            if {"history", "events"} <= tables:
                done = self._copy(con, src, cam_id, path)
            if "live_state" in tables:
                self._state(con, src, cam_id)
            return done
        finally:
            src.close()

    def _copy(self, con, src, cam_id: str, path: Path) -> dict:
        row = con.execute("SELECT sessions_id, events_id FROM history_sync WHERE camera_id = %s", (cam_id,)).fetchone()
        s_id, e_id = row or (0, 0)
        s_max, e_max = _last_id(src, "history"), _last_id(src, "events")
        if s_max < s_id or e_max < e_id:  # a new / restored SQLite file: its ids would collide with copied rows
            raise RuntimeError(f"{path} has fewer rows than already copied (replaced or restored?): run "
                               f"'python scripts/sync.py --rebuild {cam_id}' to copy it again from scratch")
        n_ev = n_s = 0
        while True:  # the first run (the migration) copies everything, BATCH rows per transaction
            events = src.execute("SELECT id, camera, slot_id, event, ts, session_id, note FROM events WHERE id > ? "
                                 "ORDER BY id LIMIT ?", (e_id, BATCH)).fetchall()
            sessions = src.execute("SELECT id, camera, slot_id, entry_ts, exit_ts, duration_min, status, note "
                                   "FROM history WHERE id > ? ORDER BY id LIMIT ?", (s_id, BATCH)).fetchall()
            if not events and not sessions:
                break
            with con.transaction():
                if sessions:
                    self._upsert_sessions(con, cam_id, sessions)
                    s_id = sessions[-1][0]
                if events:
                    with con.cursor() as cur:
                        cur.executemany(
                            "INSERT INTO parking_events (camera_id, source_id, scope, slot_id, event, at, session_id, "
                            "note) VALUES (%s, %s, %s, %s, %s, to_timestamp(%s), %s, %s) ON CONFLICT DO NOTHING",
                            [(cam_id, i, scope or "", slot, ev, ts, sid, note or "")
                             for i, scope, slot, ev, ts, sid, note in events if ts is not None and ev in ("entry", "exit")])
                    e_id = events[-1][0]
                self._cursor(con, cam_id, path, s_id, e_id)
            n_ev, n_s = n_ev + len(events), n_s + len(sessions)
        # sessions copied while open: copy their exit once SQLite has it
        open_ids = [r[0] for r in con.execute("SELECT source_id FROM parking_sessions WHERE camera_id = %s AND "
                                              "exit_at IS NULL", (cam_id,))]
        closed = []
        for i in range(0, len(open_ids), 500):
            chunk = open_ids[i:i + 500]
            closed += src.execute("SELECT id, camera, slot_id, entry_ts, exit_ts, duration_min, status, note FROM "
                                  f"history WHERE exit_ts IS NOT NULL AND id IN ({','.join('?' * len(chunk))})",
                                  chunk).fetchall()
        with con.transaction():
            if closed:
                self._upsert_sessions(con, cam_id, closed)
            self._cursor(con, cam_id, path, s_id, e_id)
        return {"events": n_ev, "sessions": n_s, "closed": len(closed)}

    @staticmethod
    def _upsert_sessions(con, cam_id, rows):
        with con.cursor() as cur:
            cur.executemany(
                "INSERT INTO parking_sessions (camera_id, source_id, scope, slot_id, entry_at, exit_at, duration_min, "
                "status, note) VALUES (%s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s), %s, %s, %s) "
                "ON CONFLICT (camera_id, source_id) DO UPDATE SET exit_at = EXCLUDED.exit_at, "
                "duration_min = EXCLUDED.duration_min, status = EXCLUDED.status, note = EXCLUDED.note, synced_at = now()",
                [(cam_id, i, scope or "", slot, t_in, t_out, dur, status or "", note or "")
                 for i, scope, slot, t_in, t_out, dur, status, note in rows if t_in is not None])

    @staticmethod
    def _cursor(con, cam_id, path, s_id, e_id):
        con.execute("INSERT INTO history_sync (camera_id, db_path, sessions_id, events_id, synced_at, error) "
                    "VALUES (%s, %s, %s, %s, now(), '') ON CONFLICT (camera_id) DO UPDATE SET db_path = "
                    "EXCLUDED.db_path, sessions_id = EXCLUDED.sessions_id, events_id = EXCLUDED.events_id, "
                    "synced_at = now(), error = ''", (cam_id, str(path), s_id, e_id))

    def _state(self, con, src, cam_id: str):
        """camera_state now, camera_events on a change, an occupancy sample once a minute."""
        row = src.execute("SELECT heartbeat, snapshot FROM live_state WHERE id = 1").fetchone()
        if row is None:
            return
        now = self.clock()
        beat = row[0] or 0.0
        snap = json.loads(row[1]) if row[1] else {}
        online = bool(beat) and now - beat < STALE
        monitoring = bool(snap.get("enabled"))
        c = snap.get("counts") or {}
        scope = snap.get("scope") or ""
        counts = (c.get("free"), c.get("busy"), c.get("unknown"), c.get("slots")) if c else (None,) * 4
        with con.transaction():
            prev = con.execute("SELECT online, monitoring, scope FROM camera_state WHERE camera_id = %s FOR UPDATE",
                               (cam_id,)).fetchone()
            changes = []
            if prev is None:
                changes.append(("first_seen", f"worker {'online' if online else 'offline'}, monitoring "
                                              f"{'on' if monitoring else 'off'}"))
            else:
                if prev[0] != online:
                    changes.append(("worker_online", "") if online else
                                   ("worker_offline", f"last heartbeat {now - beat:.0f} s ago" if beat else ""))
                if prev[1] != monitoring:
                    changes.append(("monitoring_on" if monitoring else "monitoring_off", ""))
                if scope and prev[2] and prev[2] != scope:
                    changes.append(("scope_changed", f"{prev[2]} -> {scope}"))
            con.execute(
                "INSERT INTO camera_state (camera_id, scope, online, monitoring, status, free, busy, unknown, total, "
                "heartbeat_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), now()) "
                "ON CONFLICT (camera_id) DO UPDATE SET scope = CASE WHEN EXCLUDED.scope = '' THEN camera_state.scope "
                "ELSE EXCLUDED.scope END, online = EXCLUDED.online, monitoring = EXCLUDED.monitoring, "
                "status = EXCLUDED.status, free = EXCLUDED.free, busy = EXCLUDED.busy, unknown = EXCLUDED.unknown, "
                "total = EXCLUDED.total, heartbeat_at = EXCLUDED.heartbeat_at, updated_at = now()",
                (cam_id, scope, online, monitoring, str(snap.get("status") or ""), *counts, beat or None))
            for kind, detail in changes:
                con.execute("INSERT INTO camera_events (camera_id, at, kind, detail) VALUES (%s, to_timestamp(%s), "
                            "%s, %s)", (cam_id, now, kind, detail))
            decided = (c.get("free") or 0) + (c.get("busy") or 0) > 0  # not while warming up (every slot UNKNOWN)
            if online and monitoring and decided and now - self.last_sample.get(cam_id, 0.0) >= self.sample_every:
                con.execute("INSERT INTO occupancy_samples (camera_id, at, scope, free, busy, unknown, total) VALUES "
                            "(%s, date_trunc('minute', to_timestamp(%s)), %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                            (cam_id, now, scope, *(v or 0 for v in counts)))
                self.last_sample[cam_id] = now
        for kind, detail in changes:
            self.log(f"sync: {cam_id}: {kind} {detail}".rstrip())

    def retention(self, con) -> dict[str, int]:
        """Delete rows older than the retention policy, in small batches (short locks). Open sessions stay."""
        out = {}
        jobs = []
        if self.retention_days > 0:
            jobs += [("parking_events", "at < now() - make_interval(secs => %s)", self.retention_days),
                     ("parking_sessions", "exit_at IS NOT NULL AND exit_at < now() - make_interval(secs => %s)",
                      self.retention_days),
                     ("camera_events", "at < now() - make_interval(secs => %s)", self.retention_days)]
        if self.sample_days > 0:
            jobs.append(("occupancy_samples", "at < now() - make_interval(secs => %s)", self.sample_days))
        for table, where, days in jobs:
            n = 0
            while True:
                cur = con.execute(f"DELETE FROM {table} WHERE ctid = ANY(ARRAY(SELECT ctid FROM {table} WHERE {where} "
                                  "LIMIT 5000))", (days * 86400,))
                n += cur.rowcount
                if cur.rowcount < 5000:
                    break
            out[table] = n
        return out

    def rebuild(self, cam_id: str) -> None:
        """Forget what was copied for one camera (its SQLite file was replaced): the next step copies it again."""
        con = self._conn()
        with con.transaction():
            for table in ("parking_events", "parking_sessions"):
                con.execute(f"DELETE FROM {table} WHERE camera_id = %s", (cam_id,))
            con.execute("DELETE FROM history_sync WHERE camera_id = %s", (cam_id,))


def _last_id(src, table: str) -> int:
    """The highest id a table has handed out: its AUTOINCREMENT counter (sqlite_sequence), or MAX(id) if that is higher or absent. Rows may be
    deleted (scripts/clear_events.py empties the events table) but the counter stays, so a purged log is not mistaken for a replaced file."""
    top = src.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()[0]
    try:
        seq = src.execute("SELECT seq FROM sqlite_sequence WHERE name = ?", (table,)).fetchone()
    except sqlite3.Error:  # (no AUTOINCREMENT table in this file at all)
        seq = None
    return max(top, seq[0] if seq else 0)


def sqlite_counts(path: Path) -> dict:
    """{"sessions", "events"}: the highest ids in a camera's SQLite file (how far a copy has to go)."""
    if not Path(path).exists():
        return {"sessions": 0, "events": 0}
    src = _ro(Path(path))
    try:
        if not {"history", "events"} <= _tables(src):
            return {"sessions": 0, "events": 0}
        return {"sessions": _last_id(src, "history"), "events": _last_id(src, "events")}
    finally:
        src.close()


# --- reading: PostgreSQL and the SQLite fallback, one interface -------------------------------------------------------

class PgHistory:
    """History & Analytics from PostgreSQL. A short connection per call (the API runs requests on a thread pool)."""

    source = "postgres"

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._fresh: dict[str, tuple[float, bool]] = {}

    def _conn(self):
        import psycopg
        return psycopg.connect(self.dsn, autocommit=True)

    def fresh(self, camera_id: str, max_age: float = 30.0) -> bool:
        """Has the sync copied this camera within max_age seconds? (Else the API reads its SQLite file.) Cached 2 s."""
        t, ok = self._fresh.get(camera_id, (0.0, False))
        if time.monotonic() - t < 2.0:
            return ok
        try:
            with self._conn() as con:
                row = con.execute("SELECT extract(epoch FROM now() - synced_at) FROM history_sync WHERE camera_id = %s "
                                  "AND synced_at IS NOT NULL", (camera_id,)).fetchone()
            ok = row is not None and float(row[0]) <= max_age
        except Exception:
            ok = False
        self._fresh[camera_id] = (time.monotonic(), ok)
        return ok

    def sessions(self, s: Scope, limit: int = 50, open_only: bool = False) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT source_id, slot_id, extract(epoch FROM entry_at)::float8, extract(epoch FROM exit_at)::float8, "
                "duration_min, status, note FROM parking_sessions WHERE camera_id = %s AND scope = %s "
                f"{'AND exit_at IS NULL' if open_only else ''} ORDER BY source_id DESC LIMIT %s",
                (s.camera_id, s.key, limit)).fetchall()
        return [_session(s, *r) for r in rows]

    def sessions_page(self, s: Scope, offset: int = 0, limit: int = 20, all_scopes: bool = False) -> dict:
        """One page of the History table: open sessions (still parked) first, then newest first; + "total", "open".
        all_scopes: every scope of the camera (each detector / auto grid version); the items then carry their "scope"."""
        where, args = ("camera_id = %s", (s.camera_id,)) if all_scopes else ("camera_id = %s AND scope = %s", (s.camera_id, s.key))
        with self._conn() as con:
            rows = con.execute(
                "SELECT source_id, slot_id, extract(epoch FROM entry_at)::float8, extract(epoch FROM exit_at)::float8, "
                f"duration_min, status, note, scope FROM parking_sessions WHERE {where} "
                "ORDER BY (exit_at IS NULL) DESC, source_id DESC LIMIT %s OFFSET %s", (*args, limit, offset)).fetchall()
            total, n_open = con.execute("SELECT count(*), count(*) FILTER (WHERE exit_at IS NULL) FROM parking_sessions "
                                        f"WHERE {where}", args).fetchone()
        return {"items": [_session(s, *r[:7], r[7] if all_scopes else None) for r in rows], "total": total, "open": n_open}

    def events(self, s: Scope, limit: int = 100, since_id: int = 0) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT source_id, slot_id, event, extract(epoch FROM at)::float8, session_id, note FROM parking_events "
                "WHERE camera_id = %s AND scope = %s AND source_id > %s ORDER BY source_id DESC LIMIT %s",
                (s.camera_id, s.key, since_id, limit)).fetchall()
        return [_event(s, *r) for r in rows]

    def summary(self, s: Scope) -> dict:
        with self._conn() as con:
            n, n_open = con.execute("SELECT count(*), count(*) FILTER (WHERE exit_at IS NULL) FROM parking_sessions "
                                    "WHERE camera_id = %s AND scope = %s", (s.camera_id, s.key)).fetchone()
            n_ev = con.execute("SELECT count(*) FROM parking_events WHERE camera_id = %s AND scope = %s",
                               (s.camera_id, s.key)).fetchone()[0]
        last = self.events(s, limit=1)
        return {"camera": s.camera, "detector": s.detector, "sessions": n, "open_sessions": n_open, "events": n_ev,
                "last_event": last[0] if last else None}

    def sessions_csv(self, s: Scope) -> str:
        with self._conn() as con:
            rows = con.execute("SELECT source_id, scope, slot_id, extract(epoch FROM entry_at)::float8, "
                               "extract(epoch FROM exit_at)::float8, duration_min, status, note FROM parking_sessions "
                               "WHERE camera_id = %s AND scope = %s ORDER BY source_id", (s.camera_id, s.key)).fetchall()
        return _csv(SESSION_CSV, [(i, sc, slot, _local(a, s.tz), _local(b, s.tz), "" if d is None else d, st, note,
                                   a, "" if b is None else b) for i, sc, slot, a, b, d, st, note in rows])

    def events_csv(self, s: Scope) -> str:
        with self._conn() as con:
            rows = con.execute("SELECT source_id, slot_id, event, extract(epoch FROM at)::float8, session_id, note "
                               "FROM parking_events WHERE camera_id = %s AND scope = %s ORDER BY source_id",
                               (s.camera_id, s.key)).fetchall()
        return _csv(EVENT_CSV, [(i, slot, ev, _local(t, s.tz), t, sid, note) for i, slot, ev, t, sid, note in rows])

    def analytics(self, s: Scope, days: float, all_scopes: bool = True, now: float | None = None) -> dict:
        """summarize() in SQL: aggregation in the database, in the camera's time zone."""
        now = time.time() if now is None else now
        since = now - days * 86400
        tz = tz_name(s.tz)
        scope = None if all_scopes else s.key
        p = {"cam": s.camera_id, "scope": scope, "since": since, "tz": tz}
        f = "camera_id = %(cam)s AND (%(scope)s::text IS NULL OR scope = %(scope)s)"
        label, step = bucket_of(days)
        with self._conn() as con:
            hours = {int(h): (e, x) for h, e, x in con.execute(
                "SELECT extract(hour FROM at AT TIME ZONE %(tz)s), count(*) FILTER (WHERE event = 'entry'), "
                f"count(*) FILTER (WHERE event = 'exit') FROM parking_events WHERE {f} AND at >= to_timestamp(%(since)s) "
                "GROUP BY 1", p)}
            days_ = {d.isoformat(): (e, x) for d, e, x in con.execute(
                "SELECT (at AT TIME ZONE %(tz)s)::date, count(*) FILTER (WHERE event = 'entry'), "
                f"count(*) FILTER (WHERE event = 'exit') FROM parking_events WHERE {f} AND at >= to_timestamp(%(since)s) "
                "GROUP BY 1", p)}
            n, n_open, n_closed, avg, med = con.execute(
                "SELECT count(*), count(*) FILTER (WHERE exit_at IS NULL), count(duration_min) FILTER (WHERE exit_at "
                "IS NOT NULL), avg(duration_min) FILTER (WHERE exit_at IS NOT NULL), percentile_cont(0.5) WITHIN "
                f"GROUP (ORDER BY duration_min) FILTER (WHERE exit_at IS NOT NULL) FROM parking_sessions WHERE {f} AND "
                "entry_at >= to_timestamp(%(since)s)", p).fetchone()
            edges = [hi for _, hi in DURATION_EDGES if hi is not None]
            case = " ".join(f"WHEN duration_min < {hi} THEN {i}" for i, hi in enumerate(edges))
            lengths = dict(con.execute(
                f"SELECT CASE {case} ELSE {len(edges)} END, count(*) FROM parking_sessions WHERE {f} AND exit_at IS NOT "
                "NULL AND duration_min IS NOT NULL AND entry_at >= to_timestamp(%(since)s) GROUP BY 1", p).fetchall())
            occ = con.execute(
                "SELECT date_bin(make_interval(secs => %(step)s), at AT TIME ZONE %(tz)s, "
                "date_trunc('day', at AT TIME ZONE %(tz)s)) AS b, avg(free)::float8, avg(busy)::float8, "
                "avg(unknown)::float8, max(total) "
                f"FROM occupancy_samples WHERE {f} AND at >= to_timestamp(%(since)s) GROUP BY 1 ORDER BY 1",
                {**p, "step": step}).fetchall()
            peak = con.execute(f"SELECT max(busy) FROM occupancy_samples WHERE {f} AND at >= to_timestamp(%(since)s)",
                               p).fetchone()[0]
        per_hour = [{"hour": f"{h:02d}", "entries": hours.get(h, (0, 0))[0], "exits": hours.get(h, (0, 0))[1]}
                    for h in range(24)]
        entries = sum(h["entries"] for h in per_hour)
        busiest = max(per_hour, key=lambda h: h["entries"]) if entries else None
        first = datetime.fromtimestamp(since, s.tz).date()
        last = datetime.fromtimestamp(now, s.tz).date()
        return {
            "days": days, "since": iso(since, s.tz), "bucket": label,
            "totals": {"entries": entries, "exits": sum(h["exits"] for h in per_hour), "sessions": n,
                       "closed": n_closed, "open": n_open, "avg_min": None if avg is None else round(avg, 1),
                       "median_min": None if med is None else round(med, 1),
                       "busiest_hour": busiest["hour"] if busiest else None, "peak_busy": peak},
            "per_hour": per_hour,
            "per_day": [{"day": d, "entries": days_.get(d, (0, 0))[0], "exits": days_.get(d, (0, 0))[1]}
                        for d in _days(first, last)],
            "durations": [{"bucket": b, "sessions": lengths.get(i, 0)} for i, (b, _) in enumerate(DURATION_EDGES)],
            "occupancy": [{"t": b.isoformat(timespec="minutes"), "free": round(fr, 1), "busy": round(bu, 1),
                           "unknown": round(un, 1), "total": to}
                          for b, fr, bu, un, to in occ],
        }

    def camera_events(self, camera_id: str, tz, limit: int = 100) -> list[dict]:
        with self._conn() as con:
            rows = con.execute("SELECT id, extract(epoch FROM at)::float8, kind, detail FROM camera_events WHERE "
                               "camera_id = %s ORDER BY at DESC, id DESC LIMIT %s", (camera_id, limit)).fetchall()
        return [{"id": i, "time": iso(t, tz), "ts": t, "kind": k, "detail": d} for i, t, k, d in rows]

    def overview(self, camera_ids: list[str], tz, days: float = 1, now: float | None = None) -> dict[str, dict]:
        """Per camera: entries / exits / sessions in the window, open sessions, its state (all cameras at once)."""
        now = time.time() if now is None else now
        since = now - days * 86400
        out = {c: {"entries": 0, "exits": 0, "open": 0, "state": None} for c in camera_ids}
        with self._conn() as con:
            for cam, e, x in con.execute(
                    "SELECT camera_id, count(*) FILTER (WHERE event = 'entry'), count(*) FILTER (WHERE event = 'exit') "
                    "FROM parking_events WHERE camera_id = ANY(%s) AND at >= to_timestamp(%s) GROUP BY 1",
                    (camera_ids, since)):
                out[cam].update(entries=e, exits=x)
            for cam, n in con.execute("SELECT camera_id, count(*) FROM parking_sessions WHERE camera_id = ANY(%s) AND "
                                      "exit_at IS NULL AND scope = (SELECT scope FROM camera_state cs WHERE "
                                      "cs.camera_id = parking_sessions.camera_id) GROUP BY 1", (camera_ids,)):
                out[cam]["open"] = n
            for cam, online, mon, free, busy, unknown, total, t in con.execute(
                    "SELECT camera_id, online, monitoring, free, busy, unknown, total, extract(epoch FROM updated_at)"
                    "::float8 FROM camera_state WHERE camera_id = ANY(%s)", (camera_ids,)):
                out[cam]["state"] = {"online": online, "monitoring": mon, "free": free, "busy": busy,
                                     "unknown": unknown, "total": total, "updated": iso(t, tz)}
        return out

    def sync_status(self) -> dict[str, dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT h.camera_id, h.sessions_id, h.events_id, extract(epoch FROM h.synced_at)::float8, h.error, "
                "extract(epoch FROM now() - h.synced_at)::float8, "
                "(SELECT count(*) FROM parking_sessions s WHERE s.camera_id = h.camera_id), "
                "(SELECT count(*) FROM parking_events e WHERE e.camera_id = h.camera_id), "
                "(SELECT count(*) FROM occupancy_samples o WHERE o.camera_id = h.camera_id) FROM history_sync h"
            ).fetchall()
        return {r[0]: {"sessions_id": r[1], "events_id": r[2], "synced_ts": r[3], "error": r[4],
                       "age_s": None if r[5] is None else round(r[5], 1),
                       "rows": {"sessions": r[6], "events": r[7], "samples": r[8]}} for r in rows}


def _session(s: Scope, rid, slot, t_in, t_out, dur, status, note, scope=None) -> dict:
    return {"id": rid, "camera": s.camera, "detector": s.detector, "slot_id": slot, "in_time": iso(t_in, s.tz),
            "out_time": iso(t_out, s.tz), "in_ts": t_in, "out_ts": t_out, "duration_min": dur, "status": status,
            "note": note or "", **({} if scope is None else {"scope": scope})}


def _event(s: Scope, eid, slot, kind, t, sid, note) -> dict:
    return {"id": eid, "camera": s.camera, "detector": s.detector, "slot_id": slot, "event": kind, "time": iso(t, s.tz),
            "ts": t, "session_id": sid, "note": note or ""}


class SqliteHistory:
    """The same interface over one camera's SQLite file (no PostgreSQL, or its copy is behind). Read only."""

    source = "sqlite"

    def __init__(self, path):
        self.path = Path(path)

    def _q(self, sql, args=()):
        if not self.path.exists():
            return []
        src = _ro(self.path)
        try:
            if not {"history", "events"} <= _tables(src):
                return []
            return src.execute(sql, args).fetchall()
        finally:
            src.close()

    def sessions(self, s: Scope, limit: int = 50, open_only: bool = False) -> list[dict]:
        rows = self._q("SELECT id, slot_id, entry_ts, exit_ts, duration_min, status, note FROM history WHERE camera = ? "
                       f"{'AND exit_ts IS NULL' if open_only else ''} ORDER BY id DESC LIMIT ?", (s.key, limit))
        return [_session(s, *r) for r in rows]

    def sessions_page(self, s: Scope, offset: int = 0, limit: int = 20, all_scopes: bool = False) -> dict:
        pre = f"{s.camera} | "
        where, args = (("(camera = ? OR substr(camera, 1, ?) = ?)", (s.camera, len(pre), pre)) if all_scopes
                       else ("camera = ?", (s.key,)))
        rows = self._q("SELECT id, slot_id, entry_ts, exit_ts, duration_min, status, note, camera FROM history WHERE "
                       f"{where} ORDER BY exit_ts IS NULL DESC, id DESC LIMIT ? OFFSET ?", (*args, limit, offset))
        counts = self._q(f"SELECT COUNT(*), COALESCE(SUM(exit_ts IS NULL), 0) FROM history WHERE {where}", args)
        total, n_open = counts[0] if counts else (0, 0)
        return {"items": [_session(s, *r[:7], r[7] if all_scopes else None) for r in rows], "total": total, "open": n_open}

    def events(self, s: Scope, limit: int = 100, since_id: int = 0) -> list[dict]:
        rows = self._q("SELECT id, slot_id, event, ts, session_id, note FROM events WHERE camera = ? AND id > ? "
                       "AND event IN ('entry', 'exit') ORDER BY id DESC LIMIT ?", (s.key, since_id, limit))
        return [_event(s, *r) for r in rows]

    def summary(self, s: Scope) -> dict:
        (n, n_open), = self._q("SELECT COUNT(*), SUM(exit_ts IS NULL) FROM history WHERE camera = ?", (s.key,)) or [(0, 0)]
        (n_ev,), = self._q("SELECT COUNT(*) FROM events WHERE camera = ? AND event IN ('entry', 'exit')", (s.key,)) or [(0,)]
        last = self.events(s, limit=1)
        return {"camera": s.camera, "detector": s.detector, "sessions": n, "open_sessions": n_open or 0,
                "events": n_ev, "last_event": last[0] if last else None}

    def sessions_csv(self, s: Scope) -> str:
        rows = self._q("SELECT id, camera, slot_id, entry_ts, exit_ts, duration_min, status, note FROM history WHERE "
                       "camera = ? ORDER BY id", (s.key,))
        return _csv(SESSION_CSV, [(i, sc, slot, _local(a, s.tz), _local(b, s.tz), "" if d is None else d, st, note or "",
                                   a, "" if b is None else b) for i, sc, slot, a, b, d, st, note in rows])

    def events_csv(self, s: Scope) -> str:
        rows = self._q("SELECT id, slot_id, event, ts, session_id, note FROM events WHERE camera = ? AND event IN ('entry', 'exit') "
                       "ORDER BY id", (s.key,))
        return _csv(EVENT_CSV, [(i, slot, ev, _local(t, s.tz), t, sid, note or "") for i, slot, ev, t, sid, note in rows])

    def analytics(self, s: Scope, days: float, all_scopes: bool = True, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        since = now - days * 86400
        where, args = ("", (since,)) if all_scopes else ("AND camera = ?", (since, s.key))
        events = self._q(f"SELECT ts, event FROM events WHERE ts >= ? AND event IN ('entry', 'exit') {where}", args)
        sessions = self._q(f"SELECT entry_ts, exit_ts, duration_min FROM history WHERE entry_ts >= ? {where}", args)
        return summarize(events, sessions, [], s.tz, days, now)  # occupancy samples exist only in PostgreSQL

    def camera_events(self, camera_id: str, tz, limit: int = 100) -> list[dict]:
        return []

    def overview(self, camera_ids, tz, days: float = 1, now: float | None = None):
        raise NotImplementedError  # the API builds it per camera from analytics()
