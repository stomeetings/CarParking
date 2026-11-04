"""Step 4: the central history (carparking/history_pg.py): analytics in Python, the SQLite reader, and (with
CARPARK_TEST_PG_DSN pointing at a scratch database) the SQLite -> PostgreSQL copy, its catch-up, camera state
changes, the PostgreSQL analytics (same numbers as Python), retention and a replaced SQLite file."""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.history import HistoryDB  # noqa: E402
from carparking.history_pg import (HistorySync, PgHistory, Scope, SqliteHistory, sqlite_counts,  # noqa: E402
                                   summarize)
from carparking.live import LiveStore  # noqa: E402

TZ = ZoneInfo("Asia/Karachi")
T0 = datetime(2026, 10, 20, 8, 10, tzinfo=TZ).timestamp()  # Tuesday 08:10 local
NOW = T0 + 2 * 86400


def fill(path, T0=T0, NOW=NOW) -> HistoryDB:
    """Two days of a camera: 4 sessions (one still open), the events they make, on two scopes."""
    db = HistoryDB(path, "Front", TZ, detector="yolov8s")
    for slot, t_in, t_out in ((1, T0, T0 + 20 * 60), (2, T0 + 3600, T0 + 3 * 3600), (1, T0 + 86400, T0 + 86400 + 300)):
        db.apply({"slot": slot, "state": 1, "time": t_in, "initial": False})
        db.apply({"slot": slot, "state": 0, "time": t_out, "initial": False})
    db.apply({"slot": 3, "state": 1, "time": NOW - 600, "initial": False})  # still parked
    old = HistoryDB(path, "Front", TZ, detector="yolov8n")  # an earlier detector: another scope, same camera
    old.apply({"slot": 9, "state": 1, "time": T0 - 7200, "initial": False})
    old.apply({"slot": 9, "state": 0, "time": T0 - 3600, "initial": False})
    return db


def test_summarize_hours_days_lengths_and_occupancy_in_local_time():
    events = [(T0, "entry"), (T0 + 1200, "exit"), (T0 + 3600, "entry"), (T0 + 86400, "entry")]
    sessions = [(T0, T0 + 1200, 20.0), (T0 + 3600, T0 + 3 * 3600, 120.0), (T0 + 86400, None, None)]
    samples = [(T0, 3, 2, 0, 5), (T0 + 60, 2, 3, 0, 5), (T0 + 7200, 1, 4, 0, 5)]
    a = summarize(events, sessions, samples, TZ, days=3, now=NOW)
    assert a["per_hour"][8] == {"hour": "08", "entries": 2, "exits": 1} and a["per_hour"][9]["entries"] == 1
    assert [(d["day"], d["entries"]) for d in a["per_day"]] == [("2026-10-19", 0), ("2026-10-20", 2),
                                                                 ("2026-10-21", 1), ("2026-10-22", 0)]
    assert [d["sessions"] for d in a["durations"]] == [0, 1, 0, 1, 0, 0]  # 20 min, 120 min (2-4 h)
    t = a["totals"]
    assert (t["entries"], t["exits"], t["sessions"], t["closed"], t["open"]) == (3, 1, 3, 2, 1)
    assert (t["avg_min"], t["median_min"], t["busiest_hour"], t["peak_busy"]) == (70.0, 70.0, "08", 4)
    assert a["bucket"] == "1 h" and a["occupancy"][0] == {"t": "2026-10-20T08:00", "free": 2.5, "busy": 2.5,
                                                          "unknown": 0.0, "total": 5}
    assert summarize([], [], [], TZ, days=1, now=NOW)["totals"]["avg_min"] is None


def test_sqlite_reader_matches_the_history_db(tmp_path):
    db = fill(tmp_path / "h.db")
    s = Scope("front", "Front", "yolov8s", TZ)
    lite = SqliteHistory(tmp_path / "h.db")
    assert lite.sessions(s, 10) == db.sessions(10) and lite.events(s, 10, 2) == db.events(10, 2)
    assert lite.summary(s) == db.summary()
    assert lite.sessions(s, 10, open_only=True)[0]["slot_id"] == 3
    assert "slot_id" in lite.sessions_csv(s) and lite.events_csv(s).count("\n") == 1 + 7
    assert lite.analytics(s, 3, all_scopes=True, now=NOW)["totals"]["entries"] == 5  # both scopes
    assert lite.analytics(s, 3, all_scopes=False, now=NOW)["totals"]["entries"] == 4
    assert SqliteHistory(tmp_path / "none.db").sessions(s) == [] and sqlite_counts(tmp_path / "none.db")["events"] == 0
    assert sqlite_counts(tmp_path / "h.db") == {"sessions": 5, "events": 9}


def test_sessions_page_lists_the_parked_cars_first(tmp_path):
    db = fill(tmp_path / "h.db")
    db.apply({"slot": 4, "state": 1, "time": NOW - 300, "initial": False})  # newer than the open session of slot 3
    db.apply({"slot": 4, "state": 0, "time": NOW - 200, "initial": False})
    s = Scope("front", "Front", "yolov8s", TZ)
    lite = SqliteHistory(tmp_path / "h.db")
    first = db.sessions_page(0, 2)
    assert (first["total"], first["open"]) == (5, 1)
    assert [x["slot_id"] for x in first["items"]] == [3, 4] and first["items"][0]["out_time"] is None
    assert [x["slot_id"] for x in db.sessions_page(2, 2)["items"]] == [1, 2] and db.sessions_page(4, 2)["items"][0]["slot_id"] == 1
    assert lite.sessions_page(s, 0, 2) == first and lite.sessions_page(s, 4, 2) == db.sessions_page(4, 2)
    assert SqliteHistory(tmp_path / "none.db").sessions_page(s) == {"items": [], "total": 0, "open": 0}


def test_sessions_page_can_list_every_scope_of_the_camera(tmp_path):
    db = fill(tmp_path / "h.db")  # scope "Front | yolov8s" (4 sessions) and the earlier detector's "Front | yolov8n" (1)
    s = Scope("front", "Front", "yolov8s", TZ)
    lite = SqliteHistory(tmp_path / "h.db")
    one, every = db.sessions_page(0, 20), db.sessions_page(0, 20, all_scopes=True)
    assert (one["total"], every["total"], every["open"]) == (4, 5, 1) and all("scope" not in x for x in one["items"])
    assert {x["scope"] for x in every["items"]} == {"Front | yolov8s", "Front | yolov8n"}
    assert [x["slot_id"] for x in every["items"]][0] == 3  # the parked car first, whatever the scope
    assert lite.sessions_page(s, 0, 20, all_scopes=True) == every and lite.sessions_page(s, 0, 20) == one
    other = HistoryDB(tmp_path / "h.db", "Front Gate", TZ, detector="yolov8s")  # a camera whose name starts the same: not ours
    other.apply({"slot": 7, "state": 1, "time": NOW - 50, "initial": False})
    assert db.sessions_page(0, 20, all_scopes=True)["total"] == 5 and all(x["slot_id"] != 7 for x in db.sessions_page(0, 20, True)["items"])


@pytest.fixture
def dsn():
    dsn = os.environ.get("CARPARK_TEST_PG_DSN")
    if not dsn:
        pytest.skip("set CARPARK_TEST_PG_DSN to run the PostgreSQL copy")
    from carparking.registry import PgRegistry
    PgRegistry(dsn)  # migrations
    import psycopg
    with psycopg.connect(dsn, autocommit=True) as con:  # a scratch database: start empty
        for t in ("parking_sessions", "parking_events", "occupancy_samples", "camera_state", "camera_events",
                  "history_sync"):
            con.execute(f"DELETE FROM {t}")
    return dsn


def real():
    """PostgreSQL tests run on the real clock (retention compares with now()): two days of history up to now."""
    now = float(int(time.time()))
    return now - 2 * 86400, now


def live(path, enabled=True, beat=None, counts=None, scope="Front | yolov8s"):
    store = LiveStore(path)
    store.claim("w")
    snap = {"status": "connected", "enabled": enabled, "scope": scope,
            "counts": counts or {"free": 2, "busy": 3, "unknown": 1, "slots": 6}}
    store.beat("w", snap, now=beat)
    store.close()


def test_copy_catches_up_closes_sessions_and_matches_python_analytics(dsn, tmp_path):
    T0, NOW = real()
    path = tmp_path / "h.db"
    db = fill(path, T0, NOW)
    clock = [NOW]
    sync = HistorySync(dsn, clock=lambda: clock[0], log=lambda m: None)
    live(path, beat=NOW)
    first = sync.step([("front", path)])["front"]
    assert first == {"events": 9, "sessions": 5, "closed": 0}  # the migration of the existing history
    assert sync.step([("front", path)])["front"] == {"events": 0, "sessions": 0, "closed": 0}  # idempotent
    pg, lite = PgHistory(dsn), SqliteHistory(path)
    s = Scope("front", "Front", "yolov8s", TZ)
    assert pg.fresh("front") and not pg.fresh("other")
    assert pg.sessions(s, 10) == db.sessions(10) and pg.events(s, 10) == db.events(10)
    assert pg.events(s, 10, since_id=6) == db.events(10, 6) and pg.summary(s) == db.summary()
    assert pg.sessions_csv(s) == lite.sessions_csv(s) and pg.events_csv(s) == lite.events_csv(s)

    db.apply({"slot": 3, "state": 0, "time": NOW - 60, "initial": False})  # the open session closes
    db.apply({"slot": 4, "state": 1, "time": NOW - 30, "initial": False})
    assert sync.step([("front", path)])["front"] == {"events": 2, "sessions": 1, "closed": 1}
    assert pg.sessions(s, 10) == db.sessions(10)
    assert pg.sessions_page(s, 0, 3) == db.sessions_page(0, 3) and pg.sessions_page(s, 3, 3) == db.sessions_page(3, 3)
    assert pg.sessions_page(s, 0, 10, all_scopes=True) == db.sessions_page(0, 10, all_scopes=True)  # every scope, with its "scope"
    for days in (1, 3, 30):
        for every in (True, False):
            want = lite.analytics(s, days, every, now=NOW)
            got = pg.analytics(s, days, every, now=NOW)
            assert got["totals"].pop("peak_busy") == 3 and want["totals"].pop("peak_busy") is None  # samples: PG only
            assert {k: v for k, v in got.items() if k != "occupancy"} == \
                   {k: v for k, v in want.items() if k != "occupancy"}, (days, every)
            assert got["occupancy"][0]["busy"] == 3.0
    st = pg.sync_status()["front"]
    assert st["rows"] == {"sessions": 6, "events": 11, "samples": 1} and st["events_id"] == 11


def test_camera_state_changes_samples_and_retention(dsn, tmp_path):
    T0, NOW = real()
    path = tmp_path / "h.db"
    fill(path, T0, NOW)
    clock = [NOW]
    sync = HistorySync(dsn, clock=lambda: clock[0], log=lambda m: None, retention_days=1, sample_days=1e-9)
    sync.last_retention = NOW  # the hourly retention runs below, by hand
    live(path, beat=NOW)
    sync.step([("front", path)])
    clock[0] += 30  # no heartbeat for 30 s: offline
    sync.step([("front", path)])
    live(path, enabled=False, beat=clock[0])  # back, with monitoring switched off
    sync.step([("front", path)])
    pg = PgHistory(dsn)
    kinds = [e["kind"] for e in reversed(pg.camera_events("front", TZ))]
    assert kinds == ["first_seen", "worker_offline", "worker_online", "monitoring_off"]
    ov = pg.overview(["front"], TZ, days=1, now=NOW)["front"]
    assert ov["state"]["online"] and not ov["state"]["monitoring"] and ov["open"] == 1
    import psycopg
    with psycopg.connect(dsn, autocommit=True) as con:
        assert con.execute("SELECT count(*) FROM occupancy_samples").fetchone()[0] == 1  # only while monitoring
        removed = sync.retention(con)  # 1 day: the first day's rows go; samples (1e-9 days): all
    assert removed["parking_events"] > 0 and removed["occupancy_samples"] == 1
    # open sessions are never removed, and the next round doesn't copy the removed rows again
    assert any(x["out_ts"] is None for x in pg.sessions(Scope("front", "Front", "yolov8s", TZ), 10))
    assert sync.step([("front", path)])["front"]["events"] == 0


def test_a_replaced_sqlite_file_is_reported_not_mixed_in(dsn, tmp_path):
    T0, NOW = real()
    path = tmp_path / "h.db"
    fill(path, T0, NOW)
    sync = HistorySync(dsn, clock=lambda: NOW, log=lambda m: None)
    sync.step([("front", path)])
    path.unlink()
    HistoryDB(path, "Front", TZ, detector="yolov8s").apply({"slot": 1, "state": 1, "time": NOW, "initial": False})
    assert "rebuild" in sync.step([("front", path)])["front"]["error"]
    pg = PgHistory(dsn)
    assert pg.sync_status()["front"]["error"]
    sync.rebuild("front")
    assert sync.step([("front", path)])["front"]["events"] == 1
    assert pg.sync_status()["front"]["rows"]["events"] == 1


def test_the_sync_never_writes_the_sqlite_file(dsn, tmp_path):
    T0, NOW = real()
    path = tmp_path / "h.db"
    fill(path, T0, NOW)
    live(path, beat=NOW)
    before = path.read_bytes()
    HistorySync(dsn, clock=lambda: NOW, log=lambda m: None).step([("front", path), ("ghost", tmp_path / "x.db")])
    assert path.read_bytes() == before and not (tmp_path / "x.db").exists()
    with sqlite3.connect(path) as con:
        assert json.loads(con.execute("SELECT snapshot FROM live_state").fetchone()[0])["enabled"]


def test_a_purged_event_log_still_counts_its_ids(tmp_path):
    """scripts/clear_events.py empties `events` but not its AUTOINCREMENT counter: the sync's 'fewer rows than copied' check must not take
    that for a replaced file (it stopped syncing the camera, sessions included, until the next event)."""
    import sqlite3
    fill(tmp_path / "h.db")
    con = sqlite3.connect(tmp_path / "h.db")
    with con:
        con.execute("DELETE FROM events")
    con.close()
    assert sqlite_counts(tmp_path / "h.db") == {"sessions": 5, "events": 9}
