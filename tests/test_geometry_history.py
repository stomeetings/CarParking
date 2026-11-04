"""Phase 2 history in the geometry store: the calibration lifecycle log (written in the same transaction as each change) and the
occupancy samples with their bucketed range queries. SQLite in tmp_path; PostgreSQL only with CARPARK_TEST_PG_DSN."""
import os
import time

import pytest

from carparking.geometry_store import EVENT_KEEP_DAYS, SAMPLE_KEEP_DAYS, GeometryError
from carparking.live import LiveStore

REC = {"calibration": {"confidence": "HIGH"}}
RTSP = "rtsp://admin:secret@192.168.20.10:554/Streaming/Channels/101"
DAY = 86400


@pytest.fixture(params=["sqlite", "pg"])
def make(request, tmp_path):
    if request.param == "sqlite":
        live = LiveStore(tmp_path / "live.db")
        return lambda cam="front": live.geometry(cam)
    dsn = os.environ.get("CARPARK_TEST_PG_DSN")
    if not dsn:
        pytest.skip("set CARPARK_TEST_PG_DSN to run the PostgreSQL backend")
    import psycopg
    from carparking.geometry_store import pg_geometry
    from carparking.registry import Camera, PgRegistry
    reg = PgRegistry(dsn)
    with psycopg.connect(dsn, autocommit=True) as con:
        con.execute("DELETE FROM cameras")
    for cid in ("front", "gate"):
        reg.create(Camera(cid, cid.title(), RTSP, primary=cid == "front"))
    return lambda cam="front": pg_geometry(dsn, cam)


def counts(capacity=4, occupied=1, vehicles=2, measured=2):
    return {"capacity": capacity, "occupied": occupied, "vehicles": vehicles, "measured": measured}


def test_the_lifecycle_is_logged_with_every_change(make):
    g = make()
    assert g.events() == []
    g.create(REC, created_by="admin", note="first", now=100.0)
    g.create(REC, now=101.0)
    g.activate(1, now=110.0)
    g.activate(2, now=120.0)
    g.rollback(now=130.0)
    g.mark_stale("moved 14 px", now=140.0)
    g.create(REC, now=150.0)
    g.create(REC, now=151.0)
    assert g.delete_draft(4) and g.discard_drafts() == [3]
    ev = g.events(limit=100)
    assert [(e["kind"], e["ver"]) for e in reversed(ev)] == [
        ("draft", 1), ("draft", 2), ("activated", 1), ("activated", 2), ("activated", 1), ("stale", 1), ("draft", 3), ("draft", 4),
        ("discarded", 4), ("discarded", None)]
    by = {(e["kind"], e["ver"], e["t"]): e["detail"] for e in ev}
    assert by[("draft", 1, 100.0)] == {"note": "first", "created_by": "admin"}
    assert by[("activated", 1, 110.0)] == {"previous": None, "rollback": False}
    assert by[("activated", 2, 120.0)] == {"previous": 1, "rollback": False}
    assert by[("activated", 1, 130.0)] == {"previous": 2, "rollback": True}  # a rollback says so
    assert by[("stale", 1, 140.0)] == {"reason": "moved 14 px"}
    assert [e for e in ev if e["kind"] == "discarded" and e["ver"] is None][0]["detail"] == {"versions": [3]}
    assert [e["kind"] for e in g.events(limit=3)] == ["discarded", "discarded", "draft"]  # newest first, limited
    assert [e["t"] for e in g.events(since=145.0, limit=100)][-1] > 145.0


def test_failed_transitions_and_noops_log_nothing(make):
    g = make()
    g.create(REC, now=1.0)
    g.activate(1, now=2.0)
    n = len(g.events(limit=100))
    g.activate(1, now=3.0)  # already active
    with pytest.raises(GeometryError):
        g.activate(9)
    assert g.mark_stale("x", now=4.0) == 1 and g.mark_stale("again") is None  # nothing active any more
    assert len(g.events(limit=100)) == n + 1  # only the real stale
    assert not g.delete_draft(99) and g.discard_drafts() == []
    assert len(g.events(limit=100)) == n + 1


def test_other_events_can_be_logged_but_never_carry_credentials(make):
    g = make()
    g.log_event("drift_suspected", ver=1, detail={"shift_px": 13.4}, now=50.0)
    assert g.events()[0] == {"id": g.events()[0]["id"], "t": 50.0, "kind": "drift_suspected", "ver": 1, "detail": {"shift_px": 13.4}}
    with pytest.raises(GeometryError, match="credentials"):
        g.log_event("x", detail={"url": RTSP})


def test_samples_are_bucketed_in_the_database(make):
    g = make()
    t0 = 1_800_000_000  # a multiple of 3600
    for i, occ in enumerate([0, 1, 2, 3, 4, 4, 2, 0]):  # a sample every 15 min over two hours
        g.add_sample(1, counts(occupied=occ, vehicles=occ + 1), now=t0 + i * 900)
    g.add_sample(1, counts(occupied=9), now=t0 + 900)  # the same second again: ignored
    hourly = g.history(t0, t0 + 2 * 3600, 3600)
    assert [(b["t"], b["samples"], b["occupied_avg"], b["occupied_max"], b["occupied_min"]) for b in hourly] == [
        (t0, 4, 1.5, 3, 0), (t0 + 3600, 4, 2.5, 4, 0)]
    assert hourly[0]["capacity_avg"] == 4 and hourly[0]["vehicles_avg"] == 2.5 and hourly[0]["ver"] == 1
    assert hourly[0]["first"] == t0 and hourly[0]["last"] == t0 + 2700
    assert len(g.history(t0, t0 + 2 * 3600, 900)) == 8  # finer buckets: every sample
    assert g.history(t0 + 900, t0 + 1800, 3600)[0]["samples"] == 1  # the range is [since, until)
    assert g.history(t0 - 7200, t0 - 3600, 600) == []  # nothing there
    assert g.history(t0, t0 + 7200, 0)[0]["samples"] >= 1  # a silly bucket is clamped, never a division by zero


def test_cameras_history_is_separate_and_old_rows_are_pruned(make):
    a, b = make("front"), make("gate")
    now = 1_800_000_000
    a.add_sample(1, counts(), now=now - (SAMPLE_KEEP_DAYS + 1) * DAY)
    a.add_sample(1, counts(), now=now - DAY)
    b.add_sample(2, counts(capacity=7), now=now - DAY)
    a.log_event("old", now=now - (EVENT_KEEP_DAYS + 1) * DAY)
    a.log_event("recent", now=now - DAY)
    assert a.prune_history(now=now) == 1
    assert len(a.history(0, now, 3600)) == 1 and a.history(0, now, 3600)[0]["capacity_max"] == 4
    assert b.history(0, now, 3600)[0]["capacity_max"] == 7 and b.history(0, now, 3600)[0]["ver"] == 2
    assert [e["kind"] for e in a.events(limit=10)] == ["recent"] and b.events() == []
    assert b.prune_history(now=now) == 0


def test_a_range_query_over_a_long_history_uses_the_index_and_is_fast(tmp_path):
    live = LiveStore(tmp_path / "live.db")
    g = live.geometry("front")
    g.active()  # creates the tables
    t0 = 1_700_000_000
    live.con.execute("BEGIN")  # one transaction: the connection autocommits, 130k single inserts would each be a disk sync
    live.con.executemany("INSERT INTO geometry_samples VALUES ('front', ?, 1, 4, ?, 2, 2)",
                         ((t0 + i * 60, i % 5) for i in range(90 * 1440)))  # 90 days, a sample a minute: 129,600 rows
    live.con.executemany("INSERT INTO geometry_samples VALUES ('gate', ?, 1, 4, 1, 1, 1)", ((t0 + i * 60, ) for i in range(1000)))
    live.con.execute("COMMIT")
    plan = " ".join(str(r) for r in live.con.execute(
        "EXPLAIN QUERY PLAN SELECT t / 3600, COUNT(*) FROM geometry_samples WHERE camera_id = 'front' AND t >= ? AND t < ? GROUP BY 1",
        (t0 + 40 * DAY, t0 + 41 * DAY)))
    assert "SEARCH" in plan and "SCAN" not in plan  # a range search on (camera, time), not a scan of the table
    started = time.perf_counter()
    start = (t0 + 40 * DAY) // 300 * 300  # a window that starts on a bucket boundary
    day = g.history(start, start + DAY, 300)
    took = time.perf_counter() - started
    assert len(day) == 288 and all(b["samples"] == 5 for b in day) and took < 0.5, took
    assert len(g.history(t0, t0 + 90 * DAY, DAY)) in (90, 91)  # the whole history in ~90 daily buckets (the ends are partial days)
