"""Versioned geometry records per camera (carparking/geometry_store.py): lifecycle, rollback, both backends. SQLite runs
in tmp_path; PostgreSQL only with CARPARK_TEST_PG_DSN (a scratch database, like test_registry). Never touches ./data."""
import os
import sqlite3

import pytest

from carparking.geometry_store import KEEP_SUPERSEDED, GeometryError, open_geometry
from carparking.live import LiveStore

REC = {"surface": {"polygon_px": [[0.1, 0.4], [0.9, 0.4], [0.9, 0.95], [0.1, 0.95]], "score": 0.93},
       "calibration": {"f": 1200.5, "pitch": 0.52, "h": 6.0, "confidence": "HIGH"}}
RTSP = "rtsp://admin:secret@192.168.20.10:554/Streaming/Channels/101"


@pytest.fixture(params=["sqlite", "pg"])
def make(request, tmp_path):
    """make(camera_id) -> GeometryStore on a fresh backend."""
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
    with psycopg.connect(dsn, autocommit=True) as con:  # a scratch database: start empty (geometry goes with its camera)
        con.execute("DELETE FROM cameras")
    for cid in ("front", "gate"):
        reg.create(Camera(cid, cid.title(), RTSP, primary=cid == "front"))
    return lambda cam="front": pg_geometry(dsn, cam)


def test_uncalibrated_by_default(make):
    g = make()
    assert g.active() is None and g.list() == [] and g.get(1) is None
    assert g.mark_stale("x") is None and not g.delete_draft(1)


def test_active_version_is_a_cheap_poll_that_sees_every_change(make):
    g = make()
    assert g.active_version() is None
    g.create(REC)
    g.create(REC)
    assert g.active_version() is None  # drafts are not active
    g.activate(1)
    assert g.active_version() == 1
    g.activate(2)
    assert g.active_version() == 2 and make().active_version() == 2  # another handle sees it at once: no cache
    g.rollback()
    assert g.active_version() == 1
    g.mark_stale("moved")
    assert g.active_version() is None


def test_discard_drafts_only_ever_deletes_drafts(make):
    g = make()
    for _ in range(4):
        g.create(REC)
    g.activate(1)
    g.activate(2)
    g.mark_stale()  # 2 stale, 1 superseded
    assert g.discard_drafts() == [3, 4] and sorted(v["ver"] for v in g.list()) == [1, 2]
    assert g.discard_drafts() == []
    assert {v["ver"]: v["state"] for v in g.list()} == {1: "superseded", 2: "stale"}


def test_create_is_a_numbered_draft_not_active(make):
    g = make()
    assert (g.create(REC, created_by="admin", note="first", now=100.0), g.create(REC, now=101.0)) == (1, 2)
    one = g.get(1)
    assert one["state"] == "draft" and one["record"] == REC and one["note"] == "first" and one["created_by"] == "admin"
    assert one["created_at"] == 100.0 and one["activated_at"] is None
    assert g.active() is None and [v["ver"] for v in g.list()] == [2, 1]  # newest first
    assert [v["ver"] for v in g.list("draft")] == [2, 1] and g.list("active") == []


def test_activate_supersedes_and_rolls_back(make):
    g = make()
    for t in (1, 2, 3):
        g.create({**REC, "n": t})
    assert g.activate(1, now=10.0)["state"] == "active" and g.active()["ver"] == 1
    assert g.activate(2, now=20.0)["ver"] == 2
    assert (g.get(1)["state"], g.get(2)["state"], g.get(3)["state"]) == ("superseded", "active", "draft")
    assert g.activate(2, now=25.0)["activated_at"] == 20.0  # already active: unchanged
    assert g.rollback(now=30.0)["ver"] == 1 and g.active()["record"]["n"] == 1
    assert g.get(2)["state"] == "superseded"
    assert g.rollback(now=40.0)["ver"] == 2  # rolling back again goes to the one active before (2)
    assert len([v for v in g.list() if v["state"] == "active"]) == 1


def test_rollback_needs_an_earlier_version(make):
    g = make()
    with pytest.raises(GeometryError, match="no earlier"):
        g.rollback()
    g.create(REC)
    g.activate(1)
    with pytest.raises(GeometryError, match="no earlier"):
        g.rollback()


def test_stale_means_uncalibrated_and_is_final(make):
    g = make()
    g.create(REC)
    g.activate(1)
    assert g.mark_stale("moved 12 px") == 1
    assert g.active() is None and g.get(1)["state"] == "stale" and "moved 12 px" in g.get(1)["note"]
    with pytest.raises(GeometryError, match="stale"):
        g.activate(1)
    g.create(REC)
    assert g.activate(2)["ver"] == 2 and g.get(1)["state"] == "stale"  # a new setup replaces it; stale stays stale
    with pytest.raises(GeometryError, match="no earlier"):
        g.rollback()  # stale is never a rollback target


def test_only_the_last_superseded_versions_are_kept(make):
    g = make()
    total = KEEP_SUPERSEDED + 4
    for i in range(total):
        g.create({**REC, "i": i})
        g.activate(i + 1, now=float(i))
    states = [v["state"] for v in g.list()]
    assert states.count("active") == 1 and states.count("superseded") == KEEP_SUPERSEDED
    assert g.get(1) is None and g.get(total - KEEP_SUPERSEDED) is not None  # the oldest ones are gone
    assert g.create(REC) == total + 1  # numbers keep rising


def test_delete_draft_only(make):
    g = make()
    g.create(REC)
    g.create(REC)
    g.activate(1)
    with pytest.raises(GeometryError, match="only drafts"):
        g.delete_draft(1)
    assert g.delete_draft(2) and not g.delete_draft(2) and g.active()["ver"] == 1


def test_a_draft_can_be_edited_and_nothing_else(make):
    g = make()
    g.create({**REC, "suggest": {"parent": 1, "mode": "quick"}, "grid": [{"id": 1}]})
    assert g.update_draft(1, {**REC, "suggest": {"parent": 1, "mode": "quick", "edited": True}, "grid": [{"id": 1}, {"id": 2}]})
    row = g.get(1)
    assert row["state"] == "draft" and row["record"]["grid"] == [{"id": 1}, {"id": 2}] and row["record"]["suggest"]["edited"] is True
    assert not g.update_draft(9, REC)  # an unknown version
    with pytest.raises(GeometryError):
        g.update_draft(1, {**REC, "url": RTSP})  # never a camera password
    assert g.get(1)["record"]["grid"] == [{"id": 1}, {"id": 2}]
    g.activate(1)
    with pytest.raises(GeometryError, match="only drafts"):
        g.update_draft(1, REC)  # history is not rewritten
    assert g.get(1)["record"]["grid"] == [{"id": 1}, {"id": 2}]


def test_the_events_name_a_slot_suggestions_parent(make):
    g = make()
    g.create(REC)
    g.activate(1)
    g.create({**REC, "suggest": {"parent": 1, "mode": "full"}})
    g.activate(2)
    kinds = {(e["kind"], e["ver"]): e["detail"] for e in g.events()}
    assert kinds[("draft", 2)]["parent"] == 1 and kinds[("draft", 2)]["suggest"] == "full"
    assert kinds[("activated", 2)]["parent"] == 1 and kinds[("activated", 2)]["suggest"] == "full"
    assert "suggest" not in kinds[("draft", 1)]


def test_unknown_version_and_bad_records(make):
    g = make()
    with pytest.raises(GeometryError, match="no geometry version 7"):
        g.activate(7)
    with pytest.raises(GeometryError, match="object"):
        g.create([1, 2])
    with pytest.raises(GeometryError, match="credentials"):
        g.create({"source": {"url": RTSP}})
    with pytest.raises(GeometryError, match="unknown state"):
        g.list("nope")
    assert g.list() == []  # nothing half-written


def test_cameras_are_independent(make):
    a, b = make("front"), make("gate")
    a.create(REC)
    a.activate(1)
    assert b.active() is None and b.list() == [] and b.create(REC) == 1
    b.activate(1)
    assert a.active()["camera_id"] == "front" and b.active()["camera_id"] == "gate"


def test_sqlite_table_appears_only_when_used_and_other_tables_untouched(tmp_path):
    path = tmp_path / "live.db"
    live = LiveStore(path)
    tables = lambda: {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}  # noqa: E731
    before = tables()
    assert before == {"live_config", "live_state", "live_options", "live_commands", "sqlite_sequence"}
    live.geometry("front")  # asking for the store creates nothing
    assert tables() == before
    assert live.geometry("front").active() is None  # first real use creates it
    assert tables() - before == {"camera_geometry", "geometry_samples", "geometry_events"}  # the geometry tables, together
    cfg = {"mode": 5}
    assert live.set_config(cfg) == 1 and live.config() == (cfg, 1)  # settings unaffected


def test_database_allows_only_one_active_per_camera(tmp_path):
    live = LiveStore(tmp_path / "live.db")
    g = live.geometry("front")
    g.create(REC)
    g.create(REC)
    g.activate(1)
    with pytest.raises(sqlite3.IntegrityError):
        live.con.execute("UPDATE camera_geometry SET state = 'active' WHERE ver = 2")


def test_failed_transition_changes_nothing(tmp_path):
    live = LiveStore(tmp_path / "live.db")
    g = live.geometry("front")
    g.create(REC)
    g.activate(1)
    with pytest.raises(GeometryError):
        g.activate(9)
    assert g.active()["ver"] == 1 and len(g.list()) == 1


def test_open_geometry_picks_the_backend(tmp_path):
    live = LiveStore(tmp_path / "live.db")
    assert type(open_geometry(live, "front", {}).b).__name__ == "_Sqlite"
    assert type(open_geometry(live, "front", {"CARPARK_DATABASE_URL": "postgresql://x/y"}).b).__name__ == "_Pg"
