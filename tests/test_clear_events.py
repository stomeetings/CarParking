"""scripts/clear_events.py: clearing the event log for a fresh start. Temp databases only; ./data is never touched."""
import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.history import HistoryDB, MOVES  # noqa: E402
from carparking.monitor import ParkingMonitor  # noqa: E402

spec = importlib.util.spec_from_file_location("clear_events", ROOT / "scripts" / "clear_events.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)

SHAPE = (1000, 2000, 3)
T0 = 1000.0
SLOTS = [{"id": 1, "name": "A1", "poly": [[0.05, 0.5], [0.23, 0.5], [0.23, 0.7], [0.05, 0.7]]},
         {"id": 2, "name": "A2", "poly": [[0.25, 0.5], [0.43, 0.5], [0.43, 0.7], [0.25, 0.7]]}]
CAR = {"poly": [(120, 420), (440, 420), (440, 695), (120, 695)], "box": (120, 420, 440, 695), "label": "car", "conf": 0.8, "state": 1}


class Cars:
    conf = targets = None

    def __init__(self):
        self.dets = [CAR]

    def __call__(self, frame, sliced=False):
        return [dict(d) for d in self.dets]


def make(tmp_path, camera="camera-1"):
    path = tmp_path / "h.db"
    db = HistoryDB(path, camera, detector="fake")
    det = Cars()
    mon = ParkingMonitor(det, db, fixed=SLOTS, match_rule="ground", hold=2, exit_hold=3, absent_free=True)
    return path, db, det, mon


def run(mon, t0, n):
    for t in range(n):
        mon.process(np.zeros(SHAPE, np.uint8), t=T0 + t0 + t)


def seeded(tmp_path):
    path, db, det, mon = make(tmp_path)
    run(mon, 0, 8)  # a car arrives: an entry event + an open session
    db.log_notice(1, "encroached", T0 + 9, "by A1")
    db.log_notice(0, "rejected", T0 + 10, "too_small")
    return path, db, det, mon


def counts(path):
    con = sqlite3.connect(path)
    try:
        return dict(con.execute("SELECT event, COUNT(*) FROM events GROUP BY event").fetchall())
    finally:
        con.close()


def test_dry_run_changes_nothing(tmp_path):
    path, db, det, mon = seeded(tmp_path)
    before = counts(path)
    res = ce.purge(path, "camera-1", "all", dry_run=True)
    assert res["matched"] == sum(before.values()) and res["deleted"] == 0 and counts(path) == before


def test_clears_the_whole_log_after_a_backup_and_keeps_the_sessions(tmp_path):
    path, db, det, mon = seeded(tmp_path)
    n_sessions = len(db.sessions(limit=50))
    res = ce.purge(path, "camera-1", "all", dry_run=False, backup_dir=tmp_path / "bak")
    assert res["deleted"] == res["matched"] > 0 and res["left"] == 0
    assert Path(res["backup"]).is_file() and sum(counts(Path(res["backup"])).values()) == res["matched"]  # the copy still has them
    assert db.events(limit=50, kinds=None) == []
    assert len(db.sessions(limit=50)) == n_sessions  # sessions are history, not the event log


def test_notices_only_leaves_entries_and_exits(tmp_path):
    path, db, det, mon = seeded(tmp_path)
    ce.purge(path, "camera-1", "notices", dry_run=False)
    assert set(counts(path)) <= set(MOVES) and counts(path)["entry"] == 1
    (tmp_path / "b").mkdir()
    path2, *_ = seeded(tmp_path / "b")
    ce.purge(path2, "camera-1", "moves", dry_run=False)
    assert set(counts(path2)) == {"encroached", "rejected"}


def test_only_the_named_camera_is_cleared(tmp_path):
    path, db, det, mon = seeded(tmp_path)
    other = HistoryDB(path, "camera-2", detector="fake")
    other.log_notice(0, "rejected", T0 + 20, "x")
    res = ce.purge(path, "camera-1", "all", dry_run=False)
    assert res["left"] == 1 and len(other.events(limit=10, kinds=None)) == 1  # camera-1 and "camera-1x" style names are not matched by prefix
    assert db.events(limit=10, kinds=None) == []


def test_ids_keep_counting_so_a_feed_never_sees_an_old_id_again(tmp_path):
    path, db, det, mon = seeded(tmp_path)
    last = db.events(limit=1, kinds=None)[0]["id"]
    res = ce.purge(path, "camera-1", "all", dry_run=False)
    assert res["max_id"] >= last
    db.log_notice(2, "encroached", T0 + 30, "again")
    assert db.events(limit=5, kinds=None)[0]["id"] > last  # a WebSocket client at `last` still receives it


def test_a_reopened_database_does_not_refill_the_empty_log_from_the_sessions(tmp_path):
    path, db, det, mon = seeded(tmp_path)
    ce.purge(path, "camera-1", "all", dry_run=False)
    assert HistoryDB(path, "camera-1", detector="fake").events(limit=10, kinds=None) == []  # (the backfill is for tables that do not exist)


def test_the_tracker_and_open_session_carry_on_through_a_purge(tmp_path):
    path, db, det, mon = seeded(tmp_path)
    ce.purge(path, "camera-1", "all", dry_run=False)
    run(mon, 20, 6)  # the car is still there: nothing new to log, nothing breaks
    st = mon.state()
    assert {s["id"]: s["state"] for s in st["slots"]}[1] == "BUSY" and db.events(limit=10, kinds=None) == []
    det.dets = []  # it leaves: the exit is logged against the session that was opened before the purge, with a fresh id
    run(mon, 40, 8)
    ev = db.events(limit=10, kinds=None)
    assert [e["event"] for e in ev] == ["exit"] and ev[0]["session_id"] is not None
    assert db.sessions(open_only=True) == []  # the session opened before the purge was closed by the exit
    assert db.summary()["events"] == 1


def test_a_clean_start_has_an_empty_feed(tmp_path):
    path = tmp_path / "fresh.db"
    HistoryDB(path, "camera-1", detector="fake")
    assert ce.purge(path, "camera-1", dry_run=False)["left"] == 0
    assert HistoryDB(path, "camera-1", detector="fake").events(limit=10, kinds=None) == []


def test_a_missing_database_is_an_error_not_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        ce.purge(tmp_path / "nope.db")
    assert not (tmp_path / "nope.db").exists()
    with pytest.raises(ValueError):
        ce._where(None, "bogus")


def test_cli_dry_run_by_default_and_yes_deletes(tmp_path, capsys):
    path, db, det, mon = seeded(tmp_path)
    assert ce.main(["--db", str(path), "--camera", "camera-1"]) == 0
    assert "dry run" in capsys.readouterr().out and sum(counts(path).values()) > 0
    assert ce.main(["--db", str(path), "--camera", "camera-1", "--yes", "--backup-dir", str(tmp_path / "bak")]) == 0
    assert counts(path) == {} and (tmp_path / "bak" / "h.db").is_file()
    assert ce.main(["--db", str(tmp_path / "missing.db")]) == 1
    assert ce.main(["--db", str(path), "--postgres"]) == 2  # needs --camera
    assert ce.main(["--db", str(path), "--postgres", "--pg-camera", "camera-1"]) == 2  # (no DSN here: refused before anything is connected)
