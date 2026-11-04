"""Notices in the event log (post-demo items 1 and 2): `encroached` / `encroach_cleared` of the spillage layer and `rejected` of the non-vehicle
filter are stored in the events table next to entries and exits, but every older reader still sees only movements; the API serves them on
request (/api/events?kinds=). tmp_path only: no network, no model, no ./data."""
import csv
import io
import sqlite3
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import Config, create_app  # noqa: E402
from carparking.history import MOVES, NOTICES, HistoryDB  # noqa: E402
from carparking.history_pg import Scope, SqliteHistory, summarize  # noqa: E402
from carparking.monitor import ParkingMonitor  # noqa: E402
from carparking.obstacle_filter import REASON_TEXT, RejectLog  # noqa: E402

TZ = ZoneInfo("Asia/Karachi")
T0 = 1_792_000_000.0
URL = "rtsp://admin:secret@cam.local:554/Streaming/Channels/101"


def filled(path) -> HistoryDB:
    """One session (entry + exit) and the three notices, interleaved."""
    db = HistoryDB(path, "Front", TZ, detector="yolov8s")
    db.apply({"slot": 1, "state": 1, "time": T0, "initial": False})
    db.log_notice(2, "encroached", T0 + 10, "by A1, 42 % of the bay: free but not usable")
    db.log_notice(0, "rejected", T0 + 20, "too_small: too small for the bay outside the bays, 40x70 px, not counted as a vehicle")
    db.apply({"slot": 1, "state": 0, "time": T0 + 100, "initial": False})
    db.log_notice(2, "encroach_cleared", T0 + 120, "cover below the leave share")
    return db


# ---- HistoryDB -----------------------------------------------------------------------------------------------------------------------
def test_notices_are_stored_but_the_default_reader_sees_only_movements(tmp_path):
    db = filled(tmp_path / "h.db")
    assert [e["event"] for e in db.events()] == ["exit", "entry"]  # what every older reader (Streamlit, API default) gets
    assert [e["event"] for e in db.events(kinds=MOVES)] == ["exit", "entry"]
    assert [e["event"] for e in db.events(kinds=NOTICES)] == ["encroach_cleared", "rejected", "encroached"]
    assert [e["event"] for e in db.events(kinds=None)] == ["encroach_cleared", "exit", "rejected", "encroached", "entry"]
    assert [e["event"] for e in db.events(kinds="rejected")] == ["rejected"]  # a bare string is one kind
    assert [e["event"] for e in db.events(limit=2, kinds=None)] == ["encroach_cleared", "exit"]


def test_a_notice_row_has_the_event_fields_and_no_session(tmp_path):
    db = filled(tmp_path / "h.db")
    enc = db.events(kinds="encroached")[0]
    assert set(enc) == {"id", "camera", "detector", "slot_id", "event", "time", "ts", "session_id", "note"}
    assert (enc["slot_id"], enc["session_id"], enc["ts"]) == (2, None, T0 + 10) and enc["note"].startswith("by A1")
    assert enc["time"] == db.iso(T0 + 10) and enc["camera"] == "Front"
    assert db.events(kinds="rejected")[0]["slot_id"] == 0  # concerns no bay
    assert db.sessions()[0]["slot_id"] == 1 and len(db.sessions()) == 1  # no session was opened by a notice
    assert db.open_by_slot() == {}


def test_since_id_polling_works_across_kinds(tmp_path):
    db = filled(tmp_path / "h.db")
    ids = [e["id"] for e in reversed(db.events(kinds=None))]
    assert [e["event"] for e in db.events(since_id=ids[2], kinds=None)] == ["encroach_cleared", "exit"]
    assert [e["event"] for e in db.events(since_id=ids[2])] == ["exit"]


def test_summary_and_csv_count_movements_only_unless_asked(tmp_path):
    db = filled(tmp_path / "h.db")
    s = db.summary()
    assert s["events"] == 2 and s["last_event"]["event"] == "exit"
    rows = list(csv.DictReader(io.StringIO(db.events_csv())))
    assert [r["event"] for r in rows] == ["entry", "exit"]
    rows = list(csv.DictReader(io.StringIO(db.events_csv(kinds=None))))
    assert [r["event"] for r in rows] == ["entry", "encroached", "rejected", "exit", "encroach_cleared"]
    assert rows[1]["session_id"] == "" and rows[1]["note"].startswith("by A1")
    assert [r["event"] for r in csv.DictReader(io.StringIO(db.events_csv(kinds=NOTICES)))] == ["encroached", "rejected", "encroach_cleared"]


def test_an_unknown_notice_kind_is_refused(tmp_path):
    db = HistoryDB(tmp_path / "h.db", "Front", TZ, detector="yolov8s")
    with pytest.raises(ValueError, match="unknown notice"):
        db.log_notice(1, "entry", T0)  # a movement is never written without its session
    assert db.events(kinds=None) == []


def test_scopes_do_not_share_notices(tmp_path):
    a = filled(tmp_path / "h.db")
    b = HistoryDB(tmp_path / "h.db", "Front", TZ, detector="yolov8n")
    b.log_notice(3, "rejected", T0 + 5, "aspect: x")
    assert [e["slot_id"] for e in b.events(kinds=NOTICES)] == [3] and len(a.events(kinds=NOTICES)) == 3


# ---- the other readers of the events table -----------------------------------------------------------------------------------------
def test_the_sqlite_reader_and_the_analytics_ignore_notices(tmp_path):
    db = filled(tmp_path / "h.db")
    lite, s = SqliteHistory(tmp_path / "h.db"), Scope("front", "Front", "yolov8s", TZ)
    assert [e["event"] for e in lite.events(s, 10)] == ["exit", "entry"] and lite.events(s, 10) == db.events(10)
    assert lite.summary(s) == db.summary()
    assert lite.events_csv(s).count("\n") == 1 + 2
    totals = lite.analytics(s, 30, all_scopes=True, now=T0 + 3600)["totals"]
    assert (totals["entries"], totals["exits"]) == (1, 1)  # a notice is neither
    got = summarize([(T0, "entry"), (T0 + 5, "encroached"), (T0 + 6, "rejected"), (T0 + 7, "exit")], [], [], TZ, days=1, now=T0 + 60)
    assert (got["totals"]["entries"], got["totals"]["exits"]) == (1, 1)


# ---- RejectLog -----------------------------------------------------------------------------------------------------------------------
def rej(slot=2, reason="pylon", box=(10, 20, 160, 220)):
    return {"slot": slot, "reason": reason, "box": list(box), "colors": None}


def test_a_rejection_is_logged_after_hold_seconds_and_frames_then_rate_limited():
    log, out = RejectLog(hold=5, frames=2, cooldown=600), []
    for t in range(0, 40, 1):
        out += log.update([rej()], T0 + t, {2: "A2"})
    assert len(out) == 1 and out[0]["event"] == "rejected" and out[0]["slot"] == 2 and out[0]["time"] == T0 + 5
    assert out[0]["note"].startswith("pylon: cone-coloured object in bay A2, 150x200 px")
    out2 = []
    for t in range(40, 700, 10):
        out2 += log.update([rej()], T0 + t)
    assert [e["time"] for e in out2] == [T0 + 610]  # again only after the cooldown


def test_a_flash_is_not_logged_and_slow_frames_still_need_two_frames():
    log = RejectLog(hold=5, frames=2)
    assert log.update([rej()], T0) == [] and log.update([], T0 + 1) == []
    assert log.update([rej()], T0 + 100) == []  # gap > 60 s: starts over
    slow = RejectLog(hold=5, frames=2)
    assert slow.update([rej()], T0) == []
    assert len(slow.update([rej()], T0 + 30)) == 1  # 30 s between frames: both conditions met on the second
    one = RejectLog(hold=0, frames=2)
    assert one.update([rej()], T0) == []  # hold is met at once, the second frame is not


def test_each_bay_and_reason_is_its_own_episode_and_unbayed_is_slot_zero():
    log, out = RejectLog(hold=0, frames=1), []
    out += log.update([rej(2, "pylon"), rej(2, "pylon"), rej(3, "pylon"), rej(2, "aspect"), rej(None, "too_small")], T0)
    assert sorted((e["slot"], e["note"].split(":")[0]) for e in out) == [(0, "too_small"), (2, "aspect"), (2, "pylon"), (3, "pylon")]
    assert "outside the bays" in [e for e in out if e["slot"] == 0][0]["note"]
    assert set(REASON_TEXT) == {"too_small", "aspect", "pylon", "shadow"}


def test_the_log_forgets_old_episodes():
    log = RejectLog(hold=0, frames=1, cooldown=100, gap=10)
    log.update([rej(i) for i in range(1, 20)], T0)
    assert len(log._seen) == 19
    log.update([], T0 + 500)
    assert log._seen == {}


# ---- the monitor writes them -----------------------------------------------------------------------------------------------------------
SHAPE = (1000, 2000, 3)
BAYS = [{"id": i + 1, "name": f"A{i + 1}", "poly": [[x, 0.5], [x + 0.18, 0.5], [x + 0.18, 0.7], [x, 0.7]]} for i, x in enumerate((0.10, 0.30, 0.50))]
WIDE = {"poly": [(220, 420), (744, 420), (744, 695), (220, 695)], "box": (220, 420, 744, 695), "label": "car", "conf": 0.8, "state": 1}
CONE = {"poly": [(1100, 610), (1140, 610), (1140, 690), (1100, 690)], "box": (1100, 610, 1140, 690), "label": "car", "conf": 0.6, "state": 1}
GREY = np.random.RandomState(1).randint(90, 130, SHAPE).astype(np.uint8)


class Static:
    def __init__(self, dets):
        self.dets, self.conf, self.targets = dets, None, None

    def __call__(self, frame, sliced=False):
        return [dict(d) for d in self.dets]


def monitor(tmp_path, dets, **kw):
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    return ParkingMonitor(Static(dets), db, fixed=BAYS, match_rule="ground", hold=2, exit_hold=3, absent_free=True, **kw), db


def test_the_monitor_logs_encroachment_and_rejections_and_nothing_when_both_are_off(tmp_path):
    mon, db = monitor(tmp_path, [WIDE, CONE], spillage=1, obstacle_filter=1)
    for t in range(30):
        mon.process(GREY, t=T0 + t)
    notices = db.events(kinds=NOTICES)
    assert sorted(e["event"] for e in notices) == ["encroached", "rejected"]
    enc = next(e for e in notices if e["event"] == "encroached")
    rejd = next(e for e in notices if e["event"] == "rejected")
    assert enc["slot_id"] == 2 and enc["note"].startswith("by A1, 4") and enc["note"].endswith("% of the bay: free but not usable")
    assert rejd["slot_id"] == 3 and rejd["note"].startswith("too_small: too small for the bay in bay A3, 40x80 px")
    assert [e["event"] for e in db.events()] == ["entry"] and db.open_by_slot() == {1: 1}  # only the wide car's movement; no session for a notice
    assert mon.notice_error == ""


def test_nothing_is_written_with_the_layers_off(tmp_path):
    mon, db = monitor(tmp_path, [WIDE, CONE])
    for t in range(30):
        mon.process(GREY, t=T0 + t)
    assert db.events(kinds=NOTICES) == []
    assert {e["event"] for e in db.events(kinds=None)} == {"entry"}  # the cone-sized box made a bay busy, as it always did


def test_clearing_is_logged_when_the_car_backs_off_and_when_spillage_is_switched_off(tmp_path):
    state = {"dets": [WIDE]}

    class Switch(Static):
        def __call__(self, frame, sliced=False):
            return [dict(d) for d in state["dets"]]
    db = HistoryDB(tmp_path / "h.db", "cam", detector="fake")
    mon = ParkingMonitor(Switch([]), db, fixed=BAYS, match_rule="ground", hold=2, exit_hold=3, absent_free=True, spillage=1)
    for t in range(15):
        mon.process(GREY, t=T0 + t)
    assert [e["event"] for e in db.events(kinds=NOTICES)] == ["encroached"]
    mon.tracker.set_spillage(0)
    mon.process(GREY, t=T0 + 15)
    got = [e["event"] for e in db.events(kinds=NOTICES)]
    assert got == ["encroach_cleared", "encroached"]
    assert db.events(kinds=NOTICES)[0]["note"] == "spillage switched off"
    mon.process(GREY, t=T0 + 16)
    assert len(db.events(kinds=NOTICES)) == 2  # handed out once


def test_a_failing_write_never_stops_the_frame(tmp_path):
    mon, db = monitor(tmp_path, [WIDE], spillage=1)

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    db.log_notice = boom
    for t in range(15):
        mon.process(GREY, t=T0 + t)
    assert "OperationalError" in mon.notice_error and mon.tracker.avail  # the layer went on, the error is reported
    del db.log_notice  # the real method again: the next notice is written
    mon.process(GREY, t=T0 + 15)
    mon.tracker.set_spillage(0)
    mon.process(GREY, t=T0 + 16)
    assert mon.notice_error == "" and [e["event"] for e in db.events(kinds=NOTICES)] == ["encroach_cleared"]


# ---- the API ---------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def api(tmp_path):
    env = {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
           "CARPARK_RTSP_URL": URL, "CARPARK_API_TOKEN": "t0k3n", "CARPARK_SPA_DIR": str(tmp_path / "dist")}
    with TestClient(create_app(Config(env), names_of=lambda w: (["car"], ["person"]))) as c:
        scope = c.get("/api/settings").json()["effective"]["db_detector"]
        filled_db = HistoryDB(env["CARPARK_HISTORY_DB"], "Camera 1", detector=scope)
        filled_db.apply({"slot": 1, "state": 1, "time": T0, "initial": False})
        filled_db.log_notice(2, "encroached", T0 + 10, "by A1")
        filled_db.log_notice(0, "rejected", T0 + 20, "pylon: x")
        filled_db.apply({"slot": 1, "state": 0, "time": T0 + 100, "initial": False})
        yield c


def test_api_events_default_is_movements_and_kinds_selects_the_rest(api):
    assert [e["event"] for e in api.get("/api/events").json()] == ["exit", "entry"]  # an older client sees what it always saw
    assert [e["event"] for e in api.get("/api/events?kinds=moves").json()] == ["exit", "entry"]
    r = api.get("/api/events?kinds=notices")
    assert [e["event"] for e in r.json()] == ["rejected", "encroached"] and r.headers["X-History-Source"] == "sqlite"
    assert r.json()[0]["session_id"] is None and r.json()[0]["slot_id"] == 0
    assert [e["event"] for e in api.get("/api/events?kinds=all").json()] == ["exit", "rejected", "encroached", "entry"]
    assert [e["event"] for e in api.get("/api/events?kinds=all&limit=1").json()] == ["exit"]
    assert api.get("/api/events?kinds=bogus").status_code == 422


def test_api_csv_and_summary(api):
    assert api.get("/api/events.csv").text.count("\n") == 1 + 2
    assert "encroached" in api.get("/api/events.csv?kinds=all").text and api.get("/api/events.csv?kinds=all").text.count("\n") == 1 + 4
    assert api.get("/api/events.csv?kinds=notices").text.count("\n") == 1 + 2
    assert api.get("/api/summary").json()["events"] == 2
