"""scripts/db_backup.py: consistent SQLite copies. Temp paths only; never touches ./data."""
import importlib.util
import sqlite3
from pathlib import Path

spec = importlib.util.spec_from_file_location("db_backup", Path(__file__).resolve().parents[1] / "scripts" / "db_backup.py")
db_backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_backup)


def make(path: Path, rows: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE t (x INTEGER)")
    c.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(rows)])
    c.commit()
    c.close()


def test_copies_every_db_with_relative_paths(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make(src / "a.db")
    make(src / "cameras" / "camera-2" / "b.db", 5)
    (src / "notes.txt").write_text("not a db")
    rows = db_backup.backup_all(src, dst)
    assert [r["db"] for r in rows] == ["a.db", "cameras/camera-2/b.db"]
    assert all(r["ok"] and r["check"] == "ok" for r in rows)
    assert sqlite3.connect(dst / "cameras" / "camera-2" / "b.db").execute("SELECT COUNT(*) FROM t").fetchone()[0] == 5
    assert not (dst / "notes.txt").exists()


def test_copy_includes_uncommitted_journal_state_only_when_committed(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make(src / "a.db")
    writer = sqlite3.connect(src / "a.db")
    writer.execute("INSERT INTO t VALUES (99)")  # an open, uncommitted write
    rows = db_backup.backup_all(src, dst)
    writer.rollback()
    writer.close()
    assert rows[0]["ok"]
    assert sqlite3.connect(dst / "a.db").execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3


def test_source_is_untouched_and_corrupt_file_fails(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make(src / "a.db")
    before = (src / "a.db").read_bytes()
    (src / "bad.db").write_bytes(b"this is not a sqlite database" * 50)
    rows = {r["db"]: r for r in db_backup.backup_all(src, dst)}
    assert rows["a.db"]["ok"] and not rows["bad.db"]["ok"]
    assert (src / "a.db").read_bytes() == before


def test_main_exit_codes(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make(src / "a.db")
    assert db_backup.main(["x", str(src), str(dst)]) == 0
    assert db_backup.main(["x", str(tmp_path / "empty"), str(dst)]) == 1  # nothing to back up is a failure, not a pass
    assert db_backup.main(["x"]) == 2
