"""Consistent copies of every SQLite database under a folder, using SQLite's online backup API (safe while a worker writes).

    python scripts/db_backup.py <src_dir> <dst_dir>

Every *.db under src_dir is copied to the same relative path under dst_dir (journals / WAL files are not copied: the backup
API produces a complete file), then checked with PRAGMA integrity_check. Prints one JSON line per database and exits 1 if
any copy fails its check. The source is opened read-only; nothing under src_dir is ever written."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def backup_all(src: Path, dst: Path) -> list[dict]:
    out = []
    for db in sorted(src.rglob("*.db")):
        rel = db.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        row = {"db": rel.as_posix(), "ok": False, "bytes": 0, "check": ""}
        try:
            s = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=30)
            d = sqlite3.connect(target)
            try:
                s.backup(d)
                row["check"] = d.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                d.close()
                s.close()
            row["bytes"] = target.stat().st_size
            row["ok"] = row["check"] == "ok"
        except sqlite3.Error as e:
            row["check"] = f"{type(e).__name__}: {e}"[:200]
        out.append(row)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    rows = backup_all(Path(argv[1]), Path(argv[2]))
    for r in rows:
        print(json.dumps(r))
    return 0 if rows and all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
