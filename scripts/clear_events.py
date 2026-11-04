"""Clear the event log (entries, exits and the spillage / obstacle notices) of a camera's history database, for a fresh start.

    python scripts/clear_events.py --db data/parking_history.db --camera camera-1            # DRY RUN: counts only, nothing is written
    python scripts/clear_events.py --db data/parking_history.db --camera camera-1 --yes      # backup, then delete
    ... --kinds notices      only encroached / encroach_cleared / rejected   (moves = only entry / exit; default all)
    ... --postgres           also the central copy (parking_events) of that camera: DSN from $CARPARK_DATABASE_URL
    ... --pg-camera camera-1 the PostgreSQL camera id when it is not the SQLite name (events.camera holds the display name, e.g. "Camera 1";
                             PostgreSQL holds the registry id, camera-1). --camera is the SQLite name: check with a dry run first.

What it does and does not touch:
  * only rows of the `events` table (and `parking_events` with --postgres); sessions (`history`), slots, settings and the live tables stay.
    Open sessions stay open, so the Sessions tab and the analytics keep their history; only the event log / live ticker start empty.
  * the table and its AUTOINCREMENT counter are kept: new events get ids above every old one, so a browser's WebSocket feed, the PostgreSQL
    copy's high-water mark and the Event log never see an id twice. (An emptied `events` table is NOT rebuilt from `history` on the next start:
    that only happens when the table does not exist.)
  * a database that is not there is an error, never created. Safe while the worker runs (one short transaction; SQLite waits up to 30 s).
  * with --yes a consistent copy of the database goes to --backup-dir (default recordings/event_purge_backups/<time>/), outside ./data.
A bay that is ENCROACHED when you clear stays so in the worker's memory; its later `encroach_cleared` notice then has no `encroached` before it.
To make the worker start from nothing as well, restart it (docker compose restart worker) after clearing."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

KINDS = {"all": None, "moves": ("entry", "exit"), "notices": ("encroached", "encroach_cleared", "rejected")}


def _where(camera: str | None, kinds: str) -> tuple[str, list]:
    if kinds not in KINDS:
        raise ValueError(f"kinds must be one of {sorted(KINDS)}")
    parts, args = [], []
    if camera:  # the scope of an event is "camera" or "camera | detector"
        parts.append("(camera = ? OR camera LIKE ?)")
        args += [camera, camera + " | %"]
    if KINDS[kinds] is not None:
        parts.append(f"event IN ({','.join('?' * len(KINDS[kinds]))})")
        args += list(KINDS[kinds])
    return (" WHERE " + " AND ".join(parts)) if parts else "", args


def purge(db: str | Path, camera: str | None = None, kinds: str = "all", dry_run: bool = True, backup_dir: str | Path | None = None) -> dict:
    """-> {"db", "camera", "kinds", "dry_run", "matched": rows that match, "deleted", "left": rows still in the table, "max_id", "backup"}."""
    path = Path(db)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such database (nothing is created)")
    where, args = _where(camera, kinds)
    con = sqlite3.connect(str(path), timeout=30)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'events'").fetchone():
            raise ValueError(f"{path}: no events table")
        matched = con.execute(f"SELECT COUNT(*) FROM events{where}", args).fetchone()[0]
        out = {"db": str(path), "camera": camera, "kinds": kinds, "dry_run": dry_run, "matched": matched, "deleted": 0, "backup": None}
        if not dry_run and matched:
            if backup_dir is not None:
                target = Path(backup_dir) / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                dst = sqlite3.connect(target)
                try:
                    con.backup(dst)
                    if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise RuntimeError(f"backup {target} failed its integrity check: nothing was deleted")
                finally:
                    dst.close()
                out["backup"] = str(target)
            with con:  # one transaction
                out["deleted"] = con.execute(f"DELETE FROM events{where}", args).rowcount
        out["left"] = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        out["max_id"] = (con.execute("SELECT seq FROM sqlite_sequence WHERE name = 'events'").fetchone() or [0])[0]
        return out
    finally:
        con.close()


def purge_postgres(dsn: str, camera: str, kinds: str = "all", dry_run: bool = True) -> dict:
    """The central copy: rows of parking_events for `camera` (it only ever holds entry / exit, so kinds=notices matches nothing)."""
    import psycopg
    cond, args = "camera_id = %s", [camera]
    if KINDS[kinds] is not None:
        cond += " AND event = ANY(%s)"
        args.append(list(KINDS[kinds]))
    with psycopg.connect(dsn, autocommit=True) as con:
        n = con.execute(f"SELECT COUNT(*) FROM parking_events WHERE {cond}", args).fetchone()[0]
        if not dry_run and n:
            n = con.execute(f"DELETE FROM parking_events WHERE {cond}", args).rowcount
    return {"postgres": True, "camera": camera, "kinds": kinds, "dry_run": dry_run, "matched" if dry_run else "deleted": n}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="the history database (e.g. data/parking_history.db or data/cameras/<id>/parking_history.db)")
    ap.add_argument("--camera", help="only the events of this camera (default: every camera in the file)")
    ap.add_argument("--kinds", choices=sorted(KINDS), default="all")
    ap.add_argument("--yes", action="store_true", help="really delete (without it: a dry run)")
    ap.add_argument("--backup-dir", help="where the pre-delete copy goes (default recordings/event_purge_backups/<time>)")
    ap.add_argument("--no-backup", action="store_true", help="skip the copy (not recommended)")
    ap.add_argument("--postgres", action="store_true", help="also the central parking_events of --camera ($CARPARK_DATABASE_URL)")
    ap.add_argument("--pg-camera", help="the camera id in PostgreSQL when it differs from the name in the SQLite events (\"Camera 1\" there, "
                                        "camera-1 here); default: --camera")
    a = ap.parse_args(argv)
    backup = None if a.no_backup else (a.backup_dir or Path("recordings") / "event_purge_backups" / time.strftime("%Y%m%d-%H%M%S"))
    try:
        res = purge(a.db, a.camera, a.kinds, dry_run=not a.yes, backup_dir=backup)
        print(json.dumps(res))
        if a.postgres:
            pg_cam = a.pg_camera or a.camera
            if not pg_cam:
                print("--postgres needs --camera (or --pg-camera)", file=sys.stderr)
                return 2
            dsn = os.environ.get("CARPARK_DATABASE_URL")
            if not dsn:
                print("--postgres: CARPARK_DATABASE_URL is not set", file=sys.stderr)
                return 2
            print(json.dumps(purge_postgres(dsn, pg_cam, a.kinds, dry_run=not a.yes)))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as e:
        print(f"clear_events: {e}", file=sys.stderr)
        return 1
    if not a.yes:
        print("dry run: nothing was changed. Add --yes to delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
