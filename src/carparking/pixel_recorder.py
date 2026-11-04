"""The pixel recorder (post-demo item 4): the part of pixel_replay.py that only talks to the running stack and writes files. Standard
library only (no OpenCV, numpy or tracker import), so it runs on a plain Python on the host. Format and use: see pixel_replay.py."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

FORMAT = 1
STATE_NAME = {0: "FREE", 1: "BUSY", None: "UNKNOWN"}
STATE_VALUE = {v: k for k, v in STATE_NAME.items()}
DYNAMIC_BASE = 1000  # = fixed_slots.DYNAMIC_BASE (a test keeps them equal): ids from here on are vehicle-driven bays
# worker settings (GET /api/settings: values) -> SlotTracker keyword arguments; only the ones present are passed, so an absent
# one keeps the tracker's own default instead of a number invented here
SETTING_TO_TRACKER = {"hold": "hold", "exit_hold": "exit_hold", "absent_free": "absent_free", "occlusion": "occlusion_cover",
                      "hidden_max": "hidden_max", "exit_frames": "exit_frames"}


def ensure_outside(path, *protected) -> Path:
    """-> path resolved; ValueError when it lies inside (or is) one of the protected directories (the live ./data above all)."""
    p = Path(path).resolve()
    for q in protected:
        q = Path(q).resolve()
        if p == q or q in p.parents:
            raise ValueError(f"{p} is inside {q}, which holds the live stack's files; record somewhere else")
    return p


# ---- recording -------------------------------------------------------------------------------------------------------------------
class PixelRecorder:
    """Writes one recording directory. add() appends a frame; nothing is ever read back or deleted."""

    def __init__(self, directory, header: dict):
        self.dir = Path(directory)
        if (self.dir / "header.json").exists():
            raise FileExistsError(f"{self.dir} already holds a recording")
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "header.json").write_text(json.dumps({"format": FORMAT, **header}, indent=1), encoding="utf-8")
        self._log = open(self.dir / "frames.jsonl", "a", encoding="utf-8")
        self.n, self.bytes = 0, 0

    def add(self, t: float, jpeg: bytes, dets: list[dict], states: dict) -> None:
        self.n += 1
        name = f"frames/{self.n:06d}.jpg"
        (self.dir / name).write_bytes(jpeg)
        self._log.write(json.dumps({"i": self.n, "t": float(t), "file": name, "dets": dets, "states": states},
                                   separators=(",", ":")) + "\n")
        self._log.flush()
        self.bytes += len(jpeg)

    def close(self) -> None:
        self._log.close()


def _slots_of(snap: dict) -> list[dict]:
    """The fixed map in a status snapshot: id, name, poly (fractions), capacity. Vehicle-driven bays are left out."""
    return [{"id": s["id"], "name": s.get("name", str(s["id"])), "poly": s["poly"], "capacity": s.get("capacity", 1)}
            for s in snap.get("slots", []) if s["id"] < DYNAMIC_BASE and not s.get("parent")]


def record_pixels(fetch_status, fetch_frame, out_dir, seconds: float, every: float = 1.0, max_bytes: int | None = None,
                  poll: float = 0.5, camera: str = "", settings: dict | None = None, clock=time.monotonic, sleep=time.sleep,
                  progress=None, max_errors: int = 30) -> dict:
    """fetch_status() -> the status JSON; fetch_frame() -> the raw JPEG bytes or None. Saves at most one frame per `every` seconds of
    capture time, stops after `seconds` or when the pictures have taken max_bytes. -> summary {"frames", "bytes", "mismatched",
    "stopped" ("time" / "size"), "slot_map_changed": frames whose slot map differed from the first one, "dir"}."""
    end, rec, last_ts, last_saved = clock() + seconds, None, None, None
    mismatched, changed, base_map, stopped, errors, row = 0, 0, None, "time", 0, 0
    while clock() < end:
        try:
            snap = (fetch_status() or {}).get("snapshot") or {}
            row = 0
        except (OSError, ValueError) as e:  # urllib.error.URLError, TimeoutError, json.JSONDecodeError
            errors, row = errors + 1, row + 1
            if row >= max_errors:
                if rec is not None:
                    rec.close()
                raise RuntimeError(f"{row} requests in a row failed, last: {type(e).__name__}: {e}") from e
            sleep(poll)
            continue
        ts = snap.get("frame_ts")
        due = ts is not None and (last_saved is None or ts - last_saved >= every)
        if snap.get("shape") and ts is not None and ts != last_ts and due:
            last_ts = ts
            try:
                jpeg = fetch_frame()
                again = ((fetch_status() or {}).get("snapshot") or {}).get("frame_ts") if jpeg else None
            except (OSError, ValueError):
                errors += 1
                sleep(poll)
                continue
            if jpeg and again == ts:
                if rec is None:
                    base_map = _slots_of(snap)
                    rec = PixelRecorder(out_dir, {"camera": camera, "size": list(snap["shape"]), "slots": base_map,
                                                  "settings": settings or {}, "every": every, "poll": poll,
                                                  "started": datetime.now().isoformat(timespec="seconds")})
                elif _slots_of(snap) != base_map:
                    changed += 1
                rec.add(ts, jpeg, [{"box": d["box"], "conf": d["conf"], "label": d.get("label", "")}
                                   for d in snap.get("detections", []) if d.get("kind", "vehicle") == "vehicle"],
                        {str(s["id"]): s["state"] for s in snap.get("slots", []) if s["id"] < DYNAMIC_BASE})
                last_saved = ts
                if progress:
                    progress(rec.n, rec.bytes)
                if max_bytes is not None and rec.bytes >= max_bytes:
                    stopped = "size"
                    break
            else:
                mismatched += 1
        sleep(poll)
    if rec is None:
        raise RuntimeError("the worker published no frame with a picture while recording (is monitoring on?)")
    rec.close()
    return {"frames": rec.n, "bytes": rec.bytes, "mismatched": mismatched, "errors": errors, "stopped": stopped, "slot_map_changed": changed,
            "dir": str(rec.dir)}


# ---- reading -----------------------------------------------------------------------------------------------------------------------
class PixelRecording:
    def __init__(self, directory):
        self.dir = Path(directory)
        try:
            self.header = json.loads((self.dir / "header.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"{self.dir}: not a pixel recording (no header.json)") from None
        if self.header.get("format") != FORMAT:
            raise ValueError(f"{self.dir}: recording format {self.header.get('format')!r}, expected {FORMAT}")
        self.frames: list[dict] = []
        for line in (self.dir / "frames.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                self.frames.append(json.loads(line))
            except json.JSONDecodeError:  # the last line of a recording that was killed mid-write
                break
        if not self.frames:
            raise ValueError(f"{self.dir}: the recording has no frames")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        return self.frames[-1]["t"] - self.frames[0]["t"]

    @property
    def slots(self) -> list[dict]:
        return self.header["slots"]

    def image(self, i: int):
        """The saved picture as a BGR array."""
        import cv2  # (only the replay needs it; the recorder runs without)

        img = cv2.imread(str(self.dir / self.frames[i]["file"]))
        if img is None:
            raise ValueError(f"{self.dir / self.frames[i]['file']}: picture missing or unreadable")
        return img

    def live_states(self) -> dict[int, list[int | None]]:
        """The states the running worker showed, per slot per frame (0 / 1 / None) - the baseline the live stack produced."""
        return {s["id"]: [STATE_VALUE.get(f["states"].get(str(s["id"]))) for f in self.frames] for s in self.slots}


