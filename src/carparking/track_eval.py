"""Offline tracking evaluation (item 104): would tracking IDs (ByteTrack-style) make a capacity pin's vehicle count steadier than the
count the worker has today? Nothing here touches the live system: it reads a RECORDING of the worker's own merged detections
(whole frame + slices + zoom + splits, at the worker's real, uneven frame rate) and replays it through trackers.

A recording is a JSON file (record_from_api / save_recording; synthesize() makes one with ground truth for the self-test):
    {"version": 1, "camera": str, "size": [w, h],
     "slots": [{"id", "name", "poly" (fractions), "capacity"}],
     "frames": [{"t": seconds, "dets": [{"box": [x1, y1, x2, y2] (fractions), "conf", "label"}],
                 "gt": [{"id", "box"}]  (optional: hand-labelled or synthetic ground truth, same box format)}]}

Trackers (all: update(boxes_px, confs) -> [(track id, box_px)] of the tracks seen in this frame):
  ByteTracker      ByteTrack as published (Kalman filter on x, y, aspect, height; two-stage association, the low-confidence
                   boxes continue an existing track; a track survives `buffer` frames unseen), written here in numpy only
  IouTracker       the simplest baseline: greedy IoU, a track survives `max_age` frames unseen
  UltralyticsByteTracker  the real ultralytics BYTETracker, when its `lap` and `scipy` are installed (not in the runtime image;
                   never installed from here)

Metrics (evaluate): against ground truth, if the recording has it - detection precision / recall, MOT counts (misses, false
positives, ID switches, MOTA), ID switches per vehicle-hour, how many gaps in the detections the tracker bridged with the same ID;
per slot - the vehicle count from the raw detections, from the worker's own steady rule (steady_counts), from the tracks seen in
each frame ("tracks") and from every track still held, also the ones unseen for a few frames ("held": what a count by track IDs
would use): error against the truth, share of frames exactly right, and how often the count changes per hour (flicker); and, with or without truth,
how many track IDs stood in the slot against how many vehicles. The decision rule is the one the plan set: tracking is worth a live
change only if it switches IDs less than once per vehicle-hour AND its slot counts are at least as steady and as right as today's."""
from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from carparking.fixed_slots import CAPACITY_FRAMES, footprint_poly


# --- geometry and assignment (no scipy) ---------------------------------------------------------------------------------------
def iou_matrix(a, b) -> np.ndarray:
    """IoU of every box of a with every box of b (x1, y1, x2, y2) -> (len(a), len(b))."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def hungarian(cost) -> list[tuple[int, int]]:
    """The assignment of rows to columns with the least total cost (every row of the smaller side gets one) -> [(row, column)]."""
    cost = np.asarray(cost, float)
    n, m = cost.shape
    if not n or not m:
        return []
    flip = n > m
    if flip:
        cost, n, m = cost.T, m, n
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, int), np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(m + 1, np.inf), np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
            if j0 == 0:
                break
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j]]
    return [(j, i) for i, j in pairs] if flip else pairs


def associate(cost, limit: float):
    """Hungarian assignment keeping only pairs with cost <= limit -> (matches [(row, col)], unmatched rows, unmatched cols)."""
    cost = np.asarray(cost, float)
    pairs = [(i, j) for i, j in hungarian(np.minimum(cost, limit + 1.0)) if cost[i, j] <= limit] if cost.size else []
    return (pairs, [i for i in range(cost.shape[0]) if i not in {a for a, _ in pairs}],
            [j for j in range(cost.shape[1]) if j not in {b for _, b in pairs}])


# --- trackers -----------------------------------------------------------------------------------------------------------------
def to_xyah(box) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h])


def from_xyah(z) -> np.ndarray:
    cx, cy, a, h = (float(v) for v in z[:4])
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class _Kalman:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height) and their velocities, one step = one frame (as ByteTrack)."""
    W_POS, W_VEL = 1 / 20, 1 / 160

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z):
        h, wp, wv = z[3], self.W_POS, self.W_VEL
        std = [2 * wp * h, 2 * wp * h, 1e-2, 2 * wp * h, 10 * wv * h, 10 * wv * h, 1e-5, 10 * wv * h]
        return np.r_[z, np.zeros(4)], np.diag(np.square(std))

    def predict(self, mean, cov):
        h, wp, wv = mean[3], self.W_POS, self.W_VEL
        q = np.diag(np.square([wp * h, wp * h, 1e-2, wp * h, wv * h, wv * h, 1e-5, wv * h]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(self, mean, cov, z):
        h, wp = mean[3], self.W_POS
        s = self.H @ cov @ self.H.T + np.diag(np.square([wp * h, wp * h, 1e-1, wp * h]))
        k = np.linalg.solve(s, self.H @ cov).T
        return mean + k @ (z - self.H @ mean), cov - k @ s @ k.T


@dataclass
class _Track:
    id: int
    mean: np.ndarray
    cov: np.ndarray
    conf: float
    last_frame: int
    confirmed: bool

    @property
    def box(self) -> np.ndarray:
        return from_xyah(self.mean)


class ByteTracker:
    """ByteTrack (Zhang et al., 2022) in numpy: boxes at or above `high` start and continue tracks; the boxes between `low` and `high`
    continue the tracks that were seen in the previous frame (a half-hidden car keeps its ID); a new track is confirmed by a second
    sighting (the first frame excepted); a track unseen for more than `buffer` frames is dropped. Association costs are 1 - IoU of the
    Kalman-predicted box, limits as the published defaults (0.8 / 0.5 / 0.7)."""
    name = "bytetrack"

    def __init__(self, high: float = 0.5, low: float = 0.1, new: float = 0.6, buffer: int = 30):
        self.high, self.low, self.new, self.buffer = high, low, new, buffer
        self.kf, self.tracks, self.frame, self.next_id = _Kalman(), [], 0, 1

    def update(self, boxes, confs) -> list[tuple[int, np.ndarray]]:
        self.frame += 1
        boxes, confs = np.asarray(boxes, float).reshape(-1, 4), np.asarray(confs, float).reshape(-1)
        is_high = confs >= self.high
        hi, lo = np.flatnonzero(is_high), np.flatnonzero(~is_high & (confs >= self.low))
        seen_last = lambda t: t.last_frame == self.frame - 1  # noqa: E731
        pool = [t for t in self.tracks if t.confirmed]
        for t in pool + [t for t in self.tracks if not t.confirmed]:
            if not seen_last(t):
                t.mean[7] = 0.0  # (a lost track stops changing size)
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
        hit = []

        def take(track, i):
            track.mean, track.cov = self.kf.update(track.mean, track.cov, to_xyah(boxes[i]))
            track.conf, track.last_frame, track.confirmed = float(confs[i]), self.frame, True
            hit.append(track)

        cost = 1.0 - iou_matrix([t.box for t in pool], boxes[hi]) if pool and len(hi) else np.ones((len(pool), len(hi)))
        matches, un_trk, un_hi = associate(cost, 0.8)
        for ti, di in matches:
            take(pool[ti], hi[di])
        rest = [pool[i] for i in un_trk if seen_last(pool[i])]  # (second stage: only the tracks seen in the previous frame)
        cost = 1.0 - iou_matrix([t.box for t in rest], boxes[lo]) if rest and len(lo) else np.ones((len(rest), len(lo)))
        for ti, di in associate(cost, 0.5)[0]:
            take(rest[ti], lo[di])
        left = [hi[i] for i in un_hi]
        young = [t for t in self.tracks if not t.confirmed]
        cost = 1.0 - iou_matrix([t.box for t in young], boxes[left]) if young and left else np.ones((len(young), len(left)))
        matches, un_young, un_left = associate(cost, 0.7)
        for ti, di in matches:
            take(young[ti], left[di])
        drop = {id(young[i]) for i in un_young}
        for di in un_left:
            i = left[di]
            if confs[i] >= self.new:
                mean, cov = self.kf.initiate(to_xyah(boxes[i]))
                self.tracks.append(_Track(self.next_id, mean, cov, float(confs[i]), self.frame, self.frame == 1))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if id(t) not in drop and self.frame - t.last_frame <= self.buffer]
        return [(t.id, t.box) for t in self.tracks if t.confirmed and t.last_frame == self.frame]

    def alive(self) -> list[tuple[int, np.ndarray]]:
        """Every confirmed track still held - the ones seen in this frame and the ones unseen for up to `buffer` frames (at their
        predicted box): what a count by track IDs would use."""
        return [(t.id, t.box) for t in self.tracks if t.confirmed]


class IouTracker:
    """The baseline: a box continues the track whose last box it overlaps most (IoU >= `iou`, greedy by best pair); an unmatched box
    starts a track; a track unseen for more than `max_age` frames is dropped. No motion model, no use of low confidences."""
    name = "iou"

    def __init__(self, min_conf: float = 0.3, iou: float = 0.3, max_age: int = 5):
        self.min_conf, self.iou, self.max_age = min_conf, iou, max_age
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, boxes, confs) -> list[tuple[int, np.ndarray]]:
        boxes, confs = np.asarray(boxes, float).reshape(-1, 4), np.asarray(confs, float).reshape(-1)
        keep = np.flatnonzero(confs >= self.min_conf)
        ids = list(self.tracks)
        sim = iou_matrix([self.tracks[i]["box"] for i in ids], boxes[keep])
        out, used = [], set()
        for flat in np.argsort(-sim, axis=None):
            a, b = divmod(int(flat), max(sim.shape[1], 1))
            if not sim.size or sim[a, b] < self.iou:
                break
            if ids[a] in used or keep[b] in {o[2] for o in out}:
                continue
            used.add(ids[a])
            self.tracks[ids[a]].update(box=boxes[keep[b]], age=0)
            out.append((ids[a], boxes[keep[b]], keep[b]))
        taken = {o[2] for o in out}
        for i in keep:
            if i not in taken:
                self.tracks[self.next_id] = {"box": boxes[i], "age": 0}
                out.append((self.next_id, boxes[i], i))
                self.next_id += 1
        for i in ids:
            if i not in used:
                self.tracks[i]["age"] += 1
                if self.tracks[i]["age"] > self.max_age:
                    del self.tracks[i]
        return [(i, b) for i, b, _ in out]

    def alive(self) -> list[tuple[int, np.ndarray]]:
        """Every track still held (seen in this frame or unseen for up to `max_age` frames, at its last box)."""
        return [(i, t["box"]) for i, t in self.tracks.items()]


class _Dets:
    """Detections as the ultralytics trackers read them: conf, xywh and cls arrays, and boolean / index selection."""

    def __init__(self, conf, xywh, cls):
        self.conf, self.xywh, self.cls = conf, xywh, cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Dets(self.conf[idx], self.xywh[idx], self.cls[idx])


class UltralyticsByteTracker:
    """The real ultralytics BYTETracker. It needs `lap` and `scipy`, which the runtime image does not have: availability() says so
    and nothing is ever pip-installed from here (ultralytics' own auto-install is switched off)."""
    name = "ultralytics-bytetrack"

    @staticmethod
    def _load():
        os.environ["YOLO_AUTOINSTALL"] = "False"
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace
        return BYTETracker, IterableSimpleNamespace

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        try:
            cls._load()
        except Exception as e:  # noqa: BLE001 - ImportError of lap / scipy, or a changed ultralytics
            return False, f"{type(e).__name__}: {e}"[:200]
        return True, ""

    def __init__(self):
        tracker, ns = self._load()
        args = ns(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6, track_buffer=30, match_thresh=0.8,
                  fuse_score=True)
        # older versions scale the buffer by frame_rate / 30 (30 keeps it at track_buffer frames); newer ones have no such argument
        self.inner = tracker(args, frame_rate=30) if "frame_rate" in inspect.signature(tracker.__init__).parameters else tracker(args)

    def update(self, boxes, confs) -> list[tuple[int, np.ndarray]]:
        boxes, confs = np.asarray(boxes, float).reshape(-1, 4), np.asarray(confs, float).reshape(-1)
        xywh = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]
        return [(int(r[4]), np.asarray(r[:4], float)) for r in self.inner.update(_Dets(confs, xywh, np.zeros(len(confs))))]

    def alive(self) -> list[tuple[int, np.ndarray]]:
        """Every activated track still held: tracked and lost (unseen for up to track_buffer frames)."""
        return [(int(t.track_id), np.asarray(t.xyxy, float)) for t in list(self.inner.tracked_stracks) + list(self.inner.lost_stracks)
                if t.is_activated]


TRACKERS = {"bytetrack": ByteTracker, "iou": IouTracker, "ultralytics": UltralyticsByteTracker}


# --- recordings ---------------------------------------------------------------------------------------------------------------
@dataclass
class Recording:
    camera: str
    size: tuple[int, int]
    slots: list[dict]
    frames: list[dict] = field(default_factory=list)

    @property
    def has_truth(self) -> bool:
        return any(f.get("gt") for f in self.frames)

    @property
    def duration(self) -> float:
        return (self.frames[-1]["t"] - self.frames[0]["t"]) if len(self.frames) > 1 else 0.0

    def to_dict(self) -> dict:
        return {"version": 1, "camera": self.camera, "size": list(self.size), "slots": self.slots, "frames": self.frames}


def save_recording(rec: Recording, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec.to_dict(), f, separators=(",", ":"))


def load_recording(path) -> Recording:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("version") != 1 or not d.get("frames"):
        raise ValueError(f"{path}: not a version 1 recording with frames")
    w, h = d["size"]
    return Recording(d.get("camera", ""), (int(w), int(h)), d.get("slots", []), d["frames"])


def record_from_api(fetch, seconds: float, poll: float = 0.5, clock=time.monotonic, sleep=time.sleep, camera: str = "") -> Recording:
    """Sample the worker's own published detections (GET /api/status: snapshot.detections, .slots, .shape, .frame_ts) for `seconds`,
    one frame per new frame_ts, at the worker's real rate. Read only: nothing is written to the system. fetch() -> the status JSON."""
    end, frames, slots, size, last = clock() + seconds, [], [], None, None
    while clock() < end:
        snap = (fetch() or {}).get("snapshot") or {}
        ts = snap.get("frame_ts")
        if snap.get("shape") and ts is not None and ts != last:
            last, size = ts, tuple(snap["shape"])
            slots = [{"id": s["id"], "name": s.get("name", str(s["id"])), "poly": s["poly"], "capacity": s.get("capacity", 1)}
                     for s in snap.get("slots", [])] or slots
            frames.append({"t": float(ts), "dets": [{"box": d["box"], "conf": d["conf"], "label": d.get("label", "")}
                                                    for d in snap.get("detections", []) if d.get("kind", "vehicle") == "vehicle"]})
        sleep(poll)
    if size is None:
        raise RuntimeError("the worker published no frame while recording (is monitoring on?)")
    return Recording(camera, size, slots, frames)


def synthesize(seed: int = 0, minutes: float = 10.0, fps: float = 1.0) -> Recording:
    """A known scene with ground truth, for the self-test and the tests: a pin P1 (capacity 3) with three cars arriving and one
    leaving, a one-car slot whose car is replaced, uneven frame times, jitter, missed boxes, a 5-frame dropout, weak boxes, twin
    boxes and stray false positives. The same seed gives the same recording."""
    rng = np.random.RandomState(seed)
    size = (1920, 1080)
    slots = [{"id": 9, "name": "P1", "poly": [[0.44, 0.10], [0.67, 0.10], [0.67, 0.27], [0.44, 0.27]], "capacity": 3},
             {"id": 2, "name": "B2", "poly": [[0.08, 0.38], [0.27, 0.38], [0.27, 0.62], [0.08, 0.62]], "capacity": 1}]
    total = minutes * 60.0
    cars = [(1, (0.47, 0.14, 0.52, 0.22), 0.0, total), (2, (0.53, 0.15, 0.58, 0.23), 0.1 * total, 0.7 * total),
            (3, (0.59, 0.16, 0.64, 0.24), 0.3 * total, total), (4, (0.11, 0.42, 0.24, 0.58), 0.0, 0.5 * total),
            (5, (0.11, 0.42, 0.24, 0.58), 0.55 * total, total)]
    burst = (0.4 * total, 0.4 * total + 5.0 / fps)  # car 2 is not detected for 5 frames
    frames, t = [], 0.0
    while t < total:
        dets, gt = [], []
        for cid, (x1, y1, x2, y2), t_in, t_out in cars:
            if not t_in <= t < t_out:
                continue
            gt.append({"id": cid, "box": [x1, y1, x2, y2]})
            if rng.rand() < 0.06 or (cid == 2 and burst[0] <= t < burst[1]):
                continue
            w, h = x2 - x1, y2 - y1
            j = rng.normal(0, 0.03, 4) * np.array([w, h, w, h])
            conf = float(np.clip(rng.normal(0.8, 0.08), 0.35, 0.99)) if rng.rand() > 0.12 else float(rng.uniform(0.25, 0.45))
            box = [x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3]]
            dets.append({"box": box, "conf": conf, "label": "car"})
            if rng.rand() < 0.05:  # a twin box of the same car (whole frame + zoom)
                dets.append({"box": [b + 0.003 for b in box], "conf": max(conf - 0.2, 0.2), "label": "car"})
        for _ in range(rng.poisson(0.15)):  # a stray false positive
            x, y = rng.uniform(0.05, 0.85), rng.uniform(0.05, 0.8)
            dets.append({"box": [x, y, x + 0.06, y + 0.09], "conf": float(rng.uniform(0.1, 0.5)), "label": "car"})
        frames.append({"t": round(t, 3), "dets": dets, "gt": gt})
        t += (1.0 / fps) * rng.uniform(0.85, 1.15)
    return Recording("synthetic", size, slots, frames)


# --- metrics ------------------------------------------------------------------------------------------------------------------
def steady_counts(series, window: int = CAPACITY_FRAMES) -> list[int]:
    """The worker's own steady count (FixedSlotTracker._count_loads) over a per-frame series: seen in 2 of the last `window` frames,
    kept until 4 show fewer; the first frame counts one at most."""
    out = []
    for i in range(len(series)):
        hist = list(series[max(0, i - window + 1):i + 1])
        out.append(sorted(hist, reverse=True)[1] if len(hist) > 1 else min(hist[0], 1))
    return out


def _in_slot(box_frac, poly) -> bool:
    """The middle of the vehicle's ground footprint (the bottom of its box, as the worker's capacity count) lies inside the slot."""
    x1, y1, x2, y2 = (float(v) for v in box_frac)
    fp = footprint_poly([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return cv2.pointPolygonTest(np.float32(poly).reshape(-1, 1, 2), (float(fp[:, 0].mean()), float(fp[:, 1].mean())), False) >= 0


def _series_stats(series, truth, hours: float, capacity: int) -> dict:
    series = [min(int(v), capacity) for v in series]
    changes = sum(a != b for a, b in zip(series, series[1:]))
    out = {"changes_per_hour": round(changes / hours, 1) if hours else None, "max": max(series, default=0)}
    if truth is not None:
        err = [abs(a - b) for a, b in zip(series, truth)]
        out |= {"mae": round(float(np.mean(err)), 3), "exact": round(float(np.mean([e == 0 for e in err])), 3)}
    return out


def evaluate(rec: Recording, tracker, conf_min: float = 0.3, match_iou: float = 0.5) -> dict:
    """Replay the recording through `tracker` -> the report (see the module doc). conf_min: the detector threshold of the worker
    (the raw counts use the boxes at or above it; the tracker sees every box, ByteTrack uses the weak ones)."""
    w, h = rec.size
    scale = np.array([w, h, w, h], float)
    truth_on = rec.has_truth
    ids_seen, births = set(), 0
    stat = {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "frag": 0, "det_tp": 0, "det_fp": 0, "det_fn": 0}
    last_id: dict[int, int] = {}  # ground truth object -> the track id it was last matched to
    life: dict[int, list[float]] = {}  # ground truth object -> [first t, last t]
    was_matched: dict[int, bool] = {}
    gap_before: dict[int, int | None] = {}
    gaps = {"bridged": 0, "switched": 0, "lost": 0}
    per_slot = {s["id"]: {"raw": [], "tracks": [], "held": [], "truth": [], "ids": set(), "gt_ids": set()} for s in rec.slots}
    for frame in rec.frames:
        dets = frame["dets"]
        boxes = np.array([d["box"] for d in dets], float).reshape(-1, 4) * scale
        confs = np.array([d["conf"] for d in dets], float)
        tracks = tracker.update(boxes, confs)
        births += len({i for i, _ in tracks} - ids_seen)
        ids_seen |= {i for i, _ in tracks}
        raw = [b for b, c in zip(boxes / scale, confs) if c >= conf_min]
        trk = [(i, b / scale) for i, b in tracks]
        held = [(i, b / scale) for i, b in (tracker.alive() if hasattr(tracker, "alive") else tracks)]  # (incl. tracks unseen now)
        gt = [(g["id"], np.array(g["box"], float)) for g in frame.get("gt", [])]
        for s in rec.slots:
            slot = per_slot[s["id"]]
            slot["raw"].append(sum(_in_slot(b, s["poly"]) for b in raw))
            slot["tracks"].append(sum(_in_slot(b, s["poly"]) for _, b in trk))
            slot["held"].append(sum(_in_slot(b, s["poly"]) for _, b in held))
            slot["ids"] |= {i for i, b in trk if _in_slot(b, s["poly"])}
            if truth_on:
                slot["truth"].append(sum(_in_slot(b, s["poly"]) for _, b in gt))
                slot["gt_ids"] |= {g for g, b in gt if _in_slot(b, s["poly"])}
        if not truth_on:
            continue
        # detection quality: the raw boxes against the ground truth
        sim = iou_matrix([b for _, b in gt], raw)
        det_pairs, un_gt, un_det = associate(1.0 - sim, 1.0 - match_iou) if sim.size else ([], list(range(len(gt))), list(range(len(raw))))
        stat["det_tp"] += len(det_pairs)
        stat["det_fn"] += len(un_gt)
        stat["det_fp"] += len(un_det)
        det_hit = {gt[i][0] for i, _ in det_pairs}
        # tracking: keep last frame's pairs when they still overlap, then the best assignment of the rest
        sim = iou_matrix([b for _, b in gt], [b for _, b in trk])
        pairs, used_g, used_t = [], set(), set()
        for gi, (gid, _) in enumerate(gt):
            for ti, (tid, _) in enumerate(trk):
                if last_id.get(gid) == tid and sim[gi, ti] >= match_iou:
                    pairs.append((gi, ti))
                    used_g.add(gi)
                    used_t.add(ti)
        rest_g, rest_t = [i for i in range(len(gt)) if i not in used_g], [j for j in range(len(trk)) if j not in used_t]
        if rest_g and rest_t:
            sub = 1.0 - sim[np.ix_(rest_g, rest_t)]
            pairs += [(rest_g[i], rest_t[j]) for i, j in associate(sub, 1.0 - match_iou)[0]]
        matched = {gt[gi][0]: trk[ti][0] for gi, ti in pairs}
        stat["tp"] += len(pairs)
        stat["fn"] += len(gt) - len(pairs)
        stat["fp"] += len(trk) - len(pairs)
        for gid, _ in gt:
            life.setdefault(gid, [frame["t"], frame["t"]])[1] = frame["t"]
            tid = matched.get(gid)
            if tid is not None:
                if gid in last_id and last_id[gid] != tid:
                    stat["idsw"] += 1
                if gid in was_matched and not was_matched[gid]:
                    stat["frag"] += 1
                last_id[gid] = tid
            was_matched[gid] = tid is not None
            if gid not in det_hit:  # a gap in the detections of this vehicle
                gap_before.setdefault(gid, last_id.get(gid))
            elif gid in gap_before:
                before = gap_before.pop(gid)
                if before is not None:
                    gaps["lost" if tid is None else "bridged" if tid == before else "switched"] += 1
    n = len(rec.frames)
    hours = rec.duration / 3600.0
    report = {"tracker": tracker.name, "frames": n, "duration_s": round(rec.duration, 1),
              "frame_rate": round((n - 1) / rec.duration, 3) if rec.duration else None, "ground_truth": truth_on,
              "tracks": {"births": births, "distinct_ids": len(ids_seen)}, "slots": {}}
    if truth_on:
        vehicle_hours = sum(b - a for a, b in life.values()) / 3600.0
        gt_total = stat["tp"] + stat["fn"]
        report["detection"] = {"precision": _ratio(stat["det_tp"], stat["det_tp"] + stat["det_fp"]),
                               "recall": _ratio(stat["det_tp"], stat["det_tp"] + stat["det_fn"]), "fp": stat["det_fp"], "fn": stat["det_fn"]}
        report["tracking"] = {"mota": round(1 - (stat["fn"] + stat["fp"] + stat["idsw"]) / gt_total, 3) if gt_total else None,
                              "id_switches": stat["idsw"], "fragmentations": stat["frag"], "false_positives": stat["fp"],
                              "misses": stat["fn"], "vehicle_hours": round(vehicle_hours, 3), "gt_vehicles": len(life),
                              "id_switches_per_vehicle_hour": round(stat["idsw"] / vehicle_hours, 2) if vehicle_hours else None,
                              "gaps": gaps}
    for s in rec.slots:
        d, cap = per_slot[s["id"]], int(s.get("capacity", 1))
        truth = d["truth"] if truth_on else None
        vehicles = len(d["gt_ids"]) if truth_on else None
        report["slots"][str(s["id"])] = {
            "name": s.get("name", ""), "capacity": cap, "raw": _series_stats(d["raw"], truth, hours, cap),
            "steady": _series_stats(steady_counts(d["raw"]), truth, hours, cap), "tracks": _series_stats(d["tracks"], truth, hours, cap),
            "held": _series_stats(d["held"], truth, hours, cap),
            "track_ids": {"distinct": len(d["ids"]), "max_concurrent": max(d["tracks"], default=0), "vehicles": vehicles,
                          "per_vehicle": round(len(d["ids"]) / max(vehicles if vehicles else max(d["tracks"], default=1), 1), 2)}}
    report["decision"] = decide(report)
    return report


def _ratio(a: int, b: int):
    return round(a / b, 3) if b else None


def decide(report: dict) -> dict:
    """The plan's rule: tracking is worth a live change only if IDs switch less than once per vehicle-hour AND the tracks' slot counts
    change no more often, and are no further from the truth, than today's steady count. Without ground truth it can only say so."""
    if not report.get("ground_truth"):
        return {"verdict": "undetermined: the recording has no ground truth (ID switches and count errors need it)",
                "id_switches_per_vehicle_hour": None}
    idsw = report["tracking"]["id_switches_per_vehicle_hour"]
    rows = [(s["steady"], s["held"]) for s in report["slots"].values()]  # (the candidate: a count of the tracks still held)
    flicker_ok = sum(t["changes_per_hour"] or 0 for _, t in rows) <= sum(s["changes_per_hour"] or 0 for s, _ in rows)
    error_ok = sum(t.get("mae", 0) for _, t in rows) <= sum(s.get("mae", 0) for s, _ in rows)
    ids_ok = idsw is not None and idsw < 1.0
    return {"id_switches_per_vehicle_hour": idsw, "ids_ok": ids_ok, "flicker_ok": flicker_ok, "error_ok": error_ok,
            "verdict": "tracking is worth a live change" if ids_ok and flicker_ok and error_ok
            else "keep the steady count (tracking does not beat it here)"}


def format_report(reports: list[dict]) -> str:
    """The reports as text, one block per tracker."""
    lines = []
    for r in reports:
        if "unavailable" in r:
            lines.append(f"== {r['tracker']}: not available here ({r['unavailable']})")
            continue
        lines.append(f"== {r['tracker']}: {r['frames']} frames, {r['duration_s']} s, {r['frame_rate']} frames/s, ground truth: "
                     f"{'yes' if r['ground_truth'] else 'no'}")
        lines.append(f"   tracks: {r['tracks']['births']} born, {r['tracks']['distinct_ids']} distinct ids")
        if r["ground_truth"]:
            d, t = r["detection"], r["tracking"]
            lines.append(f"   detection: precision {d['precision']}, recall {d['recall']} ({d['fp']} false, {d['fn']} missed)")
            lines.append(f"   tracking: MOTA {t['mota']}, {t['id_switches']} id switches = {t['id_switches_per_vehicle_hour']} per vehicle-hour "
                         f"({t['gt_vehicles']} vehicles, {t['vehicle_hours']} h), {t['fragmentations']} fragmentations, gaps {t['gaps']}")
        for sid, s in r["slots"].items():
            lines.append(f"   slot {sid} {s['name']} (capacity {s['capacity']}): ids {s['track_ids']['distinct']} for "
                         f"{s['track_ids']['vehicles'] if s['track_ids']['vehicles'] is not None else s['track_ids']['max_concurrent']} vehicle(s)")
            for src in ("raw", "steady", "tracks", "held"):
                v = s[src]
                lines.append(f"      {src:7s} changes/h {v['changes_per_hour']}" + (f", mae {v['mae']}, exact {v['exact']}" if "mae" in v else ""))
        lines.append(f"   verdict: {r['decision']['verdict']}")
    return "\n".join(lines)


def run_all(rec: Recording, names=("bytetrack", "iou"), conf_min: float = 0.3) -> list[dict]:
    """Evaluate the recording with each named tracker (an unavailable one is reported, not raised)."""
    out = []
    for name in names:
        cls = TRACKERS[name]
        if name == "ultralytics":
            ok, why = cls.availability()
            if not ok:
                out.append({"tracker": cls.name, "unavailable": why})
                continue
        out.append(evaluate(rec, cls(), conf_min))
    return out
