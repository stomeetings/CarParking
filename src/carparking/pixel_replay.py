"""Pixel recordings and the offline replay of them (post-demo item 4, step 0/2): the evidence layer's veto thresholds (evidence.py
EvidenceCfg: motion_ratio, motion_floor, max_bhatta, min_ncc) are computed from image content, so they can only be tuned on frames,
not on the boxes-only recording of track_eval.py. This module is the tooling for that; it carries no tuned value of its own.

  record_pixels()   sample the running worker (GET /api/status + GET /api/frame.jpg?raw=true) and write a recording directory:
                      header.json     {"format": 1, "camera", "size", "slots", "settings", "started", ...}
                      frames.jsonl    one line per saved frame {"i", "t" (capture time), "file", "dets", "states"} (appended as it goes,
                                      so a recording that is interrupted is still readable)
                      frames/NNNNNN.jpg   the raw frame the worker published for that status
                    Read only towards the stack. A frame is kept only when the status read before and after the image fetch name the
                    same frame_ts (the worker publishes snapshot and image together), so box and picture belong together.
  PixelRecording    reads one back (tolerates a half-written last line).
  replay()          feeds the saved boxes and pictures through a FixedSlotTracker, evidence_filter 0 or 1, with the recording's own slot
                    map and the worker's tracker settings; -> ReplayResult (events with the time they were reported, the shown state of
                    every slot per frame, the number of exit vetoes).

Limits of an offline replay, all reported by the scripts: boxes are the worker's published axis-aligned boxes (a segmentation polygon is not
kept), vehicle-driven bays of an auto grid are not replayed, and the zoom re-look of the veto (ParkingMonitor._relook needs the detector)
is not available, so the 'relook' reason never fires; 'appearance' and 'no_motion' do."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

from carparking import evidence as evd
from carparking.fixed_slots import FixedSlotTracker
from carparking.pixel_recorder import SETTING_TO_TRACKER, PixelRecording


# ---- replay ------------------------------------------------------------------------------------------------------------------------
def tracker_kwargs_of(settings: dict) -> dict:
    return {SETTING_TO_TRACKER[k]: v for k, v in (settings or {}).items() if k in SETTING_TO_TRACKER and v is not None}


def cfg_with(base: evd.EvidenceCfg | None = None, **overrides) -> evd.EvidenceCfg:
    """An EvidenceCfg with some fields replaced. ValueError on a name that is not a field (so a typo in a sweep is not silently ignored)."""
    names = {f.name for f in fields(evd.EvidenceCfg)}
    bad = sorted(set(overrides) - names)
    if bad:
        raise ValueError(f"not EvidenceCfg fields: {', '.join(bad)}; known: {', '.join(sorted(names))}")
    return replace(base or evd.CFG, **overrides)


def parse_cfg_value(name: str, text: str):
    """'3.5' -> the value typed like the EvidenceCfg field `name` (int fields stay int). ValueError on an unknown field or a bad number."""
    types = {f.name: type(f.default) for f in fields(evd.EvidenceCfg)}
    if name not in types:
        raise ValueError(f"not an EvidenceCfg field: {name}; known: {', '.join(sorted(types))}")
    try:
        return types[name](text) if types[name] is not int else int(float(text))
    except ValueError:
        raise ValueError(f"{name}: {text!r} is not a number") from None


@dataclass
class ReplayResult:
    label: str
    evidence_filter: int
    slot_ids: list[int]
    names: dict[int, str]
    times: list[float] = field(default_factory=list)  # capture time of each frame
    shown: dict[int, list[int | None]] = field(default_factory=dict)  # slot -> shown state per frame
    events: list[dict] = field(default_factory=list)  # {"slot", "state", "time", "reported", "initial"}
    vetoes: dict[int, int] = field(default_factory=dict)  # slot -> exit probes the veto stopped
    rejected: dict[str, int] = field(default_factory=dict)  # reason -> detections the non-vehicle filter dropped (obstacle_filter)
    cfg: dict = field(default_factory=dict)


def build_tracker(rec: PixelRecording, evidence_filter: int, cfg: evd.EvidenceCfg | None = None, tracker_kwargs: dict | None = None,
                  rule: str | None = None, min_iou: float | None = None) -> FixedSlotTracker:
    settings = rec.header.get("settings") or {}
    kw = {**tracker_kwargs_of(settings), **(tracker_kwargs or {})}
    return FixedSlotTracker(rec.slots, rule or settings.get("match_rule") or "ground",
                            min_iou if min_iou is not None else (settings.get("match_iou") or 0.15),
                            evidence_filter=evidence_filter, evidence_cfg=cfg, **kw)


def _vehicles(dets: list[dict], w: int, h: int) -> list[dict]:
    out = []
    for d in dets:
        x1, y1, x2, y2 = (round(d["box"][0] * w), round(d["box"][1] * h), round(d["box"][2] * w), round(d["box"][3] * h))
        out.append({"poly": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], "box": (x1, y1, x2, y2), "label": d.get("label", ""),
                    "conf": d["conf"], "state": 1})
    return out


def replay(rec: PixelRecording, evidence_filter: int = 0, cfg: evd.EvidenceCfg | None = None, tracker_kwargs: dict | None = None,
           rule: str | None = None, min_iou: float | None = None, label: str | None = None, progress=None) -> ReplayResult:
    """One pass of the recording through a fresh FixedSlotTracker. Frames are decoded only when the evidence layer is on (it alone reads
    pixels); the frame shape is always the first picture's, so off and on judge the same geometry."""
    tr = build_tracker(rec, evidence_filter, cfg, tracker_kwargs, rule, min_iou)
    first = rec.image(0)
    h, w = first.shape[:2]
    ids = [s["id"] for s in rec.slots]
    res = ReplayResult(label or f"evidence_filter={evidence_filter}", int(bool(evidence_filter)), ids,
                       {s["id"]: s["name"] for s in rec.slots}, shown={k: [] for k in ids},
                       cfg={f.name: getattr(tr.evidence_cfg, f.name) for f in fields(evd.EvidenceCfg)} if evidence_filter else {})
    veto_seen = {k: 0 for k in ids}
    res.vetoes = {k: 0 for k in ids}
    for n, f in enumerate(rec.frames):
        if evidence_filter or tr.obstacle_filter:
            tr.frame = first if n == 0 else rec.image(n)
            tr.relook = None  # (needs the detector; see the module doc)
        for e in tr.update(_vehicles(f["dets"], w, h), (h, w, 3), now=f["t"]):
            res.events.append({"slot": e["slot"], "state": e["state"], "time": e["time"], "reported": f["t"],
                               "initial": bool(e["initial"])})
        res.times.append(f["t"])
        for k in ids:
            res.shown[k].append(tr.shown(k) if k in tr.slots else None)
            ev = tr.ev.get(k)
            if ev is not None:
                res.vetoes[k] += max(0, ev.vetoes - veto_seen[k])  # (consecutive count: rises by one per vetoed probe, resets on its own)
                veto_seen[k] = ev.vetoes
        for r in tr.rejected:
            res.rejected[r["reason"]] = res.rejected.get(r["reason"], 0) + 1
        tr.frame = None
        if progress:
            progress(n + 1, len(rec))
    return res
