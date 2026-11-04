"""View analyser + router: decides per frame whether the axis-aligned or the OBB model should run.

Heuristic, not learned. Long Hough line segments (>= 40 px; the painted bay/lane lines), measured on a copy
resized to 1000 px:
  * angled: share of total segment length that is diagonal (15°..75° off the image axes) >= diag_share
  * dense:  many line segments per megapixel (lots of small bays seen from high up) >= dense_segments
ponytail: thresholds are uncalibrated guesses. The badge prints the measured values, so tune diag_share /
dense_segments on your own cameras. Perspective makes straight rows look diagonal too; replace with a small
learned classifier if routing mistakes show up in practice.
"""
from __future__ import annotations

import cv2
import numpy as np

NAMES = {"baseline": "YOLOv8 Axis-Aligned", "obb": "YOLO-OBB"}


def analyze_view(frame_bgr, diag_share: float = 0.55, dense_segments: float = 400) -> dict:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    s = 1000 / max(g.shape)  # fixed working size, so the thresholds don't depend on the input resolution
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150)
    # strict: only long, dominant lines (bay/lane markings), not shadows, car edges or ground texture
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
    # (N, 1, 4) or (N, 4) depending on the OpenCV version; reshape covers both
    seg = np.zeros((0, 4)) if lines is None else np.asarray(lines, dtype=float).reshape(-1, 4)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))  # 0 = horizontal, 90 = vertical
    diag = float(length[(angle > 15) & (angle < 75)].sum() / length.sum()) if length.sum() else 0.0
    density = len(seg) / (g.shape[0] * g.shape[1] / 1e6)
    return {"diag_share": round(diag, 3), "segments_per_mp": round(density, 1),
            "angled": diag >= diag_share, "dense": density >= dense_segments}


def route(frame_bgr, choice: str = "auto", available=("baseline", "obb"), **knobs) -> dict:
    """choice: "auto" | "baseline" | "obb". -> {"model", "sliced", "auto", "reason", "note", + analyze_view fields}.
    "sliced" (OBB tiles) follows the density measure in both auto and manual mode."""
    v = analyze_view(frame_bgr, **knobs)
    if v["angled"]:
        reason = f"angled layout ({v['diag_share']:.0%} of line length diagonal)"
    elif v["dense"]:
        reason = f"dense high-angle view ({v['segments_per_mp']:.0f} line segments/MP)"
    else:
        reason = f"axis-aligned, low-density layout ({v['diag_share']:.0%} diagonal)"
    pick = "obb" if v["angled"] or v["dense"] else "baseline"
    model, note = (pick if choice == "auto" else choice), ""
    if model not in available:
        other = next(iter(available))
        model, note = other, f"{NAMES[model]} weights missing, using {NAMES[other]}"
    return {**v, "model": model, "sliced": model == "obb" and v["dense"], "auto": choice == "auto",
            "reason": reason, "pick": pick, "note": note}


def badge(r: dict, mode: str) -> str:
    """Status line for the UI; mode = YoloDetector.describe(sliced)."""
    text = (f"🤖 Auto-routed to {NAMES[r['model']]} [{mode}] based on {r['reason']}" if r["auto"] else
            f"✋ Manual: {NAMES[r['model']]} [{mode}] · the analyser would pick {NAMES[r['pick']]} ({r['reason']})")
    text += f" · diagonal share {r['diag_share']:.0%}, density {r['segments_per_mp']:.0f} segments/MP"
    return f"{text} · {r['note']}" if r["note"] else text
