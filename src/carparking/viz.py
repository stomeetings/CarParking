"""Overlay bays, IDs and states on a frame, with a HUD bar above the image."""
from __future__ import annotations

import cv2
import numpy as np

# BGR. None = no label/prediction available (shown, never guessed).
COLORS = {0: (60, 200, 60), 1: (40, 40, 230), None: (160, 160, 160)}
NAMES = {0: "FREE", 1: "BUSY", None: "UNKNOWN"}
HUD_H = 34  # at 1280 px wide; scaled with the frame (see _scale)
FREE_TINT = 0.25  # FREE bays are filled green at this opacity, so an empty bay stands out, not just its outline


def _scale(frame) -> float:
    """Line / text scale: 1 up to 1280 px wide, more above. The 2560 px main stream is shown at 1280 px, so 2 px
    lines and 0.4 text drawn at full resolution end up hairline and unreadable."""
    return max(frame.shape[1] / 1280, 1.0)


def hud_text(bays, states: dict[int, int | None], title: str = "") -> str:
    """'FREE: n | BUSY: m | TOTAL: t' (+ ' | UNKNOWN: u' when some slots have no state yet), counted from states."""
    free = sum(1 for b in bays if states.get(b) == 0)
    busy = sum(1 for b in bays if states.get(b) == 1)
    hud = f"FREE: {free} | BUSY: {busy} | TOTAL: {len(bays)}"
    if unknown := len(bays) - free - busy:
        hud += f" | UNKNOWN: {unknown}"
    return f"{title}   {hud}" if title else hud


def draw(frame, bays: dict[int, np.ndarray], states: dict[int, int | None], title: str = "",
         labels: bool = False):
    """Return a new image: frame with every bay outlined in its state colour (FREE green and lightly filled, BUSY
    red, UNKNOWN grey) and its ID, plus a HUD bar on top (HUD_H x scale px taller). labels=True: the ID tag also
    says the state ("3 FREE"), for live views with a handful of slots."""
    s = _scale(frame)
    lw, fs, ft = max(2, round(2 * s)), 0.45 * s, max(1, round(s))
    out = frame.copy()
    free = [np.round(p).astype(np.int32) for b, p in bays.items() if states.get(b) == 0]
    if free:
        tint = out.copy()
        cv2.fillPoly(tint, free, COLORS[0])
        out = cv2.addWeighted(tint, FREE_TINT, out, 1 - FREE_TINT, 0)
    for slot, poly in bays.items():
        state = states.get(slot)
        pts = np.round(poly).astype(np.int32)
        cv2.polylines(out, [pts], True, COLORS[state], lw)
        cx, cy = pts.mean(axis=0).astype(int)
        label = f"{slot} {NAMES[state]}" if labels else str(slot)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        pad = round(3 * s)
        cv2.rectangle(out, (cx - tw // 2 - pad, cy - th - pad), (cx + tw // 2 + pad, cy + pad), COLORS[state], -1)
        cv2.putText(out, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft, cv2.LINE_AA)

    hud_h = round(HUD_H * s)
    out = cv2.copyMakeBorder(out, hud_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(out, hud_text(bays, states, title), (round(10 * s), round(23 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * s, (255, 255, 255), ft, cv2.LINE_AA)
    return out


def draw_dets(frame, dets: list[dict], title: str = ""):
    """YoloDetector output -> draw(): each detection's 4 corners (rotated for OBB) become a numbered bay polygon."""
    bays = {i: np.array(d["poly"], np.float32) for i, d in enumerate(dets)}
    return draw(frame, bays, {i: d["state"] for i, d in enumerate(dets)}, title)
