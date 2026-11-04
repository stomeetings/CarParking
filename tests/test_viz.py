"""Overlay (carparking/viz.py): every slot drawn in its state colour, FREE filled green, HUD counts from the states."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.history import SlotTracker  # noqa: E402
from carparking.viz import COLORS, HUD_H, draw, hud_text  # noqa: E402


def rect(x1, y1, x2, y2):
    return np.float32([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def test_free_busy_unknown_colours_fill_and_hud():
    frame = np.zeros((720, 1280, 3), np.uint8)
    bays = {1: rect(100, 100, 400, 300), 2: rect(500, 100, 800, 300), 3: rect(900, 100, 1200, 300)}
    out = draw(frame, bays, {1: 0, 2: 1, 3: None}, "RTSP", labels=True)
    assert out.shape[0] == 720 + HUD_H
    y = HUD_H  # image rows start below the HUD bar
    assert tuple(out[y + 100, 250]) == COLORS[0]  # FREE outline green
    assert tuple(out[y + 100, 650]) == COLORS[1]  # BUSY outline red
    assert tuple(out[y + 100, 1050]) == COLORS[None]  # UNKNOWN outline grey
    inside_free, inside_busy = out[y + 150, 150], out[y + 150, 550]
    assert inside_free[1] > 40 and inside_free[0] < inside_free[1]  # FREE bay filled green (25 % over black)
    assert inside_busy.sum() == 0  # BUSY bays are not filled: the car stays visible
    assert hud_text(bays, {1: 0, 2: 1, 3: None}, "RTSP") == "RTSP   FREE: 1 | BUSY: 1 | TOTAL: 3 | UNKNOWN: 1"


def test_lines_and_hud_scale_with_the_2560_px_main_stream():
    frame = np.zeros((1440, 2560, 3), np.uint8)
    out = draw(frame, {1: rect(200, 200, 800, 600)}, {1: 0})
    assert out.shape[0] == 1440 + 2 * HUD_H  # HUD twice as tall: readable after the browser's 1280 px downscale
    col = out[2 * HUD_H + 195:2 * HUD_H + 206, 500]  # the top edge of the bay, 4 px thick instead of 2
    assert sum(tuple(p) == COLORS[0] for p in col) >= 4


def test_a_car_leaving_turns_its_slot_green_in_the_overlay():
    shape = (720, 1280, 3)
    car = {"poly": [(100, 100), (400, 100), (400, 300), (100, 300)], "state": 1}
    other = {"poly": [(600, 100), (900, 100), (900, 300), (600, 300)], "state": 1}
    tr = SlotTracker(hold=2, learn_hold=2, exit_hold=3, absent_free=True)
    for t in range(6):
        tr.update([car, other], shape, now=t)
    for t in range(6, 12):  # the first car leaves
        tr.update([other], shape, now=t)
    bays, states = tr.view(shape)
    assert sorted(states.values()) == [0, 1]  # the learned slot stays, FREE: never dropped from the map
    out = draw(np.zeros(shape, np.uint8), bays, states, labels=True)
    assert out[HUD_H + 200, 150][1] > 40  # its bay is filled green
    assert hud_text(bays, states) == "FREE: 1 | BUSY: 1 | TOTAL: 2"
