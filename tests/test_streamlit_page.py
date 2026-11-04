"""The Streamlit page after its clean-up (scripts/demo_app.py): every input mode still renders, and the helpers that replaced code written out
several times (FREE/BUSY/TOTAL numbers, the webcam opening, the canvas background) do what the copies did. Renders the real page with
streamlit.testing.AppTest (no browser, no camera, no worker); the helpers are compiled out of the file with stubs for streamlit and cv2."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

PAGE = ROOT / "scripts" / "demo_app.py"


def env(tmp_path, monkeypatch):
    for k, v in {"CARPARK_HISTORY_DB": str(tmp_path / "h.db"), "CARPARK_FIXED_SLOTS": str(tmp_path / "cfg" / "f.json"),
                 "CARPARK_DATASET_RAW": str(tmp_path / "raw"), "CARPARK_RTSP_URL": "rtsp://user:secret@cam.local:554/Streaming/Channels/101",
                 "CARPARK_WORKER": "external"}.items():
        monkeypatch.setenv(k, v)


@pytest.mark.parametrize("mode", ["Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5"])
def test_every_input_mode_renders_without_an_error(tmp_path, monkeypatch, mode):
    env(tmp_path, monkeypatch)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    radio = next(r for r in at.sidebar.radio if any(str(o).startswith("Mode 1") for o in r.options))
    radio.set_value(next(o for o in radio.options if str(o).startswith(mode))).run()
    assert not at.exception, [e.value for e in at.exception]
    if mode == "Mode 3":
        assert [c.label for c in at.checkbox] == ["Start Live Webcam"] and not at.checkbox[0].value  # nothing opens a camera by itself
    if mode == "Mode 4":
        assert "Source" in [r.label for r in at.sidebar.radio]  # image upload / live webcam


# ---- the helpers, compiled out of the page -----------------------------------------------------------------------------------------------
def helpers(**stubs):
    """n_free, show_metrics, fps_line, open_camera and canvas_background from the file, in a namespace of stubs."""
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    want = {"n_free", "show_metrics", "fps_line", "open_camera", "canvas_background"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {n.name for n in body} == want
    ns = {"time": SimpleNamespace(perf_counter=lambda: 12.0), "np": np, "CANVAS_W": 960, **stubs}
    exec(compile(ast.Module(body, []), str(PAGE), "exec"), ns)
    return ns


class FakeSt:
    def __init__(self):
        self.session_state, self.errors, self.metrics = {}, [], []

    def error(self, text):
        self.errors.append(text)

    def columns(self, n):
        return [SimpleNamespace(metric=lambda label, value: self.metrics.append((label, value))) for _ in range(n)]


def test_the_counts_and_the_fps_caption_are_what_the_four_copies_were():
    st = FakeSt()
    ns = helpers(st=st)
    items = [{"state": 0}, {"state": 1}, {"state": 1}, {"state": 0}, {"state": 1}]
    assert ns["n_free"](items) == 2 and ns["n_free"]([]) == 0
    ns["show_metrics"](items)
    assert st.metrics == [("FREE", 2), ("BUSY", 3), ("TOTAL", 5)]
    assert ns["fps_line"](items, 24, 0.0) == "FREE 2 · BUSY 3 · TOTAL 5 · 2.0 fps shown"  # 24 frames in 12 s


def test_opening_a_webcam_names_the_problem_once_and_keeps_each_pages_hint():
    st = FakeSt()
    closed = SimpleNamespace(isOpened=lambda: False)
    ns = helpers(st=st, cv2=SimpleNamespace(VideoCapture=lambda cam: closed))
    assert ns["open_camera"](0) is None and ns["open_camera"](1, " (Zoom, Teams)") is None
    assert st.errors[0].startswith("Cannot open camera 0. Close other apps using it, check Windows Settings")
    assert st.errors[1].startswith("Cannot open camera 1. Close other apps using it (Zoom, Teams), check Windows Settings")
    ok = SimpleNamespace(isOpened=lambda: True)
    assert helpers(st=FakeSt(), cv2=SimpleNamespace(VideoCapture=lambda cam: ok))["open_camera"](0) is ok


def test_the_canvas_background_is_built_once_per_frame_and_parking_area():
    import cv2
    st, built = FakeSt(), []

    class Area:
        def mask(self, shape):
            built.append(shape)
            return np.full(shape[:2], 255, np.uint8)

    ns = helpers(st=st, cv2=cv2, lot_mask=lambda frame, m5: Area().mask(frame.shape))
    frame = np.zeros((120, 240, 3), np.uint8)
    m5 = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "roi": None}
    st.session_state["frame5_ts"] = 100.0
    bg1, w, h = ns["canvas_background"](frame, m5)
    assert (w, h) == (960, 480) and bg1[:2] == b"\xff\xd8" and len(built) == 1
    assert ns["canvas_background"](frame, m5)[0] is bg1 and len(built) == 1  # a rerun with nothing changed: the same bytes, no work
    assert ns["canvas_background"](frame, {**m5, "left": 0.1})[0] is not bg1 and len(built) == 2  # the parking area moved
    ns["canvas_background"](frame, {**m5, "left": 0.1})
    assert len(built) == 2
    st.session_state["frame5_ts"] = 101.0  # a new frame
    ns["canvas_background"](frame, {**m5, "left": 0.1})
    assert len(built) == 3
    ns["canvas_background"](frame, {**m5, "left": 0.1, "roi": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]})
    assert len(built) == 4  # a polygon was typed
