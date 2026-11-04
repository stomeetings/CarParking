"""Streamlit tester for the YOLO FREE/BUSY detector.

Sidebar "Model": Auto (carparking/view.py measures the layout and routes each image), or a manual override:
  * baseline: models/parking_yolo.pt (axis-aligned, SAHI tiles)
  * OBB: models/parking_yolo_obb.pt (all lots) if trained, else models/parking_yolo_obb_holdout-PUCPR.pt;
    1280 px, plus its own tiles on dense high-angle views
Inputs: image upload, video upload, live webcam (OpenCV, like scripts/live_yolo.py). All use the same route ->
YoloDetector -> draw_dets pipeline; detection always runs on the raw OpenCV frame.
Mode 4 (carparking/dynamic_iou.py): slots from the frame's painted lines, BUSY by IoU with vehicle detections.
Mode 5: RTSP IP camera (carparking/rtsp.py, newest frame only, auto-reconnect) -> learned slot map with debounced
entry/exit events (carparking/history.py) -> parking_history.db. RTSP URL from $CARPARK_RTSP_URL, times in
$CARPARK_TZ (default Asia/Karachi; the container clock is UTC).

Usage:  streamlit run scripts/demo_app.py
Runs locally; uploaded files and camera frames never leave this machine.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _drop_stale_modules(pkg: str = "carparking") -> bool:
    """Streamlit reruns this script on edit but keeps src/ modules imported; if any of them was edited since import,
    forget the whole package (else e.g. ImportError: cannot import name 'HybridDetector' until the container
    restarts). All of it, not just the edited files: an unchanged module keeps the classes it imported from the old
    one (monitor.py's SlotTracker after history.py changed -> "unexpected keyword argument 'hidden_max'"). True if
    anything was dropped."""
    mods = {name: mod for name, mod in sys.modules.items() if name == pkg or name.startswith(pkg + ".")}
    stale = any(getattr(mod, "_src_mtime", None) != os.path.getmtime(path)
                for mod in mods.values() if (path := getattr(mod, "__file__", None)) and os.path.exists(path))
    if stale:
        for name in mods:
            del sys.modules[name]
    return stale


if _drop_stale_modules():
    st.cache_resource.clear()  # free detectors built from the old classes (get_detector also keys on the version)

from carparking.dataset import save_frame, slug  # noqa: E402
from carparking.dynamic_iou import anchor_slots, bay_slots, find_slots, occupancy  # noqa: E402
from carparking.history import HistoryDB  # noqa: E402
from carparking.live import (STALE, LiveStore, code_version, decode_jpeg, heartbeat_age,  # noqa: E402
                             stream_url)
from carparking.fabric_shapes import fabric_to_polys, object_poly, polys_to_fabric  # noqa: E402
from carparking.fixed_slots import (MATCH_RULES, FixedSlotStore, draft_slots, format_points,  # noqa: E402
                                    relabel, validate_slots)

try:  # drawing on the camera image in the fixed slot editor; without it the editor offers typed coordinates only
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None
from carparking.monitor import ParkingArea, inside, parse_roi  # noqa: E402
from carparking.rtsp import LatestFrameReader, env_setting, mask_url  # noqa: E402
from carparking.view import badge, route  # noqa: E402
from carparking.viz import draw_dets  # noqa: E402
from carparking.yolo_engine import HybridDetector, YoloDetector  # noqa: E402

for _mod in [m for n, m in sys.modules.items() if n == "carparking" or n.startswith("carparking.")]:
    if getattr(_mod, "__file__", None):
        _mod.__dict__.setdefault("_src_mtime", os.path.getmtime(_mod.__file__))

OBB_FAIR = ROOT / "models" / "parking_yolo_obb.pt"
WEIGHTS = {"baseline": ROOT / "models" / "parking_yolo.pt",
           "obb": OBB_FAIR if OBB_FAIR.exists() else ROOT / "models" / "parking_yolo_obb_holdout-PUCPR.pt"}
CHOICES = {"🤖 Auto (layout analyser)": "auto",
           "YOLOv8 Axis-Aligned (Baseline)": "baseline",
           "YOLO-OBB (Oriented / Advanced)": "obb"}
STATE = {0: "FREE", 1: "BUSY"}
# Mode 4/5 vehicle sources: stock COCO models found in the repo root (drop yolov8s.pt there for the larger one), then
# per-camera fine-tunes models/cam_<camera>.pt (scripts/auto_label.py + yolo train), first so they are the default
CAM_PREFIX, COCO_PREFIX = "📷 Camera model · ", "🚗 COCO vehicles · "
COCO = {**{f"{CAM_PREFIX}{p.name}": p for p in sorted((ROOT / "models").glob("cam_*.pt"))},
        **{f"{COCO_PREFIX}{n}": ROOT / n for n in ("yolov8s.pt", "yolov8n.pt") if (ROOT / n).exists()}}
BAY_SRC = "🅿️ Bay model · PKLot (Model selector above)"
HYBRID: dict[str, tuple[Path, Path]] = {}  # Mode 5 dual-detector option -> (camera model, COCO model); per run
SLOT_ANCHOR = "Auto-Anchor Grid (Dynamic Extrapolation)"
SLOT_BAY, SLOT_HOUGH = "Trained Bay Model (Robust)", "Hough Line Detection (Dynamic)"
# Mode 5 camera: $CARPARK_RTSP_URL, else the project's .env (never in the source or the image); empty = Custom URL only
RTSP_URL = env_setting("CARPARK_RTSP_URL", ROOT / ".env")
# env overrides: tests and second instances must never share the live app's database (it holds the file open)
HISTORY_DB = Path(os.environ.get("CARPARK_HISTORY_DB", ROOT / "parking_history.db"))
# camera frames for fine-tuning (scripts/collect_dataset.py, scripts/auto_label.py)
DATASET_RAW = Path(os.environ.get("CARPARK_DATASET_RAW", ROOT / "dataset_raw"))
# Mode 5 fixed slot maps, per camera name (carparking/fixed_slots.py)
FIXED_SLOTS = Path(os.environ.get("CARPARK_FIXED_SLOTS", ROOT / "config" / "fixed_slots.json"))
SLOT_MODES = ["Dynamic (Auto)", "Fixed (Pre-defined)"]
CANVAS_W = 960  # slot drawing canvas width in px (height follows the frame's aspect ratio); slots are saved as fractions
# Mode 5 monitoring runs in scripts/worker.py (carparking/live.py); this page only views it and writes its settings.
# "embedded" (default, plain `streamlit run`): the page starts the worker as its own process when none is beating;
# "external" (docker compose): a separate service runs it
WORKER_MODE = os.environ.get("CARPARK_WORKER", "embedded")
try:
    TZ = ZoneInfo(os.environ.get("CARPARK_TZ", "Asia/Karachi"))
except Exception:  # unknown name or no tzdata: fall back to the process clock
    TZ = None


def engine_version() -> float:
    """Cache key part: detectors cached before yolo_engine.py was edited are old-class objects (missing methods,
    e.g. AttributeError: 'YoloDetector' object has no attribute 'vehicle_names'), so an edit builds fresh ones."""
    return os.path.getmtime(sys.modules[YoloDetector.__module__].__file__)


@st.cache_resource  # one cached detector per weights file (and mode), so switching back and forth doesn't reload
def _cached_detector(weights: Path, whole_imgsz: int | None, version: float):
    return YoloDetector(weights, whole_imgsz=whole_imgsz)


@st.cache_resource
def _cached_hybrid(cam: Path, coco: Path, whole_imgsz: int | None, version: float):
    return HybridDetector([_cached_detector(cam, whole_imgsz, version), _cached_detector(coco, whole_imgsz, version)],
                          ["cam", "coco"])


def n_free(items) -> int:
    """How many detections / slots are FREE (state 0); the rest are BUSY."""
    return sum(d["state"] == 0 for d in items)


def show_metrics(items):
    """The FREE / BUSY / TOTAL tiles of an image result."""
    free = n_free(items)
    c1, c2, c3 = st.columns(3)
    c1.metric("FREE", free)
    c2.metric("BUSY", len(items) - free)
    c3.metric("TOTAL", len(items))


def fps_line(items, shown: int, t0: float) -> str:
    """The live loops' caption: the counts of the last result and the frames shown per second."""
    free = n_free(items)
    return f"FREE {free} · BUSY {len(items) - free} · TOTAL {len(items)} · {shown / (time.perf_counter() - t0):.1f} fps shown"


def open_camera(cam: int, hint: str = ""):
    """A local webcam opened by this Python process on this machine (not by the browser) -> the capture, or None after showing why not."""
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        st.error(f"Cannot open camera {cam}. Close other apps using it{hint}, check Windows Settings → Privacy → Camera "
                 "(allow desktop apps), or try index 1.")
        return None
    return cap


def get_detector(weights: Path, whole_imgsz: int | None = None):
    return _cached_detector(weights, whole_imgsz, engine_version())


def get_hybrid(cam: Path, coco: Path, whole_imgsz: int | None = None):
    """Camera fine-tune + stock COCO on every frame, merged; the camera model's box wins on a duplicate."""
    return _cached_hybrid(cam, coco, whole_imgsz, engine_version())


def detect(frame, r, conf):
    """The one pipeline for every input: routed model -> detections -> RGB overlay."""
    det = get_detector(WEIGHTS[r["model"]])
    det.conf = conf
    dets = det(frame, sliced=r["sliced"])
    return cv2.cvtColor(draw_dets(frame, dets), cv2.COLOR_BGR2RGB), dets, det.describe(r["sliced"])


def show(frame, r, conf):
    img, dets, mode = detect(frame, r, conf)
    (st.success if r["auto"] else st.info)(badge(r, mode))
    show_metrics(dets)
    st.image(img, width="stretch")
    st.dataframe([{"id": i, "state": STATE[d["state"]], "label": d["label"], "conf": round(d["conf"], 3),
                   "corners": str(d["poly"])} for i, d in enumerate(dets)], hide_index=True)


def live(cam: int, every: int, choice: str, available, conf: float):
    """Same loop as scripts/live_yolo.py, rendered into Streamlit placeholders. Unticking the checkbox reruns the
    script, which stops this loop; the finally block then releases the camera."""
    cap = open_camera(cam, " (Zoom, Teams, live_yolo.py, another tab of this app)")
    if cap is None:
        return
    status, view, stats = st.empty(), st.empty(), st.empty()
    r, det, dets, i, t0 = None, None, [], 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.warning("The camera stopped delivering frames (unplugged, or taken by another app).")
                break
            if r is None:  # fixed camera: route once, on the first frame
                r = route(frame, choice, available)
                det = get_detector(WEIGHTS[r["model"]])
                (status.success if r["auto"] else status.info)(badge(r, det.describe(r["sliced"])))
            if i % every == 0:
                det.conf = conf
                dets = det(frame, sliced=r["sliced"])  # in between, the last result is drawn on the new frame
            # PNG: lossless display; inference is the bottleneck, not the transfer
            view.image(cv2.cvtColor(draw_dets(frame, dets), cv2.COLOR_BGR2RGB), output_format="PNG",
                       width="stretch")
            i += 1
            stats.caption(fps_line(dets, i, t0))
    finally:
        cap.release()


def bay_detector(frame, choice, available):
    """The routed bay model (Model selector) -> (detector, sliced, status text, auto?)."""
    r = route(frame, choice, available)
    det = get_detector(WEIGHTS[r["model"]])
    return det, r["sliced"], badge(r, det.describe(r["sliced"])), r["auto"]


def vehicle_detector(frame, vsrc, choice, available, whole_imgsz: int | None = None):
    """Mode 4/5 vehicle source -> (detector, sliced, status text, auto?). COCO = real car boxes; else the routed bay
    model, whose BUSY boxes stand in for cars. whole_imgsz: one whole-frame pass instead of SAHI tiles (Mode 5)."""
    if vsrc in HYBRID:
        det = get_hybrid(*HYBRID[vsrc], whole_imgsz)
        return det, False, f"🔀 Vehicles + objects from both models, merged: {det.describe()}", True
    if vsrc in COCO:
        det = get_detector(COCO[vsrc], whole_imgsz)
        kinds = "/".join(det.names[i] for i in det.classes)
        return det, False, f"🚗 Vehicles: {COCO[vsrc].name}, classes {det.classes} ({kinds}) [{det.describe()}]", True
    return bay_detector(frame, choice, available)


def lot_mask(frame, m4):
    """The parking area for this frame (uint8, 255 = parking): the margin rectangle, cut by the ROI polygon if set."""
    return ParkingArea.from_settings(m4).mask(frame.shape)


CUT_MAX = 60  # % of the frame a hard-wall cut may remove


def cut_input(col, label: str, key: str, default: int, help: str | None = None, save=None) -> float:
    """One hard-wall cut as % of the frame width/height: a number input (type a value; its own −/+ steps 1 %) with
    −5 % / +5 % buttons under it. The buttons change the value in a callback,
    before the rerun, so the new cut is applied (and the stream restarted with it) in that same rerun; every other
    widget keeps its value. The value also lives in a plain session key: Streamlit drops a widget's state on a run
    that doesn't show it (another mode), and the widget is re-seeded from there. save(value): also persist every
    change (Mode 5: per camera in the database, so the hard wall survives page reloads and restarts).
    -> share of the frame."""
    ss, store = st.session_state, f"{key}_value"
    ss.setdefault(store, default)
    if key not in ss:
        ss[key] = ss[store]

    def keep(v: int):
        ss[store] = ss[key] = v
        if save is not None:
            save(v)

    def typed():
        keep(ss[key])

    def nudge(d: int):
        keep(min(max(ss[key] + d, 0), CUT_MAX))

    col.number_input(f"{label} (%)", 0, CUT_MAX, step=1, key=key, help=help, on_change=typed)
    # ±1 % is the number input's own −/+ (shown at full sidebar width; 4 buttons in a row get truncated there)
    for c, d in zip(col.columns(2), (-5, 5)):
        c.button(f"{d:+d} %".replace("-", "−"), key=f"{key}_{'minus' if d < 0 else 'plus'}5", on_click=nudge,
                 args=(d,), width="stretch", help=f"{label} {d:+d} %")
    return ss[key] / 100


def slot_finder(frame, ssrc, choice, available, conf, m4):
    """Mode 4 slot source -> ((frame, vehicle dets) -> slot polygons, status text or None).
    m4: sidebar settings ("top", "bottom", "left", "right", "roi", "extend", "main", "hull")."""
    if ssrc == SLOT_ANCHOR:
        return (lambda f, dets: anchor_slots(dets, f.shape, extend=m4["extend"], frame=f, mask=lot_mask(f, m4),
                                             main_lot=m4["main"], stay_in_hull=m4["hull"])), None
    if ssrc == SLOT_BAY:
        det, sliced, text, _ = bay_detector(frame, choice, available)
        det.conf = conf
        return (lambda f, dets: bay_slots(f, det, sliced)), f"🅿️ Slots from the trained bay model: {text}"
    return (lambda f, dets: find_slots(f, ignore_top=m4["top"])), None


def iou_slots(frame, det, sliced, conf, thr, cover, slots_of, m4):
    """Mode 4: slots (from slots_of), BUSY by coverage/IoU with vehicle detections. The parking mask (lot_mask) is a
    hard wall for every slot source: vehicles outside it are dropped before the slot source sees them, and slots
    whose centre is outside it are dropped.
    -> (slot dicts, overlay BGR: outside the mask dimmed, mask edge in cyan, vehicles in yellow, then the slots)."""
    det.conf = conf
    mask = lot_mask(frame, m4)
    h, w = mask.shape
    inside = lambda x, y: 0 <= x < w and 0 <= y < h and mask[int(y), int(x)] > 0  # noqa: E731
    dets = [d for d in det(frame, sliced=sliced) if inside((d["box"][0] + d["box"][2]) / 2,
                                                           (d["box"][1] + d["box"][3]) / 2)]
    polys = [p for p in slots_of(frame, dets) if inside(*np.asarray(p, float).reshape(-1, 2).mean(0))]
    slots = occupancy(polys, dets, thr, cover)
    img = frame.copy()
    img[mask == 0] = (img[mask == 0] * 0.35).astype(np.uint8)  # not-parking area dimmed: easy to see the wall
    edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, edges, -1, (255, 255, 0), 2)
    cars = [np.array(d["poly"], np.int32) for d in dets if d["state"] == 1]
    if cars:  # cv2.polylines rejects an empty list
        cv2.polylines(img, cars, True, (0, 220, 255), 1)
    return slots, draw_dets(img, slots, "DYNAMIC IOU")


def live_iou(cam: int, every: int, vsrc: str, ssrc: str, choice: str, available, conf: float, thr: float,
             cover: float, m4: dict):
    """Mode 4 webcam loop; same structure as live(), but slots are rebuilt from the lines on every processed frame
    (a hand-held printout moves)."""
    cap = open_camera(cam)
    if cap is None:
        return
    status, slot_status, view, stats = st.empty(), st.empty(), st.empty(), st.empty()
    det, slots, i, t0 = None, [], 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.warning("The camera stopped delivering frames (unplugged, or taken by another app).")
                break
            if det is None:  # pick the vehicle model once, on the first frame
                det, sliced, text, auto = vehicle_detector(frame, vsrc, choice, available)
                (status.success if auto else status.info)(text)
                slots_of, stext = slot_finder(frame, ssrc, choice, available, conf, m4)
                if stext:
                    slot_status.info(stext)
            if i % every == 0:
                slots, img = iou_slots(frame, det, sliced, conf, thr, cover, slots_of, m4)
            else:  # in between: the last slots on the new frame
                img = draw_dets(frame, slots, "DYNAMIC IOU")
            view.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), output_format="PNG", width="stretch")
            i += 1
            stats.caption(fps_line(slots, i, t0))
    finally:
        cap.release()


def grab_frame(url: str, timeout: float = 15.0):
    """One frame from the stream -> (frame, capture time), or (None, None) if none came within timeout."""
    reader = LatestFrameReader(url)
    try:
        end = time.time() + timeout
        while time.time() < end:
            frame, _, t = reader.latest()
            if frame is not None:
                return frame, t
            time.sleep(0.1)
        return None, None
    finally:
        reader.stop()


def snapshot(url: str, camera: str, timeout: float = 15.0):
    """One raw frame from the stream to dataset_raw/ when Mode 5 is not running -> path, or None if no frame came."""
    frame, t = grab_frame(url, timeout)
    return None if frame is None else save_frame(frame, DATASET_RAW, camera, t, TZ)


def canvas_background(frame, m5: dict) -> tuple[bytes, int, int]:
    """The camera frame as the canvas background (parking area dimmed outside) -> (JPEG bytes, width, height). Every widget click
    reruns the page, so the last result is kept in the session and reused while the frame (its capture time, set whenever it is replaced)
    and the parking area are the same."""
    h, w = frame.shape[:2]
    cw, ch = CANVAS_W, round(CANVAS_W * h / w)
    area = (st.session_state.get("frame5_ts"), frame.shape, *(m5.get(k) for k in ("top", "bottom", "left", "right")), str(m5.get("roi")))
    last = st.session_state.get("_canvas_bg")
    if last is not None and last[0] == area:
        return last[1], cw, ch
    img = frame.copy()
    mask = lot_mask(frame, m5)
    img[mask == 0] = (img[mask == 0] * 0.5).astype(np.uint8)  # outside the parking area: vehicles are ignored there
    bg = cv2.imencode(".jpg", cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA),
                      [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    st.session_state["_canvas_bg"] = (area, bg)
    return bg, cw, ch


def canvas_editor(frame, m5: dict, key: str) -> list[dict]:
    """Draw mode: new slots on the camera frame (polygon by clicking corners, rectangle by dragging); the toolbar's
    edit toggle moves / reshapes / deletes any of them. -> slots in canvas order (ids and names kept by relabel)."""
    ss = st.session_state
    shape = st.radio("Shape", ["⬠ Polygon (click corners)", "▭ Rectangle (drag)"], horizontal=True,
                     key=f"{key}_shape",
                     help="Polygon: click each corner of the bay (e.g. 4 corners of an angled bay), then click the "
                          "first corner again to close it. Rectangle: click and drag. To change one slot, use its "
                          "✏️ Edit button below.")
    bg, cw, ch = canvas_background(frame, m5)
    init = ss[f"{key}_init"]
    res = st_canvas(background_image=bg, drawing_mode="polygon" if shape.startswith("⬠") else "rect",
                    initial_drawing=polys_to_fabric([s["poly"] for s in init], cw, ch), width=cw, height=ch,
                    stroke_width=3, stroke_color="#00ff00", fill_color="rgba(0,255,0,0.15)",
                    key=f"{key}_canvas{ss[f'{key}_ver']}")
    if res is None or res.json_data is None:  # first render: the canvas shows initial_drawing, nothing sent back yet
        polys, skipped = [s["poly"] for s in init], 0
    else:
        polys, skipped = fabric_to_polys(res.json_data, cw, ch)
    if skipped:
        st.caption(f"{skipped} drawn object(s) ignored: not a polygon or rectangle, or too small (a click).")
    slots = relabel(polys, ss[f"{key}_labels"])
    ss[f"{key}_labels"] = slots  # remembered by shape for the next rerun
    return slots


def slot_editor(frame, m5: dict, key: str, others: list[dict]):
    """Edit mode for one slot (ss[key_edit]): only it is editable on the canvas (orange; the others grey, locked),
    corner coordinates can be fine-tuned in a table, the name changed. Updates ss[key_edit] in place."""
    ss = st.session_state
    slot, ver = ss[f"{key}_edit"], ss[f"{key}_edit_ver"]
    st.caption("Turn on the ✏️ edit toggle in the canvas toolbar, then drag the orange slot to move it, its corners to "
               "reshape it, or its handles to resize it. Or type exact corner coordinates in the table (shares 0..1 of "
               "the frame width / height).")
    bg, cw, ch = canvas_background(frame, m5)
    drawing = polys_to_fabric([slot["poly"]], cw, ch, stroke="#ff9900", fill="rgba(255,153,0,0.25)")
    drawing["objects"] += polys_to_fabric([s["poly"] for s in others], cw, ch, stroke="#bbbbbb",
                                          fill="rgba(200,200,200,0.12)", locked=True)["objects"]
    res = st_canvas(background_image=bg, drawing_mode="rect", initial_drawing=drawing, width=cw, height=ch,
                    stroke_width=3, stroke_color="#ff9900", fill_color="rgba(255,153,0,0.25)",
                    key=f"{key}_ecanvas{ver}")
    objects = (res.json_data or {}).get("objects", []) if res is not None else []
    moved = object_poly(objects[0], cw, ch) if objects else None  # the edited slot is the first object
    if moved and len(moved) >= 3 and moved != slot["poly"]:
        slot["poly"] = moved
        ss[f"{key}_corner_ver"] += 1  # the corner table shows the new points
    c1, c2 = st.columns([1, 3])
    c1.markdown(f"**Slot {slot['id']}**")
    slot["name"] = c2.text_input("Name", slot["name"], key=f"{key}_ename{ver}", label_visibility="collapsed",
                                 placeholder="Name, e.g. A1").strip() or str(slot["id"])
    rows = pd.DataFrame(slot["poly"], columns=["x", "y"])
    edited = st.data_editor(rows, num_rows="dynamic", width="stretch",
                            key=f"{key}_corners{ver}_{ss[f'{key}_corner_ver']}",
                            column_config={c: st.column_config.NumberColumn(c, min_value=0.0, max_value=1.0,
                                                                            step=0.001, format="%.3f", required=True)
                                           for c in ("x", "y")})
    typed = [[round(float(x), 4), round(float(y), 4)] for x, y in edited.dropna().itertuples(index=False)]
    if len(typed) >= 3 and typed != slot["poly"]:  # typed corners: redraw the canvas with them
        slot["poly"] = typed
        ss[f"{key}_edit_ver"] += 1
        st.rerun()


def fixed_slot_editor(url: str, camera: str, det, conf: float, m5: dict, running: bool,
                      db: HistoryDB) -> list[dict]:
    """Fixed mode: draw slots with the mouse on the latest frame (st_canvas); every slot has its own ✏️ Edit (move /
    reshape on the canvas or fine-tune corners, rename) and 🗑️ Delete, which write config/fixed_slots.json at once
    (a deleted slot's open parking session is closed in the history); 💾 Save Slots stores all drawn slots.
    -> the saved slots (what the monitor uses)."""
    store = FixedSlotStore(FIXED_SLOTS)
    try:
        saved = store.load(camera)
    except ValueError as e:  # hand-edited file with a bad slot
        st.error(f"{FIXED_SLOTS.name}: {e}")
        saved = []
    ss, key = st.session_state, f"fx_{slug(camera)}"

    def restart(slots):  # the canvas starts over from these slots (new widget key, so its old edits are dropped)
        ss[f"{key}_init"] = ss[f"{key}_labels"] = ss[f"{key}_work"] = slots
        ss[f"{key}_ver"] = ss.get(f"{key}_ver", 0) + 1

    def flash(text):  # shown at the top of the editor for a few seconds: the remounted canvas reruns the page at once
        ss[f"{key}_flash"] = (text, time.time() + 8)

    def start_edit(sid):
        ss[f"{key}_edit"] = dict(next(s for s in ss[f"{key}_work"] if s["id"] == sid))
        ss[f"{key}_edit_ver"] = ss.get(f"{key}_edit_ver", 0) + 1
        ss[f"{key}_corner_ver"] = 0

    def stop_edit():
        ss[f"{key}_edit"] = None
        restart(ss[f"{key}_work"])

    def delete(sid):
        if any(s["id"] == sid for s in store.load(camera)):
            store.delete_slot(camera, sid, TZ)
            closed = db.close_slot(sid, note="slot deleted in the editor")
            flash(f"🗑️ Slot {sid} deleted from {FIXED_SLOTS.name}" + (" (its open parking session was closed)"
                                                                       if closed else ""))
        else:
            flash(f"🗑️ Unsaved slot {sid} removed")
        ss[f"{key}_edit"] = None
        restart([s for s in ss[f"{key}_work"] if s["id"] != sid])

    def save_edit():
        slot = ss[f"{key}_edit"]
        try:
            store.upsert_slot(camera, slot, TZ)
        except ValueError as e:
            flash(f"❌ Slot {slot['id']} not saved: {e}")
            return
        work = [s for s in ss[f"{key}_work"] if s["id"] != slot["id"]] + [validate_slots([slot])[0]]
        ss[f"{key}_work"] = sorted(work, key=lambda s: s["id"])
        flash(f"💾 Slot {slot['id']} ({slot['name']}) saved to {FIXED_SLOTS.name}")
        stop_edit()

    def save_all():  # a callback, so the list and the header below already show the saved state
        try:
            out = store.save(camera, ss[f"{key}_work"], TZ)
        except ValueError as e:
            flash(f"❌ Not saved: {e}")
            return
        where = FIXED_SLOTS.relative_to(ROOT) if FIXED_SLOTS.is_relative_to(ROOT) else FIXED_SLOTS
        flash(f"💾 Saved {len(out)} slot(s) for '{camera}' to {where}")
        restart(out)

    if f"{key}_ver" not in ss:  # the editor starts from the saved map
        restart(saved)
        ss[f"{key}_edit"] = None
    editing = ss.get(f"{key}_edit")

    with st.expander(f"✏️ Fixed slot editor · {camera} · {len(saved)} saved slot(s)", expanded=not saved or not running):
        text, until = ss.get(f"{key}_flash") or ("", 0)
        if time.time() < until:
            (st.error if text.startswith("❌") else st.success)(text)
        st.caption("Draw each slot where the vehicle's wheels stand: with the default 'Ground point' rule a vehicle "
                   "fills the slot that contains its bottom centre. Saved slots reach the running worker within a "
                   "second (the other slots keep their state).")
        if st_canvas is None:
            st.error("The slot drawing canvas needs the `streamlit-drawable-canvas` package (requirements.txt). "
                     "Install it in the running container: `docker exec carparking pip install "
                     "\"streamlit-drawable-canvas>=0.13,<0.14\"`, then restart the container (a rebuilt image "
                     "includes it).")
            return saved
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("📷 Grab latest frame", disabled=not url, key=f"{key}_grab"):
            with st.spinner("Grabbing a frame…"):
                frame, _ = grab_frame(url)
            if frame is None:
                st.error(f"No frame from {mask_url(url)}")
            else:
                ss["frame5"], ss["frame5_ts"] = frame, time.time()
        frame = ss.get("frame5")
        if frame is None:
            st.info("Press 📷 Grab latest frame to draw on the camera image (the running worker also provides it).")
            return saved

        if editing:  # one slot: canvas + corners + name; save / cancel / delete only this slot
            st.markdown(f"#### Editing slot {editing['id']} · {editing['name']}")
            slot_editor(frame, m5, key, [s for s in ss[f"{key}_work"] if s["id"] != editing["id"]])
            try:
                validate_slots([ss[f"{key}_edit"]])
                problem = None
            except ValueError as e:
                problem = str(e)
                st.error(problem)
            e1, e2, e3 = st.columns(3)
            e1.button("💾 Save this slot", type="primary", key=f"{key}_esave", disabled=bool(problem),
                      on_click=save_edit)
            e2.button("✖️ Cancel", key=f"{key}_ecancel", on_click=stop_edit)
            e3.button("🗑️ Delete this slot", key=f"{key}_edelete", on_click=delete, args=(editing["id"],))
            return saved

        if b2.button("✨ Draft from detections", disabled=det is None, key=f"{key}_draft",
                     help="One rectangle per vehicle the detector sees now (inside the parking area); replaces the "
                          "slots on the canvas (saved slots stay saved until you press Save Slots)."):
            det.conf = conf
            if m5["classes"] is not None:
                det.set_targets(m5["classes"])
            mask = lot_mask(frame, m5)
            vehicles = [d for d in det(frame) if d["state"] is not None and inside(mask, d["poly"])]
            restart(draft_slots(vehicles, frame.shape))
            st.toast(f"{len(vehicles)} slot(s) drafted from detections")
        if b3.button("↩️ Revert to saved", key=f"{key}_revert"):
            restart(saved)
        if b4.button("🗑️ Clear canvas", key=f"{key}_clear", help="Removes all shapes from the canvas; saved slots "
                                                                  "stay saved until you press Save Slots"):
            restart([])

        slots = canvas_editor(frame, m5, key)
        ss[f"{key}_work"] = slots
        problems = []
        try:
            validate_slots(slots)
        except ValueError as e:
            problems.append(str(e))
            st.error(problems[0])

        # one row per slot: its state against the saved file, ✏️ Edit, 🗑️ Delete
        by_id = {s["id"]: s for s in saved}
        st.markdown("**Slots**" if slots else "_No slots yet: draw one on the image._")
        for s in sorted(slots, key=lambda s: s["id"]):
            old = by_id.get(s["id"])
            status = ("✅ saved" if old == s else "🟠 changed, not saved" if old else "🆕 new, not saved")
            r1, r2, r3, r4 = st.columns([3, 2, 1, 1], vertical_alignment="center")
            r1.markdown(f"**{s['id']} · {s['name']}**  \n`{format_points(s['poly'])}`")
            r2.caption(status)
            r3.button("✏️ Edit", key=f"{key}_edit_{s['id']}", on_click=start_edit, args=(s["id"],),
                      help="Move or reshape this slot on the canvas, fine-tune its corners, rename it; saved at once")
            r4.button("🗑️ Delete", key=f"{key}_del_{s['id']}", on_click=delete, args=(s["id"],),
                      help="Removes only this slot (from the saved file too, at once)")
        for sid in sorted(set(by_id) - {s["id"] for s in slots}):
            r1, r2, r3, r4 = st.columns([3, 2, 1, 1], vertical_alignment="center")
            r1.markdown(f"~~{sid} · {by_id[sid]['name']}~~")
            r2.caption("removed from the canvas, still saved")
            r4.button("🗑️ Delete", key=f"{key}_del_{sid}", on_click=delete, args=(sid,))

        dirty = [(s["id"], s["name"], s["poly"]) for s in slots] != [(s["id"], s["name"], s["poly"]) for s in saved]
        s1, s2 = st.columns([1, 3])
        s1.button("💾 Save Slots", type="primary", disabled=bool(problems) or not dirty, key=f"{key}_save",
                  on_click=save_all, help="Stores all slots on the canvas (replaces the saved map of this camera)")
        if dirty and not problems:
            s2.warning("Unsaved changes: the monitor uses the saved slots until you press Save Slots.")
    return saved


def _local(iso: str | None) -> str:
    """ISO 8601 from HistoryDB (already in the camera's time zone) -> '2026-10-19 10:03:27'."""
    return "" if iso is None else iso[:19].replace("T", " ")


def history_section(db: HistoryDB):
    """'Parking History & Logs': parking sessions (In / Out time) and the raw entry/exit event log, each with a CSV
    download. -> refresh(), which redraws counts and tables (the live loop calls it after each logged event)."""
    st.subheader("🅿️ Parking History & Logs")
    caption = st.empty()
    tab_s, tab_e = st.tabs(["Sessions (In / Out)", "Event log (entry / exit)"])
    # "ignore": no rerun on click, so a running stream keeps going. The file is the data as of the last rerun.
    tab_s.download_button("⬇️ Download sessions (CSV)", db.csv(), file_name=f"parking_sessions_{slug(db.camera)}.csv",
                          mime="text/csv", on_click="ignore", key="dl_sessions")
    tab_e.download_button("⬇️ Download event log (CSV)", db.events_csv(),
                          file_name=f"parking_events_{slug(db.camera)}.csv", mime="text/csv", on_click="ignore",
                          key="dl_events")
    sessions, events = tab_s.empty(), tab_e.empty()

    def refresh():
        s = db.summary()
        caption.caption(f"{db.scope} · {s['sessions']} sessions, {s['open_sessions']} still parked · {s['events']} "
                        f"events · stored in {HISTORY_DB.name} · newest first")
        sessions.dataframe(pd.DataFrame(  # DataFrame with columns: an empty log still shows its headers
            [(r["id"], r["slot_id"], _local(r["in_time"]), _local(r["out_time"]), r["duration_min"], r["status"],
              r["note"]) for r in db.sessions(200)],
            columns=["Session", "Slot", "In Time", "Out Time", "Duration (min)", "Status", "Note"]),
            hide_index=True, width="stretch")
        events.dataframe(pd.DataFrame(
            [(_local(r["time"]), r["slot_id"], "🚗 IN" if r["event"] == "entry" else "🅿️ OUT", r["session_id"],
              r["note"]) for r in db.events(200)],
            columns=["Time", "Slot", "Event", "Session", "Note"]), hide_index=True, width="stretch")

    refresh()
    return refresh


UI5_KEYS = ("model5", "conf5", "stream5", "url5", "camera5", "vsrc5", "slotmode5", "match5", "matchiou5", "size5",
            "classes5", "extra5", "save5", "hold5", "exit5", "exitn5", "occl5", "hmax5", "lhold5", "learn5",
            "absent5", "enabled5")


def seed5(store: LiveStore) -> dict:
    """Mode 5 widgets start from the saved settings (live_config), so a new tab shows what the worker runs instead of
    the defaults (and can't push the defaults to the worker). Once per settings version: a change made in another
    tab arrives on this tab's next rerun. -> the saved widget values."""
    ss = st.session_state
    cfg, ver = store.config()
    ui = (cfg or {}).get("ui", {})
    if ss.get("_cfg_ver5") != ver:
        for k, v in ui.items():
            ss[k] = v
        ss["_cfg_ver5"] = ver
    return ui


def init5(key: str, default, options=None, saved: dict | None = None):
    """A Mode 5 widget's state before it is drawn: the session's value, else the saved one, else default; values no
    longer among options (another detector's classes, a removed model) fall back to the default."""
    ss = st.session_state
    if key not in ss:
        ss[key] = (saved or {}).get(key, default)
    if options is not None:
        if isinstance(ss[key], list):
            ss[key] = [v for v in ss[key] if v in options]
        elif ss[key] not in options:
            ss[key] = default


@st.cache_resource
def _spawned() -> dict:
    """The worker process started by this page server (embedded mode), shared by all sessions."""
    return {"proc": None, "t": 0.0}


def ensure_worker(store: LiveStore, enabled: bool) -> str | None:
    """Embedded mode: start scripts/worker.py as its own process when monitoring is on and no worker beats. It keeps
    running when tabs close; a second one would only wait on standby (lease). -> a note for the status line."""
    if WORKER_MODE != "embedded" or not enabled:
        return None
    age = heartbeat_age(store.state(images=False))
    if age is not None and age <= STALE:
        return None
    sp = _spawned()
    if sp["proc"] is not None and sp["proc"].poll() is None:
        return f"worker process {sp['proc'].pid} is starting (loading the model) or stuck: see logs/worker.log"
    if time.time() - sp["t"] < 10:
        return "starting the worker…"
    log = HISTORY_DB.parent / "logs" / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        sp["proc"] = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "worker.py")], stdout=f,
                                      stderr=subprocess.STDOUT, start_new_session=True, cwd=str(ROOT))
    sp["t"] = time.time()
    return f"started worker process {sp['proc'].pid}"


VIEW_EVERY = {"5 s": 5.0, "10 s": 10.0, "15 s": 15.0, "30 s": 30.0}  # Mode 5 viewer refresh choices


def live_view(store: LiveStore, db: HistoryDB, every: float = 10.0):
    """Mode 5 viewer: the worker's status, overlay, metrics and the history, re-read from the database every `every`
    seconds (and on 🔄 Refresh now). Display only: nothing here detects or logs, so the refresh rate never slows the
    worker, which processes and logs every frame on its own."""

    @st.fragment(run_every=every)
    def panel():
        cfg, ver = store.config()
        enabled = bool((cfg or {}).get("enabled"))
        note = ensure_worker(store, enabled)
        state = store.state()
        age, snap = heartbeat_age(state), state.get("snapshot") or {}
        down = age is None or age > STALE
        if down:
            why = "no worker has run yet" if age is None else f"no heartbeat for {age:.0f} s"
            if not enabled:
                st.info(f"⏸️ Monitoring is off ({why}). Switch it on above: a background worker then tracks the "
                        "camera and logs entries/exits, also with this tab closed.")
            else:
                hint = note or ("check the worker service: `docker compose logs worker`" if WORKER_MODE == "external"
                                else "")
                st.error(f"🔴 Worker down: {why}." + (f" {hint}" if hint else ""))
        elif not snap.get("enabled"):
            st.info(f"⏸️ Worker connected (heartbeat {age:.1f} s ago) · monitoring is off.")
        elif snap.get("cfg_ver") != ver:
            st.info("⏳ Worker connected · applying the new settings…")
        elif snap.get("waiting") and snap.get("frame_age") is None:
            st.info(f"⏳ Worker connected · {snap.get('url', '')}: {snap.get('status', '')}")
        else:
            line = (f"🟢 Worker connected · heartbeat {age:.1f} s ago · {snap.get('url', '')}: {snap.get('status', '')}"
                    + (f" · {snap['shape'][0]}×{snap['shape'][1]}" if snap.get("shape") else "")
                    + f" · {snap.get('rate', 0):.2f} inferences/s · frame age {max(snap.get('frame_age') or 0, 0):.1f} s"
                    + (f" · {snap['detector']}" if snap.get("detector") else ""))
            if snap.get("waiting") and (snap.get("frame_age") or 0) > 5:
                st.warning(f"⚠️ No new frames for {snap['frame_age']:.0f} s: {snap.get('status', '')}. "
                           "Showing the last frame.")
            else:
                st.success(line)
        if not down and snap.get("enabled"):
            if snap.get("det_text"):
                (st.success if snap.get("det_auto") else st.info)(f"{snap['det_text']} · slot map: {snap.get('scope')}")
            if snap.get("scope") and snap["scope"] != db.scope:
                st.warning(f"The worker runs {snap['scope']}; this page shows {db.scope}.")
            c = snap.get("counts")
            if c:
                raw = snap.get("raw_counts") or {}
                st.caption(f"FREE {c['free']} · BUSY {c['busy']} (hidden {c['hidden']}) · UNKNOWN {c['unknown']} · "
                           f"slots {c['slots']} · "
                           + ("fixed map" if snap.get("mode") == "fixed" else
                              f"learning {snap.get('learning', 0)} · phantom boxes ignored {snap.get('dropped', 0)}")
                           + f" · {snap.get('vehicles', 0)} vehicles · {snap.get('others', 0)} other"
                           + (" · raw " + ", ".join(f"{k} {v}" for k, v in raw.items()) if raw else ""))
            if snap.get("saved"):
                st.caption(f"📸 {snap['saved']} frame(s) saved by the worker, last: dataset_raw/{snap['last_saved']}")
        if state.get("overlay"):
            st.image(state["overlay"], output_format="JPEG", width="stretch",
                     caption="Last image from the worker (stale)" if down else None)
        if not down and snap.get("code", 0) < code_version(ROOT) - 1e-6:
            r1, r2 = st.columns([3, 1], vertical_alignment="center")
            r1.caption("The worker still runs older code (the files changed since it started).")
            if r2.button("🔄 Restart worker", key="restart5"):
                store.command("restart")  # it exits; docker compose or this page starts a fresh one
                st.toast("Restart requested")

    @st.fragment(run_every=every)
    def history():
        history_section(db)

    panel()
    history()


def decode(data: bytes):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not decode that image.")
        st.stop()
    return frame


st.set_page_config(page_title="CarParking tester", layout="wide")
st.title("CarParking: FREE / BUSY tester")

available = tuple(k for k, p in WEIGHTS.items() if p.exists())
if not available:
    st.error("No weights found in models/. Train with `python scripts/train_yolo.py` "
             "(and `--task obb` for the OBB model).")
    st.stop()

store5 = LiveStore(HISTORY_DB)  # Mode 5 settings + the background worker's published state (carparking/live.py)
saved5 = seed5(store5)
init5("model5", list(CHOICES)[0], list(CHOICES), saved5)
choice = CHOICES[st.sidebar.radio("Model", list(CHOICES), key="model5")]
mode = st.sidebar.radio("Input", ["Mode 1: Single Image Test", "Mode 2: Video Stream Test",
                                  "Mode 3: Live Webcam", "Mode 4: Dynamic IoU (Camera Independent)",
                                  "Mode 5: RTSP IP Camera (Entry/Exit History)"])
mode5 = mode.startswith("Mode 5")  # own slider: Mode 5 starts lower (shaded cars, bikes), finer steps, own value
if mode5:
    init5("conf5", 0.30, saved=saved5)
conf = st.sidebar.slider("Confidence threshold", 0.05, *((0.50, None, 0.01) if mode5 else (0.95, 0.20, 0.05)),
                         key="conf5" if mode5 else None, help="Mode 5 default 0.30 for COCO yolov8s at 1920 px: back-row "
                              "cars score 0.56–0.89, a false box across the front car's roof ~0.25. yolov8n: 0.20 "
                              "(back-row cars only 0.20–0.26). Camera model / hybrid: ~0.35. Cut fixed false boxes "
                              "away with the parking area")
st.sidebar.caption(" · ".join(f"{k}: {p.name}" + ("" if p.exists() else " (missing)") for k, p in WEIGHTS.items()))

if mode.startswith("Mode 1"):
    up = st.file_uploader("Parking image", type=["jpg", "jpeg", "png"])
    if up:
        frame = decode(up.getvalue())
        show(frame, route(frame, choice, available), conf)
elif mode.startswith("Mode 3"):
    cam = st.sidebar.number_input("Camera index", 0, 9, 0, help="0 = built-in laptop camera, 1 = first USB camera")
    every = st.sidebar.slider("Run the model every Nth frame", 1, 15, 3,
                              help="Frames in between show the last result. Raise it if the feed lags (CPU).")
    st.caption("Hold a printed parking photo (e.g. a PUCPR frame) flat, 30–50 cm from the camera, in even light. "
               "The model is chosen once, on the first frame: untick and tick again after changing the scene.")
    if st.checkbox("Start Live Webcam"):
        live(int(cam), every, choice, available, conf)
elif mode.startswith("Mode 4"):
    vsrc = st.sidebar.radio("Vehicle detector", [*COCO, BAY_SRC],
                            help="COCO: real car/bus/truck boxes (recommended). Bay model: the Model selector above; "
                                 "its BUSY bay boxes stand in for cars.")
    cover = st.sidebar.slider("Coverage threshold (BUSY above)", 0.05, 0.90, 0.25, 0.05,
                              help="Share of the slot's area covered by a vehicle box. Main rule.")
    thr = st.sidebar.slider("IoU threshold (BUSY above)", 0.05, 0.90, 0.35, 0.05, help="Second rule, either one = BUSY")
    ssrc = st.sidebar.radio("Slot Source", [SLOT_ANCHOR, SLOT_BAY, SLOT_HOUGH],
                            help="Auto-Anchor: parked vehicles are grouped into rows and the empty bays between/after "
                                 "them are extrapolated. Bay model: every bay the trained model finds (Model selector) "
                                 "is a slot. Hough: slots built from the painted lines in the frame.")
    m4 = {"top": 0.25, "bottom": 0.0, "left": 0.0, "right": 0.0, "roi": None, "extend": 2, "main": True, "hull": True}
    with st.sidebar.expander("Parking area (hard walls)", expanded=True):
        st.caption("Everything outside is dimmed in the image and ignored: no vehicles, no slots, for every slot "
                   "source. Cuts in % of the frame: type a value or step it with the buttons; the cyan line is the "
                   "edge.")
        m4["top"] = cut_input(st, "Cut top", "cut4_top", 25, "Background road, buses, buildings")
        m4["bottom"] = cut_input(st, "Cut bottom", "cut4_bottom", 0)
        m4["left"] = cut_input(st, "Cut left", "cut4_left", 0, "Left driveway")
        m4["right"] = cut_input(st, "Cut right", "cut4_right", 0, "Right driveway")
        roi_text = st.text_input(
            "Polygon (optional, for slanted edges)", placeholder="0.05,0.25; 0.95,0.25; 0.95,1; 0.05,1",
            help="Corners as x,y shares of the frame width/height (0..1), separated by ';', clockwise or not. "
                 "Combined with the cuts above: a point must be inside both.")
        try:
            m4["roi"] = parse_roi(roi_text)
        except ValueError as e:
            st.error(f"Polygon ignored: {e}")
    if ssrc == SLOT_ANCHOR:
        m4["extend"] = st.sidebar.slider("Extra bays past each row end", 0, 6, 2,
                                         help="How far a row is assumed to continue beyond its outermost car")
        m4["hull"] = st.sidebar.checkbox(
            "Stay inside the lot outline", True,
            help="Extra bays and lone cars must lie within the outline of all parked rows, so an edge row or column "
                 "can't run on into a driveway. Untick only for a lot with a single row.")
        m4["main"] = st.sidebar.checkbox(
            "Main lot only", True, help="Only the biggest group of touching rows anchors the grid; separate rows "
                                        "(e.g. street parking across a road) are left out. Untick for lots split by "
                                        "a wide internal road.")
        st.caption("Slots are extrapolated from the parked vehicles (rows in any direction): the gaps between cars "
                   "become FREE bays. A row needs at least 3 cars; a fully empty row is not seen.")
    if ssrc == SLOT_HOUGH:
        st.caption("Slots are built from the painted lines in the frame (Hough), not from slots.json. "
                   "Works on straight, roughly parallel bay lines; no visible lines = no slots.")
    st.caption("Yellow outlines = detected vehicles; slot colours come from vehicle coverage/IoU, not from the "
               "bay model's own FREE/BUSY.")
    if st.sidebar.radio("Source", ["Image upload", "Live webcam"]) == "Image upload":
        up = st.file_uploader("Parking image", type=["jpg", "jpeg", "png"])
        if up:
            frame = decode(up.getvalue())
            det, sliced, text, auto = vehicle_detector(frame, vsrc, choice, available)
            slots_of, stext = slot_finder(frame, ssrc, choice, available, conf, m4)
            slots, img = iou_slots(frame, det, sliced, conf, thr, cover, slots_of, m4)
            (st.success if auto else st.info)(text)
            if stext:
                st.info(stext)
            if not slots:
                st.warning({SLOT_ANCHOR: "No vehicles found to anchor on (try a lower confidence).",
                            SLOT_BAY: "No slots found: the bay model saw no bays (try a lower confidence)."}.get(
                    ssrc, "No parking lines found: try a sharper image or a straighter view of the lines."))
            show_metrics(slots)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
            st.dataframe([{"id": i, "state": STATE[s["state"]], "coverage": s["cover"], "best IoU": s["iou"],
                           "corners": str(s["poly"])} for i, s in enumerate(slots)], hide_index=True)
    else:
        cam = st.sidebar.number_input("Camera index", 0, 9, 0, key="cam4")
        every = st.sidebar.slider("Run the model every Nth frame", 1, 15, 3, key="every4")
        if st.checkbox("Start Live Webcam", key="live4"):
            live_iou(int(cam), every, vsrc, ssrc, choice, available, conf, thr, cover, m4)
elif mode.startswith("Mode 5"):
    streams = ["Main stream (101, full resolution)", "Sub-stream (102, lower latency)", "Custom URL"]
    init5("stream5", streams[0], streams, saved5)
    stream = st.sidebar.radio("Stream", streams, key="stream5",
                              help="The sub-stream is 640×360 on this camera: faster, but small far-away cars get lost")
    if stream == "Custom URL":
        init5("url5", "", saved=saved5)
        url = st.sidebar.text_input("RTSP URL", type="password", key="url5",
                                    help="rtsp://user:pass@host:554/... (a @ in the password is written %40). "
                                         "A digit, e.g. 0, opens a local webcam instead.")
    else:
        url = stream_url(RTSP_URL, "sub" if stream.startswith("Sub") else "main")
    init5("camera5", "Camera 1", saved=saved5)
    camera = st.sidebar.text_input("Camera name", key="camera5",
                                   help="Key for the parking area, slot map and history. Main and sub-stream of one "
                                        "camera share a name; models/cam_<name>.pt is this camera's own model "
                                        "(\"Front Parking 02\" -> cam_front_parking_02.pt).").strip() or "Camera 1"
    # order: stock COCO yolov8n (default), other COCO, this camera's own fine-tune, the hybrid of it + COCO, other
    # cameras' models, bay. The fine-tunes stay selectable but are no longer the automatic primary detector.
    own_weights = ROOT / "models" / f"cam_{slug(camera)}.pt"
    own = f"{CAM_PREFIX}{own_weights.name}"
    coco_keys = [k for k in COCO if k.startswith(COCO_PREFIX)]  # yolov8s first when present, then yolov8n
    coco_weights = COCO[coco_keys[0]] if coco_keys else None
    if own in COCO and coco_weights:
        HYBRID[f"🔀 Hybrid Dual Detector · {own_weights.name} + {coco_weights.name}"] = (own_weights, coco_weights)
    cams = sorted((k for k in COCO if k.startswith(CAM_PREFIX)), key=lambda k: k != own)
    detectors = coco_keys + cams[:1] + list(HYBRID) + cams[1:] + [BAY_SRC]
    init5("vsrc5", detectors[0], detectors, saved5)
    vsrc = st.sidebar.radio("Detector", detectors, key="vsrc5",
                            help="🚗 COCO (default): stock yolov8s (yolov8n if absent), car, motorcycle, bus, truck (classes 2, 3, 5, 7), "
                                 "whole-frame inference. 📷 Camera model: fine-tuned on one camera (car + motorcycle). "
                                 "🔀 Hybrid: the camera model and stock COCO on every frame, duplicates merged (the "
                                 "camera model's box wins); adds people/animals (shown, never a slot) and vehicle "
                                 "types the fine-tune never saw. 🅿️ Bay model: PKLot free/busy bays from the Model "
                                 "selector above; for high top-down views only.")
    init5("slotmode5", SLOT_MODES[0], SLOT_MODES, saved5)
    fixed_mode = st.sidebar.radio(
        "Slot mapping mode", SLOT_MODES, key="slotmode5", horizontal=True,
        help="Dynamic: slots are learned from where vehicles stand (default). Fixed: you draw the slots once in the "
             "editor (saved to config/fixed_slots.json per camera name); vehicles are checked against them and "
             "nothing is learned. Both log entries/exits the same way.") == SLOT_MODES[1]
    # slot map and history are per camera AND detector (and fixed slots apart from learned ones): slots another
    # detector learned are never loaded here, and slot IDs in the history always refer to the map they came from
    detector_id = (f"hybrid_{own_weights.stem}+{coco_weights.stem}" if vsrc in HYBRID else
                   COCO[vsrc].stem if vsrc in COCO else f"bay_{choice}")
    db = HistoryDB(HISTORY_DB, camera, TZ, detector=detector_id + (" | fixed" if fixed_mode else ""))
    m5 = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "roi": None, "whole": None, "classes": None,
          "extra": [], "fixed": None, "match": "ground", "match_iou": 0.15}
    det5 = None
    if fixed_mode:
        with st.sidebar.expander("Fixed slot matching", expanded=True):
            init5("match5", "ground", list(MATCH_RULES), saved5)
            m5["match"] = st.radio("A vehicle fills a slot when…", list(MATCH_RULES), key="match5",
                                   format_func=lambda r: {"ground": "its ground point is in it (low angle)",
                                                          "center": "its centre is in it (top-down)",
                                                          "iou": "its box overlaps it (IoU)"}[r],
                                   help="\n\n".join(MATCH_RULES.values()))
            if m5["match"] == "iou":
                init5("matchiou5", 0.15, saved=saved5)
                m5["match_iou"] = st.slider("Minimum IoU", 0.05, 0.8, None, 0.05, key="matchiou5")
    if vsrc != BAY_SRC:
        # yolov8s at 1920 px is what separates the back-row cars of Front Parking 02 (at 1280 it boxes the white
        # hatchback and the dark car behind it as one, like yolov8n at any size)
        big = vsrc in COCO and COCO[vsrc].name == "yolov8s.pt"
        sizes = ["Whole frame 1280 px", "Whole frame 1920 px", "SAHI 640 px tiles"]
        init5("size5", sizes[int(big)], sizes, saved5)
        size = st.sidebar.radio("Vehicle inference", sizes, key="size5",
                                help="Whole frame: a car close to the camera stays in one piece (SAHI tiles cut the "
                                     "pickup in half). 1920 px: more pixels for far, small or half-hidden vehicles, "
                                     "~1.5 s/frame for yolov8s on this CPU (default for yolov8s). Camera models: use "
                                     "the size they were trained at.")
        m5["whole"] = {"Whole frame 1280 px": 1280, "Whole frame 1920 px": 1920}.get(size)
        files = " + ".join(p.name for p in HYBRID[vsrc]) if vsrc in HYBRID else COCO[vsrc].name
        try:  # the same cached detector(s) as the stream loop
            det5 = get_hybrid(*HYBRID[vsrc], m5["whole"]) if vsrc in HYBRID else get_detector(COCO[vsrc], m5["whole"])
        except Exception as e:  # e.g. a segment/classify model or a half-copied file named cam_*.pt
            st.error(f"{files} could not be loaded as detection model(s): {e}")
            st.stop()
        kinds, others = det5.vehicle_names(), det5.other_names()  # hybrid: union of both models' classes
        if not kinds:
            st.error(f"{files} has no vehicle class. Name classes car, truck, bus, motorcycle, vehicle, van, pickup, "
                     "suv, motorbike, bike or scooter.")
            st.stop()
        init5("classes5", kinds, kinds, saved5)
        m5["classes"] = st.sidebar.multiselect("Target classes (parking)", kinds, key="classes5",
                                               help="All count as BUSY. COCO ids 2 car, 3 motorcycle, 5 bus, 7 truck; "
                                                    "camera models from auto_label.py: car (car/truck/bus) and "
                                                    "motorcycle. Filtered inside the model, so other objects can't "
                                                    "suppress a vehicle.")
        if others:  # COCO or hybrid: people, animals, ... drawn in magenta, never a slot, never an entry/exit
            init5("extra5", [n for n in ("person", "dog", "cat") if n in others] if vsrc in HYBRID else [], others,
                  saved5)
            m5["extra"] = st.sidebar.multiselect(
                "Also detect (shown only, never a slot)", others, key="extra5",
                help="Drawn in magenta with their class; they never occupy a slot or log an entry/exit.")
    with st.sidebar.expander("Parking area (hard walls)", expanded=True):
        st.caption("Vehicles and slots must lie inside (centre, ground point and 90 % of the box), else they are "
                   "ignored (building, stairs, timestamp, road). Polygon: shares of the frame; cuts: % of the frame, "
                   "typed or stepped with the buttons. Both are saved per camera name. Outside is dimmed in the "
                   "image, the edge is cyan.")
        roi_text = st.text_input(
            "Polygon (saved per camera name)", db.setting("roi"), key=f"roi5_{db.camera}",
            placeholder="0.40,0.10; 0.58,0.10; 0.58,1; 0.17,1; 0.17,0.62; 0.36,0.48; 0.40,0.30",
            help="Corners as x,y shares of the frame width/height (0..1), separated by ';'. The placeholder is the "
                 "covered lane of Front Parking 02 (building and stairs left, pillars/road right, OSD text top).")
        try:
            m5["roi"] = parse_roi(roi_text)
            if roi_text != db.setting("roi"):
                db.set_setting("roi", roi_text)
        except ValueError as e:
            st.error(f"Polygon ignored: {e}")
        for side, default, help_text in (
                ("top", 0, None), ("bottom", 0, None), ("left", 0, None),
                ("right", 40, "Default 40 %: Front Parking 02's road and driveway on the right")):
            saved_cut = db.setting(f"cut_{side}")  # per camera name, like the polygon
            m5[side] = cut_input(st, f"Cut {side}", f"cut5_{side}_{db.camera}",
                                 int(saved_cut) if saved_cut.isdigit() else default, help_text,
                                 save=lambda v, side=side: db.set_setting(f"cut_{side}", str(v)))
    with st.sidebar.expander("Dataset capture (fine-tuning)"):
        init5("save5", 0, saved=saved5)
        m5["save_every"] = st.slider("Save a raw frame every N seconds (0 = off)", 0, 600, None, 10, key="save5",
                                     help="Full-resolution frames without overlay, to dataset_raw/, while the stream "
                                          "runs. 60–120 s over several days gives varied light; see "
                                          "scripts/collect_dataset.py for unattended capture.")
    with st.sidebar.expander("Event timing"):
        for k, v in (("hold5", 15), ("exit5", 30), ("exitn5", 15), ("occl5", 0.6), ("hmax5", 30), ("lhold5", 20), ("learn5", True),
                     ("absent5", vsrc != BAY_SRC)):
            init5(k, v, saved=saved5)
        m5["hold"] = st.slider("Seconds a change must last", 0, 120, None, 5, key="hold5",
                               help="A slot turns BUSY only after a vehicle has been seen this long, so cars driving "
                                    "past don't log an entry")
        m5["exit_hold"] = st.slider("Exit buffer: seconds visibly empty before FREE", 0, 600, None, 15, key="exit5",
                                    help="BUSY → FREE needs the slot to be seen empty this long. Time while the slot "
                                         "is hidden behind another vehicle does not count.")
        m5["exit_frames"] = st.slider("Exit: consecutive frames without a vehicle", 1, 60, None, 1, key="exitn5",
                                      help="BUSY → FREE also needs this many processed frames in a row with no box in "
                                           "the slot (whole frame or zoom), on top of the exit buffer: a missed frame, "
                                           "a shadow or a short occlusion never frees a slot. At ~1.5 s per frame, 15 "
                                           "frames ≈ 22 s.")
        m5["occlusion"] = st.slider("Occlusion: vehicles in front cover ≥", 0.1, 0.9, None, 0.05, key="occl5",
                                    help="Share of a slot covered by the vehicles standing in front of it (lower in "
                                         "the image, together) that makes it HIDDEN: state kept, exit countdown "
                                         "paused. In a low-angle row every bay is ~half behind the car in front, so "
                                         "keep it above 0.5 or emptied bays stay BUSY.")
        m5["hidden_max"] = st.slider("Max seconds a hidden slot keeps its state", 10, 1800, None, 10, key="hmax5",
                                     help="After this long without any vehicle in it, a slot counts as empty even "
                                          "while hidden: BUSY → FREE on its own, the exit logged at the moment its "
                                          "car was last seen.")
        m5["learn_hold"], m5["learn"] = 20, False
        if not fixed_mode:  # fixed slots are never learned
            m5["learn_hold"] = st.slider("Seconds before a new slot is learned", 0, 300, None, 5, key="lhold5",
                                         help="A box in a new place becomes a slot once it has stayed there this long")
            m5["learn"] = st.checkbox("Learn new slots", key="learn5", help="Untick to freeze the slot map once it is complete")
        m5["absent_free"] = st.checkbox("No detection = FREE", key="absent5",
                                        help="On for COCO (it never reports empty bays). Off for the bay model: a bay "
                                             "it misses keeps its last state.")

    alive = (heartbeat_age(store5.state(images=False)) or STALE + 1) <= STALE  # a worker is beating
    init5("enabled5", False, saved=saved5)
    running = alive and st.session_state["enabled5"]
    if fixed_mode:
        live5 = store5.state()  # the worker's newest raw frame is the editor's background
        if live5.get("raw") and (live5.get("raw_ts") or 0) > st.session_state.get("frame5_ts", 0):
            st.session_state["frame5"], st.session_state["frame5_ts"] = decode_jpeg(live5["raw"]), live5["raw_ts"]
        m5["fixed"] = fixed_slot_editor(url, camera, det5, conf, m5, running, db)
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
    enabled = c1.toggle("Monitoring (background worker)", key="enabled5", disabled=not url,
                        help="On: a background worker (scripts/worker.py) reads the camera, tracks the slots and logs "
                             "entries/exits with these settings, also with this tab closed; any number of tabs only "
                             "view it. Off: the worker idles. Setting changes reach it within a second.")
    # the settings the worker runs with (live_config): the widget values (to seed other tabs) and what they mean
    rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()  # noqa: E731
    spec = ({"kind": "hybrid", "weights": [rel(own_weights), rel(coco_weights)], "whole": m5["whole"]}
            if vsrc in HYBRID else {"kind": "coco", "weights": [rel(COCO[vsrc])], "whole": m5["whole"]}
            if vsrc in COCO else {"kind": "bay", "choice": choice, "bay": {k: rel(p) for k, p in WEIGHTS.items()}})
    run5 = {"camera": camera, "stream": "custom" if stream == "Custom URL" else "sub" if stream.startswith("Sub")
            else "main", "custom_url": url if stream == "Custom URL" else "", "db_detector": db.detector,
            "conf": conf, "detector": spec, "fixed_mode": fixed_mode, "fixed": m5["fixed"] if fixed_mode else None,
            "m5": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m5.items() if k != "fixed"}}
    ui5 = {**saved5, **{k: st.session_state[k] for k in UI5_KEYS if k in st.session_state}}
    st.session_state["_cfg_ver5"] = store5.set_config({"enabled": enabled, "ui": ui5, "run": run5})
    if c3.button("📸 Save snapshot", help="One raw full-resolution frame to dataset_raw/", disabled=not url):
        if running:
            store5.command("snapshot")
            st.toast("The worker saves its next frame to dataset_raw/")
        else:
            with st.spinner("Grabbing a frame…"):
                path = snapshot(url, db.camera)
            (st.success(f"Saved dataset_raw/{path.name}") if path else st.error(f"No frame from {mask_url(url)}"))
    if c2.button("Close open sessions" if fixed_mode else "Reset slot map",
                 help="Closes open parking sessions now (history rows are kept)" +
                      ("; the fixed slots stay." if fixed_mode else " and forgets the learned slots (after moving "
                                                                     "the camera).")):
        if alive:  # the worker holds the slot map in memory: it resets both
            store5.command("reset")
        else:
            db.reset()
        st.toast("Open sessions closed" if fixed_mode else "Slot map cleared")
    st.caption("Yellow = what the detector sees in this frame. Numbered = tracked slots (grey = state not confirmed "
               "yet). Entries/exits are logged to parking_history.db after the hold time, by the background worker.")
    if not url:
        st.info("No camera URL: set CARPARK_RTSP_URL in .env (see .env.example), or choose Custom URL in the sidebar.")
    elif fixed_mode and not m5["fixed"]:
        st.warning(f"No fixed slots saved for '{camera}' yet: draw them in the editor above and press 💾 Save Slots "
                   "(or switch to Dynamic).")
        history_section(db)
    else:
        # per browser tab only (session state, not the saved settings): a change here never restarts the worker
        r1, r2 = st.columns([3, 1], vertical_alignment="bottom")
        every = VIEW_EVERY[r1.select_slider(
            "Viewer refresh", list(VIEW_EVERY), value="10 s", key="refresh5",
            help="How often this page re-reads the worker's image, status and history. Display only: the worker "
                 "keeps processing and logging every frame at full speed whatever you pick.")]
        r2.button("🔄 Refresh now", key="refresh_now5",
                  help="Re-read the latest image and history at once")  # a click reruns the page
        live_view(store5, db, every)
else:
    up = st.file_uploader("Parking video", type=["mp4", "avi", "mov"])
    stride = st.sidebar.slider("Process every Nth frame", 1, 30, 5)
    if up and st.button("Start"):
        with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
            f.write(up.getvalue())  # OpenCV needs a file path
        cap = cv2.VideoCapture(f.name)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            status, view, bar, i, r = st.empty(), st.empty(), st.progress(0.0), 0, None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    r = r or route(frame, choice, available)  # fixed camera: route once, on the first frame
                    img, _, desc = detect(frame, r, conf)
                    (status.success if r["auto"] else status.info)(badge(r, desc))
                    view.image(img, width="stretch")
                    bar.progress(min(i / total, 1.0))
                i += 1
            bar.progress(1.0)
            st.success(f"Done: {i} frames, {len(range(0, i, stride))} processed.")
        finally:
            cap.release()
            Path(f.name).unlink(missing_ok=True)
