"""The Streamlit page (scripts/demo_app.py, left as it is) and carparking.live.build_settings() must store identical Mode 5
settings for the same widget values: then a settings write from another UI (FastAPI) is a no-op for the page and never
restarts the worker's tracking. Renders the real page (streamlit.testing.AppTest, the real detector class lists) for a
set of widget values and compares what it stored with build_settings()."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from carparking.history import HistoryDB  # noqa: E402
from carparking.live import BAY_SRC, SIZES, SLOT_MODES, STREAMS, LiveStore, build_settings, detector_options  # noqa: E402

URL = "rtsp://user:secret@cam.local:554/Streaming/Channels/101"


def page_settings(tmp_path, monkeypatch, ui: dict, cuts: dict | None = None, roi: str = "", fixed: dict | None = None):
    """Render Mode 5 with these stored widget values -> (what the page stored, the paths it used)."""
    db_path, fixed_path = tmp_path / "h.db", tmp_path / "cfg" / "fixed_slots.json"
    for k, v in {"CARPARK_HISTORY_DB": str(db_path), "CARPARK_FIXED_SLOTS": str(fixed_path),
                 "CARPARK_DATASET_RAW": str(tmp_path / "raw"), "CARPARK_RTSP_URL": URL,
                 "CARPARK_WORKER": "external"}.items():  # never start a worker from a test
        monkeypatch.setenv(k, v)
    camera = (ui.get("camera5") or "Camera 1").strip() or "Camera 1"
    db = HistoryDB(db_path, camera)
    for side, v in (cuts or {}).items():
        db.set_setting(f"cut_{side}", str(v))
    if roi:
        db.set_setting("roi", roi)
    db.close()
    if fixed:
        fixed_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_path.write_text(json.dumps(fixed))
    LiveStore(db_path).set_config({"ui": ui})  # the page seeds its widgets from these
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    radio = next(r for r in at.sidebar.radio if any(str(o).startswith("Mode 5") for o in r.options))
    radio.set_value(next(o for o in radio.options if str(o).startswith("Mode 5"))).run()
    assert not at.exception, [e.value for e in at.exception]
    cfg, _ = LiveStore(db_path).config()
    return cfg, db_path, fixed_path


@pytest.mark.parametrize("name", ["defaults", "tuned", "fixed", "hybrid", "bay", "custom", "stale"])
def test_page_and_builder_store_identical_settings(tmp_path, monkeypatch, name):
    hybrid = next((d for d in detector_options(ROOT, "Front Parking 02")[0] if d.startswith("🔀")), None)
    cases = {
        "defaults": ({}, None, ""),
        "tuned": ({"stream5": STREAMS[1], "camera5": "Front Parking 02", "conf5": 0.42, "size5": SIZES[0],
                   "hold5": 20, "exit5": 45, "exitn5": 10, "occl5": 0.5, "hmax5": 60, "lhold5": 30, "learn5": False,
                   "absent5": True, "save5": 120, "enabled5": True},
                  {"top": 10, "left": 12, "right": 42}, "0.40,0.10; 0.58,0.10; 0.58,1; 0.17,1"),
        "fixed": ({"slotmode5": SLOT_MODES[1], "match5": "iou", "matchiou5": 0.3, "camera5": "Cam F"}, None, ""),
        "hybrid": ({"camera5": "Front Parking 02", "vsrc5": hybrid, "extra5": ["person"], "classes5": ["car"]},
                   None, ""),
        "bay": ({"vsrc5": BAY_SRC, "model5": "YOLO-OBB (Oriented / Advanced)"}, None, ""),
        "custom": ({"stream5": STREAMS[2], "url5": "rtsp://x:y@other/1", "enabled5": True}, None, ""),
        "stale": ({"vsrc5": "a model that was deleted", "classes5": ["car", "spaceship"], "size5": "8K"}, None, ""),
    }
    ui, cuts, roi = cases[name]
    if name == "hybrid" and hybrid is None:
        pytest.skip("no models/cam_front_parking_02.pt + COCO model here")
    fixed = {"cameras": {"Cam F": {"slots": [{"id": 1, "name": "A1",
                                              "poly": [[0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.1, 0.5]]}]}}}
    page, db_path, fixed_path = page_settings(tmp_path, monkeypatch, ui, cuts, roi, fixed if name == "fixed" else None)
    built = build_settings(page["ui"], ROOT, db_path, URL, fixed_path)
    assert built == page
    if name == "fixed":
        assert page["run"]["fixed"][0]["name"] == "A1" and page["run"]["m5"]["match"] == "iou"
    if name == "tuned":
        assert page["run"]["m5"]["top"] == 0.10 and page["run"]["m5"]["roi"][0] == [0.4, 0.1]
