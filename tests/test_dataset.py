"""Dataset helpers: carparking/dataset.py and the pure functions of scripts/auto_label.py."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from auto_label import hour_block, split_of, yolo_lines  # noqa: E402
from carparking.dataset import change, save_frame, slug  # noqa: E402


def test_slug():
    assert slug("Front Parking 02") == "front_parking_02"
    assert slug("  ") == "camera"


def test_save_frame_name_and_content(tmp_path):
    frame = np.full((36, 64, 3), 128, np.uint8)
    p = save_frame(frame, tmp_path / "raw", "Front Parking 02", t=0.0)
    assert p.exists() and p.name.startswith("front_parking_02_19700101_") and p.suffix == ".jpg"


def test_change_ignores_noise_not_a_car():
    rng = np.random.default_rng(0)
    a = rng.integers(90, 110, (360, 640, 3), dtype=np.uint8)
    noisy = np.clip(a.astype(int) + rng.integers(-3, 4, a.shape), 0, 255).astype(np.uint8)
    car = a.copy()
    car[100:220, 200:420] = 240
    assert change(a, noisy) < 1.5 < change(a, car)


def test_split_by_capture_hour():
    names = [f"cam_20261018_{h:02d}{m:02d}00_000.jpg" for h in range(8, 18) for m in (0, 30)]
    assert hour_block(names[0]) == "20261018_08"
    split = split_of(names, 5)
    val_hours = {hour_block(n) for n, s in split.items() if s == "val"}
    assert val_hours == {"20261018_12", "20261018_17"}  # every 5th hour, both frames of an hour on the same side
    few = split_of(names[:5], 50)  # fewer hours than val_every: the last 20% by time
    assert [few[n] for n in sorted(few)] == ["train"] * 4 + ["val"]


def test_yolo_lines():
    assert yolo_lines([(0, 100, 50, 300, 150)], 1000, 500) == ["0 0.200000 0.200000 0.200000 0.200000"]
