"""scripts/augment_far.py: far-away copies of a YOLO training set (shrunk onto a canvas, labels moved with them)."""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import augment_far  # noqa: E402


def test_the_picture_shrinks_onto_a_canvas_and_the_labels_move_with_it():
    img = np.full((400, 800, 3), 200, np.uint8)
    cv2.rectangle(img, (400, 200), (600, 300), (0, 0, 255), -1)  # one "car": centre (500, 250) px, 200 x 100 px
    labels = [[0, 500 / 800, 250 / 400, 200 / 800, 100 / 400]]
    out, lab = augment_far.shrink(img, labels, 0.5, random.Random(1), blur=0, quality=95)
    assert out.shape == img.shape and len(lab) == 1
    c, cx, cy, w, h = lab[0]
    assert (w, h) == (0.125, 0.125)  # half the size
    x, y = round(cx * 800), round(cy * 400)
    assert out[y, x, 2] > 200 and out[y, x, 0] < 80  # the red car is under the moved centre
    assert (out == 114).all(axis=2).any()  # a grey canvas around the shrunk picture


def test_boxes_that_become_too_small_are_dropped():
    img = np.zeros((400, 800, 3), np.uint8)
    labels = [[0, 0.5, 0.5, 0.05, 0.05], [0, 0.2, 0.2, 0.2, 0.2]]  # 40 px and 160 px wide
    _, lab = augment_far.shrink(img, labels, 0.1, random.Random(1), min_px=6.0)
    assert [round(b[3], 3) for b in lab] == [0.02]  # 4 px: gone; 16 px: kept


def test_a_dataset_gets_far_copies_of_the_train_split_only_and_a_rerun_adds_nothing(tmp_path, capsys):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(tmp_path / "images" / split / "a.jpg"), np.full((200, 400, 3), 90, np.uint8))
        (tmp_path / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    assert augment_far.main([str(tmp_path), "--copies", "2"]) == 0
    train = sorted(p.name for p in (tmp_path / "images" / "train").iterdir())
    assert train == ["a.jpg", "a_far0.jpg", "a_far1.jpg"] and [p.name for p in (tmp_path / "images" / "val").iterdir()] == ["a.jpg"]
    row = (tmp_path / "labels" / "train" / "a_far0.txt").read_text().split()
    assert row[0] == "0" and float(row[3]) < 0.3  # a smaller car
    assert augment_far.main([str(tmp_path), "--copies", "2"]) == 0
    assert len(list((tmp_path / "images" / "train").iterdir())) == 3  # no copies of copies, no duplicates
    assert augment_far.main([str(tmp_path / "nope")]) == 2
