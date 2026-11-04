"""Make "far away" copies of a YOLO training set, so a detector learns the small, low-resolution vehicles of a distant row.

A car 40 px wide in a 1920 px frame is a handful of pixels after the detector's own resize; stock COCO and most fine-tunes have seen few of them. Each
training image is shrunk by a random factor (default 0.35-0.65) and pasted onto a grey canvas of the original size (a corner / a random place), its
labels scaled and moved the same way, and then blurred + re-compressed a little like a distant, low-bitrate camera. Boxes that end up under --min-px
wide are dropped from the label (they are no longer a learnable object).

  python scripts/augment_far.py datasets/front_parking_02            # -> adds *_far0.jpg / *_far1.jpg ... to images/train and labels/train
  python scripts/augment_far.py datasets/front_parking_02 --copies 3 --scale 0.3 0.6 --split train
  then train as in scripts/auto_label.py:  yolo detect train data=datasets/front_parking_02/data.yaml model=yolov8s.pt imgsz=1280 epochs=60 ...

Only the train split is augmented (the val split stays real, or its mAP would be flattered). Safe to re-run: files ending in _farN are never augmented again and
existing ones are skipped. Works on any Ultralytics-format dataset (images/<split>, labels/<split>, "cls cx cy w h" normalised), PKLot-derived ones included.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

FAR = re.compile(r"_far\d+$")


def shrink(img: np.ndarray, labels: list[list[float]], scale: float, rng: random.Random, min_px: float = 6.0, blur: float = 0.8, quality: int = 70):
    """(image, labels): `img` scaled by `scale` onto a grey canvas of its own size at a random place, `labels` ([cls, cx, cy, w, h] normalised) moved with it
    and the ones narrower than `min_px` (of the canvas) dropped; softened by a Gaussian `blur` (sigma, 0 = none) and a JPEG round trip at `quality`."""
    h, w = img.shape[:2]
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(img, 114)  # the grey Ultralytics letterboxes with
    ox, oy = rng.randint(0, w - nw), rng.randint(0, h - nh)
    canvas[oy:oy + nh, ox:ox + nw] = small
    if blur > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    canvas = cv2.imdecode(cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
    out = []
    for c, cx, cy, bw, bh in labels:
        ncx, ncy, nbw, nbh = (ox + cx * w * scale) / w, (oy + cy * h * scale) / h, bw * scale, bh * scale
        if nbw * w >= min_px and nbh * h >= min_px:
            out.append([c, ncx, ncy, nbw, nbh])
    return canvas, out


def read_labels(path: Path) -> list[list[float]]:
    if not path.is_file():
        return []
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return [[float(v) for v in r[:5]] for r in rows if len(r) >= 5]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path, help="a dataset folder with images/<split> and labels/<split>")
    ap.add_argument("--split", default="train")
    ap.add_argument("--copies", type=int, default=2, help="far copies per image")
    ap.add_argument("--scale", type=float, nargs=2, default=(0.35, 0.65), metavar=("LOW", "HIGH"))
    ap.add_argument("--min-px", type=float, default=6.0, help="drop boxes narrower than this many pixels")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    images, labels = a.dataset / "images" / a.split, a.dataset / "labels" / a.split
    if not images.is_dir():
        print(f"no {images}", file=sys.stderr)
        return 2
    rng = random.Random(a.seed)
    made = 0
    for p in sorted(images.glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or FAR.search(p.stem):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        for k in range(a.copies):
            name = f"{p.stem}_far{k}"
            if (images / f"{name}.jpg").exists():
                continue
            im, lab = shrink(img, read_labels(labels / f"{p.stem}.txt"), rng.uniform(*a.scale), rng, a.min_px)
            cv2.imwrite(str(images / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            labels.mkdir(parents=True, exist_ok=True)
            (labels / f"{name}.txt").write_text("".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in lab))
            made += 1
    print(f"{made} far copies added to {images}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
