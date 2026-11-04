"""Pre-label camera frames with a large COCO model and build an Ultralytics dataset for fine-tuning.

dataset_raw/*.jpg (scripts/collect_dataset.py, Mode 5 snapshots)
  -> datasets/<name>/images/{train,val}, labels/{train,val} (YOLO txt), preview/ (boxes drawn), data.yaml, classes.txt

Classes: 0 car (COCO car, bus, truck: a pickup flips between car and truck in COCO, the app treats all as one parked
vehicle), 1 motorcycle. Frames without vehicles get an empty label file: they teach the model that stairs, pillars and
lamps are background, which is what removes the false positives.

Pre-labels are a starting point, not ground truth: the teacher misses what COCO misses (the covered motorbike, shaded
cars). Review every image (see the steps below) before training.

Incremental: re-run after collecting more frames; only new images are labelled and split. An image already in the
dataset keeps its split and its (reviewed) label file. --overwrite re-labels everything.

Split by capture hour: every 5th hour block goes to val, so near-identical neighbouring frames never sit on both sides
(a random split would leak and report a too-good mAP).

Steps:
  1. python scripts/auto_label.py --add-box motorcycle:0.412,0.235,0.479,0.292
         # yolov8x.pt teacher (downloaded on first use, 130 MB) + a draft box for the covered bike in every frame
  2. Review/fix: open datasets/front_parking_02 in X-AnyLabeling or Label Studio (YOLO format, classes.txt);
     check preview/ first to find the frames that need work. Add missed vehicles, delete false boxes, delete the
     bike box where the bike is gone, move it where the bike moved. Then: python scripts/auto_label.py --stats
  3. yolo detect train data=datasets/front_parking_02/data.yaml model=yolov8n.pt imgsz=1280 epochs=60 patience=15
         batch=8 project=experiments/cam name=front_parking_02
  4. copy experiments/cam/front_parking_02/weights/best.pt models/cam_front_parking_02.pt   -> Mode 5 "Camera model"
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.dataset import slug  # noqa: E402

NAMES = ["car", "motorcycle"]
COCO_TO_OURS = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 1}


def hour_block(name: str) -> str:
    """'front_parking_02_20261018_131502_123.jpg' -> '20261018_13' (capture hour); file stem if no timestamp."""
    m = re.search(r"_(\d{8})_(\d{2})\d{4}_", name)
    return f"{m.group(1)}_{m.group(2)}" if m else Path(name).stem


def split_of(names: list[str], val_every: int) -> dict[str, str]:
    """Every val_every-th hour block -> val. Fewer blocks than that: the last 20% of frames (by time) -> val."""
    blocks = sorted({hour_block(n) for n in names})
    if len(blocks) >= val_every:
        val = set(blocks[val_every - 1::val_every])
        return {n: "val" if hour_block(n) in val else "train" for n in names}
    cut = int(len(names) * 0.8)
    return {n: "train" if i < cut else "val" for i, n in enumerate(sorted(names))}


def yolo_lines(boxes, w: int, h: int) -> list[str]:
    """[(class, x1, y1, x2, y2)] in pixels -> YOLO 'class cx cy bw bh' lines (normalised)."""
    return [f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            for c, x1, y1, x2, y2 in boxes]


def parse_box(text: str) -> tuple[int, float, float, float, float]:
    """'motorcycle:0.412,0.235,0.479,0.292' (class name: x1,y1,x2,y2 as frame fractions) -> (class id, x1, y1, x2, y2)."""
    name, _, coords = text.partition(":")
    vals = [float(v) for v in coords.split(",")]
    if name not in NAMES or len(vals) != 4 or not 0 <= vals[0] < vals[2] <= 1 or not 0 <= vals[1] < vals[3] <= 1:
        raise argparse.ArgumentTypeError(f"need <{'|'.join(NAMES)}>:x1,y1,x2,y2 with 0 <= x1 < x2 <= 1, got {text!r}")
    return NAMES.index(name), *vals


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def stats(out: Path):
    """Boxes per class and split, frames without vehicles, images without a label file: check before training."""
    for s in ("train", "val"):
        imgs = sorted((out / "images" / s).glob("*.jpg"))
        per_class, empty, missing = [0] * len(NAMES), 0, 0
        for img in imgs:
            lbl = out / "labels" / s / f"{img.stem}.txt"
            if not lbl.exists():
                missing += 1
                continue
            lines = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
            empty += not lines
            for ln in lines:
                per_class[int(ln[0])] += 1
        print(f"{s:5}: {len(imgs)} images, " + ", ".join(f"{n} {c} boxes" for n, c in zip(NAMES, per_class))
              + f", {empty} without vehicles, {missing} without a label file")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "dataset_raw")
    ap.add_argument("--camera", default="Front Parking 02", help="dataset name: datasets/<slug of this>")
    ap.add_argument("--model", default="yolov8x.pt", help="teacher: the biggest COCO model you can wait for")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="teacher threshold: higher = fewer false boxes to delete, more misses to add")
    ap.add_argument("--val-every", type=int, default=5, help="every Nth capture hour goes to val")
    ap.add_argument("--overwrite", action="store_true", help="re-label images that already have a label file")
    ap.add_argument("--add-box", type=parse_box, action="append", default=[], metavar="CLASS:x1,y1,x2,y2",
                    help="draft box added to every newly labelled frame, for a static object the teacher misses "
                         "(the covered bike: motorcycle:0.412,0.235,0.479,0.292). Skipped where the teacher already "
                         "has a box of that class there. Delete it in review where the object is gone or moved.")
    ap.add_argument("--stats", action="store_true", help="only print box counts per class and split, then exit")
    a = ap.parse_args()

    out = ROOT / "datasets" / slug(a.camera)
    if a.stats:
        return stats(out)
    from ultralytics import YOLO  # lazy: --help and --stats work without it
    images = sorted(p.name for p in a.src.glob("*.jpg"))
    if not images:
        sys.exit(f"no .jpg in {a.src}: collect frames first (scripts/collect_dataset.py)")
    existing = {p.name: p.parent.name for p in (out / "images").glob("*/*.jpg")}
    split = {**split_of([n for n in images if n not in existing], a.val_every), **existing}

    model = YOLO(a.model)
    classes = [i for i, n in model.names.items() if n in COCO_TO_OURS]
    counts = {"labelled": 0, "kept": 0, "boxes": 0, "empty": 0}
    for n in images:
        s = split[n]
        img_dst, lbl = out / "images" / s / n, out / "labels" / s / f"{Path(n).stem}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        if not img_dst.exists():
            shutil.copy2(a.src / n, img_dst)
        if lbl.exists() and not a.overwrite:  # reviewed labels are never touched
            counts["kept"] += 1
            continue
        frame = cv2.imread(str(img_dst))
        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, classes=classes, agnostic_nms=True, verbose=False)[0]
        boxes = [(COCO_TO_OURS[model.names[int(c)]], *b, p) for b, c, p in
                 zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        h, w = frame.shape[:2]
        for c, x1, y1, x2, y2 in a.add_box:  # p = None marks an added draft box in the preview
            px = (x1 * w, y1 * h, x2 * w, y2 * h)
            if not any(b[0] == c and box_iou(b[1:5], px) > 0.5 for b in boxes):
                boxes.append((c, *px, None))
        lbl.write_text("\n".join(yolo_lines([b[:5] for b in boxes], w, h)) + ("\n" if boxes else ""))
        counts["labelled"] += 1
        counts["boxes"] += len(boxes)
        counts["empty"] += not boxes
        for c, x1, y1, x2, y2, p in boxes:  # preview: what to check in the review
            colour = (0, 220, 255) if c == 0 else (255, 120, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
            cv2.putText(frame, f"{NAMES[c]} {'added' if p is None else f'{p:.2f}'}", (int(x1), max(int(y1) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, colour, 2, cv2.LINE_AA)
        (out / "preview").mkdir(exist_ok=True)
        cv2.imwrite(str(out / "preview" / n), cv2.resize(frame, (1280, round(h * 1280 / w))))
        print(f"{s:5} {n}: {len(boxes)} vehicle(s)", flush=True)

    (out / "classes.txt").write_text("\n".join(NAMES) + "\n")  # for labelling tools (X-AnyLabeling, LabelImg)
    # no 'path:' key: Ultralytics then resolves train/val relative to this file, so the folder can be copied to
    # another PC (e.g. one with a GPU) and trained there unchanged
    (out / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n"
                                   + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    n_val = sum(v == "val" for v in split.values())
    print(f"\n{out}: {len(images) - n_val} train / {n_val} val images; {counts['labelled']} pre-labelled "
          f"({counts['boxes']} boxes, {counts['empty']} without vehicles), {counts['kept']} existing labels kept.")
    print("Next: review the labels (preview/ shows what the teacher drew), then train, see the steps in --help.")


if __name__ == "__main__":
    main()
