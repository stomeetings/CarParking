"""Run the parametric ground-metre layout (carparking/layout.py) on a sample picture and write overlays.

  python scripts/layout_demo.py data/sample/camera1.png --out <dir> --front 20,800,700,830 --back 790,215 --count 9 \
         --pitch 12.5 --height 3.1 --f 2184

Camera: a nominal one from --pitch (deg) --height (m) --f (px; default 1.1 x width), or --topdown (nadir camera, scale from --ref x1,y1,x2,y2,metres).
Points are in PIXELS of the picture as stored (it is scaled to 2560 px wide at most, like the sample test). Writes:
  <name>_layout.png    the lattice bays
  <name>_edited.png    after two hand edits (one moved, one widened and turned)
  <name>_refit.png     the edited bays re-fitted under a slightly different camera (should coincide with _edited)
and prints the bay sizes on the ground and the refit error in pixels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carparking import layout as L  # noqa: E402
from carparking.geometry import CameraModel  # noqa: E402
from carparking.geometry.camera import to_ground  # noqa: E402


def nums(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def camera(a, w: int, h: int):
    if a.topdown:
        x1, y1, x2, y2, m = nums(a.ref)
        ppm = float(np.hypot(x2 - x1, y2 - y1)) / m
        f = 20.0 * ppm  # (nominal 20 m high nadir camera: the scale is exactly the reference's)
        return CameraModel(f=f, cx=w / 2, cy=h / 2, pitch=np.radians(89.5), roll=0.0, h=20.0)
    return CameraModel(f=a.f or 1.1 * w, cx=w / 2, cy=h / 2, pitch=np.radians(a.pitch), roll=np.radians(a.roll), h=a.height)


def draw(img, slots, colour, tag):
    out = img.copy()
    h, w = out.shape[:2]
    for s in slots:
        p = (np.array(s["poly"]) * (w, h)).astype(np.int32)
        cv2.polylines(out, [p], True, colour, 2, cv2.LINE_AA)
        cv2.putText(out, s["name"], tuple(p.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 2400), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, tag, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--front", required=True, help="x1,y1,x2,y2: the row's front edge (pixels)")
    ap.add_argument("--back", required=True, help="x,y: one point on the row's back side (pixels)")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--depth", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=5.0)
    ap.add_argument("--f", type=float, default=None)
    ap.add_argument("--topdown", action="store_true")
    ap.add_argument("--ref", default="0,0,0,0,1")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        print(f"cannot read {a.image}")
        return 1
    h, w = img.shape[:2]
    if w > 2560:
        img = cv2.resize(img, (2560, round(h * 2560 / w)))
        h, w = img.shape[:2]
    cam = camera(a, w, h)
    fx1, fy1, fx2, fy2 = nums(a.front)
    bx, by = nums(a.back)
    lay = L.layout_from_row(cam, [[fx1 / w, fy1 / h], [fx2 / w, fy2 / h]], [[bx / w, by / h]], a.count, a.rows, a.depth, w=w, h=h)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.image).stem
    slots = lay.to_slots(cam, w, h)
    cv2.imwrite(str(out / f"{stem}_layout.png"), draw(img, slots, (0, 255, 255), f"lattice: {len(slots)} bays {lay.width:.2f} x {lay.depth:.2f} m"))
    print(f"lattice: {len(slots)} bays, {lay.width:.2f} x {lay.depth:.2f} m, heading {lay.heading:.1f} deg, flip {lay.flip}")
    # two hand edits, stored in metres
    mid = lay.total // 2
    lay.set_offset(mid, L.BayOffset(du=0.25, dv=0.1))
    lay.set_offset(min(lay.total - 1, mid + 2), L.BayOffset(dw=0.25, rot=3.0))
    edited = lay.to_slots(cam, w, h)
    cv2.imwrite(str(out / f"{stem}_edited.png"), draw(img, edited, (0, 200, 0), f"edited: bays {mid + 1} and {min(lay.total - 1, mid + 2) + 1}"))
    # recalibration: another camera (a few % off), the bays as the user sees them are re-fitted through it
    cam2 = CameraModel(f=cam.f * 1.03, cx=cam.cx, cy=cam.cy, pitch=cam.pitch + np.radians(1.5), roll=cam.roll, h=cam.h * 1.04)
    seen = lay.to_slots(cam, w, h)
    fit = L.refit(seen, cam2, w, h)
    again = fit.to_slots(cam2, w, h)
    err = max(float(np.abs((np.array(x["poly"]) - np.array(y["poly"])) * (w, h)).max()) for x, y in zip(seen, again))
    cv2.imwrite(str(out / f"{stem}_refit.png"), draw(img, again, (0, 0, 255), f"refit under a new camera: max {err:.2f} px from the edited bays"))
    print(f"edits kept as offsets: {sorted(lay.offsets)}; refit under a new camera: {fit.count} x {fit.rows}, {len(fit.offsets)} offset bays, max error {err:.2f} px")
    g = [to_ground(cam, np.array(s['poly']) * (w, h)) for s in slots]
    print("ground bay widths (m):", [round(float(np.linalg.norm(q[1] - q[0])), 2) for q in g])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
