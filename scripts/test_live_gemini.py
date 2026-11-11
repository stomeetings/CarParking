"""Live check of the vision agent against the real Gemini API: one picture, one call (a retry if the first answer is rejected).

    GEMINI_API_KEY=... python scripts/test_live_gemini.py [data/sample/freecarissue.png] [--model gemini-2.5-flash] [--pitch 40 --height 6 --focal 0.9]

The key is read from the GEMINI_API_KEY environment variable only (never a file, never an argument, never printed). The picture leaves the machine.
It prints the model's raw lines (to verify the JSON schema against the real service), then the derived parameters or why the proposal was rejected.
The camera is a rough guess (pitch / height / focal x width): a wrong guess makes the metric checks fail, the raw reply is still the schema test.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carparking.geometry.camera import CameraModel  # noqa: E402
from carparking.vision_agent.gemini import DEFAULT_MODEL, KEY_ENV, GeminiVisionAgent, VisionProviderError  # noqa: E402
from carparking.vision_agent.validate import run_calibration  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default="data/sample/freecarissue.png")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for the model")
    ap.add_argument("--pitch", type=float, default=40.0, help="camera pitch below the horizon, degrees (a guess)")
    ap.add_argument("--height", type=float, default=6.0, help="camera height in metres (a guess)")
    ap.add_argument("--focal", type=float, default=0.9, help="focal length as a multiple of the picture width (a guess)")
    ap.add_argument("--max-side", type=int, default=1280, help="the picture is scaled so its long side is at most this")
    a = ap.parse_args()
    if not os.environ.get(KEY_ENV):
        print(f"set {KEY_ENV} in the environment first", file=sys.stderr)
        return 2
    img = cv2.imread(a.image)
    if img is None:
        print(f"cannot read {a.image}", file=sys.stderr)
        return 2
    h, w = img.shape[:2]
    scale = min(1.0, a.max_side / max(h, w))
    sent = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
    cam = CameraModel(f=a.focal * w, cx=w / 2, cy=h / 2, pitch=np.radians(a.pitch), roll=0.0, h=a.height)
    print(f"picture {a.image} {w}x{h} (sent {sent.shape[1]}x{sent.shape[0]}), model {a.model}")
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "crop.png")
        cv2.imwrite(path, sent)
        agent = GeminiVisionAgent(model=a.model, timeout_s=a.timeout)
        print(f"agent: {agent!r}")
        try:
            raw = agent.propose(path, {"rows": 1})
        except VisionProviderError as e:
            print(f"RAW CALL FAILED ({type(e).__name__}, retryable={e.retryable}): {e}")
            return 1
        print(f"RAW: {len(raw.lines)} lines, estimated_count={raw.estimated_count}, confidence={raw.confidence}, notes={raw.notes!r}")
        for i, ln in enumerate(raw.lines):
            print(f"  line {i}: front={tuple(round(v, 3) for v in ln.front)} back={tuple(round(v, 3) for v in ln.back)} {ln.visible.value}")
        out = run_calibration(agent, path, {"rows": 1}, cam, w, h)
    for att in out.attempts:
        print(f"attempt {att.number}: {'ok' if not att.errors else 'REJECTED: ' + '; '.join(att.errors)}")
        for t in att.warnings:
            print(f"  warning: {t}")
    if out.ok:
        print("PARAMETERS:", out.params.model_dump())
        print("INFO:", {k: v for k, v in out.info.items() if k not in ("front_dots", "skipped", "warnings")})
        return 0
    print("NOT ACCEPTED:", out.reason)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
