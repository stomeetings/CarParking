"""The three models of the setup service, each loaded on first use (a request that doesn't need one never pays for it) from
the files setup/fetch_weights.py verified at build. CPU only. One request at a time (the service holds a lock)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

MANIFEST = Path(__file__).with_name("manifest.json")


def pins() -> dict:
    """{model: {"commit", "files": {name: sha256}}} from the manifest: what is installed, for the geometry record."""
    models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    return {k: {"commit": m["code"]["commit"], "files": {f["path"]: f["sha256"] for f in m["files"]}}
            for k, m in models.items()}


def decode(jpeg: bytes) -> np.ndarray:
    """JPEG bytes -> RGB uint8 HxWx3."""
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class Models:
    def __init__(self, weights: str | Path):
        self.root = Path(weights)
        self._sam = self._geo = self._depth = None
        self._set = None  # (sha1 of the image, True): SAM's encoder already ran on it

    def loaded(self) -> dict:
        return {"sam2": self._sam is not None, "geocalib": self._geo is not None, "depth": self._depth is not None}

    # -- SAM 2.1 -------------------------------------------------------------------------------------------------
    def segment(self, jpeg: bytes, points: list[tuple[float, float, int]], box=None) -> list[dict]:
        """Click points (x, y pixels, label 1 = ground, 0 = not ground) and/or a box -> up to three candidate masks
        [{"mask": HxW uint8 0/255, "score": float}], best first."""
        if self._sam is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            m = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]["sam2"]
            net = build_sam2(m["config"], str(self.root / "sam2" / m["files"][0]["path"]), device="cpu")
            self._sam = SAM2ImagePredictor(net)
        rgb = decode(jpeg)
        key = hashlib.sha1(jpeg).hexdigest()
        if self._set != key:  # the image encoder is the slow part: once per image, refinements reuse it
            self._sam.set_image(rgb)
            self._set = key
        kw = {}
        if points:
            kw["point_coords"] = np.array([[x, y] for x, y, _ in points], np.float32)
            kw["point_labels"] = np.array([int(label) for _, _, label in points], np.int32)
        if box is not None:
            kw["box"] = np.array(box, np.float32)
        masks, scores, _ = self._sam.predict(multimask_output=True, **kw)
        order = np.argsort(-scores)
        return [{"mask": (masks[i] > 0).astype(np.uint8) * 255, "score": float(scores[i])} for i in order]

    # -- GeoCalib ------------------------------------------------------------------------------------------------
    def geocalib(self, jpegs: list[bytes], camera_model: str = "simple_radial") -> dict:
        """Stills of one camera -> {"f", "fx", "fy", "up" (camera-frame up vector), "k1", "k2", "model",
        "focal_uncertainty", "n"}: the median focal length and the mean up vector of the stills (a fixed camera: they
        agree; the spread is returned). camera_model: "pinhole" (no distortion), "simple_radial" (k1) or "radial" (k1,
        k2): GeoCalib's own models, whose convention geometry/camera.py implements. A wide CCTV lens needs a radial
        one: on camera-1 the pinhole model gave a 20 deg field of view, simple_radial 52 deg."""
        if camera_model not in ("pinhole", "simple_radial", "radial"):
            raise ValueError(f"camera_model: pinhole, simple_radial or radial, not {camera_model!r}")
        if self._geo is None:
            from geocalib import GeoCalib
            self._geo = GeoCalib(weights="pinhole")  # torch hub cache: the file verified at build, no download
        imgs = [torch.from_numpy(decode(j)).permute(2, 0, 1).float() / 255.0 for j in jpegs]
        if len({tuple(i.shape) for i in imgs}) != 1:
            raise ValueError("the stills differ in size")
        res = self._geo.calibrate(imgs if len(imgs) > 1 else imgs[0], camera_model=camera_model,
                                  shared_intrinsics=len(imgs) > 1)
        cam = res["camera"]
        k1 = 0.0 if camera_model == "pinhole" else float(np.median(cam.k1.reshape(-1).cpu().numpy()))
        k2 = float(np.median(cam.k2.reshape(-1).cpu().numpy())) if camera_model == "radial" else 0.0
        f = cam.f.reshape(-1, 2).cpu().numpy()
        up = res["gravity"].vec3d.reshape(-1, 3).cpu().numpy()
        mean_up = up.mean(axis=0)
        mean_up /= np.linalg.norm(mean_up)
        unc = res.get("focal_uncertainty")
        return {"f": float(np.median(f.mean(axis=1))), "fx": float(np.median(f[:, 0])), "fy": float(np.median(f[:, 1])),
                "up": mean_up.tolist(), "k1": k1, "k2": k2, "model": camera_model, "up_spread_deg": float(np.degrees(np.max(
                    np.arccos(np.clip(up @ mean_up, -1, 1))))), "n": len(imgs),
                "focal_uncertainty": None if unc is None else float(unc.reshape(-1)[0])}

    # -- Depth Anything V2 (metric) ------------------------------------------------------------------------------
    def depth(self, jpeg: bytes) -> np.ndarray:
        """JPEG -> metric depth map HxW float32, metres along the optical axis."""
        if self._depth is None:
            from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
            d = str(self.root / "depth")
            self._depth = (AutoImageProcessor.from_pretrained(d), DepthAnythingForDepthEstimation.from_pretrained(d).eval())
        proc, net = self._depth
        rgb = decode(jpeg)
        with torch.no_grad():
            out = net(**proc(images=rgb, return_tensors="pt")).predicted_depth
        h, w = rgb.shape[:2]
        return torch.nn.functional.interpolate(out[:, None], size=(h, w), mode="bicubic",
                                               align_corners=False)[0, 0].float().numpy()
