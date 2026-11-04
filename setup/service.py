"""The setup service (compose profile "setup", not started by default): the heavy models behind a small HTTP API for the
api service's setup jobs. It holds no state, no database access and no camera credentials: it gets a still and returns what
the models say. Reachable only on the compose network (the port is not published).

  GET  /health      loaded models + the pinned versions / hashes
  POST /segment     {image (base64 JPEG), clicks [{x, y, label}] pixels, box?}  -> candidate masks (base64 PNG) + scores
  POST /calibrate   {images [base64 JPEG], mask? (base64 PNG, 255 = ground), depth: bool}
                    -> GeoCalib {f, up, ...} and the camera height from the metric depth of the mask's pixels
"""
from __future__ import annotations

import base64
import os
import threading
import time

import cv2
import numpy as np
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from carparking.geometry.calib import height_from_depth_map
from setup.models import Models, pins

app = FastAPI(title="Parking setup service")
models = Models(os.environ.get("SETUP_WEIGHTS", "/opt/weights"))
lock = threading.Lock()  # the models are CPU-bound and not thread-safe: one request at a time
MAX_IMAGE = 12_000_000  # bytes of one decoded-from-base64 JPEG
MAX_STILLS = 8


def jpeg_of(b64: str) -> bytes:
    try:
        raw = base64.b64decode(b64, validate=True)
    except ValueError:
        raise HTTPException(422, "image: not base64") from None
    if not raw[:2] == b"\xff\xd8" or len(raw) > MAX_IMAGE:
        raise HTTPException(422, "image: a JPEG up to 12 MB")
    return raw


class Click(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: int = Field(1, ge=0, le=1)


class SegmentIn(BaseModel):
    image: str
    clicks: list[Click] = Field(default_factory=list, max_length=20)
    box: list[float] | None = Field(None, min_length=4, max_length=4)


class CalibrateIn(BaseModel):
    images: list[str] = Field(min_length=1, max_length=MAX_STILLS)
    camera_model: Literal["pinhole", "simple_radial", "radial"] = "simple_radial"  # GeoCalib's lens model
    mask: str | None = None
    depth: bool = True


@app.get("/health")
def health():
    return {"ok": True, "loaded": models.loaded(), "pins": pins()}


@app.post("/segment")
def segment(body: SegmentIn):
    if not body.clicks and body.box is None:
        raise HTTPException(422, "give at least one click or a box")
    jpeg, t = jpeg_of(body.image), time.time()
    try:
        with lock:
            found = models.segment(jpeg, [(c.x, c.y, c.label) for c in body.clicks], body.box)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    h, w = found[0]["mask"].shape
    return {"size": [w, h], "seconds": round(time.time() - t, 2),
            "masks": [{"score": round(m["score"], 4),
                       "png": base64.b64encode(cv2.imencode(".png", m["mask"])[1].tobytes()).decode()} for m in found]}


@app.post("/calibrate")
def calibrate(body: CalibrateIn):
    jpegs, t = [jpeg_of(i) for i in body.images], time.time()
    try:
        with lock:
            geo = models.geocalib(jpegs, body.camera_model)
            depth = None
            if body.depth and body.mask:
                mask = cv2.imdecode(np.frombuffer(base64.b64decode(body.mask), np.uint8), cv2.IMREAD_GRAYSCALE)
                dmap = models.depth(jpegs[0])
                if mask is None or mask.shape != dmap.shape:
                    raise ValueError("mask: a PNG of the same size as the image")
                depth = height_from_depth_map(dmap, mask, geo["f"], geo["up"], k1=geo["k1"], k2=geo["k2"])
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"geocalib": geo, "depth": depth, "pins": pins(), "seconds": round(time.time() - t, 2)}
