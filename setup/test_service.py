"""The setup service's HTTP contract with fake models (no weights needed). Run at the setup image build:
    python -m pytest -q setup/test_service.py        (from /app, PYTHONPATH=/app/src:/app)
The real models are exercised by `python setup/smoke.py <jpeg>` against a running service (not part of the build)."""
import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from carparking.geometry import CameraModel, to_ground
from setup import service

JPEG = cv2.imencode(".jpg", np.full((108, 192, 3), 90, np.uint8))[1].tobytes()
B64 = base64.b64encode(JPEG).decode()


class FakeModels:
    def __init__(self):
        self.calls = []

    def loaded(self):
        return {"sam2": False, "geocalib": False, "depth": False}

    def segment(self, jpeg, points, box=None):
        self.calls.append(("segment", points, box))
        good, other = np.zeros((108, 192), np.uint8), np.zeros((108, 192), np.uint8)
        good[60:100, 10:180] = 255
        other[0:20, 0:20] = 255
        return [{"mask": good, "score": 0.9}, {"mask": other, "score": 0.4}]

    def geocalib(self, jpegs, camera_model="simple_radial"):
        self.calls.append(("geocalib", len(jpegs), camera_model))
        k1 = 0.0 if camera_model == "pinhole" else -0.2
        cam = CameraModel(f=150.0, cx=96, cy=54, pitch=np.radians(35), roll=0.0, h=6.0, k1=k1)
        self.cam = cam
        up = -np.array([0.0, np.cos(cam.pitch), np.sin(cam.pitch)])
        return {"f": 150.0, "fx": 150.0, "fy": 150.0, "up": up.tolist(), "k1": k1, "k2": 0.0, "model": camera_model,
                "up_spread_deg": 0.0, "n": len(jpegs), "focal_uncertainty": 3.0}

    def depth(self, jpeg):
        cam = self.cam
        uu, vv = np.meshgrid(np.arange(192) + 0.5, np.arange(108) + 0.5)
        g = to_ground(cam, np.c_[uu.ravel(), vv.ravel()])  # through the lens
        cz = (np.c_[g, np.full(len(g), -cam.h)] @ cam.R_cw())[:, 2]
        return np.where(np.isnan(cz), 0, cz).reshape(108, 192).astype(np.float32)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "models", FakeModels())
    return TestClient(service.app)


def test_health_lists_loaded_models_and_pins(client):
    r = client.get("/health").json()
    assert r["ok"] and r["loaded"] == {"sam2": False, "geocalib": False, "depth": False}
    assert set(r["pins"]) == {"sam2", "geocalib", "depth"} and len(r["pins"]["sam2"]["commit"]) == 40
    assert all(len(h) == 64 for p in r["pins"].values() for h in p["files"].values())


def test_segment_returns_masks_best_first(client):
    r = client.post("/segment", json={"image": B64, "clicks": [{"x": 50, "y": 80, "label": 1}]})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["size"] == [192, 108] and [m["score"] for m in got["masks"]] == [0.9, 0.4]
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(got["masks"][0]["png"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (108, 192) and mask[80, 50] == 255 and mask[5, 5] == 0
    assert service.models.calls == [("segment", [(50.0, 80.0, 1)], None)]


def test_segment_and_calibrate_refuse_bad_input(client):
    assert client.post("/segment", json={"image": B64, "clicks": []}).status_code == 422
    assert client.post("/segment", json={"image": base64.b64encode(b"not a jpeg").decode(),
                                         "clicks": [{"x": 1, "y": 1}]}).status_code == 422
    assert client.post("/segment", json={"image": "%%%", "clicks": [{"x": 1, "y": 1}]}).status_code == 422
    assert client.post("/calibrate", json={"images": []}).status_code == 422
    wrong = base64.b64encode(cv2.imencode(".png", np.zeros((10, 10), np.uint8))[1].tobytes()).decode()
    assert client.post("/calibrate", json={"images": [B64], "mask": wrong}).status_code == 422  # mask size differs


def test_calibrate_gives_geocalib_and_the_camera_height_from_depth(client):
    mask = np.zeros((108, 192), np.uint8)
    mask[70:105, 5:185] = 255
    png = base64.b64encode(cv2.imencode(".png", mask)[1].tobytes()).decode()
    r = client.post("/calibrate", json={"images": [B64, B64], "mask": png})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["geocalib"]["f"] == 150.0 and got["geocalib"]["n"] == 2 and len(got["geocalib"]["up"]) == 3
    assert got["geocalib"]["model"] == "simple_radial" and got["geocalib"]["k1"] == -0.2  # a wide lens is the default
    assert got["depth"]["h"] == pytest.approx(6.0, rel=0.03) and "sam2" in got["pins"]
    flat = client.post("/calibrate", json={"images": [B64], "mask": png, "camera_model": "pinhole"}).json()
    assert flat["geocalib"]["k1"] == 0.0 and flat["geocalib"]["model"] == "pinhole"
    assert client.post("/calibrate", json={"images": [B64], "camera_model": "fisheye"}).status_code == 422
    none = client.post("/calibrate", json={"images": [B64]}).json()  # no mask: GeoCalib only
    assert none["depth"] is None and none["geocalib"]["n"] == 1
