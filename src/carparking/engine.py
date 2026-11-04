"""Occupancy engine: crop each bay from a BGR frame and classify it with the trained ResNet-18.

Preprocessing must match scripts/train_model.py exactly: RGB PIL image -> Resize(128,128) -> ToTensor -> ImageNet norm.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop(frame, poly: np.ndarray):
    """Bounding-rect crop of a bay, clipped to the frame; None if it falls outside."""
    h, w = frame.shape[:2]
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    return frame[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None


class Classifier:
    """bays -> {slot_id: P(busy)} for one frame, all bays in one batch."""

    def __init__(self, weights_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = models.resnet18(weights=None)  # architecture only; trained weights come from the file
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, frame_bgr, bays: dict[int, np.ndarray]) -> dict[int, float | None]:
        ids, batch = [], []
        for slot, poly in bays.items():
            patch = crop(frame_bgr, poly)
            if patch is not None:
                ids.append(slot)
                batch.append(TRANSFORM(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))))
        probs = {slot: None for slot in bays}  # bays outside the frame stay UNKNOWN
        if batch:
            p_busy = torch.softmax(self.model(torch.stack(batch).to(self.device)), dim=1)[:, 1].tolist()
            probs.update(zip(ids, p_busy))
        return probs


def to_states(probs: dict[int, float | None], threshold: float = 0.5) -> dict[int, int | None]:
    """P(busy) -> 0 FREE / 1 BUSY / None UNKNOWN."""
    return {k: None if p is None else int(p >= threshold) for k, p in probs.items()}
