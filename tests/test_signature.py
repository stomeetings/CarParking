"""Appearance signatures (carparking/signature.py) on synthetic car crops: the same car under another exposure or at another scale
matches, another colour or another layout does not, and degenerate crops give no signature rather than a wrong one."""
import cv2
import numpy as np

from carparking import signature as sg


def car(color=(40, 40, 200), w=160, h=90, seed=0):
    """A coloured body with a windscreen, two dark wheels and a little noise: enough layout for the thumbnail."""
    img = np.full((h, w, 3), 110, np.uint8)
    cv2.rectangle(img, (w // 10, h // 5), (w - w // 10, h - h // 8), color, -1)
    cv2.rectangle(img, (w // 4, h // 4), (w - w // 4, h // 2), (30, 30, 30), -1)
    cv2.circle(img, (w // 4, h - h // 8), h // 9, (10, 10, 10), -1)
    cv2.circle(img, (w - w // 4, h - h // 8), h // 9, (10, 10, 10), -1)
    noise = np.random.default_rng(seed).integers(-6, 7, img.shape)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_same_car_other_exposure_and_scale_matches():
    a = sg.make_signature(car())
    dark = np.clip(car(seed=1).astype(np.float32) * 0.7 + 8, 0, 255).astype(np.uint8)
    small = cv2.resize(car(seed=2), (110, 62), interpolation=cv2.INTER_AREA)
    for other in (dark, small):
        b = sg.make_signature(other)
        assert sg.hist_distance(a, b) < sg.MAX_BHATTA
        assert sg.thumb_ncc(a, b) > 0.9
        assert sg.same_vehicle(a, b)


def test_other_colour_is_another_vehicle():
    a, b = sg.make_signature(car((40, 40, 200))), sg.make_signature(car((200, 120, 30)))
    assert sg.hist_distance(a, b) > sg.MAX_BHATTA
    assert not sg.same_vehicle(a, b)


def test_other_layout_is_another_vehicle():
    a = sg.make_signature(car())
    empty = np.full((90, 160, 3), 110, np.uint8)
    flat = sg.make_signature(empty)
    assert sg.thumb_ncc(a, flat) == 0.0  # a flat thumbnail has nothing to correlate
    assert not sg.same_vehicle(a, flat)
    bar = empty.copy()
    cv2.rectangle(bar, (0, 0), (160, 30), (40, 40, 200), -1)  # same colour, stripes instead of a car
    assert sg.thumb_ncc(a, sg.make_signature(bar)) < sg.MIN_NCC


def test_metric_size_must_agree_when_known():
    a, b = sg.make_signature(car(), size_m=4.2), sg.make_signature(car(seed=3), size_m=5.6)
    assert not sg.same_vehicle(a, b)
    assert sg.same_vehicle(a, sg.make_signature(car(seed=3), size_m=4.4))
    assert sg.same_vehicle(a, sg.make_signature(car(seed=3)))  # unknown on one side: no size verdict


def test_degenerate_crops_give_none():
    assert sg.make_signature(None) is None
    assert sg.make_signature(np.zeros((90, 160), np.uint8)) is None  # not colour
    assert sg.make_signature(np.zeros((7, 7, 3), np.uint8)) is None  # smaller than MIN_CROP
    assert not sg.same_vehicle(None, sg.make_signature(car()))


def test_signature_shapes_and_normalisation():
    s = sg.make_signature(car())
    assert s.hist.shape == (16 * 4 * 4,) and abs(float(s.hist.sum()) - 1.0) < 1e-5
    assert s.thumb.shape == (16, 32)
    assert abs(float(np.linalg.norm(s.unit)) - 1.0) < 1e-4 and abs(float(s.unit.mean())) < 1e-4


def test_thumb_diff_is_local_change():
    a = sg.make_signature(car())
    assert sg.thumb_diff(a, a) == 0.0
    assert sg.thumb_diff(a, sg.make_signature(np.full((90, 160, 3), 110, np.uint8))) > 10


def test_crop_box_is_a_clipped_view():
    frame = np.arange(100 * 200, dtype=np.uint8).reshape(100, 200).repeat(3).reshape(100, 200, 3)
    box = [[0.25, 0.2], [0.5, 0.2], [0.5, 0.6], [0.25, 0.6]]
    c = sg.crop_box(frame, box)
    assert c.shape[:2] == (40, 50) and np.shares_memory(c, frame)
    assert sg.crop_box(frame, [[-0.2, -0.2], [0.1, 0.1]], pad=0.5).shape[0] > 0  # never out of range
    assert sg.crop_box(frame, [[1.2, 1.2], [1.4, 1.4]]).size == 0
