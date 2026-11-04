"""Slot evidence (carparking/evidence.py): the EMA filter with its hysteresis band, the parked lock, and the exit veto, on synthetic
observation sequences. The target is the post-demo one: a stationary car logs no transition, a real exit still gets through."""
import cv2
import numpy as np
import pytest

from carparking import evidence as evd
from carparking import signature as sg
from carparking.evidence import CFG, SlotEvidence

BOX = [[0.30, 0.40], [0.50, 0.40], [0.50, 0.60], [0.30, 0.60]]


def shifted(dx, dy=0.0, scale=1.0):
    p = np.float32(BOX)
    c = p.mean(0)
    return ((p - c) * scale + c + [dx, dy]).tolist()


def car(color=(40, 40, 200), seed=0):
    img = np.full((90, 160, 3), 110, np.uint8)
    cv2.rectangle(img, (16, 18), (144, 80), color, -1)
    cv2.rectangle(img, (40, 22), (120, 45), (30, 30, 30), -1)
    cv2.circle(img, (40, 80), 10, (10, 10, 10), -1)
    cv2.circle(img, (120, 80), 10, (10, 10, 10), -1)
    noise = np.random.default_rng(seed).integers(-6, 7, img.shape)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def run(ev, obs, dt=1.0):
    """The filtered state after each observation."""
    return [evd.step(ev, o, 1.0, dt) for o in obs]


def locked_bay(now=100.0, crop=None):
    ev = SlotEvidence()
    evd.step(ev, 0.9, 1.0, 1.0)
    for t in (0.0, 30.0, 60.0, 61.0):
        evd.track_anchor(ev, BOX, now - 100 + t + 40, crop_fn=(lambda: crop) if crop is not None else None)
    assert evd.track_anchor(ev, BOX, now, crop_fn=(lambda: crop) if crop is not None else None) or ev.locked
    return ev


# ---- EMA filter ----------------------------------------------------------------------------------------------------------------

def test_first_observation_seeds_the_filter():
    ev = SlotEvidence()
    assert evd.step(ev, 0.9, 1.0, 1.0) == 1  # no seconds-long climb after a restart


def test_noise_inside_the_band_never_toggles():
    ev = SlotEvidence()
    evd.step(ev, 0.9, 1.0, 1.0)
    rng = np.random.default_rng(0)
    states = run(ev, np.clip(rng.normal(0.6, 0.25, 500), 0, 1))  # a car seen at 0.3-0.9 confidence, some frames missed
    assert set(states) == {1}


def test_stationary_car_survives_missed_frames():
    ev = SlotEvidence()
    evd.step(ev, 0.9, 1.0, 1.0)
    obs = ([0.9] * 8 + [0.0] * 3) * 20  # 3 frames in 11 missed, over and over
    assert set(run(ev, obs)) == {1}


def test_entry_is_quick_and_exit_is_slow():
    ev = SlotEvidence()
    evd.step(ev, 0.0, 1.0, 1.0)
    assert ev.seen == 0
    enter = [evd.step(ev, 0.9, 1.0, 1.0) for _ in range(30)]
    assert enter.index(1) < 5  # ~tau_enter
    leave = [evd.step(ev, 0.0, 1.0, 1.0) for _ in range(120)]
    assert 25 < leave.index(0) < 80  # ~tau_exit 30 s to reach LO from the top


def test_alpha_cap_slow_frames_need_three_misses():
    """The worker can take a minute per frame: alpha is capped at 0.5, so E goes 0.9 -> 0.45 -> 0.225 -> 0.1125 and only the third miss crosses LO."""
    ev = SlotEvidence()
    evd.step(ev, 0.9, 1.0, 1.0)
    out = [evd.step(ev, 0.0, 1.0, 60.0) for _ in range(3)]
    assert out == [1, 1, 0]
    assert ev.e == pytest.approx(0.1125, abs=1e-3)


def test_unjudged_and_invisible_leave_the_state_alone():
    ev = SlotEvidence()
    evd.step(ev, 0.9, 1.0, 1.0)
    e = ev.e
    assert evd.step(ev, None, 1.0, 100.0) == 1
    assert evd.step(ev, 0.0, 0.0, 100.0) == 1
    assert ev.e == e


def test_visibility_scales_the_step():
    a, b = SlotEvidence(), SlotEvidence()
    for ev in (a, b):
        evd.step(ev, 0.9, 1.0, 1.0)
    evd.step(a, 0.0, 1.0, 10.0)
    evd.step(b, 0.0, 0.5, 10.0)
    assert (0.9 - b.e) == pytest.approx((0.9 - a.e) / 2, rel=1e-4)


# ---- parked lock ---------------------------------------------------------------------------------------------------------------

def test_lock_needs_time_and_frames():
    ev = SlotEvidence()
    assert not evd.track_anchor(ev, BOX, 0.0)
    assert not evd.track_anchor(ev, BOX, 30.0)
    assert not ev.locked  # 3 frames but not yet 60 s
    assert evd.track_anchor(ev, BOX, 61.0)
    assert ev.locked and ev.locked_at == 61.0


def test_lock_needs_frames_even_after_the_time():
    ev = SlotEvidence()
    evd.track_anchor(ev, BOX, 0.0)
    assert not evd.track_anchor(ev, BOX, 120.0)  # 2 frames only
    assert evd.track_anchor(ev, BOX, 121.0)


def test_box_jitter_keeps_the_run_and_a_move_restarts_it():
    ev = SlotEvidence()
    evd.track_anchor(ev, BOX, 0.0)
    evd.track_anchor(ev, shifted(0.004, 0.003), 30.0)  # IoU ~0.9
    assert ev.anchor_since == 0.0 and ev.anchor_n == 2
    evd.track_anchor(ev, shifted(0.12), 50.0)  # the car rolled on: a new anchor
    assert ev.anchor_since == 50.0 and ev.anchor_n == 1 and not ev.locked


def test_lock_takes_the_signature_lazily():
    calls = []
    ev = SlotEvidence()
    for t in (0.0, 30.0, 59.0):
        evd.track_anchor(ev, BOX, t, crop_fn=lambda: calls.append(1))
    assert calls == []  # not before the lock
    evd.track_anchor(ev, BOX, 61.0, crop_fn=lambda: car())
    assert ev.sig is not None


def test_locked_anchor_does_not_follow_a_moved_box_until_released():
    ev = SlotEvidence()
    for t in (0.0, 30.0, 61.0):
        evd.track_anchor(ev, BOX, t)
    assert ev.locked
    assert not evd.track_anchor(ev, shifted(0.2), 90.0)
    assert box_same(ev.anchor, BOX)
    evd.release(ev)
    assert not ev.locked and ev.anchor is None and ev.sig is None


def box_same(a, b):
    return np.allclose(np.asarray(a), np.asarray(b))


def test_box_iou():
    assert evd.box_iou(BOX, BOX) == pytest.approx(1.0)
    assert evd.box_iou(BOX, shifted(0.5)) == 0.0
    assert 0.3 < evd.box_iou(BOX, shifted(0.1)) < 0.4


# ---- exit veto -----------------------------------------------------------------------------------------------------------------

def parked(now=100.0):
    ev = SlotEvidence()
    evd.step(ev, 0.9, 1.0, 1.0)
    for t in (0.0, 30.0, 61.0):
        evd.track_anchor(ev, BOX, t, crop_fn=lambda: car())
    assert ev.locked and ev.sig is not None
    return ev


def test_unlocked_bay_is_never_vetoed():
    assert evd.exit_veto(SlotEvidence(), sg.make_signature(car()), 1.0, True) is None


def test_veto_relook_hit():
    assert evd.exit_veto(parked(), None, None, True) == "relook"


def test_veto_appearance_same_car():
    now = sg.make_signature(np.clip(car(seed=5).astype(np.float32) * 0.8, 0, 255).astype(np.uint8))  # dimmer, same car
    assert evd.exit_veto(parked(), now, None, False) == "appearance"


def test_veto_no_motion_when_the_bay_changed_no_more_than_the_frame():
    ev = parked()
    gone = sg.make_signature(np.full((90, 160, 3), 110, np.uint8))  # the car is not there ...
    local = sg.thumb_diff(ev.sig, gone)
    # ... but the whole frame changed as much (a lighting event): that is not a car driving off
    assert evd.exit_veto(ev, gone, local, False) == "no_motion"


def test_no_veto_for_a_real_exit():
    ev = parked()
    gone = sg.make_signature(np.full((90, 160, 3), 110, np.uint8))
    assert evd.exit_veto(ev, gone, 0.5, False) is None  # bay changed a lot, the picture did not
    assert evd.exit_veto(ev, gone, None, None) is None  # no data at all = no veto


def test_other_car_in_the_bay_is_not_vetoed_as_appearance():
    ev = parked()
    other = sg.make_signature(car((200, 120, 30)))
    assert evd.exit_veto(ev, other, 0.5, False) is None


# ---- gate_exit -----------------------------------------------------------------------------------------------------------------

def test_gate_passes_everything_but_a_locked_exit():
    ev = SlotEvidence()
    assert evd.gate_exit(ev, 1, 0.0, probe=lambda: (None, None, True)) == 1
    assert evd.gate_exit(ev, 0, 0.0, probe=lambda: (None, None, True)) == 0  # not locked: nothing to protect
    assert evd.gate_exit(ev, None, 0.0) is None


def test_gate_holds_a_ghost_exit_then_cools_down_without_probing():
    ev = parked()
    calls = []

    def probe():
        calls.append(1)
        return None, None, True

    assert evd.gate_exit(ev, 0, 200.0, probe=probe) == 1
    assert ev.seen == 1 and ev.last_veto == "relook" and ev.vetoes == 1
    assert evd.gate_exit(ev, 0, 210.0, probe=probe) == 1  # cooling down: held, and the zoom is not paid again
    assert len(calls) == 1
    assert evd.gate_exit(ev, 0, 231.0, probe=probe) == 1
    assert len(calls) == 2


def test_gate_lets_a_real_exit_through_and_probes_once():
    ev = parked()
    gone = sg.make_signature(np.full((90, 160, 3), 110, np.uint8))
    assert evd.gate_exit(ev, 0, 200.0, probe=lambda: (gone, 0.5, False)) == 0
    assert ev.vetoes == 0


def test_gate_without_probe_lets_the_exit_through():
    assert evd.gate_exit(parked(), 0, 200.0) == 0


def test_a_veto_does_not_make_a_ghost_immortal():
    ev = parked()
    t = 200.0
    for _ in range(CFG.max_vetoes):
        assert evd.gate_exit(ev, 0, t, probe=lambda: (None, None, True)) == 1
        t += CFG.veto_cooldown_s + 1
    assert evd.gate_exit(ev, 0, t, probe=lambda: (None, None, True)) == 0


def test_a_veto_restores_the_band_so_one_miss_does_not_re_ask():
    ev = parked()
    for _ in range(60):
        evd.step(ev, 0.0, 1.0, 1.0)
    assert ev.seen == 0
    evd.gate_exit(ev, 0, 500.0, probe=lambda: (None, None, True))
    assert ev.seen == 1 and CFG.lo < ev.e <= CFG.hi
    assert evd.step(ev, 0.0, 1.0, 1.0) == 1


def test_end_to_end_stationary_car_with_dropouts_logs_no_exit():
    """A locked car, 30 % of frames missed and a lighting dip: the gated state never reaches 0."""
    ev = parked()
    rng = np.random.default_rng(3)
    t, out = 100.0, []
    for i in range(600):
        o = 0.0 if rng.random() < 0.3 else 0.85
        if 300 <= i < 330:
            o = 0.0  # a 30 s dip: lighting
        s = evd.step(ev, o, 1.0, 1.0)
        out.append(evd.gate_exit(ev, s, t + i, probe=lambda: (sg.make_signature(car(seed=9)), 4.0, False)))
    assert set(out) == {1}
