"""Offline tracking evaluation (carparking/track_eval.py, scripts/track_eval.py): trackers, metrics, recordings, the CLI. In-memory
and tmp_path only: no model, no network, no ./data."""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carparking.track_eval import (ByteTracker, IouTracker, Recording, UltralyticsByteTracker, associate, decide, evaluate,  # noqa: E402
                                   format_report, from_xyah, hungarian, iou_matrix, load_recording, record_from_api,
                                   run_all, save_recording, steady_counts, synthesize, to_xyah)

spec = importlib.util.spec_from_file_location("track_eval_cli", ROOT / "scripts" / "track_eval.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

CAR = np.array([100.0, 100.0, 200.0, 180.0])


def shifted(box, dx=0.0, dy=0.0):
    return np.asarray(box, float) + [dx, dy, dx, dy]


# --- primitives ---------------------------------------------------------------------------------------------------------------
def test_iou_matrix_and_the_box_conversion():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    m = iou_matrix(a, b)
    assert m.shape == (2, 3) and m[0, 0] == pytest.approx(1.0) and m[0, 1] == pytest.approx(50 / 150) and m[0, 2] == 0
    assert iou_matrix([], b).shape == (0, 3) and iou_matrix(a, np.zeros((0, 4))).shape == (2, 0)
    assert np.allclose(from_xyah(to_xyah(CAR)), CAR)


def test_hungarian_is_the_cheapest_assignment_whatever_the_shape():
    rng = np.random.RandomState(3)
    for n, m in ((3, 3), (2, 5), (5, 2), (4, 4), (1, 3)):
        cost = rng.rand(n, m)
        pairs = hungarian(cost)
        assert len(pairs) == min(n, m) and len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})
        best = min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n)) if n <= m else \
            min(sum(cost[p[j], j] for j in range(m)) for p in itertools.permutations(range(n), m))
        assert sum(cost[i, j] for i, j in pairs) == pytest.approx(best)
    assert hungarian(np.zeros((0, 3))) == []


def test_associate_drops_the_pairs_over_the_limit_and_names_who_is_left():
    cost = np.array([[0.1, 0.9], [0.8, 0.95]])
    matches, rows, cols = associate(cost, 0.5)
    assert matches == [(0, 0)] and rows == [1] and cols == [1]
    assert associate(np.zeros((0, 2)), 0.5) == ([], [], [0, 1])


# --- trackers -----------------------------------------------------------------------------------------------------------------
def ids(out):
    return sorted(i for i, _ in out)


def test_bytetrack_keeps_one_id_for_a_parked_car_and_two_for_two_cars():
    t = ByteTracker()
    seen = [ids(t.update([CAR + (i % 3) * 0.5], [0.9])) for i in range(20)]
    assert all(s == [1] for s in seen)
    t = ByteTracker()
    other = shifted(CAR, 300)
    seen = [ids(t.update([CAR, other], [0.9, 0.8])) for _ in range(10)]
    assert all(s == [1, 2] for s in seen)


def test_bytetrack_a_weak_box_continues_the_track_and_a_gap_does_not_change_the_id():
    t = ByteTracker()
    for _ in range(5):
        t.update([CAR], [0.9])
    assert ids(t.update([CAR], [0.3])) == [1]  # between low and high: it continues the existing track
    assert ids(t.update([], [])) == [] and ids(t.update([], [])) == []  # unseen
    assert ids(t.update([CAR], [0.9])) == [1]  # the same car, the same id
    t = ByteTracker(buffer=3)
    for _ in range(3):
        t.update([CAR], [0.9])
    for _ in range(6):
        t.update([], [])
    assert ids(t.update([CAR], [0.9])) == []  # a new, unconfirmed track after the buffer ran out ...
    assert ids(t.update([CAR], [0.9])) == [2]  # ... confirmed by a second sighting, with a new id


def test_bytetrack_ignores_a_one_frame_false_positive_and_a_weak_box_alone_starts_nothing():
    t = ByteTracker()
    for _ in range(3):
        t.update([CAR], [0.9])
    stray = shifted(CAR, 600, 300)
    assert ids(t.update([CAR, stray], [0.9, 0.9])) == [1]  # (unconfirmed: not output yet)
    assert ids(t.update([CAR], [0.9])) == [1]  # never seen again: gone
    assert ids(t.update([CAR, stray], [0.9, 0.3])) == [1]  # a weak box starts no track
    t = ByteTracker()
    assert ids(t.update([CAR], [0.4])) == [] and ids(t.update([CAR], [0.4])) == []


def test_alive_includes_the_tracks_unseen_this_frame_until_they_expire():
    t = ByteTracker(buffer=3)
    for _ in range(3):
        t.update([CAR], [0.9])
    assert ids(t.update([], [])) == [] and [i for i, _ in t.alive()] == [1]  # not output, still held
    t.update([], [])
    t.update([], [])
    assert [i for i, _ in t.alive()] == [1]
    t.update([], [])
    assert t.alive() == []  # unseen for more than the buffer
    s = IouTracker(max_age=2)
    s.update([CAR], [0.9])
    s.update([], [])
    assert [i for i, _ in s.alive()] == [1]
    s.update([], [])
    s.update([], [])
    assert s.alive() == []


def test_iou_tracker_survives_a_short_gap_and_ignores_weak_boxes():
    t = IouTracker(max_age=3)
    assert ids(t.update([CAR], [0.9])) == [1]
    assert ids(t.update([], [])) == [] and ids(t.update([], [])) == []
    assert ids(t.update([CAR], [0.9])) == [1]
    for _ in range(5):
        t.update([], [])
    assert ids(t.update([CAR], [0.9])) == [2]  # too long: a new id
    assert ids(t.update([shifted(CAR, 400)], [0.1])) == []  # under min_conf


# --- metrics ------------------------------------------------------------------------------------------------------------------
class Scripted:
    """A 'tracker' that returns what it was told, frame by frame (boxes in pixels)."""
    name = "scripted"

    def __init__(self, per_frame):
        self.per_frame, self.i = per_frame, 0

    def update(self, boxes, confs):
        out = self.per_frame[self.i]
        self.i += 1
        return [(i, np.asarray(b, float)) for i, b in out]


W, H = 1000, 500
SLOT = {"id": 1, "name": "P", "poly": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "capacity": 2}
BOX = [0.2, 0.2, 0.3, 0.3]
BOX_PX = [200, 100, 300, 150]


def recording(n=10, dets=None, gt=True, dt=1.0, slots=(SLOT,)):
    frames = []
    for i in range(n):
        d = [{"box": BOX, "conf": 0.9, "label": "car"}] if dets is None or dets(i) else []
        frames.append({"t": i * dt, "dets": d, **({"gt": [{"id": 7, "box": BOX}]} if gt else {})})
    return Recording("cam", (W, H), list(slots), frames)


def test_a_perfect_tracker_scores_perfectly():
    r = evaluate(recording(), Scripted([[(1, BOX_PX)]] * 10))
    t = r["tracking"]
    assert (t["mota"], t["id_switches"], t["fragmentations"], t["false_positives"], t["misses"]) == (1.0, 0, 0, 0, 0)
    assert r["detection"]["precision"] == r["detection"]["recall"] == 1.0
    assert r["slots"]["1"]["tracks"]["mae"] == 0 and r["slots"]["1"]["track_ids"] == {"distinct": 1, "max_concurrent": 1, "vehicles": 1, "per_vehicle": 1.0}
    assert r["decision"]["ids_ok"] and "worth a live change" in r["decision"]["verdict"]


def test_an_id_swap_is_counted_and_scaled_per_vehicle_hour():
    script = [[(1, BOX_PX)]] * 5 + [[(2, BOX_PX)]] * 5
    r = evaluate(recording(dt=36.0), Scripted(script))  # 9 intervals x 36 s = 324 s = 0.09 h
    t = r["tracking"]
    assert t["id_switches"] == 1 and t["vehicle_hours"] == pytest.approx(0.09, abs=1e-3)
    assert t["id_switches_per_vehicle_hour"] == pytest.approx(1 / 0.09, abs=0.05) and t["mota"] == pytest.approx(0.9)
    assert r["tracks"] == {"births": 2, "distinct_ids": 2}
    assert not r["decision"]["ids_ok"] and "keep the steady count" in r["decision"]["verdict"]


def test_a_gap_in_the_detections_is_bridged_switched_or_lost_by_the_id_after_it():
    hole = lambda i: not 3 <= i <= 5  # noqa: E731  (three frames without a detection)
    bridged = evaluate(recording(dets=hole), Scripted([[(1, BOX_PX)]] * 10))["tracking"]
    assert bridged["gaps"] == {"bridged": 1, "switched": 0, "lost": 0}
    switched = evaluate(recording(dets=hole), Scripted([[(1, BOX_PX)]] * 6 + [[(2, BOX_PX)]] * 4))["tracking"]
    assert switched["gaps"] == {"bridged": 0, "switched": 1, "lost": 0} and switched["id_switches"] == 1
    lost = evaluate(recording(dets=hole), Scripted([[(1, BOX_PX)]] * 3 + [[]] * 3 + [[]] + [[(1, BOX_PX)]] * 3))["tracking"]
    assert lost["gaps"] == {"bridged": 0, "switched": 0, "lost": 1} and lost["fragmentations"] == 1


def test_false_positives_and_misses_lower_mota():
    stray = (9, [600, 300, 700, 350])
    r = evaluate(recording(n=4), Scripted([[(1, BOX_PX), stray], [(1, BOX_PX)], [], [(1, BOX_PX)]]))
    t = r["tracking"]
    assert (t["false_positives"], t["misses"], t["id_switches"]) == (1, 1, 0) and t["mota"] == pytest.approx(1 - 2 / 4)


def test_steady_counts_is_the_workers_rule():
    assert steady_counts([0, 0, 2, 2, 2, 0, 0, 0, 0, 0]) == [0, 0, 0, 2, 2, 2, 2, 2, 0, 0]  # in after 2 frames, out after 4
    assert steady_counts([3]) == [1]  # the first frame counts one at most
    assert steady_counts([0, 2, 0, 0, 0, 0]) == [0, 0, 0, 0, 0, 0]  # a one-frame flicker never counts


def test_slot_counts_error_flicker_and_the_ids_that_stood_in_it():
    two = {"gt": [{"id": 1, "box": [0.1, 0.1, 0.2, 0.2]}, {"id": 2, "box": [0.4, 0.1, 0.5, 0.2]}]}
    frames = [{"t": float(i), "dets": [{"box": two["gt"][0]["box"], "conf": 0.9, "label": "car"}] +
               ([] if i in (3, 6) else [{"box": two["gt"][1]["box"], "conf": 0.9, "label": "car"}]), **two} for i in range(10)]
    rec = Recording("cam", (W, H), [SLOT], frames)
    boxes = [[100, 50, 200, 100], [400, 50, 500, 100]]
    script = [[(1, boxes[0]), (2, boxes[1])] for _ in range(10)]  # the tracker holds both through the missed frames
    r = evaluate(rec, Scripted(script))
    s = r["slots"]["1"]
    assert s["raw"]["mae"] == pytest.approx(0.2) and s["raw"]["exact"] == 0.8  # 2 of 10 frames lose a car
    assert s["raw"]["changes_per_hour"] > s["steady"]["changes_per_hour"] > s["tracks"]["changes_per_hour"] == 0
    assert s["tracks"]["mae"] == 0 and s["steady"]["mae"] == pytest.approx(0.1)  # (steady: one vehicle at most on the first frame)
    assert s["track_ids"] == {"distinct": 2, "max_concurrent": 2, "vehicles": 2, "per_vehicle": 1.0}
    assert r["tracking"]["gaps"]["bridged"] == 2
    assert s["held"] == s["tracks"]  # (a tracker without alive(): the held tracks are the ones it returned)
    # capacity caps every source
    one = {**SLOT, "capacity": 1}
    capped = evaluate(Recording("cam", (W, H), [one], frames), Scripted(script))
    assert capped["slots"]["1"]["tracks"]["max"] == 1 and capped["slots"]["1"]["raw"]["max"] == 1


def test_without_ground_truth_it_reports_the_proxies_and_does_not_decide():
    r = evaluate(recording(gt=False), Scripted([[(1, BOX_PX)]] * 10))
    assert not r["ground_truth"] and "tracking" not in r and "detection" not in r
    assert "mae" not in r["slots"]["1"]["raw"] and r["slots"]["1"]["track_ids"]["vehicles"] is None
    assert r["slots"]["1"]["track_ids"]["per_vehicle"] == 1.0 and "undetermined" in r["decision"]["verdict"]
    assert "undetermined" in decide({"ground_truth": False})["verdict"]


# --- the synthetic scene ------------------------------------------------------------------------------------------------------
def test_the_synthetic_scene_is_reproducible_and_has_the_hard_cases():
    a, b, c = synthesize(0, 5.0), synthesize(0, 5.0), synthesize(1, 5.0)
    assert a.frames == b.frames and a.frames != c.frames
    assert a.has_truth and a.size == (1920, 1080) and [s["capacity"] for s in a.slots] == [3, 1]
    gaps = np.diff([f["t"] for f in a.frames])
    assert 0.8 < gaps.min() and gaps.max() < 1.2 and len(a.frames) > 250  # an uneven rate, about 1 frame a second
    assert max(len(f["gt"]) for f in a.frames) == 4  # 3 in the pin + 1 in the bay
    assert any(len(f["dets"]) > len(f["gt"]) for f in a.frames)  # false positives and twin boxes
    assert any(len(f["dets"]) < len(f["gt"]) for f in a.frames)  # missed boxes


def test_trackers_on_the_synthetic_scene_hold_ids_and_the_report_is_complete():
    rec = synthesize(0, 10.0)
    byte, simple = run_all(rec, ("bytetrack", "iou"))
    for r in (byte, simple):
        assert r["ground_truth"] and r["frames"] == len(rec.frames) and set(r["slots"]) == {"9", "2"}
        assert r["detection"]["recall"] > 0.8 and r["tracking"]["gt_vehicles"] == 5
        assert r["slots"]["9"]["track_ids"]["vehicles"] == 3 and r["slots"]["2"]["track_ids"]["vehicles"] == 2
    assert byte["tracking"]["mota"] > 0.8 and byte["tracking"]["id_switches"] <= 6
    assert byte["tracking"]["gaps"]["bridged"] >= 1  # the 5-frame dropout of car 2 is bridged
    assert byte["tracks"]["distinct_ids"] < 25  # (stray boxes never become tracks: they last one frame)
    # production's steady count is already steadier than the raw boxes; the tracks must not be worse than it for the verdict
    p1 = byte["slots"]["9"]
    assert p1["raw"]["changes_per_hour"] > p1["steady"]["changes_per_hour"]
    assert p1["held"]["changes_per_hour"] < p1["tracks"]["changes_per_hour"] / 5  # holding unseen tracks is what steadies a count
    assert p1["held"]["mae"] < p1["tracks"]["mae"] and byte["slots"]["2"]["held"]["exact"] > 0.9
    assert simple["tracking"]["id_switches"] > 10 * max(byte["tracking"]["id_switches"], 1)  # the baseline loses its ids; ByteTrack does not
    assert byte["decision"]["verdict"] and "id_switches_per_vehicle_hour" in byte["decision"]
    assert json.loads(json.dumps(byte)) == byte  # plain JSON
    text = format_report([byte, simple])
    assert "== bytetrack" in text and "== iou" in text and "slot 9 P1 (capacity 3)" in text and "verdict:" in text
    assert "held    changes/h" in text


def test_the_evaluation_is_deterministic():
    rec = synthesize(2, 4.0)
    assert evaluate(rec, ByteTracker()) == evaluate(rec, ByteTracker())


# --- recordings and recording from the API ------------------------------------------------------------------------------------
def test_a_recording_round_trips_and_a_bad_file_is_refused(tmp_path):
    rec = synthesize(0, 1.0)
    path = tmp_path / "rec.json"
    save_recording(rec, path)
    back = load_recording(path)
    assert back.frames == rec.frames and back.size == rec.size and back.slots == rec.slots
    (tmp_path / "bad.json").write_text(json.dumps({"version": 2, "frames": []}))
    with pytest.raises(ValueError, match="version 1"):
        load_recording(tmp_path / "bad.json")


def status(ts, dets=(), connected=True):
    return {"connected": connected, "snapshot": {
        "shape": [2560, 1440], "frame_ts": ts,
        "slots": [{"id": 9, "name": "P1", "poly": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.4], [0.1, 0.4]], "capacity": 3, "state": "BUSY"}],
        "detections": list(dets)}}


def test_record_from_api_takes_one_frame_per_new_frame_time_and_only_vehicles():
    car = {"label": "car", "conf": 0.9, "src": "", "kind": "vehicle", "box": [0.2, 0.2, 0.3, 0.3]}
    person = {"label": "person", "conf": 0.8, "src": "", "kind": "other", "box": [0.6, 0.6, 0.65, 0.7]}
    answers = iter([status(10.0, [car, person]), status(10.0, [car]), status(11.2, [car, car]), {"snapshot": {}}, status(12.0, [])])
    now = [0.0]

    def fetch():
        return next(answers)

    def sleep(s):
        now[0] += s

    rec = record_from_api(fetch, 2.5, poll=0.5, clock=lambda: now[0], sleep=sleep, camera="camera-1")
    assert [f["t"] for f in rec.frames] == [10.0, 11.2, 12.0]  # the repeated frame_ts and the empty snapshot are skipped
    assert [len(f["dets"]) for f in rec.frames] == [1, 2, 0] and rec.frames[0]["dets"][0] == {"box": car["box"], "conf": 0.9, "label": "car"}
    assert rec.size == (2560, 1440) and rec.slots == [{"id": 9, "name": "P1", "poly": status(0)["snapshot"]["slots"][0]["poly"], "capacity": 3}]
    assert rec.camera == "camera-1" and not rec.has_truth
    with pytest.raises(RuntimeError, match="no frame"):
        record_from_api(lambda: {"snapshot": {}}, 1.0, poll=0.5, clock=lambda: now[0], sleep=sleep)


# --- the ultralytics adapter and the CLI --------------------------------------------------------------------------------------
def test_the_ultralytics_tracker_reports_when_it_cannot_run_and_never_installs_anything(monkeypatch):
    ok, why = UltralyticsByteTracker.availability()
    assert isinstance(ok, bool) and (ok or why)
    assert __import__("os").environ.get("YOLO_AUTOINSTALL") == "False" or not ok
    monkeypatch.setattr(UltralyticsByteTracker, "_load", staticmethod(lambda: (_ for _ in ()).throw(ImportError("No module named 'lap'"))))
    assert UltralyticsByteTracker.availability() == (False, "ImportError: No module named 'lap'")
    out = run_all(recording(), ("ultralytics",))
    assert out == [{"tracker": "ultralytics-bytetrack", "unavailable": "ImportError: No module named 'lap'"}]
    assert "not available here" in format_report(out)


@pytest.mark.skipif(not UltralyticsByteTracker.availability()[0], reason="needs lap and scipy (not in the runtime image)")
def test_the_ultralytics_tracker_gives_a_parked_car_one_id():
    t = UltralyticsByteTracker()
    seen = [ids(t.update([CAR], [0.9])) for _ in range(6)]
    assert all(s == [1] for s in seen)


def test_cli_selftest_prints_both_trackers_and_a_verdict():
    lines = []
    assert cli.main(["selftest", "--minutes", "3"], out=lines.append) == 0
    text = "\n".join(lines)
    assert "synthetic scene" in text and "== bytetrack" in text and "== iou" in text and "verdict:" in text


def test_cli_record_then_eval_with_a_json_report(tmp_path, monkeypatch):
    answers = [status(float(i), [{"label": "car", "conf": 0.9, "src": "", "kind": "vehicle", "box": [0.2, 0.2, 0.3, 0.3]}]) for i in range(12)]
    it = iter(answers)
    now = [0.0]
    secret = "s3cr3t-token-value"
    monkeypatch.setenv("CARPARK_VIEW_TOKEN", secret)
    lines = []
    path = tmp_path / "rec.json"
    code = cli.main(["record", "--cam", "camera-1", "--minutes", "0.1", "--poll", "0.5", "--out", str(path)], fetch=lambda: next(it),
                    out=lines.append, sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0])
    assert code == 0 and "recorded 12 frames" in lines[0] and secret not in "\n".join(lines)
    lines.clear()
    report = tmp_path / "report.json"
    assert cli.main(["eval", str(path), "--tracker", "bytetrack", "--json", str(report)], out=lines.append) == 0
    saved = json.loads(report.read_text())
    assert len(saved) == 1 and saved[0]["tracker"] == "bytetrack" and saved[0]["frames"] == 12 and not saved[0]["ground_truth"]
    assert "== bytetrack: 12 frames" in "\n".join(lines) and "undetermined" in "\n".join(lines)
    with pytest.raises(SystemExit):
        cli.main(["eval", str(path), "--tracker", "nope"])


def test_the_api_fetch_sends_the_token_only_as_a_header(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return b'{"snapshot": {}}'

    def urlopen(req, timeout=0):
        seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
        return Resp()

    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    assert cli.api_fetch("http://host:8080/", "camera-1", "tok")() == {"snapshot": {}}
    assert seen == {"url": "http://host:8080/api/status?cam=camera-1", "auth": "Bearer tok"}
    assert "tok" not in seen["url"]
