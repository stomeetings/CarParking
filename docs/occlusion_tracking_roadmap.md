# Occlusion tracking & hidden-object management: design and roadmap

Status: **planned, nothing implemented yet.** No code or live file is changed before the 2026-10-29 demo. Written 2026-10-28.
Stack: Python workers (OpenCV/numpy) + FastAPI + React. Companion plan: `docs/camera_drift_roadmap.md`.

## 1. Problem

Vehicles partly hidden by pillars, roof edges or other cars get fluctuating detection confidence. Slots can flicker BUSY/FREE, which
produces false exit/entry pairs in history. This plan fills the real gaps and gates every live change on measured evidence, as the
item 104 plan required.

## 2. Already in the code (reuse, do not rebuild)

| Need | Existing | File |
|---|---|---|
| Debounce | a state changes only after `hold` s; BUSY->FREE needs a longer `exit_hold` of *visible* emptiness | `history.py` `SlotTracker` |
| Occlusion freeze | slot covered >= `occlusion_cover` (0.6) by vehicles in front keeps its state; FREE countdown paused; `hidden_max` 30 s cap; exit flagged `occluded` | `history.py` `_hidden`, `update()` |
| Never green under a car | UNKNOWN shown (not FREE) when hidden / covered / entry pending | `SlotTracker.shown`, `unseen` |
| Partial visibility | ground-footprint matching (`footprint_poly`, `match_score`, `INSIDE_SHARE`), not raw box IoU | `fixed_slots.py`, `history.py` |
| Re-look at weak slots | zoom-crop re-detection at lower confidence for hidden / BUSY-without-box slots | `monitor.py` `_zoom` |
| Capacity smoothing | vehicle count judged over `CAPACITY_FRAMES` | `fixed_slots.py` `_count_loads` |
| Session repair | history kept equal to the live BUSY set every frame | item 96, `monitor.py` |
| Bad stream | stale/frozen picture = every slot UNKNOWN, tracker keeps state | `stream_health.py` (item 105) |
| Tracking evaluation | numpy ByteTrack + IoU tracker, recording/replay harness, `decide()` rule | `track_eval.py`, `scripts/track_eval.py` (item 104) |

## 3. Real gaps

1. **Pillars and roofs are invisible to the occlusion logic.** `_hidden` only knows vehicles in front, so a static occluder makes a slot look emptier than it is.
2. **Per-frame evidence is binary.** A frame is "seen" or "not seen": confidence 0.31 and 0.9 count the same, and one missed frame restarts the exit count. There is no confidence accumulation.
3. **No identity across gaps.** A car that vanishes behind a pillar and reappears is a new observation; only the hold/exit timers cover it. ByteTrack exists offline only, and its verdict is still "undetermined" because there is no real recording.
4. **No flicker metric** to prove a change helps (short sessions, toggles per slot-hour).

## 4. Design

### 4.1 Static occluder masks (pillars, roofs)
- Per camera, polygons for pillars / roof edge, drawn in the slot editor and stored next to the slot map (additive JSON key in `fixed_slots.json`, and for `auto_grid` pins).
- Used by the visibility math below. No new SQL migration.

### 4.2 Visibility-aware overlap
- Slot **visible footprint** `V = footprint - static_mask - fronts(vehicles)`, with visibility `v = area(V) / area(footprint)`.
- **Coverage normalised by what can be seen:** `c = area(box_footprint ∩ V) / area(V)`, used when `v < 0.7`. A car 35 % visible behind a pillar then scores its share of the *visible* slot, not 0.35 of the whole slot.
- Below `v_min` (0.25) the slot is not judged: it keeps its state and pauses its countdown, exactly like today's occlusion freeze, which this generalises.

### 4.3 Evidence filter + hysteresis
Per slot, ahead of the existing `hold` (the downstream machine is unchanged):
```
o_t   = max over matched boxes (conf_b * c_b)        # 0 if visibly empty; None if v < v_min
alpha = 1 - exp(-dt / tau)                           # time-based, safe at uneven frame rates; tau_enter ~3 s, tau_exit ~8 s
a_t   = alpha * v                                    # a half-hidden slot updates half as fast
E_t   = E_{t-1} + a_t * (o_t - E_{t-1})              # skipped when o_t is None
seen  = BUSY if E_t > 0.55, FREE if E_t < 0.20, else previous   # hysteresis band: no toggling at the edge
```
`seen` feeds the existing `hold` / `exit_hold` machine. Live option `evidence_filter` defaults to `0` (off), so behaviour is unchanged until switched on. Thresholds are exposed like `DISPLAY_TUNING`.

> **Built in `docs/post_demo_engineering_plan.md` (item 4, step 1):** `src/carparking/evidence.py` (`step`, asymmetric `tau_enter` 3 s / `tau_exit` 30 s, `alpha_max` 0.5, plus the parked lock and the exit veto) wired into `FixedSlotTracker` behind `evidence_filter` (default 0). The `tau_exit` of 30 s here replaces the ~8 s sketched above, because a parked car must not leave on a short gap. See "Step 1 as built" in that document.

### 4.4 ByteTrack persistence
- ByteTrack (already in numpy: `track_eval.ByteTracker`; Kalman, two-stage association, low-confidence boxes continue tracks) runs in the worker on the merged detections.
- A slot binds to the track that entered it. While that track is *lost but buffered* (<= 30 frames / `track_hold` s) and its Kalman-predicted footprint is still in the slot, the slot stays BUSY with no exit countdown.
- Exit needs either (a) the track seen leaving (centroid leaves the footprint) or (b) the slot visibly empty for `exit_hold`.
- **Gate:** it goes live only if `decide()` on a real camera-1 recording says it beats the steady count (< 1 ID switch per vehicle-hour; flicker and error no worse). Live option `tracking` defaults to `0`.

### 4.5 Session reopen via lightweight appearance signatures
- At a confirmed entry, store a signature with the open session: HSV histogram of the car crop (16x4x4 bins, L1-normalised), footprint size in metres when calibrated, and the label.
- When a slot that went FREE gets a car again within `reid_window` (120 s), compare signatures (Bhattacharyya < 0.35 and size within 20 %). A match **reopens the same session**: the pending exit is cancelled, so no exit/entry pair is logged.
- Also applies after a stream drop or worker restart, using the stored signature.
- No deep Re-ID network: tiny CPU cost, signatures computed only at entry and re-appearance.

> **Signature half built in `docs/post_demo_engineering_plan.md` (item 4, step 1):** `src/carparking/signature.py` (HSV 16x4x4 histogram with exposure normalisation + 32x16 grey thumbnail; Bhattacharyya and NCC; `same_vehicle`). Step 1 uses it to veto ghost exits of a locked bay. The session **reopen** itself (cancel a logged exit/entry pair within `reid_window`, step L5 there) is not built yet.

### 4.6 Performance
- Tracker: Hungarian assignment on ~20 boxes is < 1 ms (`track_eval` has scipy-free assignment).
- Static mask rasterised once per map change at working resolution; per-slot intersections run only for slots near vehicles.
- Filter is O(slots). Signatures only on entry / re-appearance events.
- No new model, no extra inference; zoom budget unchanged.
- Each step adds per-stage `timing` (ms) to the snapshot, with a test asserting the stage stays under a budget on a synthetic 20-car frame. FPS before/after is checked read-only on the live worker.

## 5. Build order (start after the demo)

Each step: snapshot first, tests, user-approved deploy.

0. **Measure first (read-only, prerequisite):**
   - `scripts/track_eval.py record` for ~30 min on camera-1, plus a short hand-labelled ground-truth slice.
   - New `scripts/flicker_report.py`: from history, sessions < 2 min and toggles per slot-hour, per slot.
   - These are the baseline numbers every later step is compared against.
1. **Static occluder masks:** editor polygons + store + visibility in `_hidden`. Helps pillars immediately.
2. **Evidence filter + hysteresis** (option off by default). Extend the `track_eval` replay to replay slot states, then compare flicker and error against step 0 before switching it on for camera-1.
3. **Live ByteTrack** behind the `tracking` option, only if the gate passes on the step-0 recording.
4. **Signature memory / session reopen.**
5. **Flicker metric on the dashboard** (toggles per slot-hour, short sessions) so regressions stay visible.

## 6. Critical files

- `src/carparking/history.py` (`SlotTracker.update`, `_hidden`, filter hook)
- `src/carparking/fixed_slots.py` (occluder-mask validation + store, `FixedSlotTracker`)
- `src/carparking/monitor.py` (tracker wiring, signatures at entry)
- `src/carparking/track_eval.py` (reuse `ByteTracker`; slot-state replay)
- `src/carparking/live.py` (options `evidence_filter`, `tracking`, timing)
- `frontend/src/components/SlotEditor.tsx` (occluder layer)
- new `scripts/flicker_report.py`
- Untouched: `app.py` / `scripts/demo_app.py` (Streamlit backup; parity test stays green).

## 7. Verification and rules

- Unit tests per step:
  - visibility math on synthetic polygons
  - filter: no toggle under confidence noise inside the band; enters within ~tau; frozen when `v < v_min`
  - tracker bridges a 20-frame pillar gap without an exit
  - signature reopen vs a different car
  - timing budget
- Replay: the step-0 recording through old vs new pipelines. The new one must show fewer toggles per slot-hour and error no worse.
- Full `docker compose up -d --build` (runs all tests), only with the user's approval.
- Live read-only check after deploy: FPS / frame age unchanged; short sessions in history drop.
- Take a snapshot before each step (`& .\scripts\snapshot.ps1 -Label occl-step<N>-pre`); roll back with `scripts\restore.ps1`.
- Additive JSON only (no new SQL migrations). Tests never touch `./data` or the live DB.
- Log every step in `ACTIVITY_LOG.md`.
- **No live code or production file is modified before the 2026-10-29 demo.**

## 8. Boundary with Night Vision

Low-light handling is a separate module: `docs/night_vision_roadmap.md`.
- **Moved out of this plan:** confidence drops caused by darkness or contrast. Lighting is not occlusion.
- **Stays here:** physical overlap, scaffolding / pillar / pole obstructions, track ID persistence, session reopen.
- **Only interface:** a scalar `light` state / quality on the frame result. The evidence filter (4.3) may scale its update weight by it; trackers never see pixels.
