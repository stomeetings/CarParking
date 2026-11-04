# Post-demo engineering plan: spillage, obstacles, distant bays, ghost transitions

Status: **approved 2026-10-29. Step 1 (item 4, ghost transitions) is wired but OFF by default** (see "Step 1 as built" at the end); the
other items are planned, nothing deployed. Stack: Python workers (OpenCV/numpy) + FastAPI + React. Companion plans:
`docs/occlusion_tracking_roadmap.md` (evidence filter 4.3 and signature reopen 4.5 are built here), `docs/camera_drift_roadmap.md`,
`docs/night_vision_roadmap.md`, `docs/PHASE2_GEOMETRY_PLAN.md`.

## Context

The 2026-10-29 demo is done. The evaluators raised four issues: big vehicles covering two or three bays, non-vehicle obstacles, distant bays the setup cannot measure, and stationary cars that flicker BUSY/FREE and log false exit/entry pairs.

What the code does today:
- **Fixed / auto-grid maps:**
  - `FixedSlotTracker._match` (`src/carparking/fixed_slots.py:402`) assigns vehicles to bays greedily, one vehicle to one bay.
  - It turns off the learned-map helpers `_claims`, `share_cover` and `unseen` (`fixed_slots.py:352`, `history.py:549,625`).
  - So the second bay under a van gets `o=0` and goes **FREE**.
- **States:** only `0` FREE, `1` BUSY and `None` UNKNOWN exist (`monitor.py:24` `STATE_NAMES`).
- **Non-vehicle classes:** they already flow through `set_targets` → `others` (`monitor.py:233-240`), but they are only drawn.
  - Zoom (`detect_region`) and SAHI (`_pass`) ckinds=moves, notices or all `_det` without `keep`, so they drop those classes (`yolo_engine.py:218,248`).
- **Open-vocabulary detectors:** none are used anywhere.
- **Far-field geometry:** the pipeline is homography + BEV raster + `pack_bays` lattice (`geometry/pipeline.py`, `geometry/capacity.py`).
  - Cars beyond `CAR_MAX_RES` 0.06 m/px (~18 m on camera 1) are not measured ("cars_too_far").
  - The runtime adds only `DEFAULT_EXTRA=1` extrapolated bay per lane (`geometry_live.py:47-50`).
- **Worker:** one single-threaded process per camera, using 5-60 s per frame on CPU (`live.py:675-763`). It records no per-stage timing.
- **Ghost transitions:**
  - `docs/occlusion_tracking_roadmap.md` already plans an EMA evidence filter (§4.3) and reopening a session from an appearance signature (§4.5).
  - Item 4 below builds on those sections; it does not duplicate them.

**How this document is used:** the four designs below, in the same style as the existing roadmaps. Each build step gets its own snapshot,
tests and an approved deploy; every live change sits behind an option that is off by default.

---

## Cross-cutting rules (apply to kinds=moves, notices or all four)

- **Additive only:**
  - New slot-level information goes in a new `avail` field next to `state`. `state` stays 0/1/None, so history, `sync_open` and the Streamlit parity test are unaffected.
  - New JSON keys go into `fixed_slots.json` and the geometry records. No SQL migrations; new event kinds use the existing events table.
- **Every feature has a live option that defaults to off:**
  - `spillage`, `obstacles`, `evidence_filter` (already planned), `parked_lock`, `far_extrude`.
  - Each is exposed like `DISPLAY_TUNING` (`history.py:116` `tuning_values`).
- **Hot-path budget:**
  - Per-frame additions must be O(slots) numpy on vectors the worker already has.
  - First add a `timing` dict to the snapshot (ms per stage: detect / near / zoom / track / new stages). Then a test asserts each new stage stays under 5 ms on a synthetic 20-car × 30-bay frame.
  - The heavy parts never run in the worker: open-vocabulary detection, tiled SAM, line detection, extrusion fit.
- **Frame-rate awareness:**
  - The worker can run at 5-60 s per frame, so every temporal filter is defined in both **seconds and frames**: `max(t_sec, n_frames)`.
  - The rate `alpha = 1-exp(-dt/tau)` is capped at `alpha_max = 0.5`, so a single slow frame can never flip the filter.
- **Shared step 0 with the occlusion roadmap:**
  - A camera-1 recording (`scripts/track_eval.py record`).
  - A new `scripts/flicker_report.py`, measuring sessions < 2 min and toggles per slot-hour.
  - Every item is judged against this baseline on replay.

---

## 1. Multi-slot encroachment and spillage

### Rule
For each vehicle `v` and bay `s` (on fixed and auto-grid maps):
- `own(s)` = the vehicle assigned by the existing greedy `_match` (unchanged, still one-to-one).
- `cover(v,s)` = the share of the bay's **ground** area under `footprint_poly(v)`. This is the "share of slot under footprint" side of `ground_on`, `fixed_slots.py:328`. Using ground area, not the image box, stops a tkinds=moves, notices or all van *in front of* a bay from counting as standing *on* it.
- A bay with no owner gets `avail = ENCROACHED` when either:
  - `max_v cover(v,s) >= ENCROACH_SHARE` (0.30), or
  - `sum_v cover(v,s) >= ENCROACH_SUM` (0.45): two cars each half a metre over the line.
- **Metric override (calibrated grids):** when the geometry version has `H` and HIGH confidence, the rule is metric instead.
  - Project the footprints to BEV and measure the bay's remaining free width.
  - If `free_width < MIN_BAY_WIDTH` (2.3 m) or `free_length < 4.5 m`, the bay is ENCROACHED, whatever the share.
  - This is the physickinds=moves, notices or ally correct test; the share is the fallback for uncalibrated Fixed maps.
- **Owned bays stay BUSY.** If a neighbour's vehicle also covers an owned bay, it only gets a note ("crowded by A3"). The owner always wins.
- **Capacity slots** (P1 type, `_count_loads`) in v1 are left alone, with a note. In v2 they get "usable length": `capacity_eff = floor(free_length / bay_length)`.

### Debounce and history
- `avail` has its own smkinds=moves, notices or all state machine, `AvailTracker`:
  - Enter after `max(hold, 2 frames)`.
  - Leave after `max(exit_hold, 3 frames)`.
- It **never opens a parking session**: an encroached bay holds no car of its own.
- It logs `encroached` / `encroach_cleared` events with the source vehicle's bay id in `note`, for audit and analytics ("bays lost to bad parking per day").
- If the owning vehicle's bay changes (re-match), the encroachment is re-derived on the next frame. The debounce stops flapping.

### Capacity
In `ParkingMonitor.state()` (`monitor.py:432`):
- `free = slots - busy - encroached - blocked - unknown`.
- A new `counts.encroached` is added, and `counts.available` = the drivable free spaces.

### Component
New module `src/carparking/spillage.py`, pure functions:
```python
@dataclass(frozen=True)
class Spill:            # one per affected bay
    slot: int
    by: tuple[int, ...]  # bay ids whose vehicles spill in
    cover: float         # max share (or sum)
    free_m: float | None # metric free width when calibrated

def spill_map(slot_polys, owner: dict[int,int], fps: list[np.ndarray],
              bev=None, cfg=SpillCfg()) -> dict[int, Spill]
```
- **Pre-filter:** an AABB overlap test on the vehicle×bay pairs, so only adjacent pairs are rasterised. That is about 2-3 pairs per vehicle, and it reuses `poly_inside` / `_rasters`.
- **Wiring:**
  - It is ckinds=moves, notices or alled at the end of `FixedSlotTracker.update`, after `_match`, and stores `self.avail`.
  - `shown()` stays as it is.
  - A new `avail_name(k)` gives the snapshot `display` (`FREE / BUSY / UNKNOWN / ENCROACHED / BLOCKED`).

### Edge cases
- **Occlusion vs encroachment:** a front car hiding a bay is already handled by `_hidden` (image-space) and shows UNKNOWN. Encroachment needs the *ground footprint* on the bay, and is evaluated only when the footprint's ground line lies within the bay's depth band (reuse `same_depth`, `history.py:290`).
- **Box jitter at the threshold:** spatial hysteresis. Enter at 0.30, leave below 0.20.
- **Vehicle moving through (manoeuvring):** the frame debounce covers it, plus a rule that the source vehicle must be BUSY-confirmed in its own bay. A car still rolling does not lock its neighbours.
- **Motorcycle or bike unplaced in a bay:**
  - It is ENCROACHED, not BUSY, when cover < 0.3 but it stands on the bay. The cover share is smkinds=moves, notices or all, so the rule is "footprint centre inside the bay".
  - This needs `parked_lock`-style persistence (60 s) first, because motorbikes are often transient.
- **Truck covering three bays:** the owner gets the middle bay or the best `match_score`, and both neighbours become ENCROACHED. That is correct capacity loss.

---

## 2. Non-vehicle obstacles and the BLOCKED state

### a) What stock COCO (YOLOv8/YOLO11, 80 classes) can already catch
| Useful as a blocker | COCO id | Reliability on a parking camera |
|---|---|---|
| potted plant | 58 | good at short/medium range |
| bench | 13 | good |
| chair | 56 | good (plastic chairs used to "reserve" spaces) |
| suitcase | 28 | medium; also catches some boxes and crates (a known false label) |
| bicycle | 1 | good (not in `VEHICLES` today, so it would count as a blocker) |
| fire hydrant, stop sign, parking meter | 10, 11, 12 | good, but usukinds=moves, notices or ally *outside* bays (useful for map checks) |
| umbrella, dining table, couch, refrigerator, tv | 25, 60, 57, 72, 62 | rare, but they work (market stkinds=moves, notices or alls, dumped furniture) |
| backpack, handbag | 24, 26 | smkinds=moves, notices or all; only near range |
| person, dog, cat, bird | 0, 16, 15, 14 | **transient: never BLOCKED**, only "activity" |

COCO has **no** traffic cone, barrier/jersey wkinds=moves, notices or all, bollard, pallet, cardboard box, debris, rubble, "reserved" A-frame sign or construction fence. Small items also lose recall past ~20 m.

### b) Outdoor items without full retraining, in order of value
1. **Class-agnostic "empty-bay appearance" check (primary, cheap, catches anything):**
   - Each bay keeps a reference patch, captured automatickinds=moves, notices or ally when the bay has been confirmed FREE with no detections for ≥ 5 min.
   - The patch is a BEV-warped or bbox crop at 96×48, grey, CLAHE, stored as edge map + gradient histogram.
   - A bank of 3 (day / dusk / night) is selected by the night-vision `light` state.
   - Every `K` frames (≈60 s), for FREE bays only, compare the live patch with the reference:
     - edge-density change + SSIM on edges (light-invariant)
     - a check that the global frame change is low, so a lighting change is not taken as an object
   - A persistent change (≥ 3 checks, ≥ 3 min) with no vehicle detection gives `BLOCKED (unknown object)`.
   - Cost: a few 96×48 ops for a handful of bays per minute, about 0.5 ms.
2. **Open-vocabulary confirmation (labels, off the hot path):**
   - YOLOE or YOLO-World (`ultralytics` `YOLOWorld("yolov8s-worldv2.pt").set_classes([...])`), prompted with: traffic cone, concrete barrier, bollard, pkinds=moves, notices or allet, cardboard box, debris, A-frame sign, construction fence.
   - It runs in a **separate low-priority process**, `scripts/obstacle_probe.py`, started by the supervisor:
     - every 5 min
     - only on crops of bays flagged by (1), or bays an operator marks
     - with results through a smkinds=moves, notices or all JSON file / LiveStore key that the worker reads (`obstacle_hits.json`)
   - It turns "unknown object" into "cone". It never alone flips a bay that (1) ckinds=moves, notices or alls unchanged, which bounds open-vocabulary false positives.
3. **Operator override:** "Mark blocked / reserved" and "Clear" on a bay, via the API and SlotEditor. Stored as `blocked_manual: {by, until?, note}` in the slot JSON. It always wins over automatic states.
4. **Later, only with data:**
   - Log open-vocab crops plus operator confirmations as a dataset.
   - A smkinds=moves, notices or all fine-tune of the camera models (`models/cam_*.pt`) with 2-3 extra classes (cone, barrier, box) using `scripts/auto_label.py`.
   - It is not needed for v1.

Detector fix (smkinds=moves, notices or all): pass `keep=self.extra` in `detect_region` and `_pass` (`yolo_engine.py:218,248`), so zoom and SAHI also return the obstacle classes. `extra` becomes the obstacle list when `obstacles=1`.

### c) BLOCKED in the state engine
- **Two layers, never mixed:**
  - `state` (occupancy: is a parked car here?) keeps 0/1/None and drives history sessions.
  - `avail` (can a car park here?) is `OPEN | ENCROACHED | BLOCKED`.
  - Display priority: BUSY > BLOCKED > ENCROACHED > FREE; UNKNOWN when the bay cannot be verified.
- **BLOCKED evidence per bay** (`ObstacleEvidence`):
  - From a COCO blocker class: footprint centre in the bay, conf ≥ 0.35, seen in ≥ 60 % of the frames over 2 min.
  - From the appearance check (1), or open-vocab (2), or a manual mark.
- **Timing:**
  - Enter: 2 min (any source), or immediately for manual.
  - Leave: 3 min clear, or immediately when a vehicle becomes BUSY in it. A car parked over a removed cone wins: occupancy beats availability.
- **A BUSY bay never becomes BLOCKED.** A blocker class seen inside a BUSY bay is ignored, because bags and people come and go around parked cars.
- **History:** `blocked` / `unblocked` events with the source (`coco:potted plant`, `appearance`, `yolow:traffic cone`, `manual`). No session. A daily "bay-hours lost to obstructions" report reads from the events.
- **Counts:** `counts.blocked`, removed from `available`. The frontend shows a hatched grey/amber bay with a label.

### Component
New module `src/carparking/obstacles.py`, with `ObstacleEvidence`, `EmptyBayModel` (reference bank + `changed(patch, light) -> float`) and `blocker_classes()`. Edge cases:
- Shadows from a parked neighbour, and snow or rain: edge-domain comparison, a global-change normalisation, and the night bank.
- A person standing in the bay: the transient class list plus 2 min persistence.
- Fkinds=moves, notices or allen leaves or litter: an area threshold. The changed edge area must be ≥ 15 % of the bay.
- Re-striped lines: the operator re-captures the reference, and the drift roadmap reference bank also helps.

---

## 3. Distant bays: tiled zoom at setup + homography extrusion

All of this runs in **setup / draft time**, in the `setup` service and the geometry pipeline. Its runtime cost is zero. The output is a child draft through the existing Suggest/Accept flow.

### 3a. Better far-field evidence (tiled and temporal)
1. **Temporal median:** take 15-30 frames over ~1 min and use the per-pixel median of the far band. This removes compression noise and passing cars, and it is the cheapest real resolution gain. It reuses the frames the setup job already grabs.
2. **Tiled zoom:**
   - The far band is everything beyond `CAR_MAX_RES`, found from `H`: the image rows where ground resolution > 0.06 m/px.
   - Split it into 50 %-overlap tiles sized so each tile upsamples ×2-×3 (Lanczos) to 1024 px.
   - On each tile run:
     - YOLO for cars (imgsz 1280, conf 0.2)
     - SAM 2.1 with **box prompts from those detections**, plus **point prompts carried over from the near surface mask** (sample points on the confirmed ground, projected into the tile)
   - Stitch the masks back with tile-centre weighting; overlapping votes count ≥ 2.
   - New endpoint `POST /segment_tiled` in `setup/service.py`, which reuses the loaded SAM predictor.
3. **Paint-line evidence:**
   - Warp the far band to BEV with `H` (`bev.py`).
   - White top-hat → LSD/Hough → segments parkinds=moves, notices or allel or perpendicular to the row direction.
   - These are stkinds=moves, notices or all dividers even where cars are too small to segment.

### 3b. Homography grid extrusion (new `src/carparking/geometry/extrude.py`)
1. **Near anchors:** bays from `pack_bays` that sit on measured cars, or on lines, within `CAR_MAX_RES`. In BEV metres, each has a centre `c_i`, row direction `u`, depth direction `n` and width `w_i`.
2. **Row model per lane:** `c(k) = c0 + k·p·u + e(k)`.
   - Fit pitch `p` (median of measured widths, or 2.5 m), `u` and `c0` by RANSAC least squares over the near anchors.
   - Far evidence then refines it with a robust (Huber) refit:
     - far car footprints (from 3a, projected to BEV)
     - painted divider positions (from 3a.3), snapped to `k·p` within ±0.4 m
3. **Extrude** `k = k_last+1 … k_max`. Stop at the first of:
   - the SAM surface ends (the mask from 3a.2)
   - `MAX_RANGE` 60 m
   - **projected bay width < `MIN_BAY_PX` (14 px)**
   - the predicted position uncertainty `σ(k)` > 0.35·p
4. **Uncertainty:**
   - Propagate the fit covariance of (`c0`, `u`, `p`) and the calibration error (GeoCalib focal and up-vector σ) through `H` with the Jacobian at each bay.
   - Store `sigma_px` and `sigma_m` per bay.
   - Bays above `σ` limits but under the hard stops are marked `far: true, confidence: LOW`, and the UI shows them dotted.
5. **Project back** to the image with `to_image` and the existing undistortion (`CameraModel`). Lens error grows towards the edges, so k1/k2 come from the near-anchor fit only.

### 3c. Runtime treatment of far bays (smkinds=moves, notices or all, opt-in `far_extrude`)
- Far bays go into the `_zoom` rotation first (`monitor.py:346`) when unmatched. This is the existing budget (`zoom_max`), with no new inference.
- Their evidence-filter weight is scaled by `min(1, px_width / 40)`. Smkinds=moves, notices or all bays move more slowly, which also cuts flicker (item 4).
- Bays the detector cannot resolve at kinds=moves, notices or all (`px_width < MIN_BAY_PX`) are not counted as FREE. They are reported as `counts.unverifiable`, which is honest capacity.

### Validation
- **Synthetic:** a known camera and `H`, true bays to 60 m, noisy near anchors only → extrusion error vs truth at 20/40/60 m, under a set tolerance.
- **Real:** camera-1 bays on hand-marked far cars from a still. Error is reported in pixels and metres.

---

## 4. Stationary-car ghost transitions

### Root causes (from the code)
- A missed detection frame makes `o=0`. The exit then needs only `exit_hold` + `exit_frames` of misses.
- Box jitter moves `ground_on` across `INSIDE_SHARE` (0.5), so the match is lost.
- Micro camera movement shifts every box relative to the bays (below `DRIFT_FRAC`).
- A night or lighting dip pushes conf below 0.30.
- A slow worker (5-60 s per frame) makes a "15 s exit_hold" mean only one or two frames.

### Layered defence (each layer is optional; evaluated on the step-0 replay)

**L1. Soft observation + spatial hysteresis** (in `FixedSlotTracker._match`):
- **Enter** needs `match_score` at the current thresholds.
- **Keep** an already BUSY bay needs only one of:
  - `ground_on >= 0.5·INSIDE_SHARE`, or
  - footprint IoU ≥ 0.5 with the bay's **parked anchor box** (see L3)
- The keep path also accepts a lower **continuation conf** of 0.15. The detector is run at 0.15 and the boxes are split by conf: ≥ 0.30 may create, ≥ 0.15 may only sustain (ByteTrack's idea, without the tracker). This costs no extra inference, because it is the same pass with a lower threshold.

**L2. EMA evidence filter:** the existing plan (occlusion roadmap §4.3), with three changes:
- asymmetric `tau_enter` 3 s / `tau_exit` 30 s
- `alpha` capped at 0.5 and a `min_frames` floor
- input `o_t = conf·cover` from L1, weighted by visibility and the far-bay px factor (3c)

**L3. Parked lock (the key ghost killer):**
- A BUSY bay whose matched footprint stays within IoU ≥ 0.7 of its first parked box for ≥ 60 s (and ≥ 3 frames) becomes `locked`.
- At lock time, store `anchor_box` and a **signature**. This reuses §4.5: an HSV histogram of the car crop plus a 32×16 grey thumbnail.
- While locked, an exit candidate (the EMA under the low threshold) must pass **kinds=moves, notices or all** of these vetoes before the exit countdown even starts:
  1. **Re-look:** one forced `_zoom` on that bay at conf 0.15 finds no vehicle. This uses the existing zoom path, once per candidacy, not per frame.
  2. **Appearance:** the live crop at `anchor_box` is no longer like the signature.
     - Bhattacharyya > 0.35 **or** thumbnail NCC < 0.6.
     - If the car still looks like the car, the detector missed it: veto, reset the candidacy, log `ghost_vetoed`.
  3. **Motion:** within the exit window, local change at the anchor exceeded `k·global change`. A real exit moves pixels; a lighting shift changes the whole frame, so normalising by the global change cancels it.
- Cost: one 32×16 compare plus one histogram, only at exit candidacy (rare), about 1 ms.

**L4. Micro-movement compensation:**
- The drift module already computes a background homography every 30 s (`geometry/drift.py`).
- When the shift is under `DRIFT_FRAC` but over 2 px, apply its translation part to the *match-time* bay polygons (not to the stored map).
- Jitter then does not change `ground_on`. Larger moves stay with the drift roadmap (freeze + re-anchor).

**L5. History safety net:**
- In `HistoryDB.apply` (`history.py:901`), an exit followed by an entry in the same bay within `reopen_window` (120 s) with a signature match **reopens the session** (§4.5).
- No pair is written; the event is kept as `reopened` for audit.
- This also covers worker restarts and stream drops (stream health already shows UNKNOWN).

**Exit latency trade-off (stated honestly):**
- Real exits from a locked bay are confirmed ~30-60 s later than today (measured in step 1: reported 48 s after the car left, 43 s later than without the layer).
- The logged exit time stays accurate, but through the existing `gone` / `expired` backdating of `SlotTracker.update` (a car absent for `hidden_max` = 30 s is timed at its first empty frame), not through `since`: a measured exit was logged at the departure (t=150) and reported at t=198. A bay that is empty for less than `hidden_max` before the filter flips would be timed at the flip.
- "Completely eliminate" is turned into a measurable target:
  - **0 ghost pairs** on the camera-1 replay of the step-0 recording
  - < 0.05 toggles per slot-hour for stationary cars
  - real exits/entries missed: 0
  - median exit delay ≤ 60 s

### Component
New module `src/carparking/evidence.py`:
```python
@dataclass
class SlotEvidence:          # one per bay, lives in tracker.ev[k]
    e: float = 0.0           # EMA evidence
    frames: int = 0
    anchor: np.ndarray | None = None  # parked box (fractions)
    sig: Signature | None = None
    locked_at: float | None = None
    exit_cand_at: float | None = None
    vetoes: int = 0

def step(ev, o_t, v, dt, cfg) -> int | None   # filtered 'seen' fed into the existing hold machine
def exit_veto(ev, crop, local_change, global_change, relook_hit, cfg) -> str | None   # reason or None
```
- `Signature` lives in `src/carparking/signature.py`, shared with occlusion §4.5 and obstacle references.
- **Hook:** the `o` computed at `history.py:558-604` passes through `evidence.step` before the debounce at 607. When `evidence_filter=0`, the path is byte-identical to today, and the Streamlit parity test stays green.

---

## Build order (after approval; each step = snapshot `scripts\snapshot.ps1 -Label pd-step<N>-pre`, tests, approved deploy)

0. **Measure (read-only, shared with occlusion step 0):** recording + `scripts/flicker_report.py` + the per-stage `timing` dict in the snapshot.
1. **Item 4, L1 + L3 + L5:** the largest correctness gain in history. Replay must show 0 ghost pairs. Then L2 (the EMA, as planned) and L4.
2. **Item 1, spillage:** share rule first, then the metric override on the camera-1 auto grid. Adds `avail`, counts and frontend colours.
3. **Item 2, BLOCKED:**
   - the `keep` fix in zoom/SAHI + COCO blocker classes + manual override
   - then the empty-bay appearance model
   - then the open-vocabulary probe process
4. **Item 3, distant bays:** temporal median + `/segment_tiled` + `extrude.py` + line snap → drafts only. Then far-bay runtime treatment.

---

## Files

- **New design doc:** `docs/post_demo_engineering_plan.md` (this task)
- **Cross-link edits (this task):**
  - `docs/occlusion_tracking_roadmap.md` (§4.3/§4.5 → item 4)
  - `docs/PHASE2_GEOMETRY_PLAN.md` (far-field risk → item 3)
  - `docs/night_vision_roadmap.md` (`light` state used by item 2's reference bank)
  - `ACTIVITY_LOG.md` (new item)
  - memory `post-demo-engineering-plan.md` + `MEMORY.md` line
- **Later code steps:**
  - new: `src/carparking/{spillage,obstacles,evidence,signature}.py`, `src/carparking/geometry/extrude.py`, `scripts/{flicker_report,obstacle_probe}.py`
  - modified: `fixed_slots.py` (`_match`, `update`), `history.py` (`update` hook, `apply` reopen), `monitor.py` (`state` counts, `_zoom` priority, forced re-look), `yolo_engine.py` (`keep` in zoom/SAHI), `live.py` (options, timing), `setup/service.py` (`/segment_tiled`), `geometry/pipeline.py`, `frontend/src/components/SlotEditor.tsx` + dashboard bay colours
- **Untouched:** `app.py`, `scripts/demo_app.py`, `./data`

## Verification

- **Step 1:** see "Step 1 as built" below (results recorded there).
- **Later steps:**
  - Unit tests per module on synthetic polygons and sequences:
    - a van over 3 bays → 1 BUSY + 2 ENCROACHED
    - jitter at the threshold → no flap
    - a stationary car with 30 % missed frames and light dips → 0 transitions
    - a real exit → 1 exit within 60 s, backdated
    - extrusion error on a synthetic camera within tolerance
    - a timing-budget test
  - Replay of the step-0 recording, old vs new pipeline, through `flicker_report.py`.
  - Full `docker compose up -d --build` (runs kinds=moves, notices or all tests) only with the user's approval.
  - A read-only live check of FPS and frame age after each deploy.

---

## Step 1 as built (2026-10-29): item 4, L1-L3 core of the ghost-transition layer, wired and OFF by default

Not deployed. The running stack is untouched; the layer is in the source and switched by one live option.

### What exists
| Piece | Where | Notes |
|---|---|---|
| Appearance signature | `src/carparking/signature.py` | HSV 16x4x4 histogram (V scaled to a fixed mean first: a lighting dip moves a car across the 4 coarse V bins, Bhattacharyya 0.79 -> same car) + 32x16 grey thumbnail; Bhattacharyya, NCC, `thumb_diff`, `same_vehicle`, `crop_box`. ~0.19 ms per signature |
| Evidence filter, lock, veto | `src/carparking/evidence.py` | `step` (asymmetric EMA, `tau_enter` 3 s / `tau_exit` 30 s, `alpha_max` 0.5, hysteresis 0.20 / 0.55), `track_anchor` (IoU >= 0.7 for >= 60 s and >= 3 frames), `exit_veto` (`relook` / `appearance` / `no_motion`), `gate_exit` (30 s cooldown, `max_vetoes` 8). ~1.5 us per `step` |
| Hook | `history.py` `SlotTracker.update` | one guarded `if self.evidence_filter:` before the debounce chain; `_evidence_obs` default returns `o`; `conf_of` / `ev_box` built only when on |
| Logic | `fixed_slots.py` `FixedSlotTracker` | `evidence_filter` / `evidence_cfg` kwargs, `_evidence_obs`, `_probe`, `_evidence_after`, `set_evidence`; per-bay `ev`, `_ev_t`, `_ev_g` |
| Pixels + re-look | `monitor.py` `process`, `_relook`, `RELOOK_CONF` 0.15 | hands the tracker the frame and a zoom ckinds=moves, notices or allable only when the layer is on |
| Switch | `live.py` `Worker._evidence` | live option `evidence_filter` ("0" default); read each frame like `_tune()`, no monitor rebuild |

### How to switch it on (when the time comes)
`store.set_option("evidence_filter", "1")` for the camera (same store as `auto_dynamic` / `slot_split`); "0" turns it off again within one frame and drops kinds=moves, notices or all its state. Do this on camera 2 or a test camera first, never before the step-0 recording is in hand.

### Rules it follows
- Off = the code path is the old one: one falsy attribute test per frame in `update`, nothing else evaluated. Proven by hash (below).
- Observation `None` (hidden, covered, unverifiable stream) passes through untouched: the layer never turns a "not judged" into evidence.
- A bay's evidence starts from its confirmed state, so switching on, editing the map or a restart does not read one missed frame as an exit.
- The frame is held for the length of one `update` only.
- Heavy work (crop signature, thumbnail diff, forced zoom) runs only when a *locked* bay wants to leave, behind a 30 s cooldown, cheapest check first (the zoom only if the other two found no reason to veto).

### Known limits (open)
- **Exit latency:** a real exit is reported ~48 s after the car leaves (EMA decay) plus the existing `exit_hold`; the logged time is the departure (see the latency note in item 4). Visible as BUSY lingering ~1 min on the dashboard.
- **Veto thresholds untuned:** `motion_ratio` 2.0, `motion_floor` 3 grey levels, `max_bhatta` 0.35, `min_ncc` 0.6 are set on synthetic scenes. They need the step-0 camera-1 recording before the option goes on anywhere real.
- **Only L1 (partly), L2, L3 are in:** the lower "continuation" confidence (0.15) for sustaining a BUSY bay (L1), micro-camera-movement compensation (L4) and session reopen across a logged exit/entry (L5) are not built. The per-stage `timing` dict in the snapshot is not built either.
- **Fixed maps and auto grids only:** the learned (dynamic) tracker has no such hook.
- **No API / UI switch yet:** the option is set through the store.

### Verification recorded (2026-10-29)
- Pre-wire snapshot: `C:\F\Backups\CarParking\20261029-181032-pd-step1-pre-wire` (OK, 444 files; restore with `scripts\restore.ps1`).
- Baseline before wiring (`docker build --target test`): 502 passed, 26 skipped, pyflakes clean.
- Off-path identity: the same 16 randomised runs x 600 frames (plain fixed, capacity pin, auto grid, auto grid + sub-bays) through the snapshot's code and the new code give the same SHA-256 over events, slot states, shown states, held, loads, new / retired slots and unplaced; the hash differs when the layer is on, so the check discriminates.
- New tests: `tests/test_signature.py`, `tests/test_evidence.py`, `tests/test_fixed_slots_evidence.py` (stationary car with dropped frames and a 75 s exposure dip: 0 exits with the layer, >= 1 without; real exit once; replacement car not vetoed; entry not delayed beyond 3 s; re-look veto; no-frame case; switching on/off; slot removal; off never reaches the evidence code). Mutation checks: veto off, EMA off and lock off each make them fail.
- Final full-suite result: see the ACTIVITY_LOG entry (item 114).

## Step 2 tooling: pixel recording and offline replay (built 2026-10-30; no threshold is set by it)

The veto thresholds are computed from pixels, and `track_eval.py record` keeps only boxes, so tuning needs a different recording. This tooling
records frames and replays them; it holds no camera-1 value and picks none. Deriving the values is the next step and needs the footage.

| Piece | File | What it does |
|---|---|---|
| Recorder | `scripts/pixel_record.py` (`record`, `info`), `src/carparking/pixel_replay.py` `record_pixels` | Reads `GET /api/status`, `/api/settings`, `/api/frame.jpg?raw=true` (read only) and writes `recordings/<camera>_pixels/<timestamp>/` = `header.json` (slot map, tracker settings), `frames.jsonl` (time, vehicle boxes / conf / label, the states the worker showed; appended per frame), `frames/NNNNNN.jpg`. A frame is kept only if the status read before and after the picture names the same `frame_ts`. `--every` thins, `--max-gb` caps, an `--out` inside `./data` is refused. |
| Metrics | `src/carparking/flicker.py` | Ghost pairs (exit then entry of one slot within a window), short states, state changes per slot-hour, report lag (event time vs the frame that reported it), exit scoring against an optional truth file. The window (default 120 s) and blink (60 s) are measurement choices. |
| Flicker report | `scripts/flicker_report.py` | `DIR` = the live states as recorded; `--source off / on / kinds=moves, notices or all` replays; `--set NAME=VALUE` overrides EvidenceCfg fields. |
| Replay harness | `scripts/replay_eval.py`, `pixel_replay.replay` | The recorded boxes and pictures through a fresh `FixedSlotTracker` (the recording's slot map, the worker's hold / exit_hold / match settings), live vs layer off vs layer on in one table; `--sweep NAME=v1,v2,...` (repeatable, cartesian) gives one row per combination with ghost pairs, exits, vetoes, lag and, with `--truth`, missed / false exits. |

Use: leave `python scripts/pixel_record.py record --cam camera-1 --minutes 600` running over a change of light (token in `CARPARK_VIEW_TOKEN`), note the real departures it contains in a truth file, then `python scripts/replay_eval.py DIR --truth truth.json --sweep motion_ratio=... --sweep min_ncc=...`. Pick the value that keeps every real exit with an acceptable lag and leaves no ghost pair on the parked cars; only then set it in `EvidenceCfg`.

Limits: boxes are the published axis-aligned ones (no segmentation polygon); vehicle-driven bays of an auto grid are not replayed; the veto's zoom re-look needs the detector and is not available offline (`appearance` and `no_motion` are); the replay-off column should agree with the live column (the agreement line says by how much). Tests: `tests/test_pixel_replay.py` (fake worker with a parked-car scene).

## Step 3 as built (2026-10-30): item 1, spillage / multi-slot encroachment, wired and OFF by default

| Piece | Where | What |
|---|---|---|
| Rules | `src/carparking/spillage.py` | `SpillCfg` (share 0.30, total 0.45, leave 0.20, align 0.30), `spill_map` (pure), `AvailTracker` (debounce: `hold` s **and** 2 frames to enter, `exit_hold` s **and** 3 frames to leave; a bay that stops being eligible is cleared at once) |
| Wiring | `fixed_slots.py` `FixedSlotTracker` | `spillage` / `spill_cfg` kwargs (default 0), `_spill` after `SlotTracker.update`, `.avail` (slot -> `Spill`), `.avail_events`, `avail_name(k)`, `set_spillage(on)` |
| State | `monitor.py` `state()` | with the layer on: per slot `avail` / `display` / `encroached_by` / `cover`; `counts.encroached`, `counts.available`, and `counts.free` no longer holds encroached bays. Off: the JSON is exactly what it was |
| Switch | `live.py` `Worker._spillage` | live option `spillage` ("0" default), read every frame like `evidence_filter`; `store.set_option("spillage", "1")` |

How it decides (a free, unowned bay of capacity 1 is the only candidate; `state` is never changed, so history, `sync_open` and the parity test are unaffected):
- **30 % single-car boundary:** one vehicle covering >= 30 % of the bay -> ENCROACHED; **or** several vehicles whose covers sum to >= 45 %.
- **Hysteresis:** once ENCROACHED it stays until the largest cover < 20 % (and the sum < 30 %), so box jitter at the line does not flap.
- **30 % alignment (depth band):** a vehicle counts only if its ground line lies within 30 % of the bay's height above its top / below its bottom: same row, same depth. A van in front of or behind the bay is not on it.
- **Sources:** a vehicle BUSY-confirmed in a bay of its own (a car still rolling locks nothing), **or one that holds no bay at kinds=moves, notices or all**. The second case was found while testing: with the default `bay_share` 0.5 a van wider than about two bays is matched to no bay (it is listed as unplaced) and every bay under it stayed FREE, which is the false free count this item exists for. Its `Spill.by` is `(0,)` ("unplaced vehicle").
- **Cover measure, a deliberate departure from the text above:** the share of the bay lying within the vehicle's lateral extent on its row, not the area under `footprint_poly`. That strip is only the bottom 30 % of the box, so a car wholly over a bay would score ~0.4 and 0.30 would mean 73 % of its width. This way 0.30 means 30 % of the bay, also for a bay drawn in perspective.

Not built / open:
- **No events in the history table.** `encroached` / `encroach_cleared` are produced (`FixedSlotTracker.avail_events`) but not persisted: the React pages show every event that is not `entry` as "vehicle out" (`Pages.tsx`, `SidePanel.tsx`), so storing them now would show as departures. Persisting needs the API / UI to filter event kinds first.
- Metric (BEV) override for calibrated grids (`Spill.free_m` stays None); capacity-pin "usable length"; frontend colours and an API / UI switch for the option; the "crowded by" note on owned bays; the 60 s `parked_lock` for motorbikes.
- Unplaced vehicles as sources can lock bays under a vehicle that is only manoeuvring slowly through the row (>= `hold` s and 2 frames over >= 30 % of a bay); the replay tooling (`replay_eval.py`) does not measure this yet.

## Step 4 as built (2026-10-30): item 2, first layer: the non-vehicle filter, wired and OFF by default

The approved item 2 above is about the BLOCKED state (an obstacle makes a *free* bay unusable). This layer is the other half of the same
problem, requested separately: **detections the model ckinds=moves, notices or alls a vehicle that are really clutter must not make a bay BUSY**. It is an
isolated filter; BLOCKED (`obstacles.py`, the empty-bay appearance model, the open-vocabulary probe, `counts.blocked`) is not built.

| Piece | Where | What |
|---|---|---|
| Rules | `src/carparking/obstacle_filter.py` | `ObstacleCfg`, `judge(box, ref_area, crop_fn)` -> reason or None, `color_breakdown(crop)` (HSV on a 48-px copy: orange / dark / bright / other + texture) |
| Wiring | `fixed_slots.py` `FixedSlotTracker` | `obstacle_filter` / `obstacle_cfg` kwargs (default 0), `_reject` at the top of `update` (before spawn, matching, loads, unplaced), `.rejected`, `set_obstacle_filter(on)` |
| Pixels | `monitor.py` `process` | hands the tracker the frame when the option is on (as for `evidence_filter`); `state()` gets a top-level `rejected` list only when on |
| Switch | `live.py` `Worker._obstacle_filter` | live option `obstacle_filter` ("0" default), read each frame; `store.set_option("obstacle_filter", "1")` |
| Replay | `scripts/replay_eval.py --obstacle` | a column with the filter on: detections dropped by reason, and whether events / shown states differ from the filter off |

Rejection reasons, first that applies (only vehicle detections, and never a box standing in a bay that is already BUSY: the filter stops
clutter making a bay busy, it does not take a parked car away):
- **too_smkinds=moves, notices or all:** box area < 0.25 of the space it stands in (the bay's bounding-box area, divided by the capacity for a pin). Relative to the bay,
  not the frame: far cars are a few hundred pixels. Skipped for a box in no bay.
- **aspect:** width / height outside [0.5, 4.5] (people, poles, bollards; fences, kerb runs).
- **pylon (colour breakdown):** the box is below 0.60 of its bay's area (not a whole vehicle) and >= 0.45 of its pixels are cone orange
  (hue 4-20, saturation >= 150, value >= 110). A whole orange car is never rejected on colour.
- **shadow:** >= 0.75 of the pixels darker than V 60 and a flat crop (mean |Sobel|/8 < 3.0). Deliberately strict: a wrongly rejected black
  car costs more than a faint shadow let through (which is the old behaviour).
- No frame at hand: only size and aspect apply (the colour rules are skipped, never guessed). An empty or unreadable crop lets the box through.

Checked against real footage: on the camera-1 pixel recording, kinds=moves, notices or all 326 sampled vehicle boxes pass (aspect 0.65-3.16; box area 0.9-2.1 x its
single bay, 0.22 x a 4-space pin = 0.9 per space; dark share <= 0.43; texture >= 11.2; no cone orange), and a replay of kinds=moves, notices or all 256 frames with the
filter on drops 0 detections, with identical events and shown states. **There is no real footage of cones, bins, people or shadows yet**, so
the rejection side is verified on synthetic scenes only; the thresholds are conservative defaults, not tuned values.

Open: BLOCKED proper (the free-bay side); rejected boxes are still drawn on the overlay as vehicles; a smkinds=moves, notices or all motorcycle (< 25 % of a car bay)
is rejected as too smkinds=moves, notices or all, adjust `min_area_share` if bikes should count; rejections are not persisted; no API / UI switch.

## Step 5 as built (2026-10-30): settings API, status payload and React UI for spillage protection and the obstacle filter

Both layers stay OFF by default and are worker options (like `model_variant`): they never touch the page-shared settings or their version, so the Streamlit page is unaffected.

| Piece | Where | What |
|---|---|---|
| Settings API | `api/main.py` `SettingsPatch` / `put_settings` / `named` | `spillage: bool`, `obstacle_filter: bool`. `GET /api/settings` returns both under `values` (false by default); `PUT /api/settings` (admin token) stores them as the LiveStore options `spillage` / `obstacle_filter` ("1" / "0"), without bumping the settings version; an explicit `null` changes nothing; a non-boolean is 422. Mixed with a page setting only that one versions |
| Live application | `live.py` `Worker._spillage` / `_obstacle_filter` | read every frame: the running tracker is switched in place, no restart, no monitor rebuild (turning spillage off emits the clearing events on the next update) |
| Status payload | `GET /api/status` -> `snapshot` | per slot (spillage on) `avail`, `display`, `encroached_by`, `cover`; `counts.encroached`, `counts.available`, and `counts.free` already without encroached bays; (obstacle filter on) `snapshot.obstacle = {"rejected": n, "reasons": {"too_smkinds=moves, notices or all": n, ...}}`. With both off the payload is what it was |
| UI: switches | `SettingsDrawer.tsx` section "Parking protection" | two toggles, "Spillage Protection" and "Non-Vehicle Obstacle Filtering", in the normal draft / Apply flow; Apply sends only the changed field with the version |
| UI: video | `LiveView.tsx`, `Grid.tsx`, `lib.ts`, `ui.tsx` | an encroached bay is drawn amber with a diagonal hatch (SVG pattern) and labelled "<id> ENCROACHED"; the hover says "FREE, not usable" and "encroached by A1 (42 % of the bay)"; the grid tiles use the amber colour |
| UI: counts | `SidePanel.tsx` | the Free card shows the usable bays (the API already excluded encroached ones); an extra "Encroached (free, not usable)" card appears only when spillage is on, and filters the video like the others (`SlotFilter` gained "ENCROACHED"; the FREE filter shows usable bays only). The Worker card gets an "Obstacle filter" line (dropped count by reason) only when the filter is on |

Not done: persisting `encroached` / `rejected` in the history (the React pages show any non-`entry` event as "vehicle out"), the metric override, drawing the dropped boxes differently on the video, and the Streamlit pages (deliberately untouched).


## Step 6 as built (2026-10-30): kinds=moves, notices or all in the event log, and dropped boxes on the video

**Event log.** `encroached`, `encroach_cleared` (spillage layer) and `rejected` (non-vehicle filter) are written to the camera's SQLite `events` table next to `entry` / `exit`, as *kinds=moves, notices or all*: no session (`session_id` NULL), `slot_id` 0 when they concern no bay, a human `note`. Nothing that reads the log for movements sees them:

| Reader | Behaviour |
|---|---|
| `HistoryDB.events / events_csv / summary` | new `kinds` argument; default `MOVES` (entry, exit), so Streamlit, the CSV download, `summary().events` / `last_event` and the PostgreSQL mirror are exactly what they were; `NOTICES` or `None` (kinds=moves, notices or all) on request; `log_notice()` refuses any other kind |
| `SqliteHistory` (history_pg.py) | events, summary, CSV and analytics filter to entry / exit; `summarize()` ignores other kinds (it used to count anything that is not `entry` as an exit) |
| PostgreSQL mirror | unchanged: the `CHECK (event IN ('entry','exit'))` and the copy filter stay; kinds=moves, notices or all are a per-camera SQLite audit |
| `GET /api/events?kinds=moves, notices or all\|notices\|all` | default `moves` = as before (PostgreSQL when fresh); `notices` / `all` are always read from SQLite (`X-History-Source: sqlite`); `422` for another value. `/api/events.csv` takes the same `kinds`. The WebSocket feed now carries every new event, so the ticker shows notices live |
| Writing | `ParkingMonitor._log_kinds=moves, notices or all` after each frame: the spillage layer's `avail_events` (note "by A1, 42 % of the bay: free but not usable"; "spillage switched off"), and `RejectLog` for the filter. A failing write sets `monitor.notice_error` and never stops the frame |

`RejectLog` (obstacle_filter.py) keeps the rejection kinds=moves, notices or all few: one `rejected` row per (bay, reason) after the same kind of detection has been dropped for 5 s **and** in 2 frames, then silent for 10 min; an episode unseen for 60 s starts over; slot 0 = in no bay. Note text: `too_small: too small for the bay in bay A3, 40x80 px, not counted as a vehicle`.

**React.** `lib.ts` `eventInfo / isMovement / eventSlot` classify an event (a kind this build does not know is a notice with its own name, never "vehicle out"); the Event log page asks `kinds=kinds=moves, notices or all` and has an All / Entries & exits / Spillage & obstacles filter (the CSV link follows it), notices get an icon label (⚠️ ENCROACHED, ✅ CLEARED, 🚫 IGNORED), a dash for no bay and no session; the live ticker gives each kind its own icon, colour and wording; Analytics' entries / exits per hour count movements only.

**Video.** A detection the filter dropped is marked in the snapshot (`detections[].rejected = reason`), no longer counted in `snapshot.vehicles`, and left out of the metric (geometry) view. The React overlay draws it grey and dashed with an `ignored: <reason>` tag and a hover that says why; the baked overlay (`render_overlay`, clean frames off) draws it as a dashed grey polygon with the same tag instead of the yellow vehicle box. With the filter off nothing is marked and the drawing is what it was.

Not done: kinds=moves, notices or all are not mirrored to PostgreSQL (so not in its analytics or the central History when PostgreSQL answers `kinds=moves`); no retention rule for notices beyond the camera file's own; the Streamlit pages (deliberately untouched) show movements only.

## Step 7 as built (2026-11-02): far-field bays

> **REVERTED 2026-11-02 at the operator's request** (it disturbed the stable auto-grid suggestion). The code described below was removed; a copy of the sources is in `recordings/reverted_far_field_20261102/`. See ACTIVITY_LOG item 129. The text is kept as the design record., setup / draft time only (item 3, 3a + 3b + the draft preview)

**Switch.** `CARPARK_FAR_EXTRUDE` (default off; `Config.far_extrude`) and, per call, `far_extrude` in `POST /api/cameras/{id}/suggest` (`null` = the server default, `false` always wins). Off = the suggest job runs exactly the code it ran before (tests compare the two drafts bay for bay). It lives behind `CARPARK_GEOMETRY_ENABLED`, like all of camera setup.

**What it does.** The far bays only ever exist in a **draft** (`record["grid"]` entries with `source: "far"`, `far`, `confidence` MEDIUM/LOW, `sigma_px`, `sigma_m`; `record["suggest"]["far"]` = `{zone, stop, model, evidence, error, warnings}`). The worker reads the ACTIVE version only, `fixed_slots.json` is never written by the job, a Fixed camera merges only ticked bays (far bays are **not** ticked by default), an auto camera's Accept names the far bays that would become real slots.

| Module | Role |
|---|---|
| `geometry/extrude.py` | `fit_row` (RANSAC over the near bays' centres, preferring lines that lead away from the camera; Huber total-least-squares with far cars and painted dividers; Huber least squares for pitch and phase, covariance from the residuals), `extrude` (continue the lane with the nearest measured bay as template; stops: 60 m, frame, bay under 14 px tall, SAM surface end, uncertainty over 0.35 of the pitch), `far_report` (read-only diagnostics) |
| `geometry/far_evidence.py` | `far_band` (rows where one pixel covers over `CAR_MAX_RES`), `band_tiles` (50 % overlap, x2 Lanczos), `temporal_median`, `boxes_from_tiles`, `paint_dividers` (bird's-eye white top-hat + Hough, only lines across the lane), `far_surface_m` |
| `setup/service.py` `POST /segment_tiled` | SAM on the upsampled band tiles, prompted with near-surface keypoints and far cars' contact points, stitched with tile-centre weights |
| `slot_suggest.py` | `measured_tiles` (the extrusion's anchors), `add_far_bays` (fresh ids above everything in use; running bays keep theirs) |
| `api/geometry_routes.py` | `far_extrusion` in the suggest job (best effort: each failed evidence step becomes a warning, an impossible extrusion becomes `far.error`, the near draft is never affected); `GET /api/cameras/{id}/far-field` (admin, read-only: no still, no model) |
| React `SuggestSlots` | "Include far bays (preview)" checkbox (off), "Check far field" button, far bays drawn violet and dotted (LOW) or dashed (MEDIUM), the capacity zone (hatched outline, "~N bays · not verifiable"), a plain-words summary |

**Uncertainty is a result, not a setting.** The calibration part of `sigma_px` is found by running the whole pipeline again under a camera whose pitch (0.5 deg) and focal length (3 %) are off, with the same image evidence: what matters is where the bay lands in the picture, because the near bays were measured with the same camera and their errors travel with it (an absolute 0.5 deg pitch error would be metres at 40 m and would stop the lane at 17 m, which is not what the picture shows). On a synthetic camera-1-like scene (h 2.78 m, pitch 14 deg, 10 measured bays out to 28 m) it continues 2 bays (to 33 m, LOW confidence) and then reports a capacity zone of about 10 bays from 34 to 59 m. Far cars and dividers tighten the pitch estimate, they do not change the calibration's share of the uncertainty.

**Not built.** Runtime treatment (plan 3c: the `_zoom` priority, the evidence weight scaled by pixel width, `counts.unverifiable`) and the real-footage validation (hand-marked far cars on a camera-1 still): both need a decision after the drafts have been looked at. The thresholds (14 px, 0.15 / 0.35 of the pitch, 0.5 deg / 3 % priors) are the plan's, not fitted to camera-1.

## Step 8 as built (2026-11-02): manual pins (P1) kept by slot suggestions

> **REVERTED 2026-11-02 at the operator's request** (it disturbed the stable auto-grid suggestion). The code described below was removed; a copy of the sources is in `recordings/reverted_far_field_20261102/`. See ACTIVITY_LOG item 129. The text is kept as the design record.

**Problem.** A Fixed camera's capacity pin (camera-1's P1, slot 9, holds 4, standing at 27-74 m) was handled like any hand-drawn slot: a suggested bay overlapping it took its id and name, otherwise it showed as "removed" in the diff, and the far bays and the capacity zone ran over the ground it already counts.

**Option.** `keep_pins` on `POST /api/cameras/{id}/suggest` (`null` = `CARPARK_SUGGEST_KEEP_PINS`, default off = the suggestion exactly as before; the editor's "Preserve existing manual pins (e.g. P1)" box is ticked by default and sends it). What counts as a pin: a Fixed camera's slot that holds more than one vehicle (`slot_suggest.fixed_pins`), an auto grid's saved pins (`source: manual`, as before).

| With the option | Effect |
|---|---|
| ids and shapes | the pin keeps its id, name, polygon and capacity; no suggested bay takes its id (it is reserved) and a suggested bay standing on it (IoU >= `PIN_IOU` 0.3, the rule `merge_pins` uses for an auto grid) is dropped: the pin is the operator's word |
| diff | the pin is out of both sides of the diff: never "removed" |
| far field | `extrude(pins_m=...)`: an extruded bay a pin covers (>= 25 %) is left out (`pinned`) and the lane goes on past it; the capacity zone is net of the capacity of the pins inside it (`zone.pins`, `zone.pin_capacity`). Pins are respected, not used as lane evidence (a pin's centre is not a bay's centre) |
| draft | `meta.keep_pins = {kind, ids}` (only when on), `pins` in the view are the live pins, drawn purple and named in a "Manual pins kept as drawn: P1 (holds 4) ..." line; never ticked for a Fixed camera's merge, and a merge leaves the pin exactly as drawn |

Camera-1's P1 sits in the zone beyond the extruded bays, so a far run there now reports a zone net of P1's 4 spaces. Not changed: the running map, the worker, `fixed_slots.json`.

## Step 9 as built (2026-11-02): "Blocked by object" (the BLOCKED state, plan item 2's unfinished half)

A FREE bay with a non-vehicle object standing in it (potted plant, vase, suitcase / box, chair, bench) is **BLOCKED**: FREE for the history and the sessions (nothing is opened or closed, no entry or exit), out of the dashboard's Free count, drawn cross-hatched slate and named BLOCKED. Off by default.

| Piece | Behaviour |
|---|---|
| option | `blocked_objects` in `PUT /api/settings` (worker option like `spillage`, no settings version, no restart; read every frame by `Worker._blocked`); Settings drawer > Advanced Smart Protections > "Blocked by Object Detection" |
| detection | while on, the object classes join the monitor's target classes (`ParkingMonitor.set_blocked`), so the detector reports them; they arrive as shown-only "others", never as vehicles; off restores the class list exactly |
| rule (`blocked.py`) | label in `BlockedCfg.labels` (potted plant, vase, suitcase, chair, bench), confidence >= 0.30, >= 50 % of the object's footprint (bottom 30 % of its box) inside a FREE, unowned, single-space bay (25 % keeps it); an object blocks the bay it stands in most; a bay a vehicle owns, a capacity pin, a person or an animal never |
| debounce | `spillage.AvailTracker` with its own event names: blocked after the entry delay AND 2 frames, cleared after the exit hold AND 3 frames |
| state | additive: slot `avail` / `display` = `BLOCKED`, `blocked_by` (labels), `counts.blocked_objects`; `counts.free` and `counts.available` leave the blocked bays; `state` stays FREE. (`counts.blocked` and a slot's `blocked` already mean "a vehicle stands in it", a different thing.) BLOCKED wins over ENCROACHED in the name |
| event log | notices `object_blocked` / `object_cleared` (never a session; not in PostgreSQL, like the other notices) |
| UI | cross-hatched slate bay + label + hover "blocked by potted plant: free, but not usable", a "Blocked by object" counter card and filter (only with the option), ticker / event-log wording |

Limits: the detector must name the object (COCO classes; an unlabelled box or a plant the model calls something else is not seen); the class list is `BlockedCfg.labels` (not a setting yet); nothing is validated on real footage (the tests are synthetic).

## Step 10 as built (2026-11-02): the sample image test

A page "Sample test" (sidebar, `#/sample`): upload a JPEG / PNG (or pick one from `<data>/samples/`), choose the layers, **Run test**. The picture is one camera frame through the camera's whole stack, drawn and counted like the live dashboard.

| Piece | Behaviour |
|---|---|
| `POST /api/sample-test?cam=` (admin) | `{image: base64}` or `{sample: name}` and optional `{layers: {spillage, obstacle_filter, blocked_objects}}` (default: the camera's live options). 422 for a bad picture, 404 unknown sample, 409 no slots / a Dynamic camera without an auto grid, 500 with the reason when the detector fails. One test at a time |
| `GET /api/sample-test/images` | the pictures in `<data dir>/samples` (jpg / png) |
| pipeline (`sample_test.py`) | the camera's own detector runs ONCE (`detect_frame`; the object classes are added only when blocked-by-object is on); a fresh `ParkingMonitor` built from the camera settings and slots (a Fixed map's, or the active HIGH calibration's auto grid with its pins) replays that answer at steps of a fifth of the hold time until every state has settled (two hold times: BUSY, then spillage / blocked), with the three layers as asked and spillage's ground metric when the picture fits the calibration |
| isolation | a throw-away tracker and an in-memory history: no worker, no live database, no saved slots, nothing written to the real event log. The notices / entries the run produced come back in the result |
| UI (`SampleTest.tsx`) | the picture with the slots coloured (Free green, Busy red, Blocked cross-hatched slate, Encroached amber hatched), the vehicle / object / ignored boxes with hover, the dashboard's own cards (Free = usable bays, Busy, Blocked by object, Encroached), the run's notices as the ticker, a slot table, and optional expected Free / Busy / Blocked counts checked ✓ / ✗ |

Not done: the live dashboard's top cards and the real event log are deliberately not touched by a test (they would mix test pictures into live numbers); the worker's zoomed re-check of hidden slots does not run (a picture has no history); a Dynamic camera that learns its slots cannot be tested.

## Step 11 as built (2026-11-02): sample pictures get their own calibration
A picture from anywhere is calibrated in a throw-away session (ground rectangle pins by default, GeoCalib + SAM + a typed distance as the advanced path), its bays are reviewed and edited, and the
full stack runs on it. In-memory only; live camera config, geometry and the event log are never touched. See ACTIVITY_LOG item 135. "Save as test case" is not built.

Step 11 follow-up (2026-11-02): the "This camera's slots" mode and its endpoints were removed; the Sample test page is the session workflow only, GeoCalib + SAM first (ground pins kept as the no-service fallback). See ACTIVITY_LOG item 136.
