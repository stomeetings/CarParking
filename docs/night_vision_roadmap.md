# Night Vision & Low-Light Enhancement Pipeline Roadmap

Status: **planned, nothing implemented yet.** No code or live file is changed before the 2026-10-29 demo. Written 2026-10-28.
Companion plans: `docs/occlusion_tracking_roadmap.md`, `docs/camera_drift_roadmap.md`.

## Architectural Separation & Pipeline Flow

Night vision and low-light optimization is strictly separated from **Occlusion & Spatial Tracking**.
* **Night Module:** Operates in the **image domain** prior to detection (luminance, contrast, and threshold policies located in `carparking/lowlight.py`).
* **Occlusion Module:** Operates in the **space and time domain** after detection (overlap, pillars, scaffolding, and track IDs). The sole interface between them is a scalar light state/quality attached to the frame result. Trackers never see raw pixels.

`carparking/lowlight.py` is planned; it does not exist yet.

### Pipeline Flow Diagram

```
camera frame
   |
   v
[stream health]  (stream_health.py: stale / frozen -> every slot UNKNOWN)
   |
   v
[LIGHT PROBE]  -> light = {state: DAY | DUSK | NIGHT, luma, contrast, quality}
   |
   +--------------------------------------------+
   | original frame                             | detection copy
   v                                            v
overlay, snapshot, drift guard          [ENHANCER]  CLAHE / gamma, only in DUSK / NIGHT
(always the ORIGINAL pixels)                    |
                                                v
                                     detector @ conf_for(light, model)
                                                |
                                                v
                                     zoom re-checks @ zoom conf_for(light)
                                                |
   ======================= image domain ends here; only `light` crosses =======================
                                                |
                                                v
                              [OCCLUSION / TRACKING]  history.py, fixed_slots.py
                                                |
                                                v
                              hysteresis (+ lighting-drop grace) -> slot states -> history
```

## Facts this plan builds on

- **Confidence:** the default is already **0.30**, not 0.50 (`live.py` `UI_DEFAULTS["conf5"]`). Zoom re-checks have their own thresholds: `zoom_conf` 0.30 and the stricter `zoom_free_conf` 0.40 (`monitor.py`). The stricter one is what keeps a visibly empty bay from turning BUSY on a weak box.
- **Hysteresis already exists and is longer than 5–8 s:** entry hold 15 s, exit buffer 30 s, plus 15 consecutive empty frames (`hold5` / `exit5` / `exitn5`). The night pipeline adds a short *lighting-drop grace* on top of these; it does not replace them.
- **Already available to reuse:** a per-frame grayscale thumbnail (`stream_health.FreezeDetector`, 64x36) and a CLAHE call (`dynamic_iou.py`).

## Specification

### 1. Light probe
- Computes mean luma, p10 / p99 luma and RMS contrast from the stream-health thumbnail. No extra resize, under 0.1 ms.
- States DAY / DUSK / NIGHT with hysteresis: enter NIGHT below `L_night` for 60 s, leave it above `L_day` for 60 s. A passing headlight never flips the state.
- An optional time-of-day prior (fixed hours, or sunrise / sunset from a configured latitude / longitude) breaks ties at dusk.
- Mode: Auto / Off / Force. The option `night_mode` defaults to `off`.

### 2. Adaptive pre-processing (CLAHE / contrast boost)
- CLAHE on the L channel of LAB (clip 2.0, 8x8 tiles), applied only in DUSK / NIGHT.
- Applied to the **detection copy** only. The overlay, snapshot and drift guard keep the original pixels.
- Run at model input size where possible (about 3–8 ms on CPU).
- Optional gamma lift for very dark frames (p10 near 0).
- IR / monochrome night picture (chroma near zero): CLAHE on grayscale only.
- Glare guard: no boost when p99 luma is saturated (headlights, floodlights), so glare halos are not amplified.

### 3. Dynamic confidence thresholding
- `conf_for(light, model)` is a per-model table (yolov8s and YOLO11m have different score distributions).
- DAY uses the configured value (0.30). NIGHT lowers it to a floor of 0.22–0.25; DUSK sits in between.
- `zoom_conf` is lowered the same way. **`zoom_free_conf` is never below 0.35**, the guard against false BUSY on empty bays.
- Every box accepted only because of the night threshold is tagged `lowlight: true`, so it can be audited.
- Exact values come from the step 0 measurement, not from this document.

### 4. Temporal hysteresis integration
- Entry hold (15 s) is unchanged.
- **Lighting-drop grace:** when luma drops sharply between frames (more than ΔL, e.g. lights switched off or a headlight sweep ending), FREE countdowns pause for 5–8 s, the same way the occlusion freeze pauses them.
- In NIGHT, `exit_hold` gets +5–8 s.
- If the occlusion roadmap's evidence filter ships, its update weight is scaled by `light.quality`. That scalar is the only coupling between the modules.

## Code / pipeline hooks

| Hook | Where | What goes there |
|---|---|---|
| Light probe | worker loop, before `self.mon.process(frame, t)` (`live.py` ~745) | `light = probe.update(thumbnail, t)`; passed into `process` |
| Pre-processing | `ParkingMonitor.process`, before `self.detector(frame, ...)` (`monitor.py` ~237) | `det_frame = enhancer(frame, light)`; detector and zoom use `det_frame` |
| Dynamic confidence | `monitor.py` ~231 (`self.detector.conf = self.conf`) | `self.detector.conf = conf_for(light, model)` |
| Zoom thresholds | `monitor.py` ~338 / ~393 (`detect_region(..., conf=...)`) | `zoom_conf` from `conf_for`; `zoom_free_conf` floor 0.35 |
| Model call | `yolo_engine.py` ~178 / ~210 | unchanged: receives the enhanced copy and the chosen conf |
| Lighting-drop grace | `SlotTracker.update` (`history.py`) | a `light_event` input that pauses FREE countdowns, like the occlusion freeze |
| Observability | worker snapshot (`live.py` snapshot builder) | `light: {state, luma, contrast, conf_used, enhanced}` + timing per stage |

Line numbers are as of 2026-10-28 and will move; check them before implementing.

## Observability & settings
- Snapshot field `light`; a "Night mode" badge on the dashboard; per-stage timing (probe, enhancer).
- Settings (display strings in the item 112 style): Night Mode (Auto / Off / Force), Night Confidence Floor, Contrast Boost Strength.

## Build order (post-demo)

Each step: snapshot first (`& .\scripts\snapshot.ps1 -Label night-step<N>-pre`), tests, user-approved deploy, option off by default.

0. **Measure offline (prerequisite).** Collect night frames from camera-1. A new `scripts/night_eval.py` replays them through raw / CLAHE / gamma × conf 0.30 / 0.25 / 0.22 against hand counts and reports recall, false positives and ms per frame. Nothing goes live without these numbers.
1. **Light probe, observe only:** snapshot field + dashboard badge; no effect on detection.
2. **CLAHE on the detection copy**, behind the `night_enhance` option.
3. **Dynamic confidence**, behind the `night_conf` option.
4. **Lighting-drop grace** in the hysteresis.
5. **Settings UI** for the night options.

## Verification and rules
- Unit tests:
  - probe state transitions with hysteresis (a single bright or dark frame never flips)
  - enhancer only in DUSK / NIGHT; original frame byte-identical after processing
  - `conf_for` floors, including `zoom_free_conf` >= 0.35
  - grace pauses FREE countdowns only, never entries
  - timing budget on a 1920 px frame
- Offline replay (step 0 frames): night recall up, with no more false BUSY than today.
- Read-only FPS / frame-age check on the live worker after each deploy.
- Additive JSON / options only (no SQL migrations). Tests never touch `./data` or the live DB. Streamlit (`app.py`, `demo_app.py`) untouched.
- Deploy only with the user's approval via `docker compose up -d --build`. Log every step in `ACTIVITY_LOG.md`.
- **No live code or production file is modified before the 2026-10-29 demo.**
