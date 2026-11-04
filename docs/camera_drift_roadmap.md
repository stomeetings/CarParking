# Camera Drift & Auto-Correction: design and roadmap

Status: **planned, nothing implemented yet.** Start after the demo (2026-10-29). Written 2026-10-28.
Stack: Python (FastAPI, workers, OpenCV/numpy) + React. There is no Node.js server in this project, so all CV work stays in Python.

## 1. What already exists

| Piece | Where | What it does |
|---|---|---|
| Drift detection | `src/carparking/geometry/drift.py` | Each calibration stores ORB features of the static background (vehicles, timestamp strip and camera-name strip masked out). Every 30 s the worker matches the live frame against them. RANSAC keeps matches that agree on one movement. Drift = 90th-percentile shift of those points (catches rotations as well as slides). Threshold `DRIFT_FRAC` = 0.6 % of frame width (~11 px at 1920). |
| Debounce | `geometry_live.py` `observe()` | First hit logs `drift_suspected`. 3 in a row (~90 s) marks the calibration `stale`. A clean check logs `drift_cleared`. |
| Three outcomes | `drift.compare` | `ok`, `drift`, `inconclusive`. Night, glare, rain, a blocked view or too few matches give `inconclusive`, which never alarms. |
| Auto-correct as a draft | `geometry/reanchor.py`, `POST /api/cameras/{cam}/setup/reanchor` | Solves how the camera turned; keeps focal length, lens, height and scale; rebuilds bays for the new view. Refuses with a reason if the camera was moved (not turned), turned more than 25 deg, has too few matches, or half the bays are out of view. Result is a child draft; only an admin Accept activates it. |
| Draft guardrails | suggest/accept endpoints in `api/geometry_routes.py` | Accept is refused while drift is `drift`/`stale` or the baseline changed (`baseline_hash`). |

## 2. Gaps

1. **Counting after drift.** When camera-1's calibration goes `stale`, the 12-bay auto grid disappears and the worker falls back to the learned map (`live.py` `_auto_slots`): shifting slot list, extra bays, wrong counts.
2. **No drift guard on Fixed-mode cameras.** A drawn Fixed map has no reference image, so a bumped camera keeps counting with misaligned boxes.
3. **No alert anyone sees.** Drift only creates events in the geometry log; nothing reaches the dashboard header or a notification channel.
4. **The reference never updates.** It is taken once at calibration. Seasons, repainting and lighting can push matches down until checks are permanently `inconclusive` (safe, but nothing is watching).

## 3. Workflow: automatic detection, automatic re-align draft, manual Accept

Never apply a correction to live counting automatically.

```
ok --drift x1--> SUSPECTED (log only)
                   | drift x3 (~90 s), or x2 if shift > 3 % of width
                   v
                DRIFTED --> 1. live counting FROZEN
                            2. admin alert: banner + event (webhook/email later)
                            3. background job: reanchor -> child draft
                                 |- success -> "Review re-align" button (diff overlay)
                                 '- refused -> "Camera moved, not turned: run Camera setup"
                   | admin Accept (diff shown, drift re-checked on a fresh frame)
                   v
                ok on the new version (history scope switches; one exit/entry pair per parked car)
```

- Why not fully automatic: re-anchoring can be confidently wrong (e.g. a large truck covering half the background gives a self-consistent wrong rotation). One click is cheap; a wrong grid feeding hours of history is not.
- Why alert anyway: the alert is separate from the fix, so the operator always learns the camera moved.
- Optional later: auto-accept small re-anchors only (rotation < 1 deg, residual < 1 px, >= 80 % of bays in view, confirmed on 3 frames minutes apart). Off by default; every auto-accept is an event that can be rolled back.

### Freeze instead of fallback (fixes gap 1)

While DRIFTED keep the last good slot list and states:
- Mark every slot `UNVERIFIED` (grey, "camera moved" note).
- Open and close no history sessions, so a misaligned grid records no fake exits/entries.
- Show counts as "last known: 4 busy / 12 (camera moved 14:02)", never as live numbers.
- Never switch to the learned map while a stale calibration exists.

Small change in `live.py` `_auto_slots` and the monitor rebuild: if the feed reports stale, return the last auto grid with `frozen=True`.

## 4. Guardrails (draft layer)

- Re-check on a fresh frame at Accept time: the draft must still match the current view (shift under `DRIFT_FRAC`), otherwise reject "camera moved again".
- Show the diff in the Suggest layer UI: old bays solid, re-anchored bays dotted, plus rotation and residual.
- One-click rollback: the previous version stays `superseded`, so "Restore previous calibration" is possible.
- Audit trail: `drift_suspected`, `drift_declared`, `reanchor_drafted`, `accepted_by`, `rolled_back` (most exist).
- Live code never changes geometry state except marking stale; only the admin API activates a version.

## 5. Lighting, shadows, weather

Already handled: ORB brightness tolerance; vehicles masked; RANSAC discards moving-shadow matches; spread check (>= 30 % x 25 % of the frame); weak evidence = `inconclusive`, not `drift`.

To add, in order of value:
1. **Reference bank** of 3-6 references per calibration (day, dusk, night IR, overcast), captured automatically on an `ok` check with high inliers under conditions not yet banked; each check uses the best match. Fixes night `inconclusive` and seasonal decay (gap 4).
2. **CLAHE** before ORB: one `cv2.createCLAHE(2.0, (8, 8))` in `_work`. Helps IR night frames and hard noon shadows.
3. **Persistent shift:** the 3 confirming shifts must agree (spread < ~30 % of their median). Real bumps repeat; shadows and sway vary.
4. **"Blinded" state:** `inconclusive` for > 30 min in daylight (by mean brightness) raises a low-priority "camera view degraded" alert (rain, dirt, spider web, camera facing a wall). Silent today.
5. **Wind-sway guard** for pole cameras: small oscillating shift is jitter; the median over the strikes handles it. Do not lower `DRIFT_FRAC` below ~0.4 %.
6. **Reflective-area mask** (wet tarmac, glass), optional, drawn at setup.

## 6. Fixed-mode cameras (closes gap 2)

- On saving a Fixed map, also store `extract_reference(still, vehicle_boxes)` with it.
- The worker runs `drift.compare` on the same 30 s cycle.
- Re-align is 2-D: fit a homography from the matched background points (`compare` already computes one) and transform the slot polygons. Result is a draft Fixed map through the same Accept flow. No camera model needed; about 80 % of the value for camera-2.

## 7. Build order (step by step)

1. **Freeze instead of fallback + dashboard drift banner.** Small, protects a demo. Tests: stale calibration keeps the last grid, no sessions opened/closed, counts labelled last-known.
2. **Drift guard on Fixed maps:** reference stored at save time; homography-based re-align drafts.
3. **Auto-run reanchor on DRIFTED** + "Review re-align" button + fresh-frame re-check at Accept.
4. **Reference bank + CLAHE + persistent-shift check + blinded alert.**
5. **Optional:** auto-accept of small re-anchors; webhook/email alerts.

## 8. Rules for implementation

- Take a snapshot first: `& .\scripts\snapshot.ps1 -Label drift-step<N>-pre`; roll back with `scripts\restore.ps1`.
- Additive only (no new SQL migrations; new fields inside existing record JSON).
- Tests never touch `./data` or the live DB. Streamlit (`app.py`, `demo_app.py`) stays untouched.
- Deploy only with the user's approval via `docker compose up -d --build`.
- Log every change in `ACTIVITY_LOG.md`.
- Hold all deploys until after the demo unless the user says otherwise.

## 9. Link to Night Vision

Step 4 here (CLAHE before ORB, day / dusk / night reference bank) reuses the light probe from `docs/night_vision_roadmap.md` to choose the reference. The drift guard always works on the original frame, never the enhanced detection copy.
