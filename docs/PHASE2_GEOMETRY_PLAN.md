# Phase 2 Metric Geometry: blueprint v2 (Admin SAM setup, zero runtime overhead)

**Planning document only. No code is written until execution starts tomorrow.**

## Context

At the live demo, stakeholders asked for:
- real-world distances;
- total parking surface area in m²;
- a dynamic capacity with occupied and free counts;
- all of it on any camera without manual geometric tuning.

Today everything works in pixels: area cuts by fraction, slots learned in pixels (Dynamic mode) or drawn by hand (Fixed mode).

Blueprint v1 learned the parking surface from occupancy history. That fails for an **empty lot on Day 1**: with no cars, there is no surface and no capacity.

This v2 adopts the Admin decision:
- **SAM 2 / SAM 3 runs once, at camera onboarding**, from one click on the ground. It produces the parking-surface polygon.
- **GeoCalib** gives camera intrinsics and orientation; **vehicle priors** give metric scale.
- From these, `pack_capacity` computes area and pre-renders FREE slots **on Day 1**.
- At runtime, the worker stays on **yolov8s + a 3×3 homography**. SAM is never loaded there.

---

## 1. Architectural flow

### 1a. Process boundaries: what runs where

| Component | Runs in | Loaded models | When |
|---|---|---|---|
| **Setup job** (new) | A separate on-demand compose service `setup` (profile `setup`), started by the API, exiting when done | SAM 2.1-small or SAM 3, GeoCalib, Depth Anything V2-Small metric | Onboarding, re-calibration, camera moved |
| Worker (existing `scripts/supervisor.py`) | `worker` | yolov8s only, as today | Every frame |
| API (existing) | `api` | none | Always |

Running setup in its own service keeps SAM, GeoCalib and Depth weights and their extra dependencies **out of the worker and api images**. That keeps runtime memory, image size and build time unchanged, and keeps the existing 185 Python tests and build path intact. If the setup service is absent, everything still works and Phase 2 metrics show "not calibrated".

### 1b. Onboarding sequence (one camera)

1. **Admin opens "Camera Setup"** in the SPA. The page reuses the Slot Editor's snapshot flow (`POST /api/fixed-slots/snapshot`: ISAPI, then go2rtc, then the raw frame), so the Admin works on a frozen still.
2. **The Admin clicks once on the parking ground.** Optionally:
   - extra clicks (+ include, − exclude: kerbs, buildings, landscaping);
   - SAM 3 only: a text prompt, "parking lot", for a no-click proposal.
3. **SAM pass, on the setup service.** From the click it produces a pixel-perfect ground mask, plus a confidence score per mask.
   - The mask is cleaned: holes from parked cars are filled, the largest component is kept, and a polygon is simplified from it.
   - Parked cars in the still would otherwise punch holes in the mask. YOLO boxes on the same still are passed to SAM as "include" hints and then filled, so occupied bays still count as surface.
4. **Admin review.** The polygon is shown on the still; the Admin can nudge vertices, then **Accept**. Only the accepted polygon is persisted.
5. **`geometry/calib.py` (setup service):**
   - GeoCalib on 5–20 stills spread over a few minutes gives the median focal length, roll, pitch and lens distortion.
   - **Scale (camera height h).** Every other geometric quantity is known, so h is the only unknown, and the system picks it by priority:
     1. **Vehicle prior**, when at least 3 cars are visible: fit a cuboid of 4.5 × 1.8 × 1.5 m and take the median h.
     2. **Painted bay lines**, on an empty lot: detect lines inside the SAM mask in BEV (LSD/Hough), take their spacing, and match it to the configured bay width, 2.5 m by default.
     3. **Depth Anything V2 metric**: fit a ground plane with RANSAC over the SAM-mask pixels. These pixels are clean ground thanks to SAM, which makes this far more reliable than v1.
     4. **Manual fallback**: type the distance between two clicked points.
   - At least two of these sources are used when available. Confidence is HIGH if they agree within 15 %, MEDIUM if only one source is available, and LOW if they disagree (a UI badge shows it).
   - The result is K, distortion, R, h, and **H** (image → ground plane in metres).
6. **`geometry/bev.py`:**
   - It projects the SAM polygon through H into a BEV polygon in metres.
   - It clips at the "horizon guard": ground beyond a range limit (default 60 m), or where one pixel covers more than 25 cm, is excluded as unmeasurable.
   - It computes **surface area in m²**.
   - It also exposes the runtime helpers: pixel → metre and car-box → ground footprint.
7. **`geometry/capacity.py`:**
   - **Bay orientation**, in order of preference: painted lines, then the PCA axis of parked cars, then the long axis of the BEV polygon.
   - It runs `pack_capacity`: rows of bays (2.5 × 5.0 m) with aisles (6 m, configurable) inside the polygon, giving the **geometric capacity**. The bays are back-projected to pixel polygons as **pre-rendered slots**.
   - **Day 1 result:** total area in m², maximum capacity, and N slots, all FREE on an empty lot.
8. **Persist and publish.** The calibration and slots are written to LiveStore as a new version (section 2). The worker picks them up the same way it hot-reloads settings today, with no restart.

### 1c. Runtime (worker, every frame): unchanged cost

The worker runs yolov8s exactly as today: whole frame, near pass, split, zoom. Then:

- `bev.annotate` adds `footprint_m` and `centroid_m` per vehicle. This is a few homography multiplies.
- Occupancy of the pre-rendered slots reuses the existing **Fixed-slot matching** (`fixed_slots.match_score`: point rule plus coverage). The packed slots are simply a machine-generated slot map, so no new occupancy logic is needed.
- Distances: shapely edge-to-edge between footprints, only for nearest neighbours.
- Counts: busy / free / unknown, with **UNKNOWN still its own bucket**.

### 1d. Mode policy (respects today's camera setup)

| Mode | Behaviour |
|---|---|
| **Fixed** (camera-2) | Admin-drawn slots stay authoritative. Packed slots appear only as an optional "suggested" layer and become real only on Admin "Adopt". |
| **Dynamic** (camera-1) | Packed slots seed the slot map on Day 1 so FREE bays exist before any car is seen. Learned slots then refine them: a learned slot that overlaps a packed bay replaces it. |
| **None calibrated** | Exactly today's behaviour. |

### 1e. Ongoing correctness

- **Drift guard:** every 60 s, the worker matches ORB features against the reference still stored with the calibration, with car boxes masked out. If the inlier reprojection error exceeds 3 px:
  - the calibration is marked STALE and metrics are hidden;
  - the Admin gets a "camera moved: re-run setup" prompt.
  
  SAM is not re-run automatically, because it is Admin-gated by design.
- **Optional usage refinement:** confirmed footprints build up an occupancy heatmap in BEV. If cars regularly park outside the SAM polygon, the Admin is shown a suggestion. The polygon is never changed silently.

---

## 2. API and data-flow strategy

### 2a. LiveStore: new `geometry` section per camera, versioned

Each camera holds a list of versions plus an `active` pointer:

- **`version`**: an integer that increases monotonically, same convention as `settings version N`.
- **`created_at`, `created_by`**: who and when (from the API token role).
- **`source_still`**: hash and timestamp of the snapshot used. The still itself is kept as a JPEG file under the data directory, referenced by path.
- **`surface`**:
  - `mask_polygon_px`, normalised 0–1 coordinates so it survives resolution changes;
  - `sam_model`, the name, version and checkpoint hash;
  - `clicks`, the click and label list, so the mask is reproducible;
  - `mask_score`;
  - `admin_edited` (bool).
- **`calibration`**:
  - `K`, `dist`, `roll`, `pitch`, `h`, `H` (3×3), `image_size`;
  - `scale_sources`: each source's h estimate;
  - `confidence`: HIGH, MEDIUM or LOW;
  - `geocalib_version`.
- **`bev`**: `surface_polygon_m`, `area_m2`, `horizon_limit_m`.
- **`capacity`**: `bay_size`, `aisle`, `orientation_deg`, `orientation_source`, `geometric_capacity`, `packed_slots`, with each slot's polygon in pixels and in metres.
- **`reference_features`**: ORB keypoints and descriptors for the drift guard, stored as a compact blob.
- **`status`**: draft, active, stale or superseded.

### 2b. Versioning rules

- Setup writes a **draft**; only Admin Accept makes it **active**. The previous active version becomes **superseded** and is kept for rollback, keeping the last 5.
- Activation bumps the camera's config version, which the worker already watches. The worker reloads H and the packed slots atomically.
- **Rollback:** setting `active` to an older version is one API call.
- The PostgreSQL and SQLite backends both store this the same way as the existing config and slot map, so tests can use temporary stores and never `./data`.

### 2c. API endpoints

All of these are Admin-token only; reads also allow the view token:

| Endpoint | Purpose |
|---|---|
| `POST /api/cameras/{id}/setup/still` | Reuse the snapshot capture and return the still id. |
| `POST /api/cameras/{id}/setup/segment` | Input: still id plus clicks (or a text prompt). Starts the setup job and returns a job id. Output: the mask polygon and its score. |
| `POST /api/cameras/{id}/setup/calibrate` | Input: still ids (plus an optional manual reference distance). Runs GeoCalib and scale fusion, then BEV and packing, and returns a draft version id. |
| `GET /api/setup/jobs/{job}` | Progress and result. Uses the same background-task pattern as `start_detection` / `asyncio.to_thread`, with the work delegated to the setup service. |
| `POST /api/cameras/{id}/geometry/{version}/activate` | Accept, or roll back. |
| `GET /api/cameras/{id}/geometry` | The active version: summary, confidence, status. |
| `GET /api/metrics/capacity?cam=` | surface_m2, geometric capacity, busy, free, unknown, confidence, stale flag. |
| `GET /api/metrics/distances?cam=` | Nearest-neighbour pairs with distance in metres ± σ. |

**Transport between the API and the setup service:** HTTP on the internal compose network (as `http://api:8000` is used today), or a job row in the DB polled by the setup service. Recommended: **HTTP plus job rows**, so a crashed job is still visible.

### 2d. Security and data hygiene

- Stills and masks stay on the host. No external upload.
- Model weights are pinned by version and hash, all at least 2 weeks old.
- Camera credentials are never written into geometry records or logs.

---

## 3. Task checklist for tomorrow's rollout

The demo freeze still applies: nothing touches the live stack until the setup service and flag are proven offline.

**A. Prep (morning)**
1. Pick SAM 2.1-small or SAM 3. Recommended: SAM 3 if its pinned release is at least 2 weeks old and runs on CPU within about 30 s per still; otherwise SAM 2.1-small.
2. Pin the versions and hashes of the SAM, GeoCalib and Depth Anything V2-Small metric weights.
3. Define the `geometry` LiveStore schema and its migration for PostgreSQL and SQLite; empty by default.

**B. Offline build and validation (scratchpad, read-only on live data)**
4. Check the SAM click → mask quality on saved camera-1 and camera-2 stills. Car-hole fill must work.
5. Check that GeoCalib plus the scale sources agree within 15 % on camera-1 and camera-2. They are the same physical camera, so their geometry must be consistent.
6. Check the BEV area and `pack_capacity` against a visual bay count on the still, including a synthetic "empty lot" test with the cars masked out.
7. Write synthetic unit tests: a known K, R, h and a rendered grid. Recovered h and distances must be within 5 %, and the packing count must be exact on rectangles.

**C. Integration (behind the feature flag `geometry.enabled = false`)**
8. Add the `setup` compose service (profile; not started by default).
9. Add the setup API endpoints and the job handling.
10. Worker: load the active geometry, run `bev.annotate`, add packed slots to Fixed-slot matching, and run the drift guard. When the flag is off, nothing is loaded.
11. SPA: the Camera Setup wizard (still → click → review mask → calibrate → review slots/area → Accept), a confidence badge, area and capacity cards (with `useHeld` smoothing), a BEV minimap, and a distance overlay toggle in `OverlayLayers`.
12. Streamlit stays untouched; the parity test must pass.

**D. Go-live gate**
13. Full suite green (185+ Python, 59+ frontend), plus the new tests.
14. Replay check on saved frames: camera-1 counts must not regress against the validated item 67–75 results.
15. Onboard camera-2 (Fixed mode) first, with the suggested-slots layer only. Then camera-1 (Dynamic mode).
16. On-site acceptance: one tape-measured distance, as a check only, within ±10 %.
17. Log everything in ACTIVITY_LOG.md (item 76+).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Empty lot with no painted lines: only depth gives scale | MEDIUM badge; Admin may enter one reference distance; auto-upgrade to HIGH once ≥ 3 cars are seen |
| Non-planar ground (ramps) | Validation spread check gives LOW; per-zone planes are a later phase |
| Far field near the horizon is inaccurate | Horizon guard excludes it from area and capacity |
| SAM mask includes the roadway or sidewalk | Admin review plus − clicks; nothing becomes active without Accept |
| Packed bays differ from real markings | Painted-line orientation preferred; Dynamic learned slots override; Fixed mode keeps Admin slots authoritative |

## Verification

- Steps 4–7 and 13–16 above.
- Plus: with the flag off, the worker's per-frame time and memory are identical to today (measured).
- With the flag on, the added per-frame cost should be under 2 ms.

## Sample test: zones (items 150-151)
A lot whose rows or blocks lie at different angles or depths does not fit one ground surface with one bay lattice. The sample test therefore has a
"Zones" layout: each zone is a drawn outline; SAM finds the parking ground inside it (cut to the outline), then GeoCalib (or the ground pins of step 1)
calibrates that zone alone, on its own mask, and `build_record` packs that zone's own grid at its own angle. The bays of every zone are merged into one list
for one occupancy run; a bay of a later zone lying on a bay of an earlier one (IoU >= 0.3, overlapping outlines) is kept once. (Item 150's manual
first-bay homography was replaced by this at the user's request.)
Item 152: only the zone's own vehicles reach its calibration, and a zone's bays are not packed by the car-anchored lattice but laid from the outline's
enclosing rectangle on the ground (row direction and length from the drawn zone, whole bays filling it, numbered along the row); "Add next N" steps more
bays on the ground at the row's spacing, and the bays can be moved / reshaped before the run.
