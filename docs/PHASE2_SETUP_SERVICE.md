# Phase 2 camera setup: the isolated setup service and its API

Status: built and tested, **not enabled** (flag off, service not started, no deploy). Plan: `docs/PHASE2_GEOMETRY_PLAN.md`.

## What runs where

| Piece | Where | Heavy libraries | When |
|---|---|---|---|
| `setup` service (SAM 2.1 small, GeoCalib, Depth Anything V2 metric small) | own image `stomeetings/carparking-setup`, compose profile `setup` | torch, transformers, kornia, sam2, geocalib, the weights | only while an Admin sets a camera up |
| Setup endpoints + jobs | the `api` service (`api/geometry_routes.py`) | none (numpy / cv2 / httpx) | only with `CARPARK_GEOMETRY_ENABLED=1` |
| Surface, scale, bird's-eye view, packing | `src/carparking/geometry/` | none (numpy / cv2) | inside a calibrate job |
| Drafts | `carparking/geometry_store.py` (SQLite per camera / PostgreSQL) | none | written by a calibrate job |
| worker | unchanged | none new | never loads any of this |

The main image's `.dockerignore` lets `setup/` into no build context, and the setup image has its own context
(`setup/Dockerfile.dockerignore`): the worker / api / ui image cannot contain the models, `transformers` or the weights.

## Pinned versions (`setup/manifest.json`)

| Model | Code pin | Weights |
|---|---|---|
| SAM 2.1 Hiera Small | facebookresearch/sam2 `2b90b9f` (2024-12-16) | `sam2.1_hiera_small.pt`, sha256 `6d1aa6f3...`, 184,416,285 bytes |
| GeoCalib (pinhole) | cvg/GeoCalib `97b8968` (2026-06-28) | `geocalib-pinhole.tar`, sha256 `86d6aeac...`, 116,074,121 bytes; weights licence CC BY 4.0 (credit the authors) |
| Depth Anything V2 Metric-Outdoor Small | HF revision `fd2c220` (2024-08-27) | `model.safetensors`, sha256 `ad065c77...` (the value Hugging Face publishes), 99,173,660 bytes |

The build downloads each file and **fails on any size or sha256 mismatch** (`setup/fetch_weights.py`). Python packages are
pinned in `setup/requirements.txt` (all at least 14 days old on 2026-10-23). SAM 3 was not used: its weights are gated
(licence acceptance on Hugging Face), which an unattended build cannot do.

## Using it

```
# once: build (downloads ~400 MB of weights, verifies them, runs the service's contract test)
docker compose --profile setup build setup
# per setup session
#   1. CARPARK_GEOMETRY_ENABLED=1 in .env, then `docker compose up -d api`   (the flag is read at API start)
#   2. docker compose --profile setup up -d setup
# ... set up the camera ...
docker compose stop setup           # and CARPARK_GEOMETRY_ENABLED=0 again if wanted
```

Endpoints (admin token; 503 while the flag is off):

| | |
|---|---|
| `POST /api/cameras/{id}/setup/still` | a still (camera snapshot, go2rtc, worker frame) kept in memory; no detection started |
| `POST /api/cameras/{id}/setup/segment` `{still, clicks:[{x,y,label}]}` (x, y normalised; label 1 = ground, 0 = not) | job: SAM 2.1 -> candidate masks -> polygon (cars on the surface filled in) |
| `POST /api/cameras/{id}/setup/calibrate` `{still, polygon, extra_stills?, reference?, bay_width?, bay_length?, aisle?, meta?, note?}` | job: GeoCalib + depth (service), vehicle scale, fusion, bird's-eye surface, area, packed bays -> a **draft** version |
| `GET /api/setup/jobs/{job}` | `{state: queued/running/done/error, stage, progress, result, error}` |
| `POST /api/cameras/{id}/setup/activate` `{version, accept_unverified?}` | **Accept**: a draft (or an earlier version: rollback) becomes the active calibration, in one transaction. A version whose scale was not anchored by a typed reference (confidence below HIGH) needs `accept_unverified`; stale 409; unknown 404 |
| `DELETE /api/cameras/{id}/setup/draft?version=N` | discard one draft (without `version`: every draft); an active, earlier or stale version is refused 409 |
| `GET /api/cameras/{id}/setup/geometry` / `.../geometry/{version}` | the versions as summaries + the active one / one version in full (the surface and the packed bays) |
| `GET /api/setup/status` | is it switched on, does the setup service answer (admin; answers while the flag is off and then contacts nothing) |
| `GET /api/cameras/{id}/geometry` | the ACTIVE calibration's summary (viewer token), read from the store on every call |

A draft changes nothing for the worker: it is used only after an explicit Accept (`/setup/activate`). Activation writes only the
versioned store and every reader reads it per call, so it takes effect on the next read with nothing restarted; the worker does not
read geometry yet (a later, flagged step that polls `GeometryStore.active_version()`). Jobs live in the API's memory (the last 50; an API
restart forgets them, a finished draft stays in the store).

## Scale sources and confidence

The camera height (metric scale) comes from, in order of trust: the Admin's typed distance between two ground points
(`reference`), the vehicles in the still (roof / contact rows of at least 3 whole cars, 1.45 m car height), the metric
depth of the SAM mask's pixels. Two or more that agree within 15 % = HIGH; one = MEDIUM; disagreeing = LOW (a warning
lists them). With none the job fails and asks for a reference distance.

## The setup page (SPA, `#/setup`, "Camera setup")

Five steps: **1 Still** - **2 Known distance** (asked for up front: click the two ends of a distance you know and type the metres:
a car's wheelbase, a bay width from the plan, a tape measure; step 3 stays locked until it is given or the Admin explicitly goes on
without, which later needs an extra confirmation to accept) - **3 Parking surface** (ground / not-ground clicks, SAM outlines it) -
**4 Calibrate** (lens model) - **5 Review & accept** (confidence badge, height, tilt, surface, bays drawn over the still, warnings;
Accept / Discard). Below: every version of the camera, with Activate, Roll back and Discard. Flag off or setup service down: the page
says so and shows the command. Verified in Chrome with the real models: HIGH, h 3.24 m, 2 bays on cars, accepted, read back at once.

## At runtime: the worker and the dashboard (all behind CARPARK_GEOMETRY_ENABLED)

With the flag on, each camera's worker (`src/carparking/geometry_live.py`) looks every 5 s for the camera's ACTIVE calibration
(one cheap query, `GeometryStore.active_version()`; the record is loaded only when the version changed) and, while there is one,
adds `geometry` to its snapshot: the measured vehicles' ground footprints, the nearest-neighbour distances (vehicle to vehicle,
and people / objects to the nearest vehicle), and the packed bays marked FREE / BUSY with capacity / occupied / free counts.
No active calibration, or any error here: the snapshot is exactly what it always was (a failing store is logged once and the last
good calibration kept for a minute). Accepting on the setup page is picked up by the running worker within seconds: no restart.

The dashboard shows it as a `Metrics` layer over the video (footprints, bay outlines, distance lines with `≈` unless the scale was
anchored by a typed reference) and a "Measured capacity" card beside the slot counters, which count exactly as before. With the
flag off the API strips `geometry` from the stream and the SPA renders nothing.

Honest limits: cars cut by the frame edge or beyond ~18 m (6 cm of ground per pixel) are counted, not measured; distances are
edge to edge between footprints modelled from the boxes (about +-0.3 m even when anchored); the metres carry the calibration's
confidence (about +-25 % unless a known distance anchored it).

## Measured occupancy on the Analytics page (behind CARPARK_GEOMETRY_ENABLED)

The Analytics page shows a "Measured occupancy" section above the legacy charts when the camera has metric history: average and peak
bay utilization, the bays available over time and the average fullness by hour of day for the selected range, plus the live metric counts
(surface, occupied / bays, vehicles measured). It reads `GET /api/cameras/{id}/geometry/history` (viewer token). With the feature off
(503), no calibration or no samples the section is simply absent and the legacy analytics are unchanged.

## Re-anchoring after a camera move (behind CARPARK_GEOMETRY_ENABLED)

When the drift guard marks a calibration stale (or the Admin knows the camera was re-aimed), the full setup is not needed if the camera
only turned: the optics, mounting height and scale are unchanged, so the calibration can be carried over to the new pose.

1. **Take a fresh still** (Setup page -> "Re-anchor calibration" on the banner, or the Re-anchor action of a version).
2. `POST /api/cameras/{id}/setup/reanchor {still, from_version?, note?}` (admin token) starts a job. It matches the stored background of
   the parent calibration to the still, solves the rotation (Kabsch + RANSAC, lens undone first) and writes a **draft** linked to its
   parent: new pitch / roll / heading, the homography H, the bays and surface in the new frame, the new background reference. No SAM,
   no depth, no setup service.
3. **Review**: turn in degrees, matches and residual, bays in view, the new bays on the new still.
4. **Accept** (`/setup/activate`, same rules as any version): becomes active, the stale condition is gone, the worker loads it on its next
   check (5 s) with a fresh drift baseline. The stale parent stays in the history; events: `reanchored`, `activated` (with `parent`).

Quality gates (the job fails with the reason and nothing is written): a stored reference is needed (calibrations made before the drift
guard have none), 25+ and 40 %+ of the matches must agree on one rotation, they must be spread over the picture, the residual stays under
2 px, the turn is at most 25 degrees, the camera stays tilted down, and at least half the bays stay in view. A shift, a zoom or a different
scene fail these gates; the answer is then the full setup. A draft is also refused at accept time when the camera has moved again since
(the worker's latest frame no longer matches the draft's own reference). Measured on camera-1's real frame turned by known rotations through
its own lens: rotation within 0.01 degrees, bay corners within 0.1-0.7 px (ACTIVITY_LOG item 83).

## The drift guard and the metric history (all behind CARPARK_GEOMETRY_ENABLED)

**Drift guard.** A calibration is only true for the picture it was made on. The calibrate job stores the still's static background
(ORB features, vehicles and the camera's date / caption strips left out) with the version. While the version is active the worker
compares each new frame's background (detected vehicles and people masked) with it every 30 s: the matches that agree on one
movement are kept and the 90th-percentile displacement of those points, as a share of the width, is the drift (above 0.6 %, about
11 px at 1920, counts as moved). Three confirmed checks in a row mark the version **stale** (an event in the log and the worker log),
and the camera falls back to the legacy snapshot until a new calibration is accepted; the setup page then says "the camera moved".
A check that cannot be judged (night, glare, other light, too few or badly spread matches) is *inconclusive* and never an alarm: the
guard then simply cannot vouch for the calibration. Versions made before the guard have no stored reference: the first frame the worker
sees after loading becomes the baseline (weaker, labelled). Tuned on real frames of the live camera (see ACTIVITY_LOG item 82).
Tunables: `CARPARK_DRIFT_EVERY` (30 s), `CARPARK_DRIFT_CONFIRM` (10), `CARPARK_DRIFT_GUARD` (off by default; the dashboard's Settings switch "Camera Drift Guard" turns the checks on or off live, without touching the active version).

**History.** The same stores (SQLite per camera / PostgreSQL) keep the calibration's lifecycle (`draft`, `activated` with the
version it replaced and whether it was a rollback, `stale` with the reason, `discarded`, `drift_suspected`, `drift_cleared`), written in
the same transaction as each change, and an occupancy sample (capacity / occupied / vehicles / measured) the worker adds every minute and
whenever the occupancy changes. Samples are kept 90 days, events a year. Range queries are bucketed in the database over the
(camera, time) key: `GET /api/cameras/{id}/geometry/history?hours=24&points=240` (up to 90 days, buckets of at least a minute) and
`GET /api/cameras/{id}/geometry/events?limit=50&since=` (viewer token). The dashboard card shows the guard's last verdict; the setup page
shows the chart, the event list and the banner.

## Lens distortion and the real-camera evaluation (2026-10-23, camera-1 "Front Parking 02" and camera-2, the same camera)

Wide CCTV lenses bend the image, and GeoCalib's pinhole model reads this camera as a 20 deg lens tilted 4.8 deg (wrong). The
calibrate job now defaults to GeoCalib's `simple_radial` model (`camera_model`: `pinhole` / `simple_radial` / `radial`) and every
pixel is undistorted (exact inverse of GeoCalib's forward model, `geometry/camera.py`) before K and the ground homography.

| | camera-1 (10-23) | camera-2 (10-21) |
|---|---|---|
| f / k1 / field of view / tilt | 1986 px / -0.46 / 59 deg / 13.8 deg | 1895 px / -0.39 / 61 deg / 14.1 deg |
| camera height from vehicles / depth | 2.67 / 2.48 m | 2.97 / 2.63 m |
| with a typed reference (own car's wheelbase) | h 3.24 m | h 3.36 m |
| the OTHER camera's car through this calibration (held out) | Celerio 2.37 m vs 2.425 (-2.1 %) | HR-V 2.62 m vs 2.61 (+0.5 %) |

**Scale: priors are consistent, not accurate.** Vehicle sizes and monocular depth agree within 8-12 %, yet an independent check
with two cars of known wheelbase (HR-V 2.61 m, Celerio 2.425 m) puts both 17-21 % short (the camera is at ~3.3 m, the priors say
~2.7); the vehicle estimate alone scatters +-15-19 % between cars and by 20 % between two stills. So the confidence is:
HIGH = a typed reference distance (+-5 %; `reference`: two ground points and the metres: a known wheelbase, a bay width from the
plan), MEDIUM = two agreeing priors (expect +-25 %), LOW = one prior or disagreeing priors. Typed reference + the lens model gave
held-out errors of 0.5-2 %. Pass `extra_stills` (more stills of the same camera) to pool the cars.

**The surface is a ground raster, not the polygon.** A SAM polygon over a car's body maps to ground metres behind the car, so the
body is cut out and the car's footprint (side-on 2.2 m deep, end-on 4.6 m) added; thin strips (< 1.2 m) and specks (< 3 m2) are
dropped; the bays follow the cars' direction, car bays first (one per car, neighbours touching), free bays packed around them.
Cars farther than ~18 m (> 6 cm of ground per pixel) get no bay: on this low, shallow camera only the near cars can be measured.

## Known limits

- Single plane: a ramp or uneven ground breaks the metres (the cars' height spread warns above 20 %).
- Orientation of the packed bays comes from the surface polygon's longer side; painted-line detection is a later step.
- GeoCalib's radial models assume the principal point at the image centre.
- Far field: beyond ~18 m (this camera) the metres are not reliable; the draft says how many cars were skipped.
- A pole or pillar can split the surface; the largest of the near-top SAM candidates is used (all are returned).
- The packed slots are a geometric proposal; Fixed mode keeps the Admin's slots authoritative.
