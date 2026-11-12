# Phase 2 (metric geometry): deployment readiness, deployment, first calibration, rollback

Status 2026-10-24: everything of Phase 2 (ACTIVITY_LOG items 76-84) is built and tested but **not deployed**. The live stack
(`carparking-*`) still runs the image built on 2026-10-22 and is untouched (demo freeze). Nothing below is to be run until the freeze
is lifted.

## 1. Audit result

| Check | Result |
|---|---|
| Python suite (`/tmp/runtests.sh`: pytest + pyflakes in a throwaway container, project mounted read-only) | 347 passed, 23 skipped (the PostgreSQL variants), pyflakes clean |
| PostgreSQL variants (store, history, guard) against a throwaway postgres:17.7 | 50 passed, none skipped (run after item 83; the later change was frontend only) |
| Frontend (vitest + `tsc`) | 115 passed, `tsc` clean, production build ok |
| Streamlit untouched | `tests/test_settings_parity.py` is in the green suite; `app.py` / `scripts/demo_app.py` unchanged |
| `docker compose config` | valid; services db, api, worker, ui, sync, go2rtc and the on-demand `setup` (profile `setup`) |
| Migrations | two additive PostgreSQL migrations, `004_camera_geometry` and `005_geometry_history`, run once by the registry at api start and tracked in `schema_migrations`; empty tables, no existing table is altered. The SQLite side creates its tables on first use, only when the flag is on. Nothing is pending: they apply by themselves on the first start of the new image |
| Uncommitted changes | **the project is not a git repository**, so "uncommitted" cannot be checked. The rollback anchor is the image: the running image (`3ebbb785c7cf`) is also tagged `stomeetings/carparking:pre-phase2` (done 2026-10-24, a local tag only) |

### Environment variables (.env.example vs the code)

Every `CARPARK_*` name the code reads is documented in `.env.example` and the reverse; the three Phase 2 ones are commented out because
their defaults are the intended ones. api, worker, ui and sync all get `.env` through `env_file`, so one line in `.env` reaches the api and
the worker (both read it). The `setup` service gets no `.env` (no tokens, no camera passwords).

| Variable | Default (unset) | Meaning |
|---|---|---|
| `CARPARK_GEOMETRY_ENABLED` | off. Only `1`/`true`/`yes`/`on` switch it on | the one switch: setup / re-anchor / history endpoints (else 503), the worker's metric feed, drift guard, history samples, the SPA metric views |
| `CARPARK_SETUP_URL` | `http://setup:8100` | where the api reaches the on-demand setup service (compose network) |
| `CARPARK_DRIFT_GUARD` | off. Only `1`/`on`/`true`/`yes` switch it on | the drift checks' default while the dashboard's Settings switch ("Camera Drift Guard", worker option `drift_guard`, applied live) was never set. Off: the worker never stales a calibration (no checks, no drift events); the active version, its bays, the metrics and the history are untouched |
| `CARPARK_DRIFT_EVERY` | 30 (seconds) | how often the worker checks the camera did not move |
| `CARPARK_DRIFT_CONFIRM` | 10 (was 3 until 12 Nov 2026) | confirmed checks in a row that mark a calibration stale (10 × 30 s = 5 minutes of continuous drift; 3 staled Camera 1 on 16 px of morning-light sway at the 15 px threshold) |

Fixed in code (not env): drift threshold 0.6 % of the picture width (about 11 px at 1920), history samples once a minute and on a change,
kept 90 days, events 1 year, the worker re-reads the active calibration every 5 s.

Finding fixed during the audit: the `.env.example` text said to set the flag "for the api service"; the worker needs it too (it already
does, through `env_file`). The wording now says `.env`.

## 2. Deployment (one command, after the freeze)

Impact: `up -d --build` recreates api, worker, ui and sync (a few seconds without video / API; the cameras reconnect, history is
unchanged). The build runs the whole test suite; a failing test aborts it and the old containers keep running.

1. Safety copies (read-only on live data):
   - `docker tag stomeetings/carparking:local stomeetings/carparking:pre-phase2` (exists already; repeat only if the image changed since)
   - `docker compose exec db pg_dump -U carparking carparking > pg-before-phase2.sql`
   - copy `data/parking_history.db` while the worker is idle, or use SQLite's backup API; keep `data/config` as is.
2. Switch it on: add `CARPARK_GEOMETRY_ENABLED=1` to `.env` (leave the other Phase 2 variables unset).
3. **The single command:** `docker compose up -d --build`
4. Verify (no side effects):
   - `docker compose ps`: all healthy; `docker compose logs worker | grep geometry` shows the feed started (and "CARPARK_GEOMETRY_ENABLED is on but no CARPARK_CAMERA_ID" is NOT present).
   - Settings page "Camera setup" opens; its status card says the feature is on and the setup service is not running yet.
   - Dashboards and counts behave as before: an uncalibrated camera is exactly the legacy view.

With the flag on and no calibration accepted nothing changes for the cameras: no metrics are shown and nothing is written except the
(empty) tables.

## 3. First calibration (interactive, per camera, about 5 minutes)

1. `docker compose --profile setup up -d setup` (the image `stomeetings/carparking-setup:local` is built; a first build downloads and
   sha256-verifies about 400 MB: `docker compose --profile setup build setup`). The first segment request loads SAM (up to a minute).
2. SPA, **Camera setup**, pick the camera, enter the admin token when asked.
3. **Still**: take a still with the camera in its final position and normal light (cars parked help the scale, not required).
4. **Known distance**: click the two end points of a distance you really know (a car's wheelbase, a bay's width, a taped length) and type it in
   metres. This makes the scale HIGH (about +-5 %); without it the calibration is MEDIUM/LOW (about +-25 %) and needs an extra confirmation.
5. **Parking surface**: click on the open ground (red clicks for what is not parking), "Find the surface", pick the candidate that fits.
6. **Calibrate**, then **Review & accept**: check the bays on the still, the camera height and tilt, the confidence badge; **Accept and
   activate**. The worker picks it up within 5 s: the dashboard then shows the metric overlays and counts, and the drift guard starts.
7. `docker compose stop setup` (it is only needed while calibrating).

Do camera-2 (Fixed mode) first if following the plan's order, then camera-1 (Dynamic).

After a camera is bumped: the stale banner offers **Re-anchor calibration** (a fresh still, no setup service, no typed distance); if it
refuses (too little background, a shift or zoom), do the full setup again.

## 4. Rollback (quickest first)

| What went wrong | Action | Restart? |
|---|---|---|
| A calibration looks wrong | Camera setup, versions table: **Roll back** to the previous version (or the camera simply stays on the legacy view if none was accepted); discard drafts | no |
| Metrics are off for a camera but the rest is fine | nothing is needed: a stale or missing calibration already falls back to the legacy snapshot | no |
| Phase 2 as a whole is unwanted | set `CARPARK_GEOMETRY_ENABLED=0` (or remove the line) in `.env`, then `docker compose up -d --no-build`; every Phase 2 endpoint answers 503, the worker loads nothing, the SPA shows no metric views, the Analytics page is legacy only | api, worker, ui, sync recreated |
| The new image misbehaves | set `CARPARK_TAG=pre-phase2` in `.env`, then `docker compose up -d --no-build` (the old image `3ebbb785c7cf`); remove the line again to return to `local` | api, worker, ui, sync recreated |

Data: the Phase 2 tables stay in place after any rollback; they are additive and the old image ignores them, so no database rollback is
needed. Only if you want to drop them: restore `pg-before-phase2.sql` into a scratch database and compare first (ask before deleting).
