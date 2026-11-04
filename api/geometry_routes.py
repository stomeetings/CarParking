"""Phase 2 camera setup endpoints (docs/PHASE2_GEOMETRY_PLAN.md), registered by api/main.py create_app.

  POST /api/cameras/{cam}/setup/still      a still of the camera (the slot editor's snapshot flow, no detection started)
  POST /api/cameras/{cam}/setup/segment    still + clicks -> SAM 2.1 in the setup service -> the parking-surface polygon (job)
  POST /api/cameras/{cam}/setup/calibrate  still + polygon -> GeoCalib + depth (setup service), vehicle scale, fusion,
                                           bird's-eye surface, bay packing -> a DRAFT geometry version (job)
  GET  /api/setup/jobs/{job}               progress / result of a job
  POST /api/cameras/{cam}/setup/reanchor   after a camera move: carry a calibration over to the new pose by matching its stored
                                           reference to a fresh still (no SAM, no depth, no setup service) -> a DRAFT (job)
  POST /api/cameras/{cam}/setup/activate   Accept: a draft (or an earlier version: rollback) becomes the camera's active
                                           calibration; a version without a typed reference needs accept_unverified
  DELETE /api/cameras/{cam}/setup/draft    discard one draft (?version=N) or all drafts; never an active / earlier version
  GET  /api/cameras/{cam}/setup/geometry   the versions (summaries) + the active one; .../geometry/{version}: one in full
  GET  /api/setup/status                   is it switched on, is the setup service up (admin; answers while the flag is off)
  GET  /api/cameras/{cam}/geometry         the ACTIVE calibration's summary (viewer token): read from the store on every
                                           call, so an activation is seen at once, without restarting anything
  GET  /api/cameras/{cam}/geometry/history occupancy over time, bucketed in the database (viewer token)
  GET  /api/cameras/{cam}/geometry/events  the calibration's lifecycle: draft, activated, stale (drift), discarded (viewer)
  POST /api/cameras/{cam}/suggest          Suggest slots: a quick or full AI bay grid as a DRAFT (job); GET lists them,
  GET|PUT|DELETE .../suggest/{version}     reads / fine-tunes / discards one, POST .../suggest/{version}/accept applies it

All of it needs the admin token and CARPARK_GEOMETRY_ENABLED (off: 503, nothing is touched). The heavy models live only in
the on-demand `setup` service (CARPARK_SETUP_URL); this module holds none of them and works with numpy / cv2 only. A draft
changes nothing for the worker: it becomes active only by an explicit activation (carparking/geometry_store.py)."""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import time
from typing import Literal

import cv2
import httpx
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from carparking.geometry.capacity import AISLE, BAY
from carparking.geometry import drift
from carparking.geometry.drift import MIN_REFERENCE, extract_reference
from carparking.geometry.reanchor import reanchor_record
from carparking.geometry.pipeline import SetupError, build_record, usable_vehicle_boxes
from carparking.geometry.surface import mask_to_polygon
from carparking.fixed_slots import MAX_EXPECTED, FixedSlotStore, validate_slots
from carparking.geometry_live import MANUAL, ActiveGeometry, GeometryFeed, merge_pins
from carparking.geometry_store import open_geometry
from carparking.slot_suggest import (EDITED, baseline_hash, bind_pin_ids, diff, merge_into_fixed, protect_baseline,
                                     suggest_slots)
from carparking.setup_jobs import JobBoard

SNAPSHOT_FRESH = 120.0  # seconds: a worker frame this recent says where the camera is now
SEGMENT_TIMEOUT = 300.0  # s: the first request loads SAM and encodes a 1080p image on a CPU
CALIBRATE_TIMEOUT = 300.0
MAX_META = 4000  # characters of the client's "meta" kept in the record
SCORE_KEEP = 0.9  # a candidate within this share of the best SAM score is a contender for "the surface"
SURFACE_AREA = (0.01, 0.9)  # a candidate mask outside this share of the image is not a parking lot
HEX32 = "^[0-9a-f]{32}$"
NOT_RUNNING = ("the setup service isn't reachable: start it with `docker compose --profile setup up -d setup` "
               "(and build it once with `docker compose --profile setup build setup`)")


class ReanchorIn(BaseModel):
    still: str = Field(pattern=HEX32)  # a fresh still of the camera at its NEW position
    from_version: int | None = Field(None, ge=1)  # the calibration to carry over (default: the one last in service that has a reference)
    note: str = Field("", max_length=200)


class ActivateIn(BaseModel):
    version: int = Field(ge=1)
    accept_unverified: bool = False  # activate a version whose scale was not anchored by a typed reference distance


class Click(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    label: int = Field(1, ge=0, le=1)  # 1 = this is parking ground, 0 = this is not


class SuggestIn(BaseModel):
    mode: Literal["quick", "full"] = "quick"  # quick: the active calibration re-tiled on a fresh still; full: SAM + GeoCalib re-measured
    still: str | None = Field(None, pattern=HEX32)  # the slot editor's snapshot (default: a new still)
    expected: int | None = Field(None, ge=1, le=MAX_EXPECTED)  # how many bays the lot has (default: the saved count)
    clicks: list[Click] = Field(default_factory=list, max_length=20)  # full: segment again from these clicks (default: the stored surface)
    note: str = Field("", max_length=200)


class SuggestEditIn(BaseModel):
    grid: list[dict] = Field(min_length=1, max_length=200)  # the draft's bays as the operator fine-tuned them


class SuggestAcceptIn(BaseModel):
    select: list[int] | None = Field(None, max_length=200)  # a Fixed camera: the suggested bays to take (default: all)


def summary(row: dict | None) -> dict | None:
    """A stored version without its heavy parts (the surface contour, the packed bays, the matrix)."""
    if row is None:
        return None
    rec = row["record"]
    cal, cap = rec.get("calibration", {}), rec.get("capacity", {})
    return {"ver": row["ver"], "state": row["state"], "note": row["note"], "created_by": row["created_by"],
            "created_at": row["created_at"], "activated_at": row["activated_at"],
            "confidence": cal.get("confidence"), "h": cal.get("h"), "scale_uncertainty": cal.get("scale_uncertainty"),
            "reference_used": "manual" in (cal.get("scale_sources") or {}), "lens": cal.get("lens"),
            "drift_guard": (rec.get("reference") or {}).get("n", 0) >= MIN_REFERENCE,
            "reanchor": rec.get("reanchor"),  # None, or how this version was carried over from its parent (rotation, yaw, residual...)
            "suggest": {k: (rec.get("suggest") or {}).get(k) for k in ("parent", "mode", "kind")} if rec.get("suggest") else None,
            "area_m2": rec.get("bev", {}).get("area_m2"), "capacity": cap.get("geometric_capacity"),
            "bays_with_cars": cap.get("with_cars"), "warnings": rec.get("warnings", [])}


class SegmentIn(BaseModel):
    still: str = Field(pattern=HEX32)
    clicks: list[Click] = Field(min_length=1, max_length=20)


class Reference(BaseModel):
    a: list[float] = Field(min_length=2, max_length=2)  # two ground points, normalised
    b: list[float] = Field(min_length=2, max_length=2)
    metres: float = Field(gt=0, le=500)


class CalibrateIn(BaseModel):
    still: str = Field(pattern=HEX32)
    extra_stills: list[str] = Field(default_factory=list, max_length=4)
    polygon: list[list[float]] = Field(min_length=3, max_length=300)  # the parking surface, normalised
    reference: Reference | None = None
    camera_model: Literal["pinhole", "simple_radial", "radial"] = "simple_radial"  # the lens model GeoCalib fits
    bay_width: float = Field(BAY[0], ge=1.8, le=4.0)
    bay_length: float = Field(BAY[1], ge=3.5, le=8.0)
    aisle: float = Field(AISLE, ge=2.0, le=12.0)
    meta: dict | None = None  # how the polygon was made (clicks, score, edited): kept in the record
    note: str = Field("", max_length=200)


def register_setup(app: FastAPI, *, cfg, ctx, current, snapshots, capture_still, start_detection, admin, setup_http, root,
                   view=(), registry=None, save_fixed=None):
    """Add the endpoints to app. ctx(cam) -> CamCtx (404 for an unknown camera), current(c) -> (settings, version),
    snapshots / capture_still / start_detection: the slot editor's still machinery in create_app, admin: the token
    dependencies, setup_http: the httpx client for the setup service (tests pass a mock transport). registry / save_fixed:
    the camera registry and main.py's "write a camera's Fixed map" (slot suggestions on a Fixed camera)."""
    from api.main import opened  # the per-request LiveStore context (imported late: main imports this module)
    from carparking.live import LiveStore

    jobs = app.state.setup_jobs = JobBoard()

    def enabled():
        if not cfg.geometry:
            raise HTTPException(503, "camera setup (Phase 2 geometry) is switched off: set CARPARK_GEOMETRY_ENABLED=1")

    guard = [*admin, Depends(enabled)]

    def camera_of(cam_id: str):
        c = ctx(cam_id)
        if c.camera is None:
            raise HTTPException(404, f"no camera {cam_id!r}")
        return c

    def still_of(c, sid: str) -> dict:
        still = snapshots.get(sid)
        if still is None or still["cam"] != c.camera.id:
            raise HTTPException(410, "this still is no longer kept (the server restarted or newer ones replaced it): "
                                     "take a new one")
        return still

    def busy(kind: str, cam_id: str):
        job = jobs.running(kind, cam_id)
        if job is not None:
            raise HTTPException(409, f"a {kind} job is already running for this camera (job {job['id']})")

    async def call_setup(path: str, payload: dict, timeout: float) -> dict:
        try:
            r = await setup_http.post(cfg.setup_url.rstrip("/") + path, json=payload, timeout=timeout)
        except httpx.TimeoutException:
            raise ValueError("the setup service didn't answer in time (a CPU can need minutes for the first request, "
                             "which loads the models): try again") from None
        except httpx.HTTPError:
            raise ValueError(NOT_RUNNING) from None
        if r.status_code == 422:
            raise ValueError("setup service: " + str(r.json().get("detail", "bad request"))[:200])
        if r.status_code != 200:
            raise ValueError(f"the setup service answered {r.status_code}")
        return r.json()

    async def vehicles_of(c, still: dict) -> list[dict]:
        """The vehicle detections on the still (the camera's own detector, once per still and settings version)."""
        settings, ver = current(c)
        if still["dets"] is None or still["ver"] != ver:
            start_detection(still, settings["run"], ver)
            still["dets"] = await asyncio.shield(still["task"][1])
            still["ver"] = ver
        return still["dets"]

    def pixel_boxes(dets: list[dict], w: int, h: int) -> list[tuple]:
        return usable_vehicle_boxes([{**d, "box": [d["box"][0] * w, d["box"][1] * h, d["box"][2] * w, d["box"][3] * h]}
                                     for d in dets], w, h)

    def png_b64(mask: np.ndarray) -> str:
        return base64.b64encode(cv2.imencode(".png", mask)[1].tobytes()).decode()

    def from_png(b64: str) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_GRAYSCALE)

    @app.post("/api/cameras/{cam_id}/setup/still", dependencies=guard)
    async def setup_still(cam_id: str):
        """A still of the camera for the setup (the slot editor's snapshot machinery: the camera's own snapshot, go2rtc,
        the worker's frame; kept in memory). No detection is started here. -> {"id", "ts", "source", "width", "height",
        "image" (data URL)}; 502 when no source answers."""
        c = camera_of(cam_id)
        sid, still = await capture_still(c)
        h, w = still["frame"].shape[:2]
        return {"id": sid, "ts": still["ts"], "source": still["source"], "width": w, "height": h,
                "image": "data:image/jpeg;base64," + base64.b64encode(still["jpeg"]).decode()}

    async def surface_candidates(c, still: dict, clicks: list[dict], report) -> dict:
        """SAM 2.1 (setup service) on a still: clicks (pixels, label 1 = ground, 0 = not) -> the parking-surface candidates."""
        h, w = still["frame"].shape[:2]
        report("segmenting (SAM 2.1)", 0.1)
        warnings = []
        seg, dets = await asyncio.gather(
            call_setup("/segment", {"image": base64.b64encode(still["jpeg"]).decode(), "clicks": clicks},
                       SEGMENT_TIMEOUT),
            vehicles_of(c, still), return_exceptions=True)
        if isinstance(seg, BaseException):
            raise seg
        if isinstance(dets, BaseException):  # the hole filling is a refinement, not a requirement
            dets = []
            warnings.append("vehicle detection failed: parked cars are not filled into the surface")
        report("building the polygon", 0.8)
        boxes = [(d["box"][0] * w, d["box"][1] * h, d["box"][2] * w, d["box"][3] * h) for d in dets]
        candidates = []
        for m in seg["masks"]:
            poly = mask_to_polygon(from_png(m["png"]), boxes)
            if poly and SURFACE_AREA[0] <= poly["area_frac"] <= SURFACE_AREA[1]:
                candidates.append({**poly, "score": m["score"]})
        if not candidates:
            raise ValueError("SAM found no usable surface at those clicks (too small or the whole image): click on "
                             "the open parking ground, or add a click on what is not parking")
        # SAM's top score often goes to a smaller piece (a pole splits the pavement); a surface wants the largest of the
        # candidates that score nearly as well. All candidates are returned: the Admin can pick another.
        top = candidates[0]["score"]  # the setup service returns them best score first
        best = max((k for k in candidates if k["score"] >= SCORE_KEEP * top), key=lambda k: k["area_frac"])
        return {"polygon": best["polygon"], "score": best["score"], "area_frac": best["area_frac"],
                "candidates": candidates, "model": "sam2.1_hiera_small", "warnings": warnings,
                "seconds": seg.get("seconds")}

    @app.post("/api/cameras/{cam_id}/setup/segment", dependencies=guard)
    async def setup_segment(cam_id: str, body: SegmentIn):
        """Start the parking-surface segmentation: the clicks (normalised, label 1 = ground, 0 = not ground) go to SAM
        2.1 in the setup service; the best candidate masks become polygons (vehicles standing on the surface are
        filled in). -> the job {"id", "state"...}; its result: {"polygon" (normalised), "score", "area_frac",
        "candidates": [{polygon, score, area_frac}], "model", "warnings"}. 409 while one runs for the camera."""
        c = camera_of(cam_id)
        still = still_of(c, body.still)
        busy("segment", cam_id)
        h, w = still["frame"].shape[:2]
        clicks = [{"x": k.x * w, "y": k.y * h, "label": k.label} for k in body.clicks]

        async def work(report):
            return await surface_candidates(c, still, clicks, report)

        return JobBoard.view(jobs.start("segment", cam_id, work))

    async def calibration_record(c, still: dict, extras: list[dict], poly: np.ndarray, *, reference: dict | None, camera_model: str,
                                 bay: tuple, aisle: float, meta_in: dict | None, report, height_hint: float | None = None) -> dict:
        """The geometry record of a camera from a still (+ extra stills) and its surface polygon: the cars on the stills, GeoCalib + depth
        in the setup service, scale, surface, bays, the drift guard's reference. The shared core of the calibrate job and of a Full
        slot suggestion (height_hint: an earlier typed-reference calibration's camera height, see build_record). Raises ValueError."""
        h, w = still["frame"].shape[:2]
        report("detecting vehicles", 0.05)
        boxes, first = [], []  # the cars of every still: one camera, one geometry, so they pool (one scene is +-20 % off)
        for n, s in enumerate((still, *extras)):
            try:
                dets = await vehicles_of(c, s)
            except Exception:  # noqa: BLE001 - the scale then rests on depth / the typed distance
                continue
            boxes += pixel_boxes(dets, w, h)
            if n == 0:
                first = dets
        # every vehicle of the main still (also those cut by the edge): their bodies are not ground, their footprints are
        obstacles = [(d["box"][0] * w, d["box"][1] * h, d["box"][2] * w, d["box"][3] * h) for d in first
                     if d.get("kind", "vehicle") == "vehicle"]
        report("models: GeoCalib + depth", 0.25)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(poly * (w, h)).astype(np.int32)], 255)
        out = await call_setup("/calibrate", {"images": [base64.b64encode(s["jpeg"]).decode() for s in (still, *extras)],
                                              "mask": png_b64(mask), "depth": True, "camera_model": camera_model}, CALIBRATE_TIMEOUT)
        report("scale, surface, bays", 0.8)
        geo = out["geocalib"]
        meta = {k: v for k, v in (meta_in or {}).items() if k in ("clicks", "score", "model", "edited")}
        if len(str(meta)) > MAX_META:
            meta = {}
        try:
            record = await asyncio.to_thread(
                build_record, width=w, height=h, polygon=poly.tolist(), f=geo["f"], up=geo["up"],
                depth=out.get("depth"), vehicles=boxes, reference=reference, bay=bay, aisle=aisle, meta=meta,
                pins=out.get("pins") or {}, k1=geo.get("k1", 0.0), k2=geo.get("k2", 0.0), lens=geo.get("model", "pinhole"),
                obstacles=obstacles, height_hint=height_hint)
        except SetupError as e:
            raise ValueError(str(e)) from None
        # the drift guard's reference: the still's static background (vehicles and people left out), kept with the version
        background = [(d["box"][0] * w, d["box"][1] * h, d["box"][2] * w, d["box"][3] * h) for d in first if d.get("box")]
        record["reference"] = await asyncio.to_thread(extract_reference, still["frame"], background)
        if record["reference"]["n"] < MIN_REFERENCE:
            record["warnings"].append("the scene has too few stable features for the drift guard: if the camera is moved, "
                                      "this calibration will not notice")
        record["source"] = {"still_ts": still["ts"], "still_source": still["source"],
                            "geocalib_up_spread_deg": round(geo.get("up_spread_deg", 0.0), 3), "stills": 1 + len(extras)}
        return record

    @app.post("/api/cameras/{cam_id}/setup/calibrate", dependencies=guard)
    async def setup_calibrate(cam_id: str, body: CalibrateIn):
        """Start the calibration of a camera from a still and its accepted surface polygon: GeoCalib (focal length,
        camera tilt) and a metric depth cross-check in the setup service; the camera height from the vehicles in the
        still, the depth, and an optional typed reference distance, fused with a confidence; the surface in metres, its
        area and the packed bays (pre-rendered FREE slots) -> a DRAFT geometry version in the camera's store
        (carparking/geometry_store.py). The result: {"version", "state": "draft", "confidence", "h", "pitch_deg",
        "area_m2", "capacity", "scale_sources", "warnings"}. The draft is not used by the worker until activated."""
        c = camera_of(cam_id)
        still = still_of(c, body.still)
        extras = [still_of(c, sid) for sid in body.extra_stills]
        busy("calibrate", cam_id)
        h, w = still["frame"].shape[:2]
        if any(e["frame"].shape[:2] != (h, w) for e in extras):
            raise HTTPException(422, "the stills differ in size")
        poly = np.asarray(body.polygon, float)
        if poly.shape[1:] != (2,) or poly.min() < 0 or poly.max() > 1:
            raise HTTPException(422, "polygon: points [x, y] between 0 and 1")

        async def work(report):
            record = await calibration_record(
                c, still, extras, poly, reference=body.reference.model_dump() if body.reference else None,
                camera_model=body.camera_model, bay=(body.bay_width, body.bay_length), aisle=body.aisle, meta_in=body.meta,
                report=report)
            report("saving the draft", 0.95)
            with opened(LiveStore, c.db) as store:
                try:
                    ver = open_geometry(store, c.camera.id, cfg.env).create(record, created_by="admin", note=body.note)
                except ValueError as e:  # GeometryError: caught as the builtin, not the class (module reloads make two copies)
                    raise ValueError(str(e)) from None
            cal = record["calibration"]
            return {"version": ver, "state": "draft", "confidence": cal["confidence"], "h": cal["h"],
                    "pitch_deg": round(float(np.degrees(cal["pitch"])), 1), "area_m2": record["bev"]["area_m2"],
                    "capacity": record["capacity"]["geometric_capacity"],
                    "bays_with_cars": record["capacity"]["with_cars"], "reference_used": "manual" in cal["scale_sources"],
                    "scale_uncertainty": cal["scale_uncertainty"], "scale_sources": cal["scale_sources"],
                    "drift_guard": record["reference"]["n"] >= MIN_REFERENCE,
                    "warnings": record["warnings"], "finished": time.time()}

        return JobBoard.view(jobs.start("calibrate", cam_id, work))

    @app.post("/api/cameras/{cam_id}/setup/reanchor", dependencies=guard)
    async def setup_reanchor(cam_id: str, body: ReanchorIn):
        """Re-anchor a calibration after the camera was bumped or re-aimed, without redoing the setup: the calibration's stored
        reference (the static background of its still) is matched to this fresh still, the camera's TURN is solved (rotation from
        matched bearings, lens undone), and the calibration is carried over to the new pitch / roll / heading with the same optics,
        height and scale (geometry/reanchor.py). No SAM, no depth, no setup service: it works with the setup service stopped.
        Quality gates refuse anything that is not a clean turn (too few agreeing matches, a shift or zoom, a turn beyond 25 deg, a
        level camera, a lot out of view): the job then fails with the reason. -> the job; its result is a DRAFT (like calibrate's:
        {"version", "parent", "confidence", "h", ..., "reanchor": {...}}), linked to its parent; nothing is used until the Admin
        accepts it (activate, which also checks the camera has not moved again). 404 for an unknown from_version; 409 while one runs."""
        c = camera_of(cam_id)
        still = still_of(c, body.still)
        busy("reanchor", cam_id)
        h, w = still["frame"].shape[:2]

        async def work(report):
            report("finding the calibration", 0.05)
            with opened(LiveStore, c.db) as store:
                rows = store_of(store, c).list()
            usable = [r for r in rows if r["state"] in ("stale", "active", "superseded")
                      and (r["record"].get("reference") or {}).get("n", 0) >= MIN_REFERENCE]
            if body.from_version is not None:
                parent = next((r for r in rows if r["ver"] == body.from_version), None)
                if parent is None:
                    raise ValueError(f"no geometry version {body.from_version}")
                if parent not in usable:
                    raise ValueError(f"version {body.from_version} cannot be re-anchored (a draft, or made before the drift guard so it "
                                     "has no stored reference): do the full setup")
            elif usable:
                parent = max(usable, key=lambda r: (r["activated_at"] or 0, r["ver"]))
            else:
                raise ValueError("no calibration of this camera can be re-anchored (none has a stored reference): do the full setup")
            report("detecting vehicles", 0.15)
            try:
                dets = await vehicles_of(c, still)
            except Exception:  # noqa: BLE001 - the vehicles are only masked out of the matching: go on without them
                dets = []
            boxes = [(d["box"][0] * w, d["box"][1] * h, d["box"][2] * w, d["box"][3] * h) for d in dets if d.get("box")]
            report("matching the background and solving the turn", 0.4)
            record, info = await asyncio.to_thread(reanchor_record, parent["record"], still["frame"], boxes)  # ReanchorError: a ValueError
            record["reanchor"]["parent"] = parent["ver"]
            record["source"] = {**(record.get("source") or {}), "still_ts": still["ts"], "still_source": still["source"],
                                "reanchored_from": parent["ver"]}
            report("saving the draft", 0.9)
            with opened(LiveStore, c.db) as store:
                g = store_of(store, c)
                ver = g.create(record, created_by="admin", note=body.note or f"re-anchored from v{parent['ver']}")
                g.log_event("reanchored", ver, {"parent": parent["ver"], **info})
            cal = record["calibration"]
            return {"version": ver, "state": "draft", "parent": parent["ver"], "parent_state": parent["state"],
                    "confidence": cal["confidence"], "h": cal["h"], "pitch_deg": round(float(np.degrees(cal["pitch"])), 1),
                    "area_m2": record["bev"]["area_m2"], "capacity": record["capacity"]["geometric_capacity"],
                    "bays_with_cars": record["capacity"]["with_cars"], "reference_used": "manual" in cal["scale_sources"],
                    "scale_uncertainty": cal["scale_uncertainty"], "scale_sources": cal["scale_sources"], "drift_guard": True,
                    "reanchor": info, "warnings": record["warnings"], "finished": time.time()}

        return JobBoard.view(jobs.start("reanchor", cam_id, work))

    @app.get("/api/setup/jobs/{job_id}", dependencies=guard)
    def setup_job(job_id: str):
        """Progress of a setup job: {"id", "kind", "cam", "state" (queued / running / done / error), "stage",
        "progress" 0-1, "result", "error"}. 404 for an unknown id (also after an API restart)."""
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job (jobs are forgotten when the API restarts)")
        return JobBoard.view(job)

    # --- accept / discard / read: the versioned store of Step 1 ----------------------------------------------------
    def store_of(store, c):
        return open_geometry(store, c.camera.id, cfg.env)

    @app.get("/api/setup/status", dependencies=admin)
    async def setup_status():
        """Whether camera setup is switched on and whether the setup service answers (admin; this one answers while the
        flag is off, so the page can say why nothing works). -> {"enabled", "service": "up" / "down" / "unchecked", "loaded"
        (models in memory), "pins" (the installed model versions)}. With the flag off nothing is contacted."""
        out = {"enabled": bool(cfg.geometry), "service": "unchecked", "loaded": None, "pins": None}
        if cfg.geometry:
            try:
                r = await setup_http.get(cfg.setup_url.rstrip("/") + "/health", timeout=3)
                ok = r.status_code == 200
                out.update(service="up" if ok else "down", **({"loaded": r.json().get("loaded"), "pins": r.json().get("pins")} if ok else {}))
            except (httpx.HTTPError, ValueError):
                out["service"] = "down"
        return out

    @app.get("/api/cameras/{cam_id}/setup/geometry", dependencies=guard)
    def setup_versions(cam_id: str):
        """The camera's stored versions, newest first, as summaries (no surface / bays), and the active one."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            rows = store_of(store, c).list()
        return {"active": summary(next((r for r in rows if r["state"] == "active"), None)), "versions": [summary(r) for r in rows]}

    @app.get("/api/cameras/{cam_id}/setup/geometry/{version}", dependencies=guard)
    def setup_version(cam_id: str, version: int):
        """One version in full (the record: calibration, the surface contour, the packed bays) - what the setup page draws
        over the still before the Admin accepts it."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            row = store_of(store, c).get(version)
        if row is None:
            raise HTTPException(404, f"no geometry version {version}")
        return {**summary(row), "record": row["record"]}

    def camera_moved_since(store, record) -> str | None:
        """For a re-anchored draft: is the camera still where the draft was made? The worker's latest frame (a few seconds old) is
        compared with the draft's own reference (the new picture's background). A confirmed move -> the reason; no recent frame, or
        a frame that cannot be judged (night) -> None: the Admin's review stands."""
        state = store.state()
        ts = state.get("raw_ts")
        if not state.get("raw") or ts is None or time.time() - ts > SNAPSHOT_FRESH:
            return None
        frame = cv2.imdecode(np.frombuffer(state["raw"], np.uint8), cv2.IMREAD_COLOR)
        res = drift.compare(record.get("reference"), frame) if frame is not None else {"status": "inconclusive"}
        if res["status"] == "drift":
            return (f"the camera moved again since this re-anchoring was made (the background is {res['shift_px']:.0f} px off the "
                    "draft's picture): take a fresh still and re-anchor again")
        return None

    @app.post("/api/cameras/{cam_id}/setup/activate", dependencies=guard)
    def setup_activate(cam_id: str, body: ActivateIn):
        """Accept: version `version` becomes the camera's active calibration (a draft, or an earlier version = rollback);
        the one active before becomes superseded (the last 5 are kept). All in one transaction: there is never a moment
        with two active versions or none in between. Nothing is restarted: whoever reads the store (GET
        /api/cameras/{cam}/geometry now, the worker's flagged read later) sees the new version on its next read.
        A version whose scale was not anchored by a typed reference distance (confidence below HIGH, ~+-25 % on the
        metres) is refused with 409 unless `accept_unverified` is true. 404 unknown version, 409 a stale one.
        -> {"active": summary, "previous": the version replaced or null}."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            g = store_of(store, c)
            row = g.get(body.version)
            if row is None:
                raise HTTPException(404, f"no geometry version {body.version}")
            confidence = (row["record"].get("calibration") or {}).get("confidence")
            if row["state"] != "active" and confidence != "HIGH" and not body.accept_unverified:
                raise HTTPException(409, f"version {body.version} has confidence {confidence}: its scale was not anchored by a "
                                         "typed reference distance (a known wheelbase, a bay width), so its metres are about "
                                         "+-25 % off. Calibrate again with a reference, or accept it unverified.")
            if row["state"] == "draft" and (row["record"].get("reanchor") or {}).get("parent"):
                moved = camera_moved_since(store, row["record"])  # the camera may have been bumped again since this draft was made
                if moved:
                    raise HTTPException(409, moved)
            before = g.active()
            try:
                now = g.activate(body.version)
            except ValueError as e:  # GeometryError: caught as the builtin, not the class (module reloads make two copies)
                raise HTTPException(409, str(e)) from None
        return {"active": summary(now), "previous": before["ver"] if before and before["ver"] != now["ver"] else None}

    @app.delete("/api/cameras/{cam_id}/setup/draft", dependencies=guard)
    def setup_discard(cam_id: str, version: int | None = Query(None, ge=1)):
        """Discard a draft the Admin does not want: ?version=N for one, without it every draft of the camera. Only drafts
        are ever deleted: an active, earlier (rollback) or stale version is refused with 409. -> {"discarded": [versions]}."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            g = store_of(store, c)
            if version is None:
                return {"discarded": g.discard_drafts()}
            try:
                if not g.delete_draft(version):
                    raise HTTPException(404, f"no geometry version {version}")
            except ValueError as e:  # GeometryError: caught as the builtin, not the class (module reloads make two copies)
                raise HTTPException(409, str(e)) from None
        return {"discarded": [version]}

    # --- Suggest slots: an AI bay grid as a DRAFT next to the slots that run now (carparking/slot_suggest.py) ----------------
    # A suggestion is a child draft of the ACTIVE calibration: record["grid"] = its bays, record["suggest"] = how it was made.
    # The worker reads the active version only, so nothing live changes until Accept: an auto-grid camera activates the draft
    # (ActiveGeometry.frozen), a Fixed camera merges the chosen bays into its hand-drawn map.
    def fixed_camera(c) -> bool:
        return bool(current(c)[0]["run"].get("fixed_mode"))

    def drift_refusal(store) -> str | None:
        snap = store.state(images=False).get("snapshot") or {}
        if ((snap.get("geometry") or {}).get("drift") or {}).get("status") in ("drift", "stale"):
            return ("the drift guard sees the camera has moved since the calibration: re-anchor or calibrate again "
                    "(Camera setup) before using suggestions")
        return None

    def running_grid(c, ag: ActiveGeometry, w: int, h: int) -> tuple[str, list[dict], list[dict]]:
        """What the camera runs now -> (kind, its slots, the saved pins): "fixed" = the hand-drawn map, "auto" = the calibration's
        grid with the operator's pins merged in (the same call the worker makes)."""
        if fixed_camera(c):
            return "fixed", [dict(s) for s in registry.slots(c.camera.id)["slots"]], []
        entry = FixedSlotStore(c.fixed).get_auto_grid(c.camera.name)
        return "auto", merge_pins(GeometryFeed._grid_slots(ag, w, h, entry["expected"]), entry["pins"]), entry["pins"]

    def save_pins(c, expected, pins) -> dict:
        """The camera's expected count + pins, stored and handed to its worker (what PUT /api/auto-pins does)."""
        saved = FixedSlotStore(c.fixed).save_auto_grid(c.camera.name, expected, pins, cfg.tz)
        with opened(LiveStore, c.db) as store:
            store.set_option("auto_grid", json.dumps({"expected": saved["expected"], "pins": saved["pins"]})
                             if saved["expected"] or saved["pins"] else "")
        return saved

    def suggestion_of(g, version: int) -> dict:
        row = g.get(version)
        if row is None or not row["record"].get("suggest") or not row["record"].get("grid"):
            raise HTTPException(404, f"no slot suggestion {version}")
        return row

    def suggestion_view(c, row: dict) -> dict:
        rec, meta = row["record"], row["record"]["suggest"]
        w, h = rec["image"]["width"], rec["image"]["height"]
        cal = rec.get("calibration") or {}
        out = {"version": row["ver"], "state": row["state"], "created_at": row["created_at"], "note": row["note"], "meta": meta,
               "grid": rec["grid"], "width": w, "height": h,
               "calibration": {"confidence": cal.get("confidence"), "h": cal.get("h"), "version": meta.get("parent")}}
        with opened(LiveStore, c.db) as store:
            active = store_of(store, c).active()
        if active is None or active["ver"] == row["ver"]:
            return {**out, "baseline": {"kind": meta.get("kind"), "slots": []}, "diff": None, "pins": [], "stale": True}
        kind, running, pins = running_grid(c, ActiveGeometry(active), w, h)
        bound = bind_pin_ids(pins, running) if kind == "auto" else []
        return {**out, "baseline": {"kind": kind, "slots": running}, "diff": diff(rec["grid"], running),
                "pins": [s for s in merge_pins(rec["grid"], bound) if s.get("source") == MANUAL],
                "stale": kind != meta.get("kind") or baseline_hash(running) != meta.get("baseline")}

    async def full_record(c, still: dict, parent: dict, body: SuggestIn, report) -> dict:
        """The Full suggestion's calibration: the surface from SAM (clicks given now, else the clicks stored with the active
        calibration, else its stored polygon), GeoCalib + depth on this still, the scale from the active calibration's camera height."""
        rec = parent["record"]
        cal = rec["calibration"]
        if "manual" not in (cal.get("scale_sources") or {}):
            raise ValueError("a full suggestion keeps the metres of the active calibration, which has no typed reference distance: "
                             "calibrate with a known distance first (Camera setup)")
        h, w = still["frame"].shape[:2]
        clicks = [{"x": k.x, "y": k.y, "label": k.label} for k in body.clicks]
        stored = (rec.get("surface") or {}).get("clicks") or []
        if not clicks:
            clicks = [{"x": float(k["x"]), "y": float(k["y"]), "label": int(k.get("label", 1))} for k in stored
                      if isinstance(k, dict) and all(isinstance(k.get(a), (int, float)) and 0 <= k[a] <= 1 for a in ("x", "y"))]
        if clicks:
            found = await surface_candidates(c, still, [{**k, "x": k["x"] * w, "y": k["y"] * h} for k in clicks], report)
            poly, meta = np.asarray(found["polygon"], float), {"clicks": clicks, "score": found["score"], "model": found["model"]}
        else:
            poly, meta = np.asarray(rec["surface"]["polygon"], float), None
        lay = rec["capacity"]
        lens = cal.get("lens") if cal.get("lens") in ("pinhole", "simple_radial", "radial") else "simple_radial"
        record = await calibration_record(c, still, [], poly, reference=None, camera_model=lens, bay=tuple(lay.get("bay") or BAY),
                                          aisle=float(lay.get("aisle") or AISLE), meta_in=meta, report=report, height_hint=float(cal["h"]))
        turn = abs(float(np.degrees(record["calibration"]["pitch"] - cal["pitch"])))
        if turn > 3.0:
            record["warnings"].append(f"GeoCalib sees the camera {turn:.1f} deg differently than the active calibration: look at "
                                      "the draft carefully before accepting")
        return record

    @app.post("/api/cameras/{cam_id}/suggest", dependencies=guard)
    async def suggest_start(cam_id: str, body: SuggestIn):
        """Suggest slots (admin, job): a fresh still (or the editor's snapshot) and the vehicles on it, then
        quick = the ACTIVE calibration (HIGH confidence) re-tiled around today's vehicles and continued up to the expected count, no
        setup service; full = SAM + GeoCalib + depth again in the setup service (stored surface, or `clicks`), its scale pinned to the
        active calibration's camera height so the result stays HIGH. Either way a DRAFT (a child of the active version) whose bays
        carry the ids of the running bays they overlap; nothing the worker reads changes. Result: {"version", "mode", "bays",
        "diff", "lined_up", "warnings"}. 409 while one runs, 410 for an unknown still."""
        c = camera_of(cam_id)
        sid = body.still
        still = still_of(c, sid) if sid else None
        busy("suggest", cam_id)

        async def work(report):
            nonlocal sid, still
            report("taking a still", 0.05)
            if still is None:
                sid, still = await capture_still(c)
            h, w = still["frame"].shape[:2]
            with opened(LiveStore, c.db) as store:
                parent = store_of(store, c).active()
                refusal = drift_refusal(store)
            if parent is None:
                raise ValueError("this camera has no active calibration: calibrate it first (Camera setup)")
            rec = parent["record"]
            if (rec.get("calibration") or {}).get("confidence") != "HIGH":
                raise ValueError("suggestions need a HIGH-confidence calibration (one anchored by a typed reference distance): "
                                 "calibrate with a known distance first (Camera setup)")
            if refusal:
                raise ValueError(refusal)
            running_g = ActiveGeometry(parent)
            if running_g.camera(w, h) is None:
                raise ValueError("this still has another shape than the calibration's picture: take a new snapshot")
            expected = body.expected or FixedSlotStore(c.fixed).get_auto_grid(c.camera.name)["expected"]
            report("detecting vehicles", 0.15)
            warnings = []
            try:
                dets = await vehicles_of(c, still)
            except Exception:  # noqa: BLE001 - the rows then line up with the calibration's own cars
                dets = []
                warnings.append("vehicle detection failed: the bays are lined up with the calibration's cars, not today's")
            if body.mode == "full":
                record = await full_record(c, still, parent, body, report)
                warnings += record["warnings"]
                new_g = ActiveGeometry({"ver": parent["ver"], "record": record})
            else:
                record, new_g = copy.deepcopy(rec), running_g
            report("laying out the bays", 0.9)
            slots, lined = await asyncio.to_thread(suggest_slots, new_g, dets, w, h, expected)
            kind, running, pins = running_grid(c, running_g, w, h)
            # the slots that run now are never re-laid, matched or moved: the draft carries them in as they are and only adds bays that do not lie on
            # them (a Fixed camera's drawn slots and pin; an auto grid's bays, with its pins kept out of the way)
            grid, protected = protect_baseline(slots, [s for s in running if s.get("source") != MANUAL],
                                               avoid=[s for s in running if s.get("source") == MANUAL])
            record["grid"] = [{k: s[k] for k in ("id", "name", "poly", "lift", "source", "protected", "capacity") if k in s} for s in grid]
            record["suggest"] = {"parent": parent["ver"], "mode": body.mode, "kind": kind, "lined_up": lined, "expected": expected,
                                 "still": sid, "still_ts": still["ts"], "baseline": baseline_hash(running), "made": time.time(),
                                 "vehicles": sum(d.get("kind", "vehicle") == "vehicle" for d in dets), "protected": protected}
            with opened(LiveStore, c.db) as store:
                ver = store_of(store, c).create(record, created_by="admin", note=body.note or f"slot suggestion ({body.mode})")
            return {"version": ver, "mode": body.mode, "bays": len(grid), "diff": diff(record["grid"], running), "lined_up": lined,
                    "warnings": warnings, "finished": time.time()}

        return JobBoard.view(jobs.start("suggest", cam_id, work))

    @app.get("/api/cameras/{cam_id}/suggest", dependencies=guard)
    def suggest_list(cam_id: str):
        """The camera's pending slot suggestions (drafts), newest first: [{"version", "created_at", "note", "meta", "bays"}]."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            rows = [r for r in store_of(store, c).list("draft") if r["record"].get("suggest") and r["record"].get("grid")]
        return {"suggestions": [{"version": r["ver"], "created_at": r["created_at"], "note": r["note"], "meta": r["record"]["suggest"],
                                 "bays": len(r["record"]["grid"])} for r in rows]}

    @app.get("/api/cameras/{cam_id}/suggest/{version}", dependencies=guard)
    def suggest_get(cam_id: str, version: int):
        """One suggestion for the editor's draft layer: {"grid" (the bays), "pins" (the operator's pins merged on top, with the bay
        each binds to), "baseline" {"kind": auto | fixed, "slots"} (what runs now), "diff" {added, removed, moved, unchanged},
        "stale" (the running slots changed since it was made), "meta", "calibration"}."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            row = suggestion_of(store_of(store, c), version)
        return suggestion_view(c, row)

    @app.put("/api/cameras/{cam_id}/suggest/{version}", dependencies=guard)
    def suggest_edit(cam_id: str, version: int, body: SuggestEditIn):
        """Fine-tune a draft: replace its bays (same checks as a slot save; a bay moved or added is marked `edited`). A bay may
        not take a pin's id. Changes nothing that runs. -> {"version", "grid"}."""
        c = camera_of(cam_id)
        try:
            clean = validate_slots(body.grid)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        reserved = set()
        if not fixed_camera(c):
            reserved = {p["slot_id"] for p in FixedSlotStore(c.fixed).get_auto_grid(c.camera.name)["pins"] if p.get("slot_id")}
        with opened(LiveStore, c.db) as store:
            g = store_of(store, c)
            row = suggestion_of(g, version)
            if row["state"] != "draft":
                raise HTTPException(409, f"version {version} is {row['state']}: only a draft can be edited")
            rec = row["record"]
            before = {s["id"]: s for s in rec["grid"]}
            given = {s["id"]: s for s in clean}
            for pid in rec["suggest"].get("protected") or ():  # the running slots of a Fixed camera's draft cannot be removed, moved or reshaped
                old, new = before.get(pid), given.get(pid)
                if old is not None and (new is None or not np.allclose(new["poly"], old["poly"], atol=1e-4)):
                    raise HTTPException(422, f"{old['name']} (slot {pid}) is a running slot: the draft keeps it exactly as it runs, "
                                             "it cannot be " + ("removed" if new is None else "moved or reshaped"))
            grid = []
            for s in clean:
                if s["id"] in reserved:
                    raise HTTPException(422, f"slot {s['id']} is a pin's own id: pick another")
                old = before.get(s["id"])
                if old is not None and old.get("protected"):
                    grid.append(old)  # (as it runs: whatever the client sent)
                    continue
                same = old is not None and np.allclose(old["poly"], s["poly"], atol=1e-4)
                grid.append({"id": s["id"], "name": s["name"], "poly": s["poly"], "lift": s.get("lift", old["lift"] if old else 0.0),
                             "source": old["source"] if same else EDITED})
            rec["grid"] = grid
            rec["suggest"]["edited"] = True
            g.update_draft(version, rec)
        return {"version": version, "grid": grid}

    @app.delete("/api/cameras/{cam_id}/suggest/{version}", dependencies=guard)
    def suggest_discard(cam_id: str, version: int):
        """Discard a suggestion (only a draft that is a slot suggestion). The running slots are untouched."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            g = store_of(store, c)
            suggestion_of(g, version)
            try:
                g.delete_draft(version)
            except ValueError as e:
                raise HTTPException(409, str(e)) from None
        return {"discarded": [version]}

    @app.post("/api/cameras/{cam_id}/suggest/{version}/accept", dependencies=guard)
    def suggest_accept(cam_id: str, version: int, body: SuggestAcceptIn):
        """Accept a suggestion. Auto-grid camera: the draft becomes the active version (the worker takes it within seconds; the
        history scope moves to the new grid version; the operator's pins keep their slot ids). Fixed camera: the chosen bays
        (`select`, default all) are merged into the hand-drawn map (a bay carrying a drawn slot's id replaces its polygon, others
        are added) and the draft is consumed. 409 when the drift guard sees the camera moved, the active calibration or the running
        slots changed since the suggestion was made, or the camera switched mode."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            g = store_of(store, c)
            row = suggestion_of(g, version)
            rec, meta = row["record"], row["record"]["suggest"]
            if row["state"] != "draft":
                raise HTTPException(409, f"version {version} is {row['state']}: only a draft can be accepted")
            parent = g.active()
            if parent is None or parent["ver"] != meta["parent"]:
                raise HTTPException(409, f"the active calibration is not version {meta['parent']} any more: ask for a new suggestion")
            if (refusal := drift_refusal(store)):
                raise HTTPException(409, refusal)
            kind, running, pins = running_grid(c, ActiveGeometry(parent), rec["image"]["width"], rec["image"]["height"])
            if kind != meta["kind"]:
                raise HTTPException(409, f"the camera switched to {kind} mode since this suggestion was made: ask for a new one")
            if baseline_hash(running) != meta["baseline"]:
                raise HTTPException(409, "the slots that run now changed since this suggestion was made: ask for a new one")
            if kind == "fixed":
                ids = {s["id"] for s in rec["grid"]}
                if body.select is not None and not set(body.select) <= ids:
                    raise HTTPException(422, f"no suggested bay {sorted(set(body.select) - ids)}")
                try:
                    saved, ver = save_fixed(c.camera, merge_into_fixed(running, rec["grid"], body.select))
                except ValueError as e:
                    raise HTTPException(422, str(e)) from None
                g.log_event("applied_to_fixed", version, {"taken": sorted(ids if body.select is None else set(body.select)),
                                                          "slots": len(saved)})
                g.delete_draft(version)
                return {"applied": "fixed", "slots": saved, "settings_version": ver}
            if (moved := camera_moved_since(store, rec)):
                raise HTTPException(409, moved)
        bound = bind_pin_ids(pins, running)  # a pin keeps the id it has now, whatever grid comes next
        if bound != pins:
            save_pins(c, meta.get("expected"), bound)
        with opened(LiveStore, c.db) as store:
            g = store_of(store, c)
            try:
                now = g.activate(version)
            except ValueError as e:  # GeometryError: caught as the builtin, not the class (module reloads make two copies)
                raise HTTPException(409, str(e)) from None
        return {"applied": "auto", "active": summary(now), "previous": parent["ver"]}

    @app.get("/api/cameras/{cam_id}/geometry", dependencies=[*view, Depends(enabled)])
    def active_geometry(cam_id: str):
        """The camera's ACTIVE calibration (summary), or null while uncalibrated (viewer token). Read from the store on
        every call: this is how an activation shows up without restarting anything."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            return {"active": summary(store_of(store, c).active())}

    # --- history: occupancy over time and the calibration's lifecycle ------------------------------------------------
    @app.get("/api/cameras/{cam_id}/geometry/history", dependencies=[*view, Depends(enabled)])
    def geometry_history(cam_id: str, hours: float = Query(24, gt=0, le=24 * 90), points: int = Query(240, ge=10, le=1000)):
        """The measured occupancy of the camera over the last `hours` (up to 90 days), in at most `points` buckets
        (>= 60 s each), aggregated in the database over an index: the cost follows `points`, not the history's length.
        -> {"since", "until", "bucket_s", "points": [{"t", "samples", "occupied_avg" / "_max" / "_min", "capacity_avg" /
        "_max", "vehicles_avg", "ver"}], "versions": [the calibration versions seen, oldest first]}. Empty while the
        camera has not been calibrated (viewer token)."""
        c = camera_of(cam_id)
        until = time.time()
        since = until - hours * 3600
        bucket = max(60, int(np.ceil(hours * 3600 / points)))
        with opened(LiveStore, c.db) as store:
            rows = store_of(store, c).history(since, until, bucket)
        return {"since": since, "until": until, "bucket_s": bucket, "points": rows, "versions": sorted({r["ver"] for r in rows})}

    @app.get("/api/cameras/{cam_id}/geometry/events", dependencies=[*view, Depends(enabled)])
    def geometry_events(cam_id: str, limit: int = Query(50, ge=1, le=500), since: float | None = None):
        """The calibration's lifecycle, newest first: draft (a setup made one), activated (Accept / rollback, with the version it
        replaced), stale (the drift guard or an Admin: the reason), discarded, drift_suspected / drift_cleared. ?since=<unix
        time> for only what is newer. -> {"events": [{"id", "t", "kind", "ver", "detail"}]} (viewer token)."""
        c = camera_of(cam_id)
        with opened(LiveStore, c.db) as store:
            return {"events": store_of(store, c).events(limit, since)}

