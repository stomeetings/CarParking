"""Sample-picture sessions: a random parking picture as its own throw-away camera (carparking/sample_session.py).

  POST   /api/sample-sessions                    {image | sample, profile_cam?}  upload a picture -> a session (the picture, upright, <= 2560 px)
  GET    /api/sample-sessions/images             the pictures in <data dir>/samples (jpg / png), to start a session from
  GET    /api/sample-sessions/models             the detectors a session can run (this camera's, stock COCO, camera fine-tunes, the PKLot bay model) + the defaults
  GET    /api/sample-sessions/service            whether the setup service (SAM, GeoCalib) answers, and whether its models are loaded
  GET    /api/sample-sessions/{id}               the session: picture, calibration summary, slots, the layers it would run with, its latest job
  DELETE /api/sample-sessions/{id}               discard it
  POST   /api/sample-sessions/{id}/surface       {clicks: [{x, y, label}]}  SAM outlines the parking surface (job; needs the setup service)
  POST   /api/sample-sessions/{id}/calibrate     {method: "rectangle", pins[4], sides[2], polygon}  or  {method: "geocalib", reference, polygon, lens}  (job)
  GET    /api/sample-sessions/{id}/job           the latest job's state / result
  PUT    /api/sample-sessions/{id}/slots         {slots}  the reviewed (edited) grid, validated like any slot save
  POST   /api/sample-sessions/{id}/zones/{z}/extend  {count}  add the next bays after the zone's last one, stepped on the ground by the zone's calibration
  PUT    /api/sample-sessions/{id}/zones/{z}/slots   {slots}  the zone's bays as edited on the page (moved, reshaped, removed)
  DELETE /api/sample-sessions/{id}/zones/{z}     forget zone z (zones: /surface and /calibrate with `zone` run SAM and GeoCalib on that zone alone; the run merges every zone's bays)
  POST   /api/sample-sessions/{id}/direct        {model?, conf?, iou?, slots?}  the PKLot bay model straight on the picture, no calibration (a comparison; changes nothing in the session); with slots (drawn by hand) it classifies those
  POST   /api/sample-sessions/{id}/run           {layers?}  the picture through the camera's detector and these slots -> the result: slots with their states, detections, counts, notices

State is in this process's memory only: at most MAX_SESSIONS sessions, each forgotten after TTL seconds idle or a restart. Nothing is written to the geometry
store, the camera registry, any slot file, the live options, any history database or the event log; a session never carries a camera's slots. `profile_cam`
only says whose DETECTOR settings (model, confidence, classes, timings) to borrow; its parking-area cuts and its slots are not used.
Admin token. One model job (detector / setup service) at a time across all sessions (409 while one runs).
"""
from __future__ import annotations

import asyncio
import base64
import copy
import secrets
import threading
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from carparking import sample_session as ss
from carparking import sample_test
from carparking.fixed_slots import validate_slots
from carparking.geometry.surface import mask_to_polygon
from carparking.setup_jobs import JobBoard

MAX_SESSIONS = 5
TTL = 1800.0
MAX_BYTES = 12_000_000
SHOW_WIDTH = 1920
SUFFIXES = (".jpg", ".jpeg", ".png")
SEGMENT_TIMEOUT = CALIBRATE_TIMEOUT = 300.0
SURFACE_AREA = (0.01, 0.9)  # a SAM candidate outside this share of the picture is not a parking lot
ZONE_AREA = 0.002  # (one zone, one row, may be a small share of the picture)
MAX_ZONES = 20
# a picture of any size is tested at a lower confidence than a live camera runs at (0.30): distant and partly hidden cars score low, and a sample test
# is read by a person who sees every box; the rest are the stock values
DEFAULT_TUNE = {"model": "default", "conf": 0.20, "iou": 0.45, "footprint": 0.50, "encroach": 0.30}
DOWNLOADABLE = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")  # Ultralytics releases, fetched into the models folder on first use
NOT_RUNNING = ("the setup service isn't reachable: start it with `docker compose --profile setup up -d setup` "
               "(the ground-pins method needs no service)")


class Point(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: int = Field(1, ge=0, le=1)


class CreateIn(BaseModel):
    image: str | None = None
    sample: str | None = Field(None, max_length=200)
    profile_cam: str | None = Field(None, max_length=100)


class Tune(BaseModel):
    """The detector and rule settings of a session (each optional: what is not sent stays as it was). Ranges are the page's sliders."""
    model: str | None = Field(None, max_length=60)  # an id from GET /api/sample-sessions/models; "default" = the profile camera's own detector
    conf: float | None = Field(None, ge=0.10, le=0.75)
    iou: float | None = Field(None, ge=0.20, le=0.70)
    footprint: float | None = Field(None, ge=0.20, le=0.80)  # the share of an object's footprint in a bay that blocks it
    encroach: float | None = Field(None, ge=0.15, le=0.60)  # the share of a bay one vehicle covers that makes it encroached


class SurfaceIn(BaseModel):
    clicks: list[Point] = Field(min_length=1, max_length=20)  # pixels of the session's picture
    tune: Tune = Tune()
    zone: int | None = Field(None, ge=1, le=MAX_ZONES)  # SAM for one zone: its outline is not the session's surface
    bound: list[list[float]] | None = Field(None, min_length=3, max_length=200)  # the zone's drawn outline (fractions): SAM's mask is cut to it


class Reference(BaseModel):
    a: list[float] = Field(min_length=2, max_length=2)
    b: list[float] = Field(min_length=2, max_length=2)
    metres: float = Field(gt=0, le=500)


class CalibrateIn(BaseModel):
    method: str = Field(pattern="^(rectangle|geocalib|topdown)$")
    polygon: list[list[float]] = Field(min_length=3, max_length=200)  # the parking surface, fractions
    pins: list[list[float]] | None = Field(None, min_length=4, max_length=4)
    sides: list[float] | None = Field(None, min_length=2, max_length=2)
    reference: Reference | None = None
    lens: str = Field("simple_radial", pattern="^(pinhole|simple_radial|radial)$")
    bay_width: float = Field(2.5, ge=1.8, le=4.0)
    bay_length: float = Field(5.0, ge=3.5, le=8.0)
    tune: Tune = Tune()
    zone: int | None = Field(None, ge=1, le=MAX_ZONES)  # calibrate one zone: GeoCalib on its surface alone, its own grid; the auto grid is left as it is
    zone_name: str = Field("", max_length=40)
    bound: list[list[float]] | None = Field(None, min_length=3, max_length=200)  # the zone's drawn outline: its cars and its row direction (default: polygon)
    expected_bays: int | None = Field(None, ge=1, le=ss.MAX_ZONE_BAYS)  # the real lot's number of bays in this zone: fixed, the layout never settles on its own count
    rows: int = Field(1, ge=0, le=6)  # rows of bays in a zone, from its front edge (1 by default; 0 = automatic: the zone's depth over a bay's length)


class LayoutIn(BaseModel):
    """Lay a calibrated zone's bays from a few user marks, in ground metres (carparking/layout.py): reference dots anywhere on the separator lines, one master bay
    to clone, or a re-fit of the bays as they stand (after the camera changed). Optional setup-time snap to the painted lines and an inward margin in metres."""
    mode: str = Field(pattern="^(dots|master|refit)$")
    count: int = Field(1, ge=1, le=ss.MAX_ZONE_BAYS)
    dots: list[dict] | None = Field(None, max_length=40)  # [{"x", "y" (fractions), "index"?, "at"? front|mid|back}]
    master: list[list[float]] | None = Field(None, min_length=4, max_length=4)  # the master bay's 4 corners, front-left first
    master_index: int = Field(0, ge=0, le=ss.MAX_ZONE_BAYS)
    width: float | None = Field(None, ge=0.3, le=8.0)
    depth: float = Field(5.0, ge=1.0, le=12.0)
    gap: float = Field(0.0, ge=0.0, le=2.0)
    rows: int = Field(1, ge=1, le=6)
    row_gap: float = Field(0.0, ge=0.0, le=10.0)
    dot_at: str = Field("front", pattern="^(front|mid|back)$")
    heading: float | None = Field(None, ge=-360, le=360)
    fit_zone: bool = False  # dots mode: the bay width is the zone's length over the count exactly (the dots then give only direction and alignment)
    snap: bool = False  # snap the separators to the painted lines (setup only)
    margin_m: float = Field(0.0, ge=0.0, le=0.5)  # pull every bay in by this many metres on the ground


class SlotsIn(BaseModel):
    slots: list[dict] = Field(min_length=1, max_length=200)
    outline: list[list[float]] | None = Field(None, min_length=3, max_length=200)  # the zone's outline / ground, when the page turned it with its bays
    polygon: list[list[float]] | None = Field(None, min_length=3, max_length=200)


class Layers(BaseModel):
    spillage: bool | None = None
    obstacle_filter: bool | None = None
    blocked_objects: bool | None = None


class DirectIn(BaseModel):
    """The PKLot bay model run straight on the picture: its own busy / free bays, no calibration, no slots, no layers. Not stored in the session."""
    model: str | None = Field(None, max_length=60)  # a bay model's id from GET /api/sample-sessions/models (kind "bay"); default: the first one
    conf: float = Field(0.25, ge=0.05, le=0.75)  # the bay models' own default
    iou: float = Field(0.45, ge=0.20, le=0.70)
    slots: list[dict] | None = Field(None, min_length=1, max_length=200)  # hand-drawn slots (fractions): the bay model then only says BUSY / FREE for each of them


class ExtendIn(BaseModel):
    count: int = Field(1, ge=1, le=50)


class RunIn(BaseModel):
    layers: Layers = Layers()
    tune: Tune = Tune()
    layout: str | None = Field(None, pattern="^(auto|zones)$")  # the page's layout (default: the last one calibrated)
    skip: list[int] = Field(default_factory=list, max_length=2000)  # zone bays the user removed
    # dynamic slots: the live auto grid's vehicle-driven rules (a bay for a vehicle off the grid, sub-bays, the wheel-footprint claim). Off (the default):
    # strictly the fixed / configured bays, nothing generated or adjusted
    dynamic: bool = False


def register_sample_sessions(app: FastAPI, *, cfg, ctx, current, admin, root, setup_http, clock=time.time):

    sessions: dict[str, dict] = {}
    app.state.sample_sessions = sessions  # (in memory only; exposed for the tests, which age sessions by hand)
    jobs = JobBoard(keep=20)
    busy = threading.Lock()  # one model job at a time: the detector and the setup service are single-threaded CPU users
    folders = [Path(cfg.data_dir) / "samples", Path(cfg.data_dir) / "sample"]  # (both spellings are in use)
    folder = folders[0]

    def model_list() -> list[dict]:
        """The detectors a session can run: [{id, label, note, kind, weights (relative path or None), ready}]. Only files in the project root and
        models/ (and the Ultralytics release names), never a path from the request."""
        out = [{"id": "default", "label": "This camera's own detector", "note": "the profile camera's weights and whole-frame size", "kind": "vehicle",
                "weights": None, "ready": True}]
        seen = set()
        for base in (Path(root), Path(root) / "models"):
            for p in sorted(base.glob("*.pt")) if base.is_dir() else []:
                rel = p.relative_to(root).as_posix()
                if p.name in seen:
                    continue
                seen.add(p.name)
                if p.name.startswith("parking_yolo") and "obb" in p.name.lower():  # option C: oriented boxes (task "obb"): tilted bays keep their slant
                    out.append({"id": rel, "label": f"PKLot bay model (oriented boxes) · {p.name}", "kind": "bay", "weights": rel, "ready": True, "option": "C", "family": "obb",
                                "note": "option C, PKLot YOLO-OBB: busy / free BAYS as rotated boxes (angled / tilted slots); trained on top-down lots, weak on low angled views"})
                elif p.name.startswith("parking_yolo"):  # option B: axis-aligned boxes
                    out.append({"id": rel, "label": f"PKLot bay model · {p.name}", "kind": "bay", "weights": rel, "ready": True, "option": "B", "family": "aabb",
                                "note": "option B, PKLot YOLO (axis-aligned boxes): trained on PKLot, it finds busy / free BAYS in top-down lots, not cars; on an angled picture it finds nothing"})
                elif p.name.startswith("cam_"):
                    out.append({"id": rel, "label": f"Camera fine-tune · {p.name}", "kind": "vehicle", "weights": rel, "ready": True,
                                "note": "fine-tuned on one camera's view"})
                else:
                    base = p.name == "yolov8s.pt"  # option A: the lightweight COCO baseline every other model is compared with
                    out.append({"id": rel, "label": f"COCO · {p.name}", "kind": "vehicle", "weights": rel, "ready": True, "family": "coco", **({"option": "A"} if base else {}),
                                "note": ("option A, the COCO baseline: " if base else "") + "stock COCO vehicles"})
        for name in DOWNLOADABLE:
            if name not in seen:
                out.append({"id": f"models/{name}", "label": f"COCO · {name} (downloads on first use)", "kind": "vehicle", "weights": f"models/{name}",
                            "ready": False, "note": "needs internet the first time"})
        return out

    def default_tune() -> dict:
        """The settings a new session starts with: the standard COCO detector (option A, yolov8s.pt, the stock car / bus / truck classes) when its weights are in the
        project, else the profile camera's own detector. Never a bay model or a camera fine-tune: those are chosen on purpose."""
        coco = next((m for m in model_list() if m.get("option") == "A" and m["ready"]), None)
        return {**DEFAULT_TUNE, "model": coco["id"] if coco else "default"}

    def tuned(s: dict, t: Tune) -> dict:
        """Merge the request's settings into the session's and return what to run with ({"weights", "bay", "conf", "iou", "footprint", "encroach", "model"})."""
        known = {m["id"]: m for m in model_list()}
        if t.model is not None and t.model not in known:
            raise HTTPException(422, f"model: {t.model!r} is not one of GET /api/sample-sessions/models")
        s["tune"].update({k: v for k, v in t.model_dump().items() if v is not None})
        tn = dict(s["tune"])
        m = known[tn["model"]]
        tn.update(weights=m["weights"], bay=m["kind"] == "bay", label=m["label"])
        return tn

    def purge():
        now = clock()
        for k in [k for k, s in sessions.items() if now - s["updated"] > TTL]:
            sessions.pop(k, None)

    def session_of(sid: str) -> dict:
        purge()
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(404, "no such sample session (it expired after 30 minutes idle, or the server restarted): upload the picture again")
        s["updated"] = clock()
        return s

    def picture(body: CreateIn) -> bytes:
        if (body.image is None) == (body.sample is None):
            raise HTTPException(422, "send either `image` (base64) or `sample` (a file name), not both and not neither")
        if body.sample is not None:
            p = next((f / Path(body.sample).name for f in folders if (f / Path(body.sample).name).is_file()), folder / Path(body.sample).name)  # a bare name
            if p.suffix.lower() not in SUFFIXES or not p.is_file():
                raise HTTPException(404, f"no sample picture {body.sample!r} in {folder.name}/")
            raw = p.read_bytes()
        else:
            try:
                raw = base64.b64decode(body.image.split(",", 1)[-1], validate=True)
            except ValueError:
                raise HTTPException(422, "image: not base64") from None
        if not raw or len(raw) > MAX_BYTES:
            raise HTTPException(422, f"image: a JPEG or PNG up to {MAX_BYTES // 1_000_000} MB")
        return raw

    def data_url(frame, quality: int = 88) -> str:
        shown = frame if frame.shape[1] <= SHOW_WIDTH else cv2.resize(frame, (SHOW_WIDTH, round(frame.shape[0] * SHOW_WIDTH / frame.shape[1])),
                                                                      interpolation=cv2.INTER_AREA)
        return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()).decode()

    def zone_view(s: dict) -> dict:
        """The zones (calibration summary, own bays) and the merged list the run will use."""
        merged, dropped = ss.merge_zones(s["zones"], s["w"], s["h"])
        return {"zones": [{"zone": z, "name": v["name"], "polygon": v["polygon"], "outline": v.get("outline"), "calibration": v["calibration"], "slots": v["slots"]} for z, v in sorted(s["zones"].items())],
                "zone_slots": merged, "zone_dropped": dropped}

    def view(s: dict) -> dict:
        job = jobs.get(s["job"]) if s.get("job") else None
        return {"id": s["id"], "width": s["w"], "height": s["h"], "image": s["image"], "profile_cam": s["cam"], "layers": s["live"], "tune": s["tune"],
                "calibration": s["calibration"], "polygon": s["polygon"], "slots": s["slots"], "layout": s["layout"], **zone_view(s), "job": None if job is None else JobBoard.view(job),
                "expires_in": max(0, round(TTL - (clock() - s["updated"])))}

    async def call_setup(path: str, payload: dict, timeout: float) -> dict:
        try:
            r = await setup_http.post(cfg.setup_url.rstrip("/") + path, json=payload, timeout=timeout)
        except httpx.TimeoutException:
            raise ValueError("the setup service didn't answer in time (the first request loads the models, minutes on a CPU): try again") from None
        except httpx.HTTPError:
            raise ValueError(NOT_RUNNING) from None
        if r.status_code == 422:
            raise ValueError("setup service: " + str(r.json().get("detail", "bad request"))[:200])
        if r.status_code != 200:
            raise ValueError(f"the setup service answered {r.status_code}")
        return r.json()

    def take_model() -> None:
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "another sample job is running (the detector and the setup service take one at a time): wait for it")

    def jpeg_of(frame) -> str:
        return base64.b64encode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()

    async def detect(s: dict, tn: dict, objects: bool = False) -> list[dict]:
        """One detector pass per (model, confidence, NMS, with / without objects): the picture does not change, the settings may."""
        key = (tn["model"], tn["conf"], tn["iou"], objects)
        if key not in s["cache"]:
            if len(s["cache"]) >= 8:
                s["cache"].pop(next(iter(s["cache"])))
            s["cache"][key] = await asyncio.to_thread(sample_test.detect_frame, s["run"], s["frame"], root, objects,
                                                      {k: tn[k] for k in ("weights", "bay", "conf", "iou")})
        return s["cache"][key]

    @app.get("/api/sample-sessions/images", dependencies=admin)
    def images():
        files = sorted({p.name for f in folders if f.is_dir() for p in f.glob("*") if p.suffix.lower() in SUFFIXES and p.is_file()})
        return {"folder": f"{folder.parent.name}/{folder.name}", "images": files[:200]}

    @app.get("/api/sample-sessions/models", dependencies=admin)
    def models():
        return {"models": [{k: v for k, v in m.items() if k != "weights"} for m in model_list()], "defaults": default_tune()}

    @app.get("/api/sample-sessions/service", dependencies=admin)
    async def service():
        """-> {"service": "up" | "down", "loaded": models in memory or None, "command"}: asked before the SAM / GeoCalib steps so the page can say why they would fail."""
        out = {"service": "down", "loaded": None, "command": "docker compose --profile setup up -d setup"}
        try:
            r = await setup_http.get(cfg.setup_url.rstrip("/") + "/health", timeout=3)
            if r.status_code == 200:
                out.update(service="up", loaded=r.json().get("loaded"))
        except (httpx.HTTPError, ValueError):
            pass
        return out

    # ---- create / read / discard --------------------------------------------------------------------------------------------------------
    @app.post("/api/sample-sessions", dependencies=admin)
    def create(body: CreateIn):
        purge()
        raw = picture(body)
        try:
            frame = ss.load_frame(raw)
        except ss.SessionError as e:
            raise HTTPException(422, str(e)) from None
        if frame.shape[0] * frame.shape[1] > 40_000_000:
            raise HTTPException(422, "image: too large (over 40 megapixels)")
        if len(sessions) >= MAX_SESSIONS:  # the oldest idle one makes room
            sessions.pop(min(sessions, key=lambda k: sessions[k]["updated"]), None)
        c = ctx(body.profile_cam)
        run = copy.deepcopy(current(c)[0]["run"])
        live = dict.fromkeys(("spillage", "obstacle_filter", "blocked_objects"), False)  # a sample test starts with every protection off; ask for each in the run
        h, w = frame.shape[:2]
        s = {"id": secrets.token_hex(8), "frame": frame, "w": w, "h": h, "image": data_url(frame), "cam": c.camera.id if c.camera else None,
             "run": run, "live": live, "cache": {}, "tune": default_tune(), "polygon": None, "record": None, "slots": [], "calibration": None,
             "layout": "auto", "zones": {}, "job": None, "created": clock(), "updated": clock()}
        sessions[s["id"]] = s
        return view(s)

    @app.get("/api/sample-sessions/{sid}", dependencies=admin)
    def read(sid: str):
        return view(session_of(sid))

    @app.delete("/api/sample-sessions/{sid}", dependencies=admin)
    def discard(sid: str):
        session_of(sid)
        sessions.pop(sid, None)
        return {"discarded": sid}

    @app.get("/api/sample-sessions/{sid}/job", dependencies=admin)
    def latest_job(sid: str):
        s = session_of(sid)
        job = jobs.get(s["job"]) if s.get("job") else None
        if job is None:
            raise HTTPException(404, "this session has no job yet")
        return JobBoard.view(job)

    # ---- the surface (SAM) --------------------------------------------------------------------------------------------------------------
    @app.post("/api/sample-sessions/{sid}/surface", dependencies=admin)
    async def surface(sid: str, body: SurfaceIn):
        s = session_of(sid)
        if (running := jobs.running("sample", sid)) is not None:
            raise HTTPException(409, f"a job is already running for this session (job {running['id']})")
        tn = tuned(s, body.tune)
        take_model()

        async def work(report):
            try:
                report("detecting vehicles", 0.1)
                raw = await detect(s, tn)
                report("segmenting (SAM 2.1)", 0.3)
                out = await call_setup("/segment", {"image": jpeg_of(s["frame"]), "clicks": [c.model_dump() for c in body.clicks]}, SEGMENT_TIMEOUT)
                boxes = [tuple(d["box"]) for d in raw if d.get("state") is not None]
                bound = None
                if body.bound is not None:  # a zone: only the ground inside its outline
                    bound = np.zeros((s["h"], s["w"]), np.uint8)
                    cv2.fillPoly(bound, [np.round(np.asarray(body.bound, float) * (s["w"], s["h"])).astype(np.int32)], 255)
                low = ZONE_AREA if body.zone is not None else SURFACE_AREA[0]
                found = []
                for m in out["masks"]:
                    mask = cv2.imdecode(np.frombuffer(base64.b64decode(m["png"]), np.uint8), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and bound is not None:
                        mask = cv2.bitwise_and(cv2.resize(mask, (s["w"], s["h"]), interpolation=cv2.INTER_NEAREST), bound)
                    poly = mask_to_polygon(mask, boxes, min_area=low) if mask is not None else None
                    if poly and bound is not None:  # (the bridges over the cars may reach out of the zone: cut once more)
                        cut = np.zeros_like(bound)
                        cv2.fillPoly(cut, [np.round(np.asarray(poly["polygon"], float) * (s["w"], s["h"])).astype(np.int32)], 255)
                        poly = mask_to_polygon(cv2.bitwise_and(cut, bound), min_area=low)
                    if poly and low <= poly["area_frac"] <= SURFACE_AREA[1]:
                        found.append({**poly, "score": m["score"]})
                if not found:
                    raise ValueError("SAM found no usable surface at those clicks" + (" inside the zone's outline" if bound is not None else "") +
                                     ": click on the open parking ground, or add a click on what is not parking")
                top = found[0]["score"]
                best = max((k for k in found if k["score"] >= 0.9 * top), key=lambda k: k["area_frac"])
                if body.zone is None:
                    s["polygon"] = best["polygon"]
                return {"polygon": best["polygon"], "score": best["score"], "area_frac": best["area_frac"], "zone": body.zone, "finished": clock()}
            finally:
                busy.release()

        job = jobs.start("sample", sid, work)
        s["job"] = job["id"]
        return JobBoard.view(job)

    # ---- the calibration --------------------------------------------------------------------------------------------------------------
    @app.post("/api/sample-sessions/{sid}/calibrate", dependencies=admin)
    async def calibrate(sid: str, body: CalibrateIn):
        s = session_of(sid)
        if body.method == "rectangle" and (body.pins is None or body.sides is None):
            raise HTTPException(422, "the rectangle method needs `pins` (4 corners) and `sides` (metres of side 1 and side 2)")
        if body.method in ("geocalib", "topdown") and body.reference is None:
            raise HTTPException(422, f"the {'GeoCalib' if body.method == 'geocalib' else 'top-down'} method needs `reference` (two ground points and the metres between them)")
        if (running := jobs.running("sample", sid)) is not None:
            raise HTTPException(409, f"a job is already running for this session (job {running['id']})")
        tn = tuned(s, body.tune)
        take_model()
        bay = (body.bay_width, body.bay_length)

        async def work(report):
            try:
                report("detecting vehicles", 0.1)
                raw = await detect(s, tn)
                outline = body.bound or body.polygon
                if body.zone is not None:  # only this zone's vehicles orient, anchor and cut its grid: a car of another row is not this row's
                    raw = ss.in_zone(raw, outline, s["w"], s["h"])
                row = outline if body.zone is not None else None
                if body.method == "rectangle":
                    report("solving the pins", 0.5)
                    rec = await asyncio.to_thread(ss.rectangle_record, s["frame"], raw, pins=body.pins, sides=body.sides, polygon=body.polygon, bay=bay,
                                                  meta={"method": "rectangle"}, row_outline=row)
                elif body.method == "topdown":  # a picture looking straight down: no GeoCalib, the typed reference is the whole scale
                    if body.reference is None:
                        raise ValueError("top-down needs two clicked ground points and the distance between them")
                    report("measuring the top-down scale", 0.5)
                    rec = await asyncio.to_thread(ss.topdown_record, s["frame"], raw, reference=body.reference.model_dump(), polygon=body.polygon, bay=bay,
                                                  meta={"method": "topdown"}, row_outline=row)
                else:
                    report("models: GeoCalib + depth", 0.3)
                    mask = np.zeros((s["h"], s["w"]), np.uint8)
                    cv2.fillPoly(mask, [np.round(np.asarray(body.polygon, float) * (s["w"], s["h"])).astype(np.int32)], 255)
                    out = await call_setup("/calibrate", {"images": [jpeg_of(s["frame"])], "mask": base64.b64encode(cv2.imencode(".png", mask)[1].tobytes()).decode(),
                                                          "depth": True, "camera_model": body.lens}, CALIBRATE_TIMEOUT)
                    report("scale, surface, bays", 0.8)
                    rec = await asyncio.to_thread(ss.geocalib_record, s["frame"], raw, geo=out["geocalib"], depth=out.get("depth"),
                                                  reference=body.reference.model_dump(), polygon=body.polygon, bay=bay, meta={"method": "geocalib"},
                                                  row_outline=row)
                if body.zone is not None:  # a zone's bays follow its outline (direction, length, rows), not the car-anchored lattice of the whole picture
                    ref = body.reference.model_dump() if body.method in ("geocalib", "topdown") else {"a": body.pins[0], "b": body.pins[1], "metres": body.sides[0]}
                    slots, sizing = await asyncio.to_thread(ss.row_grid, rec, outline, s["w"], s["h"], bay, body.rows,
                                                            [tuple(d["box"]) for d in raw if d.get("state") is not None], ref, body.expected_bays,
                                                            body.method != "topdown")  # (a top-down picture keeps the drawn rectangle: GeoCalib does not apply)
                else:
                    slots = await asyncio.to_thread(ss.grid_slots, rec, s["w"], s["h"])
                if not slots:
                    raise ValueError("no bay fits on that surface: outline more of the parking ground" + (" in this zone" if body.zone is not None else ""))
                cal = rec["calibration"]
                note = {"method": body.method, "confidence": cal["confidence"], "h": cal["h"], "f": cal["f"], "hfov_deg": cal.get("hfov_deg"),
                        "area_m2": rec["bev"]["area_m2"], "warnings": rec["warnings"], "pins_fit": rec.get("pins_fit"), "bays": len(slots)}
                if body.zone is not None:  # this zone's own camera estimate and grid; the other zones and the auto grid stay as they are
                    bays, over = ss.zone_bays(slots, body.zone, outline, s["w"], s["h"])
                    if over:
                        note["warnings"] = [*note["warnings"], f"{over} bay(s) over the {ss.MAX_ZONE_BAYS} of one zone were left out: split the zone in two"]
                    note["bays"] = len(bays)
                    shown = sizing.pop("outline", None)
                    if shown is not None:  # the outline follows the camera's slope: the page shows the rectangle the bays were cut from
                        outline = shown
                    note["sizing"] = sizing
                    s["zones"][body.zone] = {"name": body.zone_name or f"Zone {ss.zone_letter(body.zone)}", "polygon": body.polygon, "outline": outline, "record": rec,
                                             "slots": bays, "calibration": note}
                    s["layout"] = "zones"
                    return {**note, "zone": body.zone, "slots": bays, **zone_view(s), "finished": clock()}
                s.update(record=rec, slots=slots, polygon=body.polygon, layout="auto", calibration=note)
                return {**s["calibration"], "slots": slots, "finished": clock()}
            except ss.SessionError as e:
                msg = str(e)
                if body.method == "geocalib" and "too level for ground" in msg:  # GeoCalib read an overhead picture as nearly level: say what to do
                    msg += "; for an aerial / drone picture looking straight down choose the Top-down scale method"
                raise ValueError(msg) from None
            finally:
                busy.release()

        job = jobs.start("sample", sid, work)
        s["job"] = job["id"]
        return JobBoard.view(job)

    # ---- review (edit) and run ---------------------------------------------------------------------------------------------------------
    @app.put("/api/sample-sessions/{sid}/slots", dependencies=admin)
    def edit(sid: str, body: SlotsIn):
        s = session_of(sid)
        if s["record"] is None:
            raise HTTPException(409, "calibrate the picture first")
        try:
            s["slots"] = validate_slots(body.slots)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"slots": s["slots"]}

    def zone_of(s: dict, zone: int) -> dict:
        if zone not in s["zones"]:
            raise HTTPException(404, f"zone {zone} is not calibrated in this session")
        return s["zones"][zone]

    @app.post("/api/sample-sessions/{sid}/zones/{zone}/extend", dependencies=admin)
    def extend(sid: str, zone: int, body: ExtendIn):
        s = session_of(sid)
        z = zone_of(s, zone)
        try:
            in_picture = "cars" in ((z["calibration"].get("sizing") or {}).get("source") or "")  # bays laid from the cars: continue them in the picture
            bays, added = ss.extend_zone(z["record"], z["slots"], z["outline"], zone, body.count, s["w"], s["h"], in_picture)
        except ss.SessionError as e:
            raise HTTPException(409, str(e)) from None
        z["slots"] = bays
        z["calibration"] = {**z["calibration"], "bays": len(bays)}
        note = None if added == body.count else (f"{added} of {body.count} added: the next one would lie outside the picture" if added < body.count and len(bays) < ss.MAX_ZONE_BAYS
                                                 else f"{added} of {body.count} added: a zone holds at most {ss.MAX_ZONE_BAYS} bays")
        return {**zone_view(s), "added": added, "note": note}

    @app.post("/api/sample-sessions/{sid}/zones/{zone}/layout", dependencies=admin)
    def lay_zone(sid: str, zone: int, body: LayoutIn):
        """The zone's bays laid from marks in ground metres (see LayoutIn). Replaces the zone's bays; the lattice is returned so the page can show its metres."""
        from carparking import layout as lo
        from carparking import layout_extra as lx
        from carparking.layout_snap import snap_lattice
        s = session_of(sid)
        z = zone_of(s, zone)
        cam = ss.ground_camera(z["record"], s["w"], s["h"])
        if cam is None:
            raise HTTPException(409, "this zone has no usable calibration: calibrate it again")
        w, h = s["w"], s["h"]
        try:
            if body.mode == "dots":
                if not body.dots:
                    raise HTTPException(422, "dots: place at least one reference dot")
                dots = [lx.Dot((float(d["x"]), float(d["y"])), None if d.get("index") is None else int(d["index"]), d.get("at")) for d in body.dots]
                lay, info = lx.layout_from_dots(cam, dots, body.count, w, h, body.width, body.depth, body.gap, body.rows, body.row_gap, body.dot_at, body.heading,
                                                outline_px=z["outline"], fit_zone=body.fit_zone)  # (the zone polygon is the row's extent; the dots only guide)
            elif body.mode == "master":
                if body.master is None:
                    raise HTTPException(422, "master: draw the master bay (4 corners)")
                lay, info = lx.clone_master(cam, body.master, body.count, w, h, body.master_index, body.rows, body.gap, body.row_gap)
            else:
                if not z["slots"]:
                    raise HTTPException(409, "this zone has no bays to re-fit")
                lay = lo.refit(z["slots"], cam, w, h, body.count if body.count > 1 else None)
                info = {}
            report = None
            if body.snap:
                lay, report = snap_lattice(s["frame"], lay, cam)
            if body.margin_m > 0:
                slots, kept_full = lx.inset_slots(lay, cam, w, h, body.margin_m)
            else:
                slots, kept_full = lay.to_slots(cam, w, h), []
        except lo.LayoutError as e:
            raise HTTPException(422, str(e)) from None
        if not slots:
            raise HTTPException(422, "no bay of the layout is inside the picture")
        bays, over = ss.zone_bays(slots, zone, z["outline"], w, h)
        z["slots"], z["layout"] = bays, lay.to_dict()
        z["calibration"] = {**z["calibration"], "bays": len(bays)}
        warn = [f"{over} bay(s) over the {ss.MAX_ZONE_BAYS} of one zone were left out"] if over else []
        warn += list((info or {}).pop("warnings", []) or [])
        warn += [f"{n}: too thin for the {body.margin_m} m margin, kept full size" for n in kept_full]
        return {**zone_view(s), "layout": z["layout"], "info": info, "snap": report, "warnings": warn}

    @app.put("/api/sample-sessions/{sid}/zones/{zone}/slots", dependencies=admin)
    def edit_zone(sid: str, zone: int, body: SlotsIn):
        s = session_of(sid)
        z = zone_of(s, zone)
        try:
            bays = validate_slots(body.slots)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        if any(x["id"] // 100 != zone for x in bays):
            raise HTTPException(422, f"a bay of zone {zone} has an id {zone}01..{zone}99")
        for key in ("outline", "polygon"):
            pts = getattr(body, key)
            if pts is not None and any(len(q) != 2 or not all(-1e-3 <= v <= 1.001 for v in q) for q in pts):
                raise HTTPException(422, f"the zone's {key}: corners are [x, y] shares of the picture")
        if body.outline is not None:
            z["outline"] = body.outline
        if body.polygon is not None:
            z["polygon"] = body.polygon
        z["slots"] = bays
        z["calibration"] = {**z["calibration"], "bays": len(bays)}
        return zone_view(s)

    @app.delete("/api/sample-sessions/{sid}/zones/{zone}", dependencies=admin)
    def drop_zone(sid: str, zone: int):
        s = session_of(sid)
        if s["zones"].pop(zone, None) is None:
            raise HTTPException(404, f"zone {zone} is not calibrated in this session")
        return zone_view(s)

    # ---- hand-drawn slots: BUSY / FREE for each one, no calibration, no tracker, no layers ---------------------------------------------------------
    BAY_TRUST = 0.40  # a bay model's verdict on a slot counts only from this confidence: below it the PKLot models, out of their top-down domain, were measured to be wrong as often as right
    FALLBACK_CONF = 0.20  # the vehicle detector that decides a slot the bay model cannot see (the sample test's own default)

    async def judge_drawn(s: dict, sid: str, m: dict, drawn: list[dict], body: DirectIn, t0: float) -> dict:
        """A vehicle detector decides straight (BUSY = a vehicle stands in the slot, else FREE). A bay model is asked per slot, on the slot's own crop, then
        on the whole picture; a slot it cannot decide is judged by the camera's vehicle detector instead (never UNKNOWN while that detector runs), and
        each slot says which one decided it."""
        w, h = s["w"], s["h"]
        frame = s["frame"]
        found = 0
        by: dict[int, str] = {}
        verdict: dict[int, tuple[str, float]] = {}
        if m["kind"] == "bay":
            tune = {"weights": m["weights"], "bay": True, "conf": body.conf, "iou": body.iou}
            crops = await asyncio.to_thread(sample_test.classify_crops, frame, drawn,
                                            lambda crop: sample_test.detect_frame(s["run"], crop, root, False, tune))
            crops = {i: v for i, v in crops.items() if v[1] >= BAY_TRUST}
            verdict.update(crops)
            by.update({i: "bay model, slot crop" for i in crops})
            raw = await asyncio.to_thread(sample_test.detect_frame, s["run"], frame, root, False, tune)
            found = len(raw)
            for x in sample_test.classify_drawn([d for d in drawn if d["id"] not in verdict], raw, w, h):
                if x["state"] != "UNKNOWN" and x["conf"] >= BAY_TRUST:
                    verdict[x["id"]] = (x["state"], x["conf"])
                    by[x["id"]] = "bay model, whole picture"
        rest = [d for d in drawn if d["id"] not in verdict]
        label = m["label"]
        if rest:
            vehicle = m if m["kind"] != "bay" else next(x for x in model_list() if x["id"] == "default")
            cars = await asyncio.to_thread(sample_test.detect_frame, s["run"], frame, root, False,
                                           {"weights": vehicle["weights"], "bay": False, "conf": body.conf if m["kind"] != "bay" else FALLBACK_CONF, "iou": body.iou})
            if m["kind"] != "bay":
                found = len(cars)
            else:
                label += f" (+ {vehicle['label']} for {len(rest)} slot(s) it could not decide)"
            asked = {d["id"] for d in rest}
            for i, v in sample_test.classify_by_vehicles(drawn, cars, w, h).items():  # all the slots compete for each vehicle, so a car belongs to one of them
                if i in asked:
                    verdict[i] = v
                    by[i] = "vehicle detector"
        slots = [{"id": d["id"], "name": d.get("name") or f"Slot {d['id']}", "state": verdict[d["id"]][0], "conf": verdict[d["id"]][1], "by": by[d["id"]],
                  "hidden": False, "phantom": False, "duplicate_of": None, "poly": d["poly"]} for d in drawn]
        n = {k: sum(x["state"] == k for x in slots) for k in ("FREE", "BUSY", "UNKNOWN")}
        return {"shape": [w, h], "slots": slots, "detections": [], "vehicles": 0, "others": 0, "events": [],
                "counts": {"free": n["FREE"], "busy": n["BUSY"], "unknown": n["UNKNOWN"], "hidden": 0, "phantom": 0, "duplicate": 0, "slots": len(slots), "learning": 0},
                "area": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "roi": None},
                "layers": {"spillage": False, "obstacle_filter": False, "blocked_objects": False, "ground": False}, "frames": 1, "step_s": 0.0,
                "seconds": round(time.time() - t0, 2), "kind": "direct", "session": sid, "slot_count": len(slots), "source": "session", "name": None,
                "image": s["image"], "detector": label, "detected": found, "direct": {"conf": body.conf, "iou": body.iou, "drawn": True,
                                                                                    "by": {k: sum(v == k for v in by.values()) for k in sorted(set(by.values()))}}}

    # ---- the bay model, straight on the picture (a comparison next to the calibrated pipeline; nothing of the session changes) ------------------
    @app.post("/api/sample-sessions/{sid}/direct", dependencies=admin)
    async def direct(sid: str, body: DirectIn):
        s = session_of(sid)
        bays = [m for m in model_list() if m["kind"] == "bay"]
        if not bays:
            raise HTTPException(409, "no PKLot bay model (parking_yolo*.pt) in the project's models folder")
        # hand-drawn slots may be judged by any detector of the list (a vehicle detector too); the model's own boxes only by a bay model
        pool = [x for x in model_list() if x["ready"]] if body.slots is not None else bays
        m = next((x for x in pool if x["id"] == body.model), None) if body.model else bays[0]
        if m is None:
            raise HTTPException(422, f"model: {body.model!r} is not {'a model' if body.slots is not None else 'a bay model'} of GET /api/sample-sessions/models")
        drawn = None
        if body.slots is not None:
            try:
                drawn = validate_slots(body.slots)
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
        take_model()
        t0 = time.time()
        try:
            if drawn is not None:
                return await judge_drawn(s, sid, m, drawn, body, t0)
            raw = await asyncio.to_thread(sample_test.detect_frame, s["run"], s["frame"], root, False,
                                          {"weights": m["weights"], "bay": True, "conf": body.conf, "iou": body.iou})
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - a model that fails to load / run: say so
            raise HTTPException(500, f"the bay model failed: {type(e).__name__}: {e}"[:300]) from None
        finally:
            busy.release()
        w, h = s["w"], s["h"]
        ordered = sorted((d for d in raw if d.get("state") is not None), key=lambda d: (round(d["box"][1] / 40), d["box"][0]))
        slots = [{"id": i + 1, "name": f"B{i + 1}", "state": "BUSY" if d["state"] == 1 else "FREE", "hidden": False, "phantom": False, "duplicate_of": None,
                  "conf": round(float(d["conf"]), 3),
                  "poly": [[round(min(max(x / w, 0.0), 1.0), 4), round(min(max(y / h, 0.0), 1.0), 4)] for x, y in d["poly"]]} for i, d in enumerate(ordered)]
        busy_n = sum(x["state"] == "BUSY" for x in slots)
        return {"shape": [w, h], "slots": slots, "detections": [], "vehicles": 0, "others": 0, "events": [],
                "counts": {"free": len(slots) - busy_n, "busy": busy_n, "unknown": 0, "hidden": 0, "phantom": 0, "duplicate": 0, "slots": len(slots), "learning": 0},
                "area": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "roi": None},
                "layers": {"spillage": False, "obstacle_filter": False, "blocked_objects": False, "ground": False}, "frames": 1, "step_s": 0.0,
                "seconds": round(time.time() - t0, 2), "kind": "direct", "session": sid, "slot_count": len(slots), "source": "session", "name": None,
                "image": s["image"], "detector": m["label"], "detected": len(slots), "direct": {"conf": body.conf, "iou": body.iou}}

    @app.post("/api/sample-sessions/{sid}/run", dependencies=admin)
    async def run(sid: str, body: RunIn):
        s = session_of(sid)
        zoned = (body.layout or s["layout"]) == "zones"
        merged = ss.merge_zones(s["zones"], s["w"], s["h"], set(body.skip))[0] if zoned else []
        if zoned and not merged:
            raise HTTPException(409, "calibrate at least one zone first")
        if not zoned and (s["record"] is None or not s["slots"]):
            raise HTTPException(409, "calibrate the picture and review its slots first")
        layers = {k: (getattr(body.layers, k) if getattr(body.layers, k) is not None else s["live"][k]) for k in s["live"]}
        tn = tuned(s, body.tune)
        take_model()
        t0 = time.time()
        try:
            raw = await detect(s, tn, objects=layers["blocked_objects"])
            run_cfg = {**ss.session_run(s["run"]), "conf": tn["conf"]}  # the monitor takes the confidence the detections were taken at
            if zoned:  # ONE run on every zone's bays (merged, duplicates out): each vehicle is matched once against all of them. Dynamic: the footprint claim and
                # sub-bays (the surface is the hull of the zones' outlines). Spillage measures on the first zone's ground.
                first = s["zones"][min(s["zones"])]["record"]
                surface = None  # dynamic: a vehicle standing in a zone (the hull of the zones' outlines) with no bay gets one from its footprint, as on the live auto grid
                if body.dynamic:
                    pts = np.float32([p for z in s["zones"].values() if z["slots"] for p in z["outline"]]).reshape(-1, 2)
                    surface = cv2.convexHull(pts).reshape(-1, 2).tolist()
                out = await asyncio.to_thread(sample_test.run_sample, s["frame"], run_cfg, merged, raw, layers, cfg.tz,
                                              ss.ground_camera(first, s["w"], s["h"]), {k: tn[k] for k in ("footprint", "encroach", "conf")},
                                              surface, body.dynamic)
                for x in out["slots"]:
                    z = s["zones"].get(int(x["id"]) // 100) if isinstance(x.get("id"), int) else None
                    x["zone"] = z["name"] if z else None
            else:
                out = await asyncio.to_thread(sample_test.run_sample, s["frame"], run_cfg, s["slots"], raw, layers, cfg.tz,
                                              ss.ground_camera(s["record"], s["w"], s["h"]), {k: tn[k] for k in ("footprint", "encroach", "conf")},
                                              s["polygon"] if body.dynamic else None, body.dynamic)  # dynamic: the live auto grid's rules (footprint claim, bays for vehicles off the grid, sub-bays)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - a model that fails to load / run: say so
            raise HTTPException(500, f"the sample test failed: {type(e).__name__}: {e}"[:300]) from None
        finally:
            busy.release()
        return {**out, "seconds": round(time.time() - t0, 2), "kind": "session", "session": sid, "tune": {k: tn[k] for k in ("model", "conf", "iou", "footprint", "encroach")}, "detector": tn["label"],
                "detected": sum(d.get("state") is not None for d in raw), "camera": s["cam"], "slot_count": len(merged if zoned else s["slots"]), "layout": "zones" if zoned else "auto", "dynamic": body.dynamic,
                "source": "session", "name": None, "image": s["image"]}
