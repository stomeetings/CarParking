# Activity Log

> **Resume here:** section "### Start here next session" (state 2026-10-22, items 1–86): current state, how to work, open items for the next session.
> **Newest (2026-10-28):** items 107 (local snapshot / restore tooling, snapshot taken) and 108 (Suggest slots, built + tested; later rebuilt into the running stack by the user, dormant). Item 109: Camera 1 back to the 12-space baseline (auto bay creation off, live). Items 110 / 111: roadmaps only, nothing built: `docs/camera_drift_roadmap.md`, `docs/occlusion_tracking_roadmap.md` (start after the 2026-10-29 demo). Item 112: Settings wording changed in source only, NOT built or deployed (waits for the post-demo rebuild).
> Item 113: `docs/night_vision_roadmap.md` (plan only).
> **Before the 2026-10-29 demo:** no deploy, no live change. Undo for item 109: set camera-1 options `auto_dynamic` and `slot_split` back to `1`.

## 2026-10-18: Mode 5, RTSP IP camera + entry/exit history (office PC, Docker)

Setup on the office PC: project in `C:\F\Projects\CarParking`, container `carparking` (image
`stomeetings/carparking:latest`) bind-mounts it at `/app`, Streamlit on :8501. Container clock is UTC; OpenCV 5.0 with
FFmpeg. The Docker packaging / Docker Hub steps themselves were never written into this log.

### Done (tested by Claude this time)
1. **Camera probe** (from inside the container): Hikvision `192.168.20.10`, 101 = 2560×1440, 102 = 640×360, both
   open over RTSP/TCP in ~1.5 s. Scene "Front Parking 02": low covered parking, no painted bays, 1–2 vehicles.
2. **`src/carparking/rtsp.py`**: `LatestFrameReader` thread drains the stream and keeps only the newest frame (FFmpeg
   ignores CAP_PROP_BUFFERSIZE); open/read timeouts, reconnect with backoff 1→30 s; `mask_url` hides the password.
3. **`src/carparking/history.py`**: `SlotTracker` learns a slot map (candidate box stays ≥ `learn_hold` s → slot),
   matches detections by IoU ≥ 0.3, changes state only after `hold` s; event time = first sighting. Polygons stored as
   frame fractions (main and sub share a map). `HistoryDB` (SQLite `parking_history.db` in the repo root): table
   `history` (one row per session: entry, exit, duration_min, status Occupied/Available, note) and `slots` (per camera name).
4. **Mode 5 in `scripts/demo_app.py`**: stream main/sub/custom, camera name, detector (bay model via Model selector or
   COCO), hold/learn sliders, "No detection = FREE" (on for COCO), parking-area cuts, CSV download, "Reset slot map";
   live JPEG view (downscaled to 1280 px), status line (frame age, inferences/s), live history table.
   URL from `$CARPARK_RTSP_URL` (fallback = the camera URL, **password in source**), times in `$CARPARK_TZ`
   (default Asia/Karachi).
5. **Tests** `tests/test_history.py` (8 pass in a throwaway container). End-to-end on the real camera, 45 s each:
   frame age at pickup ≤ 0.13 s; COCO `yolov8n` ~0.3 inferences/s at full res, learned the pickup as a BUSY slot;
   **PKLot OBB ~6/s but 18–31 false FREE "slots" on this scene and misses the pickup** (trained on high top-down
   views of painted bays). Unreachable camera → "retrying" status, no crash.

### Later the same day: detection tuning + fine-tuning pipeline (tested by Claude)
6. **Diagnosis "COCO misses the pickup"**: it did detect it, but SAHI's 640 px tiles cut it at x=1344 into a rear
   half (0.74) and a wheel (0.14) that jump between frames. Whole frame: full pickup 0.60–0.75 on 5/5 live frames.
   The covered motorbike is not detected by COCO at any threshold (even 0.05): needs the fine-tune.
7. `YoloDetector(whole_imgsz=...)`: one whole-frame pass for detect weights (1280: 0.13 s, 1920: 0.3 s on CPU vs
   SAHI 2.4 s); `.classes` = vehicle class ids (COCO [2, 3, 5, 7]) passed into predict. Modes 1–4 unchanged.
8. Mode 5 UI: own confidence slider (0.05–0.50, default 0.15), "Vehicle inference" (whole 1280 / 1920 / SAHI),
   "Target classes" multiselect, ROI polygon saved per camera name (DB table `settings`; suggested polygon for
   Front Parking 02 in the placeholder), cyan edge, labels with confidence, "📸 Save snapshot" + auto-save every N s.
   Detector list: `models/cam_*.pt` (camera fine-tunes) first, then COCO, then bay model.
9. `scripts/collect_dataset.py` (RTSP → `dataset_raw/`, every N s, skips near-duplicates), `scripts/auto_label.py`
   (yolov8x teacher → `datasets/<camera>/` YOLO dataset, classes car/motorcycle, split by capture hour, never
   overwrites reviewed labels, preview/). Tested end-to-end incl. a 1-epoch CPU training. `yolov8x.pt` (130 MB) is in
   the repo root. Tests: `tests/test_dataset.py`; 13 pass.
10. **Incident**: a test cleanup deleted the live app's `parking_history.db` (Mode 5 test history lost,
    unrecoverable); `docker restart carparking` restored DB access. Now `$CARPARK_HISTORY_DB` / `$CARPARK_DATASET_RAW`
    let tests use temp paths.

11. **Occlusion + exit buffer** (`history.py`): a BUSY slot with no matching box is HIDDEN (state frozen, FREE
    countdown not running) when a vehicle box covers >= `occlusion_cover` (0.3) of it and its bottom edge is lower in
    the frame (in front). BUSY -> FREE needs `exit_hold` (UI default 60 s) of visible emptiness; exits after an
    occlusion get the note "was hidden behind a vehicle in front (exit = first seen empty)". Same-depth neighbours
    are not occluders. Mode 5: two sliders, "HIDDEN (kept BUSY)" label, hidden count. 18 tests pass.
12. **Covered bike workflow**: `auto_label.py --add-box motorcycle:0.412,0.235,0.479,0.292` (draft box for the
    static covered bike, measured on a live frame) and `--stats`. Tested on a read-only copy: collect 6 frames →
    labels with pickup + bike → 1 epoch at 1280 px on CPU (29 s for 4 images) → `models/cam_front_parking_02.pt`
    loads as detect {car, motorcycle}.

13. **"SlotTracker got an unexpected keyword argument 'exit_hold'"**: not a code bug. The live Streamlit process
    had imported the old `history.py`; inotify events don't cross the Windows bind mount, so the edit was never
    reloaded. Fix: `docker restart carparking` + `.streamlit/config.toml` with `fileWatcherType = "poll"` (edits to
    `src/` are now picked up). If it ever recurs: restart the container.

14. **Camera-model selection reviewed**: Mode 5 lists `models/cam_<slug of camera name>.pt` first (default), then
    other camera models, COCO, bay model; a bad `cam_*.pt` shows an error instead of crashing; Target classes = the
    model's vehicle classes (auto_label models: car + motorcycle; truck/bus are merged into car). `VEHICLES` also
    accepts vehicle/pickup/van/suv/motorbike/bike/scooter. Tested with a real 1-epoch cam model, a second camera
    model and a broken file in a project copy; 20 tests pass. Camera name must be "Front Parking 02" to match
    `cam_front_parking_02.pt` once several camera models exist.

15. **Camera model not selectable / stray grid boxes**: `models/cam_front_parking_02.pt` does not exist yet (no
    trained camera model anywhere in the project). The stray boxes were the saved slot map: "Camera 1" held 52 slots
    learned by earlier bay-model / SAHI / low-conf runs, loaded for every detector. Fixes: slot map + history scoped
    per camera AND detector (`HistoryDB(detector=...)`, scope "Front Parking 02 | cam_front_parking_02"; ROI stays
    per camera; legacy rows stay under the bare name); saved slots outside the parking area are skipped at start;
    detector labels "📷 Camera model · / 🚗 COCO vehicles · / 🅿️ Bay model ·"; sidebar note when the camera has no
    own model; `use_container_width` → `width="stretch"`; `CARPARK_MAX_FRAMES` for finite smoke runs. Real Mode 5
    loop tested on the live camera with a 1-epoch cam model in a project copy; 21 tests pass.

16. **User trained `models/cam_front_parking_02.pt`** (detect, car + motorcycle, 60 epochs, imgsz 1280). Live frame:
    covered bike motorcycle 0.99, pickup car 0.91–0.93 (COCO never saw the bike).
17. **Hybrid Dual Detector** (Mode 5 option "🔀 Hybrid Dual Detector · cam_<camera>.pt + yolov8n.pt", listed right
    after the camera model when it exists): `HybridDetector` in `yolo_engine.py` runs both models per frame and merges
    with `merge_dets` (vehicles across vehicle classes by IoU 0.5; same class also by IOS 0.8 for fragments; other
    classes only with themselves). The camera model's box wins a duplicate; "src" = cam / coco / cam+coco shown on
    the overlay; per-model raw counts in the metrics line; status banner names both files. Non-vehicle COCO classes
    ("Also detect", default person/dog/cat) come back with state None: drawn magenta, never given to the tracker.
    Own slot map scope `hybrid_cam_front_parking_02+yolov8n`. 0.27 s/frame vs 0.09 s cam only (CPU).
    `merge_dets` also applied to every single-model whole-frame output (cam model alone: 11 raw boxes → 4).
    Recommended conf for the camera model / hybrid: ~0.35 (a 0.27 false motorbike under the pickup wheel). 26 tests.
18. **2026-10-19 `ImportError: cannot import name 'HybridDetector'`**: the code was fine; the running Streamlit
    server still had the pre-hybrid `carparking.yolo_engine` in `sys.modules` (Streamlit reruns scripts/ on edit but
    never re-imports src/ modules; same cause as the earlier `exit_hold` TypeError). `demo_app.py` now drops
    `carparking.*` modules whose file changed since import (`_drop_stale_modules`, mtime stamp), so src/ edits apply
    on the next rerun without a container restart. `yolo_engine.__all__` added; `detector_label` checks
    `det.task == "hybrid"` instead of isinstance (cached detectors can predate a reload). 26 tests.
19. **2026-10-19 `AttributeError: 'YoloDetector' object has no attribute 'vehicle_names'`**: the methods exist
    (`yolo_engine.py`); `st.cache_resource` still held detectors built from the pre-edit class (item 18 reloaded the
    module, not the cached objects). `get_detector` / `get_hybrid` now key the cache on `engine_version()` (mtime of
    yolo_engine.py) and the cache is cleared whenever stale modules are dropped. Reproduced in a project copy and
    verified fixed for COCO, camera model and hybrid. Mode 5 confidence default 0.15 → **0.35** (a page reload gives
    the new default; an open session keeps its slider value). 26 tests.
20. **2026-10-19 back to stock COCO `yolov8n.pt` as the main detector** (Front Parking 02 now has 5 parked cars:
    white sedan in front, dark car, white hatchback, silver car, pickup). Mode 5 Detector order: COCO yolov8n
    (default), camera model, hybrid, other camera models, bay model (fine-tune still selectable, no longer
    automatic). Mode 5 defaults: confidence **0.20** (the back-row hatchback and silver car score 0.20–0.33 at this
    angle; 0.25 drops them), **Cut right 0.40** (road/driveway). Target classes stay COCO 2/3/5/7.
    **Merge fix** (`merge_dets`): the same-class "fragment" rule (≥ 80% inside) now also needs level bottom edges
    (`_same_depth`, within 15% of box height); before, the hatchback parked behind the dark car (its box inside the
    dark car's box) was merged away. Real duplicates still merge by IoU ≥ 0.5. Test added (27 pass).
    Live: 1280 px whole frame finds all 5 in 3 of 5 single frames (a weak back-row car flickers); the real Mode 5
    loop (70 inferences, ~1 inference/s) learned **5 slots, BUSY 5**. 1920 px was worse (merges two back cars).
    No `src/carparking/slot_tracker.py` exists; SlotTracker is in `history.py` and needed no change (candidates
    survive up to learn_hold of missed frames, exit_hold 60 s rides over flicker).
21. **2026-10-19 slots too wide (front car's slot swallowed the floor around it)**: there was no padding; a learned
    slot was the first frame's whole image box, which in this low side view includes body/roof drawn over the floor
    behind the car (whole boxes of front and back cars overlapped). `history.py`: `footprint(poly, share)` = bottom
    share of an axis-aligned box (rotated bay boxes unchanged); `SlotTracker(footprint=...)` matches and learns on
    footprints, occlusion still uses whole boxes; a learned slot = median of the boxes seen while learning (not the
    first). Mode 5 slider "Slot footprint" default **0.35** (bay model: 1.0). Old slot maps keep their wide polygons:
    press "Reset slot map" once. 30 tests (3 new).
    Live (09:25–09:30, silver car had left, 4 cars): footprint slots sit on the tyre/floor band and no longer overlap.
    Replay of 60 recorded frames: yolov8n gives 3 boxes in 52/60 frames because it boxes the white hatchback and the
    dark car behind it as one; footprint 0.35 → 3 slots, whole box → 4 only via a near-duplicate slot over that same
    combined box. Separating those two needs a stronger detector (yolov8s.pt in the repo root is picked up as COCO).
22. **2026-10-19 item 21 reverted at the user's request** (footprint slots caused detection / slot-mapping issues
    on the live stream): `history.py` is back to whole-box slots and matching, slot = first box (median removed),
    `footprint()` and the Mode 5 "Slot footprint" slider removed, its 3 tests removed. Kept: COCO yolov8n default,
    conf 0.20, Cut right 0.40, merge_dets bottom-edge fragment fix, exit_hold 60 s. Slot maps learned with
    footprints must be cleared with "Reset slot map". 27 tests; live Mode 5 loop (70 inferences): 4 slots, BUSY 4,
    4 vehicles (4 cars parked), no exceptions.
23. **2026-10-19 yolov8s.pt as default COCO detector** (downloaded 22.6 MB from the Ultralytics assets release
    v8.3.0 into the repo root; sha256 1f47a78b…a78a). Sweep on 6 live frames (4 cars parked: sedan, dark car, white
    hatchback, pickup; the silver car was away): yolov8n at 1280/1920 and yolov8s at 1280 box the hatchback + dark
    car as one; **yolov8s at 1920 px separates them** (hatchback 0.89, dark car 0.56). Ultralytics NMS iou 0.5 vs 0.7:
    no difference, kept 0.7. A ~0.25 false box across the front car's roof at 1920 → Mode 5 conf default **0.30**.
    Defaults now: Detector "🚗 COCO vehicles · yolov8s.pt" (yolov8n still listed), Vehicle inference 1920 px when
    yolov8s is selected (1280 otherwise), conf 0.30; the hybrid now pairs the camera model with yolov8s.
    Cost: ~1.5 s/frame (≈0.5 inferences/s) on this CPU, fine with hold 15 s / exit_hold 60 s.
    Real Mode 5 loop (70 inferences): 4 slots = the 4 real cars, each separate (sedan, hatchback, dark car, pickup),
    BUSY 4, no exceptions. Single frames still miss a back-row car now and then; the exit buffer covers it. 27 tests.
24. **2026-10-19 Entry/exit history & event tracking, UI-free service layer** (FastAPI/React-ready, none started):
    * `history.py` `HistoryDB`: new append-only table `events` (id, camera=scope, slot_id, event entry/exit, ts,
      time, session_id, note), written by `apply()` and by `reset()` (exit "closed by slot map reset"); an older DB
      gets its events backfilled once from `history` on first open (additive; the live DB is migrated the first
      time the live app opens it). JSON-ready readers: `sessions(limit, open_only)`, `events(limit, since_id)` (poll
      for new rows), `summary()`, `iso(t)` (ISO 8601 with offset, camera tz); `events_csv()`. Old `recent()`/`csv()` kept.
    * New `src/carparking/monitor.py` (no Streamlit import): `ParkingArea` (cuts + ROI → mask; `parse_roi` moved
      here), `ParkingMonitor(detector, db, area, targets, sliced, conf, **tracker kwargs)`: `.start(shape)` loads the
      slot map (skips saved slots outside the area, never reuses their IDs), `.process(frame, t)` → `FrameResult`
      (vehicles, others, events with entry/exit/busy/free + logged flag, `.to_json()`), `.state()` → slots + counts
      JSON. Intended API mapping is in the module docstring.
    * `demo_app.py`: Mode 5 loop now only reads, calls `ParkingMonitor`, draws and displays; `lot_mask` delegates to
      `ParkingArea`. New section **"🅿️ Parking History & Logs"** (idle and live): tabs "Sessions (In / Out)" (Session,
      Slot, In Time, Out Time, Duration, Status, Note) and "Event log (entry / exit)" (Time, Slot, 🚗 IN / 🅿️ OUT,
      Session, Note), each with a CSV download (file = data as of the last rerun); counts caption; refreshed live
      after each logged event. The old single download button moved into the tabs.
    * Tests: 32 (event log + session links + ISO/JSON + since_id + reset + backfill; tests/test_monitor.py: area
      filter, target classes, person shown-only, entry/exit logged, state/to_json JSON, ignored saved slots).
      End-to-end on a read-only backup copy of the live DB: backfill per scope (e.g. 77 sessions → 149 events), UI idle
      and live (40 inferences): 4 entries logged and shown, no exceptions.
    * Note: the user's real history is under camera name "Camera 1" (the UI default), not "Front Parking 02".
25. **2026-10-19 Fixed (pre-defined) slots mode + editor**, alongside Dynamic (default, unchanged):
    * New `src/carparking/fixed_slots.py` (no Streamlit): `FixedSlotStore` = `config/fixed_slots.json` keyed by
      camera name (`$CARPARK_FIXED_SLOTS` overrides; atomic write; slots `{"id", "name", "poly"}` as frame fractions);
      `validate_slots` (unique ids >= 1, >= 3 points in 0..1, non-zero area); `parse_points` / `format_points`
      ("x1,y1,x2,y2" rectangle or "x,y; x,y; …" polygon); `match_score` rules `ground` (box bottom-centre, 10% above
      the bottom edge, inside the slot; default, low-angle views), `center`, `iou`; `FixedSlotTracker(SlotTracker)`:
      learning off, one-to-one matching by rule; hold / exit_hold / occlusion / events = SlotTracker's;
      `draft_slots(dets)` = one rectangle per vehicle.
    * `history.py`: only change = `SlotTracker._match()` hook (the same IoU greedy as before), overridden by fixed.
    * `monitor.py`: `ParkingMonitor(fixed=[slots], match_rule, match_iou)` → FixedSlotTracker, nothing learned or
      saved to the slots table; `state()` adds "mode" and slot "name".
    * `demo_app.py` Mode 5: sidebar "Slot mapping mode" Dynamic (Auto) | Fixed (Pre-defined); Fixed adds "Fixed slot
      matching" (rule, IoU threshold), hides the learning controls, uses history scope "<camera> | <detector> | fixed",
      "Reset slot map" becomes "Close open sessions". "✏️ Fixed slot editor": 📷 Grab latest frame (the running stream
      also supplies it), ✨ Draft from detections, ↩️ Revert to saved, 🗑️ Clear table, editable table (ID, Name,
      Points; add/delete rows), live preview with a 0.1 grid and the parking area dimmed, per-row errors, 💾 Save Slots
      (disabled while invalid or unchanged). No slots saved → warning instead of the stream. Overlay shows slot names.
      No canvas: no drawing component is installed in the image (pip-installing into the container would not survive
      a rebuild); coordinates via table + grid preview + drafting.
    * Tests: 47 (15 new in tests/test_fixed_slots.py: parsing, validation, store per camera, ground vs whole-box
      matching, no learning + debounce, hidden back slot stays BUSY, monitor fixed mode → same sessions/events, draft).
      End to end (temp DB + temp json): dynamic default unchanged (BUSY 4); fixed: grab → draft 3 slots (yolov8s
      missed the dark car in that frame) → save → live 30 inferences: BUSY 3, 3 entries in the "| fixed" scope.
26. **2026-10-19 Fixed slot editor = mouse drawing canvas** (`streamlit-drawable-canvas` 0.13.0, published
    2026-09-04 by its original author; a Fabric.js v6 rewrite on Streamlit components v2, needs streamlit>=1.53 and
    renders inline, no iframe; 0.9.x does not fit Streamlit 1.65).
    * New `requirements.txt` (there was none; the image's Dockerfile is not in this folder and installed
      opencv-python numpy streamlit ultralytics sahi + CPU torch inline): adds `streamlit-drawable-canvas>=0.13,<0.14`.
    * Installed into the running container (`docker exec carparking pip install ...`) + `docker restart carparking`
      (components v2 register at server start). Survives restarts, NOT a container re-create from the image:
      rebuild the image with requirements.txt.
    * New `src/carparking/fabric_shapes.py`: `fabric_to_polys(json_data, w, h)` (Polygon/Rect -> corners via Fabric's
      transform: origin, scale, rotation, stroke, pathOffset; clipped; lines/free drawing/tiny click-rects skipped),
      `polys_to_fabric()` (saved slots as editable Polygons for initial_drawing). `fixed_slots.relabel()`: drawn
      shapes keep id/name by IoU with the previous ones (moving a slot keeps its history ID), new shapes get next id.
    * `demo_app.py`: the canvas is the only editor (typed-coordinates table and grid preview removed on the user's
      request): camera frame (parking area dimmed) as background, 960 px wide; Shape ⬠ Polygon (click corners,
      click the first corner again to close) / ▭ Rectangle (drag); toolbar edit toggle = move / resize / rotate /
      drag a corner / delete, undo / redo / reset; table below for ID + Name; Save Slots writes the canvas shapes
      (json_data → fractions). Without the package: an error with the install command, no fallback.
    * Verified in a real browser (Playwright + headless Chromium, throwaway container): 4-click polygon + dragged
      rectangle on the live frame saved with max corner error 0.0018 / 0.0024 of the frame; on the live app
      (8501, no save): canvas over the camera frame, polygon drawn → slot row "1 S1", no errors. 51 tests.
27. **2026-10-19 One-by-one slot edit / delete** in the fixed slot editor:
    * Below the canvas, one row per slot: "ID · name", corners, status (✅ saved / 🟠 changed, not saved / 🆕 new,
      not saved), **✏️ Edit** and **🗑️ Delete**; slots removed from the canvas but still saved get their own Delete.
    * ✏️ Edit = single-slot mode: canvas with only that slot editable (orange; the others grey, locked), the
      toolbar's Edit toggle moves / resizes / drags corners; a corner table (x, y, 0..1, rows addable) for
      fine-tuning, synced both ways with the canvas; name field; **💾 Save this slot** (`FixedSlotStore.upsert_slot`,
      writes the file at once, other slots untouched), ✖️ Cancel, 🗑️ Delete this slot.
    * 🗑️ Delete = `FixedSlotStore.delete_slot` at once + `HistoryDB.close_slot()` closes that slot's open parking
      session (exit event, note "slot deleted in the editor"); unsaved slots are just removed from the canvas.
    * 💾 Save Slots (all) and every per-slot action run as button callbacks, so the header / list show the new
      state immediately; confirmations stay visible 8 s (the remounted canvas reruns the page at once).
    * Engine: `FixedSlotTracker.set_slots()` swaps in an edited map keeping the state of kept slots;
      `ParkingMonitor.update_fixed()` applies it while running and closes deleted slots' sessions (API-ready; the
      Streamlit loop restarts on every edit anyway). `fabric_shapes.object_poly()`, `polys_to_fabric(locked=)`.
    * Tests 55 (store upsert/delete per slot, set_slots keeps state, update_fixed closes the session, object_poly).
      Browser (Playwright, isolated server): draw 2 rects → Save (header "2 saved slot(s)" at once) → ✏️ Edit
      slot 2: dragged +0.05 on the canvas, renamed "Pickup", Save this slot → file: slot 2 moved/renamed, slot 1
      unchanged → 🗑️ Delete slot 1 → file: only slot 2 → stream: no errors.
28. **2026-10-19 Dynamic mode: no phantom FREE slots between tightly parked cars** (`history.py`):
    * Cause (live map "Camera 1 | yolov8s"): slot 5 was learned from the red car's box stretched down over the
      white sedan in front (ground point on the sedan's body). It never held a car of its own; with the stretched
      box gone it went FREE between the two cars on every start. Same pattern: slot 3 (far car's box over the grey
      car), slot 11 (a 0.03-wide fragment lying on the far car).
    * `phantom_reason(poly, others)`: 'stretched' = holds the box of a vehicle behind it (≥ 70% inside) and that
      box + the vehicles in front cover ≥ 80% of it; 'no room' = its ground point (bottom centre) lies between two
      side-by-side vehicles whose ground points are < 1.5 × their mean width apart. Measured on live frames:
      phantoms 0.86–0.93, real cars ≤ 0.69 (the SUV holding the dark car behind it). Rectangles overstate
      occlusion, so only boxes *inside* the candidate can explain it (an earlier "visible part covered by anyone
      behind" version dropped the real red car).
    * SlotTracker: phantom detections dropped before matching/learning (`.dropped`); candidates learned only where
      the ground point rests (moves > 20% of the box = start over); an unmatched non-BUSY slot that is a phantom is
      UNKNOWN (sticky until a detection matches it, `.phantoms`) and retired after `retire_hold` (= learn_hold)
      unless it has ever held a session (`proven`); a non-BUSY slot ≥ 90% inside a vehicle in front is HIDDEN
      (UNKNOWN, not FREE). Fixed maps unchanged (`suppress_phantoms=False`).
    * HistoryDB: `slots.retired` column (auto-migrated), `retire_slots()` (closes an open session, note "phantom
      slot retired"), `proven_slots()`, `max_slot_id()` (retired IDs never reused). Monitor: `FrameResult.retired`,
      state() "phantom" per slot + count. UI: "NO ROOM (phantom)" / "HIDDEN" labels, "phantom boxes ignored N".
    * Tests 64. Live replay (throwaway container, copy of the live DB, yolov8s 1920, Mode 5 defaults, ~60 s):
      7 real cars BUSY, 0 real boxes dropped, slot 5 retired after 20 s, slots 3 / 11 UNKNOWN, no FREE between
      cars. Left as is: slot 6 = the small green sign on the building wall (FREE; outside the row; a "Reset slot
      map" or an ROI cut removes it); proven phantom slot 3 keeps its old open session.
29. **2026-10-19 Dynamic mode: a slot with a car on it is never FREE** (`history.py`, shared occupancy):
    * Cause: matching is one-to-one. Live slot 12 (learned from the rear half of the white sedan) lost the sedan's
      box to slot 2 and read FREE under the car; slot 13 (the part of the sedan seen above the hatchback) and 11 (a
      fragment on the far car) likewise.
    * An unmatched slot is observed BUSY (`.shared`) when a vehicle's ground point is in its bottom band, a vehicle
      at its depth covers ≥ 35% of it (`share_cover`), or ≥ 85% of it lies within the vehicles at its depth or in
      front (`_inside_vehicles`). Checked before the FREE / phantom / hidden logic; BUSY after the usual hold (a
      passing car logs nothing), a pending FREE countdown stops at once. Cars only beside (33%), in front
      (occlusion) or behind don't count; the red car's bay 55% behind the sedan still goes FREE when empty. Replaces
      item 28's "≥ 90% behind a car = HIDDEN". Fixed maps unchanged (`share_cover=None`).
    * Tests 66. Live replay (DB copy, ~70 s): slots 12, 13, 11 BUSY throughout with the 7 real cars' slots; FREE
      only slot 6 (wall sign) and 15 (tiny box at the far end, beyond the row).
    * Side effect of the old learned map: overlapping duplicate slots now all read BUSY, so BUSY counts 11 for 7
      cars. "Reset slot map" lets the map re-learn with the phantom filter (item 28) on.
30. **2026-10-20 Hard-wall cuts: precise controls** (`demo_app.py`, Modes 4 and 5): the Cut top / bottom / left /
    right sliders (0.05 steps) became `cut_input()`: a full-width number input in % of the frame (0–60, type a
    value or use its own −/+ for 1 %) with **−5 %** / **+5 %** buttons under it. Buttons change the value in an
    on_click callback, so the same rerun applies the cut and restarts the stream; other settings keep their values;
    the value also lives in `<key>_value` so it survives switching modes. Keys `cut4_*` / `cut5_*`; defaults
    unchanged (Mode 4 top 25 %, Mode 5 right 40 %). Two columns of inputs or four buttons per row don't fit the
    sidebar (Streamlit hides the input's −/+ when narrow, button labels truncate), hence one cut per row.
    Tested: AppTest (±5, ±1, clamping at 0 and 60, conf / camera name kept, mode switch keeps values, no exceptions);
    browser on the live camera (isolated server, temp DB): stream running, Cut top +5, +5, then the input's + →
    11 %, the cyan edge moved down 4.9 % / 4.9 % / 0.8 % of the image, stream stayed on, Cut right 40 kept, no errors.
31. **2026-10-20 Dynamic mode: strict hard wall** (`monitor.py`, `history.py`):
    * Cause: live slot 17 (`Camera 1 | yolov8s`, x 0.00–0.33 over the scaffolding) passed the old test, which
      only required the box *centre* inside the parking area (centre x 0.167: inside any left cut < 16.7 %).
    * `monitor.in_area(mask, poly)`: centre **and** ground point (bottom centre) inside, and ≥ 90 % of the box.
      Dynamic mode filters detections with it (nothing outside is learned or matched) and passes it to the tracker
      as `SlotTracker(area=...)`: slots outside are set aside in `.outside` (not shown, matched, logged; IDs and
      state kept, so they come back if the wall moves out again; not retired, since a wall change shouldn't delete
      history). `SlotTracker.set_area()` / `ParkingMonitor.set_area()` apply a changed area while running
      (API-ready; the Streamlit stream restarts on every change anyway). `mon.ignored` = slots outside. Fixed mode
      keeps the centre test.
    * Tests 68 (strict in_area incl. slot 17 at 15 % and 5 % cuts, monitor never learns/shows/logs it, set_area
      in and out). Live replay (DB copy, Cut left 15 % + right 40 %): slots 14, 15, 17 outside, 7 real cars BUSY.
32. **2026-10-20 Same request again; why slot 17 was still visible**:
    * The live app's stream loop was still running the pre-fix code: it learned slot 18 (x 0.589–0.605, straddling
      the 40 % right wall) at 16:35, 3 min after item 31 landed. A running stream only picks up `src/` changes on
      the next rerun (stop/start the stream or reload the page).
    * The cuts were per browser session only (no `cut_*` rows in `settings`): every reload put Cut left back to 0,
      where slot 17 really is inside. Mode 5 cuts are now saved per camera name like the polygon
      (`settings` keys `cut_top/bottom/left/right`, widget keys `cut5_<side>_<camera>`), written on every change.
      AppTest: left +5×3, right −5, top typed 7 → new session (reload) restores 7/0/15/35; another camera name
      keeps its own (defaults); back again → restored. 68 tests.
33. **2026-10-20 One box per vehicle: stricter duplicate merge** (`yolo_engine.merge_dets`):
    * Live check first: on 12 frames the silver hatchback got one box each time; the second box over it is the
      white sedan parked behind (its box reaches down over the silver car's roof, IoU 0.29), a real car. A blunt
      low-IoU NMS would delete it, so no.
    * Gap: a second box on a *horizontal slice* of the same car (roof + windows, or lower body) survived: the
      fragment rule needed level bottom edges, which protects cars parked behind. New `_same_column()`: the smaller
      box ≥ 80 % inside the other and spanning ≥ 85 % of its width = the same vehicle (any vehicle class). Cars
      behind are narrower and offset in this view (dark car behind the SUV 39 %, the test_engine behind case 65 %).
    * IoU threshold 0.5 → 0.45 (merge_dets default, `whole_iou`, HybridDetector): the most overlapping real pair
      on Front Parking 02 is the SUV / red car at 0.37.
    * Tests 69 (silver split into roof + lower 'truck' boxes → one car, sedan kept; the 7-car live frame stays 7).
      Live (6 frames): silver = 1 box every frame, 6–8 vehicles per frame as before.
34. **2026-10-20 Tiny stray slots retired; BUSY = real vehicles** (`history.py`, `monitor.py`):
    * Cause: after the user's "Reset slot map" (17:05) the map was 7 cars + slot 8, learned 17:11 from a stray
      box by the stairs (x 0.29–0.33, 0.14 % of the frame). It lies inside the white sedan's box, so item 29's
      "slot within cars = occupied" rule read it BUSY: 7 vehicles, BUSY 8.
    * `is_tiny(poly, refs)`: area < 15 % of the median area of the slots / vehicles at the same depth (ground
      points within half the reference's height, IoU < 0.3), relative because far cars are small too (far car
      slot 0.55 % of the frame). Live: stray 6 %, smallest real car (dark car behind the SUV) 28 %.
    * SlotTracker (`min_size=0.15`, off for fixed maps): tiny vehicle boxes dropped before matching / learning;
      tiny slots retired at once, proven or not (`retire_notes`; the monitor closes the open session with "tiny
      slot retired …").
    * Tests 72 (live map: slot 8 retired on frame 1, session closed, BUSY 7 with 8 boxes incl. a stray; is_tiny on
      the live row incl. far / lone cars; stray never learned). Item 29's test: fragment slot 11 is now retired
      (tiny) instead of BUSY. Live replay (DB copy, saved walls top 10 / left 10 / right 40 %): slot 8 retired on
      the first frame, then BUSY 7 = 7 vehicles for the whole run.
35. **2026-10-20 Emptied bays no longer stuck BUSY; out_time = departure** (`history.py`, Mode 5 UI):
    * Cause, measured on the live map (each car removed in turn, the others kept): 6 of 7 slots stuck after their
      car left. (a) Occlusion: HIDDEN (kept BUSY, exit countdown paused) whenever ONE box in front covered ≥ 30 %;
      in this low-angle row every bay is 42–50 % behind the car in front, so it never ended. (b) Item 29's
      `_inside_vehicles` forced the dark car's bay (100 % inside the SUV's box) BUSY. A restart "fixed" it because
      hidden applied only to slots already BUSY, and every slot starts unknown.
    * Fix: `_hidden(slot, cars, cover)` = the boxes in front *together* cover ≥ `occlusion_cover` (default 0.3 →
      0.6); `hidden_max` (120 s, new UI slider "Max seconds a hidden slot keeps its state"): after that long without
      any vehicle in it the slot counts as empty, hidden or not, and the FREE countdown runs from the first frame
      without the car (`s["gone"]`), so the exit = the departure (event `expired`, note "no vehicle seen in it
      while partly hidden (exit = last seen)"). `_inside_vehicles` removed: vehicles in front are occlusion, never
      occupancy. Hidden also covers not-yet-confirmed slots (UNKNOWN, not FREE, while mostly covered; also capped).
      Occlusion default in the UI 0.6. Applies to fixed maps too (same base tracker).
    * Tests 75: every car of the live row leaving in turn → its slot FREE after the exit buffer, session closed
      with out_ts = departure; half-hidden bay (56 %) exits normally; a really hidden bay keeps BUSY up to
      hidden_max then frees itself, exit timed at last seen; DB note. Updated: occlusion tests use a van covering
      87 %; fixed back bay is unknown (not FREE) while behind the front car; slot 13 hidden instead of forced BUSY.
    * Live replay (DB copy; the user had reset the map again: 6 slots, 6 cars): BUSY 6 = 6 vehicles, nothing hidden,
      no false exit over ~1 min.
36. **2026-10-20 `TypeError: SlotTracker.__init__() got an unexpected keyword argument 'hidden_max'`** (live app):
    * `SlotTracker` already had `hidden_max` (item 35). Cause: `_drop_stale_modules()` in `demo_app.py` reloaded
      only the edited file (`history.py`); unchanged `monitor.py` / `fixed_slots.py` stayed imported, still bound
      to the *old* `SlotTracker`, while the rerun page passed the new argument through them.
    * Fix: if any `carparking` module is stale, the whole package is dropped and re-imported together (and
      `st.cache_resource` cleared, as before). No lazy `carparking` imports exist, so an unedited rerun reloads
      nothing.
    * Verified with AppTest on a project copy (same process across reruns, like the server): Mode 5 run → edit
      history.py only → rerun: monitor's and fixed_slots' `SlotTracker` is the new class, no exception; rerun
      without edits: nothing reloaded. Tests 75. Any future edit to one `src/` file is picked up the same way.
37. **2026-10-20 "FREE 0, no green boxes"** (`viz.py`, Mode 5):
    * Finding: rendering and counts were consistent (viz.draw drew every slot, FREE in green; HUD counted the same
      states). The live map was reset at 18:14 (local) after the silver hatchback and white sedan had left, so
      their bays were deleted and, in Dynamic mode, a bay exists only once a car has parked in it: 6 slots = 6
      vehicles (5 cars + motorbike) → FREE 0 was correct. Learned slots are never dropped for being vacant.
    * Real defect: on the 2560 px stream (shown at 1280 px) the 2 px outlines and 0.4 labels were near invisible.
      `viz.draw`: line width / text / HUD scale with the frame width (`_scale`), FREE bays filled green (25 %),
      `labels=True` (Mode 5) tags each slot "3 FREE" / "5 BUSY"; `hud_text()` builds
      "FREE: n | BUSY: m | TOTAL: t (| UNKNOWN: u)".
    * Tests 78 (new `tests/test_viz.py`: colours per state, FREE fill, BUSY unfilled, HUD text, 2560 px scaling, a
      car leaving → its slot stays in the map, green, HUD FREE: 1). Live frame (DB copy + the silver car's old bay
      added): green-filled "7 FREE", red "BUSY" tags, HUD "FREE: 1 | BUSY: 6 | TOTAL: 7".
    * For bays that should show green before any car has parked there: Fixed (Pre-defined) mode, or don't press
      "Reset slot map" while bays are empty (it deletes the bays of absent cars).
38. **2026-10-20 One vehicle box = one slot (bike's bay stuck "3 BUSY"); hidden_max 120 → 30 s** (`history.py`):
    * Cause: item 29's shared occupancy counted any vehicle at the same depth covering ≥ 35 % of a slot. The maroon
      car's box (matched to slot 1) covers 44 % of the motorbike's bay (IoU 0.14, centre outside), so slot 3 stayed
      BUSY after the bike left.
    * Fix: a detection matched to a slot occupies that slot only. A detection that matched no slot (shape changed,
      IoU < match_iou) may occupy ONE unmatched slot (`_claims`): its best among those holding its centre or with
      IoU > `share_cover` (0.35 → 0.5). Item 29's "slot under the sedan's rear half BUSY" no longer holds (one box,
      one slot); duplicate slots like that come from old maps (learning filters now prevent them).
    * `hidden_max` default 120 → 30 s (tracker and the UI slider, now 10–1800 s, step 10).
    * Tests 80: the live bike / maroon case (slot 3 FREE at the departure, slot 1 BUSY); one box → one slot; an
      unmatched box holds the bay with its centre, only one; default hidden_max 30 s frees a hidden empty bay.
      Live (DB copy, 19:25 local): bike gone → green "3 FREE", maroon "1 BUSY", HUD FREE 3 | BUSY 3 | UNKNOWN 1
      (slot 2: the white SUV is still parked but detected narrower, matching slot 4, so its own bay reads UNKNOWN).
39. **2026-10-20 Duplicate slots over one car (slot 1 + slot 11 on the white sedan)** (`history.py`, `monitor.py`,
    `demo_app.py` overlay):
    * Live map 09:37: 11 slots, 5 of them second slots on a car that already has one: 1 (the maroon car's old bay)
      and 3 (the bike's) lie on the white sedan (slot 11) that now parks further forward; 4 and 8 on the dark car
      (slot 2); 10 on the far car (slot 9). Their IoU with the real slot is only 0.13–0.36 (different size/depth),
      so no IoU rule alone catches them; what does: 67–100 % of them lies inside that one car's box, their ground
      line on its body.
    * Learning (`duplicate_of`, `dup_iou` 0.4): a candidate overlapping a slot by IoU > 0.4, or lying ≥ 80 % in it
      (or it in the candidate) with the ground points at the same depth, never becomes a new slot. A larger one
      (≥ `grow` 1.25 × the area: a car in the bike's bay) replaces the slot's box, same ID (`.updated_slots`,
      saved by the monitor).
    * Load / refresh NMS (`dedupe_slots`, `merge_iou` 0.5): slots overlapping by IoU > 0.5 are merged at start (and
      after a slot grew), keeping the proven one, then the BUSY one, then the oldest; the other is retired with
      "duplicate slot merged into slot N" (closes its open session). The current live map has no such pair.
    * Runtime (`covered_by`, `cover_share` 0.6, `cover_hold` 60 s): a slot with no vehicle of its own that lies
      ≥ 60 % inside another vehicle's box, its ground point on it, while that vehicle matches its own slot, is a
      duplicate (`.covered`, overlay "DUPLICATE of N", state() "duplicate_of"): keeps its state, never FREE, no
      entry/exit; merged into that slot after 60 s. Protection: a car behind another on this low-angle view also
      lies wholly inside its box (dark car in the SUV's), so two slots that have held two separate vehicles in one
      frame are never duplicates (`.distinct`, new table `slot_pairs`, cleared by "Reset slot map").
    * One box → one slot was already strict (greedy one-to-one match + `_claims` gives an unmatched box one slot).
    * Fixed maps: all of this off (drawn by hand).
    * Tests 89 (+9): duplicate candidate rejected (the old code made slot 2), narrower box no new slot, bike → car
      grows the slot, NMS on load (+ proven kept, IoU 0.45 kept), one box never two BUSY, monitor retires load
      duplicates and closes their session, live 09:37 geometry (1, 3 merged into 11, never FREE), bay behind a
      parked car not a duplicate, seen-together pairs survive a restart. Two old tests now expect "duplicate"
      where they expected HIDDEN / FREE on the sedan; the phantom tests run with the merge off.
    * Live replay (DB copy, 80 s): covered {1: 11, 3: 11, 4: 2, 8: 2, 10: 9}; at 62 s 1, 3 → 11 and 4, 8 → 2
      merged; 10 got its own detection (conf 0.38) and stayed. The live DB is untouched: the app does this itself
      within a minute of the stream restart.
40. **2026-10-20 Car hidden behind the white sedan stuck HIDDEN / UNKNOWN; zoom re-check, exit 30 s**
    (`yolo_engine.py`, `monitor.py`, `history.py`, `demo_app.py`):
    * Finding: not NMS. On the live frame (10:19) YOLO gives the car behind the sedan (only roof and tail lights
      visible) NO box even at conf 0.05 with in-model NMS iou 0.95: all extra boxes are the sedan. merge_dets
      already keeps it apart from the sedan and the hatchback (different depth, 74 % of the width): new test with
      the live boxes. Mode 5 conf was already 0.30 (not 0.45). NMS and the global threshold unchanged.
    * `YoloDetector.detect_region(frame, box, conf=0.30)`: YOLO on a crop around a slot (padded 40 %), imgsz = the
      crop's own size (640–1280; shrunk to 640 the car is found on some crops only); boxes cut by the crop edge
      dropped (pieces of the cars around). HybridDetector: every member, merged. Live: the hidden car at 0.60–0.68.
    * `ParkingMonitor` zoom re-check (`zoom_max` 2 slots per frame, round robin, 0 = off; `zoom_conf` 0.30): only
      slots that are BUSY, or UNKNOWN while HIDDEN / DUPLICATE, and got no box from the whole frame. A zoom box
      counts only if its centre is inside that slot, IoU ≥ match_iou, and it repeats no known box (merge_dets);
      then the usual filters (hard wall, phantom, tiny). FREE slots are never re-checked (no ghost BUSY). Overlay
      label "[zoom]". Cost: ~1 s per frame with a re-check on CPU (2.7 vs 1.6 s).
    * Tracker: a HIDDEN frame no longer restarts a pending BUSY while the car was seen within `hold` (it is found
      on some frames only); one sighting alone never makes an entry. Exits unchanged (visible emptiness, or the
      hidden_max cap). Item 39's duplicate rule now skips BUSY slots (a car that left a bay now behind a van frees
      it on the 30 s timer instead of being held 60 s as a "duplicate"); a FREE slot lying on a car shows UNKNOWN.
    * 30 s exit: UI exit buffer default 60 → 30 s (hidden_max already 30): a departed car's slot is FREE 30 s after
      the last box (whole frame or zoom), visible or hidden, the exit logged at the departure.
    * Tests 95 (+6): detect_region crop / edge cut, zoom keeps the hidden car BUSY (and off / misfit → exit), zoom
      never makes a FREE slot BUSY, departure → FREE within 30 s (visible and behind a van), live merge boxes,
      intermittent sightings → BUSY / single sighting → nothing; item 39's live test updated (BUSY slot 1 exits at
      the 30 s cap, then merges).
    * Live replay (DB copy, 90 s): slot 1 (the hidden car) BUSY from 16.5 s on, zoom 0.66–0.68 on the frames the
      whole frame missed it; 4 cars = BUSY 4 (1, 2, 6, 11), slots 5 and 7 stay FREE; HUD FREE 2 | BUSY 4 | TOTAL 6.
41. **2026-10-20 Docker hardening (roadmap step 1)**: `Dockerfile`, `docker-compose.yml`, `.env` / `.env.example`,
    `.dockerignore`, `.gitignore`, pinned `requirements.txt`; no detection / tracking code touched.
    * Dockerfile: python:3.10-slim; torch 2.14.1+cpu / torchvision 0.29.1+cpu from the CPU index (PyPI as extra
      index: the CPU index has typing-extensions only as an sdist that fails to build); requirements pinned to the
      versions of the old image; opencv-python (pulled by ultralytics / sahi) swapped for opencv-python-headless
      5.0.0.93 (FFmpeg backend kept). Copies only app.py, src/, scripts/, models/, yolov8s.pt (.dockerignore
      allow-list). Stage `test` runs the 95 tests + pyflakes; `runtime` depends on it (a failure fails the build).
      Runtime: non-root user app (uid 1000), /data volume (CARPARK_HISTORY_DB=/data/parking_history.db,
      CARPARK_FIXED_SLOTS=/data/config/fixed_slots.json, CARPARK_DATASET_RAW=/data/dataset_raw), HEALTHCHECK on
      /_stcore/health. Image stomeetings/carparking:local 2.63 GB (old hand-built :latest 2.86 GB, left as is).
    * docker-compose.yml: service `app`, env_file .env, port ${CARPARK_PORT:-8501}, ${CARPARK_DATA_DIR:-./data}:/data,
      restart unless-stopped.
    * Secret: the camera URL with its password is gone from demo_app.py and collect_dataset.py (and the literal in
      test_mask_url, now dummy credentials, same assertion). New `rtsp.env_setting()`: $CARPARK_RTSP_URL, else the
      project's .env, so the bind-mounted dev container (no -e) keeps its camera; .env holds the URL, excluded from
      the build context and git. Without a URL Mode 5 says how to set it; collect_dataset.py exits with a message.
    * Verified: build OK (95 passed in the build); image: uid 1000, headless cv2, no password inside, yolov8s
      inference OK; `docker compose -p carparking-test` on port 8502 with a scratch data dir: healthy, HTTP 200, the
      URL arrives from .env, 2560x1440 frames from the camera, Mode 5 loads (AppTest, no exceptions), the DB is
      created in the host folder; stack removed afterwards. The live container `carparking` was not touched.
    * Not done (needs the user): switching the live app to compose (data migration below). The image was not pushed.
42. **2026-10-20 Headless background worker (roadmap step 2)**: `src/carparking/live.py` (new), `scripts/worker.py`
    (new), `scripts/demo_app.py` Mode 5 = viewer, `docker-compose.yml` worker + ui. No tracking logic changed.
    * Before: the Mode 5 loop ran inside the Streamlit script run, one per browser tab: no tab = no logging, two tabs
      = two monitors writing the same DB (double entries), any widget click restarted the tracker.
    * `live.py`: `LiveStore` (tables in the history DB: `live_config` settings JSON + version, `live_state` lease
      owner + heartbeat + status snapshot + overlay JPEG + raw JPEG, `live_commands` reset / snapshot / restart);
      `Worker` (step() = one pass: lease, settings poll, requests, newest frame -> ParkingMonitor -> publish; loop()
      reports errors and retries, never crashes, releases the lease on exit); `build_detector`, `build_monitor`,
      `render_overlay` (moved out of the page), `stream_url`, `detector_label`.
    * One worker at a time: claim() is a lease (BEGIN IMMEDIATE; heartbeat < 20 s = taken); a second worker waits on
      standby and takes over when the first stops. A fixed-map edit is applied in place (update_fixed), other
      setting changes restart the session; fixed mode without a saved map idles (never learns into its scope).
    * `scripts/worker.py`: run / `--check` (heartbeat age, compose healthcheck) / `--max-frames`; SIGTERM releases.
    * Page (Mode 5): every widget keyed and seeded from the saved settings (a new tab shows what runs and can't push
      defaults; a change in another tab arrives on the next rerun); "Start RTSP stream" -> persistent toggle
      "Monitoring (background worker)"; viewer fragments (status / overlay / metrics every 1 s, history every 5 s);
      status 🟢 Worker connected (heartbeat age) / 🔴 Worker down / ⏸️ off / applying settings / older code + 🔄
      Restart worker; snapshot and reset go to the worker as requests (direct when no worker beats); the fixed slot
      editor draws on the worker's raw frame. `CARPARK_WORKER=embedded` (default, plain `streamlit run` / the
      single dev container): the page starts scripts/worker.py as its own process when monitoring is on and no
      worker beats (log: logs/worker.log next to the DB); `external` (compose ui): never.
    * Compose: services `worker` (python scripts/worker.py, healthcheck --check) and `ui` (CARPARK_WORKER=external,
      port), same image, same /data.
    * Tests 105 (+10 test_live.py): settings versioning, lease, logging without a page, off/idle + session restart,
      fixed edit in place, reset/snapshot requests (old ones not replayed), two workers never log twice, errors
      reported, stream URLs, overlay JPEG. Build: 105 passed inside the image.
    * Verified (compose -p carparking-test, port 8502, scratch copy of the DB): both healthy; Monitoring switched on
      through the page (AppTest) and the page closed: the worker ran alone (19 frames, BUSY 3 / FREE 2, overlay
      published); `docker compose stop ui`: worker went on (frames 23 -> 32, heartbeat fresh, healthy); ui back:
      "🟢 Worker connected · heartbeat 0.8 s ago · … 2560×1440". Stack removed afterwards.
43. **2026-10-20 Slot 1 FREE with 4 cars present ("BUSY 3 | FREE 2")** (`history.py`, `monitor.py`):
    * Cause: the car between the white sedan and the hatchback (roof and tail lights only) gets no box from the whole
      frame; slot 1 (its bay) was hidden by the sedan, freed after hidden_max 30 s without a box, and FREE slots were
      never zoom-checked (item 40 safeguard). Probe 12:2x: zoom on slot 1 finds it in 8/8 frames (0.33-0.59, IoU
      0.62). Also: the zoom skipped slot 1 because the sedan's box overlaps it at IoU 0.37 >= match_iou, although
      that box is matched to slot 11.
    * Not done as asked: "a box covering >= 30 % of a slot (or its centre inside a >= 0.8 box) makes it BUSY" for
      every slot it covers: here only the sedan's box covers slot 1 (75 %), so it would mean one box holding two
      slots (contradicts item 39's rule and test), and every bay behind or beside a parked car would stay BUSY after
      its own car left (items 35/38; tests test_neighbour_beside_is_not_an_occlusion, test_hidden_slot_..._expires,
      test_shared_occupancy_is_not_triggered_by_cars_beside_in_front_or_behind fail with it).
    * Done: (1) zoom re-check also on FREE slots that the vehicles in front hide (occlusion_cover), at the stricter
      `zoom_free_conf` 0.40 (visible empty bays still never re-checked); which slots already have a box now comes
      from the tracker's one-to-one match, not raw IoU. (2) `_claims` (a box that matched no slot holds ONE slot):
      centre inside, or standing in the slot's column at its depth (centre x within the slot, bottom edges within
      0.3 x slot height) with IoU >= 0.25 (`share_cover` 0.5 -> 0.25) or covering >= 30 % (`claim_cover`). (3)
      `exit_frames`: BUSY -> FREE also needs N consecutive processed frames without a box (UI default 15, tracker
      default 0 for the old tests). FREE -> BUSY already needs >= 2 frames (hold 15 s). 'van' was already a vehicle.
    * Tests 108 (+3): reshaped box of the slot's own car still occupies it, 15 consecutive empty frames before FREE
      (a blip restarts), the live fixture (4 cars, 3 boxes, slot 1 FREE -> BUSY via the zoom at 0.52: BUSY {1, 2, 6,
      11}; at 0.35 it stays FREE). Two depth/column guards were added after the old tests caught a van in front and
      a neighbour beside occupying bays.
    * Live (real stream + yolov8s, DB copy, slot 1 forced FREE): zoom 0.46-0.68 on most frames, slot 1 BUSY at
      22.7 s and stayed; FINAL BUSY 4 | FREE 1 | TOTAL 5 (slot 5 FREE).
    * Watch: with exit_frames 15 a departure frees after max(exit buffer 30 s, 15 frames): at the 1.5 s/frame of
      one process ~30 s, slower while the CPU is shared.
44. **2026-10-20 Viewer refresh 1 s -> 10 s, selectable** (`scripts/demo_app.py` only): the Mode 5 viewer
    fragments (status / overlay / metrics, and the history, which was 5 s) re-read the database every 10 s by
    default; "Viewer refresh" select slider 5 / 10 / 15 / 30 s + "🔄 Refresh now". Per browser tab (session state,
    key refresh5, deliberately NOT in the saved worker settings: any settings change restarts the worker's
    session). The worker is untouched and keeps processing every frame. Verified: AppTest (default 10 s, switch to
    15 s, no exceptions, worker settings version unchanged), 108 tests passed, pyflakes clean.
45. **2026-10-20 Step 3, Phases 1–2: shared settings builder, worker vectors, FastAPI backend, go2rtc** (Streamlit
    files untouched by user requirement: kept as permanent backup / comparison / demo on :8501):
    * Phase 1 (`live.py`): `build_settings(ui, root, db, rtsp_url, fixed_path)` computes the Mode 5 settings
      {"enabled", "ui", "run"} exactly as the Streamlit page stores them (labels, defaults, fallbacks, cuts/polygon
      from the camera's DB settings, fixed slots from the store, class lists from the weights); `detector_options`,
      `class_names`, `bay_weights`. `tests/test_settings_parity.py` renders the real page (AppTest) for 7 sets of
      widget values (defaults, tuned+walls+polygon, fixed+IoU, hybrid, bay, custom URL, stale values) and requires
      identical dicts. Worker: a new settings version with the same `run` (and on switch) no longer restarts the
      session (another UI re-saving); the snapshot carries `seq`, `slots` (polys, states, flags), `detections`
      (boxes as fractions, label, conf, src, kind) and `area` for client-drawn overlays.
    * Phase 2 (`api/main.py`, `uvicorn --factory api.main:create_app`): /api/health, /api/status, /api/frame.jpg,
      GET /api/settings (named values, version, effective run; camera URLs masked), /api/settings/options (choices,
      class lists, slider ranges), PUT /api/settings 🔒 (partial, validated like the sliders, version check -> 409,
      a masked URL sent back keeps the stored one, cuts/polygon per camera like the page, written via
      build_settings), POST /api/commands 🔒 (to the worker; reset directly if none beats), /api/sessions,
      /api/events?since_id, /api/summary, /api/video, WS /ws/stream (one poller, state per heartbeat, event per
      logged entry/exit, ?frames=1 JPEG per new frame; slow clients drop their own oldest). 🔒 =
      CARPARK_API_TOKEN (Bearer / X-API-Token; unset = read-only). Serves frontend/dist (SPA fallback) when built.
      Short-lived DB connections per request (thread pool), the hub's under a lock.
    * go2rtc 1.9.14 (`go2rtc.yaml`): the camera is HEVC on both channels (probed), so "parking" = sub-stream
      re-encoded to H.264 (~10 % of a core), "parking_hd" = main stream as sent (HEVC: Safari/Edge HW only); RTSP
      server on 127.0.0.1 only (its ffmpeg needs it); WebRTC 8555 tcp/udp with GO2RTC_CANDIDATE = 192.168.20.237.
      Option: set the camera's sub-stream to H.264 to drop the re-encode.
    * Compose: worker, api (:8080), ui (Streamlit :8501, unchanged), go2rtc (:1984, :8555). `.env`: added
      CARPARK_RTSP_SUB_URL, CARPARK_API_TOKEN (generated), CARPARK_API_PORT, CARPARK_GO2RTC_URL, GO2RTC_CANDIDATE;
      `.env.example` documents them. requirements: fastapi 0.151.0, uvicorn[standard] 0.54.0; tests: httpx 0.28.1.
    * Tests 126 (+18: 7 parity, 2 worker, 9 API): all pass inside the image build.
    * Verified (compose -p carparking-test, ports 8081/8502/1985, scratch copy of the DB): all healthy; settings
      without password, 401 without token; enabled=true re-sent -> no new version; status with slots/detections/
      area, frame.jpg; WS state per heartbeat + JPEG per frame (seq 19, 20, 21); API wrote v3 (hold 20, exit
      frames 12) -> Streamlit showed 20 / 12 and rendering it left v3 (no rewrite, no worker reset); go2rtc:
      snapshot 24 KB, parking.mp4 = H.264 (avc1), parking_hd.mp4 = HEVC. Not yet tested: WebRTC/MSE in a real browser
      (the player comes with the React app, Phase 3). Stack removed.
    * Found, not touched: container `carparking-worker-1` (default compose project, created 09:11 UTC, idle, its
      own new ./data/parking_history.db, no settings). Not started by this session's commands (all used -p
      carparking-test); the user decides whether to remove it and ./data.
    * Open: go2rtc's API/player on :1984 has no auth (LAN only; proxy it through FastAPI later); Phase 3 = React SPA.
46. **2026-10-20/21 Step 3, Phase 3: React SPA, secured video, live migration to compose** (Streamlit untouched):
    * Clean-up (user OK): removed the stray `carparking-worker-1` + `carparking-ui-1` (default compose project,
      idle) and the empty ./data they had created.
    * Security: `CARPARK_VIEW_TOKEN` (generated in .env) guards every read (status, frames, settings, history, CSV,
      video, WebSocket; header or ?token=; the admin token also views); `/api/auth`; go2rtc's API (1984) is no
      longer published: `POST /api/webrtc?src=` proxies WebRTC signalling (SDP offer/answer, viewer token) to
      go2rtc inside the compose network, only the media port 8555 is published. WebSocket: accept, then close 4401
      without a token (a close before accept reaches browsers as a bare 403). Config: secrets from .env only for
      the real environment (tests' dicts never read the real .env: a test caught the leak). CSV endpoints.
    * `frontend/` (Vite 8.3, React 19.3, TypeScript 7, Tailwind 4.3, lucide-react 1.52, framer-motion 14,
      recharts 3.10; pinned, ≥ 2 weeks old): CoreUI-bright look (light sidebar, white cards, indigo primary, the
      CoreUI status colours, KPI widgets). Dashboard: 70 % live view = WebRTC video (go2rtc "parking", H.264) +
      SVG overlay (slots coloured by state, dashed when hidden/duplicate, labels; vehicle boxes; outside the walls
      dimmed) with hover tooltips, stream (H.264 / HD HEVC) and layer toggles, automatic fallback to the worker's
      frames (WebSocket) when video can't play; 30 % side: FREE / BUSY / TOTAL / UNKNOWN widgets, animated live
      event ticker, worker card. Header badge 🟢 Worker Connected / 🟠 Monitoring off / 🔴 Worker Down / backend
      offline. Settings drawer (Apply/Cancel, version check 409, admin token asked on Apply; unsaved changes kept
      across the token prompt: a bug the browser test found) with the everyday settings and a collapsible
      "Advanced / Debug Settings" (detector, bay model auto/axis/OBB, inference size, classes, extra classes, slot
      mode + match rule/IoU, learning, absent = FREE, dataset capture); actions snapshot / reset / restart. History
      (sessions, events, CSV) and Analytics (lazy chunk: arrivals/departures per hour, occupancy now, stay
      lengths). Fixed-slot drawing and Modes 1–4 stay in Streamlit (linked).
    * Docker: `web` stage (node 24.18.0-alpine: tsc, vitest, vite build) -> dist copied into the runtime image;
      FastAPI serves it with an SPA fallback. Build runs 129 Python + 12 frontend tests.
    * Tests: Python 129 (+3 API: viewer token, WebRTC proxy with a mock go2rtc, CSV; +1 assertion: a takeover
      resets the snapshot), frontend 12 (vitest: overlay geometry, area path, event merge, settings diff, token
      URLs, analytics buckets, badge, counters).
    * Browser E2E (Playwright 1.63 + Google Chrome for H.264, as a LAN client at 192.168.20.237): viewer token
      asked -> Worker Connected -> live counters -> WebRTC video playing (640x360, ~21 fps decoded) -> 6-8 slot
      polygons + boxes, tooltip -> settings drawer + Advanced -> Apply asks the admin token -> applied (v4/v5) ->
      Streamlit shows the new value and does not rewrite (version unchanged) -> frame fallback -> History (91 rows)
      -> Analytics charts. go2rtc :1984 refused from the LAN; /api/status 401 without a token.
    * Live migration (user go-ahead, item 41): `docker stop carparking` (kept, Exited, for rollback), copied
      parking_history.db (identical row counts), config/, dataset_raw/ (42 frames) into ./data (originals left in
      place as backup), `docker compose up -d`: worker, api :8080, ui (Streamlit) :8501, go2rtc :8555. Worker
      processing (monitoring was on), read-only browser check on :8080 passed.
    * Found and fixed during migration: the first status after a worker takeover showed the previous worker's
      snapshot (583 frames) under the new heartbeat; claim() now resets it to "starting" on a takeover.
47. **2026-10-21 Sharp video in the React app (HD H.264)**: the camera sends HEVC on both channels (no HD H.264
    profile exists), so go2rtc re-encodes. Measured on this host (CPU while watched, 1.0 = one of 6 cores): sub 360p
    0.06 / 0.4 Mbit/s; main -> 720p 0.8 / 2.0; main -> 1080p 0.9 / 4.6; main -> 1440p 1.2 / 8.0 (mostly the HEVC
    decode; only while someone watches). New default go2rtc stream `parking_hd` = main -> 1920x1080 H.264 (much
    sharper, side-by-side check); `parking` (360p) and `parking_native` (HEVC as sent) remain. Player: HD default,
    choice remembered, chip shows the decoded size, `image-rendering: pixelated` only when the browser must upscale
    (a low-res stream on a big / 2x screen); SVG overlay stays in the worker's native 2560x1440 coordinates with
    non-scaling 2 px strokes (crisp at any size). Worker overlay/raw JPEGs 1280 -> 1920 px. Deployed
    (`docker compose up -d --build`, go2rtc restarted). The browser crispness check was stopped by the user before
    it ran. Free full-resolution video: set the camera's main stream to H.264 (then no re-encode).
48. **2026-10-21 Model variant: hardware auto-detection + live switching (default / YOLO11m)** (`live.py`,
    `scripts/worker.py`, `api/main.py`, React drawer; Streamlit untouched):
    * Stored as a worker option (`live_options` table, key `model_variant`: auto / default / yolo11m), not in the
      page-shared settings: Streamlit rewrites live_config whole and would erase it; it also doesn't bump the
      settings version. PUT /api/settings {"model_variant"} (admin token, validated), GET returns it, options list.
    * Worker: `hardware()` (CUDA via torch, cores, RAM via psutil) at start; `auto_variant()`: CUDA -> yolo11m; CPU
      only with >= 16 cores and >= 16 GB (CARPARK_YOLO11M_MIN_CORES / _MIN_RAM_GB) -> yolo11m; else default. Reads
      the option every step; a change rebuilds detector + monitor from the next frame (no restart; the slot map comes
      back from the DB). Only the stock COCO detector is swapped (same classes and boxes, same slot map and history
      scope, so Streamlit and the history continue); camera models, hybrid and bay never. yolo11m.pt downloaded on
      first use into CARPARK_MODELS_DIR (default next to the DB = ./data/models, kept across rebuilds). Snapshot
      "model": setting, active, auto, auto_choice, reason, swappable, weights, hardware.
    * React: "Detector model" section in the drawer: active model + "Auto-selected" badge + the hardware reason,
      dropdown Auto / Default · Light / YOLO11m · Medium (applied with the other settings), a note when the chosen
      detector can't be swapped; worker card shows "auto-selected".
    * Tests: Python 133 (+4: hardware decisions, live switch auto(GPU)->yolo11m->default without a reader restart and
      with tracking continuing, camera models never swapped, API option/token/validation/version), frontend 13 (+1).
    * Live: this host = no CUDA, 6 cores, 15.5 GB -> auto picks the default (yolov8s), shown with the reason.
      Throwaway check: yolo11m.pt downloads (38 MB, 24 s), loads, detects; 4.2 s/frame vs 2.1 s for yolov8s at
      1920 px on this CPU (with the live worker running), which is why auto keeps the light model here.

### 2026-10-21 New plan (user): parking system enhancements, local Docker architecture
* Decisions: multi-camera, low latency and unlimited logging with Supabase/PostgreSQL run locally as a Docker
  container (no cloud); hard-coded .env camera values replaced by dynamic camera configurations (cameras.json /
  the local PostgreSQL). Work one step at a time: the user tests each step and confirms before the next.
* Step 1: fixed coordinates drawing tool (React): draw slot corners with the mouse over the live video / a still
  frame; generate, export and save the coordinates as JSON.
* Step 2: camera and coordinates management: dynamic cameras + slot coordinates (cameras.json / local Docker DB)
  instead of the .env camera.
* Step 3: multi-camera grid view in React (1x1 and 2x2).
* Step 4: local Docker Supabase/PostgreSQL: entry/exit logs and occupancy history synced and stored there.
* Constraints carried over: Streamlit (app.py, scripts/demo_app.py) stays untouched and in parallel; tests never
  write to ./data; deploy with `docker compose up -d --build` (all tests run in the build).
49. **2026-10-21 Step 1: fixed coordinates drawing tool (React)**, awaiting the user's test before Step 2:
    * `frontend/src/components/SlotEditor.tsx` (page "Slot editor", sidebar > Monitor, route #/slots) +
      `frontend/src/slotdraw.ts` (pure geometry / JSON). Background: a still camera frame (/api/frame.jpg?raw=1,
      refreshable) or the live HD video. Modes: Polygon (click corners; first corner / double-click / Enter closes;
      Esc cancels; Backspace / Ctrl+Z undo), Rectangle (drag), Edit (drag corners or whole slots, select, Delete).
      Live cursor read-out (shares 0-1 and camera pixels). Side panel: camera (load any saved map), slot list (rename,
      delete), saved/unsaved state, Save to server (admin token), Revert, Clear. JSON panel: live preview (shares or
      px), Copy, Download .json, Import (paste or file; this tool's export, a fixed_slots.json camera entry, a bare
      list, or pixels).
    * Coordinates = shares of the frame (4 decimals), the format of config/fixed_slots.json; the export adds
      `points_px` for the camera's native frame (2560x1440).
    * Backend: GET /api/fixed-slots?camera= (viewer token; + saved cameras, in_use) and PUT (admin token; the
      same file and validation as the Streamlit editor; if the settings use this camera they are rebuilt, so a
      worker in Fixed mode applies the map within a second). The Settings drawer's fixed-mode link now opens it.
    * Tests: Python 134 (+1 API: token, validation 422, fixed mode carries the map at once, page parity, another
      camera leaves the settings alone), frontend 21 (+8: add / refuse / close-near-first / rectangle / move /
      export / import / broken map). Deployed (`docker compose up -d --build`).
    * Found on deploy: live monitoring is OFF (settings v4 written 2026-10-20 13:46 UTC, enabled false; not by
      this session's checks, which were read-only on the live stack). Left as is; the user decides. Camera 1 already
      has 5 fixed slots (drawn earlier in Streamlit): the editor opens with them; saving replaces them.
50. **2026-10-21 Step 1 follow-up: parking mode switch + "no FREE/BUSY on the dashboard"** (React only):
    * Cause of the dashes: live monitoring was off (settings `enabled5: false`), the worker idle, so it published
      no counts and no slots; the dashboard rendered that as bare dashes without saying why. Rendering itself was
      fine (with monitoring on, counts and overlay appear).
    * `components/Monitor.tsx`: `monitorState()` (ok / starting / off / down / offline) + `MonitorBanner` above the
      live view with "Turn monitoring on" (PUT enabled, admin token asked if missing); the counters show the reason
      ("Monitoring off: no FREE / BUSY states", "Worker down", ...) instead of bare dashes.
    * Slot editor: "Parking mode" card (Dynamic / Fixed) for the camera the worker runs, saved through the shared
      `slot_mode` setting (same as Streamlit's "Slot mapping mode" and the Settings drawer); Fixed refused until the
      camera has saved slots (else the worker would idle); shows the saved-slot count and the monitoring state.
    * Tests: frontend 23 (+2: monitor states, banner + counters note). Deployed.
    * With the user's request to fix the missing FREE/BUSY, monitoring switched back on via the API (settings v9;
      versions 5-8 had been written meanwhile from a UI): worker tracking again, after the entry hold FREE 1 / BUSY 5
      / UNKNOWN 4 of 10 slots (3 hidden behind cars, 1 duplicate), mode Dynamic.
51. **2026-10-21 Step 2: multi-camera management, PostgreSQL registry, CCTV grid + fullscreen** (Streamlit
    untouched; it keeps showing the primary camera):
    * Decisions: plain PostgreSQL 17.7 in Docker (Supabase-compatible schema; full Supabase = ~10 more services,
      too heavy next to YOLO on this CPU); PostgreSQL = camera registry + slot coordinates; each camera's runtime
      (settings incl. monitoring + Fixed/Dynamic, heartbeat, images, history) in its own SQLite file until Step 4;
      one worker process per camera; grid tiles play the sub-streams (cheap), fullscreen 1080p HD.
    * Schema (`src/carparking/registry.py` MIGRATIONS, tracked in schema_migrations, applied under an advisory
      lock): `cameras` (id slug PK, name UNIQUE = history key, never renamed; rtsp_url, rtsp_sub_url; is_primary,
      one at most (partial unique index); sort_order; metadata JSONB; created_at / updated_at) and `camera_slots`
      (camera_id PK -> cameras ON DELETE CASCADE, slots JSONB, updated_at). Backends: PgRegistry
      (CARPARK_DATABASE_URL, psycopg 3.3.6) or JsonRegistry (cameras.json: tests / fallback).
    * Runtime per camera (`runtime_paths`): the primary keeps parking_history.db, config/fixed_slots.json,
      dataset_raw (Streamlit and the history continue); others data/cameras/<id>/{parking_history.db,
      fixed_slots.json, dataset_raw}. Slot maps: registry = source, mirrored into each camera's slot file for its
      worker; the primary's file is also Streamlit's: newer side wins (sync_primary_slots).
    * `src/carparking/supervisor.py` + `scripts/supervisor.py` (compose "worker"): one scripts/worker.py per
      camera with its env (DB, slot file, raw dir, RTSP URLs, CARPARK_CAMERA_ID/NAME); restart on a URL change,
      backoff after crashes, stop removed cameras; seeds the registry with the existing camera on first start;
      --check = every camera's heartbeat fresh. The worker itself unchanged.
    * API: every camera endpoint takes ?cam=<id> (default primary: old clients unchanged); GET /api/cameras (URLs
      masked, streams, live status), POST (admin: registers, initialises its DB with monitoring off, registers its
      go2rtc streams <id>_hd/_sd/_native at runtime), PATCH (URLs: a masked one keeps the stored one; order,
      metadata), DELETE (not the primary; files kept); per-camera settings (a camera's name can't change: its
      history key), fixed slots (registry + mirror), status, frame, history, CSV, commands, video, WS
      (/ws/stream?cam=, one broadcaster per camera, primed so a new client gets no duplicate). WebRTC for a runtime
      stream re-registers it if go2rtc restarted; signalling timeout 30 s (cold ffmpeg start on HEVC).
    * Compose: service `db` (postgres:17.7-alpine, volume pgdata, not published, healthcheck), worker/api depend
      on it; .env: POSTGRES_PASSWORD + CARPARK_DATABASE_URL (generated); .env.example documented.
    * React: header camera switcher; Cameras page (#/cameras: list + live status, per-camera monitoring toggle and
      Dynamic/Fixed (Fixed refused without saved slots), add / edit / remove, open fullscreen, draw slots);
      dashboard View 1x1 / 2x2 / 3x3 (CameraGrid: tiles with the sub-stream, live slot overlay and counts, paging,
      empty cells "Add a camera"); click = select, double-click = fullscreen (full dashboard of that camera, HD);
      back: "Back to grid", Esc or double-click; browser Full screen button; Slot editor, Settings, History,
      Analytics follow the selected camera. Player retries 2 s then up to 10 s.
    * Tests: Python 140 (+ registry/supervisor 5, + API multi-camera 1; the 2 PostgreSQL variants skip in the build
      and passed 7/7 against a scratch database `carparking_test`, dropped afterwards), frontend 26 (+3 grid).
    * Deployed: PostgreSQL seeded with Camera 1 (5 slots from the slot file); Camera 1 continued (monitoring on,
      history 105 sessions / 205 events); "Camera 2 (test)" added (the same physical camera's streams, monitoring
      off; remove it when a real second camera exists): own worker idle, own DB, go2rtc streams deliver (SD 23 KB,
      HD 199 KB frames). Browser (read-only): 2x2 both tiles playing 640x360, Camera 1 tile FREE 4 / BUSY 5,
      double-click -> Camera 2 fullscreen 1920x1080, Esc -> grid, Cameras page lists both. Found and fixed: a
      PostgreSQL migration race at startup (API + supervisor), a cold-stream first WebRTC failure (timeouts /
      retry), tiles flashing "worker down" before their first live message.
52. **2026-10-21 Step 2 follow-up: editable camera display name + native video fullscreen (NVR style)**:
    * Display name: a camera's `name` is its history key (slot map, settings and every history row are stored
      under it), so it stays; new `label` = display name, editable any time. Migration 002_camera_label (ALTER
      TABLE cameras ADD COLUMN label, applied live), Camera.display (label or name) in the API (`display`), PATCH /
      POST accept `label`. React: shown everywhere (tiles, header switcher, Cameras page, Slot editor, player
      title); edit form "Display name" editable, "History key (fixed)" read-only with the reason.
    * Video fullscreen: `frontend/src/fullscreen.tsx` (useFullscreen: Fullscreen API on the picture's own box, iPhone
      fallback webkitEnterFullscreen on the bare video; stageStyle: largest box at the camera's aspect ratio,
      centred on black, so the overlay stays aligned; FullscreenButton). A button on every grid tile and on the main
      player: only that camera's video + slot overlay fill the monitor (no sidebar, header, panels); exit with the
      button, Esc (the browser's) or a double-click. The tiles' old expand icon now only opens the dashboard view.
    * Tests: Python 142 (+2: label in registry (both backends) and API, history key unchanged), frontend 27 (+1:
      fullscreen requests the tile's own box, not the page; double-click leaves it; aspect-ratio style; display
      name on the tile). Deployed. Browser (read-only, Chrome): tile fullscreen = video 1600x900 on a 1600x900
      screen, no sidebar; main player fullscreen 1600x900 decoding 1920x1080 HD with the overlay; exit returns to
      the grid / dashboard.
53. **2026-10-21 "Viewer token not accepted"**: the server was right (viewer token: every read 200 incl.
    /api/cameras, /api/fixed-slots, /api/settings, /api/status; .env LF, container env == .env); every change
    (display name PATCH, slots, settings, monitoring) needs the ADMIN token by design (viewer = view only). The UI
    made the mix-up look like a broken token: the viewer token pasted into the admin prompt was refused silently
    (same prompt again), the attempted change was lost, and nothing showed the access level. Fixed (React only):
    TokenDialog checks each token with GET /api/auth before saving (checkTokens): field errors, e.g. "This is the
    viewer token: it only allows viewing. Changes need the admin token (CARPARK_API_TOKEN ...)"; keeps a valid
    token even if the other is wrong. onNeedAdmin(reason, retry): once the admin token is accepted the change that
    asked for it runs again by itself (camera save / remove / monitoring / mode, settings Apply, commands, slot
    map save, parking mode, Turn monitoring on). Header access badge "View only" (click = enter admin token) /
    "Admin". Tests: frontend 30 (+3 token checks). Deployed. Browser (fresh, no saved tokens): viewer token ->
    View only + cameras listed; Display name editable, history key read-only; Save -> admin prompt; viewer token in
    it -> explained, stays; admin token -> saved by itself, badge Admin; test camera name restored afterwards.
54. **2026-10-21 "Camera list empty, viewer token not accepted" (cont.)**: backend fully healthy (services up, 0
    restarts, supervisor: PgRegistry 2 cameras + 2 workers, api no errors, GET /api/cameras 200 with the viewer
    and the admin token / 401 without, migration 002 applied cleanly: label column, constraints, FK, data intact,
    no locks). The bug was in the browser: (1) readToken() prefers the stored admin token, and the old dialog stored
    tokens unchecked, so a wrong stored admin token was sent with every read -> all 401 -> empty camera list / no
    tiles, whatever viewer token was entered; (2) a refused live connection (4401) opened the mandatory viewer-token
    dialog, which stayed on top even after access recovered. Fix (React + one API header): `sanitizeTokens()` checks
    the stored tokens with /api/auth and drops rejected ones (on load, on every token change, after the dialog, and
    before reacting to a refused live connection); the mandatory prompt closes itself once access is fine; "Couldn't
    load the cameras: ..." shown on the Cameras page / dashboard instead of an empty list; index.html served with
    Cache-Control: no-cache (a new build reaches browsers on the next load). Tests: frontend 31 (+1 stored-token
    trap). Deployed. Browser: (A) wrong admin + right viewer stored -> cameras + 2x2 grid with 2 tiles, no blocking
    prompt; (B) both wrong + the viewer token entered -> accepted, cameras + 2 tiles. Viewer token = .env line 16
    (CARPARK_VIEW_TOKEN, view only); admin token = .env line 11 (CARPARK_API_TOKEN, changes).

55. **2026-10-21 Step 3: CCTV multi-view refined (React only, no backend change)**. Grid tiles: sub-stream (360p)
    with an "SD" chip, "HD" warning chip when a camera has no sub-stream; the 1x1 / opened view plays HD (1080p).
    A tile plays only while visible (hidden tab or scrolled off screen closes its WebRTC: go2rtc stops converting);
    frozen-picture watchdog (video time unchanged 8 s -> reconnect, state "Stream reconnecting…"). Health per tile
    (lib.tileHealth, most severe first): Worker offline N s (red, greyed video) > Stream reconnecting / unavailable >
    Monitoring off > Starting… > Tracking; counts FREE / BUSY / ? unknown / TOTAL + "AI x s ago". Interactions:
    click selects (header switcher follows), arrows move, Enter / double-click opens in HD, PgUp/PgDn page, a camera
    chosen in the header brings its page, page dots + animated page change, "Tour" cycles pages every 10 s; grid
    width capped so 2x2 / 3x3 fit the window without scrolling. Single view: HD poster from the latest raw frame
    (only when a frame exists), video paused in a hidden tab, "Reconnecting video…" chip. Tests: frontend 36.
    Deployed. Browser (read-only): tiles 640x360 SD, camera-1 Tracking FREE 4 / BUSY 5 / TOTAL 9, camera-2
    Monitoring off; grid bottom 860 px in a 1000 px window; ArrowRight -> header "Camera 2 (test)"; Enter -> 1920x1080.

56. **2026-10-21 Step 4: central history & analytics in PostgreSQL** (Step 3 confirmed by the user). Design: the
    workers keep writing their SQLite file (source of truth; the Streamlit page reads it, untouched); a new compose
    service `sync` (scripts/sync.py, src/carparking/history_pg.py) copies every registered camera's history into
    PostgreSQL every 2 s, read-only on the SQLite side. Migration `003_history` (in registry.MIGRATIONS):
    parking_sessions, parking_events (both keyed (camera_id, source_id) = the SQLite ids, so ids stay identical),
    occupancy_samples (FREE/BUSY/UNKNOWN/TOTAL once a minute while monitoring and not warming up), camera_state,
    camera_events (first_seen, worker_online/offline, monitoring_on/off, scope_changed), history_sync (per camera:
    last copied ids, synced_at, error). Indexes: (camera_id, scope, source_id DESC), (camera_id, at), partial index on
    open sessions, BRIN on time for retention. Incremental + idempotent (ON CONFLICT); open sessions get their exit
    copied when SQLite closes them; a replaced SQLite file (fewer rows than copied) is reported, never mixed in
    (`scripts/sync.py --rebuild <id>`). Retention hourly, batched deletes, PostgreSQL only: CARPARK_RETENTION_DAYS
    (365) for sessions/events/camera events, CARPARK_SAMPLE_RETENTION_DAYS (90) for samples; open sessions kept.
    API: /api/sessions, /events, /summary, CSVs served from PostgreSQL while the camera's copy is < 30 s old, else
    the old SQLite path (header X-History-Source); new /api/analytics?days=&scope=all|current (SQL aggregation in the
    camera's time zone: totals, per hour, per day, stay lengths, occupancy over time), /api/analytics/overview,
    /api/camera-events, /api/history/sync. React: Analytics page (24 h / 7 / 30 / 90 days, all history vs current
    slot map, KPI cards, all-cameras table, occupancy-over-time, per-day chart, source badge); History page (source
    badge "PostgreSQL · synced N s ago", Camera status tab). Tests: Python 145 + 7 PostgreSQL cases (run against a
    throwaway postgres:17.7 container: 32 passed incl. SQL-vs-Python analytics parity), frontend 38. Deployed:
    first sync migrated camera-1 222 sessions / 424 events (`sync.py --verify`: in sync); 90-day analytics ~0.1 s.
    One warm-up sample (0 free / 0 busy / 9 unknown, 08:00 UTC) was recorded before the warm-up filter was added;
    it is harmless and ages out (or delete it from occupancy_samples by hand).

57. **2026-10-21 "page reloads every 1–3 s" fix + Slot editor locked in Dynamic mode.** (a) Reload: no
    location.reload anywhere; measured in a browser (MutationObserver): in the 1×1 / fullscreen view the HD video
    kept retrying (connecting -> failed after 10 s -> connecting), LiveView switched the WebSocket frame feed on/off
    with it, and useLive reset state + ticker to empty and refetched events on every switch: banner, counters,
    ticker and status card blanked and refilled (4892 DOM changes in 8 s). Fix: useLive clears only on another
    camera / token (frames on/off reconnects in the background, keeps state, ticker and the last frame; link stays
    "open"); LiveView keeps the frame images after a video failure until the video really plays. After: 29 DOM
    changes in 8 s (overlay ticks). Tests: hooks.test.tsx. (b) Rule: Fixed mode = draw / edit; Dynamic mode =
    read-only. API PUT /api/fixed-slots -> 409 "Dynamic Mode automatically detects parking spaces. Switch to Fixed
    Mode to manually draw or edit slots." for a registered camera in Dynamic mode (checked before any write; the
    legacy unregistered-name slot file path unchanged); GET adds mode / editable / locked. React SlotEditor: in
    Dynamic mode no Polygon / Rectangle / Edit / Save / Revert / Clear / Import, the message over the image and in the
    Slots card, the AI's live slots drawn read-only (FREE / BUSY colours) with a list; Fixed unchanged. Parking mode
    card: Fixed is allowed before a map exists (drawing needs Fixed; the worker idles until a map is saved);
    switching to Dynamic with unsaved drawing is refused. Streamlit's own editor (app.py) is untouched and not bound
    by this rule. Tests: sloteditor.test.tsx, test_api (409, mode fields, multi-camera). (c) Secrets: uvicorn's
    access / WebSocket log lines printed ?token= of <img>/<video>/WebSocket URLs: a log filter (MaskTokens) now
    writes token=***. Older log lines still hold the token until the api container is recreated / logs rotate.

58. **2026-10-21 Slot NMS 0.35 (camera-agnostic) + Refresh data / sync badge.** The user reported slots 11 and 20
    "both on the silver car". Checked live (frame + boxes): two cars, the silver car (slot 20, ground y 0.86) and the
    white sedan parked right behind it (slot 11, ground y 0.57; IoU 0.33, 74 % inside slot 20); 7 vehicles = 7 BUSY.
    Every overlapping pair of the live map is in slot_pairs (has held two vehicles at once). The user asked again for
    a general 30-40 % rule, so (history.py, any camera): merge_iou 0.5 -> 0.35 and dup_iou 0.4 -> 0.35; slot NMS
    (_dedupe) now runs every frame, not only on load, and never merges a .distinct pair; a candidate overlapping a
    slot that another box holds in the same frame is learned as a second vehicle (+ distinct) unless the two boxes are
    one vehicle (same_vehicle: IoU >= 0.5 at one depth = a split box), and such box pairs never mark a pair distinct.
    One box = one slot was already strict (greedy one-to-one by IoU; detector: class-agnostic NMS + merge_dets).
    Effect on the live map: none (all its overlapping pairs are distinct). Tests: test_history (live merge, car
    behind a car, split box, highest-IoU slot only; on-load merge test updated to the 0.35 rule). UI: dashboard view
    bar (single and grid) "Refresh data" button (useLive refresh counter: reconnect + ticker refetch in the background,
    camera list reloaded; no page reload; spins until the next state) and badge "Auto-syncing (~1 s) · updated X s
    ago" (amber "No update for X s" after 5 s, "Reconnecting…" when the socket is down). Tests: sync.test.tsx,
    hooks.test.tsx.

59. **2026-10-21 Auto-sync 20 s + no false FREE in the far background.** (a) Interval: WS /ws/stream?every=N
    (per client): routine state messages at most every N s, but a change of state_key (connection, settings
    version, monitoring, worker status, counts, any slot's shown state) is sent at once, entries / exits and frames
    always. React: interval picker in the sync badge (20 s default, 30 s, 5 s, live ~1 s; localStorage
    carparking.syncEvery), "updated X s ago · next <= Y s", amber after N + 10 s; Refresh data still reconnects for
    the current state at once; grid tiles use the same interval. (b) The user saw green slots at the far end. Live
    frame: one pickup there (slot 6) plus a second white car behind its bed at the pillar (slot 22, found by the
    zoom pass: real); FREE slots 13 / 17 / 18 / 21 lay behind vehicles nearer the camera (FREE only because a hidden
    slot turns FREE after hidden_max), 2 / 5 had the hatchback's / pickup's wheels in their floor (unmarked lot:
    slots learned from earlier cars overlap). New, any camera, display only (history unchanged): SlotTracker.unseen,
    a FREE slot is shown and counted UNKNOWN while "blocked" (stands_in: a vehicle's ground point in the bottom 40 %
    / inner 80 % of the slot) or "hidden" (vehicles in front cover >= occlusion_cover, also after hidden_max).
    Monitor.state: slot "blocked" flag, counts.blocked, hidden incl. unseen-hidden. Replay of the live frame: FREE 6
    -> 0, UNKNOWN 0 -> 6, BUSY unchanged (6). Overlay: dashed outline + hover text. Tests: test_history (3),
    test_api (every throttle), sync.test.tsx, hooks.test.tsx. A parking-area polygon (Settings) can still cut the
    far background off entirely (hard wall).

60. **2026-10-22 One shown state for banner + dashboard; one vehicle = one BUSY; no FREE over a vehicle box.**
    The user's screenshot: video banner "FREE 5 | BUSY 6 | TOTAL 11" vs cards "0 Free | 6 Busy | 5 Unknown". Cause:
    item 59's unseen rule was applied in Monitor.state() only; the overlay (live.render -> tracker.view) drew raw
    states. Now SlotTracker.shown(k) is the one source: view() (banner, overlay colours) and Monitor.state() (API,
    cards, counts, PostgreSQL samples) both use it. New display rules (dynamic maps, history unchanged): a BUSY slot
    without its own box this frame, on a vehicle matched to another slot (IoU > vehicle_iou 0.2 or stands_in), is
    UNKNOWN "on the vehicle of slot N" unless hidden / .shared (live: the pickup matched 6 and stood in 5 -> BUSY
    6 -> 5); a FREE slot with any vehicle box over it (IoU > 0.2) is UNKNOWN "overlaps a vehicle". Slot JSON "note"
    (reason), shown in the overlay hover. Replay of the reported frame: FREE 0, BUSY 5 (1 red car, 6 pickup, 11 white
    sedan, 12 hatchback, 20 silver), UNKNOWN 6. Not a double count (checked on the frame): slot 1 is the maroon car
    behind the white sedan (slot 11); slot 22 is a real white car behind the pickup at the pillar (zoom pass), which
    makes 6 vehicles when it is detected. Tests: test_history (+3, the beside-the-bay case moved under IoU 0.2),
    lib.test.ts. ROOT CAUSE of the remaining double BUSY (found live after the first deploy): cv2.intersectConvexConvex
    returns nonsense for nearly shared edges, depending on the argument order: the small car box at the pillar (x2
    0.5680) vs slot 6 (x2 0.5676) gave IoU 11.85, so the greedy matcher gave that car slot 6 and pushed the pickup's
    box onto slot 5 (two BUSY for one pickup). dynamic_iou.overlap / intersection_area now clip exactly
    (Sutherland-Hodgman, numpy, symmetric, capped at the smaller area); yolo_engine ios / _iou_ios (merge_dets NMS)
    use it too. tests/test_overlap.py: the live case, 3000 random boxes with shared / nearly shared edges vs the
    analytic IoU in both orders, rotated boxes vs a raster. Replay of the live frame: BUSY 6 -> 5. Live after deploy: banner == cards in every sample, the pickup only in slot 6;
    BUSY 6 = 6 vehicle boxes (incl. slot 22, the car behind the pickup). Last gap: a matched slot whose entry is still
    in its hold showed FREE under its car (slot 22, ~20 s): now UNKNOWN "a vehicle is in it (entry being confirmed)".

61. **2026-10-22 FREE judged on the ground footprint (user's 80 % rule).** The user saw "2 clearly visible vacant
    bays" while the dashboard showed 0 FREE / 5 BUSY / 6 UNKNOWN (their screenshots never reach the session). Shown
    state of a FREE slot is now judged on its ground footprint (bottom 40 % of its box, history.footprint): UNKNOWN
    "hidden" only when vehicles in front cover > hidden_ground 0.8 of it (was >= 0.6 of the whole box), "overlaps a
    vehicle" when a vehicle at its depth / behind covers > ground_share 0.4 of it (replaces the whole-box IoU > 0.2
    rule; a neighbour spilling 30 % over the side leaves a bay FREE), "blocked" (a vehicle's wheels in it) as
    before. Measured on the live frame, the 6 UNKNOWN stay UNKNOWN under that rule too: ground behind vehicles in
    front 2 77 % + wheels in it, 5 67 % + pickup's wheels, 13 / 18 / 22 100 %, 21 93 %. Asked the user which two bays
    they mean. Possibly a bay with no slot: slots 19 (x .35-.45, y .34-.47, retired 10-21 04:54) and 3 (retired
    10-20) lay left of the white sedan in front of the stairs; a vacant bay without a slot can never show FREE.
    OUTCOME: the user was right. My measurement used a frame from before the silver hatchback left (exit logged
    17:13:36). After it left, the old rules kept both vacant bays UNKNOWN: slot 20 by the whole-box IoU > 0.2 with
    the sedan's box (0.31), slot 21 as "hidden" (the sedan's box covered 79 % of it, >= 0.6). With the footprint rule
    both are FREE: live banner "FREE: 2 | BUSY: 5 | TOTAL: 11 | UNKNOWN: 4" == cards 2 / 5 / 4; BUSY = 11 sedan,
    1 maroon, 12 hatchback, 18 a car that parked 17:11, 6 pickup.

62. **2026-10-22 Counting thresholds adjustable in Settings (defaults unchanged).** history.DISPLAY_TUNING =
    {hidden_ground 0.8 [0.5-1], ground_share 0.4 [0.1-0.9], vehicle_iou 0.2 [0.05-0.6]} (the shown-state rules
    of items 60-61), tuning_values() clamps / defaults, SlotTracker.tune() applies them. Stored per camera as the
    worker option "tuning" (JSON, LiveStore.live_options, like model_variant: Streamlit's page-shared settings and
    the settings version untouched); the Worker reads it every step (_tune) and applies it to the running tracker on
    the next frame (no restart; a new monitor gets it too). API: GET /api/settings values hidden_ground /
    ground_share / vehicle_iou (stored or default), PUT (admin, range-checked by SettingsPatch, merged into the
    stored JSON), /api/settings/options tuning_defaults + ranges. React Settings drawer: section "Counting thresholds
    (Dynamic mode)", three sliders, Reset to defaults, applied with Apply. Not exposed on purpose: the slot-merge
    thresholds (a merge retires slots for good). Tests: test_history, test_live, test_api.

63. **2026-10-22 Video stall watchdog on decoded frames + Refresh data reconnects the video (demo prep).** User:
    video OSD 17:52:02 vs event log 18:04, "video stalled". Measured: the camera's OSD clock is ~15 min SLOW (frame
    captured 18:11:53 server PKT shows 17:56:56 on the OSD), so most of that gap is the camera clock (fix: NTP /
    time sync on the Hikvision). go2rtc parking_hd was flowing (1.2 MB/s in and to 3 consumers). Real gap found
    anyway: the frozen-picture watchdog used video.currentTime, which Chrome advances with the clock on a live
    WebRTC stream even without new frames, so a camera / go2rtc stall froze the picture for good. Now useWebRtc reads
    WebRTC stats (inbound-rtp framesDecoded) every second; no new frame for STALL_MS 5 s while connected -> reconnect
    (frame images meanwhile, as on any failure). useWebRtc(refresh): Refresh data reconnects LiveView, every grid
    tile's video and the poster still. Slot 11 "vehicle out" 18:04:06: it was FREE in the backend at once (the white
    sedan left; slot 21's entry the same second = a motorbike parked by the stairs); state changes are pushed at
    once even with the 20 s auto-sync. Test: webrtc.test.tsx (fake RTCPeerConnection).

64. **2026-10-22 Counter cards filter the video's slot overlay (demo).** Dashboard (single / fullscreen view):
    Free / Busy / Unknown cards are toggle buttons (aria-pressed, hover lift, glow ring when active, the others
    dimmed); the video's SVG overlay then draws only those slots, the rest faded to an 18 % outline without fill or
    label; Total slots, the active card again or the "Showing FREE only · n of m ✕" chip on the video shows all.
    Front-end only (App state slotFilter, reset on camera change; survives every state update / frame since it is
    React state, not tied to the stream). In frame-image fallback (the worker's images have every slot drawn) the
    matching slots are outlined on top. Test: filter.test.tsx.

65. **2026-10-22 Slot editor "Test slots" preview (read-only, additive).** New src/carparking/slot_preview.py
    (preview_slots: validate_slots + fixed_slots.match_score under the camera's match rule, one-to-one best pairs
    = FixedSlotTracker._match; no hold, nothing stored) and POST /api/fixed-slots/preview?cam= (viewer token; the
    drawn slots, saved or not, + optional only=[ids]; vs the vehicles of the worker's latest snapshot; 409 without
    live detections, 422 bad slot). No change to the tracker, worker, database or saved map. React SlotEditor
    (Fixed mode): "Test all slots" / "Test selected slot", canvas overlay (tested slots FREE green / BUSY red,
    untested dashed grey, the vehicles dashed yellow), banner "Preview mode – active · n FREE · m BUSY · vehicles
    from X s ago · rule", "Back to edit" (banner, card, Esc); editing / drawing keys paused meanwhile, a list edit
    leaves the preview; drawings, unsaved edits and the selection are kept. api.query = a read-only POST with the
    viewer token. Tests: test_slot_preview.py (incl. parity with FixedSlotTracker on random boxes, 3 rules),
    test_api (read-only: map + settings unchanged), sloteditor.test.tsx.

66. **2026-10-22 Test slots without monitoring (on-demand frame + detection).** The user got "no live detections
    ... switch Monitoring on". POST /api/fixed-slots/preview is now async: the worker's detections are used only
    while fresh (heartbeat ok, frame <= PREVIEW_FRESH 15 s; "source": "worker"); otherwise grab_frame() takes one JPEG
    from go2rtc /api/frame.jpeg (the camera's native stream, else HD; ~0.4 s) and slot_preview.detect_on_demand()
    runs the camera's own detector on it in a thread (live.build_detector with run["detector"], its conf, target
    classes, the parking-area mask and Fixed mode's keep rule; models cached in the API process, one inference at a
    time; no zoom pass) -> "source": "on-demand" + "image" (the tested frame as a data URL). 502 when no frame can be
    had, 500 if detection fails, 422 bad slot (checked first). Still stores nothing. Editor: the tested frame is the
    canvas background during the preview, banner "frame grabbed now" / "live monitoring, n s ago", "Detecting…" with
    a note (the first test loads the model). Tests: test_api (on-demand with a mocked go2rtc + stubbed detector,
    read-only), test_slot_preview (detect_on_demand filters like the worker), sloteditor.test.tsx.

67. **2026-10-22 (day 2) Slow entries, "slot 6 missing" in History, pagination, front car FREE.** User: 4 requests.
    * **Findings (live, read-only).** (a) Slot 6 was never missing: its open session 180 (parked since 2026-10-19
      19:03, "already parked when tracking started") was row 85 of 149 in the History table (newest first, limit 200).
      BUSY always comes with an open session (tracker state 1 is set only together with db.apply). (b) Entry latency
      = entry hold 15 s (+ >= 2 frames) on a ~4.3 s frame period => ~20-25 s; the WebSocket sends slot changes at
      once (state_key), so no UI delay. Benchmark on the live frame (with the worker running): yolov8s whole frame
      1920 px 1.3-2.9 s, 1280 0.45-1.0 s, 960 ~0.2 s, 640 ~0.1 s, zoom crop 0.4-0.8 s each. **Lower resolution is
      NOT safe here**: at 1280 / 960 / 640 the two white cars become one box. (c) The front car (maroon hatchback):
      1920 px finds it at only 0.23-0.24 with a wrong box (conf 0.23 -> lost on many frames), 640 / 960 px at
      0.93-0.94 with the right box. That caused the false FREE / exit: session 265 of slot 20 closed 09:23:18 while
      the car stood there; camera-2 Test slots found 3 vehicles (maroon car missing, S1 given the white cars' box).
    * **Near pass** (yolo_engine.YoloDetector near_imgsz / near_band 0.3; live.NEAR_IMGSZ = 640, used by
      build_detector for every whole-frame detector > 640 px -> worker, API Test slots): a second whole-frame pass at
      640 px whose boxes count only if their bottom edge is in the lowest 30 % of the frame (near the lens); they win
      a duplicate against the main pass's box (merge_dets, src "near"). Far rows stay 1920 px. Live: maroon car 0.925
      [near], slot 20 BUSY, slot 21 no longer wrongly FREE (UNKNOWN hidden), 5 BUSY = 5 open sessions.
    * **Fixed slots, point rules** (fixed_slots.match_score; POINT_TOL 0.15, POINT_COVER 0.3): a ground / centre
      point just outside the slot (<= 15 % of the vehicle box's own height) still matches when the box covers >= 30 %
      of the slot; score = coverage (< 1), below every point-inside match (1 + coverage), so a slot that holds the
      point always wins the vehicle. MATCH_RULES texts say so. Camera-2 after: S1 BUSY (maroon, near 0.90); S2 still
      FREE because S2 is drawn on the white sedan's body (y 0.38-0.48, its wheels at 0.56 = 24 % of its height
      away): open item, see Start here 4.
    * **Faster entries**: camera-1 entry hold 15 s -> 3 s (PUT /api/settings, version 20; Settings > hold; exit
      hold 30 s / 15 empty frames unchanged). Tracker: new seen_window (15 s): the "hidden car seen on some frames
      only" window and the "initial" (already parked) cut-off are max(hold, seen_window), so a short hold neither
      loses hidden cars nor logs parked cars as arrivals; default behaviour (hold 15) unchanged. One test changed on
      purpose (test_hidden_car_seen_on_some_frames_only_turns_busy: the gap is now > 15 s).
      Measured after deploy: ~3.6 s / frame with no viewer (33 frames in 120 s; go2rtc 0 % CPU); 7.5 s / frame while
      go2rtc was failing and re-spawning transcodes (150 % CPU). Expected entry time now: first frame that sees the
      car + the next one = ~6-9 s after it stops (was ~20-25 s). 2-3 s needs ~1 frame/s: not possible on this CPU
      at 1920 px (a GPU, or the SD stream on the dashboard to free go2rtc's CPU).
    * **History pagination**: GET /api/sessions/page?page=&size= (default 20, max 200; viewer token) ->
      {items, total, open, page, size, pages}; open sessions first, then newest first (HistoryDB.sessions_page,
      PgHistory / SqliteHistory.sessions_page; PostgreSQL or SQLite as before, X-History-Source). /api/sessions
      unchanged (Streamlit, CSV). React History > Sessions: 20 rows per page, pager under the table (Previous /
      page numbers with "…" / Next, "Showing 1–20 of 151 sessions · 5 parked now (listed first)"), page 1 again on
      a camera change. Live: page 1 = slots 20, 12, 11, 18, 6 parked, then the newest closed ones (8 pages).
    * **go2rtc incident** (not caused by the code): after the deploy every go2rtc snapshot was empty and WebRTC
      returned 500 ("wrong response on SETUP" from the camera; HEVC "VPS 0 does not exist"); 38 such WebRTC errors
      in the last 24 h. `docker compose restart go2rtc` fixed it (frames 291 KB / 203 KB, WebRTC 201). If the video
      goes black again with these errors: restart go2rtc.
    * Tests: Python 176 passed + 7 skipped (+ PostgreSQL 7 on a throwaway postgres:17.7), frontend 54 (+2): near
      pass, point tolerance, seen_window, sessions_page (SQLite, PostgreSQL, API), pager, pageList.

68. **2026-10-22 Front Parking 02 (Dynamic): 6 cars, 5 BUSY.** User: the two white cars in the row merge into one
    detection; adjust NMS / IoU so the BUSY count matches the 6 parked cars.
    * **Finding (live frame, read-only).** Row: maroon (slot 20), white sedan (11), **white hatchback behind it
      (slot 1)**, silver hatchback (12), white car (18), pickup (6). The whole frame (yolov8s 1920) boxes the sedan +
      hatchback as ONE box (0.248-0.556 x 0.25-0.593, 0.83); even at conf 0.05 and NMS iou 0.9 (also 1280) there is
      no separate hatchback box: **not an NMS / threshold problem**, the model sees one white object. Slot 1 shown
      UNKNOWN "hidden". The zoom re-check of slot 1 found nothing: its crop (slot + 40 %) contains the sedan again,
      and the hatchback's box ends where the sedan hides it = on the crop edge = dropped as "cut".
    * **Behind crop** (YoloDetector.detect_region behind=True): the crop ends at the slot's ground line (+ 2 % of its
      height) and boxes on that bottom edge are kept. Slot 1 -> the hatchback alone at 0.92 (IoU 0.64 with slot 1).
      ParkingMonitor._zoom tries it when the normal zoom finds nothing on a hidden / flagged slot.
    * **Swallowed-box rule** (ParkingMonitor._apart, SWALLOW 0.15): the merge took the hatchback's box for a slice of
      the sedan's box (as wide). It is kept as a vehicle of its own when every box it would merge into is matched to
      a slot and rises above that slot's top by > 15 % of its height (the sedan's box: 0.25 vs slot 11's 0.333 = 32 %:
      it swallowed the car behind) and the new box reaches above that top too. First try (distinct pairs + slot top)
      also turned slot 13 BUSY from the pickup's upper half -> replaced: the pickup's box fits its own slot 6, never
      split; a lone sedan's roof (box starts at its slot's top) neither.
    * Replay (throwaway container, copy of the live DB, camera-1 settings + detector, the live frame every 3.6 s):
      before 5 BUSY for 20 frames; now slot 1 entry at frame 8, then **6 BUSY** (1, 6, 11, 12, 18, 20) for 30
      frames, slot 13 stays out. Cost: one extra crop (~0.4 s) only when the normal zoom of a hidden slot finds nothing.
    * First deploy, live: slot 1 turned BUSY, then flickered UNKNOWN "on the vehicle of slot 11" (and slot 12, the
      silver hatchback with a weak 0.26 box, "on the vehicle of slot 18"): the zoom's 2 looks / frame rotate over ~7
      hidden slots, so a BUSY car behind another was confirmed only every ~30 s (> hidden_max) and a new one never
      twice within the 15 s window; slot 1 also counted as a phantom on some frames (never zoomed). Fixes:
      - zoom order: BUSY slots without a box and slots with a pending entry (cand 1) first, each in turn; the
        hidden FREE / unknown ones share the rest (monitor._zoom);
      - proven slots taken for phantoms are zoomed too, at zoom_free_conf 0.40; a phantom frame (like a hidden one)
        no longer restarts a pending BUSY (history.update);
      - display: a BUSY slot is not "on the vehicle of slot N" when that vehicle stands in front of it (bottom lower)
        and the pair has held two cars at once (.distinct); slot 5 under the pickup (behind / same depth) still is
        (history._unseen; test_one_vehicle_never_makes_two_busy_slots unchanged).
      Replay on both live frames (5 s / frame): slot 1 entry at frame 7, then 6 BUSY steady for the rest.
      **Live after the second deploy: 6 BUSY = the 6 cars** (20 maroon, 11 sedan, **2 white hatchback**, 12 silver
      hatchback, 18 white car, 6 pickup), checked on the frame with the boxes drawn: the zoom alternates slot 12
      [0.38-0.55 x 0.21-0.30] (the silver car) and slot 2 [0.33-0.55 x 0.21-0.34] (the white hatchback's visible
      upper part, its tail light at the box's bottom right). The hatchback lands in slot 2 (overlaps slot 1; it was
      tracked there this morning: slot 2 sessions 07:02-07:52 and 08:15-08:56, the latter closed as "no vehicle seen
      while partly hidden" = the 5-instead-of-6 the user saw). 50 frames: 6 BUSY except one frame (slot 20 one frame
      "on the vehicle of slot 11" when its box dropped; history unaffected).
    * Tests: Python 180 (+4: behind crop in test_engine; swallowed / lone sedan / pickup slice and BUSY-first +
      phantom zoom in test_monitor; hidden car not "on the vehicle" in front in test_history), zoom fakes take
      behind=. NMS / conf unchanged (they can't split the box).

69. **2026-10-22 Slot editor: snapshot-based flow (Camera 2, Fixed).** User: the live stream drops while editing and
    Test slots / Test all fail with connection errors; wanted: a still captured on open, frozen as the canvas, tests
    on it, a "Refresh snapshot" button.
    * Cause: the canvas was the worker's raw frame (none for camera-2, monitoring off) and every Test grabbed a new
      frame from go2rtc, which answered with empty bodies whenever the camera refused its RTSP SETUP (item 67).
    * **POST /api/fixed-slots/snapshot?cam=** (viewer token): one still captured now, kept in memory (SNAPSHOT_KEEP 3
      per camera; gone on an API restart). Sources, first that answers: (1) **the camera's own HTTP snapshot**
      (Hikvision ISAPI /ISAPI/Streaming/channels/<ch>/picture, digest auth from the camera's RTSP URL, channel from its
      path; live: 2560x1440, ~465 KB, 0.7 s, no video stream involved; credentials never logged), (2) go2rtc's
      snapshot of the native / HD stream, (3) the worker's last frame (<= 10 min old). -> {id, ts, source, width,
      height, image (data URL)}; 502 with "Refresh snapshot" advice when none answers. In Fixed mode the vehicle
      detection on the still starts at once in the background (start_detection).
    * **POST /api/fixed-slots/preview** takes "snapshot": the vehicles detected on that still (the camera's own
      detector, once per still and settings version, then reused; awaits the warm-up job) -> source "snapshot";
      410 when the still is gone (the editor then takes a new one and retries, no error shown). Without "snapshot"
      it works as before (worker / on-demand).
    * React SlotEditor: a snapshot is captured on open and on a camera change ("Capturing a snapshot…", or the error
      + Try again); it is the canvas (pill "Snapshot"; "Live video" stays for reference); label "Snapshot 09:41:02 ·
      camera snapshot"; toolbar button **Refresh snapshot**; Test all / Test selected send the snapshot id and switch
      the canvas to it; banner "snapshot of <time>". Pixel export uses the still's size when the worker has none.
    * Live (camera-2, monitoring off): API snapshot 0.7 s; Test all 8.4 s for the first detection after an API
      restart (model load), 2.3 s on a new still, then 0.1 s for further tests of the same still. Browser (Chrome,
      Playwright): snapshot on open 2.1 s, Test all 8.5 s (first after deploy), Test selected 0.2 s, Refresh
      snapshot 1.2 s (new image), Test all on it 1.4 s; no error shown, no console errors; result 2 FREE · 4 BUSY.
    * Tests: Python 181 (+1 test_api: ISAPI still, tests reuse one detection, 410, fallback to the worker frame, 502,
      keep 3, warm-up in Fixed mode; the network mocked, camera_still uses the app's HTTP client), frontend 56 (+2:
      snapshot on open / Test on it / Refresh; 410 -> new still + retry, no error).

70. **2026-10-22 Live player: overlay toggles remembered.** User: Slots / Boxes / Area switched off came back on
    after a refresh. LiveView's layers are now kept in localStorage ("carparking.layers", savedLayers(); missing /
    unreadable = all on; private window: this view only), read whenever the player mounts. The toggles exist in the
    single player (1x1 / a camera opened in HD), not in the grid tiles. Live (Chrome): Boxes + Area off stayed off
    over 25 s of live updates, Refresh data and a page reload; no console errors. Tests: frontend 58 (+2,
    layers.test.tsx). Noted: camera-1 conf is 0.30 (settings version 21, set from Settings, not by Claude).

71. **2026-10-22 Front Parking 02: one hidden car, one BUSY slot (7 -> 6).** Found while checking item 70: 7 BUSY
    for 6 cars, slots 1 AND 2 open for the white hatchback (slot 2 since 11:17, slot 1 since 12:07); since 10:31
    slot 1 had opened / closed 23-50 s sessions every 5-15 min ("no vehicle seen while partly hidden").
    * Cause (replay, copy of the live DB, the live frame): the hatchback hidden behind the sedan is found by the
      zoom with a differently cut box each time (its upper part + the silver car's roof [0.33,0.21,0.56,0.39]; cut at
      the sedan's roof line [0.26,0.26,0.55,0.44]; with the sedan's windows [0.23,0.29,0.55,0.50]), each matching
      another learned slot (2, 1, 21); each turned BUSY in turn. Those slots were even recorded as "distinct" (two
      boxes of one car in one frame). Also: a slot with a pending entry that was not hidden dropped out of the zoom.
    * Fix, dynamic maps (SlotTracker._owned, OWN_IOU 0.3): a box that would start / confirm an entry in slot k while
      a BUSY slot j has no box of its own in that frame, and the box is j's vehicle (IoU >= 0.3 with the vehicle box
      j last had, .slots[j]["last"], or >= match_iou with j's slot), is j's car seen again (j kept, timers refreshed)
      and no evidence for k (no entry, not FREE either). Not for slots already BUSY (they keep their own box); the
      distinct pairs are ignored here. Zoom: slots with a pending entry are in the want list (monitor._zoom).
      Trade-off: a car leaving j while another parks in an overlapping slot within j's exit buffer is counted in j.
    * Replays (4 scenarios: the current frame from the live state and from FREE, the two earlier frames; 18 frames
      each): 6 BUSY steady in all, never 7 (the hatchback in 21, 1 or 2 depending on the frame: slots 1 / 2 / 21
      overlap heavily in the learned map, but one car = one BUSY).
    * Tests: Python 182 (+1 test_history: the three differently cut boxes never enter slots 1 / 21 while slot 2 holds
      the car; a separate car next to a BUSY slot with its own box still enters).
    * Live after that deploy: the silver hatchback had left (12 closed 12:34) -> 5 cars, 5 BUSY for 31 of 40 frames,
      but slot 24 (a strip on the white car of slot 18) entered at 12:38 and stayed BUSY 14 min: as a BUSY slot
      without a box it is zoomed every frame, the crop finds the white car's upper part, inside that car's whole-frame
      box but no merge duplicate (other bottom / width) -> kept itself alive. Fix (monitor._zoom, INSIDE 0.8): a zoom
      box that repeats a box already found (merge) or lies >= 80 % inside a whole-frame box is that vehicle's piece,
      accepted only if that box swallowed a car behind it (_apart, unchanged test; now also for the normal crop).
      Another zoom box is not used for "inside" (loosely cut: the hatchback's includes the silver car's roof).
      Test test_monitor (slot 24 released, 18 stays BUSY, busy 1).
    * Slot map caveat (learned, dynamic): after the silver car left, the tracker retired slot 12 as a duplicate of
      slot 2 (slot 2 overlaps it and the hatchback); slots 1 / 2 / 21 overlap the hatchback. Replays against the
      current map: the 5-car frame -> 5, the two 6-car frames -> 6; the old frame with the silver car -> 5 (no own
      slot for it any more until it is learned again). For a guaranteed demo count: Fixed mode with drawn slots, or
      Reset slot map so the row is learned afresh.
    * Live after the final deploy (13:0x-13:2x; the maroon car left 13:05 = slot 20's exit, correct): 5 cars
      (sedan 11, hatchback 1, car 18, pickup 6, the car behind the pillar 23) -> 5 BUSY on 26 of 40 frames; no
      duplicate entries any more (slot 24 released 13:06). Still open: hidden cars blink (slot 1 and slot 23 close
      with "no vehicle seen while partly hidden" and reopen 2-5 min later: the hidden_max 30 s expiry when the zoom
      misses them, more often since conf 0.30 = the old open item 3, slot 22's cycles) -> the count dips by one for
      stretches; slot 5 one 1-min false session (13:21). Proposed, awaiting the user: hidden_max 30 -> ~120 s for
      camera-1 (Settings; exits keep the last-seen time). Slot 2's session from before the fix (since 11:17) stays
      open until slot 2 is seen FREE.

72. **2026-10-22 Demo prep: hidden max 120 s, slot map reset, steady counter cards.** User: 3 items.
    * camera-1 hidden_max 30 -> 120 s (PUT /api/settings, version 22; hold 3, conf 0.30, exit hold 30 unchanged).
    * Reset slot map on camera-1 (POST /api/commands reset, run by the worker): open sessions closed "closed by slot
      map reset" (incl. slot 2's leftover since 11:17), learned slots and pairs deleted, history kept. Before it, a
      read-only backup of the map: scratchpad camera1_slotmap_before_reset.json (25 slot rows, 37 pairs, 6 open
      sessions). Relearned within ~1 min: 4 slots, IDs 1-4 (1 sedan, 2 white car, 3 pickup, 4 car at the pillar),
      4 BUSY. **But 5 cars are parked**: the new slot 1 was learned from the whole frame's merged box of the sedan
      + the white hatchback behind it (top 0.252), so the swallowed-box rule (the box rises above its own slot) can't
      split the hatchback any more; a hidden car is only learned once seen in the whole frame. -> 4 BUSY for 5
      cars. Not fixed; options given to the user (restore the backup map, or Fixed mode with drawn slots).
    * Counter cards: measured first (/api/status every s for 3 min: never blank, max heartbeat age 5.3 s of 15;
      useLive keeps its state on auto-sync / Refresh, clears only on a camera / token change). The one real blink:
      the worker's "loading the model" messages without counts after a settings save / reset. Fix: useHeld
      (hooks.ts) keeps the cards' last counts for up to 30 s while counts are missing; dropped at once on a camera
      change or Monitoring off. Chrome: 30 samples / 60 s all 0/4/0/4, no blank, no console errors. Tests:
      frontend 59 (+1 held.test.tsx).

73. **2026-10-22 Front Parking 02: the maroon hatchback near the left edge counted again.** User: after the reset the
    front maroon car (back after an in/out) wasn't counted; keep camera-1 Dynamic, camera-2 Fixed.
    * Finding (live frame 13:58): the car IS detected (near pass 0.92, box 0.070-0.452 x 0.450-0.817) but dropped by
      the parking-area wall: 87 % of its box inside the left cut (12 %), the rule wants 90 % (in_area). Parked a little
      further left than in the morning (then 90.4 %). Not a tracking-confirmation problem: the tracker never saw it.
    * Fix (monitor.in_area, NEAR_BAND 0.3, NEAR_INSIDE 0.75): a vehicle close to the lens (bottom edge in the lowest
      30 % of the frame) and whole in the picture (not cut by the frame's left / right side) needs 75 % inside; centre
      and ground point inside as before; everything else 90 %. The 2026-10-19 scaffolding box (reaches x 0) stays
      out even at 85 %. Applies to detections and learned slots (dynamic). Test test_monitor (maroon kept, scaffold
      out, the same box farther away out).
    * Replay (copy of the live DB, the live frame): maroon learned as slot 5, entry at frame 5. Live after deploy:
      slot 5 learned + BUSY within ~25 s; 4 BUSY = sedan (1), white car (2), pickup (3), maroon (5).
    * Still not counted (dynamic limits, not changed): the white hatchback behind the sedan (merged into the sedan's
      box since the reset, item 72) and the car behind the pillar (slot 4; after the worker restart its visible sliver
      box no longer matches slot 4 and is dropped as a tiny box, < 15 % of the cars at its depth, the filter that
      keeps railings / mirrors out). Modes unchanged: camera-1 dynamic (monitoring on, hidden max 120), camera-2 fixed.
    * Tests: Python 184 (+1).

74. **2026-10-22 Area coverage 90 % -> 70 % (Dynamic), demo freeze.** User: lower the 90 % coverage threshold to
    70 % without loosening the small-box filters. The only 90 % threshold in Dynamic mode is the parking-area wall
    (in_area; slot matching is IoU 0.30). Now MIN_INSIDE 0.7 for a whole box; EDGE_INSIDE 0.9 kept for boxes cut by
    the frame's left / right side (the scaffolding phantom of 2026-10-19); centre and ground point still inside;
    is_tiny / phantom filters untouched; the item-73 near rule is folded into it. Checked on the live frame: no box
    newly admitted (maroon 87 %, sedan, pickup kept; nothing else). Test updated (maroon kept, also farther away;
    scaffold out; a whole box at 66 % out). Python 184. Live: before the deploy 5 BUSY (1, 2, 3, 4, 5); after the
    worker restart 4 BUSY (1, 3, 4, 5): slot 2 (the white car) can't be re-confirmed because the detector's sedan
    box now reaches up over it (top 0.206). Worker restarts reset the slots' confirmed states (not changed).

75. **2026-10-22 Front Parking 02: one box over the sedan + the hatchback behind it is split in two.** User: split the
    merged box so both cars get their own slots (no reset).
    * Not an NMS fix: the 1920 px pass never outputs a separate hatchback box (conf 0.05 / NMS 0.9, item 68). A crop of
      the merged box gives inconsistent boxes. What works on the live frames: a second whole-frame pass at 1280 px
      boxes the sedan alone (same bottom, top ~0.34 vs the merged 0.225-0.255) on 3 of 4 frames, and the behind crop
      over the strip above it finds the hatchback (0.36-0.79, its bottom at the sedan's roof) on 4 of 4.
    * ParkingMonitor._split (dynamic only, before the zoom; SPLIT_MIN_H 0.25, SPLIT_IMGSZ 1280, SPLIT_EVERY 30 s):
      a whole-frame box >= 25 % of the frame tall is split when the 1280 pass has a box with its bottom (+-8 %) whose
      top is >= 15 % lower and >= 80 % of its width (the front car alone) AND the behind crop over the strip above it
      finds a car (bottom near the front car's top, top near the merged box's top, >= 40 % of its width, >= 70 % inside
      it) -> the front car's box + that car's box, "src" "split". A lone tall car: the 1280 box = the same box ->
      never split, the strip not looked at. The result (split or not) is reused 30 s while the box stays (IoU >= 0.9):
      one extra pass + crop per half minute; the car behind is seen on every frame meanwhile.
    * Replays (copy of the live DB): current frame 5 BUSY (hatchback learned as a new slot), six_raw 6, dup7 5,
      six_raw2 unchanged (the 1280 pass doesn't separate there: no split, no harm).
    * Live (14:1x OSD): 5 BUSY steady over 29 frames = maroon (5, near pass), white sedan (7, the split's front box,
      newly learned), white hatchback (2, the split's rear box), white car behind it (6), pickup (3); the slot learned
      from the merged box (1) shows UNKNOWN "blocked". Test test_monitor (split into two slots, 2 BUSY; a lone car
      never split; the 1280 pass run once). Python 185.

76. **2026-10-23 Phase 2 started: metric geometry core (`src/carparking/geometry/`), not deployed, behind a future
    `geometry.enabled` flag.** Master plan: `docs/PHASE2_GEOMETRY_PLAN.md` (blueprint v2: one-time Admin SAM click in an
    isolated `setup` service, GeoCalib + vehicle priors, yolov8s + a 3x3 homography at runtime, `pack_capacity` for Day-1
    FREE slots on an empty lot). Done today: `camera.py` (pinhole + flat ground, pixel <-> metres, horizon, ground
    resolution), `bev.py` (surface polygon with horizon guard, area, vehicle footprint, edge-to-edge distance, nearest
    pairs), `capacity.py` (double-row bay packing with aisles, orientation from the polygon), `calib.py` (camera height
    from vehicle boxes via roof/contact rows, MAD outliers; fusion of scale sources with HIGH/MEDIUM/LOW). numpy + cv2
    only, no new dependency. `tests/test_geometry.py` (14 tests, synthetic camera h = 6 m / pitch 30 deg / roll 1.5 deg:
    height recovered within 3 %, area within 2 %, exact packing counts). Run in a throwaway container with the project
    mounted read-only (no pytest on the host); Streamlit untouched (parity test passes). Nothing in the worker, API or
    live data changed; no deploy. Next: LiveStore `geometry` versions, the `setup` service (SAM / GeoCalib / depth).

77. **2026-10-23 Phase 2 step 1: versioned geometry storage per camera (`src/carparking/geometry_store.py`), not
    deployed.** One `GeometryStore` class over two backends: SQLite (the camera's own history DB, via the new lazy
    `LiveStore.geometry(camera_id)`) and PostgreSQL (table `camera_geometry`, migration `004_camera_geometry` added to
    `registry.MIGRATIONS`, rows cascade with their camera). Each version is one JSON record with a state: `draft`
    (setup output) -> `active` (Admin Accept; at most one per camera, enforced by a partial unique index) ->
    `superseded` (kept: the last 5, for rollback) or `stale` (drift guard; final, not re-activatable). API of the store:
    create / get / list / active / activate / rollback (re-activates the superseded version activated last) /
    mark_stale / delete_draft (drafts only); `open_geometry` picks PostgreSQL when CARPARK_DATABASE_URL is set. Records
    containing a URL with credentials are refused. **Zero effect on what exists:** the SQLite table is created on first
    use only (LiveStore() alone adds nothing: tested), settings / live_* tables untouched, no active version =
    uncalibrated = `active()` None, so the worker needs no change. `tests/test_geometry_store.py` (13 tests x 2
    backends): defaults, numbering, activate / rollback chain, stale, pruning, draft delete, camera isolation, one-active
    constraint, failed transitions change nothing. Verified: SQLite in tmp_path; PostgreSQL against a throwaway
    postgres:17.7 container on its own network (removed afterwards, the live `db` untouched) together with
    test_registry + test_history_pg; full Python suite 212 passed, 16 skipped (PG cases without a DSN), pyflakes clean;
    parity test passes. Live containers not restarted. Next: the `setup` service (SAM / GeoCalib) and its API endpoints.

78. **2026-10-23 Phase 2 step 2: the isolated setup service + setup API (built and tested, NOT deployed, flag off).**
    * **Setup service** (`setup/`, compose profile `setup`, not started by `up`): own image `stomeetings/carparking-setup`
      (Python 3.12 - current kornia / matplotlib need >= 3.11; the main image stays 3.10; 3.48 GB), FastAPI on :8100, no
      published port, no `.env`, no `/data`. SAM 2.1 Hiera Small (sam2 `2b90b9f`), GeoCalib pinhole (`97b8968`, 2026-06-28,
      the newest commit older than 14 days), Depth Anything V2 Metric-Outdoor Small (HF `fd2c220`); `setup/manifest.json`
      pins commits + sha256 + sizes, `setup/fetch_weights.py` verifies at build (all 5 files matched; the depth hash equals
      Hugging Face's own). SAM 3 not used (gated weights). Models load lazily; endpoints /health, /segment, /calibrate.
      The build runs `setup/test_service.py` (4 contract tests with fake models). Isolation checked: the main
      `.dockerignore` allowlist lets no `setup/` into the main build context (verified by listing it); default
      `docker compose config --services` has no `setup`; no heavy package is in the main requirements.
    * **API** (`api/geometry_routes.py`, registered in `create_app`; `capture_still` split out of `take_snapshot`, same
      behaviour): `POST /api/cameras/{id}/setup/still`, `.../segment`, `.../calibrate`, `GET /api/setup/jobs/{job}`; admin
      token; **503 unless `CARPARK_GEOMETRY_ENABLED=1`** (default off; `CARPARK_SETUP_URL`, default http://setup:8100).
      Jobs in memory (`carparking/setup_jobs.py`, last 50). Calibrate -> a DRAFT in the Step 1 geometry store
      (PostgreSQL or the camera's SQLite); nothing the worker reads changes. A setup service that isn't running gives
      the job an error naming the compose command.
    * **numpy-only logic** (`src/carparking/geometry/`): `surface.py` (mask -> polygon, cars on the surface filled),
      `pipeline.py` (GeoCalib + depth + vehicles + typed reference -> record with area, packed bays, warnings),
      `calib.py` (+ GeoCalib up vector -> camera, height from depth map / reference distance).
    * Tests: `test_geometry_pipeline.py` (15), `test_setup_api.py` (4, mock setup service + flag/auth/failures/draft),
      full suite 231 passed, 16 skipped. Docs: `docs/PHASE2_SETUP_SERVICE.md`.
    * **Real-model run on a live camera-1 still (throwaway DB, service on a localhost-only port, then removed):** the whole
      chain works (SAM ~18-24 s on CPU, GeoCalib + depth ~45-50 s, first call loads the models). SAM follows the brick
      pavement and stops at the road kerb; a steel pole splits it, so the polygon is picked as the largest candidate within
      90 % of SAM's top score (the top-scoring one was half the pavement). **Calibration on this camera is NOT usable:**
      GeoCalib's pinhole model returns f = 5500 px (20 deg field of view) and 4.8 deg tilt; simple_radial gives 52 deg /
      13.8 deg with k1 = -0.46 (strong barrel distortion), i.e. a wide lens the pinhole pipeline doesn't undistort. The
      draft came out LOW (vehicles 2.6 m vs depth 0.8 m), 0 bays, and now carries a field-of-view warning (35-130 deg
      plausible). Next: undistort with GeoCalib's radial model (k1) in the pipeline and re-check on both cameras.

79. **2026-10-23 Phase 2 step 3: lens distortion + re-evaluation on real stills (setup service / geometry modules only; not
    deployed, flag off, worker untouched).**
    * **Radial distortion** (`geometry/camera.py`): GeoCalib's forward model (`f * p * (1 + k1 r2 + k2 r4) + c`, read from the
      pinned source) with an EXACT inverse (bisection on the monotone branch; GeoCalib's own undistort is first-order and poor at
      k1 ~ -0.46; points beyond the lens model's range are NaN). `CameraModel` gets k1, k2; `to_ground` / `ground_resolution` /
      `surface_polygon` take the camera (undistort first, then K and H); `to_image` distorts; `horizon_px` is the bent horizon.
      `calib.py`: vehicle height, typed-distance height and depth back-projection all undistort first. Service `/calibrate` takes
      `camera_model` (pinhole / simple_radial / radial, default simple_radial) and returns k1, k2; API body `camera_model`.
      Synthetic camera shaped like camera-1: area within 2 % of truth (3.5 % off ignoring the lens), height within 5 %.
    * **Real stills (camera-1 live frame, camera-2 frame of 2026-10-21; same physical camera, different cars):** pinhole GeoCalib
      said 20 deg FOV / 4.8 deg tilt (wrong); radial gives camera-1 f 1986 / k1 -0.46 / 13.8 deg / 59 deg FOV, camera-2 f 1895 /
      k1 -0.39 / 14.1 deg / 61 deg: consistent. Vehicles vs depth scale agree within 8 % / 12 % (confidence LOW -> HIGH).
    * **But an independent check says agreement is not accuracy.** Wheel-ground contact points of two known cars through the
      calibration: Honda HR-V (2.610 m) came out 2.06 m (-21 %), Suzuki Celerio (2.425 m) 2.02 m (-17 %): both priors are
      ~20 % low. Per-car height scatters +-15-19 %; two stills of the same camera give 2.65 vs 3.17 m. Hypothesis "the box's
      depth extent biases it" tested: no single extent fits both cars. So **confidence is now evidence-based**: HIGH only with
      a typed reference distance (+-5 %), MEDIUM = two agreeing priors (expect +-25 %, the warning says so), LOW = one prior or
      disagreeing. **Held-out validation of the reference route:** calibrate camera-1 with the HR-V wheelbase -> h 3.24 m; the
      never-seen Celerio through it: 2.37 m vs 2.425 (-2.1 %); camera-2 with the Celerio -> h 3.36 m, HR-V through it 2.62 vs
      2.61 (+0.5 %); the two calibrations' h differ 3.8 %. Lens, tilt and focal length are right; only unaided scale priors are off.
    * Painted lines as a scale source: not applicable here (brick paving, no painted bays); not implemented.
    * **Surface and bays rebuilt** (`bev.py` Grid / surface_grid / car_footprint, `capacity.py` pack_bays, `pipeline.py`): a SAM
      polygon covering a car's BODY stretched the surface ~10 m behind it (82.7 m2 on camera-1), so the surface is now a ground
      raster: cells projected into the image and kept on mask pixels with the vehicles' boxes cut out, strips < 1.2 m and pieces
      < 3 m2 dropped, each whole car's footprint added (7.5-8.3 m2 vs a real ~7.9 m2), bays oriented from the cars (robust
      vote: camera-1 175.8 deg, camera-2 2 deg = the same axis), car bays placed first (centred on each car, neighbours touching),
      free bays on the lattice around them, packing on the raster (poles stop bays). `mask_to_polygon` bridges a pole (3.5 % of
      the width). Vehicles of all stills are pooled for the scale. **Physical limit found:** this camera (3.2 m high, 14 deg
      tilt) looks along the row, so beyond ~18 m one pixel is > 6 cm of ground and a far car's position is meters off: those
      cars get no bay and the draft says so (`bev.cars_too_far`). Result: camera-1 2 bays on its 2 near cars (3 far cars ignored),
      camera-2 3 bays on 3 (3 ignored); overlays checked: each bay sits under its car (wheels inside). No free bays fit.
    * Fixed on the way: bay layout phase after padding, 6 cm/px car range, `isinstance` -> duck typing in `to_ground` (Streamlit's
      test runner purges modules, a second class copy broke the check in the full suite only).
    * Tests: 254 passed, 16 skipped (was 231): `test_geometry_distortion` (10), `test_geometry_raster` (11), pipeline / API
      additions (pooled stills, lens forwarding, confidence policy). Setup image rebuilt (4 contract tests in the build).
      Next: Admin Accept / activate endpoints, the setup page, then the worker's flagged read of the active geometry.

80. **2026-10-23 Phase 2 step 4: Accept / discard endpoints + the Camera setup page (built, tested in a real browser; NOT
    deployed, flag off, worker untouched).**
    * **API** (`api/geometry_routes.py`; admin token + `CARPARK_GEOMETRY_ENABLED` on all but the status): `POST
      /api/cameras/{id}/setup/activate {version, accept_unverified}` (a draft, or an earlier version = rollback; one
      transaction in the Step 1 store, never two active versions; a version without a typed reference distance - confidence
      below HIGH - is refused with 409 unless `accept_unverified`; stale 409, unknown 404), `DELETE .../setup/draft?version=N`
      (one draft, or every draft without it; an active / earlier / stale version is refused 409), `GET .../setup/geometry`
      (summaries + the active one), `GET .../setup/geometry/{version}` (one in full), `GET /api/setup/status` (answers while the
      flag is off, contacts nothing then), `GET /api/cameras/{id}/geometry` (viewer token: the ACTIVE calibration). Store:
      `active_version()` (one cheap query, for the worker's later flagged poll) and `discard_drafts()`.
    * **"Without restarting anything":** activation only writes the geometry store; every reader reads the store per call (no
      cache), so the next read sees it - tested on the same running app (`test_accept_rollback_and_discard_flow`), and in a real
      browser: Accept, then the very next `GET .../geometry` returned the new version. The worker does not read geometry yet
      (a later, flagged step), so nothing live changes; no `live_commands` entry is sent, `live_config` / the slot map are
      unchanged (asserted).
    * **Reference distance up front** (`frontend/src/setupflow.ts`, `components/CameraSetup.tsx`, nav "Camera setup", `#/setup`):
      five steps - Still, **Known distance** (two clicks on the still + the metres; step 3 is locked until it is given or the
      Admin ticks "go on without", which makes accepting need an extra confirmation), Parking surface (ground / not-ground
      clicks -> SAM, candidates), Calibrate (lens model), Review & accept (HIGH badge, height / tilt / surface / bays, warnings,
      the bays drawn over the still, Accept / Discard). A versions table (Activate, Roll back, Discard; a non-HIGH version asks
      again). Flag off: the page says so and offers nothing; setup service down: the compose command is shown.
    * Tests: Python 260 passed, 18 skipped (254 + 6; pyflakes clean); frontend 74 (was 59: `setupflow.test.ts`,
      `components/setup.test.tsx`), `tsc` clean, `vite build` compiles. **Real browser** (Chrome via Playwright, throwaway API
      with a made-up token + the real setup service, both removed afterwards): still -> HR-V wheelbase 2.61 m -> 5 ground / 3 not
      ground clicks -> SAM 32 s -> calibrate 64 s -> HIGH, h 3.24 m, 2 bays on cars -> Accept -> active; no console errors.
    * Fixed on the way: routes catch the builtin `ValueError` (GeometryError's base) instead of the `GeometryError` class: in the
      full test run Streamlit's module purge made two copies of the class and the error escaped as a 500.
    * To use it: `CARPARK_GEOMETRY_ENABLED=1`, `docker compose up -d --build api` (the image needs the new code and page;
      NOT done: demo freeze), `docker compose --profile setup up -d setup`. Next: the worker's flagged read of the active
      geometry (`active_version()` poll) and the metric overlays / counts.

81. **2026-10-23 Phase 2 step 5: the worker's flagged read of the active geometry + metric overlays / counts on the dashboard
    (built, tested end to end with a real worker + real browser; NOT deployed, flag off).**
    * **One switch:** `geometry_enabled(env)` (CARPARK_GEOMETRY_ENABLED; in the numpy-free `geometry_store.py`) is used by the API
      Config and by `scripts/worker.py`, so they can't disagree. Off (the default): the worker imports nothing new, reads nothing,
      the snapshot is byte-for-byte what it was, the API strips any `geometry` key from the stream (a mismatched worker can't leak
      it), the SPA renders nothing.
    * **Worker** (`src/carparking/geometry_live.py`, `Worker(geometry=...)` hook in `live.py`, `scripts/worker.py`): the feed polls
      `GeometryStore.active_version()` every 5 s (one cheap query; the record is loaded only when the version changed), for its
      camera (`CARPARK_CAMERA_ID`, already set by the supervisor; PostgreSQL when CARPARK_DATABASE_URL is set, else the camera's
      SQLite). No active version = `None` = legacy snapshot. A failing store (e.g. the table not migrated yet) is logged once, never
      raises into the loop, the last good calibration is served for 60 s, then uncalibrated; a failing metrics computation costs
      no frame. With an active calibration each frame's snapshot gains `geometry`: per measured vehicle a ground footprint
      (image fractions) and centroid, the nearest-neighbour distance between vehicles and to people / objects (edge to edge, exact
      closest points), the packed bays marked FREE / BUSY by whether a footprint centre is inside, and counts (capacity /
      occupied / free, vehicles, measured, beyond range). Cars cut by the frame edge or beyond 6 cm of ground per pixel are
      counted, not measured; a duplicate detection of one car is one car; the calibration scales to the frame's resolution
      (same picture at 2x gives the same metrics), another aspect ratio = `valid: false`.
    * **Dashboard:** `Metrics` layer (footprints, dashed bay outlines red / green, distance lines + labels, `≈` unless the scale
      was anchored by a typed reference) over the video / frames, toggle remembered in its own key (the Slots / Boxes / Area
      storage is untouched), and a "Measured capacity" card (bays / occupied / free, m2, vehicles measured, calibration version,
      +-% when not HIGH) beside the unchanged slot counters. The stream's throttle key notices a new calibration or a change in
      the bay counts.
    * **Verified end to end** (throwaway DB, real YOLO worker loop on camera-1's live frame with the flag on, throwaway API with a
      made-up token, real Chrome; all removed afterwards): before any calibration the snapshot has no `geometry`; Accept written
      to the store while the worker keeps running -> within seconds `geometry` v1 HIGH, 2 bays both occupied, 2 of 6 vehicles
      measured (4 too far / cut), 1.5 m between the two near cars; the layer lines up with the cars in the worker's frame; toggle
      persists across a reload; marking the calibration stale made the same worker fall back to the legacy snapshot, no errors.
      **API flag off while the worker still publishes:** `/api/status` has no `geometry`, no pill / overlay / card, slot counters
      as before.
    * Tests: Python 271 passed, 18 skipped (260 + 11: `test_geometry_live` x7, `test_live` x3 for the worker hook, the stream
      gating in `test_setup_api`), pyflakes clean; frontend 82 (74 + 8 in `metrics.test.tsx`), `tsc` clean.
    * To use it: `CARPARK_GEOMETRY_ENABLED=1`, `docker compose up -d --build` (worker, api and the SPA all need the new code;
      NOT done: demo freeze), an accepted calibration on the Camera setup page. Next: the drift guard (ORB reference check ->
      `mark_stale`) and metric history.

82. **2026-10-24 Phase 2 step 6: the drift guard + metric history (built, tested end to end with a real worker + browser + a real
    PostgreSQL; NOT deployed, flag off, demo freeze kept).**
    * **Drift guard** (`geometry/drift.py`, `GeometryFeed.observe` in `geometry_live.py`, hooked per frame in `Worker._metrics`):
      the calibration stores the still's static-background ORB features (600, ~50 KB, vehicles and the OSD strips left out) in its
      record (`reference`, written by the calibrate job); the worker compares the current frame (detected vehicles / people
      masked) every 30 s: RANSAC keeps the matches that agree, the 90th-percentile displacement of THOSE matched points is the drift
      (> 0.6 % of the width = ~11 px). **3 confirmed checks in a row** mark the active version STALE (`mark_stale(ver=...)`: only if
      it is still the one that was checked, so an Admin accepting a newer calibration meanwhile is never staled by an old check),
      the feed drops it at once, the worker publishes the legacy snapshot, the story is in the event log and the worker log.
      **Inconclusive never alarms** (night, glare, other light, too few or badly spread matches). Older versions without a
      reference get a worker-side baseline (first frame after loading; weaker, labelled). Env: CARPARK_DRIFT_EVERY / _CONFIRM.
    * **Tuned on REAL frames, and it mattered:** camera-2's frame (same camera, two days earlier, other light and cars) first read
      "drift 27 px" from 48 matches that were all on the burned-in caption "Front Parking 02" (a thin strip; the transform fitted
      to it was extrapolated across the picture) - a false alarm AND a trap the other way (burned-in text stays put when the camera
      moves, so it would vouch for a moved camera). Fixes: displacement measured at the matched points, never extrapolated;
      the caption strip (bottom 8 %) masked like the clock (top 10 %); matches must cover the picture; 90th percentile instead of the
      median (a 2 deg roll: median 10 px, p90 17 px). Real-frame results: shifts read within ~1-3 px (12 px -> drift), 2 deg roll ->
      drift, 1 deg / dusk / night-ish / blur -> ok, black frame and camera-2's frame -> inconclusive.
    * **History** (`geometry_store.py`, same backends as the versions: SQLite in the camera DB, PostgreSQL migration
      `005_geometry_history`): lifecycle EVENTS (draft, activated incl. rollback, stale, discarded, drift_suspected, drift_cleared)
      written in the same transaction as each change, and occupancy SAMPLES (capacity / occupied / vehicles / measured; the worker
      stores one a minute and at once on a change, >= 10 s apart; pruned hourly: samples 90 days, events 1 year). Range queries are one
      indexed scan bucketed in SQL (`t / bucket`, GROUP BY 1): 24 h from 130 k rows in ~6 ms, plan `SEARCH`, same query on PostgreSQL.
    * **API** (flag + viewer token): `GET /api/cameras/{id}/geometry/history?hours&points` (<= 90 days, buckets >= 60 s) and
      `.../geometry/events?limit&since`; version summaries carry `drift_guard`; the metrics payload carries `drift` + `drift_guard`.
      **SPA:** a drift line on the capacity card, on the setup page a "the camera moved" banner (stale and nothing accepted since),
      a Measured occupancy chart (6 h / 24 h / 7 days), the Calibration events list, a Drift guard column.
    * **Verified live:** throwaway DB, real YOLO worker with the flag on (drift every 6 s, confirm 2), a calibration accepted with its
      real reference while it ran -> guard `ok` 0.1 px; frame swapped for a copy shifted 30 px right / 6 down -> `drift` 32.2 px,
      2 checks later "version 1 marked STALE", `geometry` gone from the snapshot while frames and counters carried on, API events
      draft / activated / drift_suspected / stale, history sample, setup page banner + events + history + guard column, dashboard
      back to the legacy view; then everything removed. A numpy float32 in the drift result (not JSON) had silently dropped the
      suspected / cleared events and would have broken publishing: caught by a test, fixed (plain floats, JSON assertions added).
    * Tests: Python 306 passed, 23 skipped (271 + 35: drift module x14, guard / feed x12, history store x6, history API x3), the
      PostgreSQL variants run separately against a throwaway postgres:17.7 (53 passed with the registry / history_pg suites),
      pyflakes clean; frontend 92 (82 + 10), `tsc` clean. With the flag off: no history written, the new endpoints answer 503, the
      worker imports nothing (unchanged tests prove it).
    * To use it: CARPARK_GEOMETRY_ENABLED=1 + `docker compose up -d --build` (NOT done). Calibrations made after this step carry
      the reference; a version accepted before it uses the worker baseline. Limits: the guard abstains at night and in other light
      (it cannot vouch then); a 1 deg roll is below its threshold; a camera moved while the worker was down is only caught against
      a stored reference.

83. **2026-10-24 Phase 2 step 7: re-anchoring a calibration after a camera move, without the full setup (built, tested on real frames;
    NOT deployed, flag off, demo freeze kept).**
    * **Idea:** a bumped camera is (almost always) a pure ROTATION: the optics, the mounting height and the metric scale are unchanged,
      only pitch / roll / heading moved. So instead of SAM + depth + a typed distance again, the calibration's stored background
      (the drift guard's ORB reference) is matched to a fresh still, the rotation is solved and the calibration is carried over.
    * **Math** (`geometry/reanchor.py`): matched pixels are undone with the calibration's own lens, turned into bearing vectors, Kabsch
      (SVD) gives R with b_new = R b_old, RANSAC on the induced homography (3 px at 1920) + 3 refinement passes. New pitch / roll come
      from the up vector, the heading change from the remaining rotation about the vertical; the old ground is rotated by -yaw into the
      new frame. Kept: focal length, lens, height, scale and its confidence. Rebuilt: H, bay polygons (image and metres), surface
      outline, bay orientation, and the drift reference (the new picture's background). `drift.matched()` is the matching shared with
      the drift check (behaviour of the guard unchanged, its 26 tests untouched).
    * **Refuses what is not a clean turn** (the job fails with a reason and writes nothing): no stored reference (made before the drift
      guard), fewer than 25 agreeing matches / under 40 % of them, matches clumped in one corner, residual above 2 px, a turn beyond
      25 deg, a camera that ends up level (< MIN_PITCH), fewer than half the bays still in view. Shift / zoom / another scene fail
      these gates, so it can not silently produce a wrong calibration; the answer is then "do the full setup".
    * **API** (`api/geometry_routes.py`, flag + admin token): `POST /api/cameras/{id}/setup/reanchor {still, from_version?, note?}` ->
      a job (409 while one runs, 410 unknown still, 404 / error for an unknown or unusable parent; default parent = the version last
      in service that has a reference, stale included). Needs NO setup service (no SAM / GeoCalib / depth: the mock asserts it is
      never called). Result = a DRAFT linked to its parent (`reanchor.parent`, event `reanchored`, `draft` / `activated` events carry
      `parent`, version summaries carry `reanchor`). **Activation** keeps all existing checks (HIGH or accept_unverified) and for a
      re-anchored draft also compares the worker's latest frame (< 2 min old) with the draft's own reference: if the camera moved
      AGAIN since, 409 "take a fresh still". Accepting makes it the active version (clears the stale condition: the banner goes,
      the feed picks it up within 5 s with a fresh drift state); the stale parent stays `stale` in the history.
    * **SPA:** a "Re-anchor calibration" button on the "the camera moved" banner and a Re-anchor action per version (disabled, with the
      reason, for versions without a reference), a card (`components/Reanchor.tsx`): fresh still -> Re-anchor -> report (turned N deg,
      heading / tilt / roll, inliers / matches, residual, bays in view) with the new bays drawn on the new still -> Accept and activate
      / Discard / Do the full setup instead. The versions table shows "from vN"; the events list names the parent.
    * **Real frames:** camera-1's live frame turned by KNOWN rotations through its own distorted lens (k1 = -0.46, f = 1973): solved
      angle exact to 0.01 deg, pitch / roll / yaw within 0.05 deg, bay corners land within 0.1-0.7 px of the independent ground truth
      (turns of 0 / 1 / 5 / 8.5 deg); the old reference reads "drift", the new one "ok". One turn (-2 deg pitch, +3 deg yaw) was REFUSED
      (only 31 of 98 matches agreed: this low-texture camera sees mostly pavement and sky) - the safe failure, the Admin does the
      full setup. Limit worth knowing: a re-anchor needs enough background texture; a mostly bare view may refuse.
    * Tests: Python 347 passed, 23 skipped (306 + 29 re-anchor core + 7 API + 1 feed hand-over + parametrised cases), pyflakes clean;
      the PostgreSQL variants of store / history / guard run against a throwaway postgres:17.7 (50 passed, none skipped); frontend 101
      (92 + 9), `tsc` clean. With the flag off the endpoint answers 503 and nothing imports it in the worker.
    * To use it: CARPARK_GEOMETRY_ENABLED=1 + `docker compose up -d --build` (NOT done). Only calibrations made with the drift guard
      (item 82 onwards) carry the reference it needs; older ones need the full setup once.

84. **2026-10-24 Phase 2 step 8: the metric history on the main Analytics page (frontend only; built and tested; NOT deployed, demo
    freeze kept).**
    * New "Measured occupancy" section on the Analytics page (`components/MetricAnalytics.tsx`, rules in `analyticsMetrics.ts`), above the
      legacy charts and driven by the page's own range (24 h / 7 / 30 / 90 days -> `GET /api/cameras/{id}/geometry/history?hours=&points=120`,
      bucketed in the database, polled once a minute): KPI tiles (average and peak bay utilization, bays, and - from the live frame's
      calibrated metrics - surface m² with its confidence, occupied now / bays, vehicles now with measured / beyond-range), a bay
      utilization trend (average occupied area, peak line, bays dashed) and the average fullness by hour of day (sample-weighted).
    * **Flag-gated and clean fallback:** no new server code. With CARPARK_GEOMETRY_ENABLED off the history endpoint answers 503, with no
      calibration / no samples it is empty: the section then renders NOTHING (the page is exactly the legacy one, no layout gap, no
      console error - asserted by tests for 503, 404, no samples, a network failure and no camera). The live "now" tiles only appear
      while the frame is the calibrated picture (`geometry.valid`).
    * Tests: frontend 115 (101 + 14: rules x4, section x6, page legacy fallback / addition x2, plus the failure matrix), `tsc` clean,
      production build ok (into a scratch dir). Python untouched (347 passed, 23 skipped as of item 83).

85. **2026-10-24 Phase 2 final audit and deployment readiness (docs + one wording fix; NOT deployed, demo freeze kept).**
    * Audit: Python 347 passed / 23 skipped (PostgreSQL variants: 50 passed on a throwaway postgres:17.7, none skipped), pyflakes clean;
      frontend 115 passed, `tsc` clean; compose config valid; the parity test (Streamlit untouched) green. Every CARPARK_* name the code
      reads is in `.env.example` and vice versa. Phase 2 migrations 004 / 005 are additive, run once at api start, nothing pending.
      The project is not a git repo, so "uncommitted changes" can not be checked: the rollback anchor is the image, tagged
      `stomeetings/carparking:pre-phase2` (= the running image 3ebbb785c7cf; a local tag, nothing restarted).
    * Fixed: `.env.example` said to set the flag for the api service only; the worker reads it too (both get `.env`): reworded.
    * New `docs/PHASE2_DEPLOYMENT.md`: audit table, env defaults, the single-command deployment (`docker compose up -d --build` after
      setting CARPARK_GEOMETRY_ENABLED=1 in `.env`), the first interactive calibration, and a four-level rollback (UI roll back, flag off,
      previous image tag; data tables are additive and stay).

86. **2026-10-24 Phase 2 DEPLOYED live (authorized by the user), flag on, nothing calibrated yet.**
    * Before: PG dump + SQLite backups in `backups/` (pg-before-phase2.sql, parking_history-before-phase2.db, the camera-2 DB); image
      tagged `stomeetings/carparking:pre-phase2`. `.env`: `CARPARK_GEOMETRY_ENABLED=1` appended (no other change).
    * `docker compose --profile setup up -d --build` (the build ran the whole test suite, exit 0): api, worker, ui, sync, go2rtc
      recreated, the setup service created; db untouched. All healthy 25 s later.
    * Verified: migrations 004_camera_geometry and 005_geometry_history applied (tables empty); /api/health 200; /api/setup/status
      enabled, setup service up, model pins match the manifest (models load on first use); the worker logs "geometry: reading the active
      calibration of camera-1 / camera-2 every 5 s; drift check every 30 s ... history every 60 s". No errors in api / worker logs.
    * Open: the first interactive calibration of camera-1 / camera-2 (needs the Admin's typed distance); until one is accepted the
      cameras show exactly the legacy view and no drift guard / history runs. Stop the setup service afterwards (`docker compose stop setup`).

### Start here next session (state 2026-10-24, items 1–86; everything below is deployed and verified live)

**PHASE 2 (metric geometry) - read this first (items 76-86, newest state).** Deployed live 2026-10-24 (item 86) with
CARPARK_GEOMETRY_ENABLED=1 in .env; migrations 004 / 005 applied; api, worker, ui, sync, go2rtc recreated, the `setup`
service (SAM 2.1 / GeoCalib / Depth, profile `setup`) is running. **NOT YET CALIBRATED: the Admin still has to do the first
interactive calibration** (SPA "Camera setup": still, a typed known distance for HIGH confidence, surface click, Calibrate,
Accept; then `docker compose stop setup`). Until accepted the cameras show the legacy view and no drift guard / history runs.
After the first accept: check read-only that snapshot.geometry appears (footprints, distances, capacity card), the drift guard
reports ok, history samples accrue (GET /api/cameras/{id}/geometry/history) and the Analytics page shows "Measured occupancy".
What exists: geometry core + versioned store (76-77), setup service (78), lens distortion (79), accept / discard + setup UI (80),
worker feed + dashboard overlays (81), drift guard + history (82), re-anchoring after a camera move (83), Analytics metric section
(84), audit + docs/PHASE2_DEPLOYMENT.md (85, with the rollback ladder: UI roll back, flag off + `up -d --no-build`, CARPARK_TAG=
pre-phase2). Backups before the deploy: backups/ (PG dump, SQLite copies); image tag stomeetings/carparking:pre-phase2.
Tests now: Python 347 (+23 PG skips, verified on a throwaway postgres), frontend 115. Not a git repo: no commit history.
Docs: docs/PHASE2_GEOMETRY_PLAN.md, PHASE2_SETUP_SERVICE.md, PHASE2_DEPLOYMENT.md. The older text below is the pre-Phase-2 state.

**Where we are.** The user's 4-step plan is done (Steps 1–3 confirmed by the user, Step 4 = PostgreSQL history /
analytics deployed). Since then, all deployed and checked live, the user's latest tests pending:
- 57 page "reload" fix (useLive no longer blanks the dashboard) · Dynamic mode = read-only slot editor (API 409)
- 58 slot NMS (merge IoU > 0.5, or > 0.35 at the same depth; never pairs that held 2 cars) · Refresh data button
- 59 auto-sync 20 s (picker 20/30/5 s/live; changes sent at once) · display rules: no FREE behind / under vehicles
- 60 one shown state for video banner + cards (SlotTracker.shown) · one vehicle = one BUSY · EXACT polygon overlap
  (cv2.intersectConvexConvex gave IoU 11.85 on nearly shared edges: dynamic_iou._clip_area now)
- 61 FREE judged on the ground footprint (UNKNOWN only if > 80 % of it is behind vehicles in front)
- 62 Settings > "Counting thresholds (Dynamic mode)": hidden_ground 0.8, ground_share 0.4, vehicle_iou 0.2,
  worker option "tuning", applied live, defaults unchanged
- 63 video stall watchdog on decoded frames (5 s) + Refresh data reconnects the video
- 64 counter cards filter the video's slot overlay (Free / Busy / Unknown; Total = all)
- 65-66 Slot editor "Test all / selected slots" preview (read-only), works with monitoring OFF: grabs a frame from
  go2rtc and runs the camera's detector in the API (first test ~8 s, then ~2.6 s)
- 67 near pass (640 px, cars close to the lens: the maroon car 0.24 -> 0.93) · Fixed point rules tolerate a car
  parked slightly forward · camera-1 entry hold 3 s (entries ~6-9 s instead of ~20-25 s) · History paginated
  (20 / page, parked first) · go2rtc restarted after "wrong response on SETUP"
- 68 the white hatchback hidden behind the white sedan (one merged box) counted again: zoom "behind" crop + the
  swallowed-box rule -> 6 cars = 6 BUSY
- 69 slot editor works on a snapshot (the camera's own HTTP still, 0.7 s): captured on open, Refresh snapshot,
  Test all / selected on it (no video stream needed; detection warmed up in Fixed mode)
- 70 overlay toggles (Slots / Boxes / Area) remembered in localStorage · 71 one hidden car = one BUSY slot
  (SlotTracker._owned): Front Parking 02 7 -> 6 BUSY
- 72 hidden max 120 s · slot map reset (4 slots relearned; the hatchback behind the sedan is NOT counted: 4 BUSY
  for 5 cars, awaiting the user's choice) · counter cards hold their counts through model reloads (useHeld)
- 73 a whole car close to the lens may overhang the area wall (75 % inside): the maroon car counted again; the
  hatchback behind the sedan and the car behind the pillar are not (dynamic limits, see item 73)
- 74 area coverage 70 % for whole boxes (90 % kept for boxes cut by the frame side); DEMO FREEZE requested by the user
- 75 merged sedan + hatchback box split (1280 px second pass + strip crop): live 5 BUSY = the 5 cars

**Live stack** (docker compose, project "carparking"): worker (scripts/supervisor.py, one worker per camera), api
(FastAPI + React SPA :8080), ui (Streamlit :8501, UNTOUCHED backup: app.py / scripts/demo_app.py, guarded by
tests/test_settings_parity.py), go2rtc (WebRTC, media 8555), db (PostgreSQL 17.7, volume pgdata), sync
(scripts/sync.py, SQLite -> PostgreSQL every 2 s). Data in ./data. Old container `carparking` stopped (rollback).
Cameras: camera-1 "Front Parking 02" (Dynamic, monitoring on, scope "Camera 1 | yolov8s"); camera-2 "Camera 2
(test)" = the same physical camera, Fixed mode, monitoring OFF, 5 drawn slots (S1-S3, S5, S6).

**How to work** (keep doing it this way):
- Deploy: `docker compose up -d --build` (the build runs all tests: now 172 Python + 52 frontend). A failed build
  leaves the running stack untouched.
- Quick Python tests without a build: `MSYS_NO_PATHCONV=1 docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v
  "C:\F\Projects\CarParking:/src:ro" -w /src stomeetings/carparking:local sh -c "pip install -q pytest==8.4.2
  httpx==0.28.1 >/dev/null; python -m pytest -q -p no:cacheprovider tests/<file>"`. Frontend: `cd frontend; npx
  vitest run` and `npx tsc -b --noEmit`. PostgreSQL tests: throwaway postgres:17.7-alpine (see item 56).
- Live checks: read-only only (GET /api/status?cam=, /api/frame.jpg?raw=1, SQLite opened `file:...?mode=ro`
  inside the worker container, `docker compose cp` scripts to /tmp in a container with MSYS_NO_PATHCONV=1).
  Browser checks: mcr.microsoft.com/playwright:v1.63.0-noble, channel chrome, scripts in the session scratchpad.
- The user's screenshots / images do NOT reach the session: always look at the live frame (draw slots + boxes).
- Secrets: tokens / camera passwords only in .env; never print them (API logs mask ?token= since item 57).
- Tests never touch ./data or the live DB; ask before deleting anything. Log every change here.

**Open / for tomorrow** (nothing half-done in the code):
1. User to test items 57-66 (Ctrl+F5 first). Test slots on camera-2 with monitoring off.
2. Worker speed: ~3.6 s / frame with no viewer, ~7.5 s while go2rtc transcodes at 150 % CPU (item 67). Smaller
   detection sizes are ruled out (they merge the white cars). Still open: SD stream on the dashboard (less go2rtc
   CPU), fewer zoom re-checks, or a GPU for the 2-3 s target.
3. Slot 22 (car behind the pickup at the pillar) logs false entry/exit pairs every 10-40 min (exits are hidden_max
   timeouts, the car never left). Proposed, awaiting the user's OK: don't close a session on the timeout while
   >= 90 % of the slot is hidden; or cut that corner with the parking-area polygon.
4. Camera-2 match rule is "ground" but S2 / S6 are drawn on the cars' bodies (S2: the white sedan's wheels are 24 %
   of its height below it, beyond the 15 % tolerance of item 67, so S2 shows FREE). User to choose: draw the slots
   on the ground (where the wheels stand), or match rule "overlap" / "centre point".
5. Camera clock: on 2026-10-21 the Hikvision OSD was ~15 min slow (frame 18:11:53 server = 17:56:56 OSD); fix via
   NTP on the camera (Configuration > System > Time Settings). Check again.
6. Retired slots 3 / 19 (in front of the stairs, left of the white sedan): a vacant bay there has no slot, so it
   can never show FREE until a car parks there again (Dynamic mode learns bays from cars).
7. One bogus occupancy sample (0/0/9, 2026-10-21 08:00 UTC) in PostgreSQL occupancy_samples: harmless, ages out;
   delete only if the user asks.

#### Older start-here notes (items <= 43, before the compose migration; kept for reference)
0000. LIVE NOW = docker compose (project "carparking"): worker, api + React app http://<host>:8080 (viewer token
      CARPARK_VIEW_TOKEN, admin CARPARK_API_TOKEN, both in .env), Streamlit http://<host>:8501, go2rtc (media 8555).
      Data in ./data (CARPARK_DATA_DIR). Code is baked into the image: after editing, `docker compose up -d --build`
      (the build runs all tests). Old single container `carparking` is stopped (Exited) for rollback:
      `docker compose down; docker start carparking` (it uses the pre-migration files in the project root).
000. Items 42-43: the live container (bind-mounted) runs the new page after a reload: Mode 5 is a viewer; switch
     "Monitoring (background worker)" on once (it starts off), the page then starts scripts/worker.py itself
     (embedded). Close any old tab still running the in-page loop first (two loggers until it reruns).
00. Item 41 (Docker): to move the live app to compose, with the user's go-ahead: `docker stop carparking`, then copy
    parking_history.db, config/ and dataset_raw/ into ./data/ (DB not in use once stopped), `docker compose up -d
    --build`, check http://localhost:8501, and only then remove the old container. Roadmap step 2 next: the
    headless worker (monitoring independent of browser tabs).
0. Item 39 (duplicate slots): after the restart, expect slots 1, 3, 4, 8 to show "DUPLICATE of N" and disappear
   within a minute (event log notes "duplicate slot merged into slot N"); check no real bay behind a car goes.
1. **Restart the RTSP stream (or reload the page)** so items 35–40 are live (item 40: zoom re-check, exit buffer 30 s; the sidebar exit slider default changed, so a page reload resets it to 30). Since item 36 an edit to any `src/`
   file reloads the whole `carparking` package on the next rerun; a stream loop that is already running keeps the
   old code until it is stopped/started. Check: BUSY = vehicle count, an emptied bay turns green ("N FREE") after
   the exit buffer (60 s) or at most hidden_max (30 s) + exit buffer when it is hidden behind cars, a neighbour's
   box never keeps a bay BUSY, the In/Out sessions close at the departure time.
2. Live setup now: camera name **`Camera 1`** (scope `Camera 1 | yolov8s`). The user reset the slot map again on
   2026-10-19 18:14 (local): slots 1–6 = maroon car, white SUV, motorbike, dark car, far cars; slot 7 learned later
   (small box at the far end by the pillar, FREE: check it). Evening 19:25: motorbike gone → slot 3 FREE (green);
   the white SUV is detected narrower at night, matching slot 4, so slot 2 reads UNKNOWN. Saved walls in
   `settings`: cut_top 10, cut_left 10 (cut_right default 40). COCO yolov8s, 1920 px, conf 0.30, Dynamic mode.
   UI defaults: occlusion 0.6, hidden_max 30 s, exit buffer 60 s, hold 15 s.
3. Dynamic-mode pipeline, in order (fixed maps: only the match + occlusion steps):
   `merge_dets` duplicates (IoU 0.45 + same-width slices, item 33) → hard wall `in_area` (31) → phantom boxes
   `phantom_reason` (28) → tiny boxes `is_tiny` (34) → one-to-one IoU match → an unmatched box may hold ONE bay
   (centre inside or IoU > 0.5, `_claims`, 38) → duplicate slots on another car merged after 60 s (39) → phantom slots UNKNOWN / retired (28), tiny slots retired (34) →
   occlusion HIDDEN only if the cars in front cover ≥ 60 % together, at most hidden_max (35) → exit buffer →
   events → HistoryDB sessions (out_time = first frame without the car).
4. Overlay (item 37): FREE green-filled + "N FREE", BUSY red, UNKNOWN grey, HUD "FREE | BUSY | TOTAL | UNKNOWN",
   scaled for the 2560 px stream. In Dynamic mode a bay exists only after a car has parked in it; "Reset slot map"
   deletes the bays of absent cars (FREE 0 after a reset with empty bays is correct). For always-visible empty
   bays use Fixed (Pre-defined) mode (items 24–27, canvas editor, not yet used live by the user).
5. Watch over a day: real entries/exits and durations, retirements (notes in the event log), whether any real car
   is ever dropped ("phantom boxes ignored N" in the status line), how the night (IR) detections behave.
6. Not yet checked by the user in the browser: the hybrid option, magenta person boxes, the "raw cam N, coco M" metrics.

### Open
- Map hygiene: slot 3's old open session (pre-reset map) is closed by the reset; overlapping duplicate slots from
  older maps are gone after the reset. If BUSY ≠ vehicles again, look at the slot map first
  (`SELECT slot_id, poly FROM slots WHERE camera = 'Camera 1 | yolov8s' AND retired IS NULL`).
- Slot 6 of the old map (sign on the wall) was inside the walls; with cut_top 10 % it is out. A saved ROI polygon
  would make the lane exact (placeholder in the UI = the covered lane of Front Parking 02).
- The canvas package (`streamlit-drawable-canvas`) is only pip-installed in the running container: rebuild the image
  from `requirements.txt` before re-creating the container.
- Model robustness: the current `cam_front_parking_02.pt` was trained on daytime frames. Collect more over several
  days incl. night (IR), rain, frames without the bike / with other vehicles; review labels, retrain (CPU only).
- False positive: 0.27 "motorcycle" under the pickup's rear wheel (inside the ROI); conf 0.35 removes it; a retrain
  with that frame reviewed should fix it properly.
- Legacy history/slots under scope "Camera 1" (pre per-detector scopes) are still in `parking_history.db`; migrate
  or delete them only with the user's OK and with the stream stopped (never touch the live DB from tests).
- **Security**: the camera password is in `demo_app.py` as the RTSP fallback. Move it to `$CARPARK_RTSP_URL` before
  rebuilding/pushing `stomeetings/carparking` to Docker Hub.
- OBB is not usable on this camera; Bay model only for high top-down views of painted bays.
- Hybrid costs 0.27 s/frame vs 0.09 s cam only on CPU: fine for parking (hold ≥ 15 s); use the sub-stream if the
  PC gets busy.
- Each browser tab running Mode 5 runs its own loop and writes to the same DB: keep one tab open.
- Universal product goal (later phases): per-camera models `models/cam_<slug>.pt` via the same collect → auto_label
  → review → train pipeline; multi-camera running without a browser tab (background worker) not started.

## 2026-10-12 → 2026-10-13: OBB results, auto-router, live webcam, Mode 4 (Dynamic IoU)

Note: Claude had no shell in these sessions. **None of the code below has been run or tested by Claude**
(`pytest` included). Run `pytest tests/test_dynamic.py` first.

### State
| Item | Status |
|---|---|
| OBB holdout run (`experiments/yolo/pklot_obb_holdout-PUCPR-3`, 20 epochs) | Done. PUCPR mAP50 fell after epoch 1 (domain shift, labels verified correct), so best.pt = epoch 1 → `models/parking_yolo_obb_holdout-PUCPR.pt` |
| Fair OBB model (`models/parking_yolo_obb.pt`, day split, all lots) | **Not trained yet.** The app/scripts use it automatically once it exists |
| App modes | 1 image · 2 video · 3 live webcam (OpenCV) · 4 Dynamic IoU (new) |
| Git | **Still no commits** |

### Done (in order)
1. **OBB inference tuning** (`yolo_engine.py`): OBB runs at `obb_imgsz=1280`, `obb_iou=0.45`; optional own tiling
   (`tile_starts`) merged with `rotated_nms` (IOS 0.6). Confidence slider default 0.20.
2. **Auto-router** (`src/carparking/view.py`): HoughLinesP measures the diagonal share of line length and segment
   density → Auto picks Baseline or OBB (+ tiles on dense views); manual override kept; badge shows the measurements.
   Fixes: HoughLinesP shape `reshape(-1, 4)`; stricter Hough (threshold 60, minLineLength 40, maxLineGap 10) and
   `diag_share` 0.55 after straight layouts were routed to OBB. Thresholds are still uncalibrated.
3. **Comparison script** `scripts/compare_models.py`: baseline vs OBB over a folder of own images (e.g. Pakistani
   lots) → side-by-side jpgs + `counts.csv` (incl. auto pick and router measurements) in `experiments/compare/`.
4. **Live webcam**: `scripts/live_yolo.py` (cv2 window) and Mode 3 in the app (OpenCV `VideoCapture` loop in
   `st.empty()` placeholders, route once on the first frame, model every Nth frame, PNG display). Replaced
   `st.camera_input`.
5. **Mode 4: Dynamic IoU (camera independent)**, new module `src/carparking/dynamic_iou.py`; Modes 1–3 untouched.
   - **Slot sources** (sidebar "Slot Source"):
     - *Auto-Anchor Grid* (default, `anchor_slots`): vehicles grouped into left-right rows (`vehicle_rows`), bay pitch
       from neighbouring cars, empty bays interpolated in gaps and `extend` bays past each row end.
     - *Trained Bay Model* (`bay_slots`): every bay the routed bay model finds (free or busy) is a slot.
     - *Hough Line Detection* (`find_slots`): CLAHE + adaptive threshold + closing → Canny → HoughLinesP → dominant
       angle ±15° → dividers → slots; filters: aspect 1.3–4.0, area, >30% green = trees, optional top-of-frame cut,
       continuous rows of ≥3 equally sized bays, duplicate removal.
   - **Vehicles**: COCO `yolov8n.pt` (car/bus/truck, default; `yolov8s.pt` used if placed in the repo root) or the bay
     model's BUSY boxes.
   - **Occupancy** (`occupancy`): BUSY if a vehicle covers > 25% of the slot or IoU > 0.35 (both sliders). Polygons are
     convex-hulled first, so OBB corners in any order work.
   - UI: yellow vehicle outlines, numbered red/green slots, FREE/BUSY/TOTAL, table with coverage and IoU, image upload
     and live webcam.
6. **Tests** (`tests/test_dynamic.py`): OBB tiles + rotated NMS, view router, Hough slots on a noisy synthetic image,
   coverage vs IoU, bay-model slots, anchor-grid row clustering and gap interpolation.
7. **2026-10-14, Auto-Anchor fixes** (after visual test on PUCPR): slots are now rectangles rotated to the row angle
   (row direction from the car centres, pitch measured along the row). Row clustering tightened: once a row has 2
   cars, a new car must lie within 0.35 × car height of the row's line (perpendicular), so parallel lanes no longer
   merge; side-by-side spacing 0.5–4 car widths; single-car rows link only within 35° of horizontal. New test with
   two slanted (20°) parallel lanes.
8. **2026-10-15, Stage 1 of the Mode 4 blueprint: evaluation.** `src/carparking/evaluate.py` (PKLot XML → GT bays,
   Hungarian matching on IoU, per-image + aggregate metrics) and `scripts/eval_slots.py` (runs a slot source over a
   lot, prints table + summary, CSV to `experiments/eval/`). Tests: `tests/test_evaluate.py`. `scipy`, `pandas` added
   to requirements (already installed via ultralytics/streamlit). Blueprint stages 2–5 (rectification + direction-first
   clustering, tracking + persistent slot map, hysteresis + camera-shift, source fusion) wait for these numbers.
9. **2026-10-15, Auto-Anchor in any orientation**: left-edge columns (cars stacked top-bottom) were never grouped,
   because rows had to run within 35° of horizontal. `vehicle_rows` now runs one clustering pass per direction
   (0°, 90°, 45°, 135°, each ±35°) and accepts the longest row found, then repeats on the remaining cars.
   `anchor_slots` points each row's direction along its own order. Test: a vertical column next to a horizontal row.
10. **2026-10-15, Auto-Anchor parking-area limits** (capacity had jumped to 142, with slots on roads and trees).
    Only virtual bays are filtered; detected cars always keep their slot. A row needs ≥3 cars with at least two side
    by side; gaps of more than 3 empty bays are not filled; a virtual bay must be within 2.5 car sizes of a parked car;
    foliage (>30% green) dropped; optional manual `roi` polygon; duplicates (IoU > 0.3) dropped.
11. **2026-10-15, real cars filtered too**: cars on the street still got red slots, because the checks above only
    applied to virtual bays. Now every slot passes the frame / ROI / foliage checks, and the parking area is the
    neighbourhood of trusted rows: a lone car (single or untrusted chain) keeps a slot only within 2.5 car sizes of
    a trusted-row car. A frame without any trusted row now gives no slots.
12. **2026-10-15, ROI as a hard wall + default boundary**: street parking formed its own trusted rows and widened
    the parking area. `roi` now removes vehicles outside it *before* clustering (no rows, no anchors), and no slot
    centre outside it is kept. Default without ROI: `main_lot=True` keeps only the biggest group of touching trusted
    rows (`_main_lot`). App (Mode 4): "Parking ROI" text field (x,y fractions, e.g. `0.05,0.25; 0.95,0.25; 0.95,1;
    0.05,1`), applied to every slot source and drawn in cyan; "Main lot only" checkbox. `eval_slots.py` has no ROI
    option yet.
13. **2026-10-15, top cut-off**: boxes still spilled onto the background road at the top of the frame.
    `anchor_slots(ignore_top=...)` drops vehicles whose centre is in the top band before clustering and rejects slot
    centres there. App: one "Ignore top part of the frame" slider for every slot source (default 0.25), applied to
    the vehicle detections before any slot source sees them, drawn as a cyan line.
14. **2026-10-15, side spill-over**: an edge column extended down into the driveway, and cars beside the lot kept
    slots. `anchor_slots(stay_in_hull=True)`: extra bays past a row's end and lone cars must lie inside the lot
    outline (convex hull of trusted-row car centres + 0.5 car size), so extension only fills toward where other rows
    already reach; a single-row lot gets no end extensions. New `mask` argument (any-shape parking mask). App: a
    "Parking area (hard walls)" box with Cut top/bottom/left/right sliders + optional polygon, combined into one mask
    used by every slot source; outside is dimmed in the image, edge in cyan; "Stay inside the lot outline" checkbox.
    Fix: OpenCV 5 rejected the column view in `convexHull` → `np.ascontiguousarray`.
15. **2026-10-15, cars under trees**: detected cars under the canopy were dropped by the foliage check. Detected cars
    now tolerate up to `car_green=0.6` foliage (empty bays stay at `max_green=0.3`); `hull_margin` 0.5 → 1.0 car
    sizes so cars at the canopy edge stay inside the lot outline.
16. **2026-10-15, lone cars keep their slots**: cars under trees were detected (yellow) but classed as lone and
    failed the reach check. `keep_lone=True` (default): a detected car only has to pass the hard walls (frame, top
    cut, ROI, mask) and the foliage limit; reach and lot outline now only limit virtual bays. Street cars must be
    blocked with the mask. `keep_lone=False` = the old strict rule. Open: `test_anchor_grid_lot_outline` failed
    once with only `[100]` at x=480 (unexplained; diagnostic command sent, not run yet).

### Known limits
- Hough slots on PUCPR are fragile (faint, foreshortened lines); three tuning rounds so far.
- Auto-Anchor: a row needs ≥2 parked cars; empty rows are invisible; how far a row continues is a guess (`extend`);
  rows must run roughly left-right.
- Bay-model slots: a bay the model misses is not counted.

### Next steps: start here
```powershell
cd C:\Projects\CarParking
pytest tests/test_dynamic.py
streamlit run app.py      # Mode 4 → Auto-Anchor Grid → COCO vehicles → PUCPR image
git add -A; git commit -m "Auto-router, live webcam, Mode 4 dynamic IoU"
python scripts\train_yolo.py --task obb --epochs 20 --batch 4 --workers 0 --device 0   # fair OBB model
python scripts\compare_models.py data\test_images   # calibrate the router from counts.csv
```

## 2026-10-09 → 2026-10-12: Dynamic YOLO path, SAHI tuning, YOLO-OBB Stage 1, dual-model app

Note: Claude had no shell in these sessions. **None of the code below has been run or tested by Claude**
(`pytest` included). The user ran the training and the backups.

### State
| Item | Status |
|---|---|
| Axis-aligned YOLO (~15 h on PKLot, day split) | Trained (resumed after a crash). Run: `experiments/yolo/pklot`. Weights: `models/parking_yolo.pt` |
| Backup of the axis-aligned model | Done by user: `models/backup_axis_aligned/parking_yolo.pt` + `run_pklot/` (weights, results.csv, plots). Engine code frozen in `src/carparking/legacy_engine.py` |
| YOLO-OBB model | **Not trained yet.** Target: `models/parking_yolo_obb_holdout-PUCPR.pt` |
| App (`streamlit run app.py`) | Sidebar switch between Baseline and OBB. ResNet path removed from `demo_app.py` (user decision, Option A) |
| Git | **No commits yet.** `models/`, `data/` and `experiments/yolo/` are git-ignored |

### Done (in order)
1. **Resume training**: `scripts/resume_train.py` resumes `experiments/yolo/pklot/weights/last.pt` and copies
   `best.pt` → `models/parking_yolo.pt`. **Do not run it again**: it would overwrite the original weights.
2. **App switched to YOLO** (`scripts/demo_app.py`): `YoloDetector` instead of the ResNet `Classifier`; boxes drawn
   via the existing `viz.draw`; FREE/BUSY/TOTAL counted from detections; confidence slider.
3. **Inference tuning** in `src/carparking/yolo_engine.py`, all without retraining:
   - SAHI sliced inference (640 tiles, overlap 0.3, CPU by default); `sahi` added to `requirements.txt`.
   - YOLO NMS: `iou=0.7`, `agnostic_nms=True`, `max_det=1000`, set via `model.overrides` (SAHI does not pass them on).
   - SAHI merge: `NMS` + `IOS` 0.8, class-agnostic. The old default (GREEDYNMM, IOS 0.5) merged neighbouring bays.
   - Top-band pass: top 15% of the frame re-run with 320 tiles, overlap 0.5, `iou` 0.85 (zoom for far, small bays).
   - Result (user): dense rows improved a lot; **steep diagonals in the top-left were still missed**. Cause: axis-aligned
     boxes of neighbouring diagonal bays overlap, a limit of the box shape itself.
4. **YOLO-OBB Stage 1** (decision D12 in `docs/decisions.md`):
   - `src/carparking/pklot.py`: keeps the bay polygons. `yolo_lines` (axis-aligned) gives the same labels as before;
     new `obb_lines` writes 4 corners from `cv2.minAreaRect`.
   - `scripts/train_yolo.py`: `--task obb` and `--holdout-lot PUCPR`. Each non-default run gets its own data dir, run
     name and weights file (suffix `_obb_holdout-PUCPR`), so it can't overwrite the baseline. No options = old behaviour.
5. **Dual-model app**:
   - `yolo_engine.py` reads the task from the weights file. `detect` keeps SAHI and the top-band pass; `obb` runs one
     whole-frame Ultralytics pass (SAHI merges only axis-aligned boxes).
   - Every detection has `"poly"` (4 corners) plus `"box"`.
   - `demo_app.py`: sidebar radio "YOLOv8 Axis-Aligned (Baseline)" / "YOLO-OBB (Oriented / Advanced)"; if the weights
     file is missing, the app shows an error with the training command.
6. **Tests** added in `tests/test_dynamic.py`: top-band merge, OBB labels (incl. a 45° bay), OBB detection output.

### Blueprint: remaining stages (each needs separate approval)
1. ~~OBB labels + honest split~~ (done, not yet trained).
2. **Baseline report**: held-out-lot mAP and per-image count error, axis-aligned vs OBB, in `docs/experiments.md`.
   Caveat: the current baseline saw PUCPR days in training, so on PUCPR it has an advantage. For a fair comparison,
   train an axis-aligned model with `--holdout-lot PUCPR` too.
3. **High-res + augmentation run**: `--imgsz 1280`, perspective/rotation/HSV augmentation (safe with OBB).
4. **Auto slot discovery**: detector over N frames → stable slot map → per-slot classification every frame.
5. **Cross-domain test on CNR-EXT**, once the E0 camera CSV blocker is solved.
6. **Fine-tune on target CCTV footage** (200–500 labelled frames per camera), once footage exists.

### Next steps: start here
```powershell
cd C:\Projects\CarParking
git add -A; git commit -m "Axis-aligned YOLO + SAHI baseline, OBB Stage 1, dual-model app"
pytest
python scripts\train_yolo.py --task obb --holdout-lot PUCPR --convert-only
python scripts\train_yolo.py --task obb --holdout-lot PUCPR --epochs 1 --batch 8   # smoke test
```
- Open `experiments\yolo\pklot_obb_holdout-PUCPR\train_batch0.jpg`: the rotated labels must sit on the bays.
- Then do the full run (same command with `--epochs 20` or more), then `streamlit run app.py` and switch models on a
  PUCPR image with diagonal bays.
- If OBB misses small, far-away cars: it has no SAHI zoom. Try training with `--imgsz 1280`.

## 2026-10-06 (end of day): current state, resume here

### State
| Item | Status |
|---|---|
| `official_repo/` | **Cloned** (verified: `Dockerfile`, `docker-compose.yml`, `LICENSE.md` present). Excluded via `.gitignore` |
| Docker Desktop | **Installed** (user-reported) |
| Image `ghcr.io/lotvulture/lotvulture:latest` (~3.0 GB) | **Pull in progress** at end of day. Not yet confirmed complete |
| Custom workspace (`app.py`, `models/parking_resnet18.pth`, `data/slots.json`, sample jpg/mp4) | Frozen, untouched |
| Root `.venv` | Last known broken (no `pyvenv.cfg`/site-packages). Rebuild before running the custom app (see the 2026-10-06 investigation section below) |

### Tomorrow: exact steps
1. **Confirm the pull finished.** `docker images ghcr.io/lotvulture/lotvulture` should list `latest`.
   If it was interrupted: `cd official_repo; docker compose pull`.
2. **Start Lot Vulture**
   ```
   cd official_repo
   Set-Content .env "TZ=Asia/Karachi"     # optional: local timezone instead of America/Chicago
   docker compose up -d
   docker compose logs -f                 # wait for the server to start, then Ctrl+C
   ```
   Open http://localhost:8000. Note whether it asks for login/licence and whether it works offline (it sends a heartbeat to license.lotvulture.com every 10 min).
3. **Phase 3a: sample image**
   - New terminal: `cd C:\Projects\CarParking\data; python -m http.server 8080`
   - In Lot Vulture add a snapshot camera: `http://host.docker.internal:8080/sample_parking.jpg`
   - Draw the 16 slots over P1–P16 and record each slot's state vs `expected` in `data/slots.json`.
   - Same image in our app (`streamlit run app.py`, Mode 1) → compare both against `expected`, slot by slot.
4. **Phase 3b: sample video**: our app: Mode 2 with `data/sample_parking.mp4` (one slot FREE→BUSY at 2–3 s,
   one BUSY→FREE at 4–5 s; `generate_sample_video.py` prints which). Lot Vulture needs an RTSP shim
   (MediaMTX + FFmpeg looping the mp4). Decide before setting it up.
5. **Phase 4: laptop camera + print**: print `sample_parking.jpg` on matte A4/A3, tape it flat, 30–50 cm from the
   camera, even light. Our app: `python scripts/infer_video.py --video 0`. **Slots must be re-marked in webcam
   coordinates first** (click-to-define-slots tool still to be built; needs approval). Lot Vulture: needs the webcam
   as RTSP/snapshot (MediaMTX or similar). Decide then.
6. **Record results** (per-slot table for both systems) here and in `docs/experiments.md` (EV0).
7. **Cleanup when finished**: `cd official_repo; docker compose down -v`.

## 2026-10-06 (later): Option 2 chosen: Docker / source reference, strict isolation

### Done
- `.gitignore`: added `official_repo/` (Lot Vulture code is source-available with a non-compete clause; never committed).
- Inspected the official repo **remotely** (GitHub API tree + raw `Dockerfile`, `docker-compose.yml`, `.env.example`).
  No custom workspace files were touched (`app.py`, `models/`, `data/slots.json`, `.venv` unchanged).

### Not done
- **Clone not performed.** Claude had no shell in this session. Run it yourself:
  `git clone https://github.com/lotvulture/lotvulture official_repo` (now git-ignored).
- Docker Desktop / WSL2 not installed or checked.

### Findings (repo inspection)
| Item | Finding | Impact |
|---|---|---|
| Container image | `docker-compose.yml` uses **prebuilt** `ghcr.io/lotvulture/lotvulture:${VULTURE_VERSION:-latest}`, port `8000:8000`, volume `lotvulture-data:/var/lib/lotvulture`, `restart: unless-stopped` | No local compile needed. Docker route is viable |
| Engine | `Dockerfile` (base `nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04`) pip-installs a prebuilt `vulturevision-1.0.0-cp312-linux_x86_64.whl` from **`http://download.lotvulture.com`** (plain HTTP, no hash pin) | Only matters if building the image ourselves. Prefer the published ghcr image |
| Model `.onnx` | Not in the repo. It ships inside the engine/image | Black box, as expected |
| Phone-home | `.env.example`: `VULTURE_BILLING_PORTAL_URL` (default `https://license.lotvulture.com`), `VULTURE_HEARTBEAT_INTERVAL_SECONDS=600` | **Container contacts the vendor every 10 min.** Only our test media goes in (dataset collage), never private footage. Check behaviour when offline during EV0 |
| GPU | GPU block commented out; GPU is Commercial-only | Community = CPU. Our 4 GB GPU is unused |
| Timezone | Default `TZ=America/Chicago` | Override in a local `.env` (e.g. `TZ=Asia/Karachi`) inside `official_repo/` |
| Native Windows build | `windows/build.sh`, `build.ps1`, GStreamer/PyGObject builds via wine/mingw | Not needed for the Docker route |
| Inputs | RTSP / ONVIF / snapshot URL | Our image/video need a local shim (http.server snapshot, or MediaMTX for video) |

### Tomorrow: Docker route, start here
1. Clone (if not yet): `git clone https://github.com/lotvulture/lotvulture official_repo`
2. Install Docker Desktop with the WSL2 backend (needs admin + reboot; your decision). Check: `docker version`, `wsl --status`.
3. `cd official_repo` → create `.env` with `TZ=Asia/Karachi` → `docker compose up -d` → open http://localhost:8000.
4. Phase 3 media test: in `data/` run `python -m http.server 8080`; from inside the container the host is
   `http://host.docker.internal:8080/sample_parking.jpg`. Add it as a snapshot camera, draw the 16 slots, and compare
   with `expected` in `data/slots.json`. Video: needs an RTSP shim (MediaMTX + FFmpeg loop of `sample_parking.mp4`); decide then.
5. Remove afterwards: `docker compose down -v` (also deletes the data volume).

## 2026-10-06: Official Lot Vulture repo: investigation (setup NOT completed)

### Frozen custom workspace (untouched today)
These were not modified: `app.py`, `scripts/*`, `src/carparking/*`, `models/parking_resnet18.pth`, `data/slots.json`,
`data/sample_parking.jpg`, `data/sample_parking.mp4`, `.venv/`.
- Custom app: `streamlit run app.py` → `scripts/demo_app.py` (ResNet-18 FREE/BUSY over the slots in `data/slots.json`).
- Known issue: the root `.venv` was found broken (`python.exe` present, but no `pyvenv.cfg` and no site-packages). It must be rebuilt
  before the custom app runs (commands in the "Tomorrow" section).

### What was done today
- Read the official docs of https://github.com/lotvulture/lotvulture: `README.md`, `docs/DEVELOPMENT.md`, `windows/INSTALL.md`.
- **Not done**: cloning and setup. Claude had no shell in this session, and the docs show a blocker (below).

### Findings: why a "from source, inside official_repo/" setup is not possible on Windows today
- Backend: Python 3.12, FastAPI/uvicorn on port 8000 (`pip install -r requirements.txt`, `python backend/init_db.py`,
  `uvicorn backend.app:app --port 8000`). Frontend: Vue 3, `npm install` / `npm run dev`.
- **Inference engine `vulturevision` (C++, ONNX Runtime) must be compiled.** The Windows build is a *cross-compile on
  Linux* (wine + mingw, `vulturevision/build_windows.sh`). No prebuilt engine wheel is documented. Without it, the
  backend cannot run inference.
- Where the model `.onnx` comes from is not documented in these files. The licence mentions "encrypted neural network
  assets" and "cloud attestation keys". Whether it needs internet/activation is unknown.
- The documented Windows path is the **installer `LotVulture_Setup_<version>.exe`** (GitHub Releases). It installs a
  **system service running as NT AUTHORITY\SYSTEM, auto-starts on boot**, stores data in `C:\ProgramData\LotVulture\`.
  That is a system-wide install. It does *not* live inside `official_repo/`.
- Input: RTSP / ONVIF / HTTP snapshot URL only. No local file or webcam input, so a local snapshot/RTSP shim is needed for our media.
- Licence: Lot Vulture Source License 1.0.0 (source-available, non-compete, no model substitution). Its code must not be
  copied into our code. Keep `official_repo/` out of our git history (add `official_repo/` to `.gitignore`; awaiting approval).

### Decision needed (user)
Pick one route for EV0:
1. **Windows installer (.exe)**: fastest to "running"; installs a SYSTEM service + ProgramData; uninstall afterwards.
2. **Docker**: needs Docker Desktop + WSL2 (Linux container `ghcr.io/lotvulture/lotvulture`); isolated, easy to remove.
3. **Source only**: clone into `official_repo/` for reading/learning; cannot run inference without building the engine on Linux.

### Tomorrow: start here
0. Decide the route above. Clone for reference (safe, read-only use):
   `git clone https://github.com/lotvulture/lotvulture official_repo`
1. Rebuild the custom app's venv (root `.venv` is broken):
   ```
   Remove-Item -Recurse -Force .venv
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   .venv\Scripts\python -m pytest -q
   ```
2. **Phase 3: Sample media test**
   - Custom: `streamlit run app.py` → Mode 1 `data/sample_parking.jpg` (compare with `expected` in slots.json),
     Mode 2 `data/sample_parking.mp4` (P-slot changes at 2–3 s and 4–5 s; printed by `generate_sample_video.py`).
   - Lot Vulture: serve the image as a snapshot URL (`python -m http.server 8080` in `data/` →
     `http://localhost:8080/sample_parking.jpg`), draw the 16 slots in its editor, compare its states slot by slot.
3. **Phase 4: Laptop camera + printed image**
   - Print `data/sample_parking.jpg` (matte, A4/A3), tape it flat, 30–50 cm from the camera, even light.
   - Custom: `python scripts/infer_video.py --video 0`. **Slots must be re-marked in webcam-frame coordinates**
     (needs a small click-to-define-slots tool, not built yet).
   - Lot Vulture: needs the webcam exposed as RTSP or snapshot (e.g. MediaMTX + FFmpeg), to be decided.

## 2026-10-25: item 87, auto slot grid in Dynamic mode (HIGH calibration)
- `GeometryFeed.auto_slots()` (geometry_live.py): the active calibration's packed bays as fixed-slot dicts, only when confidence is HIGH and the frame fits.
- `Worker` (live.py): a Dynamic run with such a grid builds its monitor as a fixed map (ground-point match, no learning, history scope
  "<detector> | auto v<N>"). Every bay exists as FREE and flips BUSY/FREE as vehicles come and go. Grid appears / changes / disappears
  (new calibration, STALE, option `auto_slots` = 0) -> monitor rebuilt, back to the learned map when there is no grid. Fixed runs untouched.
- Snapshot `auto_slots {version, count}`; Worker card shows "auto grid · N bays from calibration vN". Test: tests/test_live.py. Not deployed yet.
- Item 87b: `auto_slots()` no longer uses the stored 6 m-aisle packing (camera-1 v3: 3 bays). `ActiveGeometry.tiles()` re-tiles the measured
  surface (bev.surface_m) with standard bays and no aisles (anchored on the calibration's car bays, cover 0.8), then a second pass turns
  the bays 90 degrees where the ground is too narrow crosswise (strips beside the pillar). camera-1 v3: 9 tiles, checked read-only on a
  dataset_raw frame. Not deployed yet (rebuild + worker restart).
- Item 87c (live frame 2026-10-25 13:00, camera-1 v3): the truck at the far end stood in the notch the calibration cut out of the surface
  (a car too far to measure is not ground), and its "ground" point (10 % of the box height above the bottom) fell past the thin far
  tiles. Fixes: `tiles()` closes notches up to CLOSE_M = 8 m before tiling (tiles 11, was 9; cover stays 0.8, no tiles over the wall or road);
  new match rule `foot` (fixed_slots.FOOT, NOT in MATCH_RULES, so the parity test / UIs are untouched): box bottom centre, "just outside"
  tolerance without the coverage test. The auto grid runs on it. Checked on the live frame: car -> bay 6, truck -> bay 11, one slot each.
  test_api's two snapshot tests failed once when run in the same pytest call as the others (timing), pass alone twice. Not deployed yet.
- Item 87d (camera-1 v3, live frame 13:22): the tilted lattice (polygon axis 175 deg) and cover 0.8 left a bay on the left lobe over the
  stairs. Now `tiles()` snaps the bay orientation to the camera axes when within SNAP_DEG = 10 (unless the orientation came from parked
  vehicles) and needs TILE_COVER = 0.9 -> 10 axis-aligned bays in one lane column x -2.9..2.1 m, y 4.7..29.7, none on the stairs.
  Each slot carries `lift` (image shift of VEHICLE_HALF = 1 m of ground): the `foot` rule tests the vehicle's centre, not its near wheels
  (fixed_slots.validate_slots keeps an optional `lift`). Live frame: car -> bay 7, truck -> bay 10. A free bay behind a vehicle in front
  still reads UNKNOWN (display rule: its ground is hidden). Not deployed yet.
- Item 87e (user request): one extra far-end bay. `tiles()` continues the lane's own row pitch FAR_BAYS = 1 tile past the last measured tile
  (camera-1 v3: 11 bays, last one y 29.7..32.2 m). EXTRAPOLATED, not measured: only 17 % of it is measured surface (the far car's ground was
  cut out at calibration). Live frame: car -> bay 7, truck -> bay 11 (inside). The surface itself reaches y 34.2 m only on a 3 m wide strip,
  so a bay up to 35 m would need a second extrapolated tile (FAR_BAYS = 2), not done. Not deployed yet.
- Item 88 (user request): hybrid auto grid = measured bays + extrapolated bays + operator-pinned bays. Not deployed yet.
  `geometry_live`: `FAR_BAYS` is gone. `ActiveGeometry.measured()` (the old tiling) + `tiles(expected)` -> (polygon, source) with source
  measured / extrapolated; the lane is continued until `expected` bays exist, at most MAX_EXTRAPOLATED = 3 (fixed_slots), DEFAULT_EXTRA = 1 when
  no count is set (today's behaviour). `auto_slots(shape, now, expected, pins)`: an extrapolated bay with < EDGE_KEEP = 50 % of its area in the
  frame is dropped with everything beyond it; `merge_pins` (PIN_IOU = 0.3): a pin replaces the grid bays it overlaps and takes the lowest of
  their ids (the other bays keep theirs, so history stays), a pin that overlaps none is added after the last id; pins have lift 0, source
  manual. Result cached per (version, frame, expected, pins): the frame loop asked on every frame. Bays now carry `source`.
  Storage: `FixedSlotStore.get_auto_grid / save_auto_grid` under its own top-level key "auto_grid" in fixed_slots.json (save() and the registry
  mirror rewrite "cameras" only, never the pins). Worker option "auto_grid" (JSON) carries it to the worker: GET/PUT /api/auto-pins (PUT is
  admin; 422 on a bad count or pin; no settings version, no restart). `live.py`: the auto key includes the stored input so an edit rebuilds the
  monitor; status `auto_slots` = {version, count, measured, extrapolated, manual, expected, sources{id: source}, short?}.
  Frontend: AutoGridPins card in the slot editor (Dynamic mode with an auto grid): draw rectangle / polygon pins over the snapshot with the grid as
  a guide, expected count, Save; LiveView dots extrapolated bays and marks pinned ones "P"; the Worker card says "11 bays (9 measured · 1
  extrapolated · 1 pinned)" and how many expected bays could not be placed. Pins are in image fractions, so calibration error does not touch them.
- Handoff (end of 2026-10-25, to start from tomorrow): items 87-88 (auto grid, far-end bay, hybrid pins) are written and tested but NOT
  deployed. Last full runs: Python 353 passed / 23 skipped (Docker), frontend 122 passed, tsc clean.
  Next steps: (1) confirm which camera "Front Parking 02" is (expected bays / pins are per camera; only camera-1 v3 was checked live);
  (2) get approval, then `docker compose up -d --build` (the build runs all tests); (3) in the slot editor (Dynamic mode) set expected bays = 11
  and check the truck shows BUSY in the last bay, bay 10 FREE, car in bay 7; (4) pin the far bay if the extrapolated one is off;
  (5) not done: a bay up to ~35 m would need MAX_EXTRAPOLATED / expected beyond what is measured (ask first); replay on the live frame after
  the rewrite was not re-run (scratch scripts in %TEMP%\chk).
- Item 89 (user request, 2026-10-26): vehicle-driven bays and footprint matching on the auto grid. Not deployed yet.
  Only on an auto grid (Dynamic mode + HIGH calibration, `FixedSlotTracker(auto={"surface": ...})`); the learned-map dynamic mode (no
  calibration) is unchanged. In `fixed_slots.py`: (1) after the point rule (`foot`), a vehicle with no bay is given the bay its GROUND
  FOOTPRINT (bottom FOOT_SHARE = 30 % of its box) overlaps by IoU >= FOOT_IOU = 0.25; between several, the one whose centre is nearest the
  footprint's; one vehicle, one bay. The verified assignments (point rule first) are untouched. (2) A vehicle with conf >= SPAWN_CONF =
  0.45, its foot on the measured surface, no bay and footprint IoU < SPAWN_IOU = 0.20 with every bay, seen in >= 2 frames over SPAWN_HOLD =
  3 s, gets a bay (ids from DYNAMIC_BASE = 1000, names D1, D2 ..., the footprint as polygon); added before the frame's matching so it
  is BUSY after the normal hold and logs a normal entry. (3) Such a bay is removed (tracker.retired, so the monitor closes its session)
  when it is not BUSY and empty for CLEAR_FRAMES = 30 frames in a row. Grid bays and pins (ids < 1000) are never removed; a new grid
  (set_slots) keeps the vehicle-driven bays. `poly_iou` moved to fixed_slots. The surface = `ActiveGeometry.surface_outline()` (measured
  ground with the notches closed, 0.2 m simplified) projected by `GeometryFeed.auto_surface()`; the worker passes it in run["auto"]. Option
  `auto_dynamic` = 0 turns (1)+(2) off (the monitor is rebuilt). Status auto_slots gets `dynamic` and sources[id] = "dynamic"; the Worker
  card says "+ N vehicle-driven", the live view marks them "D" with a dash-dot outline. Total slots in counts include them.
  Tests: Python 360 passed / 23 skipped, frontend 122 passed, tsc clean. NOT checked on the live camera frames yet (run the replay first).
  Open points: dynamic ids restart at 1000 after a worker restart (history rows of an earlier D1 share the id); IoU on a thin far bay
  against a tall vehicle's footprint is small by nature, so the footprint rule mostly helps wheels just past a bay's near edge.
- Item 90 (user request, 2026-10-27): sub-bays for vehicles sharing a slot, persistent vehicle-driven bays, dashboard preferences. Not deployed.
  `fixed_slots.py`: `FixedSlotTracker(split=True)` (any map: hand-drawn fixed or auto grid): an unmatched vehicle (conf >= 0.45) whose footprint is
  >= SPLIT_INSIDE = 0.9 inside a slot ANOTHER vehicle holds and is a different vehicle (footprint IoU < SPLIT_DISTINCT = 0.3) gets a sub-bay of
  that parent ("9P.2", ".3" ...; ids from 1000; `tracker.parent`). Parent keeps its vehicle: total slots +1 per extra vehicle, busy = vehicles.
  Debounce for every vehicle-driven bay is now SPAWN_FRAMES = 3 consecutive frames AND SPAWN_HOLD = 3 s. Persistence: new bays go to
  `tracker.new_slots` (the monitor already saves those with HistoryDB.save_slots, scope-specific), `ParkingMonitor.start` calls
  `tracker.restore(db.load_slots(), db.max_slot_id() + 1)` (a bay whose centre is in a map slot is that slot's sub-bay), so a restart brings
  them back and ids are never reused in the scope; a cleared bay is retired (not restored). Removal unchanged (not BUSY + empty 30 frames);
  grid bays and pins never removed; a new map keeps vehicle-driven bays, a sub-bay goes with a removed parent. A sub-bay can end up holding
  the vehicle while the parent reads FREE (one-to-one by score): the count of busy bays is still the number of vehicles.
  `live.py`: option `slot_split` (default on, 0 = off, rebuilds the monitor); status: `vehicle_bays` (fixed runs), auto_slots.dynamic/sub and
  sources[id] = "dynamic" | "sub"; monitor.state() slots carry `parent`. Frontend: sub-bays drawn "S" with a dash-dot outline.
  Dashboard preferences: could NOT reproduce a reset. New `src/app.test.tsx` mounts the whole App against a fake backend/socket: layers, Metrics and
  the BUSY filter survive 8 state messages, one without geometry, a "worker reloading" message and a socket drop/reconnect (all pass on the
  existing code). Hardening anyway: the counter-card filter is now per camera in sessionStorage (`useSlotFilter`), so a remount/reload keeps it
  (layers/Metrics were already in localStorage). If it still happens, I need the exact steps (single view or grid, which pill, what the
  Auto-sync badge says): the first thing to check is whether the page itself reloads (new tab state, proxy) rather than the data refreshing.
  Tests: Python 365 passed / 23 skipped; frontend 124 passed, tsc clean. NOT run on live frames yet.

## 91. Slot 2 BUSY from a tyre or bumper over the line (2026-10-26)
Report: with "Ground taken" and "One vehicle, one BUSY slot" at 80 %, slot 2 on the auto grid still showed BUSY when a white SUV's rear tyre touched its edge.
- **Finding:** those two sliders (`ground_share`, `vehicle_iou`) are read and applied correctly, but only in `SlotTracker._unseen` (what is shown as UNKNOWN). They never decide BUSY. BUSY comes from `FixedSlotTracker._match` / `match_score`, whose near-miss rule (foot rule: any foot point within 15 % of the box height outside the bay) and footprint fallback (IoU >= 0.25) had no "how much of the vehicle is on the bay" test. `vehicle_iou` was also clamped to 60 %, so a typed 80 % was stored as 60 %.
- **Fix:** `fixed_slots.ground_on` (the larger of the vehicle's ground footprint inside the bay and the bay under the footprint). The near-miss rule and the footprint fallback now need `bay_share` (default 50 %, Settings slider "Bay held", 10-95 %). A foot point inside the bay is unchanged. `bay_share` rides the existing `tuning` option (applied live, no restart); `vehicle_iou` max is now 90 %. `slot_preview.preview_slots` uses the same default, so Test slots agrees. Settings text says the first three sliders only affect UNKNOWN.
- **Tests:** Python 366+ passed in the touched files (158 in the 6 affected files), frontend 124 passed, tsc clean. New tests: spill over a line, the share decides who holds a bay and is applied live.
- **Not done:** not replayed on the SUV's real frame (the saved camera-1 frame is from 2026-10-25 and has no SUV); not deployed. To fix it on the dashboard: deploy, then set "Bay held" to 80 %. If slot 2 still goes BUSY, send the slot's note/ids so the claiming branch can be identified.

### 91b. Follow-up: slot 2 still BUSY with "Bay held" at 80 % (2026-10-26)
- **Audit of every path that can set BUSY on a fixed map / auto grid:** (1) `match_score` point-inside rule (foot point or lifted foot point inside the bay): had NO ground test, so it bypassed "Bay held"; (2) the near-miss rule and (3) the footprint fallback: gated in item 91; (4) `SlotTracker._claims` / `_owned`: off for fixed maps (`share_cover=None`, `suppress_phantoms=False`); (5) zoom re-check (`monitor._zoom`) only re-looks at BUSY / hidden slots and needs the box centre inside the slot; (6) `_spawn` D / sub-bays are bays of their own, not slot 2. The worker does apply `bay_share` to the auto-grid tracker (`Worker._tune` runs every step for any tracker).
- **Fix:** the point-inside rule is gated too: `ground_on >= min(bay_share, INSIDE_SHARE = 0.5)`. Not the full 80 %: measured on the saved camera-1 frame (2026-10-25) the real car in bay 7 has ground_on 0.74 and the truck in bay 11 0.77, so an 80 % gate there would un-match real vehicles. The stray matches there (car on 6: 0.27, truck on 10: 0.06) are removed.
- **Diagnostics:** each snapshot slot now has `ground` (ground_on of the vehicle matched to it): if slot 2 is red, its `ground` says how much of the SUV's ground is on it (and None = matched by something else, e.g. the zoom or a stale BUSY).
- Tests: Python 368 passed / 23 skipped; tsc clean. Not deployed, not replayed on the SUV frame.

## 92. Overlay toggles (Slots / Boxes / Area / Metrics) really independent (2026-10-26)
Report: with Slots, Area and Metrics off and only Boxes on, other overlay drawing still showed.
- **Finding:** no MJPEG stream exists. In video (WebRTC) mode the picture is clean and every layer is its own SVG shape behind its own toggle (already correct; now tested for all 8 Slots/Boxes/Area combinations). The leak was **frame mode** (the "Frames" pill, or when the video cannot play): the frame images are the worker's `render_overlay` JPEG with the parking area dimmed + cyan edge, vehicle boxes, slot fills/outlines and labels baked in, so the toggles could not remove them (the SVG drew only invisible hover shapes).
- **Fix:** the worker now also publishes the clean frame with every overlay image (option `clean_frames`, default on, "0" = as before; snapshot flag `clean_frames`). The WebSocket frame feed sends that clean frame (`api/main.py` Broadcaster.tick) and falls back to the baked overlay for an older worker / option off. `LiveView` draws all layers itself on a clean frame (`baked` = frame mode on a non-clean feed keeps the old hover-only behaviour, so nothing is ever drawn twice). Metrics was already an SVG layer.
- **Kept:** `overlay` (baked) is still published and `/api/frame.jpg` still serves it, so the Streamlit pages are untouched. Cost: one more JPEG encode per frame in the worker.
- **Lost in frame mode:** the baked texts "HIDDEN / NO ROOM / DUPLICATE" on the picture; slot names/state and hover details are still drawn by the SVG.
- Tests: Python 369 passed / 23 skipped; frontend 141 passed (17 new: 8 combos in video, 8 in clean frame mode, 1 older worker), tsc clean. Not deployed.

### 92b. "window is not defined" in sync.test.tsx (2026-10-26)
Cause: not the test or the Docker build. The `web` stage (`docker build --target web`) passes all 141 tests. The failure appears only when vitest is started from the repo root (the log's `frontend/src/...` path): there is no vite config there, so the environment is Node, not jsdom, and all 3 tests of the file fail (`window` / `document` missing). jsdom is installed only in `frontend/node_modules`, so a docblock cannot fix a root run. Run the tests from `frontend/` (`npm test`, `npx vitest run`). The test now reads `globalThis.location` (no ReferenceError by itself).

## 93 - Busy count lower than the visible cars (sub-bays / capacity)  (2026-10-26, not deployed)
- Replayed 3 vehicles in one pin + a 4th in its own slot with "Bay held" 50/80/95% and jittering boxes: all counted
  BUSY, so the item-91 gates are not the cause. What was missing was any way to see why a drawn vehicle is not counted.
- `FixedSlotTracker.unplaced` / `state()["unplaced"]` + `counts.unplaced`, `counts.sub_bays`: vehicles that hold no bay, with
  the reason (confidence under 0.45, same vehicle as one already counted, waiting n/3 steady frames).
- A second box of a sub-bay's own car no longer spawns another sub-bay (compared with every holder of the slot, not only the parent's).
- `counts.slots` stays the capacity incl. sub-bays (9 + 1 = 10, BUSY 4). A user-set `capacity` field on a pin is NOT built (asked).
- Tests: test_fixed_slots (3 shares, unplaced reasons, twin box), test_monitor counts. Nothing deployed.

## 94 - Manual capacity per slot / pin  (2026-10-26, not deployed)
- A slot or auto-grid pin can be given `capacity` N (1-20, default 1; only stored when > 1): `validate_slots`, `merge_pins`, the
  Slot editor and the Auto grid pins list (new "cap" field per row), JSON export/import.
- `FixedSlotTracker._count_loads`: the distinct vehicle boxes (footprint IoU < 0.3) whose ground is on the slot (ground_on >=
  min(Bay held, 0.5)) and held by no other slot; the most seen in the last 3 frames, capped at N. A capacity slot makes no auto sub-bays.
- `ParkingMonitor.state()`: counts are spaces: total = sum of capacities, BUSY = vehicles on BUSY slots (min 1, max N), FREE = the
  rest of those slots + FREE slots. Slots carry `capacity` and `vehicles`. Capacity 1 everywhere = as before.
- Tests: test_fixed_slots (validation, cap 4 / 2 with 3 vehicles, mixed total, pins), slotdraw.test.ts. Nothing deployed.

## 94b - Capacity count steadied  (2026-10-26, not deployed)
- Reviewed `_count_loads`: the count followed the max of the last 3 frames, a car half over the pin's line (< 50% of its footprint
  inside) was not counted, and two cars with overlapping boxes (footprint IoU >= 0.3) were merged into one, so a new car next to
  a stronger box could add nothing or lose a count.
- Now: on the slot = footprint middle inside it OR ground_on >= min(Bay held, 0.5); one vehicle = footprint IoU >= 0.5
  (CAPACITY_DISTINCT); steady = rises when seen in 2 of the last 5 frames, falls after 4 frames with fewer.
- `count_note` per slot (why a box was left out: "held by slot N", "same vehicle as a stronger one"), shown in the slot's flags.
- Tests: steady count, overlapping vs duplicate boxes, half-over car. Python suite subset + vitest 144 pass. Not deployed.

## 95 - History sessions follow a capacity slot's vehicles  (2026-10-26, not deployed)
- Cause: `history` keeps one session per slot (entry on BUSY, exit on FREE), so a capacity-4 pin with 3 vehicles showed 1 session
  while the dashboard counted 3 BUSY (5 BUSY on the page vs 3 sessions in the log).
- `HistoryDB.set_open(slot, target, t)`: a BUSY capacity slot has one open session per counted vehicle (extra ones are added with an
  entry event, the newest closed with an exit event); FREE / delete / retire close every open session of the slot.
  `ParkingMonitor.process` calls it each frame with `FixedSlotTracker.vehicles_in(k)`: the same number `state()` shows as BUSY.
- No spike: the count is the steady one (rises after 2 of 5 frames, falls after 4; the first frame after a start counts 1 at most),
  so a single-frame extra box opens no session. Tests: test_fixed_slots (sessions == busy through 1,2,3,2,2,0 vehicles).

## 96 - Open sessions synced to the live count on every frame  (2026-10-26, not deployed)
- Read-only check of the live data (2026-10-26): camera-1 scope "auto v10": P1 (slot 9, capacity 4) had 3 open sessions in the
  worker's SQLite and in PostgreSQL = the 3 vehicles counted, so item 95 worked for P1 (the 5-vs-3 was the old one-session-per-slot
  behaviour). Left over: slot 10 on camera-1 (7 open, 6 BUSY) and slots 3 and 5 on camera-2 (5 open, 3 BUSY) = sessions of slots that
  are FREE / not on the map, never closed; `set_open` also did nothing for a BUSY slot with no session (no backfill).
- `HistoryDB.sync_open(want, undecided, t)` replaces `set_open`: each frame (fixed maps / auto grids) the open sessions of a scope
  are made to match the tracker: BUSY slot -> one per counted vehicle (missing ones opened = backfill, extras closed newest first),
  FREE slot or a slot no longer on the map -> all closed, undecided (UNKNOWN) slots left alone. The normal entry/exit events still
  fire first with their own notes and times; this only repairs the difference. One cheap GROUP BY query per frame.
- Test: test_open_sessions_are_synced_... (stale sessions of a gone slot and an empty slot closed at start, count change closes one,
  deleted sessions are backfilled). Dynamic (learned) maps unchanged. Not deployed. The sync service copies new / closed rows by id.

## 97 - Third visible vehicle not counted in P1  (2026-10-26, not deployed)
- Read-only check of the live worker (2026-10-26, items 94-96 deployed): camera-1 had 3 vehicle boxes, BUSY 2. History was
  already in step (open sessions == BUSY on both cameras: 2 and 3), so the stale-session report was before item 96 took effect.
- The uncounted car (conf 0.86, box bottom 0.265) stood 0.010 of the frame below P1's lower edge (y ~0.241), in the gap above
  B7: ground_on 0.16, footprint middle outside P1, held by no slot (counted in `counts.unplaced`) -> not in P1's 1 counted.
- `_count_loads`: an unheld vehicle whose footprint middle is within CAPACITY_REACH 0.25 x its box height of the slot also counts,
  with a count_note "counted: stands n% of the frame outside its edge (draw the pin larger)". Here 0.010 vs a 0.0215 reach.
  A vehicle held by another slot is still that slot's. The pin itself can also be redrawn larger. History follows (item 96).

## 98 - History page lag  (2026-10-26, not deployed)
- Checked the PostgreSQL sync read-only: open sessions PG == SQLite (35 == 35, all scopes), camera-1's current scope 3 open (slot 7,
  and P1's 2 sessions at that moment), history_sync lag 1 s, sync every 2 s, no error. The sync inserts, updates and closes rows
  correctly (new rows by id, closes copied for sessions open in PG), so nothing to change there.
- The History page polled `/api/sessions/page` and `/api/events` every 15 s -> now 3 s (Pages.tsx); the table follows a count
  change within ~5 s (frame -> SQLite -> sync <= 2 s -> poll <= 3 s). A browser still holding the old bundle needs a hard refresh after a rebuild.
- Noticed: P1's extra sessions open and close every 1-3 minutes (424-430), i.e. the third vehicle's detection comes and goes;
  item 97 (the car just outside P1's edge) should make it steadier.

## 99 - History page read the wrong scope  (2026-10-26, not deployed)
- Cause (live check, 2026-10-26): the worker writes camera-1's history to the scope "Camera 1 | yolov8s | auto v10" (Worker._auto_run
  adds " | auto vN"), but the API read it by the stored settings' `db_detector` ("Camera 1 | yolov8s"): `/api/sessions/page`
  returned 268 rows ending 25 October with 3 old open sessions, while the real scope had 431 rows and the live sessions.
  PostgreSQL and the sync were right (PG == SQLite); the page asked for the wrong scope, so no refresh could help. Camera-2 (fixed) was fine.
- `live_detector(run, snapshot)` (api/main.py): the detector is taken from the worker's snapshot scope ("<camera> | <detector>")
  when it belongs to this camera, else the settings'. Used by `history_db`, `scope_of` (PostgreSQL + analytics) and the WebSocket's
  new-event feed. Test: test_api (page follows the auto-v10 scope; another camera's scope ignored).
- Item 98's 3 s poll stays. Not deployed (api container only needs a rebuild).

## 100 - Fixed maps are strict  (2026-10-27, not deployed)
- `live.py:_auto_run`: sub-bays (`split`) only when the run is the auto grid (`auto is not None`); `Worker.split` holds the effective
  value and the "monitor rebuild on option change" check compares `(slot_split and auto) != split`, so on a drawn Fixed map the
  `slot_split` option neither makes sub-bays nor rebuilds the monitor.
- `FixedSlotTracker.restore()` returns the ids it skipped: a Fixed map (no auto grid, no sub-bays) does not restore vehicle-driven
  bays; `ParkingMonitor.start()` always calls it and retires those ids (`HistoryDB.retire_slots`: closes their open sessions, marks
  the saved bay retired, note "a Fixed map has no sub-bays"). Camera-2's saved sub-bay 1001 goes this way on the next worker start.
- `_count_loads`: the grace margin (CAPACITY_REACH) only when `self.auto is not None`; a Fixed capacity slot counts strictly inside.
- Fixed trackers fill `unplaced` too (`_unplaced_fixed`): a vehicle no slot holds or counts -> "no free slot (Fixed map: capacity N)"
  (it stands on a slot) or "in no slot of the Fixed map". On the auto grid a vehicle a capacity pin counted is removed from `unplaced`.
- Tests: test_fixed_slots (Fixed: 2 slots + 3 vehicles -> 2 BUSY, reason; strict vs margin; restore skip + monitor retire),
  test_live (Fixed never splits and is not rebuilt; the grid toggles with the option). Python 389 passed, 23 skipped. Not deployed.
- Known, left alone: an auto-grid vehicle counted by a capacity pin could still spawn a vehicle-driven bay when it stands on the
  measured surface and overlaps no bay (not the case on camera-1 today: no D bay was ever made for it).

## 101 - Pin grace: the capacity pin's margin as a live setting  (2026-10-27, not deployed)
- `history.DISPLAY_TUNING["pin_reach"] = (0.25, 0.0, 0.5)` (the old `CAPACITY_REACH` constant is gone); `SlotTracker.pin_reach`;
  `FixedSlotTracker._count_loads` uses `self.pin_reach` (still auto grid only; 0 = strictly inside; a Fixed map ignores it).
  It travels the `bay_share` path: `SettingsPatch.pin_reach` (range from DISPLAY_TUNING), stored in the worker option "tuning",
  applied by `Worker._tune` in place, in `tuning_defaults` and `ranges`. Default 0.25: no behaviour change.
- Settings drawer (Counting thresholds): "Pin grace: vehicle height past a capacity pin's edge", 0-50 %, hint "Auto grid only ...
  A Fixed map is always strict"; the Reset button covers 5 values. `types.ts`: `SettingsValues.pin_reach`, tuning_defaults key.
- Tests: test_history (tuning_values, clamp, 0 kept), test_api (defaults, ranges, stored, 422 over 0.5, 0 accepted), test_live (applied
  in place), test_fixed_slots (near / mid / far car at reach default, 0 and 0.5; a Fixed tracker ignores it), new
  settingsdrawer.test.tsx (field + hint + range, Apply sends only {pin_reach, version}, Reset). Python 390 passed, 23 skipped;
  vitest 146 passed; tsc clean. Not deployed.

## 102 - History across grid versions, no stale cache  (2026-10-27, not deployed)
- Cache: every history / analytics answer carries `Cache-Control: no-store` (`json_from`, analytics, overview, history/sync, the two CSVs).
- `HistoryDB.close_open(note, t)` (this scope), `stale_open(only)` / `close_stale(note, t, dry_run, only)` (this camera's OTHER scopes;
  an exit event under each session's own scope; never deletes; a camera whose name merely starts the same is not ours).
- `Worker._close_older_grids`: entering the auto grid scope "<detector> | auto vN" closes the open sessions of the same camera's
  and detector's older versions with the note "grid replaced by vN" (newer versions, other detectors and the learned map are left alone).
- `POST /api/commands {"action": "close_stale", "dry_run": true|false}` (admin): counts, then closes, the open sessions of every scope
  but the one in use; answers `{done, dry_run, sessions, scopes}`; independent of the worker (it only writes its own scope).
- `GET /api/sessions/page?all=1`: every scope of the camera, each item with `scope` (HistoryDB, SqliteHistory and PgHistory.sessions_page(all_scopes)).
- History page: "This grid / All versions" pills (page back to 1), a Scope column in All versions, Refresh = `usePoll.reload`
  (title "Refresh"); `types.Session.scope`.
- Tests: test_history (close/stale/dry run/events/other camera), test_history_pg (SQLite and, run against a scratch PostgreSQL
  container, PgHistory all-scopes == HistoryDB), test_api (no-store on 11 endpoints, ?all=1, close_stale 401 / dry run / apply), test_live (v2 closes
  v1 only), history.test.tsx (toggle + Scope column, Refresh refetches). Python 396 passed, 23 skipped (the PostgreSQL variants ran
  separately: 8 passed); vitest 148 passed; tsc clean. Not deployed. Today's 35 orphan open sessions are NOT touched: run close_stale (dry run first) on the user's go.

## 103 - Mismatch visibility, no second loop  (2026-10-27, not deployed)
- No extra thread: the check runs inside the frame the worker already processes. `ParkingMonitor._check_sessions` compares, every
  frame and for every map type, the vehicles the tracker counts BUSY (a capacity slot counts its vehicles) with the open sessions of the
  decided slots (UNKNOWN slots keep theirs and are left out): `session_check = {expected, open, since, at}`; `HistoryDB.open_by_slot()` (one GROUP BY).
- A failed repair is no longer fatal: `sync_open` raising `sqlite3.Error` (a locked / read-only file) is caught, kept in
  `session_error`, and the frame goes on (before, it would have stopped the session). It repairs itself in the next frame that can write.
- Fixed maps / auto grids are repaired every frame (item 96); learned maps are only reported (their merges and retirements are their own).
- Snapshot `sessions = {expected, open, mismatch_s, error}`; `Worker._watch_sessions` logs ONE line when it has lasted 30 s
  ("history differs from the live count ...") and one when it is equal again; `/api/history/sync` cameras[].live (None without a worker).
- History page: a warning bar (role=alert) with the counts, duration and last error from 30 s on (`MISMATCH_WARN_S`).
- Tests: test_history (open_by_slot), test_monitor (learned map: reported, not repaired), test_fixed_slots (failed repair visible,
  clears; UNKNOWN slots not a mismatch), test_live (snapshot, one log line, cleared line), test_api (history/sync live), history.test.tsx
  (warns >= 30 s with the error; quiet for 12 s, equal, no worker). Not deployed.

## 104 - Offline tracking evaluation  (2026-10-27, not deployed; nothing live touched)
- `src/carparking/track_eval.py` + `scripts/track_eval.py` (`selftest`, `record`, `eval`). Read only: it replays a recording, it never
  writes to the system, ./data or the history. `record` samples the running worker's own merged detections through GET /api/status at
  its real, uneven rate (token from an env var, sent as a header only, never printed); `eval` replays a JSON recording.
- Trackers: `ByteTracker` (ByteTrack in numpy: Kalman, two-stage association, the weak boxes continue a track, buffer 30, a track is
  confirmed by a second sighting), `IouTracker` (baseline), `UltralyticsByteTracker` (the real one when `lap` + `scipy` exist; they are
  NOT in the runtime image and nothing is installed from here: YOLO_AUTOINSTALL is switched off, it reports "not available here").
  Verified once in a throwaway container with lap + scipy: same tracks / ID switches as my numpy ByteTrack, held-count error 0.056 vs 0.054.
- Metrics: detection precision / recall; MOTA, ID switches and per vehicle-hour, fragmentations, gaps in the detections bridged /
  switched / lost; per slot, the count from raw boxes, from the worker's steady rule, from the tracks seen this frame and from the tracks
  HELD (also the ones unseen for a few frames: what a count by track IDs would use): error against the truth, share exactly right,
  count changes per hour (flicker), track ids per vehicle. The plan's decision rule is coded (`decide`): ids < 1 switch per vehicle-hour
  AND held-track flicker and error no worse than the steady count; without ground truth it says "undetermined".
- Synthetic scene (`synthesize`): a capacity-3 pin + a bay, uneven frame times, jitter, misses, a 5-frame dropout, weak boxes, twins, strays.
  Result, 20 min: ByteTrack 0 id switches, held count 15 changes/h (steady 123, raw 1354) but error 0.054 vs steady 0.039 -> "keep the
  steady count". A synthetic scene only proves the harness; the real answer needs a real recording of camera-1 (`record` ~30 min) - and
  hand-labelled ground truth for ID switches (without it: ids per vehicle only).
- Tests: tests/test_track_eval.py (25: IoU, Hungarian vs brute force, Kalman round trip, every tracker behaviour, metrics with scripted
  trackers, deterministic synthetic scene, recording round trip, record_from_api, CLI, token not printed, adapter unavailable path;
  1 skipped without lap + scipy). Python 426 passed, 24 skipped (the 23 PostgreSQL variants + the Ultralytics test). No live change.

## 105 - Stream health: jittered reconnects, frozen picture, UNKNOWN while the stream is bad  (2026-10-27, not deployed)
- `src/carparking/stream_health.py` (new). `FreezeDetector`: a 64x36 grayscale thumbnail of every new frame, compared with an ANCHOR (the
  first thumbnail of the quiet stretch), not with the previous frame; the anchor moves when a frame differs by >= `FREEZE_DIFF` (0.5 grey
  levels, mean absolute difference); the same picture for `FREEZE_S` (20 s) = frozen. Anchored because a live quiet scene drifts away
  from its anchor (exposure, shadows) even when no two consecutive frames differ much; a repeated frame never does. Measured on the
  saved frames in data/dataset_raw: real pairs 2-12 s apart differ by 0.8-1.5 grey levels, so 0.5 is above a repeat and below a live scene
  there, but consecutive VIDEO frames of a still night were not measured: the snapshot's stream.diff shows the live value to check it
  on camera-1 first (CARPARK_FREEZE_DIFF, CARPARK_FREEZE_S; 0 = off). `StreamHealth`: active / stale (no new frame for `STALE_S` = 10 s,
  CARPARK_STALE_S) / frozen; a gap longer than that drops the anchor (a still night scene is not "frozen" the moment the stream returns).
- `rtsp.py` (A1): the reader already kept only the newest frame, over TCP, with open / read timeouts and a backoff; what was missing
  is jitter: every wait is now backoff + up to half of it at random (`LatestFrameReader.delay`), also the wait after a dropped stream (was
  a fixed 1 s), so cameras that dropped together do not reconnect in lockstep.
- `monitor.py` (A3): `ParkingMonitor.stream`; while it is not "active", `state()` shows every slot UNKNOWN (counts: unknown = all spaces,
  busy = free = 0, note "stream stale / frozen"). The tracker and the history are not touched.
- `live.py` (A3): `Worker` takes a `health_factory` (scripts/worker.py: from the environment). A stale or frozen stream is not processed
  (no detector call, no raw frame saved, no sync_open), so the open sessions stay open exactly as they were; the heartbeat snapshot then
  carries the slots (UNKNOWN), no detections and `stream`; every processed frame carries `stream` too. One log line when it turns bad,
  one when it is live again.
- Frontend: `Snapshot.stream`; a warning banner over the video ("Camera stream frozen: the same picture for 25 s. Slots are shown UNKNOWN
  and open sessions are kept until it is live again.").
- NOT done, on purpose: A4 as written (an affine fitted by `estimateAffinePartial2D`, applied to the slot polygons on its own for a 1-4 %
  shift): `geometry/drift.py` documents a fitted transform that claimed 27 px of drift for frames whose matched points had moved 0.1 px,
  and it already measures the displacement of the matched points, marks the calibration stale after 3 checks in a row and falls back; and a
  silent rewrite of an operator's Fixed map is not wanted. A5 (GPU batching service) is an architecture blueprint, not code.
- Tests: tests/test_stream_health.py (11: thumbnail, 20 s freeze, slow drift never frozen, scene change / off switch, stale and recovery,
  gap drops the anchor, env thresholds, jittered waits differ per camera, worker: stale and frozen show UNKNOWN, no processing, sessions
  open, one log line each way, recovery); vitest banner (152 passed). tests/test_live.py's fake camera repeats one picture for ever, so its
  workers run with the freeze check off. No live change; not deployed.
- Build fix (105): two heavy app-level vitest tests (src/app.test.tsx, "survive many state messages ..." and "come back after the page is
  opened again") timed out at the default 5 s in the container build. They are unrelated to the stream-health code: they take 4.2 s and
  2.0 s with the container limited to 0.5 CPU. `frontend/vite.config.ts` now sets `test.testTimeout: 15000` (the rest of the config is kept).

## 106 - Heartbeat from a thread of its own  (2026-10-27, not deployed)
- Cause: the worker's heartbeat was written only by `step()` (`beat` / `publish`), between frames. On a CPU without a GPU one inference can
  take longer than `STALE` (15 s), so the page said "The worker is down (no heartbeat for N s)" while the worker was busy working, and
  `claim()` (the same heartbeat, `LEASE` 20 s) let a second worker take the lease.
- `live.py`: `LiveStore.pulse(owner, now)` (heartbeat only, only for the lease holder; snapshot and images untouched) and `Pulse`, a daemon
  thread with its own store connection that calls it every `PULSE_EVERY` (3 s, CARPARK_PULSE_S) while the worker holds the lease. The loop
  stamps `Pulse.progress` at the start and end of every step; after `PULSE_STALL` (180 s) without a stamp the pulse stops, so a HUNG worker
  still shows as down while a slow one does not. Stopped in `loop()`'s finally before `release()`. `Worker(pulse_every=...)`, 0 = off (tests);
  scripts/worker.py turns it on. The timeouts themselves are unchanged.
- Visible effect: during a long frame the page shows the worker as running with an ageing frame (`frame_age`), instead of "down".
- Tests (test_live.py): pulse touches only the holder's heartbeat and leaves the snapshot; inactive / stalled / lost-lease ticks write nothing;
  a worker blocked in the detector keeps a moving heartbeat (real thread); the pulse follows the lease and stops with the loop.

## 107 - Task 2 step 0: local snapshot + rollback tooling (no git)  (2026-10-28, tooling only; no running service changed)
- `scripts/snapshot.ps1 [-Label] [-Cold] [-SaveImages] [-Dest]` (default `C:\F\Backups\CarParking\<stamp>-<label>\`, outside the project so
  a mirror-restore cannot delete it and the Docker build context never sees it): `code\` (robocopy of everything except data, backups,
  node_modules, dist, __pycache__; includes `.env`, so keep the folder private), `data\` (plain files by robocopy; every SQLite DB through
  SQLite's online backup API in a one-off `docker compose run --no-deps worker` container, so a running worker is fine; `-Cold` stops
  worker + sync for the copy), `pg\carparking.dump` (`pg_dump -Fc`, copied out with `docker compose cp`), image tags `:snap-<stamp>` of
  the app and setup images (`-SaveImages` also writes tars, ~6 GB), `manifest.json` (sha256 of every file, image ids, compose state).
  Self-check: files re-hashed, `PRAGMA integrity_check` on each DB copy, `pg_restore --list`, code mirror compared with the source.
- `scripts/db_backup.py` (+ `tests/test_db_backup.py`, 4 pass): the online-backup copier; source opened read-only.
- `scripts/restore.ps1 -Snapshot <dir> [-Code] [-Images] [-Data] [-Pg] [-Force] [-WhatIf]`: default = code + images (history since is
  kept); `-Data` / `-Pg` need `-Force`. Verifies the snapshot's hashes, takes a `pre-restore` snapshot, stops worker/sync/ui/api, mirrors
  code (files added since are removed), mirrors data, `pg_restore --clean --single-transaction`, re-tags the images, `up -d --no-build`
  (setup too if it was running), waits for healthy + /api/health.
- Taken: `C:\F\Backups\CarParking\20261028-094007-task2-pre` (428 files, code 323 MB, data 52 MB, pg dump 0.4 MB, both SQLite copies
  `ok`, image tags snap-20261028-094007 = ids dc22c2e3961d / 02a8bdc17b7c). Rehearsed: `restore.ps1 -WhatIf` (also flags files changed
  since); the dump restored with exit 0 into a throwaway postgres:17.7 (12 tables, counts equal the live ones apart from minutes of new
  rows). Not rehearsed: a full live restore (it would stop the stack).
- Additive-only rule for Task 2: no SQL migrations; new data lives inside the geometry `record` JSON and as extra pin keys.
  `validate_slots` copies known keys only, so older code just drops an unknown pin key such as `slot_id`: a code-only rollback is safe.

## 108 - Task 2: Suggest slots (AI draft layer, accept / discard)  (2026-10-28, built and tested, NOT deployed)
Decisions (user): Accept depends on the camera's mode (auto grid -> a new grid version, Fixed -> merge into the drawn map); the
button has a Quick default and a Full (SAM + GeoCalib) option; P1 is a persisted manual anchor with a stable slot id bound to the
nearest generated bay. Snapshot to roll back to: `C:\F\Backups\CarParking\20261028-094007-task2-pre` (item 107).
- **Engine** `src/carparking/slot_suggest.py` (numpy only): `suggest_slots` = the ACTIVE HIGH calibration's surface re-tiled around the
  whole vehicles on a fresh still (`ActiveGeometry.retile(anchors, expected)`, the old `measured()` split into `_tile(anchors)`; with
  fewer than 2 usable vehicles the calibration's own cars line the rows up), `carry_ids` (a suggested bay over a running one by IoU >= 0.5
  keeps its id and name, the rest get fresh ids above everything and never a pin's), `diff` (added / removed / moved / unchanged),
  `baseline_hash`, `bind_pin_ids`, `merge_into_fixed`. `GeometryFeed.slots_from_tiles` = the old `_grid_slots` loop, shared.
- **P1 / pins**: pins get an optional `slot_id` (`validate_slots`, 1..999). `merge_pins`: a pin with a `slot_id` keeps it whatever the grid
  does (a bay that holds that id steps aside; a pin without one behaves exactly as before) and every pin carries `binds_to` = the id of the
  nearest bay of the grid (a label, nothing is renumbered). On the first Accept `bind_pin_ids` writes the id each pin has on the running
  grid (P1 keeps its history id, capacity 4 and open sessions); capacity logic is untouched.
- **Drafts** are child geometry versions of the active one: `record.grid` (the bays) + `record.suggest` {parent, mode, kind auto|fixed,
  lined_up, expected, still, baseline hash}. The worker reads the active version only, so a draft changes nothing live. An accepted draft
  is a normal activation: `ActiveGeometry.frozen` makes `_grid_slots` return the stored bays (the expected count was applied at suggestion
  time). `GeometryStore.update_draft` (new; drafts only, no credentials), draft / activated events carry `parent` + `suggest` mode.
- **API** (admin + CARPARK_GEOMETRY_ENABLED, `api/geometry_routes.py`): `POST /api/cameras/{cam}/suggest {mode, still?, expected?, clicks?}`
  -> job (409 while one runs); `GET .../suggest` (pending drafts), `GET|PUT|DELETE .../suggest/{ver}` (view with baseline + diff + pins +
  `stale`; fine-tune: moved bays marked `edited`, a bay may not take a pin's id; discard), `POST .../suggest/{ver}/accept {select?}`.
  Refused (409): no HIGH active calibration, drift guard says `drift`/stale, the active calibration or the running slots (hash) changed
  since, the camera changed mode, the camera moved again (`camera_moved_since`). Fixed accept: chosen bays merged through `save_fixed`
  (factored out of `PUT /api/fixed-slots`, so `update_fixed` keeps the state of unchanged slots), draft consumed, event `applied_to_fixed`.
  Full mode: `build_record(height_hint=)` pins the metres to the active calibration's camera height (needs a typed-reference calibration;
  it stays HIGH), surface from SAM on new / stored clicks else the stored polygon, GeoCalib + depth; warns when the pitch differs > 3 deg.
  `calibration_record` and `surface_candidates` were extracted from the calibrate / segment jobs (behaviour unchanged, their 21 tests pass).
  `docker-compose.yml`: `setup` gets `cpus: "2"` so a Full run cannot starve the workers (takes effect when the setup service is recreated).
- **UI** `components/SuggestSlots.tsx` (+ `suggest.ts`), under the slot editor: "Auto-detect / suggest slots" (Quick | Full, expected
  count) -> progress -> the draft over the editor's snapshot. Layers (remembered): Compare (draft coloured added / moved / edited, removed
  running bays red dashed), Both, Draft, Running; pins in purple with "slot N · next to Bx". Drag a bay or its corners, remove a bay,
  Save draft / Revert, Test draft (the existing preview on the snapshot), Accept (inline confirmation saying what changes; saves unsaved
  edits first; Fixed: tick the bays to take), Discard. Hidden when the geometry flag is off; asks for the admin token when missing.
- **Tests**: Python 467 passed / 26 skipped (was 426 / 24): `test_slot_suggest` (12), `test_suggest_api` (9), store (update_draft,
  events; PostgreSQL variants run on a throwaway postgres:17.7: 30 passed), `test_db_backup` (4); pyflakes clean. Frontend 172 passed
  (was 152), `tsc` clean.
- **Not done / to know**: not deployed (needs `docker compose up -d --build`, then recreate `setup` for the CPU cap). Not tried on the live
  camera. Accepting on camera-1 changes the history scope (`auto vN`), so each parked vehicle shows one exit + entry at the switch (item
  102 closes the old scope's sessions). The pin's `binds_to` uses image-space distance, not metres. A suggestion shows up in the Camera
  setup versions table too (it carries `suggest`); activating it there skips the pin binding, use Accept in the slot editor.

## 109 - Camera 1 back to the 12-space baseline for the demo  (2026-10-28 10:3x, live change, user request)
- Finding (read-only): Camera 1 was running the calibration auto grid (v10: 8 measured bays + pin P1 capacity 4 = 12 spaces) PLUS vehicle-driven
  and sub-bays that appeared on their own: D12, B7.2, B6.2 -> 15 slots, and every extra bay could add a BUSY (6-7 busy for 5 cars). There is
  NO saved 12-slot manual map anywhere (registry, data/config, backups, the task2-pre snapshot): the saved Camera 1 Fixed map is S1-S5
  (5 slots), Camera 2's is S1-S2. So "the 12 slots" = the auto grid's 12 spaces, and a switch to the saved Fixed map would have shown 5.
- Change (live, reversible, no restart, camera-1 only): worker options in `/data/parking_history.db` -> `live_options`: `auto_dynamic = 0`
  (no vehicle-driven bays) and `slot_split = 0` (no sub-bays). Both were unset before (= on). They are stored in the database, so they
  survive worker / stack restarts and rebuilds. Undo: set both to `1` (LiveStore.set_option). Nothing else touched: calibration v10, pins,
  settings (slot_mode), camera-2, history scope `Camera 1 | yolov8s | auto v10`, sessions.
- Result 10:37 local: counts slots 12 (dynamic 0, sub 0), BUSY 4 (B2 sedan, B4, B6, P1 pickup), FREE 2 (B1, B8), UNKNOWN 3 (B3, B5, B7:
  bays hidden behind the cars in front, shown UNKNOWN not FREE), the motorcycle is outside the grid (counts.unplaced 1). The 5 open
  sessions are slots 2, 4, 6, 7, 9; the spawned bays' sessions are gone.
- Task 2 (Suggest slots): the stack was rebuilt by the user at ~10:2x (new image 52da4e47...), so the Task 2 code (and items 93-106) IS in the
  running containers now, dormant: no suggestion draft exists, calibration v10 is still active, pins unchanged, nothing applies a draft
  without an admin pressing Accept. The `setup` container is still the old one (no CPU cap yet).
- Not done: a real Fixed-mode lock (it would drop to the 5-slot S1-S5 map); the dashboard HUD "TOTAL: 9" counts slots, not P1's capacity
  (the cards say 12).

## 110 - Camera drift roadmap  (2026-10-28, planning only, no code or live change)
- wrote `docs/camera_drift_roadmap.md` (camera drift and auto-correction: what exists,
  gaps, workflow, guardrails, lighting/weather handling, 5-step build order). To be implemented step by step after the 2026-10-29 demo;
  step 1 is "freeze instead of fallback + drift banner".

## 111 - Occlusion tracking roadmap  (2026-10-28, planning only, no code or live change)
- wrote `docs/occlusion_tracking_roadmap.md` (occlusion tracking and hidden-object
  management: static occluder masks, visibility-aware overlap, evidence filter + hysteresis, ByteTrack persistence, appearance-signature
  session reopen, build order 0-5, verification rules). To be implemented step by step after the 2026-10-29 demo.

## 112 - Settings panel wording  (2026-10-28, display strings only; prepared, NOT built or deployed; the running stack is untouched)
- `frontend/src/components/SettingsDrawer.tsx`: labels, helper texts and tooltips set to the user's exact wording for Monitoring, Stream
  Selection, AI Detection Model, Detection Confidence, Entry Delay, Exit Verification Buffer (seconds and frames), Covered Slot Memory
  (coverage and seconds), Single Car Multi-Slot Protection, Slot Alignment Tolerance, Pin Boundary Grace Margin, Parking Zone Crop.
  `ui.tsx`: `Field` got an optional `tip` prop (becomes the `title` tooltip). No state keys, defaults, ranges or logic changed.
  `settingsdrawer.test.tsx`: the label and hint strings it looked for were updated. tsc clean, vitest 172 passed. Shows up after the next
  rebuild (after the 2026-10-29 demo).

## 113 - Night Vision & Low-Light roadmap  (2026-10-28, planning only, no code or live change)
- wrote `docs/night_vision_roadmap.md` (Night Vision & Low-Light pipeline, kept
  separate from occlusion tracking: light probe, CLAHE on the detection copy only, dynamic confidence with a zoom_free_conf floor of
  0.35, lighting-drop grace in the hysteresis, code hooks, build order 0-5). Noted that the default confidence is already 0.30 (not
  0.50) and the existing hold is 15 s / exit 30 s. Cross-link sections appended to the occlusion and drift roadmaps.

## Session end 2026-10-28: state for the next session
- **Live:** stack as rebuilt by the user ~10:2x (items 93-108 in the containers; Suggest slots dormant, no drafts). Camera 1 runs the
  calibration v10 auto grid with `auto_dynamic=0`, `slot_split=0` (item 109): 12 spaces (8 bays + pin P1 capacity 4), no spawned bays.
- **In source, not built:** item 112 (Settings wording). Appears after the next `docker compose up -d --build`.
- **Plans only (docs/):** `camera_drift_roadmap.md` (110), `occlusion_tracking_roadmap.md` (111), `night_vision_roadmap.md` (113),
  cross-linked. Each starts with a read-only measurement step and puts every live change behind an option that is off by default.
- **Rollback point:** `C:\F\Backups\CarParking\20261028-094007-task2-pre` (`scripts\restore.ps1 -Snapshot <dir>`; default code + images).
- **Rules until after the 2026-10-29 demo:** no deploy, no live change. Undo item 109 by setting camera-1 `auto_dynamic` / `slot_split` to `1`.
- **Open after the demo:** decide on a real Fixed 12-slot map for Camera 1 or keep the auto grid; rebuild for item 112; recreate `setup`
  for its CPU cap; first Suggest-slots Accept trial on camera-2; then the roadmaps, step 0 first (drift step 1 = freeze instead of fallback).

## 114 - Post-demo plan, step 1: ghost-transition layer wired, OFF by default  (2026-10-29, source only; NOT built or deployed, the running stack is untouched)
- wrote `docs/post_demo_engineering_plan.md` (the approved spec for the four evaluator issues: multi-slot spillage, non-vehicle obstacles
  / BLOCKED, distant-bay extrusion, stationary ghost transitions; build order 0-4) with a "Step 1 as built" section; cross-links added to
  `docs/occlusion_tracking_roadmap.md` 4.3 and 4.5.
- snapshot first: `C:\F\Backups\CarParking\20261029-181032-pd-step1-pre-wire` (OK: 444 files, SQLite + pg_dump checked, 2 image tags;
  restore with `scripts\restore.ps1`).
- new modules: `src/carparking/signature.py` (HSV 16x4x4 + 32x16 grey thumbnail, Bhattacharyya / NCC / thumb_diff, `same_vehicle`) and
  `src/carparking/evidence.py` (asymmetric EMA `tau_enter` 3 s / `tau_exit` 30 s / `alpha_max` 0.5 with a 0.20-0.55 band, parked lock
  IoU >= 0.7 for >= 60 s and >= 3 frames, `exit_veto` = relook / appearance / no_motion, `gate_exit` with a 30 s cooldown).
- wiring: `history.py` `SlotTracker.update` got one guarded hook (`if self.evidence_filter:` -> `_evidence_obs`, default identity);
  the logic is in `fixed_slots.py` `FixedSlotTracker` (`evidence_filter` kwarg, default 0); `monitor.py` hands it the frame and a
  re-look (`_relook`, `RELOOK_CONF` 0.15) only when on; `live.py` `Worker._evidence` follows the live option `evidence_filter`
  ("0" default, applied to the running tracker every frame, no monitor rebuild).
- found while testing: a lighting dip (x0.7) moved a car across the 4 coarse V bins (Bhattacharyya 0.79 = "another car"), which would
  have switched the appearance veto off exactly when it is needed; V is now scaled to a fixed mean before the histogram.
- results: `docker build --target test` before wiring 502 passed, 26 skipped; after wiring and the new tests **522 passed, 26 skipped,
  pyflakes clean** (Streamlit parity test included). Off-path identity: 16 randomised runs x 600 frames (plain fixed, capacity pin,
  auto grid, auto grid + sub-bays) through the snapshot's code and the new code give the same SHA-256 over events, slot states, shown
  states, loads and unplaced; with the layer on the hash differs. Stationary-car scenario (7 s dropouts every 60 s + a 75 s 55 %
  exposure dip with the detector losing the car): 0 exits with the layer, >= 1 without; mutation checks (veto / EMA / lock disabled) make
  the tests fail.
- measured trade-off: a real exit is reported ~48 s after the car leaves (155 -> 198 in the test scene) but logged at the departure time
  (existing `gone` / `expired` backdating). Veto thresholds (`motion_ratio`, `motion_floor`, `max_bhatta`, `min_ncc`) are set on
  synthetic scenes and need the step-0 camera-1 recording before the option is switched on anywhere real.
- not done: L1 continuation confidence, L4 micro-movement compensation, L5 session reopen, per-stage `timing`, API / UI switch;
  items 1-3 of the plan. Switch (when the time comes): `store.set_option("evidence_filter", "1")` on camera 2 / a test camera first.
- Streamlit (`app.py`, `scripts/demo_app.py`) untouched; `./data` untouched; no `docker compose up`.

## 115 - Post-demo plan, step 2 tooling: pixel recorder, flicker report, offline replay  (2026-10-30, source only; NOT built or deployed, the running stack is untouched)
- why: the veto thresholds (`motion_ratio`, `motion_floor`, `max_bhatta`, `min_ncc`) are computed from pixels; `track_eval.py record` keeps only
  boxes, and no camera-1 footage exists in the project. So they cannot be tuned yet. This item builds the tooling only; **no threshold was
  changed or chosen** (EvidenceCfg is untouched).
- new: `src/carparking/pixel_replay.py` (recorder, reader, `replay()` through `FixedSlotTracker` with evidence_filter 0 / 1 and EvidenceCfg
  overrides), `src/carparking/flicker.py` (ghost pairs, short states, changes per slot-hour, report lag, exit scoring vs an optional truth
  file), `scripts/pixel_record.py` (`record` / `info`), `scripts/flicker_report.py`, `scripts/replay_eval.py` (side-by-side live | off | on,
  `--sweep`, `--truth`), `tests/test_pixel_replay.py` (17 tests, fake worker), "Step 2 tooling" section in `docs/post_demo_engineering_plan.md`.
- recorder: read only (GET /api/status, /api/settings, /api/frame.jpg?raw=true); writes `recordings/<camera>_pixels/<timestamp>/`, refuses
  any path inside `./data`; a frame is kept only when the status before and after the picture names the same frame_ts. `.dockerignore`
  already excludes `recordings/` from the build context.
- results: `docker build --target test` **539 passed, 26 skipped** (522 + 17 new), pyflakes clean. In the new tests a parked car with
  dropouts and an exposure dip shows >= 1 ghost pair in the replay with the layer off and 0 with it on; a real exit is still found with a
  measured lag; the sweep, truth scoring, report lag, truncated-recording and refuse-`./data` cases are covered. Existing code untouched:
  no change to `fixed_slots.py`, `history.py`, `monitor.py` or `live.py`, so the off-path identity of item 114 stands.
- not done / next: the recorder has not been run against the real stack (it needs the viewer token in `CARPARK_VIEW_TOKEN`); no camera-1
  footage yet. Next: `python scripts/pixel_record.py record --cam camera-1 --minutes 600` across a change of light, list the real
  departures in a truth file, sweep with `scripts/replay_eval.py`, then set the values. The replay cannot use the worker's zoom re-look.
- Streamlit (`app.py`, `scripts/demo_app.py`) untouched; `./data` untouched; no `docker compose up`.

## 116 - Camera-1 pixel recording: first real run, sweep on it, thresholds NOT changed  (2026-10-30, source + recordings/ only; running stack untouched)
- real run: `scripts/pixel_record.py record --cam camera-1 --minutes 1` against the live API (viewer token read from `.env` into the process
  environment only, sent as a header, never printed or saved): 9 frames over 63 s -> `recordings/camera1_pixels/test-1min/` (header with the
  9-slot map B1-B8 + pin P1 capacity 4 and the worker's settings; `frames.jsonl` 9 rows, timestamps strictly increasing; 9 JPEGs, 4 MB).
  Only GET requests; nothing written outside `recordings/`; `./data` not opened by the scripts (the live stack keeps writing it as usual).
- finding: the camera-1 worker publishes ~1 frame per 7 s (9 frames/min), so a 6 h recording is ~3000 frames, ~1.4 GB. The evidence layer's
  60 s parked lock then spans only ~9 frames: worth remembering when reading the lock and EMA numbers.
- baseline (`flicker_report.py --source all`): live / replay off / replay on = 0 state changes, 0 exits, 0 entries, 0 ghost pairs in all three.
  Replay off agrees with live. Slots B1, B3, B4, B7, P1 BUSY; B2, B5, B6, B8 FREE throughout.
- sweep (`replay_eval.py --sweep` over motion_ratio 1.5/2/3 x motion_floor 2/3/5 x max_bhatta 0.25/0.35/0.45 x min_ncc 0.5/0.6/0.7 = 81 runs):
  every combination identical: 0 ghost pairs, 0 exits, 0 vetoes, no lag. Nothing in 63 s of footage dropped out or left, so the footage
  cannot tell the settings apart. **No threshold was selected or changed**: `evidence.py` is as in item 114 (motion_ratio 2.0,
  motion_floor 3.0, max_bhatta 0.35, min_ncc 0.6, still synthetic-scene values). A value picked from this would be arbitrary.
- recorder hardening found by the first run (a container's route to the host timed out mid-run and killed the recording): `record_pixels` now
  counts and retries failed requests (`max_errors` 30 in a row ends the run with the error, closing what was recorded); the recorder moved to
  `src/carparking/pixel_recorder.py` (stdlib only, so it runs on the host Python, which has no OpenCV); `pixel_replay.py` keeps the replay.
  2 new tests (retry / dead stack, no-OpenCV import + DYNAMIC_BASE kept equal to fixed_slots).
- tests: `docker build --target test` **541 passed, 26 skipped**, pyflakes clean (539 + 2).
- started: a bounded read-only recording `recordings/camera1_pixels/long-<timestamp>/` (360 min or 3 GB), running from this session's shell.
- next: when it has finished (or whenever there is a stretch with a real departure and a change of light), write the departures into a
  truth file and run `scripts/replay_eval.py DIR --truth truth.json --sweep ...`; only then set the values.

## 117 - Post-demo plan item 1: spillage / multi-slot encroachment, wired, OFF by default  (2026-10-30, source only; NOT built or deployed, the running stack is untouched)
- new: `src/carparking/spillage.py` (`SpillCfg` share 0.30 / total 0.45 / leave 0.20 / align 0.30, `spill_map`, `AvailTracker`), wired into
  `FixedSlotTracker` (`spillage` kwarg default 0, `_spill`, `.avail`, `.avail_events`, `avail_name`, `set_spillage`), `monitor.py` `state()`
  (per slot `avail` / `display` / `encroached_by` / `cover`; `counts.encroached`, `counts.available`; `counts.free` excludes encroached
  bays), `live.py` `Worker._spillage` (live option `spillage`, "0" default). Spec: "Step 3 as built" in `docs/post_demo_engineering_plan.md`.
- rules: a free, unowned, capacity-1 bay is ENCROACHED when one vehicle covers >= 30 % of it, or several sum to >= 45 %; leaves below 20 %;
  only vehicles on the bay's row count (ground line within 30 % of the bay height); debounce = hold s AND 2 frames to enter, exit_hold s AND
  3 frames to leave. `state` is never changed (history, `sync_open`, Streamlit parity untouched); with the option off `state()` JSON has no new keys.
- found while testing: with the default `bay_share` 0.5 a van wider than ~2 bays is matched to no bay (unplaced), so every bay under it stays
  FREE and is counted free: the exact false count this item targets. Vehicles that hold no bay therefore count as sources (`Spill.by = (0,)`).
- deliberate departure from the spec text: cover = share of the bay inside the vehicle's lateral extent on its row, not the area under the
  footprint strip (the strip is the bottom 30 % of the box, a car wholly over a bay would score ~0.4, so 0.30 would mean 73 % of the width).
- not done: encroach events are NOT written to the history table (the React pages show any non-`entry` event as "vehicle out"; needs an
  API / UI kind filter first); metric (BEV) override; capacity-pin usable length; frontend colours / API switch; `crowded by` notes.
- tests: `tests/test_spillage.py` 23 tests (30 % boundary, sum rule, hysteresis, depth band, unowned sources, config validation, debounce in
  seconds and frames, van over 3 bays, jitter at the threshold 120 frames 0 flips, clears after exit_hold, rolling car, car arriving in the
  bay, capacity pin ignored, on/off identical states and events, switching off, monitor counts on/off/stale, the worker option).
  `docker build --target test`: **564 passed, 26 skipped** (541 + 23), pyflakes clean.
- other: the long camera-1 recording is still running (163 frames at last look); the 4 evidence thresholds are still the synthetic ones.

## 118 - Post-demo plan item 2, first layer: non-vehicle filter (size / aspect / HSV colour breakdown), wired, OFF by default  (2026-10-30, source only; NOT built or deployed, the running stack is untouched)
- scope note: the approved item 2 is the BLOCKED state for free bays; this layer is the other half, requested separately: detections of clutter
  must not make a bay BUSY. BLOCKED itself (obstacles.py, empty-bay appearance model, open-vocabulary probe) is NOT built.
- new: `src/carparking/obstacle_filter.py` (`ObstacleCfg`, `judge`, `color_breakdown`); `FixedSlotTracker` kwargs `obstacle_filter` (default 0) /
  `obstacle_cfg`, `_reject` at the top of `update`, `.rejected`, `set_obstacle_filter`; `monitor.py` hands over the frame when on and adds a
  top-level `rejected` list to `state()` only when on; `live.py` `Worker._obstacle_filter` (live option `obstacle_filter`, "0" default);
  `scripts/replay_eval.py --obstacle`; `pixel_replay.replay` decodes frames for the colour rules and counts rejections by reason.
- rules (first that applies; vehicle detections only; a box in a BUSY bay is never judged): too_small (area < 0.25 of its bay's space),
  aspect (outside 0.5-4.5), pylon (not a whole vehicle, < 0.60 of the bay, and >= 0.45 cone-orange pixels), shadow (>= 0.75 dark and flat).
  Without a frame only size/aspect apply. Spec: "Step 4 as built" in `docs/post_demo_engineering_plan.md`.
- real-data check (camera-1 pixel recording, read only): all 326 sampled real vehicle boxes pass; aspect 0.65-3.16, box/bay area >= 0.9 per space,
  dark <= 0.43, texture >= 11.2, orange 0. Replay of all 256 recorded frames (45.6 min) with the filter on: 0 detections dropped, events and shown
  states identical to the filter off. No real footage of cones / people / shadows exists, so rejection is verified on synthetic scenes only; the
  numbers are conservative defaults, not tuned values.
- bugs the tests found: `crop_box` takes a polygon in fractions (a pixel box gave an empty crop, which fails open); the first shadow thresholds
  (0.60 dark, texture < 6) would have rejected a synthetic black car, so they were tightened to 0.75 / 3.0 (a rejected real car costs more than a
  shadow let through).
- tests: `tests/test_obstacle_filter.py` 19 tests (each rule and its pass cases, real-vehicle aspects, pin per-space size, orange car passes, black car
  passes, filter off/on identical on a 300-frame vehicle-only run, clutter never busy incl. 200 frames of random transients, BUSY bay exempt, no-frame
  behaviour, on/off switching, default off, 20 cars x 30 bays under 5 ms, monitor state on/off, worker option). `docker build --target test`:
  **583 passed, 26 skipped** (564 + 19), pyflakes clean. Off path: one falsy attribute test per update; existing suites unchanged.
- open: BLOCKED proper; overlay still draws rejected boxes; small motorcycles (< 25 % of a car bay) count as too small; no persistence / API / UI switch.
- long camera-1 recording: 256 frames, still running; the evidence-layer thresholds are still the synthetic ones.

## 119 - Settings API, status payload and React UI for spillage protection and the obstacle filter  (2026-10-30, source only; NOT built into the containers or deployed, the running stack is untouched)
- settings API: `SettingsPatch` gained `spillage` / `obstacle_filter` (bool). `GET /api/settings` returns both in `values` (false by default); `PUT` (admin
  token) stores them as the LiveStore options of the same name; they are worker options like `model_variant` (no settings version, no Streamlit
  impact; mixed with a page setting only that one versions). The worker already read both options every frame (items 117 / 118), so a change
  applies to the running `FixedSlotTracker` on the next frame, no restart.
- status payload: slot `avail` / `display` / `encroached_by` / `cover`, `counts.encroached` / `available` (item 117) were already in `snapshot`;
  new: `snapshot.obstacle = {"rejected", "reasons"}` (`Worker._stats`), present only while the filter is on. Both off = the payload is what it was.
- frontend: `Parking protection` section in the Settings drawer (two switches, Apply flow); encroached bays amber + diagonal hatch + "ENCROACHED"
  label + hover text on the video and amber on the grid tiles; Free card = usable bays, new "Encroached (free, not usable)" card (also a filter;
  `SlotFilter` gained "ENCROACHED", the FREE filter shows usable bays only); Worker card line "Obstacle filter: N dropped (reasons)". Everything is
  absent while the options are off.
- tests: `tests/test_api.py` +1 (defaults, auth, validation, persistence, version untouched, null ignored, mixed PUT), `tests/test_live.py` +1 (worker
  status with the options on / off / on while running: encroached fields, obstacle summary, counts, BUSY-bay exemption), `frontend/src/components/
  protection.test.tsx` 12 vitest tests (drawer switches and PUT bodies, hatch / labels / hover / filters on the video, counters, Worker card, helpers).
- found while testing: a cone-sized box that already made a bay BUSY is (by design) not judged afterwards, so the live test switches the filter on
  before the first frame.
- results: `docker build --target test` **585 passed, 26 skipped** (583 + 2), pyflakes clean; `docker build --target web` (tsc --noEmit, vitest, vite build): **24 files, 186 tests passed**, build OK. The long camera-1 recording is still running.


## 120 - Notices in the event log (encroached / encroach_cleared / rejected) and dropped boxes on the video  (2026-10-30, source only; NOT built into the containers or deployed, the running stack is untouched)
- why: the React pages show every event that is not `entry` as "vehicle out", and `summarize()` counted anything that is not `entry` as an exit, so
  the new events could not simply be written into the log. They are now stored as *notices* and every reader of the log keeps seeing only movements.
- storage (`history.py`): `MOVES` / `NOTICES`, `HistoryDB.log_notice()` (no session: `session_id` NULL; `slot_id` 0 = no bay; refuses a movement
  kind), `events(limit, since_id, kinds=MOVES)`, `events_csv(kinds=MOVES)`; `summary()` counts movements only. `SqliteHistory` (history_pg.py)
  filters entry / exit in events, summary, CSV and analytics; `summarize()` ignores other kinds. PostgreSQL untouched (its CHECK and the copy filter
  already admit only entry / exit): notices are a per-camera SQLite audit.
- writing: `ParkingMonitor._log_notices` after each frame: the spillage layer's events (note "by A1, 42 % of the bay: free but not usable";
  "spillage switched off" when the option goes off) and `obstacle_filter.RejectLog` (one `rejected` row per bay and reason after 5 s AND 2 frames,
  then silent 10 min; unseen 60 s = a new episode). A failed write sets `notice_error`, never stops the frame. Both layers off = nothing written.
- API: `GET /api/events?kinds=moves|notices|all` (default `moves` = as before; the other two always from SQLite, 422 otherwise), `/api/events.csv`
  takes the same; the WebSocket feed carries every new event (the ticker shows notices live).
- React: `eventInfo / isMovement / eventSlot` in `lib.ts` (an unknown kind is a notice with its own name, never "vehicle out"); the Event log page
  has All / Entries & exits / Spillage & obstacles (CSV follows), icon labels, a dash for no bay, no session for notices; the live ticker gives each
  kind its own icon / colour / wording; Analytics' per-hour counts movements only.
- video: dropped detections carry `rejected: <reason>` in the snapshot, are not counted in `snapshot.vehicles` and are left out of the metric view
  (`geometry`). React: grey dashed rectangle + `ignored: <reason>` tag + hover text; `render_overlay` (baked frames): dashed grey polygon + tag instead
  of the yellow box. Boxes stay direct children of the overlay SVG (an existing test counts them so). Filter off = drawing and payload as before.
- found while testing: the baked overlay image is 53 px taller than the frame (`draw()` adds a banner), which the first overlay test missed.
- tests: `tests/test_history_notices.py` (storage, kinds filtering, CSV, summary, other readers, RejectLog, monitor writing / clearing / failing,
  API kinds, CSV), `tests/test_api.py` +1 (WebSocket notice), `tests/test_live.py` (snapshot marks the dropped box, vehicles count),
  `tests/test_obstacle_filter.py` +3 (baked overlay grey dashed / yellow solid / helper), `frontend/src/components/eventlog.test.tsx` 12 vitest tests
  (helpers, ticker, History filter and URLs, CSV link, dropped boxes grey / dashed / tag / hover / structure).
- results: `docker build --target test` **606 passed, 26 skipped** (585 + 21), pyflakes clean; `docker build --target web` (tsc --noEmit, vitest,
  vite build): **25 files, 198 tests passed** (186 + 12), build OK.
- open: notices are not in PostgreSQL / central analytics; no retention rule for them beyond the camera file; the Streamlit pages show movements only.
- the long camera-1 recording is still running.

## 121 - Settings drawer reorganised into collapsible groups  (2026-10-30, source only; NOT built into the containers or deployed)
- why: the drawer had grown into one long list of mixed sections. Front end only: no API, payload, default or toggle behaviour changed.
- `SettingsDrawer.tsx`: five cards with icon, title and one-line summary, each folded/unfolded from its header (`aria-expanded`): General & Monitoring
  (Monitoring, camera, stream), AI & Detection Engine (model, confidence), Timing & Hysteresis (entry delay, exit buffers, covered-slot memory),
  Advanced Smart Protections (spillage, non-vehicle filter), Dynamic Counting & Thresholds (hidden, ground taken, multi-slot, alignment, pin grace),
  plus Layout & Parking Zone (auto grid, crop). Advanced / Debug and Actions unchanged. General and AI start unfolded.
- folded bodies stay mounted (`hidden`), so the draft, Apply payload (`PUT /api/settings`, changed fields + version), tuning reset and the
  immediate overflow switch behave as before; a folded header shows "N changed" for unsaved edits in it.
- hints: clearer subtitle/tooltip on the exit buffers and the two hidden/ground sliders. Bhatta / NCC are evidence-layer thresholds, not user
  settings, so they have no slider here.
- tests: `settingsgroups.test.tsx` (+2); existing drawer/protection tests pass untouched.

## 122 - Spillage on the ground: calibrated wheel-footprint distance instead of pixel overlap  (2026-10-30, source only; NOT built into the containers or deployed)
- why: the pixel cover (share of the bay inside the vehicle box's x-range) read a box's side wall / mirrors / perspective as standing on the next bay.
- `spillage.GroundMeasure` (per frame, from the camera's active calibration): the vehicle's ground-contact line (bottom edge of its box, trimmed by
  `SpillCfg.wheel_inset` 0.06 of the width at each end: the wheels stand inside the body) is projected to metres (`geometry.to_ground`); the bay to its
  metric quad (short axis = width axis); cover = metres of the bay's width the wheels' line reaches / bay width, same row rule (`align`).
  Reaching less than `SpillCfg.margin_m` (0.15 m) counts as nothing. Same thresholds after that (share 0.30 / total 0.45 / leave 0.20).
- fallback, per pair: above the horizon, beyond `CAR_MAX_RES` m/px, no camera, any exception -> the unchanged pixel rule for that pair; no calibration
  (or spillage off) = byte-identical behaviour (test). `Spill` gains `metric` and `free_m` (metres of the bay's width left, ground metric only);
  `monitor.state()` publishes `free_m` per encroached slot only when measured; React adds "N m of ground left" to the slot hover.
- wiring: `GeometryFeed.camera(shape)` (None when uncalibrated / other shape / stale, never raises), `Worker._ground()` hands it to
  `FixedSlotTracker.ground_cam` each frame while spillage is on. No new option or API field: calibrated camera + spillage on = ground metric.
- untuned: `margin_m` 0.15 and `wheel_inset` 0.06 are engineering estimates, not fitted to camera-1 footage (no calibrated recording with ground truth yet).
- tests: `tests/test_metric_spillage.py` (14), `frontend/src/components/protection.test.tsx` +1.

## 123 - Ground-metric spillage hardened: footprint area on any axis, UNKNOWN bays, tuning fallback  (2026-10-30, source only; NOT built/deployed)
- symptom: on the live dashboard the encroached count stayed 0 although vehicles stood on neighbouring bays (camera angles where cars have a visible gap).
- diagnosis (item 122's measure, found by reading camera-1's slot map in `recordings/camera1_pixels/*/header.json`, no ./data read):
  1. the measure used only the vehicle's CONTACT LINE (the bottom edge of its box) projected onto the bay's short axis. Camera-1's bays B1..B7 are
     strips across the lane stacked one behind the other, so the neighbour is behind, not beside: the contact line is a point on that axis and its
     reach was 0 -> "no encroachment" for every pair (the pixel rule only looked sideways as well).
  2. the tracker only offered FREE bays as candidates; a bay behind a vehicle is UNKNOWN (that vehicle's box hides it), i.e. exactly the bay its
     far end stands on was never examined (reproduced in `test_tracker_sees_the_stacked_neighbour...`).
  3. (checked, not the cause) `wheel_inset` / `margin_m` / camera height: shares cancel a scale error in the calibration (test with the camera
     25 % higher/lower: same verdict, cover within 0.10 because the guessed footprint depth is in metres); the margin is only metres and is now
     capped at a quarter of the bay's short side.
- fix (`spillage.py`): `GroundMeasure` now builds the vehicle's FOOTPRINT (`bev.car_footprint`: contact line trimmed `wheel_inset`, depth guessed
  by view, scaled by `SpillCfg.depth_trust` 0.85 because the far end is unseen) and the bay's convex metric polygon, and cover = overlap AREA /
  the smaller of bay and footprint (Sutherland-Hodgman clipping, no new dependency): it works along whichever axis the neighbours lie. The margin:
  the footprint must reach `min(margin_m, short side / 4)` inside the bay's edges. `Spill.free_m` = the bay's short side x (1 - total cover).
- fix (`fixed_slots.py`, `monitor.py`): with a ground camera, UNKNOWN bays are candidates too (pixel mode unchanged: FREE only). `counts.encroached`
  counts every encroached bay, but only a FREE one leaves `counts.free` (UNKNOWN stays in `unknown`).
- tuning fallback: live option `spillage_ground` = `auto` (default, ground reading), `union` (the higher of the ground and the pixel cover per
  pair: use when the ground reading is too cautious on some camera), `off` (pixels only); read each frame by `Worker._ground`, no restart, no API
  field (set it in the live options table). Per pair the pixel rule still answers where the calibration cannot (horizon, > CAR_MAX_RES, shape).
- not verified on real footage: no calibrated camera-1 recording with ground truth; `depth_trust`, `wheel_inset`, `margin_m` remain estimates.
- tests: `tests/test_metric_spillage.py` rewritten (23): side-by-side and stacked layouts, own-bay, gap/van, union, margin cap, two cars,
  camera height +-25 %, 6 deg roll, fallback, broken camera, byte-identical without calibration, worker option, state counts with an UNKNOWN bay.

## 124 - Clearing the event log for a fresh start: `scripts/clear_events.py`  (2026-10-30, source only; the purge itself was NOT run, ./data untouched)
- script: `python scripts/clear_events.py --db <history.db> --camera camera-1 [--kinds all|moves|notices] [--yes] [--postgres]`. Dry run unless `--yes`;
  with `--yes` a consistent SQLite copy goes first to `recordings/event_purge_backups/<time>/` (outside ./data; `--no-backup` skips), then one
  transaction `DELETE FROM events ...`. `--postgres` also deletes that camera's `parking_events` ($CARPARK_DATABASE_URL), because `/api/events` (moves)
  is served from PostgreSQL when fresh and would otherwise still list the old rows.
- deliberately kept: sessions (`history`), slots, settings, live tables, the `events` table and its AUTOINCREMENT counter. New events therefore get
  ids above every old one: a browser's WebSocket feed (`since_id`) and PostgreSQL's sync high-water mark never see an id twice; and an emptied table
  is not re-seeded from `history` on the next start (that backfill only runs when the table does not exist).
- restart: a purge needs no rebuild. The worker's in-memory state (FixedSlotTracker, spillage tracker, open sessions) is untouched; a bay that is
  ENCROACHED at that moment will later log `encroach_cleared` without an `encroached` before it. `docker compose restart worker` after clearing starts
  that memory from nothing too.
- UI: the ticker already said "No events logged yet."; the Event log page now shows the same in an empty-state row (`event-empty`).
- tests: `tests/test_clear_events.py` (10: dry run, backup + delete, kinds, per-camera, ids keep counting, no re-seed, tracker and an open session
  carry on and the exit is logged, fresh DB, missing DB not created, CLI) and `eventlog.test.tsx` +1.

## 125 - Event purge executed on the live stack (camera-1), and the sync check it tripped  (2026-10-30)
- run (authorised by the user): the api image has no `scripts/clear_events.py` (built before it existed), so the file was copied into the running
  api container (`docker compose cp`, lost on recreate). Two corrections to the given command: `--db /data/parking_history.db` is required, and the
  SQLite events carry the display name `Camera 1` while PostgreSQL has the registry id `camera-1` -> new `--pg-camera`. `--camera camera-1` would
  have matched nothing in SQLite. Dry run first: 1587 SQLite events, 1520 PostgreSQL rows. Then `--yes --postgres`: both deleted; backup of the
  database in `recordings/event_purge_backups/20261030-161104/parking_history.db` (consistent copy, taken before the delete). `docker compose
  restart api worker` as asked. Sessions, slots, settings untouched. Camera-2 / test not touched.
- found: `HistorySync._copy` refused camera-1 ("fewer rows than already copied") because it compared the table's MAX(id) (0 after the purge) with
  the copied high-water mark; the earlier claim that the counter protects the sync was wrong for that check. It healed by itself at the first new
  event (id 1588), no data lost. Fixed in source: `history_pg._last_id` (AUTOINCREMENT counter or MAX(id)) in `_copy` and `sqlite_counts`;
  regression test in `tests/test_history_pg.py`. The running sync container still has the old code until rebuilt (harmless: next purge before a
  rebuild would pause the sync until the next event).
- state afterwards: `/api/events` kinds=all/moves/notices were `[]` right after the restart; then the restarted worker logged one real event
  (id 1588, entry slot 6, "already parked when tracking started": tracking restart re-opens a car that is parked). The page shows
  "No events logged yet." (in the served bundle) when the list is empty.
- observed, not caused by the purge: "database is locked" from sync (since 11:01 UTC), worker and occasional API 500s on `/api/events?kinds=notices`
  (always SQLite): several containers share one rollback-journal SQLite file on a bind mount. Pre-dates the purge. Not changed.

## 126 - Rebuild and recreate of api, worker, sync  (2026-10-30)
- `docker compose build api` (all four app services share `stomeetings/carparking:local`): the runtime stage is gated on the test stage. The first two
  attempts built nothing (the tag stayed at the 10:34 image): the test stage failed with 1-3 timing failures in `tests/test_setup_*` (calibration
  jobs "did not finish"), a different test each time, while other builds / the full test run competed for CPU (25 min instead of ~10). Rebuilt with
  nothing else running: passed, new image `sha256:1dd865f8ae56...`, created 12:35 UTC.
- `docker compose up -d --no-deps api worker sync` (recreate, not restart: a restart keeps the old image). All three healthy within seconds; the
  `ui` (Streamlit) container was not asked for and still runs the previous image.
- the image now holds: `scripts/clear_events.py` (items 124/125, `--pg-camera`), the sync fix `history_pg._last_id` (item 125), ground-metric spillage with
  footprint area / UNKNOWN bays / `spillage_ground` option (items 122/123), grouped Settings drawer (121). Checked inside the running containers.
- after: `/api/events` 200, sync logs only worker online/offline lines (no "fewer rows" / lock error), worker log without traceback in the first 2 minutes.
- summary of this session's items: 115-120 recording/replay tooling, spillage, obstacle filter, settings/UI, notices; 121 settings groups;
  122-123 ground-metric spillage and its fix; 124-125 event purge (executed: camera-1, 1587 SQLite + 1520 PostgreSQL rows, backup in
  `recordings/event_purge_backups/20261030-161104/`); 126 this rebuild.
- open: SQLite "database is locked" between worker / sync / api (shared rollback-journal file, pre-existing); evidence-layer and ground-metric
  thresholds untuned (no truth data); notices not mirrored to PostgreSQL; the long camera-1 pixel recording keeps running.

## 127 - Far-field bay extrusion as a draft preview (plan item 3, setup time only)  (2026-11-02, source only; NOT built into the containers or deployed)
- toggle: `CARPARK_FAR_EXTRUDE` (default off, `.env.example`) + per-request `far_extrude` on `POST /api/cameras/{id}/suggest`. Off = the suggest job is unchanged
  (test: two drafts, nothing / `false`, are identical bay for bay, no new key on a bay). Runs only with `CARPARK_GEOMETRY_ENABLED`.
- `geometry/extrude.py`: RANSAC lane over the measured bays (lines leading away from the camera preferred, so a wide lot's rows do not win),
  Huber TLS line fit with far cars / painted dividers, Huber LS pitch + phase (dividers at half indices), extrusion from the nearest bay as template, stops
  (60 m, frame, < 14 px tall, SAM surface end, uncertainty > 0.35 pitch), sigma = fit + calibration (pipeline re-run under pitch +-0.5 deg / f +-3 %), capacity zone
  beyond, `far_report` diagnostics. `geometry/far_evidence.py`: far band, 50 % tiles x2, temporal median, tile detections back in picture pixels, painted dividers
  in a bird's-eye resampling, `far_surface_m`. `setup/service.py` `POST /segment_tiled`. `slot_suggest.measured_tiles` / `add_far_bays`.
- API: the draft gets `source: "far"` bays (`far`, `confidence`, `sigma_px`, `sigma_m`; fresh ids above everything in use) and `meta.far = {zone, stop, model, evidence,
  error, warnings}`; `GET /api/cameras/{id}/far-field` read-only diagnostics + dry extrusion (no still, no model). Fine-tuning a far bay keeps its uncertainty.
- never automatic: worker reads the active version only; `fixed_slots.json` untouched; on a Fixed camera far bays are NOT ticked for the merge; an auto camera's confirmation
  names the far bays that would become real slots.
- UI (`SuggestSlots`): "Include far bays (preview)" checkbox (off), "Check far field", far bays violet and dotted (LOW) / dashed (MEDIUM), capacity zone with
  "~N bays - not verifiable", summary line, per-bay badge with the uncertainty.
- finding: with an honest calibration prior the lane is trusted only a little past the measured surface (synthetic camera-1: 2 bays to 33 m, then a zone of ~10 to 59 m); more
  would need better calibration or far evidence, not a looser threshold. Thresholds are the plan's, untuned on real footage.
- not built: plan 3c (runtime zoom priority, evidence weight, `counts.unverifiable`) and the real-footage validation.
- tests: `tests/test_extrude.py` (12), `tests/test_far_evidence.py` (8), `tests/test_far_suggest_api.py` (7), `setup/test_service.py` +2, `frontend/src/components/farfield.test.tsx` (12).

## 128 - Slot suggestions keep the operator's manual pins (P1)  (2026-11-02, source only; NOT built into the containers or deployed)
- cause: a Fixed camera's capacity pin (P1, slot 9, holds 4) was an ordinary drawn slot to `carry_ids` / `diff` (a suggested bay on it took its id, else it showed as
  "removed") and the far-field run ignored it (bays and zone over the ground it already counts).
- option: `keep_pins` on `POST /api/cameras/{id}/suggest` (null = `CARPARK_SUGGEST_KEEP_PINS`, default off: the suggestion is exactly as before; `.env.example`).
  `slot_suggest.fixed_pins` (capacity > 1) and `pin_ground` (pins in metres); `extrude(pins_m=...)` leaves out bays a pin covers >= 25 % and nets the zone of the pins' capacity;
  a suggested bay on a pin (IoU >= 0.3) is dropped, the pin's id reserved, the diff excludes pins; `meta.keep_pins` only when on; the view's `pins` are the live pins.
- UI (`SuggestSlots`): "Preserve existing manual pins (e.g. P1)" box, ticked by default, sends `keep_pins: true`; the draft names the kept pins and what that did to the far bays;
  pins/far bays are never ticked for a Fixed camera's merge. `suggest.test.tsx` body expectation updated for the new default field.
- tests: `tests/test_pin_suggest.py` (9), `frontend/src/components/suggestpins.test.tsx` (6).

## 129 - Far-field extrusion (items 127) and keep_pins (item 128) REVERTED  (2026-11-02, source; the running containers still hold the removed code until rebuilt)
- why: the operator reported the stable auto slot suggestion disturbed (overlapping slots, wrong counts, wrong additions / removals).
- removed: `geometry/extrude.py`, `geometry/far_evidence.py`; `slot_suggest` (`FAR`, `FAR_KEYS`, `measured_tiles`, `fixed_pins`, `pin_ground`, `add_far_bays`, the `MEASURED` / `to_ground` imports);
  `api/geometry_routes.py` (`SuggestIn.far_extrude` / `keep_pins`, `far_extrusion`, `GET .../far-field`, the pin logic in the suggest job, `suggestion_view`, the fine-tune FAR_KEYS carry-over,
  the result's `far`); `api/main.py` (`Config.far_extrude`, `suggest_keep_pins`); `setup/service.py` `POST /segment_tiled` and its 2 tests; `.env.example` (`CARPARK_FAR_EXTRUDE`,
  `CARPARK_SUGGEST_KEEP_PINS`); the Net `/segment_tiled` branch in `tests/test_setup_api.py`; React `suggest.ts` / `SuggestSlots.tsx` restored to their text before item 127 (no far / pins
  checkboxes, no violet bays, no zone, no "Check far field"); tests `test_extrude`, `test_far_evidence`, `test_far_suggest_api`, `test_pin_suggest`, `farfield.test.tsx`, `suggestpins.test.tsx`
  deleted; `suggest.test.tsx` body expectation back to `{mode, still, expected}`.
- kept (unrelated, earlier): the sync `_last_id` fix (125), `clear_events.py` (124), the ground-metric spillage (122-123), grouped Settings (121).
- backup of every touched / removed file as it was just before the revert: `recordings/reverted_far_field_20261102/` (outside ./data). Docs sections "Step 7" / "Step 8" in
  `docs/post_demo_engineering_plan.md` stay as the design record, marked REVERTED. Items 127 / 128 above are history.
- not changed: the running slots, `fixed_slots.json`, the live database. Whether the regression came from the far-field code itself or from the pins box (ticked by default, so
  a Fixed camera's capacity pin was dropped from the diff and a suggested bay on it was discarded) was not isolated: both are out.
- result of the revert: `docker build --target test` **640 passed, 26 skipped**, pyflakes clean (= the count before item 127 plus the later sync test); `--target web` 26 files / 202 tests,
  tsc and vite build OK (= before item 127). A first run after the revert showed 24 failures: the revert had also cut `Config.setup_url` in `api/main.py` (it sat inside the removed block);
  restored, all pass. Nothing was rebuilt or recreated: the running api / worker / sync / setup images still contain the removed code.

## 130 - A Fixed camera's running slots (B1..B8, P1) are protected in a slot suggestion  (2026-11-02, source only; NOT built/deployed)
- symptom: the draft marked running slots "removed" (red) and laid new bays (B10, B11 ...) over them. Cause: `carry_ids` only keeps a running slot's id for a suggested bay that
  overlaps it by IoU >= 0.5; a drawn slot sitting inside a calibration tile (IoU ~0.3) was left unmatched = "removed", and the tile became a fresh bay on top of it.
- fix (Fixed cameras only; an auto-grid camera's draft is untouched, its Accept replaces the grid by design): `slot_suggest.protect_baseline` builds the draft around the running slots:
  every drawn slot (B1.., the pin P1 with its capacity) is carried in exactly as it runs (same id, name, polygon, capacity; `source: "running"`, `protected: true`); of the suggested bays
  only those that do not lie on a running slot (<= 10 % of the smaller one on the other, `overlap_share`; side by side or one behind the other = ~0) are added, with fresh ids above every
  running id. So the diff has no "removed" and no "moved" for a running slot, and no green bay sits on one. `meta.protected` = the ids (Fixed drafts only).
- the fine-tune endpoint refuses (422) to move, reshape or remove a protected slot in the draft; Accept on a Fixed camera merges only what is ticked and a running slot comes out as drawn
  either way.
- UI: protected bays are solid, badged "running" (with "holds N" for the pin), not draggable, no corners, no remove button, not ticked for the merge; a line says how many running slots
  are protected and how many new bays were added; the Accept confirmation says they stay as they run.
- the far-field extension is not part of the code (reverted, item 129); if it comes back it must add through `protect_baseline`'s overlap rule.
- tests: `tests/test_suggest_protect.py` (8: the rule, the old behaviour reproduced with the same polygons, API on a Fixed camera, fine-tune refusals, accept, auto camera unchanged),
  `frontend/src/components/suggestprotect.test.tsx` (6).

## 131 - Protection extended to the AUTO-GRID camera (camera-1 is one)  (2026-11-02, source only; NOT built/deployed)
- item 130 protected Fixed-map cameras only. camera-1 is an auto-grid camera (calibration v10: B1..B8 measured, P1 a manual pin holding 4; `mode: fixed` in the status is the tracker class),
  so its draft still went through `carry_ids` and showed running bays removed / moved with fresh bays over them. Reproduced on its real data (`tests/fixtures/camera1_auto_grid.json`:
  v10 record, the running grid, one frame's vehicles): old matching = removed [2, 5], moved [1, 3, 4, 6, 7]; protected = removed [], moved [], unchanged B1..B8.
- fix: the suggest job now builds EVERY draft with `protect_baseline` (Fixed and auto): the running bays are copied in as they are (id, name, shape, source, capacity; `protected`),
  never matched, moved or dropped; of the suggested bays only those that do not lie on a running bay or on a pin are added, with fresh ids above every running id and pin id; the pin
  stays a pin (merged by the worker on top of the grid, shown purple in the draft). The fine-tune endpoint refuses to move / reshape / remove a protected bay (422).
- what Accept does on an auto camera: the draft becomes the ACTIVE version and its bays are the grid. With a protected draft that is the running bays (unchanged) plus the added bays;
  with an unprotected draft (the old behaviour, and any draft made before this fix) it is only the bays in the draft, i.e. the red ones would leave the live grid (the previous version stays
  superseded; "rollback" = activate it again; pins and the history of the removed bays are kept).
- tests: `test_suggest_protect.py` 9 (+ the real camera-1 case), `test_suggest_api.py` fine-tune test rewritten for the new rule (a running bay cannot be moved or removed), vitest +1.

## 132 - "Blocked by object" (BLOCKED) slot state  (2026-11-02, source only; NOT built/deployed)
- a FREE bay with a potted plant / vase / suitcase / chair / bench standing in it is BLOCKED: FREE for the history and sessions (no entry / exit), out of the Free count, drawn
  cross-hatched slate. Option `blocked_objects` (Settings > Advanced Smart Protections), default OFF; worker option read every frame (no restart, no settings version).
- `blocked.py` (rule + config), `FixedSlotTracker(blocked=...)` / `set_blocked` / `_block` / `avail_name` (BLOCKED wins), `AvailTracker(events=, notes=)` (defaults = spillage's),
  `ParkingMonitor.set_blocked` (object classes join the targets only while on), `state()` (`avail`/`display` BLOCKED, `blocked_by`, `counts.blocked_objects`, `free` / `available` net of it),
  notices `object_blocked` / `object_cleared` in `history.NOTICES`, `Worker._blocked`, `SettingsPatch.blocked_objects`.
- off = nothing runs: tracker output equal to a tracker without the layer (test), no new key in `state()`, the class list untouched; the live-worker test shows the pot not even reported
  until the option is on and gone again when it is turned off.
- React: switch, BLOCKED drawing (`BlockHatch`), hover line, "Blocked by object" card + filter, ticker / event-log labels.
- names: `counts.blocked` / slot `blocked` already mean "a vehicle stands in it", so the new count is `counts.blocked_objects`.
- tests: `tests/test_blocked_objects.py` (20), `test_live.py` +1, `test_api.py` +2, `frontend/src/components/blocked.test.tsx` (11).
- not done: validation on real footage; the label list as a setting; PostgreSQL copy of the notices.

## 133 - Streamlit page clean-up (authorised by the operator; it had been kept untouched until now)  (2026-11-02, source only; NOT built/deployed)
- backup first: the whole code tree (202 files, everything but data, backups, node_modules, dist, recordings, runs, datasets, models, logs, weights and databases) copied to
  `backups/pre-streamlit-cleanup-20261102-142435/code` with `manifest.json` (sha256 of every file); every copy re-hashed against its source (0 mismatches). It contains `.env` (secrets): keep it private.
  Restore: copy `code/app.py`, `code/scripts/demo_app.py` (and `src/carparking/fabric_shapes.py`, `.streamlit/`) back.
- findings: `scripts/demo_app.py` (1118 lines; `app.py` is a 5-line launcher) had no unused imports, constants or functions (pyflakes clean; an AST pass found no top-level name that is never
  referenced), so nothing was deleted for being dead. What was redundant:
  * the FREE / BUSY / TOTAL numbers written out four times (image test, Mode 4 upload, both webcam loops) -> `n_free`, `show_metrics`, `fps_line`;
  * the webcam open + error text written twice -> `open_camera` (each page keeps its own hint in the message);
  * `canvas_background` masked, resized and JPEG-encoded the camera frame on EVERY rerun of the fixed slot editor (every click) -> kept in the session while the frame (its capture time)
    and the parking area (cuts, polygon) are unchanged.
  Net +20 lines (the helpers carry docstrings); behaviour, widgets, keys, stored settings unchanged.
- tests: `tests/test_settings_parity.py` (7) unchanged and green; `tests/test_streamlit_page.py` (8): all five input modes render, the helpers do what the copies did, the canvas
  background is rebuilt exactly when the frame or the parking area changes.
- not touched: the worker / FastAPI / React code; `.streamlit/config.toml` (the poll watcher is needed on the Windows bind mount).

## 134 - Sample image test (upload a picture, run it through the whole stack, see it validated)  (2026-11-02, source only; NOT built/deployed)
- backend: `src/carparking/sample_test.py` (`detect_frame`, `Replay`, `schedule`, `run_sample`), `api/sample_routes.py` (`GET /api/sample-test/images`, `POST /api/sample-test`, admin token,
  one at a time, 12 MB / 40 MP limits, a bare file name only for the samples folder), registered in `create_app`. The detector runs once; a throw-away `ParkingMonitor` (in-memory history)
  replays its answer until the states settle (two hold times); layers default to the camera's live options and can be overridden per request; ground metric for spillage when the picture fits
  the active calibration. A Fixed camera uses its drawn slots, an auto-grid camera the active HIGH calibration's grid with its pins; a Dynamic camera without a grid gets a 409 saying so.
- nothing live is touched (settings, saved slots, the real event log, the worker): checked by test.
- React: sidebar item "Sample test" (`#/sample`), `SampleTest.tsx` (upload / pick, layer switches from the live options, Run test, picture with SlotShape / BoxShape overlay, cards via `Counters`,
  notices via `EventTicker`, slot table, expected-vs-got chips), `SlotShape` / `BoxShape` exported from `LiveView.tsx`, `Button` forwards `data-testid`.
- the dashboard's top cards and the real event log are NOT updated by a test on purpose: the page shows the same cards and notices for the picture itself.
- tests: `tests/test_sample_test.py` (16: the pipeline with each layer and all together, schedule, replay, endpoints, auth, validation, isolation), `frontend/src/components/sample.test.tsx` (11).

## 135 - Sample pictures from anywhere: their own calibration and grid  (2026-11-02, source only; NOT built/deployed)
- why: item 134 answered 409 for a picture that is not the view of a configured camera. A session now gives such a picture a throw-away calibration and grid, then runs the same stack.
- backend: `src/carparking/geometry/pins.py` (`camera_from_rectangle`: 4 clicked corners of a known ground rectangle give homography, f, tilt and height with no models; crossing pins refused),
  `src/carparking/sample_session.py` (upright/resized load, `rectangle_record`, `geocalib_record`, `grid_slots`, `session_run`), `api/sample_sessions.py`
  (`POST/GET/DELETE /api/sample-sessions`, `/surface` (SAM), `/calibrate` (rectangle default | geocalib + typed distance), `PUT /slots`, `POST /run`; admin token; in memory, max 5 sessions, 30 min TTL,
  one model job at a time -> 409). The session uses the chosen camera's detector and confidence only, never its area cuts. Nothing is written to the geometry store, registry, slots, options, history or event log.
- React: `Sample test` page has two modes ("This camera's slots", "Calibrate a picture"): `SampleSession.tsx` (upload, pins or reference, surface outline / SAM, calibrate job, bay review with drag and delete, layers, run),
  shared `SampleResult.tsx`.
- tests: `tests/test_pins.py` (11), `tests/test_sample_session.py` (6), `tests/test_sample_sessions.py` (11), `frontend/src/components/samplesession.test.tsx` (11). Vitest: 30 files / 241 tests pass; tsc clean.
- not built: "Save as test case". Known gap unchanged: `reanchor_record` does not rotate a frozen `record["grid"]`.

## 136 - Sample test page: one workflow (GeoCalib + SAM), the "This camera's slots" mode removed  (2026-11-02, source only; NOT built/deployed)
- removed: the "This camera's slots" tab (`SampleTest.tsx`, the page-level mode switch) and its endpoints `GET /api/sample-test/images`, `POST /api/sample-test` (`api/sample_routes.py`, `register_sample`).
  `carparking/sample_test.py` stays: it is the pipeline every session run uses. The page (`#/sample`) opens straight into `SampleSession`.
- the workflow is now GeoCalib first: default scale method "GeoCalib + distance" (two clicked ground points + a typed distance), then SAM for the surface, Calibrate, review the bays, Run test.
  "Ground pins (no service)" is kept as a fallback inside the same flow for when the setup service is not running (it needs no models).
- SAM "unresponsive": the job's progress was shown only in the Calibrate card, far from the button, and a stopped setup service was reported only after the job ran. Now:
  `GET /api/sample-sessions/service` (setup `/health`: up/down, models loaded) drives a banner with the start command and a Check button (re-checked after any failed job);
  the SAM card shows the job stage and a spinner, says what the clicks mean (turquoise = parking, Shift-click red = not parking, sent as label 0) and confirms the outline when it arrives.
- `GET /api/sample-sessions/images` + a picker: start a session from a picture in `data/samples/`.
- tests: `tests/test_sample_test.py` keeps the pipeline tests only; `tests/test_sample_sessions.py` +2 (service, images, negative SAM click, old endpoints gone, 13 total);
  `sample.test.tsx` now tests the result panel (`ResultPanel`); `samplesession.test.tsx` 16 (banner, picker, SAM progress and failure). Vitest 30 files / 240 tests, tsc clean.

## 137 - Sample test: detection settings (model, confidence, NMS, footprint, encroach), detector findings, far-vehicle fine-tune guide  (2026-11-02, source only; NOT built/deployed)
- finding (measured on `data/sample/Authorities barred from collecting parking fee.png`, 640x480): the sample test ran the profile camera's detector = stock COCO yolov8s at 1920 px, NOT PKLot weights.
  The PKLot weights (`parking_yolo.pt`) are bay detectors and find 0 on this picture. yolov8s vehicles found: 22 at conf 0.30, 27 at 0.20, 30 at 0.10; yolov8x 25 at 0.30. In a calibrated session
  26 were detected but 18 counted: the rest stood outside the parking surface. Free/Busy/Encroached evaluated correctly on bays placed over the real cars (4 busy, free road free); no classification bug
  found in `run_sample`, the misses came from confidence 0.30, the surface cut and a grid that is not on the cars.
- backend: `Tune` (model, conf 0.10-0.75, iou 0.20-0.70, footprint 0.20-0.80, encroach 0.15-0.60) on `/surface`, `/calibrate`, `/run`; the session keeps the last values; defaults conf 0.20, iou 0.45, footprint 0.50,
  encroach 0.30. `GET /api/sample-sessions/models` (this camera's detector, stock COCO weights in the project root / models/, camera fine-tunes, the PKLot bay model flagged, yolo11n/s/m downloadable on first use; ids only,
  never a path from the request). `sample_test.detect_frame(..., tune)` applies conf / iou (model NMS and the duplicate merge) for the call and puts the shared detector back; `detector_spec`; `run_sample(..., tune)` sets the
  Blocked footprint share (hysteresis half of it, min object conf not above the confidence used) and the spillage encroach share. One detector pass per (model, conf, iou, objects), up to 8 cached per session. The result
  carries `tune`, `detector`, `detected`. Pictures in `data/sample/` are listed with `data/samples/`.
- React: a collapsible "Detection settings" card on `#/sample` (model select with a note, sliders + number boxes, Reset), the values summarised in its title; the result summary says how many vehicles were found vs counted
  (outside the surface / dropped by the filter) and the settings used.
- `scripts/augment_far.py` (far-away copies of a YOLO train split) and `docs/FINETUNE_DISTANT_VEHICLES.md` (causes, the order of fixes, the train commands).
- checked end to end with the real detector through the API on that picture (yolov8s / yolov8x, conf 0.30 / 0.20 / 0.10): counted vehicles 17 / 18 / 19 of 22 / 26 / 29 found.
- tests: `test_sample_test.py` +4 (detect_frame tune and restore, spec, footprint, encroach), `test_sample_sessions.py` +5, `test_augment_far.py` 3, `samplesession.test.tsx` +3.

## 138 - Model audit, trained-model findings and the Camera 1 database repair  (2026-11-02)
- model inventory answered (React / API / worker only, Streamlit-only ResNet-18 and SAHI left out): live YOLOv8s (stock COCO, 1920 px); optional YOLO11m (`model_variant`); `cam_front_parking_02.pt` and the PKLot bay
  models selectable but not used by any camera; YOLOv8n / YOLOv8x selectable in the Sample test only; setup service: SAM 2.1 Hiera Small, GeoCalib (pinhole), Depth Anything V2 Metric Small.
- checked from the weight files' training metadata: `parking_yolo.pt` (PKLot, 20 epochs, 640 px), `parking_yolo_obb_holdout-PUCPR.pt` (PKLot OBB from yolov8n-obb, 1 epoch), `cam_front_parking_02.pt` (yolov8n fine-tune on
  Front Parking 02 frames, 60 epochs, 1280 px); yolov8s / yolov8n are stock COCO. Live config: Camera 1, Camera 2 (test) and test all run `yolov8s.pt` whole 1920; no trained model is in the live path.
- Sample picture (`data/sample/Authorities barred ...png`, 640x480): PKLot models find nothing at 640 px; at 1920 px 7-17 bays (mostly busy) at conf 0.25-0.10. Correction of the earlier "0 detections" statement, which held only for 640 px.
  The Front Parking 02 fine-tune gives 36-92 boxes (45 at conf 0.30 / 1280 px): duplicates, car fragments, a person as a motorcycle, and it misses the large front-left cars; COCO yolov8s is better on this view.
- guidance given (no code): slots come from SAM + GeoCalib, busy/free from a vehicle detector matched to those slots; a custom model fits only as a vehicle-class model (a Camera 1 fine-tune, not built); changing the model in the
  Sample test needs only Run test, no recalibration; PKLot as a second opinion / COCO + PKLot combined mode NOT built, decision left open.
- incident: `GET /api/cameras` 500 "Couldn't load the cameras" and worker unhealthy. Cause: `data/parking_history.db` (Camera 1) had a corrupt `live_state` table ("database disk image is malformed", integrity_check:
  overflow list of page 44 truncated); every other table was intact (history 949, events 880, slots 176, settings, live_config, live_options, live_commands, slot_pairs). Fix: backed up the file to
  `backups/corrupt-db-20261102-195400/parking_history.db.corrupt`, rebuilt it table by table with `live_state` recreated empty (counts equal, integrity ok), stopped worker / api / sync / ui, swapped the file, started them again
  (no rebuild, no redeploy). Result: `/api/cameras` 200, worker healthy, no `malformed` errors. Camera 2 and test databases were checked: ok. Root cause of the corruption not determined (my tests mounted ./data read-only);
  Docker Desktop bind-mounted SQLite on Windows is a known risk: schedule `scripts/db_backup.py`.

## 139 - Sample test: "Direct bay model (no calibration)" comparison run  (2026-11-02, source only; NOT built/deployed)
- why: show the trained PKLot bay model next to the SAM + GeoCalib pipeline in the demo. The default pipeline is unchanged and stays the main flow.
- backend: `POST /api/sample-sessions/{id}/direct` {model?, conf 0.05-0.75 (0.25), iou 0.20-0.70 (0.45)} (admin): the chosen bay model (parking_yolo*.pt only; 422 for any other id, 409 when none is installed) runs on the session picture
  through `sample_test.detect_frame` (whole frame, no SAM / GeoCalib / calibration / grid / monitor / layers); its own boxes come back as BUSY / FREE bays in the same result shape (`kind: "direct"`, `detector`, `direct` settings).
  It stores nothing: the session's calibration, slots and settings are untouched, nothing is written to a database or the event log. One model job at a time (409). `detect_frame` also sets `obb_iou` so IoU reaches oriented-box weights.
- React: a "Direct bay model (no calibration)" card appears once a picture is loaded (bay model select, confidence, IoU, Run direct bay model); its result shows in its own panel under the calibrated one so both can be read together;
  the calibrated flow's code path is unchanged.
- real run on the sample picture through the endpoint: `parking_yolo.pt` 0 bays at conf 0.25 and 0.10; OBB 4 busy / 0 free (0.25), 6 busy / 0 free (0.10).
- tests: `test_sample_sessions.py` +3 (direct run on an uncalibrated picture, session unchanged, calibrated run identical before/after, validation, 409), `samplesession.test.tsx` +3.

## 140 - Verification of item 139 and the PKLot bay-model evaluation  (2026-11-02)
- PKLot / bay model evaluated on `data/sample/Authorities barred ...png` without code changes (before item 139): through the app's own path at conf 0.25 `parking_yolo.pt` found 0 bays and the OBB model 4 busy / 0 free; drawn on the picture, 2 of the 4 OBB
  boxes sit on one car (partly) and 2 on building windows (false); ~28 cars are parked, none of them found as a whole. At 1920 px the baseline gave 7-17 bays (mostly busy). Neither model finds a free bay. Conclusion: the oblique 640 px view makes
  the PKLot models fail (top-down training data); the SAM + GeoCalib grid with a vehicle detector stays the right pipeline for Camera 1.
- item 139 verified: full `docker build --target test` (pytest + pyflakes) exit 0 and `--target web` (tsc + vitest 246 + vite build) exit 0, each run alone. Still source only: the live containers do not have the Direct bay model option.

## 141 - Sample sessions follow the live auto-grid rules; no parking-area cut in a session  (2026-11-02, source only; NOT built/deployed)
- analysis (low-res oblique pictures): the detector finds 26 of ~28 cars; the loss was in how a session evaluated them. (1) The session cut the picture by the surface outline, and a fixed map keeps a box only if its CENTRE is in
  the mask: on an oblique view the centre is on the car body, above the ground the outline covers. (2) `run_sample` ran with `auto: None`, so a car off the packed tiles was ignored, unlike live Camera 1 (`Worker._auto_run`), which
  gives it the bay its wheel footprint overlaps or a bay of its own. (3) the packed grid is a lattice from the measured ground; far bays at 640 px get no tiles.
- change (session path only; the worker, `FixedSlotTracker` constants and live settings are untouched): `sample_session.session_run` applies no area cut (neither the profile camera's margins nor the surface);
  `sample_test.run_sample(..., auto_surface=None, split=False)` (defaults = the old behaviour) builds the tracker as the live auto grid does (footprint claim `FOOT_IOU`, spawned D-bays for vehicles standing on the surface at conf >= 0.45,
  sub-bays) and returns `unplaced` (vehicles that hold no bay, with the reason); `POST /api/sample-sessions/{id}/run` passes the session surface and `split=True`. The result summary says how many bays were added for vehicles off the grid
  and why other vehicles hold no bay.
- measured on the sample picture through the API (same pins and hand-placed bays as before): counted vehicles 18 -> 26 of 26 detected (conf 0.20); 11 D-bays added, the 7 placed bays keep their states.
- pre-processing measured, not added: yolov8s at 1920 px, vehicles found at conf 0.30 / 0.20: plain 22 / 27; CLAHE (clip 2) 22 / 25; CLAHE (clip 3) 22 / 28; Lanczos 2x 24 / 27; Lanczos 2x + CLAHE 25 / 29. Differences of +-2 with
  no way to tell real cars from false ones: no `enhance` option was built. Super-resolution was not tried (it invents texture and needs a heavy model).
- tests: `test_sample_test.py` +1 (off-grid car gets D1, low confidence is reported, default off), existing session tests updated (no area roi, D-bays in the slot list), vitest +1; vitest 247, tsc clean.

## 142 - Direct bay model: draw your own fixed slots (rectangle / polygon, zoom, pan) and get BUSY / FREE per slot  (2026-11-03, source only; NOT built/deployed)
- why: on low-resolution oblique pictures the SAM + GeoCalib grid is imprecise and the tight spillage layer can flag false Encroached; the PKLot bay model's own boxes are not your bays either. A hand-drawn slot map removes the geometry from the question:
  the bay model only has to say busy or free for each slot you drew.
- frontend (Sample test page only): new `DirectSlotCanvas.tsx`, shown inside the "Direct bay model (no calibration)" card behind a checkbox "Draw fixed slots" (off by default: the plain direct run is unchanged). Tools: Rectangle (drag), Polygon (click corners; first corner /
  double-click / Enter finishes, Esc cancels, Backspace undoes a corner), Select / edit (select, move, drag corners, Delete), Pan (also the middle mouse button). Zoom 100-1200 % with the wheel (about the pointer) or +/- buttons, Fit; coordinates stay shares of the picture at every zoom.
  Slots are labelled Slot 1, Slot 2 ... on the canvas. Buttons: Save slots (this browser, per picture size, localStorage), Load default slot preset (your saved slots for that size, else a starter 12-slot grid to move onto the bays), Clear all, Delete selected.
  The run button needs at least one slot in this mode. `SampleResult`: summary line for a drawn-slot run (N busy, N free, N unknown, bays the model found).
- backend (`api/sample_sessions.py`, `sample_test.py`): `POST /api/sample-sessions/{id}/direct` takes an optional `slots` list (same fraction format and `validate_slots` as every slot map, 1-200, else 422 before any model work). With it, the bay model runs once and
  `sample_test.classify_drawn` gives each slot the state of the model bay it overlaps most (IoU >= 0.25 or the bay covers >= 50 % of the slot): BUSY / FREE, else UNKNOWN. No SAM, GeoCalib, tracker, spillage, obstacle filter or blocked layer runs, so no ENCROACHED / BLOCKED can appear;
  `layers` in the answer is all false. The session stores nothing (calibration and slots stay as they were).
- untouched: the worker, Camera 1, the auto-grid and calibrated pipeline, the database, Streamlit.
- tests: pytest +3 (`classify_drawn` overlap rule; drawn slots end to end with BUSY / FREE / UNKNOWN and no layers; validation refuses bad slots before the detector runs), vitest +6 (toggle and request body, rectangle + polygon + labels, select / move / reshape / delete, zoom and wheel, pan, preset / save / clear, drawn-slot result).


## 143 - Hand-drawn slots: no more all-UNKNOWN; slot crops, a vehicle-detector fallback and a model choice that works  (2026-11-03, source only; NOT built/deployed)
- cause (read from the code and measured on the sample picture): item 142 asked the PKLot model for its own bays on the whole 640 px picture and gave each drawn slot the bay it overlapped. On this oblique view the PKLot models find (almost) no bays, so no slot overlapped anything and every slot was
  UNKNOWN, whatever bay model was chosen. The "default model" was not even in the selector (bay models only), and the model choice could not matter because the answer came from the same empty detection list.
- measured: each of 13 slot crops (10 under cars, 3 on empty ground) through `parking_yolo.pt` and the OBB model, enlarged to 640 px, conf 0.25 / 0.10: no verdict on 12-13 of 13, and the few it gave were wrong (a "free" on a car, a "busy" on empty ground). The PKLot models are bay DETECTORS trained on top-down lots,
  not busy / free classifiers; cropping gives them the same wrong view closer up. So cropping alone cannot fix this picture.
- change (`sample_test.py`, `api/sample_sessions.py`, Sample test page only): `classify_crops` runs the bay model on every slot's own crop (+30 % context, enlarged to 640 px); slots it cannot decide get the whole-picture overlap; slots still undecided are judged by the camera's vehicle detector
  (`classify_by_vehicles`: a vehicle whose foot point is in the slot or whose box covers >= 25 % of it = BUSY, none = FREE) at conf 0.20. A vehicle detector chosen as the model (the camera's own, COCO, the camera fine-tune) judges every slot that way. With drawn slots `POST .../direct` accepts any ready model
  of `GET /api/sample-sessions/models` (without slots it is still bay models only). Every slot carries `by` ("bay model, slot crop" / "bay model, whole picture" / "vehicle detector"), the summary counts them, and the detector label says when the fallback was used. UNKNOWN is only left if a detector fails (a 500 says so).
- check on the sample picture: 12 slots drawn under the 31 detected cars + 3 empty patches: 15 of 15 right with the vehicle rule (the slots were derived from the same detector, so this checks the geometry, not the detector).
- frontend: in "Draw fixed slots" mode the model list is every ready model (the bay model stays the default); switching the mode off returns to bay models only; each slot row shows who decided it.
- tests: pytest +2 (`classify_crops`, `classify_by_vehicles`), the drawn-slot API test rewritten (stages, `by`, no UNKNOWN, default model, plain run still bay-only), vitest +1 (model choice sent / reset). Camera 1, worker, calibrated pipeline, DB, Streamlit untouched.

## 144 - Sample test: slot edit / delete, closed panels by default, empty slot 11 no longer BUSY  (2026-11-03, source only; NOT built/deployed)
- slot 11 false BUSY (cause, from the code): the vehicle fallback of item 143 called a slot BUSY when a vehicle's FOOT POINT was in it or the vehicle's whole BOX covered >= 25 % of it, and every vehicle could make several slots busy. On an angled view a car's box is mostly body
  above the ground it stands on, so a car in slot 10 (or behind slot 11) leaned its box over the empty slot 11 and made it BUSY. A weak "busy" from a PKLot bay model on empty ground (measured: 0.16-0.20 confidence) could do the same.
- fix (`sample_test.classify_by_vehicles`, `api/sample_sessions.py`): only the lower 40 % of a vehicle's box (its wheel footprint) is compared with the slots, and each vehicle is given to ONE slot, the one holding most of its footprint (>= 25 % of it, or a slot it covers >= 25 % of); all drawn slots compete for each
  vehicle, so a car between slot 10 and 11 makes only one busy. A bay model's verdict (slot crop or whole picture) now counts only from confidence 0.40 (`BAY_TRUST`); below it the slot goes to the vehicle detector. Not verified on your exact picture (I do not have your slot 11): tests reproduce the leaning-box and the weak-bay cases.
- slot edit / delete (`DirectSlotCanvas.tsx`): a list of the drawn slots with Edit (selects it and switches to the edit tool) and Delete per row; the corner handles of the selected slot show and drag in every tool, not only Select; a rectangle's corner resizes it as a rectangle (kept a rectangle, flips about the fixed corner);
  polygons still move corner by corner; moving a whole slot and Delete / the Delete selected button stay.
- panels: "Direct bay model" and "GeoCalib + SAM (calibrated grid)" are collapsible sections, both CLOSED by default (`fold-*` in `SampleSession.tsx`); their content stays mounted while closed, so a drawing or a half-done calibration is not lost. Results stay below.
- tests: pytest +1 and `classify_by_vehicles` rewritten (leaning box stays FREE, one car = one slot); vitest +4 (closed by default and independent, state kept when closed, list Edit / Delete, rectangle resize incl. past the opposite corner), 2 existing tests open the panel first. vitest 259, tsc clean. Camera 1, worker, DB, Streamlit untouched.

## 145 - Sample test step 2: hand-drawn surface corners lead clearly on to Calibrate  (2026-11-03, source only; NOT built/deployed)
- report: after "Draw corners" the step-3 button "Calibrate and lay bays" did not seem to wake up, unlike after SAM. Checked the handler chain: `click` appends corners, `reason()` gates the button on (3+ corners) and (the scale: 2 ground points + a typed distance, or 4 pins + 2 sides), identical for a SAM polygon and a drawn one; the calibrate request
  carries `polygon` either way. A vitest that draws 2 points, types the distance and draws 4 corners shows the button enabled and the request sent: I found no event-handler fault, so what follows removes what makes the flow look dead.
- what was missing: (1) nothing told you the surface was complete (SAM prints "outlined ... go on to Calibrate"); (2) the reason the button stays off was a small grey line ("To do: ..."); (3) a double-click added the same corner twice; (4) three corners in a line passed the 3-corner check and went to the server;
  (5) no undo for a single corner.
- change (`SampleSession.tsx`, Sample test page only): after 3+ corners a green "Surface drawn (N corners)" line with a **Done: go on to the scale** button (switches to the missing scale step, so the next click is a ground point, not another corner); the disabled reason is a highlighted box
  ("The button wakes up when this is done: ..."); the second click of a double-click is ignored; corners enclosing almost no area are refused with the reason; **Undo corner** next to Clear.
- tests: vitest +3 (GeoCalib + hand-drawn corners calibrate and send the polygon; double-click / undo / flat outline; Done moves to the scale step). Backend untouched.

## 146 - "the surface polygon has no measurable ground": say why, and do not fail when the outline lies over the parked cars  (2026-11-03, source only; NOT built/deployed)
- where it comes from: `geometry/pipeline.build_record` -> `bev.surface_grid` returned None. Four things empty the raster: the outline's pixels at / above the horizon (a wrong tilt or scale), ground beyond 60 m or coarser than 25 cm per pixel, strips thinner than 1.2 m or pieces under 3 m2,
  and every detected vehicle's box being CUT OUT of the outline before it is measured (a car body is not ground). An outline drawn over a packed row of cars (typical for hand-drawn corners; SAM leaves the cars out itself) is cut into slivers by the boxes and nothing is left.
- change (setup/geometry path only, error and fallback; a record that built before builds identically): (1) if the grid is empty after the vehicles are cut out, `build_record` measures the outline as drawn (the cars' footprints are laid back as before) and adds the warning
  "the outline lies mostly over parked vehicles ..."; (2) if it is still empty, the error keeps its first sentence and adds the shares of the outline's pixels that are at/above the horizon, too far, too coarse, measurable, plus which way to look (scale/tilt vs slivers). New `bev.explain_surface` (diagnostic only).
- tests: `test_geometry_pipeline.py` +2 (outline over cars is measured with the warning and a normal outline has none; the error names the horizon share for a sky outline and the sliver advice for a thin one). Frontend and the live worker untouched.
- not known: which of the four caused it on your picture (I have only the message). If it still fails the new message says which: mostly "above the horizon / far" = check the typed distance and the two ground points; mostly "slivers" = draw the outline around the open ground.

## 147 - Sample test: the Direct bay output and the SAM + GeoCalib output are never on screen together  (2026-11-03, source only; NOT built/deployed)
- before: both results were kept and drawn one under the other (the calibrated panel, then the direct one), so after trying both you saw two result panels with different slot sets.
- now (`SampleSession.tsx`): a `shown` state says whose result is displayed = the test run last pressed. A Direct bay / fixed-slots run shows only the direct result (its own drawn or model slots, BUSY / FREE); a calibrated Run test shows only the calibrated grid result. The other result is kept in memory
  but not rendered, and comes back only by running that test again; Start over / a new picture clears both. The calibrated grid editing canvas lives in the closed-by-default GeoCalib + SAM panel and the drawing canvas in the Direct panel, so their overlays do not share one picture either.
- tests: the comparison test now checks one result panel at a time in both orders. vitest 262, tsc clean. Backend untouched.

## 148 - Sample test: Spillage protection, Obstacle filter and Blocked by object start unchecked  (2026-11-03, source only; NOT built/deployed)
- before: a session took the three switches from the profile camera's live options (Camera 1 runs Blocked by object, so it started ticked) and a `run` without `layers` did the same.
- now (`api/sample_sessions.py`): a session starts with all three off whatever the live camera runs; the page shows them unchecked and a run applies exactly the boxes you tick (a request's `layers` still overrides). The live options are no longer read by the sample test at all.
- tests: pytest test renamed/rewritten (live spillage on, sample off, request turns it on); vitest +1 (all three unchecked) and the run-body expectations updated. Frontend otherwise unchanged.

## 149 - Sample test: zoom and pan on the GeoCalib + SAM calibration picture  (2026-11-03, source only; NOT built/deployed)
- why: on oblique or low-resolution pictures the 4 ground pins, the two ground points, the surface corners and the SAM clicks need to sit exactly on the paint lines; at 100 % a click is several metres of ground.
- change (frontend, Sample test page only): the calibration picture (panel "GeoCalib + SAM") has the same controls as the Fixed slots drawing canvas: wheel zoom about the pointer (100-1200 %), zoom in / out buttons, Fit, a % readout, a **Pan** tool (drag the picture instead of placing points) and the middle mouse button
  to pan in every mode. Pins, ground points, corners, SAM clicks, bays and their labels keep their size on screen while zoomed, and a click is turned into a share of the picture through the zoomed layer, so points land where the pointer is at any zoom.
  New shared hook `useZoomPan.ts`; `DirectSlotCanvas.tsx` now uses it too (same behaviour, one implementation).
- tests: vitest +2 (zoom buttons / wheel / fit and constant mark size; Pan tool and middle button pan without placing a point, a plain click still places one), the Direct canvas pan test scoped to its own canvas. vitest 265, tsc clean. Backend untouched.

## 150 - Sample test: Row zones (one mapping per parking row) with sequential bays, run as one occupancy list  (2026-11-03, source only; NOT built/deployed)
- why (read from the code): the auto grid keeps ONE record / ONE ground plane / ONE bay lattice per session (`build_record` + `pack_bays`, one global angle, anchored on cars). Rows at another angle, slope or depth get tiles that skew, overlap or spill into the driveway.
  GeoCalib is not the cause and running it per zone cannot fix it: there is one camera and one lens.
- change: a "Layout" switch in the GeoCalib + SAM panel: **Auto grid** (unchanged) | **Row zones**. Each zone = one parking row:
  * the 4 ground corners of its first bay (B1, 1->2 along the row, 2->3 its depth) and its size (W x L, gap) fix the row's own plane-to-image homography (`frontend/src/rowzones.ts`, pure, 8x8 solve);
  * bay i = H applied to [i(W+g), i(W+g)+W] x [0, L]: with gap 0 neighbours share their edge exactly (no gap, no overlap) and a homography keeps the order along the row (no jump, no split);
  * "Add next 1 / 2 / N", "Remove last", per-bay delete (holes keep the numbering), Reverse;
  * "Fit to row end": the far edge of the last bay (2 points) re-solves H between two measured edges, so a click error no longer grows along the row (3 px error: bay 10 within 3 px instead of > 15 px);
  * stop rules said on screen (beyond the horizon, leaves the picture, too small); cross-zone overlap (IoU > 0.10) flagged in red.
  Rendering: inactive zones dimmed and not clickable, the active zone on top with its draggable corners / end points (live regeneration); a click on another row's bay selects that row. Zoom / pan of item 149 apply.
- server (`api/sample_sessions.py`): `PUT /api/sample-sessions/{id}/zones` {zones, slots}: validates (ids, B1 fractions, bay ranges, count 1..60, slot ids = zone id x 100 + n via `validate_slots`), checks each zone's B1 as ground pins with
  `camera_from_rectangle` (reprojection px, camera height, hfov; > 4 px or a PinsError = a warning on that zone only). `POST /run` takes `layout` ("auto" | "zones"); in zone mode it needs no calibration, runs ONE `run_sample`
  on every zone's bays (a vehicle is matched once against all of them), with spillage off (no single ground plane), no spawned D-bays / sub-bays; each result slot carries its zone name. A calibration sets the layout back to auto.
- tests: vitest +9 (`rowzones.test.ts`: homography, true-perspective reproduction, shared edges / order, gap / reverse, row-end drift, removed bays, stop rules, overlap) +4 (UI: sequence and independent zones, drag / row end / overlap, save + run as one list, auto unchanged);
  2 existing run-body expectations gained `layout: "auto"`. pytest +2 (validation and per-zone pin check; one-engine run without calibration, layout switch, nothing live moved). vitest 278, tsc clean.
- real check: `data/sample/image.png` (640x480, near top-down), a 19-bay row from B1 + row end: every bay on one car or one empty place, A6 / A7 on the empty places; a second row from B1 alone drifted about one bay over six (why the row end exists). The 1672x941 picture was not available here.
- Camera 1, worker, `build_record` / `pack_bays`, DB, Streamlit untouched.

## 151 - Sample test: zones redone as SAM + GeoCalib PER ZONE (item 150's manual B1 homography removed)  (2026-11-03, source only; NOT built/deployed)
- asked: the B1-based rows failed for the user ("Zone A stops at A2", broken generation); each zone must instead be a drawn polygon in which the backend runs SAM and GeoCalib separately and lays its grid automatically,
  as the working single-surface mode does; no manual sizing or stop rules; all zones' bays merged into one run without duplicates.
- removed: `frontend/src/rowzones.ts` (+ tests), the B1 / width / length / Add next / row end UI, `PUT /zones` and its B1 pin check.
- now (GeoCalib + SAM panel, Layout "Zones (SAM + GeoCalib per zone)"; step 1 Scale is shared: GeoCalib + a typed distance, or the ground pins):
  * per zone: draw its outline -> optional "Find the ground with SAM" (`POST /surface` with `zone` + `bound`: SAM's mask is cut to the outline, also after the car bridges; a zone may be 0.2 % of the picture;
    the session's own surface is not replaced) -> "Calibrate and lay bays" (`POST /calibrate` with `zone`, `zone_name`): GeoCalib on that zone's own mask (one setup-service call per zone), `build_record` / `grid_slots` on that zone alone,
    its bays renumbered zone x 100 + n and named A1, B1 ... (max 99 per zone, the rest said in a warning). The whole-picture grid of the session is left as it is. `DELETE /zones/{z}` forgets a zone.
  * merged engine (`sample_session.merge_zones`): all zones' bays in one list; a later zone's bay with IoU >= 0.3 to an earlier zone's bay is left out and listed ("B1 = A2"); a bay the user removes (`skip`) still owns its place,
    so its twin does not come back. `POST /run` with `layout: "zones"`: ONE `run_sample` on that list (a vehicle is matched once), the auto grid's footprint claim on, no spawned D-bays (they would fall outside every zone), no sub-bays;
    spillage measures with the first zone's calibrated camera. Result slots carry their zone's name.
  * canvas: each zone's outline (dashed), SAM surface (filled), its bays in the zone's colour; inactive zones dimmed; a click on another zone's bay selects it. Zoom / pan of item 149 apply; pins / reference points show in both layouts.
- tests: pytest +3 (two zones -> two GeoCalib calls with different masks, own grids, auto grid untouched, delete; SAM cut to the zone outline, surface not replaced; merged run: duplicates once, skip without resurrecting the twin,
  spillage with ground metric, 409 without zones / without auto grid, 422 bad layout, nothing live moved) replacing item 150's 2; vitest +3 replacing item 150's 13 (per-zone SAM + GeoCalib requests and merged list,
  run with skip + zone delete, whole-picture layout unchanged). vitest 268, tsc clean.
- found while testing: removing a bay used to let its duplicate from the overlapping zone back into the run (fixed in `merge_zones`).
- not checked here: the real SAM / GeoCalib per zone on the user's 1672x941 picture (the setup service is part of the live stack and was not started; the picture is not in data/sample).

## 152 - Sample test zones: bays follow the zone's row, "Add next 1 / 2 / N" back, bays editable  (2026-11-03, source only; NOT built/deployed)
- cause of the skewed / out-of-order zone bays (read from the code, reproduced in a test): a zone's calibration used EVERY car of the picture, so the bay angle came from the cars of other rows (`bay_angle_from_cars`),
  their footprints were added to the zone's ground and anchored the lattice (`pack_bays`): a bay could stand across the row or outside the zone (measured: A2 rotated 90 deg, A3 beyond the zone).
  The bays were also numbered "front to back" of the whole picture, not along the row.
- fix:
  * only the zone's own vehicles (box bottom-centre in the drawn outline, `sample_session.in_zone`) reach its calibration;
  * `build_record(row_outline=...)`: the orientation comes from the drawn outline on the ground (bays across its longer side, source "zone outline");
  * `sample_session.row_grid`: a zone's bays are laid straight from its outline on the ground: its enclosing rectangle's long side is the row, cut into whole bays (~ bay width, stretched / squeezed by at most half a bay so they
    fill the zone's length and share their lines), its depth into whole rows (~ bay length). No car anchors, no lattice. (The whole-picture layout still uses `grid_slots` unchanged.)
  * `order_along_row`: A1, A2 ... along the row, row after row across the zone.
- "Add next 1 / 2 / N" (per calibrated zone): `POST /api/sample-sessions/{id}/zones/{z}/extend {count}` steps new bays on the ground after the last one, by the last two bays' centre distance (one bay: its width along the row), with the zone's
  own camera; stops only where a bay would be wholly outside the picture (and says "k of n added"). Names re-read along the row.
- editing: "Edit bays" / a row's Edit selects a bay; drag it to move, drag its corners to reshape (as the whole-picture review). Edits are saved (`PUT .../zones/{z}/slots`, validated, ids must stay the zone's) before
  an extension and before the run. Delete still removes a bay from the run.
- tests: pytest +4 (zone oriented by its outline and blind to other rows' cars, 4 bays of 2.5 x 5 m inside a 10 m zone; row-by-row order; extension at the same 2.5 m spacing, stop at the picture's edge; extend / edit endpoints and their 404 / 422),
  vitest +2 (outline sent, add next 2 / N; select, move, reshape, saved before the run). vitest 270, tsc clean.
- not checked: the user's own picture with the real SAM / GeoCalib (setup service not started here).

## 153 - Sample test zones: bay width no longer squeezed / turned (any resolution); Dynamic slots switch (off by default)  (2026-11-03, source only; NOT built/deployed)
- reported: on 1672x941 pictures a zone's bays were too narrow, two of them on one car.
- investigated: the scale path is resolution-consistent (reference points are fractions x the frame's own w, h; GeoCalib's f is at the picture's own size; the typed distance sets the camera height and wins over every prior);
  a synthetic 1672x941 camera now gives the same 2.5 x 5 m bays as 1920x1080 (test). The narrow / split bays came from item 152's `row_grid`:
  * the row direction was the outline's longer side ON THE GROUND. An outline drawn over the cars has its top edge on their roofs, which lands metres behind them on the ground, so on an oblique view the "depth" outgrew the row
    and the bays were laid across the cars (2.5 m slices through each car);
  * `round(length / width)` stretched or squeezed the bays by up to half a bay (a 3.7 m zone became two 1.85 m bays).
- fix (`sample_session.row_grid`):
  * the row direction = the outline's longer side IN THE PICTURE (a zone is drawn along its row), measured on the ground at the outline's lower part;
  * bays are the typed size (2.5 x 5 m by default), in `rows` rows (new per-zone setting, default 1) from the outline's front edge, away from the camera; along the row they are fitted to the zone's length only within +-12 %,
    otherwise exactly the bay width (never squeezed);
  * with 3+ cars of the zone along the row, their own spacing (folded: a gap of k bays = k pitches, accepted at 0.7-1.4 x the bay width) is the bay width and the bay lines fall halfway between the cars, so one car = one bay even
    when the typed scale is a few % off. The calibration note says the size and where it came from ("Bays 2.5 x 5 m (bay width)" / "(the zone's cars)").
- Dynamic slots switch (both run cards, default OFF): off = the run judges exactly the fixed / configured bays (no bay for a vehicle off the layout, no sub-bays, no footprint claim); on = the live auto grid's
  vehicle-driven rules as before (whole picture: spawned D-bays + sub-bays + footprint claim; zones: footprint claim + sub-bays). `POST /run {dynamic}`; the result says which. Sample test only: Camera 1's live auto grid is unchanged.
- tests: pytest +2 (same metric bays at 1672x941 and 1920x1080 with an outline deeper than the row; a 3.7 m zone = one 2.5 m bay; cars 3 m apart set a 3 m pitch with each car mid-bay (measured 3.13 m: the box bottom centre is a
  few % off the wheel line); 2 rows) and the run test now checks dynamic off = fixed bays only, on = a D-bay for the car off the grid. vitest +1 (dynamic off by default, sent when on; rows sent), run-body expectations gained `dynamic: false`.
  vitest 271, tsc clean.
- not checked: the user's own 1672x941 picture with the real GeoCalib (setup service not started here).

## 154 - Sample test zones: one car = one bay on the two sample pictures (high and low resolution); bays laid from the zone's cars  (2026-11-03, source only; NOT built/deployed)
- reported: data/sample/aftercalibaration.png (image2.png, 1672x941): the zone got 16 narrow bays for ~7 cars.
- measured on the real pipeline (a scratch copy of the setup image `cp-setup-scratch` on its own network, the live `carparking-setup-1` untouched; yolov8s conf 0.20; the user's zone and reference read off the screenshot;
  low-res = "Authorities barred from collecting parking fee.png", 640x480, which is image2 resized x 0.385 / y 0.459, i.e. out of aspect):
  * GeoCalib's focal length scales exactly with the picture: f / width = 0.817, 0.825, 0.833 at 1672, 836, 2508 px wide; 0.950 / 0.960 at 640 / 1280 (tilt and roll the same within 0.6 deg). No resolution bug in f, the
    reference points (fractions x the frame's own size) or the metres.
  * the scale sources disagree by 2x: camera height from the typed 2.6 m wheelbase 9.3 m (hi) / 11.0 m (lo), from the cars 4.9 / 5.2 m, from depth 3.0 / 4.0 m. Roll (-5.4 deg) and tilt (13 deg) change the far end's metres a lot:
    the cars' feet along the row came out 2.8 m apart at the near end and 7-10 m at the far end at roll -5.4, 4-6 m at roll 0. In the picture itself the cars are 108-139 px apart all along the row. No single camera fits these
    pictures, so any metric grid puts several bays on the far cars: reproduced exactly, 16 metric bays of 2.46 m (hi) and 12 (lo) in the user's zone.
- fix (`sample_session.car_row_grid`, used by `row_grid` when 2+ cars stand in the zone; the metric grid stays the fallback):
  * the outline's front and back edges are the row's; its short sides meet at the vanishing point of the bay lines;
  * each car's position (`car_position`: box centre across, a quarter of its height above its bottom) is carried along those lines onto the front edge (t: 0..1 of the edge, a ratio: the same at any resolution);
  * `anchored_edges`: the bays start at the outline's first corner and are as wide as the neighbouring cars are apart there (interpolated along the row; one car seen twice = one car; a gap of 1.8+ spacings = that many empty bays).
    A spacing is a difference of two car positions, so a car position's own offset cancels; no f, h, tilt, roll or pixel size enters.
  * "Add next" continues such bays in the picture (each next bay shares the last one's far line, width x the last two bays' ratio, at most +-15 %); metric zones still extend in metres. The calibration note says
    "about X m wide (fitted to the zone's N cars in the picture)".
  * removed: item 153's metric car-pitch folding (`car_pitch`): it folded the far 2x gaps back to 2.5 m and so kept the split.
- result on both pictures, the user's zone: 7 bays for the 7 cars (SUV, silver van, white car half hidden behind it, white van, hatchback, red car, silver car), each car's front at the curb inside its own bay; the low-res
  bays are the high-res ones resized within a quarter of a bay. (A bay line drawn on the ground looks as if it crossed the nearer car's body on this angled view: the ground between two cars is hidden behind the nearer one.)
  Overlays: scratchpad z_grid_image.png / z_grid_Autho.png (new), z_metric_image.png (the old 16).
- tests: pytest +3 with the two pictures' real detections and the user's zone as fixtures (7 = 7 at both resolutions, same bays resized; bias cancels / duplicates / empty bays / shrinking spacing; Add next lands on the next bay
  within a third of a bay); the synthetic spacing test now checks each car in a bay of its own. vitest unchanged (271), tsc clean.

## 155 - Sample test zones: bays never far too wide or too narrow: the zone's own metric frame (outline + focal length + reference) checks the cars  (2026-11-03, source only; NOT built/deployed by me)
- reported after item 154 was deployed by the user: zone bays "way too wide". The screenshot was not in the workspace; the weak point of item 154 found by test: the bay width was the cars' MEDIAN spacing, so a zone whose
  cars are sparse (empty bays, cars under a canopy missed by the detector, 2 cars far apart) got bays 2-4 bays wide. With only 2 cars there was no way to tell.
- measured (image2.png + its low-res copy, the user's zone): GeoCalib's f is right and resolution-consistent (item 154); the zone outline as a ground rectangle with that f gives the zone's proportions without the camera's
  tilt / roll / height (the wrong ones): length / depth = |K^-1 h1| / |K^-1 h2| = 5.14 (hi) / 5.15 (lo); the typed reference through the rectified frame then gives 25.4 x 5.1 m (hi) and 23.4 x 5.2 m (lo):
  the depth is one bay length, the length ~1.35x the 7 cars' row (the outline's far corner was cut by the picture's edge, 15 deg off square).
- fix (`sample_session.ZoneFrame`, `zone_frame`, `frame_grid`; `anchored_edges(ts, bay_t)`; `row_grid(..., reference)`):
  * a gap between two cars holds empty bays only when the frame makes it at least MULTIPLE = 2.5 bays (2.0 split one car of image2; 2.5 split none), split into round(gap / frame bay); two detections closer than 0.45 of a frame
    bay are one car;
  * a zone with fewer than 2 cars is cut by the frame alone: round(L / 2.5) equal bays in the rectified frame (the row's true perspective), `rows` rows across;
  * the camera (GeoCalib) grid is only the last resort (no reference).
  * `build_record(min_surface_m2=...)`: a zone passes 0 (a row drawn over its parked cars left 5.2 m2 of open ground on the low-res picture and the zone was refused "too small to be a parking lot"); defaults unchanged.
  * the API gives the zone's reference (the typed distance, or pins 1-2 and side 1) to row_grid.
- result, both pictures, the user's zone: all cars 7 bays (= 7 cars); every other car 8 / 10; first + last car only 10 / 9; no cars 10 / 10; bay widths 2.4-2.9 m in every case (before: 16 / 12 metric, and 2-4x wide with sparse cars).
- Camera 1 live is not changed: the worker's auto grid calls build_record with the old defaults (row_outline None, min_surface 20 m2); geometry pipeline tests green.
- tests: pytest +3 (frame alike at both resolutions; every car-count case 7-10 bays of 2.2-3.2 m and 7 with all cars; frame_grid widths / rows). geometry + session suites 64 passed.
- (155, cont.) generic rule for sparse cars: with the zone frame, when some cars stand side by side (a gap < ADJACENT = 1.5 frame bays) other gaps hold empty bays only from MULTIPLE = 2.5; when none do (a car in every
  other bay, missed detections) every gap is cut into round(gap / frame bay) bays (with 2.5 alone, a right frame and a car in every other bay gave double-wide bays).
  New synthetic test with known truth: a 6-bay row (15 x 5 m) at 20 / 45 / 80 deg tilt, 640-2560 px, no cars / all / every other / the two end cars: 6 bays of 2.5 x 5 m (+-0.3) in all 16 cases. The two sample pictures
  unchanged (all cars 7 = 7; 2.4-2.9 m otherwise). Live-path regression test: build_record without the zone options is byte-for-byte the record with their defaults, a small lot still refused.
- the first full build of 155 ran 42 min (I ran pytest in parallel) and 4 Camera-1 setup-API tests hit their 10 s job timeout; the same three files fail 6 tests on the OLD code (item 154 image) and 4 on the new one
  under the live worker's ~74 % CPU: load-dependent timeouts, not a regression (one of them took 20 s on the old code, 6.8 s on the new).

## 156 - Dynamic bays laid along the zone's rows (Sample test, dynamic mode only)
- Problem: a dynamic bay (D1, D2 ... a vehicle off the grid) was the vehicle's upright footprint rectangle (`footprint_poly`), so its sides stood straight up while the zone's rows slant with the perspective.
- `sample_session.DynamicBays(zones, w, h)`: for a vehicle standing in a zone (ground point inside the outline, 3 % margin) the bay is cut by the zone's separator lines (through the vanishing point of the outline's short
  sides, so left / right slant exactly as the zone's do), its front line is parallel to the zone's front edge through the box bottom (the tyres), its back runs to the zone's back edge along the same separators. The width is
  the car's own box width (0.85 of it, or the zone frame's bay width at that place when the zone has a reference), kept between 0.5 and 1.0 of the box width: cars set the width, the zone's ground perspective the angle.
  A vehicle outside every zone gets None: no bay (a sub-bay keeps the old rectangle).
- `FixedSlotTracker.spawn_shape` (default None = exactly the old footprint rectangle), `sample_test.run_sample(spawn_shape=None)`; only the zone run with Dynamic ON passes it. The surface for D-bays in a zone run is the hull of
  the zones' outlines (before: [], no D-bays at all). The zone keeps its reference ("reference") so the run can rebuild its frame. Fixed mode (Dynamic OFF), the whole-picture run and Camera 1's worker are unchanged.
- tests: pytest +3 (sample pictures at both resolutions, with and without frame: front parallel to the zone front, both sides point to the separators' vanishing point, car's tyres inside, width <= car; None outside a zone;
  the tracker default is the old rectangle). session / sessions / fixed_slots / sample_test / geometry / parity suites green.
- (156, cont.) the first version was only unit-tested; run on the real pictures (scratch container, 3 fixed bays kept so the other cars get D-bays) it showed the bay centred on the box bottom (the car's nearest corner),
  i.e. shifted off its car, and two D-bays for one van (two duplicate boxes). Now the bay is centred on the car's footprint point (car_position) carried along the separators onto the front edge and cut exactly like the
  zone's own bays (front edge to back edge, same vanishing point), and a vehicle whose shaped bay overlaps a bay already there (IoU >= 0.5) gets none. Overlays checked on image2.png and the low-res copy: the D-bays
  continue the B-bays' lattice. Checked with the script in the scratchpad (dyn.py), not shipped.

## 157 - Item 156 rolled back (dynamic shape generator removed)
- User: the zones' dynamic shape generator gave far cars the same-size / uniform bays, against the camera's perspective. Removed completely: `sample_session.DynamicBays`, `FixedSlotTracker.spawn_shape` / `_shaped`,
  `run_sample(spawn_shape)`, the zone's stored "reference", the item-156 tests. Dynamic bays are again the vehicle's own footprint (its box, so a far car's bay is small, a near one's large), and a zone run with Dynamic ON
  passes a surface of [] as before (sub-bays / footprint claim only). Fixed mode, zones, Add next, edit and Camera 1 untouched. A copy of the source before this rollback: backups/pre-rollback-zones-20261104-104553.

## 158 - Zone bays shrink with distance even when laid from the cars; Dynamic works in zones (Sample test)
- Measured on image2.png / the low-res copy (all 7 cars): the zone's bay widths near->far were [77, 67, 51, 56, 81, 73, 51] (a far bay wider than a near one: bays took the cars' uneven spacing); with no cars they were already
  71 -> 28 (perspective). `_perspective_edges` (car_row_grid): the bays' edges, which the cars place, are put on the zone's ground lattice: the outline's four corners give the unit-square homography (no camera model),
  equal steps of its u between the first and last edge = equal bays on the ground. Now [100, 84, 71, 61, 53, 46, 41] / [110, 90, 75, 64, 54, 47, 33]: steadily narrower, the separators converge on the vanishing point.
  A row at one depth keeps equal bays (synthetic test). Camera 1's worker / auto grid do not call this code.
- Dynamic slots in a zone run: ON gives the vehicle-driven rules a surface (hull of the zones' outlines) again, so a car standing in a zone with no bay gets a D-bay from its own footprint (its box: small far, large near,
  as the live auto grid); OFF = the fixed bays only. The toggle (item 153) is on both layouts. Item 156's shape generator stays removed.
- tests: pytest +1 (bays shrink steadily from near to far at both resolutions, one-depth row stays equal); two bounds relaxed for the true geometry (hi / lo bay agreement 1/3 bay; all-cars width <= 3.7 m metric, image2's
  frame is 1.35x long). session / sessions / fixed_slots / sample_test / geometry / parity suites: 152 passed.

## 159 - Sample test on data/sample/camera1.png; "Whole picture" removed from the view; zone outline corner pairing fixed
- Test on camera1.png (1915 x 1079, Camera 1's Front Parking, cars nose-in along the paving), real pipeline in scratch containers (YOLOv8s, GeoCalib in a scratch setup container on its own network; live stack untouched).
  Hand-drawn row outline (700,830)(1075,215)(790,215)(20,800), reference the SUV's wheelbase 2.6 m: GeoCalib f 2184 px, pitch 12.5 deg, h 3.1 m, 9 bays whose front widths run 168, 127, 99, 79, 65, 54, 46, 39, 34 px
  (steadily narrower, separators converge on one vanishing point). Dynamic ON with 3 fixed bays: the other cars got D1 / D2 from their own footprint, the far one smaller than the near one; OFF: the fixed bays only.
- Bug found by that test: `_outline_sides` paired the front's corners with the back's by distance; on a thin, strongly foreshortened outline it picked the wrong diagonal (vanishing point inside the picture, bow-tie bay, 1 bay
  of 650 px). Now the back corners are the ones next to the front's across the outline (its own cycle); a bow-tie click order is put in order first. Test added. (Sample zone code only: Camera 1's worker does not use it.)
- Sample view: the "Whole picture" layout (pill, its parking-surface / calibrate / review cards, canvas layer, handlers, state) is removed: a picture is laid out by zones only, the first zone (Zone A) is ready after the picture
  loads, run sends layout "zones". The Dynamic slots switch stays (zones). Backend endpoints unchanged. vitest: whole-picture tests deleted, zone tests no longer click a layout pill, +1 test (no Whole picture option); 256 passed.
- Limits seen on camera1.png: with the typed wheelbase picked by eye the zone's depth measured 3.3 m (a bay about 1.8 m wide), the far cars bunch near the row's end (t 0.95-0.99), so one bay per car does not hold there.

## 160 - Sample test: "Select zone & slot adjustment" (frontend only)
- Backend and the zone generation (perspective, dynamic) untouched. New `frontend/src/bayfine.ts` (pure functions on fractions, sizes / angles in pixels): `shiftBays` (the chosen bays move together, stopped at the picture's
  edge), `resizeBay` (about the bay's centre along the bay's own front and side directions, so its slant is kept), `rotateBay`, `tuneBays`.
- SampleSession (Edit bays): a "Select zone & slot adjustment" box. Target: the selected bay or Whole <zone>; step 1 / 2 / 5 / 10 px; arrow buttons and the Up / Down / Left / Right keys (Shift = 5x; ignored while typing in a
  box); Narrower / Wider / Shorter / Longer (2 % each), Shrink 2 % (all round, to keep a bay off its neighbour's car) and turn 0.5 deg either way. Edits go through the same path as dragging: kept in the zone and the merged list,
  the zone marked edited, saved (PUT /zones/{z}/slots) before the next bays are added and before the run.
- tests: vitest +5 (4 for bayfine.ts, 1 for the panel: keys, buttons, bay vs whole zone, typing ignored, saved before run). tsc clean; 261 passed. Not deployed.

## 161 - Sample test: "Top-down (aerial)" scale method (GeoCalib reads overhead pictures as nearly level)
- Cause: on data/sample/Aerial View of a Busy Car Park.png (a drone shot looking nearly straight down) GeoCalib has no horizon, no vertical edges and almost no convergence to read gravity from (focal length and tilt cannot
  be told apart in a near-orthographic view): it answered a pitch of -14.8 deg, and `build_record` rightly refused a camera that looks level (MIN_PITCH 3 deg; `geometry/pipeline.py`, unchanged). The threshold is not
  the fault: the pitch value is wrong, so lowering it would not help.
- New third scale method, chosen by the user (never an automatic fallback: on an oblique picture that would silently give wrong metres): `ss.topdown_record` models a nadir camera (pitch 89.5 deg, no roll; focal
  length set from the typed reference for a nominal 20 m height, so the scale is exactly the reference's) and goes through the same `build_record` as the other methods. API: `method: "topdown"` (needs `reference`, no setup
  service); a GeoCalib job that fails with "too level for ground" now ends "...; for an aerial / drone picture looking straight down choose the Top-down scale method" (sample API only, pipeline text unchanged).
  UI: a "Top-down (aerial)" pill in the Scale card (no lens choice, no setup-service remark).
- Real check (YOLOv8s, scratch container): zone along one column of the aerial picture, reference one 2.5 m bay: 18 bays from 10 detected cars at 2.4-2.5 m x 6.2 m, 17 bays at 2.54-2.64 m with no cars.
- tests: pytest +3 (topdown_record: 10 bays of 2.5 x 5 m everywhere, same-point reference refused; API top-down needs no service; level GeoCalib answer carries the hint), vitest +2. Not deployed. Not touched: pipeline.py,
  reanchor.py, Camera 1, GeoCalib / pins methods, zone generation, dynamic slots, fine adjustment.

## 162 - Sample test: a zone two cars deep gets two rows automatically; Split bay (aerial picture)
- Cause (reproduced on data/sample/Aerial View of a Busy Car Park.png, YOLOv8s, top-down scale): a zone drawn over two columns of cars side by side is ~11 m deep, and a zone is laid as ONE row of bays (rows = 1), so each bay
  was 11 m long and spanned both cars. The scale (2.4-3 m along the row, from the reference) was right; the depth was not divided.
- Fix 1: `rows` 0 = Auto (new default in the UI; the API's own default stays 1): `ss.auto_rows` = round(the zone frame's depth / bay length), 1..6, 1 without a reference. The aerial zone: 28 bays in 2 rows of 5.5 m
  (14 along), one cell per car; the sample pictures (depth ~5 m) stay one row (the select still offers 1-6). The Rows box shows "Auto".
- Fix 2: "Split the selected bay" in Select zone & slot adjustment: `bayfine.splitBay` cuts a bay into two by the square-to-quad homography of its corners, so the cut is equal on the ground, left | right or front / back; the first
  part keeps the id and gets the name + "a", the second takes the zone's next free id and + "b"; saved (PUT /zones/{z}/slots) before the next bays and the run.
- tests: pytest +1 (two rows for a deep zone, one for a one-car-deep zone, bays one car long), vitest +4 (splitBay x3, the split in the panel and its save). Backend 168 passed, vitest 267 passed. Not deployed.

## 163 — Sample test: far cars hold their thin bay; Edit bays Undo and moves along the bay
- **Occupancy:** the point rules (`match_score`) take a car's foot point; a small far car's point lands just past its thin bay (more than 15 % of the box height), so the bay stayed FREE although the car had a box. New last-resort pass `FixedSlotTracker._cover_pass` (`fixed_slots.py`): a vehicle no bay claimed takes the free bay its box covers >= `COVER_MIN` (0.6) with the bay's centre inside the box, nearest to the box bottom first (a bay behind a car is not taken for it). Off by default (`cover_fallback = False`); only `run_sample` turns it on, so the live trackers and Camera 1 are unchanged. Test `test_cover_pass_gives_a_far_car_its_thin_bay`.
- **Edit bays:** `Undo` button (and Ctrl+Z) takes back the last bay edit (drag, corner drag, arrows, size, turn, split), 50 deep; the restored zone is saved again before the run. New moves along the bay's own directions (`shiftAlong` in `bayfine.ts`): Row ◂ / ▸, Back ↥ / Front ↧. The screen-direction arrows (←↑↓→) were already there.
- Not deployed. Verified: related backend suites 138 passed, tsc clean, vitest 269/269. The user's own screenshot (image_284b1d.png) is an overlay, not a runnable input, so the occupancy fix is checked on a synthetic case only.
