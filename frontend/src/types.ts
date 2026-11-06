// Shapes of the FastAPI backend's JSON (api/main.py) and the worker's snapshot (carparking/live.py).

export type SlotState = "FREE" | "BUSY" | "UNKNOWN";
/** The dashboard's slot filter (the counter cards): only these slots drawn on the video; null = all. */
export type SlotFilter = SlotState | "ENCROACHED" | "BLOCKED" | null;

export interface Slot {
  id: number;
  name: string;
  state: SlotState;
  hidden: boolean;
  blocked?: boolean; // FREE, but another vehicle stands in it: shown UNKNOWN
  note?: string | null; // why it is shown UNKNOWN (e.g. "on the vehicle of slot 6")
  phantom: boolean;
  duplicate_of: number | null;
  parent?: number | null; // a sub-bay: the slot it was split from
  capacity?: number; // vehicles the operator says this slot holds (default 1)
  count_note?: string | null; // capacity slots: why boxes on it were or were not counted
  vehicles?: number; // vehicles counted in it now (BUSY only)
  ground?: number | null; // fixed map / auto grid: share (0..1) of the matched vehicle's ground on this slot
  /** Spillage protection on (option "spillage"): a FREE bay a neighbour's vehicle stands on is "ENCROACHED" (state stays FREE; not usable). */
  avail?: "ENCROACHED" | "BLOCKED" | null;
  /** Blocked by object on (option "blocked_objects"): what stands in a FREE bay that is "BLOCKED" (potted plant, vase ...); the state stays FREE. */
  blocked_by?: string[];
  display?: string; // what to show: ENCROACHED, else the state
  encroached_by?: string[]; // the bays' names whose vehicles stand on it ("unplaced vehicle" = one that holds no bay)
  free_m?: number; // ground metric (calibrated camera): metres of the bay's width the vehicles leave
  cover?: number | null; // the largest share (0..1) of this bay under one of them
  poly: [number, number][]; // fractions of the frame
}

export interface Detection {
  label: string;
  conf: number;
  src: string; // "" | "zoom" | "cam" | "coco" | "cam+coco"
  kind: "vehicle" | "other";
  box: [number, number, number, number]; // x1, y1, x2, y2 as fractions
  poly?: [number, number][]; // an oriented (rotated) box: its 4 corners as fractions, drawn instead of the box
  /** Non-vehicle filter on: why the tracker did not let this box count as a vehicle (too_small / aspect / pylon / shadow). */
  rejected?: string;
}

export interface Area {
  top: number;
  bottom: number;
  left: number;
  right: number;
  roi: [number, number][] | null;
}

export interface Counts {
  free: number;
  busy: number;
  unknown: number;
  hidden: number;
  phantom: number;
  duplicate: number;
  slots: number; // total capacity: the map's slots plus sub-bays (a second vehicle in a slot)
  learning: number;
  sub_bays?: number;
  unplaced?: number; // vehicles seen that hold no bay yet (they are not in BUSY until they do)
  /** Spillage protection on only: FREE bays a neighbour's vehicle stands on (already taken out of `free`), and the drivable free spaces. */
  encroached?: number;
  available?: number;
  /** Blocked by object on only: FREE bays with an object standing in them (already taken out of `free` and `available`). */
  blocked_objects?: number;
}

export interface ModelInfo {
  setting: "auto" | "default" | "yolo11m"; // what is stored
  active: "default" | "yolo11m"; // what runs
  auto: boolean; // picked from the hardware at worker start
  auto_choice: "default" | "yolo11m";
  reason: string;
  swappable: boolean; // false: a camera model, the hybrid or the bay model is chosen (never swapped)
  weights: string | null;
  hardware: { cuda: boolean; gpu: string | null; cores: number; ram_gb: number };
}

/** Where an auto-grid bay comes from: measured ground, a continuation of the lane past it, or a bay the operator pinned. */
export type BaySource = "measured" | "extrapolated" | "manual" | "dynamic" | "sub";

/** The auto grid's make-up (Worker._stats). sources: slot id -> source (JSON keys are strings); short: expected bays it could
 * not place (pin them). */
export interface AutoSlots {
  version: number;
  count: number;
  measured?: number;
  extrapolated?: number;
  manual?: number;
  sub?: number; // ... of which sub-bays: a second vehicle in a slot another one holds (named "<slot>.2")
  dynamic?: number; // bays the worker gave to vehicles that stood in none (ids from 1000, gone when the vehicle leaves)
  expected?: number | null;
  short?: number;
  sources?: Record<string, BaySource>;
}

export interface Snapshot {
  status?: string;
  enabled?: boolean;
  cfg_ver?: number;
  code?: number;
  url?: string;
  frame_ts?: number;
  frame_age?: number | null;
  waiting?: boolean;
  /** Stream health (item 105): "stale" = no new frame, "frozen" = the same picture; both show every slot UNKNOWN and keep the open sessions. */
  stream?: { state: "active" | "stale" | "frozen"; age_s: number; for_s: number; diff: number; same_s: number };
  shape?: [number, number];
  seq?: number;
  camera?: string;
  scope?: string;
  mode?: "dynamic" | "fixed";
  clean_frames?: boolean; // the worker's frame images carry no drawing (the page draws every layer); absent / false: baked overlay
  auto_slots?: AutoSlots; // Dynamic mode on a HIGH-confidence calibration: the slots are its packed bays
  counts?: Counts;
  /** Non-vehicle filter on (option "obstacle_filter"): detections it dropped in the last frame, by reason (too_small / aspect / pylon / shadow). */
  obstacle?: { rejected: number; reasons: Record<string, number> };
  detector?: string;
  det_text?: string;
  rate?: number;
  frames?: number;
  logged?: number;
  saved?: number;
  last_saved?: string;
  learning?: number;
  dropped?: number;
  raw_counts?: Record<string, number>;
  vehicles?: number;
  others?: number;
  slots?: Slot[];
  detections?: Detection[];
  area?: Area;
  model?: ModelInfo;
  geometry?: GeometryLive; // Phase 2, see GeometryLive
}

/** Phase 2 (snapshot.geometry): the worker's metric view of the frame from the camera's ACTIVE calibration. Present only while
 * CARPARK_GEOMETRY_ENABLED is on and the camera has an accepted calibration. */
export interface GeometryLive {
  version: number;
  valid: boolean; // false: the frame is not the picture the calibration was made on (reason says so)
  reason?: string;
  confidence: "HIGH" | "MEDIUM" | "LOW" | "NONE" | null;
  uncertainty: number | null; // relative error of the metres (0.05 with a typed reference, about 0.25 without)
  reference_used: boolean;
  h: number | null; // camera height, metres
  lens: string | null;
  area_m2: number;
  counts?: { capacity: number; occupied: number; free: number; vehicles: number; measured: number; beyond_range: number };
  vehicles: { i: number; poly: [number, number][]; centroid_m: [number, number] }[]; // footprints, image fractions
  distances: { kind: "vehicle" | "object"; label?: string; m: number; p: [number, number]; q: [number, number] }[];
  bays: { id: number; poly: [number, number][]; state: "FREE" | "BUSY"; car: boolean }[];
  /** The drift guard's last check (carparking/geometry/drift.py): is the camera where the calibration was made? */
  drift?: { status: "ok" | "drift" | "inconclusive" | "stale"; shift_px?: number | null; at: number; version: number;
            baseline: "calibration" | "worker"; why?: string } | null;
  drift_guard?: boolean; // the version carries its own reference (false: an older calibration, baseline from the first frame)
}

export interface StateMessage {
  type: "state";
  connected: boolean;
  heartbeat_age: number | null;
  settings_version: number;
  snapshot: Snapshot;
  time: number;
}

export interface ParkingEvent {
  id: number;
  slot_id: number;
  /** entry / exit = a vehicle moved (they open and close sessions); the others are notices: encroached / encroach_cleared (a free bay a
   * neighbour's vehicle stands on, spillage protection) and rejected (a detection the non-vehicle filter dropped). */
  event: "entry" | "exit" | "encroached" | "encroach_cleared" | "rejected" | "object_blocked" | "object_cleared";
  time: string; // ISO 8601 in the camera's time zone
  ts: number;
  session_id: number | null; // none for notices
  // slot_id 0 on a notice = it concerns no bay
  note: string;
}

export interface Session {
  id: number;
  slot_id: number;
  in_time: string | null;
  out_time: string | null;
  in_ts: number | null;
  out_ts: number | null;
  duration_min: number | null;
  status: string;
  note: string;
  scope?: string; // all-versions listing only: "<camera> | <detector>[ | auto vN]" the session was logged in
}

/** GET /api/sessions/page: the History table, open sessions (still parked) first, then newest first. */
export interface SessionPage {
  items: Session[];
  total: number;
  open: number;
  page: number;
  size: number;
  pages: number;
}

export interface SettingsValues {
  enabled: boolean;
  camera: string;
  stream: "main" | "sub" | "custom";
  custom_url: string;
  detector: string;
  bay_model: "auto" | "baseline" | "obb";
  inference: "1280" | "1920" | "sahi";
  conf: number;
  classes: string[];
  extra: string[];
  slot_mode: "dynamic" | "fixed";
  match_rule: "ground" | "center" | "iou";
  match_iou: number;
  save_every: number;
  hold: number;
  exit_hold: number;
  exit_frames: number;
  occlusion: number;
  hidden_max: number;
  learn_hold: number;
  learn: boolean;
  absent_free: boolean;
  cut_top: number;
  cut_bottom: number;
  cut_left: number;
  cut_right: number;
  roi: string;
  model_variant: "auto" | "default" | "yolo11m";
  // worker options like model_variant (both off by default; applied live, no restart, no settings version)
  spillage?: boolean; // a free bay a neighbour's vehicle stands on is ENCROACHED, not counted free
  blocked_objects?: boolean; // a free bay with a potted plant, vase, box ... in it is BLOCKED, not counted free
  obstacle_filter?: boolean; // cones, people, shadows and other clutter do not make a bay BUSY
  // counting thresholds of the shown state (Dynamic mode; worker options, applied live): shares 0..1
  hidden_ground: number; // vehicles in front cover more of a free slot's ground: UNKNOWN (default 0.8)
  ground_share: number; // a vehicle covers more of a free slot's ground: not FREE (default 0.4)
  vehicle_iou: number; // a BUSY slot without its own box overlapped this much by another slot's vehicle (default 0.2)
  bay_share: number; // fixed maps / auto grid: a vehicle just outside a bay holds it only with this share of its ground inside (default 0.5)
  pin_reach: number; // auto grid only: a capacity pin also counts a vehicle no bay holds within this share of its height outside the pin (default 0.25)
}

export interface SettingsResponse {
  version: number;
  values: SettingsValues;
}

/** GET / PUT /api/overflow: the worker options auto_dynamic + slot_split as one switch (enabled = both on). */
export interface Overflow {
  enabled: boolean;
  auto_dynamic: boolean;
  slot_split: boolean;
}

export interface Options {
  stream: string[];
  detector: string[];
  bay_model: string[];
  inference: string[];
  slot_mode: string[];
  match_rule: string[];
  classes: string[];
  extra: string[];
  detector_selected: string;
  ranges: Record<string, [number, number]>;
  tuning_defaults?: Record<"hidden_ground" | "ground_share" | "vehicle_iou" | "bay_share" | "pin_reach", number>;
}

export interface AuthInfo {
  view: boolean;
  admin: boolean;
  view_required: boolean;
  writes_enabled: boolean;
}

export interface CameraStatus {
  connected: boolean;
  heartbeat_age: number | null;
  enabled: boolean; // monitoring switched on (its settings)
  running: boolean; // the worker is tracking it now
  slot_mode: "dynamic" | "fixed";
  counts: Counts | null;
  state: string | null;
}

/** A registered camera (GET /api/cameras): URLs masked, its go2rtc streams, its live status. */
export interface CameraInfo {
  id: string;
  name: string; // its history key: never changes
  label: string; // the display name, editable ("" = name)
  display: string; // what to show: label or name
  primary: boolean;
  sort_order: number;
  metadata: Record<string, unknown>;
  rtsp_url: string;
  rtsp_sub_url: string;
  streams: { hd: string; sd: string | null; native: string };
  status: CameraStatus;
}

/** GET /api/analytics: one camera over a window (PostgreSQL, or SQLite as fallback). */
export interface Analytics {
  days: number;
  since: string;
  bucket: "5 min" | "1 h" | "1 day";
  source: "postgres" | "sqlite";
  scope: "all" | "current";
  cam: string | null;
  totals: {
    entries: number; exits: number; sessions: number; closed: number; open: number;
    avg_min: number | null; median_min: number | null; busiest_hour: string | null; peak_busy: number | null;
  };
  per_hour: { hour: string; entries: number; exits: number }[];
  per_day: { day: string; entries: number; exits: number }[];
  durations: { bucket: string; sessions: number }[];
  occupancy: { t: string; free: number; busy: number; unknown: number; total: number }[];
}

/** GET /api/analytics/overview: every camera at once. */
export interface Overview {
  days: number;
  source: "postgres" | "sqlite";
  cameras: { id: string; display: string; entries: number; exits: number; open: number; online: boolean;
             monitoring: boolean; free: number | null; busy: number | null; total: number | null }[];
}

/** GET /api/camera-events: a camera's state changes (recorded by the sync service). */
export interface CameraEvent {
  id: number;
  time: string;
  ts: number;
  kind: string; // first_seen / worker_online / worker_offline / monitoring_on / monitoring_off / scope_changed
  detail: string;
}

/** GET /api/history/sync: where History & Analytics are served from, how far the PostgreSQL copy is. */
export interface SyncStatus {
  backend: "postgres" | "sqlite";
  error: string | null;
  cameras: { id: string; display: string; sqlite: { sessions: number; events: number };
             behind: { sessions: number; events: number };
             rows: { sessions: number; events: number; samples: number } | null;
             age_s: number | null; error: string; source: "postgres" | "sqlite";
             live?: SessionCheck | null }[];
}

/** The worker's check of its history against the live count (snapshot.sessions, /api/history/sync cameras[].live). */
export interface SessionCheck {
  expected: number; // vehicles counted BUSY
  open: number; // open sessions in the history
  mismatch_s: number; // how long they have differed (0 = equal)
  error: string; // a failed write of the history ("" = none)
}
