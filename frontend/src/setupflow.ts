// Camera setup (Phase 2): the types of its API and the pure rules of the workflow (unit-tested in setupflow.test.ts).
// The page itself is components/CameraSetup.tsx; the endpoints are api/geometry_routes.py.

import { api } from "./api";

export type Pt = [number, number];

export interface SetupStatus {
  enabled: boolean; // CARPARK_GEOMETRY_ENABLED on the server
  service: "up" | "down" | "unchecked"; // the on-demand setup service (SAM / GeoCalib / depth)
  loaded: Record<string, boolean> | null;
  pins: Record<string, { commit: string }> | null;
}

export type Confidence = "HIGH" | "MEDIUM" | "LOW" | "NONE";

export interface SetupJob<T = unknown> {
  id: string;
  kind: "segment" | "calibrate" | "reanchor";
  cam: string;
  state: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number; // 0..1
  result: T | null;
  error: string | null;
}

export interface Candidate {
  polygon: Pt[];
  score: number;
  area_frac: number;
}
export interface SegmentResult extends Candidate {
  candidates: Candidate[];
  model: string;
  warnings: string[];
}

export interface CalibrateResult {
  version: number;
  state: "draft";
  confidence: Confidence;
  h: number;
  pitch_deg: number;
  area_m2: number;
  capacity: number;
  bays_with_cars: number;
  reference_used: boolean;
  scale_uncertainty: number | null;
  scale_sources: Record<string, number>;
  warnings: string[];
}

/** A stored version without its heavy parts (GET /api/cameras/{id}/setup/geometry). */
export interface GeometrySummary {
  ver: number;
  state: "draft" | "active" | "superseded" | "stale";
  note: string;
  created_at: number;
  activated_at: number | null;
  confidence: Confidence | null;
  h: number | null;
  scale_uncertainty: number | null;
  reference_used: boolean;
  lens: string | null;
  area_m2: number | null;
  capacity: number | null;
  bays_with_cars: number | null;
  drift_guard?: boolean; // the version carries a stored reference for the drift guard
  reanchor?: ReanchorReport | null; // set when this version was carried over from an earlier one after a camera move
  warnings: string[];
}

/** How a re-anchored version was derived from its parent (geometry/reanchor.py). */
export interface ReanchorReport {
  parent: number | null;
  method: string;
  matches: number;
  inliers: number;
  rms_px: number;
  rotation_deg: number;
  yaw_deg: number;
  pitch_before_deg: number;
  pitch_after_deg: number;
  roll_before_deg: number;
  roll_after_deg: number;
  bays: number;
  bays_in_view: number;
}

export interface ReanchorResult extends Omit<CalibrateResult, "state"> {
  state: "draft";
  parent: number;
  parent_state: GeometrySummary["state"];
  reanchor: ReanchorReport;
}

/** The versions a calibration can be carried over from: ones that were in service and carry a stored background (drift guard). */
export const reanchorParents = (versions: GeometrySummary[]) =>
  versions.filter((v) => (v.state === "stale" || v.state === "active" || v.state === "superseded") && v.drift_guard);

/** The default parent, as the server picks it: the most recently activated one that has a reference. */
export function defaultParent(versions: GeometrySummary[]): GeometrySummary | null {
  const ok = reanchorParents(versions);
  return ok.length ? ok.reduce((a, b) => ((b.activated_at ?? 0) > (a.activated_at ?? 0) || ((b.activated_at ?? 0) === (a.activated_at ?? 0) && b.ver > a.ver) ? b : a)) : null;
}

/** Can a re-anchoring start, and if not why. It needs no setup service (no SAM, no depth), only a calibration with a reference. */
export function reanchorGate(i: { still: boolean; parents: number; busy: boolean }): { ok: boolean; why: string | null } {
  if (i.parents === 0) return { ok: false, why: "No calibration of this camera has a stored reference: do the full setup." };
  if (!i.still) return { ok: false, why: "Take a fresh still first, with the camera where it is now." };
  if (i.busy) return { ok: false, why: "A job is running." };
  return { ok: true, why: null };
}

/** The re-anchoring report as lines for the review: what turned, how well the background matched. */
export function reanchorLines(r: ReanchorReport): string[] {
  const d = (v: number) => `${v >= 0 ? "+" : "-"}${Math.abs(v).toFixed(1)}°`;
  return [
    `The camera turned ${r.rotation_deg.toFixed(1)}° (heading ${d(r.yaw_deg)}, tilt ${d(r.pitch_after_deg - r.pitch_before_deg)}, roll ${d(r.roll_after_deg - r.roll_before_deg)})`,
    `${r.inliers} of ${r.matches} background points agree, residual ${r.rms_px.toFixed(1)} px`,
    `${r.bays_in_view} of ${r.bays} bays are still in view`,
  ];
}

export interface GeometryVersion extends GeometrySummary {
  record: {
    surface: { polygon: Pt[] };
    capacity: { slots: { id: number; poly: Pt[]; car: boolean }[] };
    calibration: { f: number; k1: number; pitch: number };
  };
}

export interface Reference {
  a: Pt | null;
  b: Pt | null;
  metres: string; // what the Admin typed
}

export const EMPTY_REFERENCE: Reference = { a: null, b: null, metres: "" };
export const MIN_SEPARATION = 0.03; // two reference points closer than this share of the frame are one click twice
export const METRES = { min: 0.3, max: 200 };

/** A click on the still as shares of its size, or null when the box has no size (not laid out). */
export function clickPoint(rect: { left: number; top: number; width: number; height: number }, x: number, y: number): Pt | null {
  if (rect.width <= 0 || rect.height <= 0) return null;
  const c = (v: number) => Math.min(1, Math.max(0, v));
  return [Math.round(c((x - rect.left) / rect.width) * 1e4) / 1e4, Math.round(c((y - rect.top) / rect.height) * 1e4) / 1e4];
}

/** Distance between two points in pixels of a frame of that size. */
export const pixelDistance = (a: Pt, b: Pt, size: [number, number]) => Math.hypot((a[0] - b[0]) * size[0], (a[1] - b[1]) * size[1]);

/** Why the reference can't be used yet, or null. */
export function referenceProblem(ref: Reference, size: [number, number] = [1920, 1080]): string | null {
  if (!ref.a || !ref.b) return "Click the two end points of the distance you know on the still.";
  if (pixelDistance(ref.a, ref.b, size) < MIN_SEPARATION * size[0]) return "The two points are too close: click the two ends of the distance.";
  const m = Number(ref.metres.replace(",", "."));
  if (!ref.metres.trim() || !Number.isFinite(m)) return "Type the distance between the two points in metres.";
  if (m < METRES.min || m > METRES.max) return `The distance must be between ${METRES.min} and ${METRES.max} m.`;
  return null;
}

/** The reference as the API takes it, or null while it is incomplete. */
export function referenceBody(ref: Reference, size?: [number, number]): { a: Pt; b: Pt; metres: number } | null {
  if (referenceProblem(ref, size) || !ref.a || !ref.b) return null;
  return { a: ref.a, b: ref.b, metres: Number(ref.metres.replace(",", ".")) };
}

export interface CalibrateInputs {
  still: boolean;
  polygon: Pt[] | null;
  reference: Reference;
  withoutReference: boolean; // the Admin chose to go on without one (the metres stay about +-25 %)
  service: SetupStatus["service"] | null;
  busy: boolean;
  size?: [number, number];
}

/** Can the calibration run, and if not why. A typed reference is asked for up front: without one (or an explicit decision
 * to go without) the step stays closed. */
export function calibrateGate(i: CalibrateInputs): { ok: boolean; why: string | null } {
  if (!i.still) return { ok: false, why: "Take a still first." };
  if (referenceProblem(i.reference, i.size) && !i.withoutReference) {
    return { ok: false, why: "Give a distance you know (step 2), or choose to go on without one." };
  }
  if (!i.polygon || i.polygon.length < 3) return { ok: false, why: "Mark the parking surface first (step 3)." };
  if (i.service === "down") return { ok: false, why: "The setup service is not running." };
  if (i.busy) return { ok: false, why: "A job is running." };
  return { ok: true, why: null };
}

/** Accepting a version: fine when its scale was anchored (HIGH); otherwise the Admin has to say so. */
export const needsUnverifiedOk = (confidence: Confidence | null | undefined) => confidence !== "HIGH";

export function confidenceNote(confidence: Confidence | null | undefined, uncertainty: number | null | undefined): string {
  const pct = uncertainty != null ? `±${Math.round(uncertainty * 100)} %` : "";
  switch (confidence) {
    case "HIGH": return `Anchored by your typed distance ${pct}`;
    case "MEDIUM": return `From vehicle sizes and depth only ${pct}: type a known distance to anchor it`;
    case "LOW": return `One estimate, or estimates that disagree ${pct}: type a known distance`;
    default: return "Not calibrated";
  }
}

export const CONFIDENCE_TONE: Record<string, string> = {
  HIGH: "bg-success/10 text-success ring-success/30",
  MEDIUM: "bg-warning/15 text-ink ring-warning/40",
  LOW: "bg-danger/10 text-danger ring-danger/30",
  NONE: "bg-body text-muted ring-line",
};

/** Poll a setup job until it is done or failed (the first look at once, then every `interval` ms). */
export async function waitForJob<T>(id: string, onUpdate: (j: SetupJob<T>) => void, stop: { cancelled: boolean }, interval = 1000): Promise<SetupJob<T>> {
  for (;;) {
    const job = await api.get<SetupJob<T>>(`/api/setup/jobs/${id}`);
    onUpdate(job);
    if (job.state === "done" || job.state === "error") return job;
    if (stop.cancelled) throw new Error("cancelled");
    await new Promise((r) => setTimeout(r, interval));
  }
}

// --- history and the drift guard (Phase 2, step 6) -----------------------------------------------------------------------

export interface HistoryPoint {
  t: number; // start of the bucket (unix seconds)
  samples: number;
  occupied_avg: number;
  occupied_max: number;
  occupied_min: number;
  capacity_avg: number;
  capacity_max: number;
  vehicles_avg: number;
  ver: number; // the newest calibration version in the bucket
}
export interface HistoryResponse {
  since: number;
  until: number;
  bucket_s: number;
  points: HistoryPoint[];
  versions: number[];
}
export interface GeometryEvent {
  id: number;
  t: number;
  kind: string; // draft / activated / stale / discarded / drift_suspected / drift_cleared
  ver: number | null;
  detail: Record<string, unknown>;
}
/** The worker's last drift check (snapshot.geometry.drift). */
export interface DriftStatus {
  status: "ok" | "drift" | "inconclusive" | "stale";
  shift_px?: number | null;
  at: number;
  version: number;
  baseline: "calibration" | "worker"; // the reference came with the calibration, or is the first frame the worker saw
  why?: string;
}

/** One line of the calibration's story. */
export function eventText(e: GeometryEvent): string {
  const d = e.detail as { previous?: number | null; rollback?: boolean; reason?: string; versions?: number[]; shift_px?: number; parent?: number | null; rotation_deg?: number };
  switch (e.kind) {
    case "draft": return d.parent ? `Calibration v${e.ver} made (a draft, from v${d.parent})` : `Calibration v${e.ver} made (a draft)`;
    case "reanchored": return `v${e.ver} re-anchored from v${d.parent}${d.rotation_deg != null ? ` after a ${d.rotation_deg.toFixed(1)}° turn` : ""}`;
    case "activated": return d.rollback ? `Rolled back to v${e.ver}${d.previous ? ` (from v${d.previous})` : ""}` : `v${e.ver} accepted${d.previous ? `, replacing v${d.previous}` : ""}${d.parent ? ` (re-anchored from v${d.parent})` : ""}`;
    case "stale": return `v${e.ver} marked stale${d.reason ? `: ${d.reason}` : ""}`;
    case "discarded": return `Draft${(d.versions?.length ?? 0) > 1 ? "s" : ""} ${(d.versions ?? [e.ver]).map((v) => `v${v}`).join(", ")} discarded`;
    case "drift_suspected": return `The camera may have moved${d.shift_px != null ? ` (${Math.round(d.shift_px)} px)` : ""}: checking again`;
    case "drift_cleared": return "The camera is back where it was";
    default: return e.kind;
  }
}

/** The drift guard's line on the capacity card. tone: ok / warn / muted. */
export function driftNote(drift: DriftStatus | null | undefined, guard: boolean | undefined, now: number): { text: string; tone: "ok" | "warn" | "muted" } {
  if (!drift) return { text: guard === false ? "Drift guard: no stored reference, using the first frame as the baseline" : "Drift guard: waiting for the first check", tone: "muted" };
  const ago = `checked ${Math.max(0, Math.round(now - drift.at))} s ago`;
  const base = drift.baseline === "worker" ? " · baseline: first frame seen (an older calibration)" : "";
  switch (drift.status) {
    case "ok": return { text: `Camera position unchanged${drift.shift_px != null ? ` (${Math.round(drift.shift_px)} px)` : ""} · ${ago}${base}`, tone: "ok" };
    case "drift": return { text: `The camera may have moved (${Math.round(drift.shift_px ?? 0)} px): confirming · ${ago}`, tone: "warn" };
    case "stale": return { text: "The camera moved: this calibration was marked stale", tone: "warn" };
    default: return { text: `Camera position can't be checked now${drift.why ? ` (${drift.why})` : ""} · ${ago}`, tone: "muted" };
  }
}

/** The camera moved: the newest event says the calibration went stale and nothing has been accepted since. */
export function staleBanner(events: GeometryEvent[], activeVersion: number | null | undefined): GeometryEvent | null {
  if (activeVersion) return null;
  const e = events.find((x) => x.kind === "stale" || x.kind === "activated" || x.kind === "draft");
  return e?.kind === "stale" ? e : null;
}

/** The chart's lines in a w x h box: occupied (area + max line) and the bays available, scaled to the largest value. */
export function chartLines(points: HistoryPoint[], since: number, until: number, w: number, h: number) {
  const top = Math.max(1, ...points.map((p) => Math.max(p.capacity_max, p.occupied_max)));
  const x = (t: number) => ((t - since) / Math.max(1, until - since)) * w;
  const y = (v: number) => h - (v / top) * h;
  const line = (f: (p: HistoryPoint) => number) => points.map((p) => `${x(p.t).toFixed(1)},${y(f(p)).toFixed(1)}`).join(" ");
  const area = points.length
    ? `${x(points[0].t).toFixed(1)},${h} ${line((p) => p.occupied_avg)} ${x(points[points.length - 1].t).toFixed(1)},${h}`
    : "";
  return { top, occupiedMax: line((p) => p.occupied_max), area, capacity: line((p) => p.capacity_max) };
}
