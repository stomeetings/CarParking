// Fixed slot drawing: pure geometry and JSON (unit-tested in slotdraw.test.ts). Points are shares of the frame
// (0..1), the format of config/fixed_slots.json (carparking/fixed_slots.py), so a map drawn on any stream size fits
// every other: the worker's 2560x1440 frame, the 1080p video, the 360p sub-stream.

export type Pt = [number, number];

export interface DrawnSlot {
  id: number;
  name: string;
  poly: Pt[];
  /** Vehicles this slot holds (a big pin with a truck and a car): the total counts it, BUSY counts the vehicles on it. Default 1. */
  capacity?: number;
}

/** The most vehicles one slot can be given (fixed_slots.MAX_CAPACITY). */
export const MAX_CAPACITY = 20;

/** The server refuses a slot smaller than this share of the frame (fixed_slots.validate_slots). */
export const MIN_AREA = 1e-4;

export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
export const round4 = (v: number) => Math.round(v * 1e4) / 1e4;
export const pt = (x: number, y: number): Pt => [round4(clamp01(x)), round4(clamp01(y))];

export function nextId(slots: DrawnSlot[]): number {
  return slots.reduce((m, s) => Math.max(m, s.id), 0) + 1;
}

/** Shoelace area, as a share of the frame. */
export function area(poly: Pt[]): number {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

/** Corners of the rectangle dragged from a to b, clockwise from the top left. */
export function rectPoly(a: Pt, b: Pt): Pt[] {
  const [x1, x2] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  const [y1, y2] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];
  return [pt(x1, y1), pt(x2, y1), pt(x2, y2), pt(x1, y2)];
}

/** A click this close to the first corner (in pixels on screen) closes the polygon. */
export function nearFirst(draft: Pt[], p: Pt, size: [number, number], tolPx = 12): boolean {
  if (draft.length < 3) return false;
  const [w, h] = size;
  return Math.hypot((draft[0][0] - p[0]) * w, (draft[0][1] - p[1]) * h) <= tolPx;
}

/** Why a polygon can't be a slot, or null. */
export function polyProblem(poly: Pt[]): string | null {
  if (poly.length < 3) return "a slot needs at least 3 corners";
  if (poly.some(([x, y]) => x < 0 || x > 1 || y < 0 || y > 1)) return "a corner lies outside the frame";
  if (area(poly) < MIN_AREA) return "the slot is too small (a line or a dot)";
  return null;
}

/** True when p lies inside (or on the edge of) the polygon (ray casting). */
export function pointInPoly(p: Pt, poly: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** The polygon grown outward by d (frame fractions): every corner moves along the mitre of its two edge normals, so each edge ends up d from where it was.
 * For a convex polygon of any winding; the mitre is capped so a sharp corner does not shoot off. Corners are not clamped to the frame (the picture clips them). */
export function growPoly(poly: Pt[], d: number): Pt[] {
  const n = poly.length;
  if (n < 3 || d === 0) return poly.map(([x, y]) => [x, y] as Pt);
  const turn = poly.reduce((a, [x, y], i) => a + x * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * y, 0) >= 0 ? 1 : -1;
  const normal = (a: Pt, b: Pt): Pt => {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy) || 1;
    return [(turn * dy) / len, (-turn * dx) / len];
  };
  return poly.map((q, i) => {
    const n1 = normal(poly[(i + n - 1) % n], q);
    const n2 = normal(q, poly[(i + 1) % n]);
    const k = d / Math.max(0.25, 1 + n1[0] * n2[0] + n1[1] * n2[1]);
    return [round4(q[0] + (n1[0] + n2[0]) * k), round4(q[1] + (n1[1] + n2[1]) * k)] as Pt;
  });
}

/** Add the drawn polygon as a new slot -> the slots, or an error. */
export function addSlot(slots: DrawnSlot[], poly: Pt[], name?: string): { slots: DrawnSlot[]; error?: string } {
  const problem = polyProblem(poly);
  if (problem) return { slots, error: problem };
  const id = nextId(slots);
  return { slots: [...slots, { id, name: name?.trim() || `S${id}`, poly: poly.map(([x, y]) => pt(x, y)) }] };
}

/** Set a slot's capacity from the typed text: empty or 1 = the default (the key is dropped), anything else is kept as typed
 * (a non-number is reported by mapProblems, never silently changed). */
export function setCapacity(slots: DrawnSlot[], id: number, text: string): DrawnSlot[] {
  return slots.map((s) => {
    if (s.id !== id) return s;
    const { capacity: _old, ...rest } = s;
    const t = text.trim();
    return t === "" || t === "1" ? rest : { ...rest, capacity: Number(t) };
  });
}

export function moveVertex(slots: DrawnSlot[], id: number, index: number, p: Pt): DrawnSlot[] {
  return slots.map((s) => (s.id === id ? { ...s, poly: s.poly.map((q, i) => (i === index ? pt(p[0], p[1]) : q)) } : s));
}

export function moveSlot(slots: DrawnSlot[], id: number, dx: number, dy: number): DrawnSlot[] {
  return slots.map((s) => {
    if (s.id !== id) return s;
    // keep the whole slot inside the frame
    const xs = s.poly.map((p) => p[0]);
    const ys = s.poly.map((p) => p[1]);
    const ddx = Math.min(Math.max(dx, -Math.min(...xs)), 1 - Math.max(...xs));
    const ddy = Math.min(Math.max(dy, -Math.min(...ys)), 1 - Math.max(...ys));
    return { ...s, poly: s.poly.map(([x, y]) => pt(x + ddx, y + ddy)) };
  });
}

/** Problems of the whole map (what the server would refuse), or []. */
export function mapProblems(slots: DrawnSlot[]): string[] {
  const out: string[] = [];
  const ids = new Set<number>();
  for (const s of slots) {
    if (ids.has(s.id)) out.push(`slot ${s.id}: duplicate id`);
    ids.add(s.id);
    const p = polyProblem(s.poly);
    if (p) out.push(`slot ${s.id}: ${p}`);
    if (s.capacity != null && (!Number.isInteger(s.capacity) || s.capacity < 1 || s.capacity > MAX_CAPACITY)) {
      out.push(`slot ${s.id}: the capacity must be a whole number from 1 to ${MAX_CAPACITY}`);
    }
  }
  return out;
}

/** The export: the map as the server stores it (shares of the frame) plus the same corners in camera pixels. */
export function toJson(camera: string, slots: DrawnSlot[], size: [number, number]): string {
  const [w, h] = size;
  return JSON.stringify(
    {
      camera,
      frame: { width: w, height: h },
      slots: slots.map((s) => ({
        id: s.id,
        name: s.name,
        poly: s.poly,
        ...(s.capacity ? { capacity: s.capacity } : {}),
        points_px: s.poly.map(([x, y]) => [Math.round(x * w), Math.round(y * h)]),
      })),
    },
    null,
    2,
  );
}

/** Read a map: this tool's export, config/fixed_slots.json's per-camera {"slots": [...]}, or a bare list. Corners
 * as shares of the frame, or (points_px / values above 1 with a frame size) camera pixels. */
export function fromJson(text: string, size: [number, number]): DrawnSlot[] {
  const data = JSON.parse(text);
  const list = Array.isArray(data) ? data : Array.isArray(data?.slots) ? data.slots : null;
  if (!list) throw new Error('expected a list of slots, or {"slots": [...]}');
  const [w, h] = data?.frame?.width && data?.frame?.height ? [data.frame.width, data.frame.height] : size;
  const slots: DrawnSlot[] = list.map((s: { id?: number; name?: string; poly?: number[][]; points_px?: number[][]; capacity?: number }, i: number) => {
    const raw = s.poly ?? s.points_px;
    if (!Array.isArray(raw)) throw new Error(`slot ${i + 1}: no corners ("poly")`);
    const px = !s.poly || raw.some(([x, y]) => x > 1 || y > 1);
    const poly = raw.map(([x, y]) => (px ? pt(x / w, y / h) : pt(x, y)));
    const id = Number.isInteger(s.id) && (s.id as number) > 0 ? (s.id as number) : i + 1;
    return { id, name: String(s.name ?? id), poly, ...(s.capacity && s.capacity > 1 ? { capacity: s.capacity } : {}) };
  });
  const problems = mapProblems(slots);
  if (problems.length) throw new Error(problems[0]);
  return slots;
}
