// Fine adjustment of bays already laid (Sample test zones): shift, widen / lengthen / shrink, turn. Pure functions on fractions of the picture;
// sizes and angles are worked out in pixels (w x h), so a turn stays a turn on a 16:9 picture. The backend's generation is not involved.
import { type DrawnSlot, type Pt, pt } from "./slotdraw";

type V = [number, number];
const centroid = (poly: Pt[]): V => [poly.reduce((m, p) => m + p[0], 0) / poly.length, poly.reduce((m, p) => m + p[1], 0) / poly.length];
const unit = (v: V): V | null => {
  const n = Math.hypot(v[0], v[1]);
  return n < 1e-9 ? null : [v[0] / n, v[1] / n];
};
const back = (q: V, w: number, h: number): Pt => pt(q[0] / w, q[1] / h);

/** Move the bays `ids` by (dx, dy) (fractions), all by the same amount, stopped where the first of them meets the picture's edge. */
export function shiftBays(all: DrawnSlot[], ids: number[], dx: number, dy: number): DrawnSlot[] {
  const own = all.filter((s) => ids.includes(s.id));
  if (!own.length) return all;
  const xs = own.flatMap((s) => s.poly.map((p) => p[0]));
  const ys = own.flatMap((s) => s.poly.map((p) => p[1]));
  const ddx = Math.min(Math.max(dx, -Math.min(...xs)), 1 - Math.max(...xs));
  const ddy = Math.min(Math.max(dy, -Math.min(...ys)), 1 - Math.max(...ys));
  return all.map((s) => (ids.includes(s.id) ? { ...s, poly: s.poly.map(([x, y]) => pt(x + ddx, y + ddy)) } : s));
}

/** The bays `ids`, each moved along its OWN directions (not the picture's): `across` pixels along its front edge (to its right), `along` pixels towards its back
 * (negative: towards its front). A row that lies at a slant moves along the row, whatever the picture's axes are. Each bay is stopped at the picture's edge. */
export function shiftAlong(all: DrawnSlot[], ids: number[], w: number, h: number, across: number, along: number): DrawnSlot[] {
  let out = all;
  for (const s of all) {
    if (!ids.includes(s.id) || s.poly.length !== 4) continue;
    const p = s.poly.map(([x, y]): V => [x * w, y * h]);
    const u = unit([p[1][0] - p[0][0] + p[2][0] - p[3][0], p[1][1] - p[0][1] + p[2][1] - p[3][1]]);
    const v = unit([p[3][0] - p[0][0] + p[2][0] - p[1][0], p[3][1] - p[0][1] + p[2][1] - p[1][1]]);
    if (!u || !v) continue;
    out = shiftBays(out, [s.id], (across * u[0] + along * v[0]) / w, (across * u[1] + along * v[1]) / h);
  }
  return out;
}

/** A bay (front-left, front-right, back-right, back-left) stretched about its centre: `across` times as wide along its front, `along` times as long
 * towards its back. Its own slant (the perspective of its row) is kept: the two directions are the bay's own, not the picture's axes. */
export function resizeBay(poly: Pt[], w: number, h: number, across: number, along: number): Pt[] {
  if (poly.length !== 4) return poly;
  const px = poly.map(([x, y]): V => [x * w, y * h]);
  const c = centroid(px as unknown as Pt[]);
  const u = unit([px[1][0] - px[0][0] + px[2][0] - px[3][0], px[1][1] - px[0][1] + px[2][1] - px[3][1]]);
  const v = unit([px[3][0] - px[0][0] + px[2][0] - px[1][0], px[3][1] - px[0][1] + px[2][1] - px[1][1]]);
  const det = u && v ? u[0] * v[1] - u[1] * v[0] : 0;
  if (!u || !v || Math.abs(det) < 1e-6) return poly;
  return px.map((q) => {
    const d: V = [q[0] - c[0], q[1] - c[1]];
    const a = (d[0] * v[1] - d[1] * v[0]) / det; // d = a u + b v
    const b = (u[0] * d[1] - u[1] * d[0]) / det;
    return back([c[0] + a * across * u[0] + b * along * v[0], c[1] + a * across * u[1] + b * along * v[1]], w, h);
  });
}

/** A bay turned by `deg` degrees (clockwise on the screen) about its centre. */
export function rotateBay(poly: Pt[], w: number, h: number, deg: number): Pt[] {
  const px = poly.map(([x, y]): V => [x * w, y * h]);
  const c = centroid(px as unknown as Pt[]);
  const r = (deg * Math.PI) / 180;
  const cos = Math.cos(r);
  const sin = Math.sin(r);
  return px.map((q) => back([c[0] + (q[0] - c[0]) * cos - (q[1] - c[1]) * sin, c[1] + (q[0] - c[0]) * sin + (q[1] - c[1]) * cos], w, h));
}

/** The bays `ids` run through `fn` (a new polygon from the old), the others as they were. */
export function tuneBays(all: DrawnSlot[], ids: number[], fn: (poly: Pt[]) => Pt[]): DrawnSlot[] {
  return all.map((s) => (ids.includes(s.id) ? { ...s, poly: fn(s.poly) } : s));
}

/** A bay (front-left, front-right, back-right, back-left) cut into `parts` equal bays along its `width` (side by side, left to right) or its `length`
 * (front to back), in the bay's own ground perspective: the cuts are equal on the ground, not on the picture (the square-to-quad homography of its
 * four corners). Each part keeps the corner order. A bay laid across two cars becomes one bay per car. */
export function splitBay(poly: Pt[], axis: "width" | "length", parts = 2): Pt[][] {
  if (poly.length !== 4 || parts < 2) return [poly];
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = poly;
  const sx = x0 - x1 + x2 - x3;
  const sy = y0 - y1 + y2 - y3;
  let a: number, b: number, d: number, e: number, g = 0, h = 0;
  if (Math.abs(sx) < 1e-12 && Math.abs(sy) < 1e-12) {
    a = x1 - x0; b = x3 - x0; d = y1 - y0; e = y3 - y0;
  } else {
    const dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2;
    const den = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(den) < 1e-14) return [poly];
    g = (sx * dy2 - dx2 * sy) / den;
    h = (dx1 * sy - sx * dy1) / den;
    a = x1 - x0 + g * x1; b = x3 - x0 + h * x3; d = y1 - y0 + g * y1; e = y3 - y0 + h * y3;
  }
  const at = (u: number, v: number): Pt => {
    const k = g * u + h * v + 1;
    return pt((a * u + b * v + x0) / k, (d * u + e * v + y0) / k);
  };
  return Array.from({ length: parts }, (_, i) => {
    const lo = i / parts, hi = (i + 1) / parts;
    return axis === "width" ? [at(lo, 0), at(hi, 0), at(hi, 1), at(lo, 1)] : [at(0, lo), at(1, lo), at(1, hi), at(0, hi)];
  });
}

/** The middle of a set of polygons (the mean of all their corners), as fractions: what a whole group turns about. */
export function groupCentre(polys: Pt[][]): Pt {
  const all = polys.flat();
  if (!all.length) return [0.5, 0.5];
  return [all.reduce((m, p) => m + p[0], 0) / all.length, all.reduce((m, p) => m + p[1], 0) / all.length];
}

/** Corners turned `deg` degrees clockwise on the screen about the point `c` (fractions), worked out in pixels. Not clamped: a corner may land outside
 * the picture (see inFrame); the result is rounded to 4 decimals like every drawn point. */
export function turnPoints(poly: Pt[], c: Pt, w: number, h: number, deg: number): Pt[] {
  const r = (deg * Math.PI) / 180;
  const cos = Math.cos(r);
  const sin = Math.sin(r);
  return poly.map(([x, y]): Pt => {
    const dx = (x - c[0]) * w;
    const dy = (y - c[1]) * h;
    return [Math.round(((c[0] * w + dx * cos - dy * sin) / w) * 1e4) / 1e4, Math.round(((c[1] * h + dx * sin + dy * cos) / h) * 1e4) / 1e4];
  });
}

/** Every corner lies on the picture (0..1, a hair of rounding allowed). */
export const inFrame = (poly: Pt[]): boolean => poly.every(([x, y]) => x >= -1e-4 && x <= 1.0001 && y >= -1e-4 && y <= 1.0001);

const onFrame = (poly: Pt[]): Pt[] => poly.map(([x, y]): Pt => [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))]);

/** The bays `ids` turned together as one rigid block about `centre` (default: the middle of those bays): their spacing and angles to each other are kept, the
 * whole grid swings. null when a corner would leave the picture (nothing is changed then). */
export function rotateGroup(all: DrawnSlot[], ids: number[], w: number, h: number, deg: number, centre?: Pt): DrawnSlot[] | null {
  const own = all.filter((s) => ids.includes(s.id));
  if (!own.length || !deg) return all;
  const c = centre ?? groupCentre(own.map((s) => s.poly));
  const turned = own.map((s) => turnPoints(s.poly, c, w, h, deg));
  if (!turned.every(inFrame)) return null;
  const by = new Map(own.map((s, i) => [s.id, onFrame(turned[i])]));
  return all.map((s) => (by.has(s.id) ? { ...s, poly: by.get(s.id)! } : s));
}

/** How the depth of a group is changed: `k` times as deep, or the back edge `px` pixels up the picture (negative: down). The front line stays where it is. */
export type DepthSpec = { k: number } | { px: number };

/** The back / top edge of a whole group of bays moved on its own (perspective depth): every point of the bays (and of `extra` polygons such as a zone's
 * outline) is moved along the group's depth direction `v`, in proportion to its distance from the group's FRONT line, so the front baseline does not move
 * and the side edges stay straight lines (the rows' perspective lines are kept, only the depth changes). `v` is the bays' mean back direction, the front
 * line runs through every bay's front corners. null when there is no depth direction, or a corner would leave the picture (nothing is changed then). */
export function depthGroup(all: DrawnSlot[], ids: number[], w: number, h: number, spec: DepthSpec, extra: Pt[][] = []): { slots: DrawnSlot[]; extra: Pt[][] } | null {
  const own = all.filter((s) => ids.includes(s.id) && s.poly.length === 4);
  if (!own.length) return null;
  const px = own.map((s) => s.poly.map(([x, y]): V => [x * w, y * h]));
  const sum = (f: (p: V[]) => V): V => px.reduce((m, p) => ((a) => [m[0] + a[0], m[1] + a[1]] as V)(f(p)), [0, 0] as V);
  const v = unit(sum((p) => [p[3][0] - p[0][0] + p[2][0] - p[1][0], p[3][1] - p[0][1] + p[2][1] - p[1][1]]));
  const u = unit(sum((p) => [p[1][0] - p[0][0], p[1][1] - p[0][1]]));
  if (!v || !u) return null;
  const fronts = px.flatMap((p) => [p[0], p[1]]);
  const c: V = [fronts.reduce((m, q) => m + q[0], 0) / fronts.length, fronts.reduce((m, q) => m + q[1], 0) / fronts.length];
  const vu = v[0] * u[1] - v[1] * u[0]; // v x u
  if (Math.abs(vu) < 1e-3) return null; // the depth runs along the front line: no depth to change
  const t = (q: V) => ((q[0] - c[0]) * u[1] - (q[1] - c[1]) * u[0]) / vu; // how far along v q lies from the front line
  const depth = Math.max(...px.flat().map(t));
  if (!(depth > 1e-6)) return null;
  const k = "k" in spec ? spec.k : 1 + ((v[1] < 0 ? 1 : -1) * spec.px) / depth;
  if (!(k > 0.05)) return null;
  const move = (poly: Pt[]): Pt[] => poly.map(([x, y]): Pt => {
    const q: V = [x * w, y * h];
    const d = (k - 1) * t(q);
    return [Math.round(((q[0] + d * v[0]) / w) * 1e4) / 1e4, Math.round(((q[1] + d * v[1]) / h) * 1e4) / 1e4];
  });
  const slots = all.map((s) => (ids.includes(s.id) && s.poly.length === 4 ? { ...s, poly: move(s.poly) } : s));
  const more = extra.map((p) => (p.length ? move(p) : p));
  if (![...slots.filter((s) => ids.includes(s.id)).map((s) => s.poly), ...more].every(inFrame)) return null;
  return { slots, extra: more };
}
