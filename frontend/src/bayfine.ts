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
