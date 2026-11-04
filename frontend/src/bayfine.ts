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
