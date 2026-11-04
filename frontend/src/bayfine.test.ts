import { describe, expect, it } from "vitest";
import { resizeBay, rotateBay, shiftBays, tuneBays } from "./bayfine";
import type { DrawnSlot, Pt } from "./slotdraw";

const W = 1600;
const H = 900;
const bay = (id: number, x: number, y = 0.5): DrawnSlot => ({ id, name: `A${id}`, poly: [[x, y + 0.2], [x + 0.1, y + 0.2], [x + 0.12, y], [x + 0.02, y]] });
const px = (poly: Pt[]) => poly.map(([x, y]) => [x * W, y * H]);
const len = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);

describe("fine adjustment of bays", () => {
  it("shifts the chosen bays together and stops at the picture's edge without squeezing them", () => {
    const all = [bay(1, 0.1), bay(2, 0.3), bay(3, 0.8)];
    const moved = shiftBays(all, [1, 2], 0.01, -0.02);
    expect(moved[0].poly[0]).toEqual([0.11, 0.68]);
    expect(moved[1].poly[0]).toEqual([0.31, 0.68]);
    expect(moved[2]).toBe(all[2]); // another bay is not touched
    const edge = shiftBays(all, [1, 3], 0.5, 0); // bay 3 ends at 0.92: both stop 0.08 on, their distance stays
    expect(edge[2].poly[2][0]).toBeCloseTo(1, 4); // its far corner is on the edge
    expect(edge[0].poly[0][0] - 0.1).toBeCloseTo(0.08, 4);
  });

  it("widens, lengthens and shrinks a bay about its centre along the bay's own directions", () => {
    const poly = bay(1, 0.4).poly;
    const wide = resizeBay(poly, W, H, 1.1, 1);
    const a = px(poly);
    const b = px(wide);
    expect(len(b[0], b[1]) / len(a[0], a[1])).toBeCloseTo(1.1, 1); // the front is 10 % wider
    expect(len(b[0], b[3]) / len(a[0], a[3])).toBeCloseTo(1, 1); // its side is as long (up to the shear of the slant)
    const small = px(resizeBay(poly, W, H, 0.9, 0.9));
    expect(len(small[0], small[1]) / len(a[0], a[1])).toBeCloseTo(0.9, 1);
    expect(len(small[0], small[3]) / len(a[0], a[3])).toBeCloseTo(0.9, 1);
    const mid = (p: number[][]) => [p.reduce((m, q) => m + q[0], 0) / 4, p.reduce((m, q) => m + q[1], 0) / 4];
    expect(mid(small)[0]).toBeCloseTo(mid(a)[0], 0); // it shrinks about its own centre
    expect(mid(small)[1]).toBeCloseTo(mid(a)[1], 0);
    resizeBay(poly, W, H, 1, 1).forEach(([x, y], i) => (expect(x).toBeCloseTo(poly[i][0], 3), expect(y).toBeCloseTo(poly[i][1], 3))); // 1 x 1: unchanged
  });

  it("turns a bay about its centre keeping its size, and a full turn is the same bay", () => {
    const poly = bay(1, 0.4).poly;
    const a = px(poly);
    const t = px(rotateBay(poly, W, H, 2));
    expect(len(t[0], t[1])).toBeCloseTo(len(a[0], a[1]), 0);
    expect(len(t[0], t[3])).toBeCloseTo(len(a[0], a[3]), 0);
    const ang = (p: number[][]) => Math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]) * 180 / Math.PI;
    expect(ang(t) - ang(a)).toBeCloseTo(2, 0);
    const full = px(rotateBay(poly, W, H, 360));
    full.forEach((q, i) => expect(Math.hypot(q[0] - a[i][0], q[1] - a[i][1])).toBeLessThan(1));
  });

  it("tuneBays changes only the chosen bays and keeps ids and names", () => {
    const all = [bay(1, 0.1), bay(2, 0.3)];
    const out = tuneBays(all, [2], (p) => rotateBay(p, W, H, 1));
    expect(out[0]).toBe(all[0]);
    expect([out[1].id, out[1].name]).toEqual([2, "A2"]);
    expect(out[1].poly).not.toEqual(all[1].poly);
  });
});
