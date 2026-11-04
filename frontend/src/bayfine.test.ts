import { describe, expect, it } from "vitest";
import { resizeBay, rotateBay, shiftBays, splitBay, tuneBays } from "./bayfine";
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

describe("splitBay", () => {
  const ground = (poly: Pt[]) => poly; // (the bay below is a perspective quad: its far side is narrower)
  const quad: Pt[] = [[0.3, 0.8], [0.5, 0.8], [0.46, 0.4], [0.34, 0.4]];

  it("cuts a bay into two that tile it exactly, in the bay's own order, with no gap or overlap", () => {
    const [a, b] = splitBay(quad, "width");
    expect(a[1]).toEqual(b[0]); // they share the cut
    expect(a[2]).toEqual(b[3]);
    expect(a[0]).toEqual(quad[0]);
    expect(b[1]).toEqual(quad[1]);
    expect(b[2]).toEqual(quad[2]);
    const [f, k] = splitBay(quad, "length");
    expect(f[3]).toEqual(k[0]);
    expect(f[2]).toEqual(k[1]);
    expect(f[0]).toEqual(quad[0]);
    expect(k[3]).toEqual(quad[3]);
  });

  it("cuts equal on the ground, not on the picture: the near half looks longer than the far half", () => {
    const [near, far] = splitBay(quad, "length");
    const side = (p: Pt[]) => Math.hypot(p[3][0] - p[0][0], p[3][1] - p[0][1]); // the left side of each half
    expect(side(near)).toBeGreaterThan(side(far));
    const mid = near[3][1];
    expect(mid).toBeLessThan(0.6); // the picture's midpoint of the sides is at y 0.6: the ground's is farther
    const [l, r] = splitBay(quad, "width");
    expect(Math.hypot(l[1][0] - l[0][0], l[1][1] - l[0][1])).toBeCloseTo(Math.hypot(r[1][0] - r[0][0], r[1][1] - r[0][1]), 3); // a front edge halves evenly
  });

  it("a plain rectangle splits in half, thirds work, and a bay that is not four-cornered is left alone", () => {
    const rect: Pt[] = [[0.2, 0.6], [0.4, 0.6], [0.4, 0.2], [0.2, 0.2]];
    const [a, b] = splitBay(rect, "width");
    expect(a[1][0]).toBeCloseTo(0.3, 4);
    expect(b[0][0]).toBeCloseTo(0.3, 4);
    expect(splitBay(rect, "length", 3)).toHaveLength(3);
    expect(splitBay([[0, 0], [1, 0], [1, 1]], "width")).toHaveLength(1);
    expect(ground(quad)).toBe(quad);
  });
});
