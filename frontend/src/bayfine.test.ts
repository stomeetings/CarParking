import { describe, expect, it } from "vitest";
import { depthGroup, groupCentre, inFrame, resizeBay, rotateBay, rotateGroup, shiftBays, splitBay, tuneBays, turnPoints } from "./bayfine";
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

describe("turning a whole group as one block", () => {
  const all = [bay(1, 0.3), bay(2, 0.45), bay(3, 0.6)];
  const dist = (a: DrawnSlot[], i: number, j: number) => len(px(a[i].poly)[0], px(a[j].poly)[0]);

  it("keeps every bay's size and the spacing between bays (a rigid turn), while the grid swings about its middle", () => {
    const turned = rotateGroup(all, [1, 2, 3], W, H, 5)!;
    expect(dist(turned, 0, 1)).toBeCloseTo(dist(all, 0, 1), 0);
    expect(dist(turned, 0, 2)).toBeCloseTo(dist(all, 0, 2), 0);
    const side = (a: DrawnSlot[], i: number) => len(px(a[i].poly)[0], px(a[i].poly)[1]);
    expect(side(turned, 1)).toBeCloseTo(side(all, 1), 0);
    const c = (a: DrawnSlot[]) => groupCentre(a.map((s) => s.poly));
    expect(c(turned)[0]).toBeCloseTo(c(all)[0], 3); // about its middle
    expect(c(turned)[1]).toBeCloseTo(c(all)[1], 3);
    expect(turned[0].poly).not.toEqual(all[0].poly);
    // the bays turned together swing by the angle: the line through the first and last bay's corners turns 5 degrees
    const ang = (a: DrawnSlot[]) => Math.atan2(px(a[2].poly)[0][1] - px(a[0].poly)[0][1], px(a[2].poly)[0][0] - px(a[0].poly)[0][0]);
    expect(((ang(turned) - ang(all)) * 180) / Math.PI).toBeCloseTo(5, 1);
  });

  it("only the chosen bays turn; the other bays stay; a turn that would leave the picture changes nothing", () => {
    const some = rotateGroup(all, [1, 2], W, H, 3)!;
    expect(some[2]).toBe(all[2]);
    const wide = [bay(1, 0), bay(2, 0.45), bay(3, 0.88)]; // as wide as the picture
    expect(rotateGroup(wide, [1, 2, 3], W, H, 90)).toBeNull(); // turned a quarter: its ends leave the frame
    expect(rotateGroup(all, [1, 2, 3], W, H, 90)).not.toBeNull(); // the narrower grid still fits
    expect(rotateGroup(all, [1, 2, 3], W, H, 0)!.map((s) => s.poly)).toEqual(all.map((s) => s.poly));
  });

  it("turnPoints is a pixel-space turn (a 16:9 picture keeps right angles) and inFrame tells when a corner is off the picture", () => {
    const sq: Pt[] = [[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]];
    const t = px(turnPoints(sq, [0.45, 0.45], W, H, 37));
    expect(len(t[0], t[1])).toBeCloseTo(0.1 * W, 0); // the sides keep their pixel length (160 x 90), so the angle at each corner stays a right angle
    expect(len(t[1], t[2])).toBeCloseTo(0.1 * H, 0);
    expect(len(t[0], t[2])).toBeCloseTo(Math.hypot(0.1 * W, 0.1 * H), 0);
    expect(inFrame(sq)).toBe(true);
    expect(inFrame([[0.5, 0.5], [1.2, 0.5], [0.5, 0.9]])).toBe(false);
  });
});

describe("the back edge of a group (perspective depth)", () => {
  // a row of bays: front edge low on the picture (y 0.7), back edge higher (y 0.5); the back is up the picture
  const row = [bay(1, 0.3), bay(2, 0.45), bay(3, 0.6)];
  const front = (a: DrawnSlot[]) => a.flatMap((s) => [s.poly[0], s.poly[1]]);
  const backs = (a: DrawnSlot[]) => a.flatMap((s) => [s.poly[2], s.poly[3]]);

  it("moves only the back edge: the front corners stay exactly, every back corner moves up by the step, widths keep their ratio", () => {
    const r = depthGroup(row, [1, 2, 3], W, H, { px: 18 })!; // 18 px up of 900 = 0.02
    expect(front(r.slots)).toEqual(front(row));
    backs(r.slots).forEach((q, i) => expect(q[1]).toBeCloseTo(backs(row)[i][1] - 0.02, 3));
    // the side edges stay straight lines through the old front corner: the back corner moved along the bay's own side, not straight up
    const [a, b, c] = [row[0].poly[1], row[0].poly[2], r.slots[0].poly[2]];
    const cross = (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
    expect(Math.abs(cross)).toBeLessThan(1e-3);
    const down = depthGroup(row, [1, 2, 3], W, H, { px: -18 })!; // pulled back down
    backs(down.slots).forEach((q, i) => expect(q[1]).toBeCloseTo(backs(row)[i][1] + 0.02, 3));
  });

  it("a ratio scales the depth about the front line (and the extra polygons, like a zone's outline, with it); an impossible change returns null", () => {
    const outline: Pt[] = [[0.25, 0.7], [0.8, 0.7], [0.82, 0.45], [0.27, 0.45]];
    const r = depthGroup(row, [1, 2, 3], W, H, { k: 0.5 }, [outline])!;
    expect(r.slots[0].poly[3][1]).toBeCloseTo(0.6, 3); // the old depth was 0.2: half of it
    expect(r.extra[0][0]).toEqual(outline[0]); // its front corners are on the front line: unchanged
    expect(r.extra[0][3][1]).toBeGreaterThan(0.45); // its back corners came down
    expect(depthGroup(row, [1, 2, 3], W, H, { px: 5000 })).toBeNull(); // off the picture
    expect(depthGroup(row, [1, 2, 3], W, H, { k: 0 })).toBeNull();
    expect(depthGroup(row, [], W, H, { k: 1.1 })).toBeNull();
  });
});
