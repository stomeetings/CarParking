import { describe, expect, it } from "vitest";
import { addSlot, area, fromJson, mapProblems, moveSlot, moveVertex, nearFirst, pointInPoly, growPoly, rectPoly, setCapacity, toJson, type DrawnSlot } from "./slotdraw";

const SIZE: [number, number] = [2560, 1440];
const square = (x: number, y: number, d = 0.1) => [[x, y], [x + d, y], [x + d, y + d], [x, y + d]] as [number, number][];

describe("drawing", () => {
  it("adds a clicked polygon as the next slot, rounded to 4 decimals", () => {
    const r = addSlot([], [[0.1, 0.3], [0.30004, 0.3], [0.3, 0.5], [0.1, 0.5]]);
    expect(r.error).toBeUndefined();
    expect(r.slots).toEqual([{ id: 1, name: "S1", poly: [[0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.1, 0.5]] }]);
    expect(addSlot(r.slots, square(0.5, 0.5), "A2").slots[1]).toMatchObject({ id: 2, name: "A2" });
  });
  it("refuses what the server would refuse", () => {
    expect(addSlot([], [[0.1, 0.1], [0.2, 0.2]]).error).toMatch(/3 corners/);
    expect(addSlot([], [[0.1, 0.1], [0.10001, 0.1], [0.1, 0.10001]]).error).toMatch(/too small/);
  });
  it("closes the polygon when the click lands near the first corner (screen pixels)", () => {
    const draft = [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]] as [number, number][];
    expect(nearFirst(draft, [0.103, 0.104], SIZE)).toBe(true); // 7.7 px, 5.8 px away
    expect(nearFirst(draft, [0.11, 0.1], SIZE)).toBe(false); // 25.6 px away
    expect(nearFirst(draft.slice(0, 2), [0.1, 0.1], SIZE)).toBe(false); // needs 3 corners first
  });
  it("turns a drag into a rectangle, whichever way it was dragged", () => {
    expect(rectPoly([0.4, 0.5], [0.2, 0.3])).toEqual([[0.2, 0.3], [0.4, 0.3], [0.4, 0.5], [0.2, 0.5]]);
    expect(area(rectPoly([0.2, 0.3], [0.4, 0.5]))).toBeCloseTo(0.04);
  });
  it("moves a corner, and a whole slot without leaving the frame", () => {
    const slots: DrawnSlot[] = [{ id: 3, name: "A", poly: square(0.85, 0.1) }];
    expect(moveVertex(slots, 3, 1, [1.2, 0.05])[0].poly[1]).toEqual([1, 0.05]);
    expect(moveSlot(slots, 3, 0.2, 0)[0].poly.map((p) => p[0])).toEqual([0.9, 1, 1, 0.9]); // stopped at the edge
  });
});

describe("JSON export / import", () => {
  const slots: DrawnSlot[] = [{ id: 1, name: "A1", poly: [[0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.1, 0.5]] }];
  it("exports shares of the frame and camera pixels", () => {
    const out = JSON.parse(toJson("Camera 1", slots, SIZE));
    expect(out.camera).toBe("Camera 1");
    expect(out.frame).toEqual({ width: 2560, height: 1440 });
    expect(out.slots[0].poly).toEqual(slots[0].poly);
    expect(out.slots[0].points_px).toEqual([[256, 432], [768, 432], [768, 720], [256, 720]]);
  });
  it("reads its own export, the server file's per-camera entry, a bare list, or pixels", () => {
    expect(fromJson(toJson("Camera 1", slots, SIZE), SIZE)).toEqual(slots);
    expect(fromJson(JSON.stringify({ updated: "x", slots }), SIZE)).toEqual(slots);
    expect(fromJson(JSON.stringify(slots), SIZE)).toEqual(slots);
    const px = [{ id: 1, name: "A1", points_px: [[256, 432], [768, 432], [768, 720], [256, 720]] }];
    expect(fromJson(JSON.stringify(px), SIZE)).toEqual(slots);
  });
  it("refuses a broken map with the reason", () => {
    expect(() => fromJson("{}", SIZE)).toThrow(/list of slots/);
    expect(() => fromJson(JSON.stringify([{ id: 1, poly: [[0.1, 0.1], [0.2, 0.2]] }]), SIZE)).toThrow(/3 corners/);
    expect(mapProblems([...slots, { ...slots[0] }])).toEqual(["slot 1: duplicate id"]);
  });
});

describe("capacity (vehicles per slot)", () => {
  const base: DrawnSlot[] = [{ id: 1, name: "P1", poly: [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]] }];
  it("is set from the typed text; empty or 1 is the default and drops the key", () => {
    const two = setCapacity(base, 1, "2");
    expect(two[0].capacity).toBe(2);
    expect("capacity" in setCapacity(two, 1, "")).toBe(false);
    expect("capacity" in setCapacity(two, 1, "1")[0]).toBe(false);
  });
  it("a bad value is reported, never changed silently", () => {
    for (const t of ["0", "2.5", "21", "abc"]) expect(mapProblems(setCapacity(base, 1, t))[0]).toContain("capacity");
    expect(mapProblems(setCapacity(base, 1, "4"))).toEqual([]);
  });
  it("survives the JSON export and import", () => {
    const text = toJson("cam", setCapacity(base, 1, "3"), [1000, 500]);
    expect(fromJson(text, [1000, 500])[0].capacity).toBe(3);
    expect("capacity" in fromJson(toJson("cam", base, [1000, 500]), [1000, 500])[0]).toBe(false);
  });
});

describe("capacity pin helpers", () => {
  const box: [number, number][] = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]];
  it("finds the points inside a polygon (a bay centre under a pin)", () => {
    expect(pointInPoly([0.5, 0.5], box)).toBe(true);
    expect(pointInPoly([0.7, 0.5], box)).toBe(false);
    expect(pointInPoly([0.5, 0.3], box)).toBe(false);
    expect(pointInPoly([0.1, 0.1], [[0, 0], [1, 0]])).toBe(false);
  });
  it("grows a polygon outward by a distance, whichever way it is wound", () => {
    expect(growPoly(box, 0.1)).toEqual([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]);
    expect(growPoly([...box].reverse(), 0.1).map(([x, y]) => [x, y])).toEqual([[0.3, 0.7], [0.7, 0.7], [0.7, 0.3], [0.3, 0.3]]);
    expect(growPoly(box, 0)).toEqual(box);
    expect(growPoly([[0, 0], [1, 1]], 0.1)).toEqual([[0, 0], [1, 1]]);
  });
});
