import { describe, expect, it } from "vitest";
import {
  type Bay, type SuggestView, acceptBlock, acceptText, diffText, gridBody, pinText, sameGrid, statusOf,
} from "./suggest";

const sq = (x: number, y = 0.5): Bay["poly"] => [[x, y], [x + 0.1, y], [x + 0.1, y + 0.1], [x, y + 0.1]];
const bay = (id: number, x: number, extra: Partial<Bay> = {}): Bay => ({ id, name: `B${id}`, poly: sq(x), lift: 0.01, source: "measured", ...extra });
const diff = { added: [4], removed: [3], moved: [2], unchanged: [1] };

const view = (over: Partial<SuggestView> = {}): SuggestView => ({
  version: 2, state: "draft", created_at: 1, note: "", grid: [bay(1, 0.1)], width: 1920, height: 1080,
  meta: { parent: 1, mode: "quick", kind: "auto", lined_up: "vehicles", expected: null },
  calibration: { confidence: "HIGH", h: 3.2, version: 1 }, baseline: { kind: "auto", slots: [] }, diff, pins: [], stale: false, ...over,
});

describe("how a draft bay compares with what runs now", () => {
  it("names added, moved and unchanged bays, and an edited bay is edited whatever the diff says", () => {
    expect(statusOf(bay(4, 0.4), diff)).toBe("added");
    expect(statusOf(bay(2, 0.2), diff)).toBe("moved");
    expect(statusOf(bay(1, 0.1), diff)).toBe("unchanged");
    expect(statusOf(bay(4, 0.4, { source: "edited" }), diff)).toBe("edited");
    expect(statusOf(bay(9, 0.9), null)).toBe("unchanged"); // nothing to compare with
  });

  it("says it in a line", () => {
    expect(diffText(diff)).toBe("1 added · 1 moved · 1 removed · 1 unchanged");
    expect(diffText(null)).toContain("nothing runs");
  });
});

describe("pins are anchors", () => {
  it("shows the slot id, the bay it binds to and the capacity", () => {
    expect(pinText({ ...bay(9, 0.5), name: "P1", binds_to: 7, capacity: 4 })).toBe("P1 · slot 9 · next to B7 · holds 4");
    expect(pinText({ ...bay(9, 0.5), name: "P2" })).toBe("P2 · slot 9");
  });
});

describe("accepting", () => {
  it("tells an auto-grid camera that the live grid switches, and a Fixed camera that its drawn slots stay", () => {
    expect(acceptText(view({ pins: [bay(9, 0.5, { source: "manual" })] }), 3)).toMatch(/Switch the live grid.*1 added, 1 moved, 1 removed.*1 pin\(s\) keep their slot ids/s);
    const fixed = view({ meta: { parent: 1, mode: "quick", kind: "fixed", lined_up: "vehicles", expected: null } });
    expect(acceptText(fixed, 2)).toMatch(/Merge 2 suggested bay\(s\) into the hand-drawn map.*stay as drawn/s);
  });

  it("is blocked for a stale or finished draft, and for a Fixed camera with nothing ticked", () => {
    expect(acceptBlock(view(), 1)).toBeNull();
    expect(acceptBlock(view({ stale: true }), 1)).toMatch(/changed since/);
    expect(acceptBlock(view({ state: "active" }), 1)).toMatch(/is active/);
    const fixed = view({ meta: { parent: 1, mode: "quick", kind: "fixed", lined_up: "vehicles", expected: null } });
    expect(acceptBlock(fixed, 0)).toMatch(/at least one/);
    expect(acceptBlock(fixed, 1)).toBeNull();
  });
});

describe("the draft that is saved", () => {
  it("sends ids, names, shapes and lift only, and compares by those", () => {
    expect(gridBody([bay(1, 0.1, { source: "edited", binds_to: 3 })])).toEqual({ grid: [{ id: 1, name: "B1", poly: sq(0.1), lift: 0.01 }] });
    expect(sameGrid([bay(1, 0.1)], [bay(1, 0.1, { source: "edited" })])).toBe(true); // the source is the server's to decide
    expect(sameGrid([bay(1, 0.1)], [bay(1, 0.11)])).toBe(false);
    expect(sameGrid([bay(1, 0.1)], [])).toBe(false);
  });
});
