import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { BLOCK_HATCH, SLOT_COLORS } from "../lib";
import type { Slot } from "../types";
import { Sidebar } from "./Layout";
import { type Expected, ResultPanel, type SampleResult, observed } from "./SampleResult";

/** The result of a sample test: the slots coloured on the picture, the dashboard's cards for it, the notices the run produced, and the check against
 * what you expect. (The flow that produces it is in samplesession.test.tsx.) */
afterEach(cleanup);

const box = (x: number): [number, number][] => [[x, 0.2], [x + 0.1, 0.2], [x + 0.1, 0.4], [x, 0.4]];
const slot = (id: number, state: Slot["state"], x: number, extra: Partial<Slot> = {}): Slot =>
  ({ id, name: `A${id}`, state, hidden: false, phantom: false, duplicate_of: null, poly: box(x), ...extra }) as Slot;

const RESULT: SampleResult = {
  shape: [2000, 1000],
  counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 4, learning: 0, encroached: 1, available: 1, blocked_objects: 1 },
  slots: [
    slot(1, "BUSY", 0.05, { avail: null, display: "BUSY", blocked_by: [], encroached_by: [] }),
    slot(2, "FREE", 0.25, { avail: "ENCROACHED", display: "ENCROACHED", encroached_by: ["A1"], cover: 0.42, blocked_by: [] }),
    slot(3, "FREE", 0.45, { avail: null, display: "FREE", blocked_by: [], encroached_by: [] }),
    slot(4, "FREE", 0.65, { avail: "BLOCKED", display: "BLOCKED", blocked_by: ["potted plant"], encroached_by: [] }),
  ],
  detections: [
    { label: "car", conf: 0.9, src: "", kind: "vehicle", box: [0.06, 0.2, 0.3, 0.4] },
    { label: "potted plant", conf: 0.7, src: "", kind: "other", box: [0.68, 0.3, 0.72, 0.4] },
    { label: "car", conf: 0.5, src: "", kind: "vehicle", box: [0.9, 0.9, 0.92, 0.95], rejected: "too_small" },
  ],
  area: { top: 0, bottom: 0, left: 0, right: 0, roi: null }, vehicles: 1, others: 1, obstacle: { rejected: 1, reasons: { too_small: 1 } },
  events: [
    { id: 1, slot_id: 1, event: "entry", time: "2026-11-02T12:00:00+05:00", ts: 1, session_id: 1, note: "" },
    { id: 2, slot_id: 2, event: "encroached", time: "2026-11-02T12:00:20+05:00", ts: 2, session_id: null, note: "by A1, 42 % of the bay: free but not usable" },
    { id: 3, slot_id: 4, event: "object_blocked", time: "2026-11-02T12:00:25+05:00", ts: 3, session_id: null, note: "by potted plant: free but not usable" },
  ],
  layers: { spillage: true, obstacle_filter: true, blocked_objects: true, ground: false }, frames: 16, step_s: 3, seconds: 2.4, kind: "fixed", slot_count: 4,
  source: "upload", name: null, image: "data:image/jpeg;base64,AAAA",
};

const NONE: Expected = { free: "", busy: "", blocked: "" };
const showResult = (expected: Expected = NONE) => render(<ResultPanel result={RESULT} label="lot.jpg" expected={expected} />);

describe("a result", () => {
  it("zooms the result picture in and out, and fits it again, so a single bay can be inspected", () => {
    showResult();
    const layer = screen.getByTestId("res-layer");
    expect(screen.getByTestId("res-zoom").textContent).toBe("100 %");
    expect((screen.getByTestId("res-zoom-out") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByTestId("res-zoom-in"));
    expect(screen.getByTestId("res-zoom").textContent).toBe("150 %");
    expect(layer.style.transform).toContain("scale(1.5)");
    fireEvent.click(screen.getByTestId("res-zoom-reset"));
    expect(screen.getByTestId("res-zoom").textContent).toBe("100 %");
    expect(screen.getByTestId("sample-overlay")).toBeTruthy(); // the slots and boxes live inside the zoomed layer
  });

  it("draws an oriented (YOLO-OBB) detection as a tilted yellow outline, an axis-aligned one as a box", () => {
    const tilted = { label: "busy_slot", conf: 0.8, src: "", kind: "vehicle" as const, box: [0.1, 0.1, 0.3, 0.3] as [number, number, number, number],
                     poly: [[0.2, 0.1], [0.3, 0.2], [0.2, 0.3], [0.1, 0.2]] as [number, number][] };
    render(<ResultPanel result={{ ...RESULT, detections: [tilted] }} label="lot.jpg" expected={NONE} />);
    const obb = within(screen.getByTestId("sample-overlay")).getByTestId("overlay-obb");
    expect(obb.getAttribute("stroke")).toBe("#f9b115");
    expect(obb.getAttribute("points")?.split(" ")).toHaveLength(4);
    expect(within(screen.getByTestId("sample-overlay")).queryAllByTestId("overlay-box")).toHaveLength(0);
  });

  it("", () => {
    showResult();
    const overlay = screen.getByTestId("sample-overlay");
    expect((screen.getByAltText("Sample picture") as HTMLImageElement).src).toBe(RESULT.image); // the picture that was tested
    const drawn = within(overlay).getAllByTestId("overlay-slot");
    const fill = (i: number) => drawn[i].querySelector("polygon")!.getAttribute("fill");
    expect(drawn.map((s) => s.getAttribute("data-avail") ?? s.getAttribute("data-state"))).toEqual(["BUSY", "ENCROACHED", "FREE", "BLOCKED"]);
    expect(fill(0)).toBe(SLOT_COLORS.BUSY.fill);
    expect(fill(2)).toBe(SLOT_COLORS.FREE.fill);
    expect(fill(3)).toBe(`url(#${BLOCK_HATCH})`); // cross-hatched slate
    expect(drawn[3].textContent).toContain("BLOCKED");
    expect(drawn[3].querySelector("polygon")!.getAttribute("stroke")).toBe(SLOT_COLORS.BLOCKED.stroke);
    const boxes = within(overlay).getAllByTestId("overlay-box");
    expect(boxes.map((b) => b.getAttribute("stroke"))).toEqual(["#f9b115", "#d63384", "#9da5b1"]); // vehicle, other object, ignored
    expect(boxes[2].getAttribute("data-rejected")).toBe("too_small");
  });

  it("", () => {
    showResult();
    const cards = within(screen.getByTestId("counters"));
    expect(cards.getByText("Free").previousSibling!.textContent).toBe("1");
    expect(cards.getByText("Busy").previousSibling!.textContent).toBe("1");
    expect(screen.getByTestId("blocked-card").textContent).toContain("Blocked by object (free, not usable)");
    expect(screen.getByTestId("blocked-card").firstElementChild!.textContent).toContain("1");
    expect(screen.getByTestId("encroached-card").firstElementChild!.textContent).toContain("1");
    expect(screen.getByTestId("sample-summary").textContent).toContain("1 vehicle(s), 1 other object(s) in 2.4 s");
    expect(screen.getByTestId("sample-summary").textContent).toContain("Spillage protection, Obstacle filter, Blocked by object");
    expect(screen.getByTestId("sample-summary").textContent).toContain("obstacle filter dropped 1");
  });

  it("", () => {
    showResult();
    const items = within(screen.getByTestId("ticker")).getAllByRole("listitem").map((li) => li.textContent ?? "");
    expect(items[0]).toContain("Slot 4 · bay blocked by an object");
    expect(items[1]).toContain("Slot 2 · bay encroached by a neighbour");
    expect(items[2]).toContain("Slot 1 · vehicle in");
    const rows = screen.getAllByTestId("sample-slot-row");
    expect(rows.map((r) => r.getAttribute("data-display"))).toEqual(["BUSY", "ENCROACHED", "FREE", "BLOCKED"]);
    expect(rows[3].textContent).toContain("blocked by potted plant");
    expect(screen.getByTestId("sample-legend").textContent).toContain("Blocked by object (cross-hatched)");
  });

  it("", () => {
    showResult({ free: "1", busy: "2", blocked: "1" });
    const chips = within(screen.getByTestId("sample-checks")).getAllByText(/expected/);
    expect(chips.map((c) => [c.getAttribute("data-ok"), c.textContent])).toEqual([
      ["1", "✓ Free: expected 1, got 1"], ["0", "✗ Busy: expected 2, got 1"], ["1", "✓ Blocked: expected 1, got 1"]]);
  });

  it("observed() reads the usable free count and a missing blocked count as 0", () => {
    expect(observed(RESULT.counts)).toEqual({ free: 1, busy: 1, blocked: 1 });
    expect(observed({ free: 3, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 4, learning: 0 })).toEqual({ free: 3, busy: 1, blocked: 0 });
  });
});

describe("the sidebar", () => {
  it("has a Sample test page", () => {
    const pages: string[] = [];
    render(<Sidebar page="dashboard" open onNavigate={(p) => pages.push(p)} onClose={() => {}} />);
    fireEvent.click(screen.getAllByText("Sample test")[0]);
    expect(pages).toEqual(["sample"]);
  });
});
