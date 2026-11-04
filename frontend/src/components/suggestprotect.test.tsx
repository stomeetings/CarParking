import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { type Bay, type SuggestView, acceptText, protectedText } from "../suggest";
import type { Snapshot } from "../types";
import { SuggestSlots } from "./SuggestSlots";

/** A Fixed camera's draft keeps its running slots (B1.., the pin P1) as they run: shown as protected, not draggable or removable, not ticked for the
 * merge, and the new bays are the only ones that can change. */
beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0, toJSON: () => ({}) });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const snapshot = { shape: [2000, 1000] } as unknown as Snapshot;
const sq = (x: number, y = 0.5): Bay["poly"] => [[x, y], [x + 0.1, y], [x + 0.1, y + 0.05], [x, y + 0.05]];
const run = (id: number, x: number, extra: Partial<Bay> = {}): Bay => ({ id, name: id === 9 ? "P1" : `B${id}`, poly: sq(x), lift: 0, source: "running", protected: true, ...extra });
const add = (id: number, x: number): Bay => ({ id, name: `B${id}`, poly: sq(x, 0.3), lift: 0.01, source: "measured" });

const VIEW: SuggestView = {
  version: 2, state: "draft", created_at: 1700000000, note: "", width: 2000, height: 1000,
  meta: { parent: 1, mode: "quick", kind: "fixed", lined_up: "vehicles", expected: null, vehicles: 3 },
  grid: [run(1, 0.1), run(2, 0.25), run(9, 0.6, { capacity: 4 }), add(10, 0.1), add(11, 0.3)],
  calibration: { confidence: "HIGH", h: 3.2, version: 1 },
  baseline: { kind: "fixed", slots: [run(1, 0.1), run(2, 0.25), run(9, 0.6, { capacity: 4 })].map((b) => ({ ...b, source: undefined, protected: undefined })) },
  diff: { added: [10, 11], removed: [], moved: [], unchanged: [1, 2, 9] }, pins: [], stale: false,
};

function serve(view: SuggestView = VIEW) {
  const calls: { method: string; url: string; body: unknown }[] = [];
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    calls.push({ method, url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    if (url.endsWith("/suggest") && method === "GET") return json({ suggestions: [] });
    if (url.endsWith("/suggest") && method === "POST") {
      return json({ id: "j1", kind: "suggest", state: "done", stage: "done", progress: 1, error: null, result: { version: 2, bays: 5, mode: "quick", lined_up: "vehicles", warnings: [], diff: view.diff } });
    }
    if (/\/suggest\/2$/.test(url)) return json(view);
    if (url.includes("/accept")) return json({ applied: "fixed" });
    return json({});
  });
  return calls;
}
const show = () => render(<SuggestSlots camId="camera-1" cam="camera-1" snapshot={snapshot} image="data:image/jpeg;base64,xx" stillId={"a".repeat(32)}
                                         tokenKey="k" onNeedAdmin={vi.fn()} />);
const suggestNow = async () => {
  fireEvent.click(screen.getByText(/Auto-detect \/ suggest slots/));
  await screen.findByTestId("suggest-banner");
};

describe("the draft with protected running slots", () => {
  it("says which running slots are kept, none removed or moved, and what was added", async () => {
    serve();
    show();
    await suggestNow();
    expect(screen.getByTestId("suggest-protected").textContent).toBe(
      "3 running slot(s) are protected (B1, B2, P1): the draft keeps them exactly as they run, never marks them removed or moved, and adds 2 new bay(s) that do not lie on them.");
    expect(screen.getByTestId("suggest-diff").textContent).toBe("2 added · 0 moved · 0 removed · 3 unchanged");
    expect(screen.queryAllByTestId("removed-bay")).toHaveLength(0);
  });

  it("marks the protected bays, draws them solid and lets only the new bays be dragged or removed", async () => {
    serve();
    show();
    await suggestNow();
    const bays = screen.getAllByTestId("draft-bay");
    const [b1, , , n10] = bays;
    expect(b1.getAttribute("data-protected")).toBe("true");
    expect(b1.querySelector("polygon")!.getAttribute("stroke-dasharray")).toBeNull();
    expect(n10.getAttribute("data-protected")).toBeNull();
    expect(n10.querySelector("polygon")!.getAttribute("stroke-dasharray")).toBe("3 4");
    expect(screen.getAllByTestId("protected-badge")).toHaveLength(3);
    // a click selects but offers no corners to drag
    fireEvent.mouseDown(b1.querySelector("polygon")!, { button: 0, clientX: 150, clientY: 260 });
    expect(screen.queryAllByTestId("vertex")).toHaveLength(0);
    fireEvent.mouseDown(n10.querySelector("polygon")!, { button: 0, clientX: 150, clientY: 160 });
    expect(screen.queryAllByTestId("vertex")).toHaveLength(4);
    // no remove button on a protected row, one on each new row
    const rows = within(screen.getByTestId("suggest-bays")).getAllByRole("listitem");
    expect(rows.map((r) => r.querySelector('button[title="Remove this bay from the draft"]') !== null)).toEqual([false, false, false, true, true]);
    expect(screen.getByText("holds 4")).toBeTruthy();
  });

  it("dragging a protected bay does not move it", async () => {
    serve();
    show();
    await suggestNow();
    const poly = screen.getAllByTestId("draft-bay")[0].querySelector("polygon")!;
    const before = poly.getAttribute("points");
    fireEvent.mouseDown(poly, { button: 0, clientX: 150, clientY: 260 });
    fireEvent.mouseMove(screen.getByTestId("suggest-svg"), { clientX: 400, clientY: 300 });
    fireEvent.mouseUp(screen.getByTestId("suggest-svg"));
    expect(screen.getAllByTestId("draft-bay")[0].querySelector("polygon")!.getAttribute("points")).toBe(before);
  });

  it("does not tick the running slots for the merge, and the confirmation says they stay as they run", async () => {
    serve();
    show();
    await suggestNow();
    const ticked = (name: string) => (screen.getByLabelText(`take ${name}`) as HTMLInputElement).checked;
    expect([ticked("B1"), ticked("B2"), ticked("P1"), ticked("B10"), ticked("B11")]).toEqual([false, false, false, true, true]);
    fireEvent.click(screen.getByText("Accept"));
    expect(screen.getByTestId("suggest-confirm").textContent).toContain("The running slots in the draft are protected: they stay exactly as they run.");
  });

  it("an auto-grid camera draft shows none of it", async () => {
    serve({ ...VIEW, meta: { ...VIEW.meta, kind: "auto" }, grid: [add(1, 0.1), add(2, 0.3)], baseline: { kind: "auto", slots: [] }, diff: { added: [], removed: [], moved: [], unchanged: [1, 2] } });
    show();
    await suggestNow();
    expect(screen.queryByTestId("suggest-protected")).toBeNull();
    expect(screen.queryAllByTestId("protected-badge")).toHaveLength(0);
  });
});

describe("an auto-grid camera draft (camera-1)", () => {
  it("keeps its running bays too, with none removed or moved, and the confirmation says they keep their ids and shapes", async () => {
    const auto: SuggestView = { ...VIEW, meta: { ...VIEW.meta, kind: "auto" }, grid: [run(1, 0.1, { source: "measured" }), run(2, 0.25, { source: "measured" }), add(10, 0.1)],
                                baseline: { kind: "auto", slots: [] }, diff: { added: [10], removed: [], moved: [], unchanged: [1, 2] },
                                pins: [{ ...run(9, 0.6, { capacity: 4, source: "manual", protected: undefined }), binds_to: 2 }] };
    serve(auto);
    show();
    await suggestNow();
    expect(screen.getByTestId("suggest-protected").textContent).toContain("2 running slot(s) are protected (B1, B2)");
    expect(screen.getByTestId("suggest-diff").textContent).toBe("1 added · 0 moved · 0 removed · 2 unchanged");
    expect(screen.getAllByTestId("pin-bay")).toHaveLength(1); // P1 stays a pin
    fireEvent.click(screen.getByText("Accept"));
    expect(screen.getByTestId("suggest-confirm").textContent).toContain("The running bays in the draft are protected: they keep their ids and shapes exactly.");
  });
});

describe("the helpers", () => {
  it("protectedText is null without protected bays and acceptText only mentions them when a draft holds some", () => {
    expect(protectedText([add(1, 0.1)])).toBeNull();
    expect(acceptText({ ...VIEW, grid: [add(1, 0.1)] }, 1)).not.toContain("protected");
    expect(acceptText({ ...VIEW, meta: { ...VIEW.meta, kind: "auto" }, grid: [add(1, 0.1)] }, 2)).not.toContain("protected"); // (nothing protected, nothing said)
  });
});
