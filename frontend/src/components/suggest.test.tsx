import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Bay, Diff, SuggestView } from "../suggest";
import type { Snapshot } from "../types";
import { SuggestSlots } from "./SuggestSlots";

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0, toJSON: () => ({}) });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  try {
    localStorage.clear();
  } catch {
    /* no storage */
  }
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const snapshot = { shape: [2000, 1000] } as unknown as Snapshot;
const sq = (x: number, y = 0.5): Bay["poly"] => [[x, y], [x + 0.1, y], [x + 0.1, y + 0.1], [x, y + 0.1]];
const bay = (id: number, x: number, extra: Partial<Bay> = {}): Bay => ({ id, name: `B${id}`, poly: sq(x), lift: 0.01, source: "measured", ...extra });
const DIFF: Diff = { added: [3], removed: [9], moved: [2], unchanged: [1] };

function viewOf(over: Partial<SuggestView> = {}): SuggestView {
  return {
    version: 2, state: "draft", created_at: 1700000000, note: "", width: 2000, height: 1000,
    meta: { parent: 1, mode: "quick", kind: "auto", lined_up: "vehicles", expected: null, vehicles: 3 },
    grid: [bay(1, 0.1), bay(2, 0.3), bay(3, 0.5)],
    calibration: { confidence: "HIGH", h: 3.2, version: 1 },
    baseline: { kind: "auto", slots: [bay(1, 0.1), bay(2, 0.31), bay(9, 0.7), bay(7, 0.8, { source: "manual" })] },
    diff: DIFF, pins: [{ ...bay(7, 0.8, { source: "manual" }), name: "P1", capacity: 4, binds_to: 3 }], stale: false, ...over,
  };
}

interface Net { calls: { method: string; url: string; body: unknown }[]; }
function serve(opts: { view?: SuggestView; list?: unknown[]; listStatus?: number; accept?: [number, unknown] } = {}): Net {
  const net: Net = { calls: [] };
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    net.calls.push({ method, url, body });
    if (url.endsWith("/suggest") && method === "GET") return opts.listStatus ? json({ detail: "x" }, opts.listStatus) : json({ suggestions: opts.list ?? [] });
    if (url.endsWith("/suggest") && method === "POST") {
      return json({ id: "j1", kind: "suggest", state: "done", stage: "done", progress: 1, error: null,
                    result: { version: 2, bays: 3, mode: body.mode, lined_up: "vehicles", warnings: ["w"], diff: DIFF } });
    }
    if (url.includes("/accept")) return opts.accept ? json(opts.accept[1], opts.accept[0]) : json({ applied: "auto" });
    if (url.includes("/fixed-slots/preview")) return json({ results: [{ id: 1, state: "BUSY" }, { id: 2, state: "FREE" }] });
    if (/\/suggest\/2$/.test(url)) {
      if (method === "PUT") return json({ version: 2, grid: [] });
      if (method === "DELETE") return json({ discarded: [2] });
      return json(opts.view ?? viewOf());
    }
    return json({});
  });
  return net;
}

const show = (props: Partial<Parameters<typeof SuggestSlots>[0]> = {}) => {
  const onNeedAdmin = vi.fn();
  render(<SuggestSlots camId="camera-1" cam="camera-1" snapshot={snapshot} image="data:image/jpeg;base64,xx" stillId={"a".repeat(32)}
                       tokenKey="k" onNeedAdmin={onNeedAdmin} {...props} />);
  return onNeedAdmin;
};
const suggestNow = async () => {
  fireEvent.click(screen.getByText(/Auto-detect \/ suggest slots/));
  await screen.findByTestId("suggest-banner");
};

describe("the card's gate", () => {
  it("does not exist when the geometry feature is off, and asks for the admin token when it is missing", async () => {
    serve({ listStatus: 503 });
    const { container } = render(<SuggestSlots camId="camera-1" cam="camera-1" snapshot={snapshot} image={null} stillId={null} tokenKey="k" onNeedAdmin={vi.fn()} />);
    await waitFor(() => expect(container.innerHTML).toBe(""));
    cleanup();
    serve({ listStatus: 401 });
    const need = show();
    fireEvent.click(await screen.findByText("Enter admin token"));
    expect(need).toHaveBeenCalledWith(expect.stringContaining("admin token"), expect.any(Function));
  });
});

describe("asking for a suggestion", () => {
  it("posts the mode, the snapshot and the expected count, then shows the draft over the running slots", async () => {
    const net = serve();
    show();
    fireEvent.click(await screen.findByText("Full (SAM + GeoCalib)"));
    fireEvent.change(screen.getByTestId("suggest-expected"), { target: { value: "11" } });
    await suggestNow();
    const post = net.calls.find((c) => c.method === "POST")!;
    expect(post.url).toBe("/api/cameras/camera-1/suggest");
    expect(post.body).toEqual({ mode: "full", still: "a".repeat(32), expected: 11 });
    expect(screen.getByTestId("suggest-banner").textContent).toContain("live monitoring is unchanged");
    expect(screen.getByTestId("suggest-diff").textContent).toBe("1 added · 1 moved · 1 removed · 1 unchanged");
    const bays = screen.getAllByTestId("draft-bay");
    expect(bays.map((b) => b.getAttribute("data-status"))).toEqual(["unchanged", "moved", "added"]);
    expect(screen.getAllByTestId("removed-bay")).toHaveLength(1); // the running bay the draft drops
    expect(screen.getByTestId("suggest-pins").textContent).toContain("P1 · slot 7 · next to B3 · holds 4");
    expect(screen.getAllByTestId("pin-bay")).toHaveLength(1);
  });

  it("refuses a bad expected count before asking", async () => {
    const net = serve();
    show();
    fireEvent.change(await screen.findByTestId("suggest-expected"), { target: { value: "0" } });
    fireEvent.click(screen.getByText(/Auto-detect \/ suggest slots/));
    expect(await screen.findByTestId("suggest-message")).toBeTruthy();
    expect(net.calls.some((c) => c.method === "POST")).toBe(false);
  });

  it("lists pending drafts and opens one", async () => {
    serve({ list: [{ version: 2, created_at: 1700000000, note: "", bays: 3, meta: { parent: 1, mode: "quick", kind: "auto", lined_up: "vehicles", expected: null } }] });
    show();
    const list = await screen.findByTestId("suggest-list");
    fireEvent.click(within(list).getByText("Open"));
    await screen.findByTestId("suggest-banner");
  });
});

describe("layers", () => {
  it("shows only the draft, or only the running slots, and remembers the choice", async () => {
    serve();
    show();
    await suggestNow();
    fireEvent.click(screen.getByText("Running"));
    expect(screen.queryAllByTestId("draft-bay")).toHaveLength(0);
    expect(screen.getAllByTestId("baseline-bay").length).toBeGreaterThan(0);
    expect(localStorage.getItem("carparking.suggestLayer")).toBe("baseline");
    fireEvent.click(screen.getByText("Draft"));
    expect(screen.queryAllByTestId("baseline-bay")).toHaveLength(0);
    expect(screen.queryAllByTestId("removed-bay")).toHaveLength(0);
    expect(screen.getAllByTestId("draft-bay")).toHaveLength(3);
  });

  it("opens on the layer chosen last time", async () => {
    localStorage.setItem("carparking.suggestLayer", "both");
    serve();
    show();
    await suggestNow();
    expect(screen.getAllByTestId("baseline-bay").length).toBeGreaterThan(0); // the running slots solid, the draft dotted on top
    expect(screen.getAllByTestId("draft-bay")).toHaveLength(3);
  });
});

describe("fine-tuning", () => {
  it("drags a corner, marks the draft unsaved and saves only ids, names and shapes", async () => {
    const net = serve();
    show();
    await suggestNow();
    const save = screen.getByText("Save draft").closest("button")!;
    expect(save.disabled).toBe(true);
    fireEvent.click(within(screen.getByTestId("suggest-bays")).getAllByRole("button", { name: /B1/ })[0]);
    const corners = screen.getAllByTestId("vertex");
    expect(corners).toHaveLength(4);
    fireEvent.mouseDown(corners[0], { button: 0, clientX: 100, clientY: 250 });
    fireEvent.mouseMove(screen.getByTestId("suggest-svg"), { clientX: 200, clientY: 300 });
    fireEvent.mouseUp(screen.getByTestId("suggest-svg"));
    expect(save.disabled).toBe(false);
    fireEvent.click(save);
    await waitFor(() => expect(net.calls.some((c) => c.method === "PUT")).toBe(true));
    const put = net.calls.find((c) => c.method === "PUT")!;
    expect(put.url).toBe("/api/cameras/camera-1/suggest/2");
    const grid = (put.body as { grid: { id: number; poly: number[][]; source?: string }[] }).grid;
    expect(grid[0].poly[0]).toEqual([0.2, 0.6]); // the dragged corner, as shares of the frame
    expect(grid[1].poly).toEqual(sq(0.3)); // the others as they were
    expect("source" in grid[0]).toBe(false);
  });

  it("moves a whole bay by dragging it, and a removed bay leaves the draft", async () => {
    serve();
    show();
    await suggestNow();
    const first = screen.getAllByTestId("draft-bay")[0].querySelector("polygon")!;
    fireEvent.mouseDown(first, { button: 0, clientX: 150, clientY: 275 });
    fireEvent.mouseMove(screen.getByTestId("suggest-svg"), { clientX: 250, clientY: 275 });
    fireEvent.mouseUp(screen.getByTestId("suggest-svg"));
    expect((screen.getByText("Save draft").closest("button") as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(screen.getByText("Revert edits"));
    expect((screen.getByText("Save draft").closest("button") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(within(screen.getByTestId("suggest-bays")).getAllByTitle("Remove this bay from the draft")[2]);
    expect(screen.getAllByTestId("draft-bay")).toHaveLength(2);
  });
});

describe("test, discard, accept", () => {
  it("tests the draft on the snapshot and colours the bays", async () => {
    const net = serve();
    show();
    await suggestNow();
    fireEvent.click(screen.getByText("Test draft"));
    await waitFor(() => expect(screen.getAllByText(/B1 BUSY/).length).toBeGreaterThan(0));
    const t = net.calls.find((c) => c.url.includes("/fixed-slots/preview"))!;
    expect(t.url).toContain("cam=camera-1");
    expect((t.body as { snapshot: string }).snapshot).toBe("a".repeat(32));
    expect((t.body as { slots: unknown[] }).slots).toHaveLength(3);
  });

  it("discards the draft and says the running slots were not touched", async () => {
    const net = serve();
    show();
    await suggestNow();
    fireEvent.click(screen.getByText("Discard"));
    expect((await screen.findByTestId("suggest-message")).textContent).toContain("not touched");
    expect(net.calls.some((c) => c.method === "DELETE" && c.url.endsWith("/suggest/2"))).toBe(true);
    expect(screen.queryByTestId("suggest-banner")).toBeNull();
  });

  it("asks for a confirmation before accepting an auto grid, then accepts", async () => {
    const net = serve();
    show();
    await suggestNow();
    fireEvent.click(screen.getByText("Accept").closest("button")!);
    const box = screen.getByTestId("suggest-confirm");
    expect(box.textContent).toContain("Switch the live grid");
    expect(net.calls.some((c) => c.url.includes("/accept"))).toBe(false); // nothing yet
    fireEvent.click(within(box).getByText("Confirm accept"));
    await waitFor(() => expect(net.calls.some((c) => c.url.includes("/accept"))).toBe(true));
    expect(net.calls.find((c) => c.url.includes("/accept"))!.body).toEqual({});
    expect((await screen.findByTestId("suggest-message")).textContent).toContain("Accepted");
  });

  it("saves unsaved edits first, so the accepted draft is the one on screen", async () => {
    const net = serve();
    show();
    await suggestNow();
    fireEvent.mouseDown(screen.getAllByTestId("draft-bay")[0].querySelector("polygon")!, { button: 0, clientX: 150, clientY: 275 });
    fireEvent.mouseMove(screen.getByTestId("suggest-svg"), { clientX: 250, clientY: 275 });
    fireEvent.mouseUp(screen.getByTestId("suggest-svg"));
    fireEvent.click(screen.getByText("Accept").closest("button")!);
    fireEvent.click(within(screen.getByTestId("suggest-confirm")).getByText("Confirm accept"));
    await waitFor(() => expect(net.calls.some((c) => c.url.includes("/accept"))).toBe(true));
    const order = net.calls.filter((c) => c.method === "PUT" || c.url.includes("/accept")).map((c) => c.method);
    expect(order).toEqual(["PUT", "POST"]);
  });

  it("on a Fixed camera takes only the ticked bays", async () => {
    const net = serve({ view: viewOf({ meta: { parent: 1, mode: "quick", kind: "fixed", lined_up: "vehicles", expected: null }, pins: [] }) });
    show();
    await suggestNow();
    fireEvent.click(screen.getByLabelText("take B2"));
    fireEvent.click(screen.getByText("Accept").closest("button")!);
    expect(screen.getByTestId("suggest-confirm").textContent).toContain("Merge 2 suggested bay(s)");
    fireEvent.click(within(screen.getByTestId("suggest-confirm")).getByText("Confirm accept"));
    await waitFor(() => expect(net.calls.some((c) => c.url.includes("/accept"))).toBe(true));
    expect(net.calls.find((c) => c.url.includes("/accept"))!.body).toEqual({ select: [1, 3] });
    await screen.findByTestId("suggest-message"); // (the draft is consumed: the card is back to asking)
    expect(screen.queryByTestId("suggest-job")).toBeNull();
  });

  it("cannot accept a stale draft, and reports a refusal from the server", async () => {
    serve({ view: viewOf({ stale: true }) });
    show();
    await suggestNow();
    expect(screen.getByTestId("suggest-stale").textContent).toContain("can no longer be accepted");
    expect((screen.getByText("Accept").closest("button") as HTMLButtonElement).disabled).toBe(true);
    cleanup();
    serve({ accept: [409, { detail: "the drift guard sees the camera has moved" }] });
    show();
    await suggestNow();
    fireEvent.click(screen.getByText("Accept").closest("button")!);
    fireEvent.click(within(screen.getByTestId("suggest-confirm")).getByText("Confirm accept"));
    expect((await screen.findByTestId("suggest-message")).textContent).toContain("camera has moved");
  });
});
