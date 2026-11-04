import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CameraInfo, Snapshot } from "../types";
import { DYNAMIC_LOCKED, SlotEditor } from "./SlotEditor";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const BAY = { id: 1, name: "A1", poly: [[0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.1, 0.5]] };
const camera: CameraInfo = {
  id: "front", name: "Front", label: "", display: "Front", primary: true, sort_order: 0, metadata: {}, rtsp_url: "", rtsp_sub_url: "",
  streams: { hd: "front_hd", sd: "front_sd", native: "front_native" },
  status: { connected: true, heartbeat_age: 1, enabled: true, running: true, slot_mode: "dynamic", counts: null, state: null },
};
const snapshot = {
  shape: [2560, 1440], mode: "dynamic",
  counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 2, learning: 0 },
  slots: [
    { id: 1, name: "S1", state: "FREE", hidden: false, phantom: false, duplicate_of: null, poly: [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]] },
    { id: 2, name: "S2", state: "BUSY", hidden: false, phantom: false, duplicate_of: null, poly: [[0.4, 0.1], [0.5, 0.1], [0.5, 0.2], [0.4, 0.2]] },
  ],
} as unknown as Snapshot;

let stills = 0; // snapshots taken (POST /api/fixed-slots/snapshot)
let gone = false; // the next preview of a snapshot answers 410 (the server restarted)

function serve(mode: "dynamic" | "fixed", puts: string[]) {
  stills = 0;
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    if (url.startsWith("/api/fixed-slots/snapshot")) {
      stills += 1;
      return json({ id: `${stills}`.padStart(32, "0"), ts: 1792650000 + stills, source: "camera", width: 2560, height: 1440,
                    image: `data:image/jpeg;base64,STILL${stills}` });
    }
    if (url.startsWith("/api/fixed-slots/preview") && gone && JSON.parse(String(init?.body)).snapshot === "1".padStart(32, "0")) {
      gone = false;
      puts.push(`preview ${init?.body}`);
      return json({ detail: "this snapshot is no longer kept (the server restarted): Refresh snapshot" }, 410);
    }
    if (url.startsWith("/api/fixed-slots/preview")) {
      puts.push(`preview ${init?.body}`);
      const body = JSON.parse(String(init?.body));
      const all = body.slots.map((x: { id: number; name: string }) => ({ id: x.id, name: x.name, state: x.id === 1 ? "BUSY" : "FREE", score: x.id === 1 ? 1.5 : 0,
                                                                         vehicle: x.id === 1 ? { label: "car", conf: 0.9, box: [0.1, 0.2, 0.3, 0.5] } : null }));
      return json({ results: body.only ? all.filter((r: { id: number }) => body.only.includes(r.id)) : all, rule: "ground", match_iou: 0.15,
                    frame_ts: 1, frame_age: 0.1, vehicles: [[0.1, 0.2, 0.3, 0.5]], source: "on-demand", image: "data:image/jpeg;base64,AAAA" });
    }
    if (url.startsWith("/api/fixed-slots")) {
      if (init?.method === "PUT") {
        puts.push(url);
        return mode === "fixed" ? json({ slots: [BAY], settings_version: 2 }) : json({ detail: DYNAMIC_LOCKED }, 409);
      }
      return json({ camera: "Front", cam: "front", updated: null, slots: mode === "fixed" ? [BAY] : [], cameras: ["Front"],
                    in_use: mode === "fixed", mode, editable: mode === "fixed", locked: mode === "fixed" ? null : DYNAMIC_LOCKED });
    }
    if (url.startsWith("/api/settings")) return json({ version: 1, values: { camera: "Front", slot_mode: mode, enabled: true } });
    return json({});
  });
}

const editor = () => render(<SlotEditor snapshot={snapshot} tokenKey="" onNeedAdmin={() => {}} live={null} cam="front"
                                        cameras={[camera]} onCam={() => {}} />);

describe("slot editor: Fixed mode draws, Dynamic mode is read-only", () => {
  it("Dynamic: no drawing tools, the message, the AI's slots; clicks draw nothing", async () => {
    const puts: string[] = [];
    serve("dynamic", puts);
    editor();
    await waitFor(() => expect(screen.getByTestId("dynamic-lock").textContent).toContain(DYNAMIC_LOCKED));
    await waitFor(() => expect(screen.getByTestId("mode-dynamic")).toBeTruthy()); // the mode card's requests done
    expect(DYNAMIC_LOCKED).toBe("Dynamic Mode automatically detects parking spaces. Switch to Fixed Mode to manually draw or edit slots.");
    expect(screen.queryByText("Polygon")).toBeNull();
    expect(screen.queryByText("Rectangle")).toBeNull();
    expect(screen.queryByText("Save to server")).toBeNull();
    expect(screen.queryByTitle("Import JSON")).toBeNull();
    expect(screen.getAllByTestId("detected-slot")).toHaveLength(2);
    expect(screen.getByTestId("detected-list").textContent).toContain("BUSY");
    const svg = screen.getByTestId("draw-svg");
    for (const [x, y] of [[100, 100], [200, 100], [200, 200], [100, 100]]) fireEvent.mouseDown(svg, { button: 0, clientX: x, clientY: y });
    fireEvent.keyDown(window, { key: "Enter" });
    expect(screen.queryByTestId("slot-list")).toBeNull();
    expect(puts).toEqual([]);
  });

  it("Fixed: the drawing tools and Save are there", async () => {
    const puts: string[] = [];
    serve("fixed", puts);
    editor();
    await waitFor(() => expect(screen.getByText("Polygon")).toBeTruthy());
    await waitFor(() => expect(screen.getByTestId("mode-fixed")).toBeTruthy());
    expect(screen.queryByTestId("dynamic-lock")).toBeNull();
    expect(screen.getByText("Rectangle")).toBeTruthy();
    expect(screen.getByText("Save to server")).toBeTruthy();
    expect(screen.getByTestId("slot-list").textContent).toContain("1");
    expect(screen.queryAllByTestId("detected-slot")).toHaveLength(0);
  });
});

describe("Edit slots panel (Fixed mode)", () => {
  it("moves, turns, splits and undoes on the drawn slots; the edited map is what Save sends and what Test slots checks; Dynamic has no panel", async () => {
    const puts: string[] = [];
    serve("fixed", puts);
    editor();
    await waitFor(() => expect(screen.getByTestId("slot-fine")).toBeTruthy());
    await waitFor(() => expect(screen.getByTestId("mode-fixed")).toBeTruthy()); // (the mode card's answer resets the selection once)
    const save = () => screen.getByText("Save to server").closest("button") as HTMLButtonElement;
    expect(save().disabled).toBe(true); // nothing edited
    fireEvent.click(screen.getByTestId("slot-list").querySelector("li")!); // select A1
    fireEvent.change(screen.getByTestId("slot-fine-step"), { target: { value: "10" } });
    fireEvent.click(screen.getByTestId("slot-fine-down"));
    expect(save().disabled).toBe(false); // an edit
    fireEvent.click(screen.getByTestId("slot-undo-edit"));
    expect(save().disabled).toBe(true); // undone: the saved map again
    fireEvent.click(screen.getByTestId("slot-redo-edit"));
    fireEvent.click(screen.getByTestId("slot-turn-cw")); // the whole map as one block
    fireEvent.click(screen.getByTestId("slot-split-width"));
    expect(screen.getByTestId("slot-list").querySelectorAll("li")).toHaveLength(2);
    // the edited, unsaved slots are what the real-time test is asked about
    fireEvent.click(screen.getByText("Test all slots"));
    await waitFor(() => expect(puts.some((u) => u.startsWith("preview "))).toBe(true));
    const sent = JSON.parse(puts.find((u) => u.startsWith("preview "))!.slice(8));
    expect(sent.slots).toHaveLength(2);
    expect(sent.slots[0].poly).not.toEqual(BAY.poly);
    fireEvent.click(screen.getAllByText("Back to edit")[0]); // and Save sends them to the server
    fireEvent.click(save());
    await waitFor(() => expect(puts.some((u) => u.startsWith("/api/fixed-slots"))).toBe(true));
  });

  it("Dynamic mode: no adjustment panel", async () => {
    serve("dynamic", []);
    editor();
    await waitFor(() => expect(screen.getByTestId("dynamic-lock")).toBeTruthy());
    expect(screen.queryByTestId("slot-fine")).toBeNull();
  });
});

describe("Test slots preview (Fixed mode, read-only)", () => {
  it("previews FREE / BUSY, pauses editing, and Back to edit returns with nothing lost", async () => {
    const puts: string[] = [];
    serve("fixed", puts);
    editor();
    await waitFor(() => expect(screen.getByText("Test all slots")).toBeTruthy());
    await waitFor(() => expect(screen.getByTestId("mode-fixed")).toBeTruthy());
    // an unsaved edit first: draw a second slot (rectangle)
    fireEvent.click(screen.getByText("Rectangle"));
    const svg = screen.getByTestId("draw-svg");
    screen.getByTestId("draw-surface").getBoundingClientRect = () => ({ left: 0, top: 0, width: 1000, height: 562, right: 1000, bottom: 562, x: 0, y: 0, toJSON: () => ({}) });
    fireEvent.mouseDown(svg, { button: 0, clientX: 600, clientY: 300 });
    fireEvent.mouseUp(svg, { clientX: 800, clientY: 450 });
    await waitFor(() => expect(screen.getByTestId("slot-list").textContent).toContain("2"));
    const before = screen.getByTestId("slot-list").textContent;
    expect(screen.getByText("Test selected slot").closest("button")?.disabled).toBe(false); // the new slot is selected

    fireEvent.click(screen.getByText("Test all slots"));
    await waitFor(() => expect(screen.getByTestId("preview-banner").textContent).toContain("Preview mode – active"));
    expect(screen.getByTestId("preview-banner").textContent).toContain("1 FREE · 1 BUSY");
    expect(screen.getAllByTestId("preview-slot").map((g) => g.getAttribute("data-state"))).toEqual(["BUSY", "FREE"]);
    expect(screen.getAllByTestId("preview-vehicle")).toHaveLength(1);
    expect(screen.getByTestId("preview-banner").textContent).toContain("frame grabbed now"); // monitoring not needed
    expect(screen.getByTestId("preview-image").getAttribute("src")).toBe("data:image/jpeg;base64,AAAA"); // the tested frame
    expect(puts.some((p) => p.startsWith("preview") && p.includes('"id":2'))).toBe(true); // the unsaved slot was tested
    expect(puts.filter((p) => !p.startsWith("preview"))).toEqual([]); // nothing saved
    // editing paused: a click on the image draws nothing
    fireEvent.mouseDown(svg, { button: 0, clientX: 100, clientY: 100 });
    fireEvent.mouseUp(svg, { clientX: 300, clientY: 300 });
    expect(screen.getByTestId("slot-list").textContent).toBe(before);

    fireEvent.click(screen.getByTestId("preview-back"));
    expect(screen.queryByTestId("preview-banner")).toBeNull();
    expect(screen.queryAllByTestId("preview-slot")).toHaveLength(0);
    expect(screen.queryByTestId("preview-image")).toBeNull(); // back on the editor's own background
    expect(screen.getByTestId("slot-list").textContent).toBe(before); // the unsaved slot is still there
    expect(screen.getByText("Save to server").closest("button")?.disabled).toBe(false); // still unsaved

    fireEvent.click(screen.getByText("Test selected slot"));
    await waitFor(() => expect(screen.getByTestId("preview-banner").textContent).toContain("1 FREE · 0 BUSY"));
    expect(screen.getAllByTestId("preview-slot").map((g) => g.getAttribute("data-state"))).toEqual(["untested", "FREE"]);
    fireEvent.keyDown(window, { key: "Escape" }); // Esc leaves the preview too
    await waitFor(() => expect(screen.queryByTestId("preview-banner")).toBeNull());
  });

  it("snapshot: captured on open as the frozen canvas, Test runs on it, Refresh snapshot takes a new one", async () => {
    const puts: string[] = [];
    serve("fixed", puts);
    editor();
    await waitFor(() => expect((screen.getByTestId("still-image") as HTMLImageElement).src).toContain("STILL1"));
    expect(screen.getByTestId("still-info").textContent).toContain("camera snapshot");
    await waitFor(() => expect((screen.getByText("Test all slots").closest("button") as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByText("Test all slots"));
    await waitFor(() => expect(screen.getByTestId("preview-banner")).toBeTruthy());
    expect(JSON.parse(puts.find((x) => x.startsWith("preview "))!.slice(8)).snapshot).toBe("1".padStart(32, "0"));
    fireEvent.click(screen.getByTestId("preview-back"));
    fireEvent.click(screen.getByText("Refresh snapshot"));
    await waitFor(() => expect((screen.getByTestId("still-image") as HTMLImageElement).src).toContain("STILL2"));
    expect(stills).toBe(2);
  });

  it("snapshot gone on the server (410): a new one is taken and the test runs on it, no error", async () => {
    const puts: string[] = [];
    serve("fixed", puts);
    gone = true;
    editor();
    await waitFor(() => expect((screen.getByTestId("still-image") as HTMLImageElement).src).toContain("STILL1"));
    await waitFor(() => expect((screen.getByText("Test all slots").closest("button") as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByText("Test all slots"));
    await waitFor(() => expect(screen.getByTestId("preview-banner")).toBeTruthy());
    const asked = puts.filter((x) => x.startsWith("preview ")).map((x) => JSON.parse(x.slice(8)).snapshot);
    expect(asked).toEqual(["1".padStart(32, "0"), "2".padStart(32, "0")]);
    expect(stills).toBe(2);
    expect(screen.queryByText(/Test failed/)).toBeNull();
  });
});

describe("slot editor: the auto grid's pins", () => {
  it("Dynamic mode with an auto grid offers the pin editor; without one it does not", async () => {
    serve("dynamic", []);
    const grid = { ...snapshot, auto_slots: { version: 1, count: 2 } } as unknown as Snapshot;
    const { unmount } = render(<SlotEditor snapshot={grid} tokenKey="" onNeedAdmin={() => {}} live={null} cam="front" cameras={[camera]} onCam={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("dynamic-lock")).toBeTruthy());
    expect(screen.getByTestId("auto-grid-pins")).toBeTruthy();
    unmount();
    editor();
    await waitFor(() => expect(screen.getByTestId("dynamic-lock")).toBeTruthy());
    expect(screen.queryByTestId("auto-grid-pins")).toBeNull();
  });
});
