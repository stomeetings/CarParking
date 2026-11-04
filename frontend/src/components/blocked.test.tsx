import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { BLOCK_HATCH, SLOT_COLORS, eventInfo, slotColorKey, slotFlags, slotMatches } from "../lib";
import type { CameraInfo, ModelInfo, Options, SettingsValues, Slot, Snapshot } from "../types";
import { LiveView } from "./LiveView";
import { SettingsDrawer } from "./SettingsDrawer";
import { Counters, EventTicker } from "./SidePanel";

/** Blocked by Object Detection: the Settings switch (off by default), BLOCKED bays on the video (cross-hatched grey, named), the counter card and the
 * filter, the event notices; a bay with an object in it is FREE by state but never counted free. */
beforeAll(() => {
  class Silent {
    onopen = null;
    onmessage = null;
    onclose = null;
    close() {}
    addTransceiver() {}
    createOffer() {
      return new Promise(() => {});
    }
  }
  vi.stubGlobal("RTCPeerConnection", Silent);
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
});
afterAll(() => vi.unstubAllGlobals());
afterEach(() => {
  cleanup();
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

const base: SettingsValues = {
  enabled: true, camera: "Front", stream: "main", custom_url: "", detector: "yolov8s", bay_model: "auto", inference: "1280", conf: 0.3,
  classes: ["car"], extra: [], slot_mode: "fixed", match_rule: "ground", match_iou: 0.15, save_every: 60, hold: 20, exit_hold: 20,
  exit_frames: 0, occlusion: 0.8, hidden_max: 600, learn_hold: 20, learn: true, absent_free: true, cut_top: 0, cut_bottom: 0,
  cut_left: 0, cut_right: 0, roi: "", model_variant: "auto", hidden_ground: 0.8, ground_share: 0.4, vehicle_iou: 0.2, bay_share: 0.5,
  pin_reach: 0.25, spillage: false, obstacle_filter: false,
};
const options: Options = {
  stream: ["main"], detector: ["yolov8s"], bay_model: ["auto"], inference: ["1280"], slot_mode: ["dynamic", "fixed"], match_rule: ["ground"],
  classes: ["car"], extra: [], detector_selected: "yolov8s", ranges: {},
};

function serve(puts: Record<string, unknown>[], stored: Partial<SettingsValues> = {}) {
  let values = { ...base, ...stored };
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    if (String(url).includes("/api/settings/options")) return json(options);
    if (String(url).includes("/api/overflow")) return json({ enabled: false, auto_dynamic: false, slot_split: false });
    if (init?.method === "PUT") {
      const body = JSON.parse(String(init.body));
      puts.push(body);
      const { version: _v, ...changed } = body;
      values = { ...values, ...changed };
      return json({ version: 3, values });
    }
    return json({ version: 3, values });
  });
}
const drawer = () => render(<SettingsDrawer open onClose={() => {}} onNeedAdmin={() => {}} tokenKey="" codeStale={false}
                                            model={undefined as unknown as ModelInfo} cam={null} />);
const sw = (id: string) => screen.getByTestId(id).querySelector("[role=switch]") as HTMLButtonElement;

describe("the Settings switch", () => {
  it("is in Advanced Smart Protections, off by default, explained, and sends only its own field", async () => {
    const puts: Record<string, unknown>[] = [];
    serve(puts);
    drawer();
    await screen.findByTestId("blocked-switch");
    expect(screen.getByTestId("group-toggle-protect").textContent).toContain("Advanced Smart Protections");
    expect(screen.getByText("Blocked by Object Detection")).toBeTruthy();
    expect(sw("blocked-switch").getAttribute("aria-checked")).toBe("false");
    expect(screen.getByText(/stays FREE for the history and the sessions: no entry or exit is/)).toBeTruthy();
    fireEvent.click(sw("blocked-switch"));
    expect(screen.getByTestId("group-pending-protect").textContent).toBe("1 changed");
    fireEvent.click(screen.getByText("Apply"));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ blocked_objects: true, version: 3 });
  });

  it("shows what the worker runs with when it is on, and switching it off sends false", async () => {
    const puts: Record<string, unknown>[] = [];
    serve(puts, { blocked_objects: true });
    drawer();
    await screen.findByTestId("blocked-switch");
    expect(sw("blocked-switch").getAttribute("aria-checked")).toBe("true");
    fireEvent.click(sw("blocked-switch"));
    fireEvent.click(screen.getByText("Apply"));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ blocked_objects: false, version: 3 });
  });
});

// ---- slots on the video --------------------------------------------------------------------------------------------------------------
const box = (x: number): [number, number][] => [[x, 0.2], [x + 0.1, 0.2], [x + 0.1, 0.4], [x, 0.4]];
const slot = (id: number, state: Slot["state"], x: number, extra: Partial<Slot> = {}): Slot =>
  ({ id, name: `A${id}`, state, hidden: false, phantom: false, duplicate_of: null, poly: box(x), ...extra });
const blocked = slot(2, "FREE", 0.3, { avail: "BLOCKED", display: "BLOCKED", blocked_by: ["potted plant"], encroached_by: [], cover: null });
const snap = (extra: Partial<Snapshot> = {}): Snapshot => ({
  shape: [2560, 1440], frame_ts: 1,
  counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 3, learning: 0, encroached: 0, available: 1, blocked_objects: 1 },
  slots: [slot(1, "BUSY", 0.1, { avail: null, display: "BUSY", blocked_by: [] }), blocked, slot(3, "FREE", 0.5, { avail: null, display: "FREE", blocked_by: [] })],
  detections: [], area: { top: 0, bottom: 0, left: 0, right: 0, roi: null }, ...extra,
}) as unknown as Snapshot;
const camera = { id: "front", display: "Front", streams: { hd: "front_hd", sd: "front_sd", native: "front_native" } } as CameraInfo;
const view = (snapshot: Snapshot, slotFilter: Parameters<typeof LiveView>[0]["slotFilter"] = null) =>
  render(<LiveView snapshot={snapshot} frameUrl={null} onNeedFrames={() => {}} camera={camera} tokenKey="" slotFilter={slotFilter} />);
const drawn = () => screen.getAllByTestId("overlay-slot");

describe("BLOCKED bays on the video", () => {
  it("are cross-hatched slate and named BLOCKED, the others as they were", () => {
    view(snap());
    const [busy, b, free] = drawn();
    expect(b.getAttribute("data-avail")).toBe("BLOCKED");
    expect(b.getAttribute("data-state")).toBe("FREE"); // by state it is still FREE
    expect(b.textContent).toContain("BLOCKED");
    expect(b.querySelector("polygon")!.getAttribute("fill")).toBe(`url(#${BLOCK_HATCH})`);
    expect(b.querySelector("polygon")!.getAttribute("stroke")).toBe(SLOT_COLORS.BLOCKED.stroke);
    expect(screen.getByTestId("overlay").querySelector(`pattern#${BLOCK_HATCH}`)).toBeTruthy();
    expect(busy.textContent).toContain("BUSY");
    expect(free.querySelector("polygon")!.getAttribute("fill")).toBe(SLOT_COLORS.FREE.fill);
  });

  it("say what blocks them on hover", () => {
    view(snap());
    fireEvent.mouseMove(drawn()[1], { clientX: 10, clientY: 10 });
    expect(screen.getByText("FREE, not usable")).toBeTruthy();
    expect(screen.getByText("blocked by potted plant: free, but not usable")).toBeTruthy();
  });

  it("the BLOCKED filter shows only them, the FREE filter only the usable ones", () => {
    view(snap(), "BLOCKED");
    expect(drawn().map((s) => s.getAttribute("data-faded"))).toEqual(["1", "0", "1"]);
    expect(screen.getByTestId("filter-chip").textContent).toContain("Showing BLOCKED only · 1 of 3");
    cleanup();
    view(snap(), "FREE");
    expect(drawn().map((s) => s.getAttribute("data-faded"))).toEqual(["1", "1", "0"]);
  });

  it("a snapshot without the option draws and words everything as before", () => {
    view(snap({ slots: [slot(1, "BUSY", 0.1), slot(2, "FREE", 0.3), slot(3, "FREE", 0.5)] }));
    expect(drawn().every((s) => s.getAttribute("data-avail") == null && !s.textContent!.includes("BLOCKED"))).toBe(true);
    expect(screen.getByTestId("overlay").querySelector(`pattern#${BLOCK_HATCH}`)).toBeTruthy(); // (the pattern is defined, never used)
    expect(drawn()[1].querySelector("polygon")!.getAttribute("fill")).toBe(SLOT_COLORS.FREE.fill);
  });
});

describe("helpers", () => {
  it("slotColorKey / slotMatches / slotFlags", () => {
    expect(slotColorKey(blocked)).toBe("BLOCKED");
    expect(slotColorKey({ ...blocked, avail: "ENCROACHED" })).toBe("ENCROACHED");
    expect(slotMatches(blocked, "FREE")).toBe(false); // not counted free
    expect(slotMatches(blocked, "BLOCKED")).toBe(true);
    expect(slotMatches(blocked, "ENCROACHED")).toBe(false);
    expect(slotMatches(slot(3, "FREE", 0.5), "BLOCKED")).toBe(false);
    expect(slotMatches(slot(3, "FREE", 0.5), "FREE")).toBe(true);
    expect(slotMatches(blocked, null)).toBe(true);
    expect(slotFlags(blocked)).toEqual(["blocked by potted plant: free, but not usable"]);
    expect(slotFlags({ ...blocked, blocked_by: [] })[0]).toBe("blocked by an object: free, but not usable");
  });
});

describe("counts", () => {
  it("a Blocked card appears with the option, the Free card holds the usable bays, and it filters", () => {
    const picks: unknown[] = [];
    render(<Counters snapshot={snap()} filter={null} onFilter={(f) => picks.push(f)} />);
    expect(screen.getByText("Free").previousSibling!.textContent).toBe("1");
    const card = screen.getByTestId("blocked-card");
    expect(card.textContent).toContain("Blocked by object (free, not usable)");
    expect(card.firstElementChild!.textContent).toContain("1");
    fireEvent.click(screen.getByTestId("filter-blocked"));
    fireEvent.click(screen.getByTestId("filter-free"));
    expect(picks).toEqual(["BLOCKED", "FREE"]);
  });

  it("without the option the cards are exactly as before", () => {
    render(<Counters snapshot={{ counts: { free: 2, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 3, learning: 0 } }} />);
    expect(screen.queryByTestId("blocked-card")).toBeNull();
    expect(screen.queryByText(/Blocked by object/)).toBeNull();
  });
});

describe("the event log", () => {
  it("files the notices apart from entries and exits", () => {
    const ev = (id: number, event: "object_blocked" | "object_cleared", note: string) =>
      ({ id, slot_id: 2, event, time: "2026-11-02T12:00:00+05:00", ts: 100 + id, session_id: null, note });
    expect(eventInfo(ev(1, "object_blocked", "")).group).toBe("notice");
    expect(eventInfo(ev(1, "object_blocked", "")).label).toBe("🪴 BLOCKED");
    expect(eventInfo(ev(2, "object_cleared", "")).text).toBe("object gone from the bay");
    render(<EventTicker events={[ev(2, "object_cleared", "object gone"), ev(1, "object_blocked", "by potted plant: free but not usable")]} />);
    const text = screen.getAllByRole("listitem").map((li) => li.textContent ?? "");
    expect(text[0]).toContain("Slot 2 · object gone from the bay");
    expect(text[1]).toContain("Slot 2 · bay blocked by an object");
    expect(text[1]).toContain("by potted plant");
    expect(screen.queryAllByText(/vehicle out/)).toHaveLength(0);
  });
});
