import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { slotColorKey, slotFlags, slotMatches, SLOT_COLORS } from "../lib";
import type { CameraInfo, ModelInfo, Options, SettingsValues, Slot, Snapshot } from "../types";
import { LiveView } from "./LiveView";
import { SettingsDrawer } from "./SettingsDrawer";
import { Counters, StatusCard } from "./SidePanel";

/** Spillage Protection and Non-Vehicle Obstacle Filtering: the Settings drawer switches, the encroached bays on the video and the
 * counter cards, the Worker card line. */
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

// ---- the Settings drawer -------------------------------------------------------------------------------------------------------------
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
      return json({ version: 3, values }); // a worker option: the settings version does not move
    }
    return json({ version: 3, values });
  });
}
const drawer = () => render(<SettingsDrawer open onClose={() => {}} onNeedAdmin={() => {}} tokenKey="" codeStale={false}
                                            model={undefined as unknown as ModelInfo} cam={null} />);
const sw = (id: string) => screen.getByTestId(id).querySelector("[role=switch]") as HTMLButtonElement;

describe("Parking protection switches in the Settings drawer", () => {
  it("shows both off by default, with their labels and explanations", async () => {
    serve([]);
    drawer();
    await screen.findByTestId("spillage-switch");
    expect(screen.getByText("Spillage Protection")).toBeTruthy();
    expect(screen.getByText("Non-Vehicle Obstacle Filtering")).toBeTruthy();
    expect(sw("spillage-switch").getAttribute("aria-checked")).toBe("false");
    expect(sw("obstacle-switch").getAttribute("aria-checked")).toBe("false");
    expect(screen.getByText(/ENCROACHED and is taken out of the Free count/)).toBeTruthy();
    expect(screen.getByText(/a bay that is already BUSY is never affected/)).toBeTruthy();
  });

  it("each switch sends only its own field, with the version, on Apply", async () => {
    const puts: Record<string, unknown>[] = [];
    serve(puts);
    drawer();
    await screen.findByTestId("spillage-switch");
    fireEvent.click(sw("spillage-switch"));
    expect(sw("spillage-switch").getAttribute("aria-checked")).toBe("true");
    fireEvent.click(screen.getByText("Apply"));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ spillage: true, version: 3 });
    await waitFor(() => expect(screen.getByText(/Applied \(version 3\)/)).toBeTruthy());
    fireEvent.click(sw("obstacle-switch"));
    fireEvent.click(screen.getByText("Apply"));
    await waitFor(() => expect(puts).toHaveLength(2));
    expect(puts[1]).toEqual({ obstacle_filter: true, version: 3 });
    expect(sw("spillage-switch").getAttribute("aria-checked")).toBe("true"); // still on, and not re-sent
  });

  it("the camera drift guard is a fourth switch, off by default, sent alone on Apply", async () => {
    const puts: Record<string, unknown>[] = [];
    serve(puts);
    drawer();
    await screen.findByTestId("drift-switch");
    expect(screen.getByText("Camera Drift Guard")).toBeTruthy();
    expect(sw("drift-switch").getAttribute("aria-checked")).toBe("false");
    expect(screen.getByText(/nothing is ever marked stale/)).toBeTruthy();
    fireEvent.click(sw("drift-switch"));
    fireEvent.click(screen.getByText("Apply"));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ drift_guard: true, version: 3 });
    await waitFor(() => expect(sw("drift-switch").getAttribute("aria-checked")).toBe("true"));
  });

  it("shows what the worker runs with (stored on) and switching one off sends false", async () => {
    const puts: Record<string, unknown>[] = [];
    serve(puts, { spillage: true, obstacle_filter: true });
    drawer();
    await screen.findByTestId("spillage-switch");
    expect(sw("spillage-switch").getAttribute("aria-checked")).toBe("true");
    expect(sw("obstacle-switch").getAttribute("aria-checked")).toBe("true");
    fireEvent.click(sw("obstacle-switch"));
    fireEvent.click(screen.getByText("Apply"));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ obstacle_filter: false, version: 3 });
  });
});

// ---- slots on the video ---------------------------------------------------------------------------------------------------------------
const box = (x: number): [number, number][] => [[x, 0.2], [x + 0.1, 0.2], [x + 0.1, 0.4], [x, 0.4]];
const slot = (id: number, state: Slot["state"], x: number, extra: Partial<Slot> = {}): Slot =>
  ({ id, name: `A${id}`, state, hidden: false, phantom: false, duplicate_of: null, poly: box(x), ...extra });
const encroached = slot(2, "FREE", 0.3, { avail: "ENCROACHED", display: "ENCROACHED", encroached_by: ["A1"], cover: 0.42 });
const snap = (extra: Partial<Snapshot> = {}): Snapshot => ({
  shape: [2560, 1440], frame_ts: 1,
  counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 3, learning: 0, encroached: 1, available: 1 },
  slots: [slot(1, "BUSY", 0.1, { avail: null, display: "BUSY", encroached_by: [], cover: null }), encroached, slot(3, "FREE", 0.5)],
  detections: [], area: { top: 0, bottom: 0, left: 0, right: 0, roi: null }, ...extra,
}) as unknown as Snapshot;
const camera = { id: "front", display: "Front", streams: { hd: "front_hd", sd: "front_sd", native: "front_native" } } as CameraInfo;
const view = (snapshot: Snapshot, slotFilter: Parameters<typeof LiveView>[0]["slotFilter"] = null) =>
  render(<LiveView snapshot={snapshot} frameUrl={null} onNeedFrames={() => {}} camera={camera} tokenKey="" slotFilter={slotFilter} />);
const drawn = () => screen.getAllByTestId("overlay-slot");

describe("encroached bays on the video", () => {
  it("are hatched amber and labelled ENCROACHED, the others as before", () => {
    view(snap());
    const [a1, a2, a3] = drawn();
    expect(a2.getAttribute("data-avail")).toBe("ENCROACHED");
    expect(a2.getAttribute("data-state")).toBe("FREE"); // the state is unchanged
    expect(a2.querySelector("polygon")!.getAttribute("fill")).toBe("url(#encroach-hatch)");
    expect(a2.querySelector("polygon")!.getAttribute("stroke")).toBe(SLOT_COLORS.ENCROACHED.stroke);
    expect(a2.textContent).toContain("ENCROACHED");
    expect(a1.getAttribute("data-avail")).toBeNull();
    expect(a1.querySelector("polygon")!.getAttribute("fill")).toBe(SLOT_COLORS.BUSY.fill);
    expect(a3.querySelector("polygon")!.getAttribute("fill")).toBe(SLOT_COLORS.FREE.fill);
    expect(screen.getByTestId("overlay").querySelector("pattern#encroach-hatch")).toBeTruthy();
  });

  it("the hover says who is on the bay and how much", () => {
    view(snap());
    fireEvent.mouseMove(drawn()[1], { clientX: 10, clientY: 10 });
    expect(screen.getByText(/encroached by A1 \(42 % of the bay\): free, but not usable/)).toBeTruthy();
    expect(screen.getByText("FREE, not usable")).toBeTruthy();
  });

  it("the FREE filter shows only usable bays and the ENCROACHED filter only the encroached", () => {
    const free = view(snap(), "FREE");
    expect(drawn().map((s) => s.getAttribute("data-faded"))).toEqual(["1", "1", "0"]);
    expect(screen.getByTestId("filter-chip").textContent).toContain("1 of 3");
    free.unmount();
    view(snap(), "ENCROACHED");
    expect(drawn().map((s) => s.getAttribute("data-faded"))).toEqual(["1", "0", "1"]);
    expect(screen.getByTestId("filter-chip").textContent).toContain("Showing ENCROACHED only · 1 of 3");
  });

  it("without spillage protection nothing changes: no ENCROACHED anywhere, FREE keeps its colour", () => {
    view(snap({ slots: [slot(1, "BUSY", 0.1), slot(2, "FREE", 0.3)], counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 2, learning: 0 } }));
    expect(drawn().every((s) => s.getAttribute("data-avail") == null && !s.textContent!.includes("ENCROACHED"))).toBe(true);
    expect(drawn()[1].querySelector("polygon")!.getAttribute("fill")).toBe(SLOT_COLORS.FREE.fill);
  });
});

describe("helpers", () => {
  it("slotColorKey / slotMatches / slotFlags", () => {
    expect(slotColorKey(encroached)).toBe("ENCROACHED");
    expect(slotColorKey(slot(3, "FREE", 0.5))).toBe("FREE");
    expect(slotMatches(encroached, "FREE")).toBe(false);
    expect(slotMatches(encroached, "ENCROACHED")).toBe(true);
    expect(slotMatches(slot(3, "FREE", 0.5), "FREE")).toBe(true);
    expect(slotMatches(slot(3, "FREE", 0.5), "ENCROACHED")).toBe(false);
    expect(slotMatches(encroached, null)).toBe(true);
    expect(slotFlags(encroached)).toEqual(["encroached by A1 (42 % of the bay): free, but not usable"]);
    expect(slotFlags({ ...encroached, encroached_by: ["unplaced vehicle"], cover: null })[0]).toBe("encroached by unplaced vehicle: free, but not usable");
  });
});

// ---- counter cards and the Worker card ---------------------------------------------------------------------------------------------------
describe("counts", () => {
  it("the Free card holds the usable bays and an Encroached card appears beside it only with spillage protection", () => {
    render(<Counters snapshot={snap()} />);
    expect(screen.getByText("Free").previousSibling!.textContent).toBe("1");
    const card = screen.getByTestId("encroached-card");
    expect(card.textContent).toContain("Encroached (free, not usable)");
    expect(card.firstElementChild!.textContent).toContain("1");
    expect(card.querySelector("[title]")!.getAttribute("title")).toContain("(1)"); // the usable count in the hint
  });

  it("the Encroached card is a filter button when the cards filter", () => {
    const picks: unknown[] = [];
    render(<Counters snapshot={snap()} filter={null} onFilter={(f) => picks.push(f)} />);
    fireEvent.click(screen.getByTestId("filter-encroached"));
    fireEvent.click(screen.getByTestId("filter-free"));
    expect(picks).toEqual(["ENCROACHED", "FREE"]);
  });

  it("without the option the four cards are exactly as before", () => {
    render(<Counters snapshot={{ counts: { free: 2, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 3, learning: 0 } }} />);
    expect(screen.queryByTestId("encroached-card")).toBeNull();
    expect(screen.queryByText(/Encroached/)).toBeNull();
  });

  it("the Worker card lists what the obstacle filter dropped, only while it is on", () => {
    const { rerender } = render(<StatusCard snapshot={snap({ obstacle: { rejected: 3, reasons: { too_small: 2, pylon: 1 } } })} heartbeat={1} />);
    expect(screen.getByText("Obstacle filter")).toBeTruthy();
    expect(screen.getByText(/3 dropped \(too small 2, pylon 1\) in the last frame/)).toBeTruthy();
    rerender(<StatusCard snapshot={snap({ obstacle: { rejected: 0, reasons: {} } })} heartbeat={1} />);
    expect(screen.getByText("0 dropped in the last frame")).toBeTruthy();
    rerender(<StatusCard snapshot={snap()} heartbeat={1} />);
    expect(screen.queryByText("Obstacle filter")).toBeNull();
  });
});

describe("ground-metric encroachment", () => {
  it("adds the metres left only when the server measured them", () => {
    const base = { avail: "ENCROACHED", encroached_by: ["A1"], cover: 0.42 } as Slot;
    expect(slotFlags(base).join(" ")).toContain("(42 % of the bay): free, but not usable");
    expect(slotFlags({ ...base, free_m: 1.4 } as Slot).join(" ")).toContain("(42 % of the bay, 1.4 m of ground left)");
  });
});
