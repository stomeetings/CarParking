import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { LAYERS_KEY } from "./components/LiveView";
import { METRICS_KEY } from "./components/Metrics";

// The whole dashboard against a fake backend: the viewer's choices (layers, Metrics, the counter-card filter) must survive
// every state message the live socket delivers, however often it comes and whatever it carries.

class FakeSocket {
  static all: FakeSocket[] = [];
  binaryType = "";
  onopen: (() => void) | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  constructor(public url: string) {
    FakeSocket.all.push(this);
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {}
}
const socket = () => FakeSocket.all[FakeSocket.all.length - 1];

const box = (x: number): [number, number][] => [[x, 0.2], [x + 0.1, 0.2], [x + 0.1, 0.4], [x, 0.4]];
const slot = (id: number, state: string, x: number) => ({ id, name: String(id), state, hidden: false, phantom: false, duplicate_of: null, poly: box(x) });
const state = (n: number, over: Record<string, unknown> = {}) => ({
  type: "state", connected: true, heartbeat_age: 0.5, settings_version: 1, time: 1000 + n,
  snapshot: {
    status: "streaming", shape: [2560, 1440], frame_ts: 1000 + n, seq: n, camera: "Front", mode: "dynamic",
    counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 2, learning: 0 },
    slots: [slot(1, "FREE", 0.1), slot(2, "BUSY", 0.5)], detections: [], area: { top: 0, bottom: 0, left: 0, right: 0, roi: null },
    geometry: { version: 3, valid: true, confidence: "HIGH", uncertainty: 0.05, reference_used: true, h: 2.3, lens: null, area_m2: 156,
                counts: { capacity: 2, occupied: 1, free: 1, vehicles: 1, measured: 1, beyond_range: 0 }, bays: [], vehicles: [], distances: [] },
    ...over,
  },
});
const camera = {
  id: "front", name: "Front", label: "", display: "Front", primary: true, sort_order: 0, metadata: {}, rtsp_url: "", rtsp_sub_url: "",
  streams: { hd: "front_hd", sd: "front_sd", native: "front_native" },
  status: { connected: true, heartbeat_age: 1, enabled: true, running: true, slot_mode: "dynamic", counts: null, state: null },
};

beforeAll(() => {
  class Silent {
    close() {}
    addTransceiver() {}
    addEventListener() {}
    removeEventListener() {}
    createOffer() {
      return new Promise(() => {});
    }
  }
  vi.stubGlobal("RTCPeerConnection", Silent);
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: () => "blob:x", revokeObjectURL: () => {} }));
});
beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  FakeSocket.all = [];
  vi.stubGlobal("fetch", (url: string) => {
    const json = (b: unknown) => Promise.resolve(new Response(JSON.stringify(b), { status: 200, headers: { "Content-Type": "application/json" } }));
    if (url.startsWith("/api/auth")) return json({ view: true, admin: false, view_required: false, writes_enabled: true });
    if (url.startsWith("/api/cameras")) return json([camera]);
    if (url.startsWith("/api/events")) return json([]);
    return json({});
  });
});
afterEach(cleanup);

const push = (msg: unknown) => act(async () => socket().onmessage?.({ data: JSON.stringify(msg) }));
const pill = (label: string) => screen.getByText(label).closest("button")!;
const on = (label: string) => pill(label).className.includes("bg-primary");
const dashboard = async () => {
  render(<App />);
  await screen.findByTestId("camera-switcher"); // the camera list is in: the live state is for this camera from here on
  await act(async () => new Promise((r) => setTimeout(r, 300))); // sockets settle (frame images on / off reconnect once)
  await push(state(1));
  await waitFor(() => expect(screen.getAllByTestId("overlay-slot")).toHaveLength(2));
};

describe("the dashboard keeps the viewer's choices through the live updates", () => {
  it("layers, Metrics and the BUSY filter survive many state messages, one without geometry, and a reconnect", async () => {
    await dashboard();
    for (const label of ["Slots", "Boxes", "Area", "Metrics"]) fireEvent.click(pill(label));
    fireEvent.click(screen.getByTestId("filter-busy"));
    const chosen = () => ["Slots", "Boxes", "Area", "Metrics"].map(on);
    expect(chosen()).toEqual([false, false, false, false]);
    expect(screen.getByTestId("filter-chip").textContent).toContain("Showing BUSY only");
    for (let n = 2; n <= 6; n++) await push(state(n));
    await push(state(7, { geometry: undefined })); // a message without Phase 2 data
    await push({ type: "state", connected: false, heartbeat_age: null, settings_version: 1, time: 1, snapshot: { status: "loading the default model" } }); // worker reloading
    await push(state(8));
    expect(chosen()).toEqual([false, false, false, false]);
    expect(screen.getByTestId("filter-chip").textContent).toContain("Showing BUSY only");
    await act(async () => socket().onclose?.({ code: 1006 })); // the socket drops and reconnects
    await act(async () => new Promise((r) => setTimeout(r, 1200)));
    await push(state(9));
    expect(chosen()).toEqual([false, false, false, false]);
    expect(screen.getByTestId("filter-chip").textContent).toContain("Showing BUSY only");
    expect(JSON.parse(localStorage.getItem(LAYERS_KEY) ?? "{}")).toEqual({ slots: false, boxes: false, area: false });
    expect(localStorage.getItem(METRICS_KEY)).toBe("0");
  });

  it("the BUSY filter and the hidden layers come back after the page is opened again, and another camera has its own filter", async () => {
    await dashboard();
    fireEvent.click(pill("Boxes"));
    fireEvent.click(screen.getByTestId("filter-busy"));
    cleanup(); // the page is closed and opened again (reload, or the app remounting)
    FakeSocket.all = [];
    await dashboard();
    expect(screen.getByTestId("filter-chip").textContent).toContain("Showing BUSY only");
    expect(on("Boxes")).toBe(false);
    expect(on("Slots")).toBe(true);
    expect(JSON.parse(sessionStorage.getItem("carparking.slotFilter") ?? "{}")).toEqual({ front: "BUSY" });
  });
});
