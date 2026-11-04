import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { CameraInfo, Snapshot } from "../types";
import { LAYERS_KEY, LiveView, savedLayers } from "./LiveView";

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
beforeEach(() => localStorage.removeItem(LAYERS_KEY));
afterEach(cleanup);

const box = (x: number): [number, number][] => [[x, 0.2], [x + 0.1, 0.2], [x + 0.1, 0.4], [x, 0.4]];
const snap = (ts: number) => ({
  shape: [2560, 1440], frame_ts: ts,
  counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 2, learning: 0 },
  slots: [
    { id: 1, name: "1", state: "FREE", hidden: false, phantom: false, duplicate_of: null, poly: box(0.1) },
    { id: 2, name: "2", state: "BUSY", hidden: false, phantom: false, duplicate_of: null, poly: box(0.5) },
  ],
  detections: [{ label: "car", conf: 0.9, src: "", kind: "vehicle", box: [0.5, 0.2, 0.6, 0.4] }],
  area: { top: 0.1, bottom: 0, left: 0, right: 0, roi: null },
}) as unknown as Snapshot;
const camera = { id: "front", display: "Front", streams: { hd: "front_hd", sd: "front_sd", native: "front_native" } } as CameraInfo;
const player = (ts: number, refresh = 0) => (
  <LiveView snapshot={snap(ts)} frameUrl={null} onNeedFrames={() => {}} camera={camera} tokenKey="" refresh={refresh} />
);
const pressed = (label: string) => screen.getByText(label).closest("button")!.className.includes("bg-primary");

describe("overlay toggles (Slots / Boxes / Area) survive live updates", () => {
  it("a layer switched off stays off on new frames, a Refresh, and when the player mounts again", () => {
    const view = render(player(1));
    expect([pressed("Slots"), pressed("Boxes"), pressed("Area")]).toEqual([true, true, true]); // default: all on
    const slotsDrawn = screen.getAllByTestId("overlay-slot").length;
    fireEvent.click(screen.getByText("Boxes"));
    fireEvent.click(screen.getByText("Slots"));
    expect([pressed("Slots"), pressed("Boxes"), pressed("Area")]).toEqual([false, false, true]);
    expect(screen.queryAllByTestId("overlay-slot")).toHaveLength(0);
    view.rerender(player(2)); // a new frame / state message
    view.rerender(player(3, 1)); // Refresh data
    expect([pressed("Slots"), pressed("Boxes"), pressed("Area")]).toEqual([false, false, true]);
    expect(screen.queryAllByTestId("overlay-slot")).toHaveLength(0);
    view.unmount(); // the player is rebuilt (reconnect, camera page again)
    render(player(4));
    expect([pressed("Slots"), pressed("Boxes"), pressed("Area")]).toEqual([false, false, true]);
    expect(screen.queryAllByTestId("overlay-slot")).toHaveLength(0);
    fireEvent.click(screen.getByText("Slots")); // switched on again: remembered too
    expect(screen.getAllByTestId("overlay-slot")).toHaveLength(slotsDrawn);
    expect(savedLayers()).toEqual({ slots: true, boxes: false, area: true });
  });

  it("unreadable or missing preferences fall back to all layers on", () => {
    localStorage.setItem(LAYERS_KEY, "{not json");
    expect(savedLayers()).toEqual({ slots: true, boxes: true, area: true });
    localStorage.setItem(LAYERS_KEY, JSON.stringify({ area: false }));
    expect(savedLayers()).toEqual({ slots: true, boxes: true, area: false });
  });
});


// what is actually drawn in the overlay SVG: slot outlines, vehicle boxes, the dimmed area (invisible "ghost" shapes, kept for
// hovering, are not drawn)
const drawn = () => {
  const svg = screen.getByTestId("overlay");
  const seen = (el: Element) => el.getAttribute("stroke") !== "transparent";
  return {
    slots: screen.queryAllByTestId("overlay-slot").filter((g) => seen(g.querySelector("polygon")!)).length,
    boxes: [...svg.querySelectorAll(":scope > rect")].filter(seen).length,
    area: svg.querySelectorAll(":scope > path").length,
  };
};

describe("each overlay layer is drawn by its own toggle alone", () => {
  const combos = [0, 1, 2, 3, 4, 5, 6, 7].map((n) => ({ slots: !!(n & 1), boxes: !!(n & 2), area: !!(n & 4) }));
  it.each(combos)("video: %o", (want) => {
    localStorage.setItem(LAYERS_KEY, JSON.stringify(want));
    render(player(1));
    expect(drawn()).toEqual({ slots: want.slots ? 2 : 0, boxes: want.boxes ? 1 : 0, area: want.area ? 1 : 0 });
  });

  const frames = (clean: boolean | undefined) => {
    const s = { ...snap(1), clean_frames: clean } as unknown as Snapshot;
    render(<LiveView snapshot={s} frameUrl="blob:x" onNeedFrames={() => {}} camera={camera} tokenKey="" refresh={0} />);
    fireEvent.click(screen.getByText("Frames")); // the worker's frame images instead of the video
    expect(screen.getByTestId("frame-image")).toBeTruthy();
  };
  it.each(combos)("frame images without drawing (clean_frames): %o", (want) => {
    localStorage.setItem(LAYERS_KEY, JSON.stringify(want));
    frames(true);
    expect(drawn()).toEqual({ slots: want.slots ? 2 : 0, boxes: want.boxes ? 1 : 0, area: want.area ? 1 : 0 });
  });
  it("frame images with the worker's own drawing (an older worker): the SVG only hovers, it never draws twice", () => {
    frames(undefined);
    expect(drawn()).toEqual({ slots: 0, boxes: 0, area: 0 });
  });
});


describe("auto grid bays by source", () => {
  it("dots an extrapolated bay and marks a pinned one", () => {
    const base = snap(1);
    const s = { ...base, auto_slots: { version: 1, count: 2, sources: { "1": "extrapolated", "2": "manual" } } } as unknown as Snapshot;
    render(<LiveView snapshot={s} frameUrl={null} onNeedFrames={() => {}} camera={camera} tokenKey="" refresh={0} />);
    const [a, b] = screen.getAllByTestId("overlay-slot");
    expect([a.getAttribute("data-source"), b.getAttribute("data-source")]).toEqual(["extrapolated", "manual"]);
    expect(a.querySelector("polygon")!.getAttribute("stroke-dasharray")).toBe("2 4");
    expect(b.textContent).toContain("2P");
  });
});

describe("stream health banner", () => {
  const withStream = (stream: unknown) => ({ ...snap(1), stream }) as unknown as Snapshot;
  const view = (s: Snapshot) => render(<LiveView snapshot={s} frameUrl={null} onNeedFrames={() => {}} camera={camera} tokenKey="" refresh={0} />);

  it("warns when the stream is stale or frozen, and says why", () => {
    view(withStream({ state: "stale", age_s: 42.4, for_s: 32, diff: 0, same_s: 0 }));
    expect(screen.getByTestId("stream-banner").textContent).toContain("stale: no new frame for 42 s");
    cleanup();
    view(withStream({ state: "frozen", age_s: 0.4, for_s: 5, diff: 0.02, same_s: 25.3 }));
    expect(screen.getByTestId("stream-banner").textContent).toContain("frozen: the same picture for 25 s");
  });

  it("stays quiet while the stream is active or the worker does not report one", () => {
    view(withStream({ state: "active", age_s: 0.4, for_s: 0, diff: 3, same_s: 1 }));
    expect(screen.queryByTestId("stream-banner")).toBeNull();
    cleanup();
    view(snap(1));
    expect(screen.queryByTestId("stream-banner")).toBeNull();
  });
});
