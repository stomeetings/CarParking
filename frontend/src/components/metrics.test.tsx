import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { CameraInfo, GeometryLive, Snapshot } from "../types";
import { LAYERS_KEY, LiveView, savedLayers } from "./LiveView";
import { METRICS_KEY, MetricCard, metres, savedMetrics } from "./Metrics";

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
beforeEach(() => {
  localStorage.removeItem(LAYERS_KEY);
  localStorage.removeItem(METRICS_KEY);
});
afterEach(cleanup);

const quad = (x: number, y: number): [number, number][] => [[x, y], [x + 0.1, y], [x + 0.1, y + 0.1], [x, y + 0.1]];
const geometry = (over: Partial<GeometryLive> = {}): GeometryLive => ({
  version: 3, valid: true, confidence: "HIGH", uncertainty: 0.05, reference_used: true, h: 3.24, lens: "simple_radial", area_m2: 41.4,
  counts: { capacity: 3, occupied: 2, free: 1, vehicles: 4, measured: 2, beyond_range: 2 },
  vehicles: [{ i: 0, poly: quad(0.1, 0.5), centroid_m: [0, 10] }, { i: 1, poly: quad(0.3, 0.5), centroid_m: [0, 12] }],
  distances: [{ kind: "vehicle", m: 0.43, p: [0.2, 0.55], q: [0.3, 0.55] }, { kind: "object", label: "person", m: 1.8, p: [0.6, 0.6], q: [0.4, 0.55] }],
  bays: [{ id: 1, poly: quad(0.1, 0.5), state: "BUSY", car: true }, { id: 2, poly: quad(0.3, 0.5), state: "BUSY", car: true }, { id: 3, poly: quad(0.5, 0.5), state: "FREE", car: false }],
  ...over,
});
const snap = (g?: GeometryLive) => ({
  shape: [2560, 1440], frame_ts: 1,
  counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 2, learning: 0 },
  slots: [{ id: 1, name: "1", state: "FREE", hidden: false, phantom: false, duplicate_of: null, poly: quad(0.1, 0.1) },
          { id: 2, name: "2", state: "BUSY", hidden: false, phantom: false, duplicate_of: null, poly: quad(0.5, 0.1) }],
  detections: [{ label: "car", conf: 0.9, src: "", kind: "vehicle", box: [0.5, 0.2, 0.6, 0.4] }],
  area: { top: 0.1, bottom: 0, left: 0, right: 0, roi: null },
  ...(g ? { geometry: g } : {}),
}) as unknown as Snapshot;
const camera = { id: "front", display: "Front", streams: { hd: "front_hd", sd: "front_sd", native: "front_native" } } as CameraInfo;
const player = (s: Snapshot) => <LiveView snapshot={s} frameUrl={null} onNeedFrames={() => {}} camera={camera} tokenKey="" />;
const pressed = (label: string) => screen.getByText(label).closest("button")!.className.includes("bg-primary");

describe("the metric layer on the video", () => {
  it("without geometry (flag off, or the camera not calibrated) nothing of it exists", () => {
    render(player(snap()));
    expect(screen.queryByText("Metrics")).toBeNull();
    expect(screen.queryByTestId("metric-overlay")).toBeNull();
    expect(screen.getAllByTestId("overlay-slot")).toHaveLength(2); // the slot overlay as ever
    expect(savedLayers()).toEqual({ slots: true, boxes: true, area: true });
  });

  it("draws bays, footprints and distances over the slots, a ≈ on the metres unless the scale is anchored", () => {
    const view = render(player(snap(geometry({ confidence: "LOW", uncertainty: 0.3 }))));
    expect(screen.getByText("Metrics")).toBeTruthy();
    expect(screen.getAllByTestId("metric-bay").map((b) => b.getAttribute("data-state"))).toEqual(["BUSY", "BUSY", "FREE"]);
    expect(screen.getAllByTestId("metric-footprint")).toHaveLength(2);
    const labels = screen.getAllByTestId("metric-distance");
    expect(labels).toHaveLength(2);
    expect(labels[0].textContent).toContain("≈ 0.4 m");
    expect(labels[1].getAttribute("data-kind")).toBe("object");
    expect(labels[1].textContent).toContain("person to the nearest vehicle");
    expect(screen.getAllByTestId("overlay-slot")).toHaveLength(2); // the slots are still there, untouched
    view.rerender(player(snap(geometry()))); // anchored by a typed distance: exact
    expect(screen.getAllByTestId("metric-distance")[0].textContent).toContain("0.4 m");
    expect(screen.getAllByTestId("metric-distance")[0].textContent).not.toContain("≈");
  });

  it("switches off and on, remembered on its own and leaving the other layers' storage alone", () => {
    const view = render(player(snap(geometry())));
    expect(pressed("Metrics")).toBe(true);
    fireEvent.click(screen.getByText("Metrics"));
    expect(pressed("Metrics")).toBe(false);
    expect(screen.queryByTestId("metric-overlay")).toBeNull();
    expect(screen.getAllByTestId("overlay-slot")).toHaveLength(2);
    expect(savedMetrics()).toBe(false);
    expect(localStorage.getItem(LAYERS_KEY)).toBeNull(); // Slots / Boxes / Area preferences were never touched
    view.unmount();
    render(player(snap(geometry()))); // the player is rebuilt: still off
    expect(pressed("Metrics")).toBe(false);
    fireEvent.click(screen.getByText("Metrics"));
    expect(screen.getByTestId("metric-overlay")).toBeTruthy();
    expect(savedMetrics()).toBe(true);
  });

  it("a calibration that does not fit this picture draws nothing", () => {
    render(player(snap(geometry({ valid: false, reason: "the frame is not the picture this calibration was made on", counts: undefined, bays: [], vehicles: [], distances: [] }))));
    expect(screen.queryByTestId("metric-overlay")).toBeNull();
  });
});

describe("the measured capacity card", () => {
  it("shows bays, occupied and free beside the slot counters, with what was measured", () => {
    render(<MetricCard geometry={geometry()} />);
    expect(screen.getByTestId("metric-bays").textContent).toBe("3");
    expect(screen.getByTestId("metric-occupied").textContent).toBe("2");
    expect(screen.getByTestId("metric-free").textContent).toBe("1");
    expect(screen.getByTestId("metric-confidence").textContent).toBe("HIGH");
    const notes = screen.getByTestId("metric-notes").textContent ?? "";
    expect(notes).toContain("41 m² measured");
    expect(notes).toContain("2 of 4 vehicles measured (2 too far to place)");
    expect(notes).toContain("calibration v3");
    expect(notes).not.toContain("±"); // anchored: no caveat
  });

  it("says how far the metres can be trusted when they are not anchored", () => {
    render(<MetricCard geometry={geometry({ confidence: "MEDIUM", uncertainty: 0.25 })} />);
    expect(screen.getByTestId("metric-notes").textContent).toContain("about ±25 %");
  });

  it("shows what the drift guard last saw, with its tone", () => {
    const now = Date.now() / 1000;
    const { rerender } = render(<MetricCard geometry={geometry({ drift: { status: "ok", shift_px: 1.8, at: now - 12, version: 3, baseline: "calibration" }, drift_guard: true })} />);
    expect(screen.getByTestId("metric-drift").textContent).toMatch(/^Camera position unchanged \(2 px\) · checked (11|12|13) s ago$/);
    expect(screen.getByTestId("metric-drift").className).toContain("text-success");
    rerender(<MetricCard geometry={geometry({ drift: { status: "drift", shift_px: 15, at: now, version: 3, baseline: "calibration" }, drift_guard: true })} />);
    expect(screen.getByTestId("metric-drift").textContent).toContain("may have moved (15 px)");
    expect(screen.getByTestId("metric-drift").className).toContain("text-warning");
    rerender(<MetricCard geometry={geometry({ drift: { status: "inconclusive", at: now, version: 3, baseline: "calibration", why: "only 4 matches" }, drift_guard: true })} />);
    expect(screen.getByTestId("metric-drift").textContent).toContain("can't be checked now (only 4 matches)");
    rerender(<MetricCard geometry={geometry({ drift: null, drift_guard: false })} />);
    expect(screen.getByTestId("metric-drift").textContent).toContain("no stored reference");
  });

  it("renders nothing without geometry, and explains an invalid one", () => {
    const { container, rerender } = render(<MetricCard geometry={undefined} />);
    expect(container.textContent).toBe("");
    rerender(<MetricCard geometry={geometry({ valid: false, reason: "wrong picture", counts: undefined })} />);
    expect(screen.getByTestId("metric-invalid").textContent).toBe("wrong picture");
    expect(screen.queryByTestId("metric-card")).toBeNull();
  });

  it("formats metres", () => {
    expect(metres(0.43, "HIGH")).toBe("0.4 m");
    expect(metres(5.96, "LOW")).toBe("≈ 6.0 m");
    expect(metres(12.4, "MEDIUM")).toBe("≈ 12 m");
    expect(metres(1, null)).toBe("≈ 1.0 m");
  });
});
