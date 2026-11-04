import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { hourProfile, metricLabel, metricRows, summarize } from "../analyticsMetrics";
import type { HistoryPoint } from "../setupflow";
import type { GeometryLive } from "../types";
import AnalyticsPage from "./Analytics";
import { MetricAnalytics } from "./MetricAnalytics";

// jsdom has no layout: recharts' ResponsiveContainer needs a size to draw anything
beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const pt = (t: number, occ: number, cap: number, extra: Partial<HistoryPoint> = {}): HistoryPoint => ({
  t, samples: 10, occupied_avg: occ, occupied_max: Math.ceil(occ), occupied_min: Math.floor(occ), capacity_avg: cap, capacity_max: cap, vehicles_avg: occ, ver: 1, ...extra,
});
const hour = (h: number) => new Date(2026, 9, 20, h, 0, 0).getTime() / 1000; // local time, so the hour of day is what it says

describe("metric analytics rules", () => {
  it("rows: utilization and peak as percent of the bays, null without bays", () => {
    const r = metricRows([pt(1, 3, 12, { occupied_max: 6 }), pt(2, 0, 0, { occupied_max: 0 })]);
    expect(r[0]).toMatchObject({ occupied: 3, peak: 6, capacity: 12, util: 25, peakUtil: 50 });
    expect(r[1]).toMatchObject({ util: null, peakUtil: null });
  });
  it("summary: sample-weighted average, the highest peak, the newest bay count", () => {
    const s = summarize([pt(1, 2, 10, { samples: 30, occupied_max: 4 }), pt(2, 8, 12, { samples: 10, occupied_max: 11 })]);
    expect(s.samples).toBe(40);
    expect(s.avgOccupied).toBe(3.5); // (2*30 + 8*10) / 40
    expect(s.avgUtil).toBeCloseTo(33.3, 1); // 3.5 occupied of 10.5 bays (also sample-weighted)
    expect(s.peakUtil).toBeCloseTo(91.7, 1); // 11 of 12
    expect(s.peakOccupied).toBe(11);
    expect(s.capacity).toBe(12);
    expect(summarize([])).toEqual({ samples: 0, avgUtil: null, peakUtil: null, peakOccupied: null, capacity: null, avgOccupied: null });
  });
  it("hour profile: 24 entries, empty hours null, same hours pooled", () => {
    const p = hourProfile([pt(hour(9), 6, 12), pt(hour(9) + 86400, 3, 12), pt(hour(17), 12, 12)]);
    expect(p).toHaveLength(24);
    expect(p[9].util).toBe(37.5); // (6 + 3) / (12 + 12)
    expect(p[17].util).toBe(100);
    expect(p[3].util).toBeNull();
  });
  it("labels: a time within a day, a date and time beyond", () => {
    expect(metricLabel(hour(9.5 as number), 24)).toMatch(/^\d\d:\d\d$/);
    expect(metricLabel(hour(9), 168)).toBe("20/10 09:00");
  });
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const LIVE = {
  version: 1, valid: true, confidence: "HIGH", uncertainty: 0.05, reference_used: true, h: 3.2, lens: null, area_m2: 212.4,
  counts: { capacity: 12, occupied: 5, free: 7, vehicles: 6, measured: 5, beyond_range: 1 }, vehicles: [], distances: [], bays: [],
} as GeometryLive;
const HISTORY = { since: hour(0), until: hour(23), bucket_s: 3600, versions: [1, 2], points: [pt(hour(9), 6, 12, { occupied_max: 8 }), pt(hour(10), 9, 12, { occupied_max: 11 })] };

function serve(history: () => Promise<Response>) {
  const calls: string[] = [];
  vi.stubGlobal("fetch", (url: string) => {
    calls.push(url);
    if (url.includes("/geometry/history")) return history();
    return json({});
  });
  return calls;
}

describe("the Measured occupancy section", () => {
  it("shows the metric KPIs, trend and hour profile next to the live metric counts", async () => {
    const calls = serve(() => json(HISTORY));
    render(<MetricAnalytics tokenKey="" cam="front" days={7} geometry={LIVE} />);
    await screen.findByTestId("metric-analytics");
    expect(calls).toEqual(["/api/cameras/front/geometry/history?hours=168&points=120"]);
    const kpis = screen.getByTestId("metric-kpis").textContent ?? "";
    expect(kpis).toContain("Average utilization");
    expect(kpis).toContain("63 %"); // (6 + 9) / 24 bays
    expect(kpis).toContain("92 %"); // the peak: 11 of 12
    expect(kpis).toContain("212 m²");
    expect(kpis).toContain("HIGH ±5 %");
    expect(kpis).toContain("5 / 12");
    expect(kpis).toContain("1 beyond range");
    expect(screen.getByTestId("metric-analytics").textContent).toContain("v1, v2");
    expect(screen.getByTestId("metric-trend")).toBeTruthy();
    expect(screen.getByTestId("metric-hours")).toBeTruthy();
  });

  it("without a live calibrated frame it still shows the history (no 'now' tiles)", async () => {
    serve(() => json(HISTORY));
    render(<MetricAnalytics tokenKey="" cam="front" days={1} geometry={undefined} />);
    await screen.findByTestId("metric-analytics");
    const kpis = screen.getByTestId("metric-kpis").textContent ?? "";
    expect(kpis).not.toContain("Occupied now");
    expect(kpis).not.toContain("Surface");
    expect(kpis).toContain("Average utilization");
  });

  it("a frame that is not the calibrated picture shows no live metric tiles", async () => {
    serve(() => json(HISTORY));
    render(<MetricAnalytics tokenKey="" cam="front" days={1} geometry={{ ...LIVE, valid: false }} />);
    await screen.findByTestId("metric-analytics");
    expect(screen.getByTestId("metric-kpis").textContent).not.toContain("Occupied now");
  });

  it.each([
    ["the feature off (503)", () => json({ detail: "geometry is off" }, 503)],
    ["no calibration (404)", () => json({ detail: "no such camera" }, 404)],
    ["no samples yet", () => json({ ...HISTORY, points: [], versions: [] })],
    ["a network failure", () => Promise.reject(new Error("offline"))],
  ])("renders nothing for %s, without console errors", async (_name, reply) => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const calls = serve(reply);
    const { container } = render(<MetricAnalytics tokenKey="" cam="front" days={7} geometry={undefined} />);
    await waitFor(() => expect(calls.length).toBeGreaterThan(0));
    await new Promise((r) => setTimeout(r, 30));
    expect(container.innerHTML).toBe("");
    expect(err).not.toHaveBeenCalled();
  });

  it("asks nothing without a camera", async () => {
    const calls = serve(() => json(HISTORY));
    const { container } = render(<MetricAnalytics tokenKey="" cam={null} days={7} geometry={undefined} />);
    await new Promise((r) => setTimeout(r, 30));
    expect(calls).toEqual([]);
    expect(container.innerHTML).toBe("");
  });
});

describe("the Analytics page", () => {
  const ANALYTICS = {
    days: 7, since: "x", bucket: "1 h", source: "postgres", scope: "all", cam: "front",
    totals: { entries: 4, exits: 3, sessions: 4, closed: 3, open: 1, avg_min: 30, median_min: 25, busiest_hour: "9", peak_busy: 5 },
    per_hour: [], per_day: [], durations: [], occupancy: [],
  };
  const page = (history: () => Promise<Response>) => {
    vi.stubGlobal("fetch", (url: string) => {
      if (url.includes("/geometry/history")) return history();
      if (url.startsWith("/api/analytics/overview")) return json({ cameras: [] });
      if (url.startsWith("/api/analytics")) return json(ANALYTICS);
      return json({}, 404);
    });
    return render(<AnalyticsPage tokenKey="" snapshot={{ geometry: LIVE } as never} cam="front" />);
  };

  it("legacy analytics only when the feature is off: same KPIs, no metric section, no console errors", async () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    page(() => json({ detail: "off" }, 503));
    await waitFor(() => expect(screen.getByTestId("kpis").textContent).toContain("Arrivals"));
    await new Promise((r) => setTimeout(r, 30));
    expect(screen.queryByTestId("metric-analytics")).toBeNull();
    expect(screen.getByTestId("analytics-bar")).toBeTruthy();
    expect(err).not.toHaveBeenCalled();
  });

  it("adds the measured section above the legacy charts when there is history", async () => {
    page(() => json(HISTORY));
    const section = await screen.findByTestId("metric-analytics");
    expect(screen.getByTestId("kpis").textContent).toContain("Arrivals"); // the legacy KPIs are still there
    expect(section.compareDocumentPosition(screen.getByTestId("analytics-bar")) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy(); // after the bar
  });
});
