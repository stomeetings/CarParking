import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GeometryEvent, HistoryResponse } from "../setupflow";
import type { CameraInfo } from "../types";
import { CameraSetup } from "./CameraSetup";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
beforeEach(() => {
  // jsdom does not lay anything out: the still's overlay is 1000 x 500 for every click
  vi.spyOn(SVGElement.prototype, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0, toJSON: () => ({}) });
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const camera: CameraInfo = {
  id: "front", name: "Front", label: "", display: "Front", primary: true, sort_order: 0, metadata: {}, rtsp_url: "", rtsp_sub_url: "",
  streams: { hd: "front_hd", sd: "front_sd", native: "front_native" },
  status: { connected: true, heartbeat_age: 1, enabled: true, running: true, slot_mode: "dynamic", counts: null, state: null },
};
const POLY = [[0.1, 0.5], [0.9, 0.5], [0.9, 0.9], [0.1, 0.9]];
const summary = (ver: number, state: string, confidence: string | null, extra = {}) => ({
  ver, state, note: "", created_at: 1792650000 + ver, activated_at: null, confidence, h: 3.2, scale_uncertainty: confidence === "HIGH" ? 0.05 : 0.25,
  reference_used: confidence === "HIGH", lens: "simple_radial", area_m2: 40, capacity: 2, bays_with_cars: 2, warnings: [], ...extra,
});

interface Server {
  calls: { method: string; url: string; body: unknown }[];
  status: { enabled: boolean; service: "up" | "down" | "unchecked" };
  confidence: "HIGH" | "LOW";
  versions: ReturnType<typeof summary>[];
  unauthorized: boolean;
  events: GeometryEvent[];
  history: HistoryResponse;
}

function serve(over: Partial<Server> = {}): Server {
  const s: Server = { calls: [], status: { enabled: true, service: "up" }, confidence: "HIGH", versions: [], unauthorized: false, events: [],
                       history: { since: 1792650000, until: 1792660000, bucket_s: 60, points: [], versions: [] }, ...over };
  const jobs: Record<string, unknown> = {};
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    s.calls.push({ method, url, body });
    if (s.unauthorized) return json({ detail: "missing or wrong API token" }, 401);
    if (url === "/api/setup/status") return json({ ...s.status, loaded: null, pins: null });
    if (url === "/api/cameras/front/setup/geometry") return json({ active: s.versions.find((v) => v.state === "active") ?? null, versions: s.versions });
    if (url.startsWith("/api/cameras/front/geometry/events")) return json({ events: s.events });
    if (url.startsWith("/api/cameras/front/geometry/history")) return json(s.history);
    if (url === "/api/cameras/front/setup/still") return json({ id: "a".repeat(32), ts: 1792650000, source: "camera", width: 1920, height: 1080, image: "data:image/jpeg;base64,STILL" });
    if (url === "/api/cameras/front/setup/segment") {
      jobs.s = { id: "s", kind: "segment", cam: "front", state: "done", stage: "done", progress: 1, error: null,
                 result: { polygon: POLY, score: 0.8, area_frac: 0.3, model: "sam2.1", warnings: [],
                           candidates: [{ polygon: [[0, 0], [1, 0], [1, 1]], score: 0.5, area_frac: 0.1 }, { polygon: POLY, score: 0.8, area_frac: 0.3 }] } };
      return json(jobs.s);
    }
    if (url === "/api/cameras/front/setup/calibrate") {
      const ver = Math.max(0, ...s.versions.map((v) => v.ver)) + 1;
      s.versions = [summary(ver, "draft", s.confidence), ...s.versions];
      jobs.c = { id: "c", kind: "calibrate", cam: "front", state: "done", stage: "done", progress: 1, error: null,
                 result: { version: ver, state: "draft", confidence: s.confidence, h: 3.24, pitch_deg: 13.8, area_m2: 41, capacity: 2, bays_with_cars: 2,
                           reference_used: s.confidence === "HIGH", scale_uncertainty: s.confidence === "HIGH" ? 0.05 : 0.25, scale_sources: {},
                           warnings: s.confidence === "HIGH" ? [] : ["the scale rests on one prior"] } };
      return json(jobs.c);
    }
    if (url.startsWith("/api/setup/jobs/")) return json(jobs[url.endsWith("/s") ? "s" : "c"]);
    if (/\/setup\/geometry\/\d+$/.test(url)) {
      return json({ ...s.versions[0], record: { surface: { polygon: POLY },
                    capacity: { slots: [{ id: 1, poly: [[0.2, 0.6], [0.3, 0.6], [0.3, 0.7], [0.2, 0.7]], car: true }, { id: 2, poly: [[0.4, 0.6], [0.5, 0.6], [0.5, 0.7], [0.4, 0.7]], car: false }] },
                    calibration: { f: 1986, k1: -0.46, pitch: 0.24 } } });
    }
    if (url === "/api/cameras/front/setup/activate") {
      s.versions = s.versions.map((v) => (v.ver === body.version ? { ...v, state: "active" } : v.state === "active" ? { ...v, state: "superseded" } : v));
      return json({ active: s.versions.find((v) => v.ver === body.version), previous: null });
    }
    if (url.startsWith("/api/cameras/front/setup/draft") && method === "DELETE") {
      const ver = Number(new URL(url, "http://x").searchParams.get("version"));
      s.versions = s.versions.filter((v) => v.ver !== ver);
      return json({ discarded: [ver] });
    }
    return json({ detail: `unexpected ${method} ${url}` }, 404);
  });
  return s;
}

const page = (onNeedAdmin = vi.fn()) => ({ onNeedAdmin, ...render(<CameraSetup cam="front" cameras={[camera]} onCam={() => {}} tokenKey="" onNeedAdmin={onNeedAdmin} />) });
const click = (x: number, y: number) => fireEvent.click(screen.getByTestId("setup-svg"), { clientX: x, clientY: y });
const posted = (s: Server, suffix: string) => s.calls.filter((c) => c.method === "POST" && c.url.endsWith(suffix)).map((c) => c.body);

/** still -> the two end points + the metres -> next */
async function giveReference(metres = "2.61") {
  fireEvent.click(await screen.findByText("Take a still"));
  await screen.findByTestId("setup-svg");
  click(200, 350);
  click(400, 375);
  fireEvent.change(screen.getByTestId("reference-metres"), { target: { value: metres } });
}

describe("camera setup page", () => {
  it("switched off on the server: says so and offers nothing else", async () => {
    const s = serve({ status: { enabled: false, service: "unchecked" } });
    page();
    await waitFor(() => expect(screen.getByTestId("setup-status").textContent).toContain("CARPARK_GEOMETRY_ENABLED=1"));
    expect(screen.queryByTestId("steps")).toBeNull();
    expect(screen.queryByText("Take a still")).toBeNull();
    expect(s.calls.map((c) => c.url)).toEqual(["/api/setup/status"]); // nothing else was asked
  });

  it("asks for the admin token when the server refuses", async () => {
    serve({ unauthorized: true });
    const { onNeedAdmin } = page();
    await waitFor(() => expect(onNeedAdmin).toHaveBeenCalled());
    expect(onNeedAdmin.mock.calls[0][0]).toContain("admin token");
  });

  it("the setup service being down is shown, and Find the surface stays closed", async () => {
    serve({ status: { enabled: true, service: "down" } });
    page();
    await waitFor(() => expect(screen.getByTestId("setup-status").textContent).toContain("docker compose --profile setup up -d setup"));
    fireEvent.click(await screen.findByText("Take a still"));
    await screen.findByTestId("setup-svg");
    fireEvent.click(screen.getByTestId("step-3"), {}); // blocked: no distance yet
    expect((screen.getByTestId("step-3") as HTMLButtonElement).disabled).toBe(true);
  });

  it("the known distance comes first: step 3 stays shut until it is given (or waived)", async () => {
    serve();
    page();
    fireEvent.click(await screen.findByText("Take a still"));
    await screen.findByTestId("setup-svg");
    expect(screen.getByTestId("step-2").getAttribute("aria-current")).toBe("step"); // taken to the distance, not the surface
    expect(screen.getByTestId("reference-hint").textContent).toContain("Type one distance you really know");
    expect((screen.getByTestId("step-3") as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByText("Next: parking surface") as HTMLButtonElement).disabled).toBe(true);
    click(200, 350);
    expect(screen.getAllByTestId("reference-point")).toHaveLength(1);
    expect(screen.getByTestId("reference-problem").textContent).toContain("two end points");
    click(400, 375);
    expect(screen.getAllByTestId("reference-point")).toHaveLength(2);
    expect(screen.getByTestId("reference-problem").textContent).toContain("Type the distance");
    fireEvent.change(screen.getByTestId("reference-metres"), { target: { value: "2.61" } });
    expect(screen.queryByTestId("reference-problem")).toBeNull();
    expect((screen.getByText("Next: parking surface") as HTMLButtonElement).disabled).toBe(false);
    click(300, 100); // a third click starts the distance over
    expect(screen.getAllByTestId("reference-point")).toHaveLength(1);
    expect((screen.getByText("Next: parking surface") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByTestId("reference-skip")); // the explicit decision to go without
    expect((screen.getByText("Next: parking surface") as HTMLButtonElement).disabled).toBe(false);
  });

  it("the full flow with a known distance: HIGH, accepted without further ado, nothing restarted", async () => {
    const s = serve({ confidence: "HIGH" });
    page();
    await giveReference();
    fireEvent.click(screen.getByText("Next: parking surface"));
    click(500, 400); click(700, 450);
    fireEvent.click(screen.getByText("Not ground"));
    click(900, 50);
    expect(screen.getByTestId("clicks-count").textContent).toContain("2 ground · 1 not ground");
    fireEvent.click(screen.getByText("Find the surface"));
    await screen.findByTestId("segment-result");
    expect(posted(s, "/segment")).toEqual([{ still: "a".repeat(32), clicks: [{ x: 0.5, y: 0.8, label: 1 }, { x: 0.7, y: 0.9, label: 1 }, { x: 0.9, y: 0.1, label: 0 }] }]);
    expect(screen.getByTestId("candidate-1").textContent).toContain("30 %"); // the server's pick is selected
    fireEvent.click(screen.getByText("Next: calibrate"));
    expect(screen.getByTestId("inputs-summary").textContent).toContain("Known distance 2.61 m");
    fireEvent.click(screen.getByText("Calibrate"));
    await screen.findByTestId("review");
    expect(posted(s, "/calibrate")).toEqual([{
      still: "a".repeat(32), polygon: POLY, camera_model: "simple_radial",
      reference: { a: [0.2, 0.7], b: [0.4, 0.75], metres: 2.61 }, meta: { clicks: expect.any(Array), score: 0.8, model: "sam2.1" } }]);
    expect(screen.getByTestId("result-confidence").textContent).toBe("HIGH");
    expect(screen.getAllByTestId("bay-car")).toHaveLength(1); // the bays drawn over the still
    expect(screen.getAllByTestId("bay-free")).toHaveLength(1);
    expect(screen.queryByTestId("unverified-ok")).toBeNull(); // anchored: no extra confirmation
    const accept = screen.getByText("Accept and activate") as HTMLButtonElement;
    expect(accept.disabled).toBe(false);
    fireEvent.click(accept);
    await screen.findByTestId("result-active");
    expect(posted(s, "/activate")).toEqual([{ version: 1, accept_unverified: false }]);
    expect(screen.getByTestId("setup-message").textContent).toContain("nothing was restarted");
    expect(within(screen.getByTestId("version-1")).getByText("active")).toBeTruthy();
  });

  it("without a known distance: LOW, and accepting needs the extra yes", async () => {
    const s = serve({ confidence: "LOW" });
    page();
    fireEvent.click(await screen.findByText("Take a still"));
    await screen.findByTestId("setup-svg");
    fireEvent.click(screen.getByTestId("reference-skip"));
    fireEvent.click(screen.getByText("Next: parking surface"));
    click(500, 400);
    fireEvent.click(screen.getByText("Find the surface"));
    await screen.findByTestId("segment-result");
    fireEvent.click(screen.getByText("Next: calibrate"));
    expect(screen.getByTestId("inputs-summary").textContent).toContain("No known distance");
    fireEvent.click(screen.getByText("Calibrate"));
    await screen.findByTestId("review");
    expect((posted(s, "/calibrate")[0] as { reference?: unknown }).reference).toBeUndefined();
    expect(screen.getByTestId("result-confidence").textContent).toBe("LOW");
    expect(screen.getByTestId("result-warnings").textContent).toContain("one prior");
    const accept = screen.getByText("Accept and activate") as HTMLButtonElement;
    expect(accept.disabled).toBe(true);
    fireEvent.click(screen.getByTestId("unverified-ok"));
    expect(accept.disabled).toBe(false);
    fireEvent.click(accept);
    await screen.findByTestId("result-active");
    expect(posted(s, "/activate")).toEqual([{ version: 1, accept_unverified: true }]);
  });

  it("a draft can be discarded instead", async () => {
    const s = serve({ confidence: "HIGH" });
    page();
    await giveReference();
    fireEvent.click(screen.getByText("Next: parking surface"));
    click(500, 400);
    fireEvent.click(screen.getByText("Find the surface"));
    await screen.findByTestId("segment-result");
    fireEvent.click(screen.getByText("Next: calibrate"));
    fireEvent.click(screen.getByText("Calibrate"));
    await screen.findByTestId("review");
    fireEvent.click(screen.getByText("Discard draft"));
    await waitFor(() => expect(screen.getByTestId("setup-message").textContent).toContain("Draft 1 discarded"));
    expect(s.calls.some((c) => c.method === "DELETE" && c.url === "/api/cameras/front/setup/draft?version=1")).toBe(true);
    expect(screen.queryByTestId("review")).toBeNull();
    expect(screen.getByTestId("versions-empty")).toBeTruthy();
  });

  it("the calibrations list: a HIGH draft activates at once, an unverified one asks first; rollback; discard", async () => {
    const s = serve({ versions: [summary(4, "draft", "LOW"), summary(3, "draft", "LOW"), summary(2, "draft", "HIGH"), summary(1, "superseded", "HIGH")] });
    page();
    await screen.findByTestId("versions-table");
    fireEvent.click(screen.getByTestId("version-discard-4")); // a draft the Admin does not want
    await waitFor(() => expect(s.calls.some((c) => c.method === "DELETE" && c.url === "/api/cameras/front/setup/draft?version=4")).toBe(true));
    await waitFor(() => expect(screen.queryByTestId("version-4")).toBeNull());
    fireEvent.click(screen.getByTestId("version-activate-3")); // not anchored: it asks, nothing is sent
    expect(screen.getByTestId("version-confirm-3").textContent).toContain("±25 %");
    expect(posted(s, "/activate")).toEqual([]);
    fireEvent.click(screen.getByText("Cancel"));
    expect(screen.queryByTestId("version-confirm-3")).toBeNull();
    fireEvent.click(screen.getByTestId("version-activate-2")); // anchored: straight away
    await waitFor(() => expect(posted(s, "/activate")).toEqual([{ version: 2, accept_unverified: false }]));
    await waitFor(() => expect(within(screen.getByTestId("version-2")).getByText("active")).toBeTruthy());
    fireEvent.click(screen.getByTestId("version-activate-1")); // rollback to the earlier anchored one
    await waitFor(() => expect(posted(s, "/activate")[1]).toEqual({ version: 1, accept_unverified: false }));
    fireEvent.click(screen.getByTestId("version-activate-3"));
    fireEvent.click(screen.getByTestId("version-confirm-yes-3")); // confirmed: sent as accepted unverified
    await waitFor(() => expect(posted(s, "/activate")[2]).toEqual({ version: 3, accept_unverified: true }));
    await waitFor(() => expect(within(screen.getByTestId("version-3")).getByText("active")).toBeTruthy());
    expect(screen.queryByTestId("version-discard-3")).toBeNull(); // an active version can not be discarded from the page
  });
});

const point = (i: number, avg: number, max: number) => ({ t: 1792650000 + i * 600, samples: 10, occupied_avg: avg, occupied_max: max, occupied_min: 0, capacity_avg: 4, capacity_max: 4, vehicles_avg: avg, ver: 1 });
const event = (id: number, kind: string, ver: number | null, detail: Record<string, unknown>, t: number): GeometryEvent => ({ id, t, kind, ver, detail });

describe("history, events and the drift guard on the setup page", () => {
  it("draws the measured occupancy and lets the range change", async () => {
    const s = serve({ history: { since: 1792650000, until: 1792660000, bucket_s: 600, versions: [1, 2],
                                 points: [point(0, 1, 2), point(1, 2.5, 4), point(2, 0.5, 1)] } });
    page();
    await screen.findByTestId("history-chart");
    expect(screen.getByTestId("history-capacity").getAttribute("points")?.split(" ")).toHaveLength(3);
    expect(screen.getByTestId("history-peak").getAttribute("points")?.split(" ")).toHaveLength(3);
    expect(screen.getAllByTestId("history-dot")).toHaveLength(3); // a visible dot per sample: a single early sample is not a blank chart
    expect(screen.getByTestId("history-legend").textContent).toContain("calibrations v1, v2");
    expect(s.calls.some((c) => c.url === "/api/cameras/front/geometry/history?hours=24&points=96")).toBe(true);
    fireEvent.click(screen.getByText("7 days"));
    await waitFor(() => expect(s.calls.some((c) => c.url === "/api/cameras/front/geometry/history?hours=168&points=96")).toBe(true));
  });

  it("says so when there is nothing to draw yet", async () => {
    serve();
    page();
    expect((await screen.findByTestId("history-empty")).textContent).toContain("No samples yet");
    expect((await screen.findByTestId("events-empty")).textContent).toBe("Nothing yet.");
    expect(screen.queryByTestId("stale-banner")).toBeNull();
  });

  it("lists the calibration's events and warns when the camera moved and nothing was accepted since", async () => {
    serve({ versions: [summary(1, "stale", "HIGH")],
            events: [event(4, "stale", 1, { reason: "drift: the background moved 31 px (1.6 % of the width), confirmed on 3 checks in a row" }, 1792650300),
                     event(3, "drift_suspected", 1, { shift_px: 30.5 }, 1792650100), event(2, "activated", 1, { previous: null, rollback: false }, 1792650000),
                     event(1, "draft", 1, {}, 1792649990)] });
    page();
    const banner = await screen.findByTestId("stale-banner");
    expect(banner.textContent).toContain("The camera moved.");
    expect(banner.textContent).toContain("v1 marked stale: drift: the background moved 31 px");
    expect(banner.textContent).toContain("the dashboard works as before");
    expect(screen.getByTestId("event-stale").textContent).toContain("v1 marked stale");
    expect(screen.getByTestId("event-drift_suspected").textContent).toContain("may have moved (31 px)");
    expect(screen.getByTestId("event-activated").textContent).toContain("v1 accepted");
    expect(screen.getByTestId("event-draft").textContent).toContain("Calibration v1 made");
    expect(screen.getAllByRole("listitem").length).toBeGreaterThanOrEqual(4);
  });

  it("no banner while a calibration is active, however old the stale event", async () => {
    serve({ versions: [summary(2, "active", "HIGH"), summary(1, "stale", "HIGH")],
            events: [event(5, "activated", 2, { previous: null, rollback: false }, 1792650500), event(4, "stale", 1, { reason: "drift" }, 1792650300)] });
    page();
    await screen.findByTestId("events-list");
    expect(screen.queryByTestId("stale-banner")).toBeNull();
  });

  it("shows which calibrations carry a drift reference", async () => {
    serve({ versions: [summary(2, "active", "HIGH", { drift_guard: true }), summary(1, "superseded", "HIGH", { drift_guard: false })] });
    page();
    await screen.findByTestId("versions-table");
    expect(screen.getByTestId("version-guard-2").textContent).toBe("✓");
    expect(screen.getByTestId("version-guard-1").textContent).toBe("–");
  });
});

