import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { GeometryEvent, ReanchorReport } from "../setupflow";
import type { CameraInfo } from "../types";
import { CameraSetup } from "./CameraSetup";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const camera: CameraInfo = {
  id: "front", name: "Front", label: "", display: "Front", primary: true, sort_order: 0, metadata: {}, rtsp_url: "", rtsp_sub_url: "",
  streams: { hd: "front_hd", sd: "front_sd", native: "front_native" },
  status: { connected: true, heartbeat_age: 1, enabled: true, running: true, slot_mode: "dynamic", counts: null, state: null },
};
const POLY = [[0.1, 0.5], [0.9, 0.5], [0.9, 0.9], [0.1, 0.9]];
const REPORT: ReanchorReport = {
  parent: 1, method: "feature matching + rotation (Kabsch)", matches: 400, inliers: 310, rms_px: 0.8, rotation_deg: 4.6, yaw_deg: 4.0,
  pitch_before_deg: 32, pitch_after_deg: 34, roll_before_deg: 0.5, roll_after_deg: -0.5, bays: 12, bays_in_view: 12,
};
const summary = (ver: number, state: string, extra = {}) => ({
  ver, state, note: "", created_at: 1792650000 + ver, activated_at: state === "draft" ? null : 1792650100 + ver, confidence: "HIGH", h: 6, scale_uncertainty: 0.05,
  reference_used: true, lens: "simple_radial", area_m2: 40, capacity: 12, bays_with_cars: 2, warnings: [], drift_guard: true, reanchor: null, ...extra,
});

interface Server {
  calls: { method: string; url: string; body: unknown }[];
  versions: ReturnType<typeof summary>[];
  events: GeometryEvent[];
  fail: string | null; // the re-anchoring job fails with this
  confidence: "HIGH" | "MEDIUM";
}

function serve(over: Partial<Server> = {}): Server {
  const s: Server = { calls: [], versions: [summary(1, "stale")], fail: null, confidence: "HIGH",
                       events: [{ id: 1, t: 1792650200, kind: "stale", ver: 1, detail: { reason: "the background moved 30 px" } }], ...over };
  let job: unknown = null;
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    s.calls.push({ method, url, body });
    if (url === "/api/setup/status") return json({ enabled: true, service: "down", loaded: null, pins: null });  // the setup service is stopped
    if (url === "/api/cameras/front/setup/geometry") return json({ active: s.versions.find((v) => v.state === "active") ?? null, versions: s.versions });
    if (url.startsWith("/api/cameras/front/geometry/events")) return json({ events: s.events });
    if (url.startsWith("/api/cameras/front/geometry/history")) return json({ since: 0, until: 1, bucket_s: 60, points: [], versions: [] });
    if (url === "/api/cameras/front/setup/still") return json({ id: "b".repeat(32), ts: 1792650300, source: "camera", width: 1920, height: 1080, image: "data:image/jpeg;base64,NEW" });
    if (url === "/api/cameras/front/setup/reanchor") {
      if (s.fail) return json((job = { id: "r", kind: "reanchor", cam: "front", state: "error", stage: "failed", progress: 1, result: null, error: s.fail }));
      const ver = Math.max(...s.versions.map((v) => v.ver)) + 1;
      s.versions = [summary(ver, "draft", { reanchor: REPORT }), ...s.versions];
      return json(job = { id: "r", kind: "reanchor", cam: "front", state: "done", stage: "done", progress: 1, error: null,
                    result: { version: ver, state: "draft", parent: body.from_version, parent_state: "stale", confidence: s.confidence, h: 6, pitch_deg: 34,
                              area_m2: 40, capacity: 12, bays_with_cars: 2, reference_used: true, scale_uncertainty: s.confidence === "HIGH" ? 0.05 : 0.25,
                              scale_sources: {}, reanchor: REPORT, warnings: ["re-anchored after a camera move"] } });
    }
    if (url.startsWith("/api/setup/jobs/")) return json(job);
    if (/\/setup\/geometry\/\d+$/.test(url)) {
      return json({ ...s.versions[0], record: { surface: { polygon: POLY },
                    capacity: { slots: [{ id: 1, poly: [[0.2, 0.6], [0.3, 0.6], [0.3, 0.7], [0.2, 0.7]], car: true }, { id: 2, poly: [[0.4, 0.6], [0.5, 0.6], [0.5, 0.7], [0.4, 0.7]], car: false }] },
                    calibration: { f: 1500, k1: 0, pitch: 0.6 } } });
    }
    if (url === "/api/cameras/front/setup/activate") {
      s.versions = s.versions.map((v) => (v.ver === body.version ? { ...v, state: "active" } : v));
      s.events = [{ id: 9, t: 1792650400, kind: "activated", ver: body.version, detail: { previous: null, parent: 1 } }, ...s.events];
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

const page = () => render(<CameraSetup cam="front" cameras={[camera]} onCam={() => {}} tokenKey="" onNeedAdmin={() => {}} />);
const posted = (s: Server, suffix: string) => s.calls.filter((c) => c.method === "POST" && c.url.endsWith(suffix)).map((c) => c.body);

describe("re-anchoring a calibration after a camera move", () => {
  it("the stale banner offers it; a fresh still, the report, the bays on the new still, accept: no setup service needed", async () => {
    const s = serve();
    page();
    fireEvent.click(await screen.findByTestId("stale-reanchor"));
    expect(screen.getByTestId("reanchor-parent").textContent).toContain("v1");
    expect((screen.getByTestId("reanchor-run").closest("button") as HTMLButtonElement).disabled).toBe(true); // a fresh still first
    expect(screen.getByTestId("reanchor-gate").textContent).toContain("fresh still");

    fireEvent.click(screen.getByTestId("reanchor-still"));
    await screen.findByTestId("reanchor-surface");
    fireEvent.click(screen.getByTestId("reanchor-run"));
    await screen.findByTestId("reanchor-review");
    expect(posted(s, "/setup/reanchor")).toEqual([{ still: "b".repeat(32), from_version: 1 }]);
    expect(screen.getByTestId("reanchor-report").textContent).toContain("turned 4.6°");
    expect(screen.getByTestId("reanchor-report").textContent).toContain("310 of 400");
    expect(screen.getByTestId("reanchor-report").textContent).toContain("12 of 12 bays");
    expect(screen.getAllByTestId("reanchor-bay-car")).toHaveLength(1);
    expect(screen.getAllByTestId("reanchor-bay-free")).toHaveLength(1);
    expect(s.calls.some((c) => c.url.includes("/setup/segment") || c.url.includes("/setup/calibrate"))).toBe(false); // no SAM, no calibrate job
    expect(s.versions[0].state).toBe("draft"); // nothing is used until accepted
    expect(screen.getByTestId("version-parent-2").textContent).toContain("from v1");

    fireEvent.click(screen.getByTestId("reanchor-accept"));
    await waitFor(() => expect(screen.getByTestId("reanchor-message").textContent).toContain("Version 2 is now this camera's active calibration"));
    expect(posted(s, "/setup/activate")).toEqual([{ version: 2, accept_unverified: false }]);
    expect(screen.queryByTestId("stale-banner")).toBeNull(); // the banner is gone once a calibration is active
  });

  it("a failed re-anchoring says why and offers the full setup; the draft list is untouched", async () => {
    const s = serve({ fail: "the stored background and this picture do not match: do the full setup" });
    page();
    fireEvent.click(await screen.findByTestId("stale-reanchor"));
    fireEvent.click(screen.getByTestId("reanchor-still"));
    await screen.findByTestId("reanchor-surface");
    fireEvent.click(screen.getByTestId("reanchor-run"));
    await waitFor(() => expect(screen.getByTestId("reanchor-message").textContent).toContain("do the full setup"));
    expect(screen.queryByTestId("reanchor-review")).toBeNull();
    expect(s.versions).toHaveLength(1);
    fireEvent.click(screen.getByTestId("reanchor-full"));
    expect(screen.queryByTestId("reanchor")).toBeNull();
    expect(screen.getByTestId("step-1").getAttribute("aria-current")).toBe("step");
  });

  it("the review can be discarded", async () => {
    const s = serve();
    page();
    fireEvent.click(await screen.findByTestId("stale-reanchor"));
    fireEvent.click(screen.getByTestId("reanchor-still"));
    await screen.findByTestId("reanchor-surface");
    fireEvent.click(screen.getByTestId("reanchor-run"));
    fireEvent.click(await screen.findByTestId("reanchor-discard"));
    await waitFor(() => expect(screen.queryByTestId("reanchor-review")).toBeNull());
    expect(s.versions.map((v) => v.ver)).toEqual([1]);
  });

  it("an unanchored scale needs the extra yes before accepting", async () => {
    serve({ confidence: "MEDIUM" });
    page();
    fireEvent.click(await screen.findByTestId("stale-reanchor"));
    fireEvent.click(screen.getByTestId("reanchor-still"));
    await screen.findByTestId("reanchor-surface");
    fireEvent.click(screen.getByTestId("reanchor-run"));
    await screen.findByTestId("reanchor-review");
    const accept = screen.getByTestId("reanchor-accept").closest("button") as HTMLButtonElement;
    expect(accept.disabled).toBe(true);
    fireEvent.click(screen.getByTestId("reanchor-unverified"));
    expect(accept.disabled).toBe(false);
  });

  it("per version in the table; a version without a stored reference cannot be re-anchored", async () => {
    const s = serve({ versions: [summary(2, "active"), summary(1, "superseded", { drift_guard: false })], events: [] });
    page();
    const old = (await screen.findByTestId("version-reanchor-1")).closest("button") as HTMLButtonElement;
    expect(old.disabled).toBe(true);
    expect(old.title).toContain("no stored background");
    expect(screen.queryByTestId("stale-banner")).toBeNull();
    fireEvent.click(screen.getByTestId("version-reanchor-2"));
    expect(screen.getByTestId("reanchor-parent").textContent).toContain("v2");
    fireEvent.click(screen.getByTestId("reanchor-still"));
    await screen.findByTestId("reanchor-surface");
    fireEvent.click(screen.getByTestId("reanchor-run"));
    await screen.findByTestId("reanchor-review");
    expect(posted(s, "/setup/reanchor")).toEqual([{ still: "b".repeat(32), from_version: 2 }]);
  });

  it("with no stored reference anywhere it says to do the full setup", async () => {
    serve({ versions: [summary(1, "stale", { drift_guard: false })] });
    page();
    fireEvent.click(await screen.findByTestId("stale-reanchor"));
    expect(screen.getByTestId("reanchor-none").textContent).toContain("full setup");
    expect((screen.getByTestId("reanchor-still").closest("button") as HTMLButtonElement).disabled).toBe(true);
  });
});
