import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SampleSession } from "./SampleSession";
import type { SampleResult } from "./SampleResult";

/** The Sample test page: upload, give it a scale (pins or GeoCalib + a distance), outline the surface, calibrate (a job), review and fix the bays, run the
 * stack and read the result. The server is mocked; the clicks are real events on the picture. */
beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0, toJSON: () => ({}) });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const sq = (x: number, y = 0.5): [number, number][] => [[x, y], [x + 0.1, y], [x + 0.1, y + 0.1], [x, y + 0.1]];
const SLOTS = [1, 2, 3].map((i) => ({ id: i, name: `B${i}`, poly: sq(0.1 + 0.2 * i), lift: 0.01, source: "measured" }));
const SESSION = { id: "s1", width: 2000, height: 1000, image: "data:image/jpeg;base64,AAAA", layers: { spillage: false, obstacle_filter: false, blocked_objects: false },
                  calibration: null, polygon: null, slots: [], job: null, expires_in: 1800, profile_cam: "camera-1" };
const CAL = { method: "rectangle", confidence: "HIGH", h: 6.02, f: 1498, hfov_deg: 65.2, area_m2: 140.5, warnings: ["the cars give a spread of 25 % in height"],
              pins_fit: { reprojection_px: 0.4 }, bays: 3 };
const RESULT: SampleResult = {
  shape: [2000, 1000],
  counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 3, learning: 0, encroached: 0, available: 1, blocked_objects: 1 },
  slots: [{ id: 1, name: "B1", state: "BUSY", poly: sq(0.3) }, { id: 2, name: "B2", state: "FREE", poly: sq(0.5), avail: "BLOCKED", display: "BLOCKED", blocked_by: ["potted plant"] },
          { id: 3, name: "B3", state: "FREE", poly: sq(0.7) }] as unknown as SampleResult["slots"],
  detections: [], area: { top: 0, bottom: 0, left: 0, right: 0, roi: null }, vehicles: 1, others: 1, events: [], layers: { spillage: false, obstacle_filter: false, blocked_objects: true, ground: false },
  frames: 12, step_s: 1, seconds: 1.2, tune: { model: "default", conf: 0.2, iou: 0.45, footprint: 0.5, encroach: 0.3 }, detector: "This camera's own detector", detected: 30, kind: "session", slot_count: 3, source: "session", name: null, image: "data:image/jpeg;base64,AAAA",
};

const DEFAULTS = { model: "default", conf: 0.2, iou: 0.45, footprint: 0.5, encroach: 0.3 };
const MODELS = [{ id: "default", label: "This camera's own detector", note: "the profile camera's weights", kind: "vehicle", ready: true },
                { id: "yolov8x.pt", label: "COCO · yolov8x.pt", note: "stock COCO vehicles", kind: "vehicle", ready: true },
                { id: "models/parking_yolo.pt", label: "PKLot bay model · parking_yolo.pt", note: "trained on PKLot: it finds busy / free BAYS", kind: "bay", ready: true }];

const DIRECT_RESULT = { ...RESULT, kind: "direct", detector: "PKLot bay model · parking_yolo.pt", detected: 3, direct: { conf: 0.25, iou: 0.45 }, tune: undefined,
                        counts: { ...RESULT.counts, free: 1, busy: 2, slots: 3, available: undefined, blocked_objects: undefined } };

// a zone's own grid (ids zone * 100 + n) and the merge the server answers after zone n was calibrated (zone 2's B1 lies on A2)
const ZONE_BAYS = (z: number) => [1, 2].map((n) => ({ id: z * 100 + n, name: `${String.fromCharCode(64 + z)}${n}`, poly: sq(0.1 + 0.4 * (z - 1) + 0.1 * n) }));
const ZONE_MERGE = (upTo: number) => ({
  zones: Array.from({ length: upTo }, (_, i) => ({ zone: i + 1, name: `Zone ${String.fromCharCode(65 + i)}`, polygon: sq(0.1), calibration: { ...CAL, bays: 2 }, slots: ZONE_BAYS(i + 1) })),
  zone_slots: Array.from({ length: upTo }, (_, i) => ZONE_BAYS(i + 1)).flat().filter((b) => b.id !== 201),
  zone_dropped: upTo > 1 ? [{ id: 201, name: "B1", same_as: "A2" }] : [],
});

interface Net { calls: { method: string; url: string; body?: Record<string, any> }[] }
function serve(opts: { jobs?: unknown[]; fail?: [number, string]; failRun?: [number, string]; down?: boolean; images?: string[] } = {}): Net {
  const net: Net = { calls: [] };
  const queue = [...(opts.jobs ?? [])];
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    net.calls.push({ method, url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    if (url === "/api/sample-sessions/service") return json({ service: opts.down ? "down" : "up", loaded: null, command: "docker compose --profile setup up -d setup" });
    if (url === "/api/sample-sessions/models") return json({ defaults: DEFAULTS, models: MODELS });
    if (url === "/api/sample-sessions/images") return json({ folder: "data/samples", images: opts.images ?? [] });
    if (url === "/api/sample-sessions" && method === "POST") return opts.fail ? json({ detail: opts.fail[1] }, opts.fail[0]) : json(SESSION);
    if (/\/job$/.test(url)) return json(queue.shift() ?? { id: "j", state: "done", stage: "done", progress: 1, result: { ...CAL, slots: SLOTS }, error: null });
    if (/\/calibrate$/.test(url) || /\/surface$/.test(url)) {
      const b = JSON.parse(String(init!.body));
      if (b.zone != null) queue.unshift(/\/surface$/.test(url)
        ? { id: "j", state: "done", stage: "done", progress: 1, result: { polygon: sq(b.zone === 1 ? 0.12 : 0.52, 0.55), zone: b.zone, score: 0.9, area_frac: 0.01 }, error: null }
        : { id: "j", state: "done", stage: "done", progress: 1, result: { ...CAL, zone: b.zone, slots: ZONE_BAYS(b.zone), ...ZONE_MERGE(b.zone) }, error: null });
    }
    if (/\/calibrate$/.test(url) || /\/surface$/.test(url)) return json({ id: "j", state: "running", stage: "detecting vehicles", progress: 0.1, result: null, error: null });
    if (/\/s1\/slots$/.test(url) && method === "PUT") return json({ slots: JSON.parse(String(init!.body)).slots });
    if (/\/zones\/\d+$/.test(url) && method === "DELETE") return json(ZONE_MERGE(1));
    if (/\/zones\/1\/extend$/.test(url)) {
      const n = JSON.parse(String(init!.body)).count as number;
      const more = [...ZONE_BAYS(1), ...Array.from({ length: n }, (_, k) => ({ id: 103 + k, name: `A${3 + k}`, poly: sq(0.4 + 0.1 * k) }))];
      return json({ ...ZONE_MERGE(1), zones: [{ ...ZONE_MERGE(1).zones[0], slots: more }], zone_slots: more, added: n, note: null });
    }
    if (/\/zones\/1\/slots$/.test(url) && method === "PUT") {
      const b = JSON.parse(String(init!.body));
      return json({ ...ZONE_MERGE(1), zones: [{ ...ZONE_MERGE(1).zones[0], slots: b.slots }], zone_slots: b.slots });
    }
    if (/\/direct$/.test(url)) return json(DIRECT_RESULT);
    if (/\/run$/.test(url)) return opts.failRun ? json({ detail: opts.failRun[1] }, opts.failRun[0]) : json(RESULT);
    if (url.startsWith("/api/sample-sessions/") && method === "DELETE") return json({ discarded: "s1" });
    return json({});
  });
  return net;
}
const show = (onNeedAdmin = vi.fn()) => {
  render(<SampleSession cam="camera-1" tokenKey="k" onNeedAdmin={onNeedAdmin} />);
  return onNeedAdmin;
};
const upload = async () => {
  fireEvent.change(screen.getByTestId("sess-file"), { target: { files: [new File(["x"], "lot.jpg", { type: "image/jpeg" })] } });
  await screen.findByTestId("sess-canvas");
};
const clickAt = (x: number, y: number) => fireEvent.click(screen.getByTestId("sess-canvas"), { clientX: x * 1000, clientY: y * 500 });
const posts = (net: Net, suffix: string) => net.calls.find((c) => c.method === "POST" && c.url.endsWith(suffix))!;

// a scale (two ground points and a distance) and a drawn zone outline: what a zone needs before Calibrate
const zoneReady = () => {
  clickAt(0.2, 0.8);
  clickAt(0.4, 0.8);
  fireEvent.change(screen.getByTestId("sess-ref-metres"), { target: { value: "5" } });
  fireEvent.click(screen.getByTestId("zone-draw"));
  [[0.1, 0.9], [0.4, 0.9], [0.4, 0.4], [0.1, 0.4]].forEach(([a, b]) => clickAt(a, b));
};

describe("starting", () => {
  it("has no Whole picture option: a picture is laid out by zones only, with the first zone ready to draw", async () => {
    serve();
    show();
    await upload();
    expect(screen.queryByTestId("sess-layout")).toBeNull();
    expect(screen.queryByText(/Whole picture/)).toBeNull();
    expect(screen.getAllByTestId("zone-row")).toHaveLength(1);
    expect(screen.getByTestId("zone-reason").textContent).toContain("draw Zone A's outline");
    expect(screen.getByTestId("sess-dynamic")).toBeTruthy(); // the Dynamic slots switch is on the one layout there is
  });

  it("opens straight into the GeoCalib + SAM workflow: no camera-slots tab, GeoCalib is the scale method, the service is checked", async () => {
    serve();
    show();
    expect(screen.queryByText("This camera's slots")).toBeNull();
    expect(screen.queryByTestId("sample-mode")).toBeNull();
    expect(screen.getByTestId("sample-session")).toBeTruthy();
    expect((await screen.findByTestId("sess-service")).textContent).toContain("Setup service is running");
    await upload();
    expect(screen.getByText(/Place points \(0\/2\)/)).toBeTruthy(); // GeoCalib's two points, not pins
  });

  it("says how to start the setup service when it is down, and checks again on request", async () => {
    const net = serve({ down: true });
    show();
    await waitFor(() => expect(screen.getByTestId("sess-service").textContent).toContain("docker compose --profile setup up -d setup"));
    fireEvent.click(screen.getByTestId("sess-service-check"));
    await waitFor(() => expect(net.calls.filter((c) => c.url === "/api/sample-sessions/service")).toHaveLength(2));
  });

  it("starts a session from a picture in the samples folder", async () => {
    const net = serve({ images: ["a.jpg", "b.png"] });
    show();
    const pick = await screen.findByTestId("sess-sample");
    expect(within(pick).getAllByRole("option").map((o) => o.textContent)).toEqual(["…or pick from data/samples/", "a.jpg", "b.png"]);
    fireEvent.change(pick, { target: { value: "b.png" } });
    await screen.findByTestId("sess-canvas");
    expect(posts(net, "/api/sample-sessions").body).toEqual({ sample: "b.png", profile_cam: "camera-1" });
  });

  it("uploads the picture into a session named after the profile camera, and refuses a file that is not a JPEG or PNG", async () => {
    const net = serve();
    show();
    fireEvent.change(screen.getByTestId("sess-file"), { target: { files: [new File(["x"], "a.gif", { type: "image/gif" })] } });
    expect((await screen.findByTestId("sess-error")).textContent).toContain("JPEG or PNG");
    await upload();
    expect(posts(net, "/api/sample-sessions").body).toMatchObject({ profile_cam: "camera-1" });
    expect(String(posts(net, "/api/sample-sessions").body!.image)).toMatch(/^data:/);
  });

  it("shows the server's refusal and asks for the admin token on a 401", async () => {
    serve({ fail: [422, "image: could not be decoded"] });
    show();
    fireEvent.change(screen.getByTestId("sess-file"), { target: { files: [new File(["x"], "a.jpg", { type: "image/jpeg" })] } });
    expect((await screen.findByTestId("sess-error")).textContent).toContain("could not be decoded");
    cleanup();
    serve({ fail: [401, "unauthorized"] });
    const need = show();
    fireEvent.change(screen.getByTestId("sess-file"), { target: { files: [new File(["x"], "a.jpg", { type: "image/jpeg" })] } });
    await waitFor(() => expect(need).toHaveBeenCalled());
  });
});

describe("the detection settings", () => {
  it("is a collapsed card showing the values in use, defaulting to confidence 0.20, IoU 0.45, blocked 50 %", async () => {
    serve();
    show();
    await waitFor(() => expect(screen.getByTestId("sess-settings-summary").textContent).toContain("conf 0.20"));
    expect(screen.getByTestId("sess-settings-summary").textContent).toContain("IoU 0.45 · blocked 50 %");
    expect(screen.queryByTestId("sess-settings")).toBeNull();
    fireEvent.click(screen.getByTestId("sess-settings-toggle"));
    expect(screen.getByTestId("sess-settings")).toBeTruthy();
    expect(within(screen.getByTestId("sess-model")).getAllByRole("option").map((o) => o.textContent)).toEqual(
      ["This camera's own detector", "COCO · yolov8x.pt", "PKLot bay model · parking_yolo.pt"]);
  });

  it("takes values from a slider or the number box, keeps them in range, and says what the model is", async () => {
    serve();
    show();
    fireEvent.click(screen.getByTestId("sess-settings-toggle"));
    await waitFor(() => expect(within(screen.getByTestId("sess-model")).getAllByRole("option")).toHaveLength(3)); // the list has arrived
    fireEvent.change(screen.getByTestId("sess-conf"), { target: { value: "0.12" } });
    expect((screen.getByTestId("sess-conf-n") as HTMLInputElement).value).toBe("0.12");
    fireEvent.change(screen.getByTestId("sess-conf-n"), { target: { value: "0.01" } }); // below the range: held at 0.10
    expect((screen.getByTestId("sess-conf-n") as HTMLInputElement).value).toBe("0.1");
    fireEvent.change(screen.getByTestId("sess-iou-n"), { target: { value: "0.9" } });
    expect((screen.getByTestId("sess-iou-n") as HTMLInputElement).value).toBe("0.7");
    fireEvent.change(screen.getByTestId("sess-footprint-n"), { target: { value: "65" } }); // shown as a percentage
    expect(screen.getByTestId("sess-settings-summary").textContent).toContain("blocked 65 %");
    fireEvent.change(screen.getByTestId("sess-model"), { target: { value: "models/parking_yolo.pt" } });
    expect(screen.getByTestId("sess-model-note").textContent).toContain("finds busy / free BAYS");
    fireEvent.click(screen.getByTestId("sess-settings-reset"));
    expect(screen.getByTestId("sess-settings-summary").textContent).toContain("conf 0.20 · IoU 0.45 · blocked 50 %");
  });

  it("sends the settings with the zone's SAM, Calibrate and Run, and shows what the run used and found", async () => {
    const net = serve();
    show();
    await upload();
    fireEvent.click(screen.getByTestId("sess-settings-toggle"));
    fireEvent.change(screen.getByTestId("sess-conf-n"), { target: { value: "0.15" } });
    fireEvent.change(screen.getByTestId("sess-iou-n"), { target: { value: "0.3" } });
    fireEvent.change(screen.getByTestId("sess-footprint-n"), { target: { value: "60" } });
    fireEvent.change(screen.getByTestId("sess-encroach-n"), { target: { value: "40" } });
    fireEvent.change(screen.getByTestId("sess-model"), { target: { value: "yolov8x.pt" } });
    const want = { model: "yolov8x.pt", conf: 0.15, iou: 0.3, footprint: 0.6, encroach: 0.4 };
    zoneReady();
    fireEvent.click(screen.getByTestId("zone-sam-mode"));
    clickAt(0.2, 0.6);
    fireEvent.click(screen.getByTestId("zone-sam"));
    await screen.findByTestId("zone-surface", {}, { timeout: 4000 });
    expect(posts(net, "/surface").body!.tune).toEqual(want);
    fireEvent.click(screen.getByTestId("zone-calibrate"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay")).toHaveLength(2), { timeout: 4000 });
    expect(posts(net, "/calibrate").body!.tune).toEqual(want);
    fireEvent.click(screen.getByTestId("zone-run"));
    await screen.findByTestId("sample-overlay", {}, { timeout: 4000 });
    expect(posts(net, "/run").body!.tune).toEqual(want);
    expect(screen.getByTestId("sample-summary").textContent).toContain("30 found, 29 outside the parking surface or dropped by the filter");
    expect(screen.getByTestId("sample-summary").textContent).toContain("at conf 0.2, IoU 0.45, blocked 50 %, encroach 30 %");
  });
});

describe("the direct bay model", () => {
  it("is offered once a picture is loaded, with no calibration, and sends the bay model and its own settings", async () => {
    const net = serve();
    show();
    expect(screen.queryByTestId("direct-run")).toBeNull(); // nothing to run on yet
    await upload();
    fireEvent.click(screen.getByTestId("fold-direct-toggle"));
    const pick = await screen.findByTestId("direct-model");
    expect(within(pick).getAllByRole("option").map((o) => o.textContent)).toEqual(["PKLot bay model · parking_yolo.pt"]); // bay models only
    fireEvent.change(screen.getByTestId("direct-conf"), { target: { value: "0.1" } });
    fireEvent.change(screen.getByTestId("direct-iou"), { target: { value: "0.5" } });
    expect((screen.getByTestId("direct-run") as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(screen.getByTestId("direct-run"));
    await screen.findByTestId("direct-result");
    expect(posts(net, "/direct").url).toBe("/api/sample-sessions/s1/direct");
    expect(posts(net, "/direct").body).toEqual({ model: "models/parking_yolo.pt", conf: 0.1, iou: 0.5 });
    expect(net.calls.some((c) => c.url.endsWith("/calibrate") || c.url.endsWith("/run"))).toBe(false); // the calibrated pipeline was not touched
    const text = within(screen.getByTestId("direct-result")).getByTestId("sample-summary").textContent ?? "";
    expect(text).toContain("PKLot bay model · parking_yolo.pt at conf 0.25, IoU 0.45");
    expect(text).toContain("no calibration");
  });

  it("shows only the output of the last test run, direct or zones, never both", async () => {
    const net = serve();
    show();
    await upload();
    zoneReady();
    fireEvent.click(screen.getByTestId("zone-calibrate"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay")).toHaveLength(2), { timeout: 4000 });
    fireEvent.click(screen.getByTestId("zone-run"));
    await waitFor(() => expect(screen.getAllByTestId("sample-summary")).toHaveLength(1), { timeout: 4000 });
    fireEvent.click(screen.getByTestId("fold-direct-toggle"));
    fireEvent.click(screen.getByTestId("direct-run"));
    await screen.findByTestId("direct-result");
    expect(screen.getAllByTestId("sample-summary")).toHaveLength(1); // only the direct result: the two outputs are never on screen together
    expect(within(screen.getByTestId("direct-result")).getByTestId("sample-summary").textContent).toContain("no calibration");
    fireEvent.click(screen.getByTestId("zone-run")); // the zone run again: now only that one
    await waitFor(() => expect(screen.queryByTestId("direct-result")).toBeNull(), { timeout: 4000 });
    expect(screen.getAllByTestId("sample-summary")).toHaveLength(1);
    expect(posts(net, "/run").body).toEqual({ layers: { spillage: false, obstacle_filter: false, blocked_objects: false }, tune: DEFAULTS, layout: "zones", skip: [], dynamic: false }); // all three protections start off
  });

  it("shows the server's reason when it cannot run, and a Start over clears it", async () => {
    vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (url === "/api/sample-sessions" && method === "POST") return json(SESSION);
      if (url.endsWith("/direct")) return json({ detail: "no PKLot bay model (parking_yolo*.pt) in the project's models folder" }, 409);
      if (url === "/api/sample-sessions/models") return json({ defaults: DEFAULTS, models: MODELS });
      if (url === "/api/sample-sessions/service") return json({ service: "up", loaded: null, command: "" });
      if (url === "/api/sample-sessions/images") return json({ folder: "data/samples", images: [] });
      return json({});
    });
    show();
    await upload();
    fireEvent.click(await screen.findByTestId("direct-run"));
    expect((await screen.findByTestId("sess-error")).textContent).toContain("no PKLot bay model");
    expect(screen.queryByTestId("direct-result")).toBeNull();
  });
});

describe("drawing fixed slots for the direct bay model", () => {
  const svg = () => screen.getByTestId("draw-svg");
  const drag = (x1: number, y1: number, x2: number, y2: number) => {
    fireEvent.mouseDown(svg(), { clientX: x1 * 1000, clientY: y1 * 500, button: 0 });
    fireEvent.mouseMove(svg(), { clientX: x2 * 1000, clientY: y2 * 500 });
    fireEvent.mouseUp(svg(), { clientX: x2 * 1000, clientY: y2 * 500 });
  };
  const open = async () => {
    const net = serve();
    show();
    await upload();
    fireEvent.click(screen.getByTestId("fold-direct-toggle"));
    fireEvent.click(await screen.findByTestId("direct-draw-toggle"));
    return net;
  };

  it("is off by default (the plain direct run sends no slots) and the run needs a slot once it is on", async () => {
    const net = await open();
    expect(screen.getByTestId("draw-canvas")).toBeTruthy();
    expect((screen.getByTestId("direct-run") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByTestId("direct-draw-toggle"));
    expect(screen.queryByTestId("draw-canvas")).toBeNull();
    fireEvent.click(screen.getByTestId("direct-run"));
    await screen.findByTestId("direct-result");
    expect(posts(net, "/direct").body).toEqual({ model: "models/parking_yolo.pt", conf: 0.25, iou: 0.45 });
  });

  it("draws a rectangle and a polygon, labels them Slot 1, Slot 2 and sends them with the run", async () => {
    const net = await open();
    drag(0.1, 0.2, 0.2, 0.4); // the rectangle tool is the default
    expect(screen.getAllByTestId("draw-slot")).toHaveLength(1);
    fireEvent.click(screen.getByText("Polygon"));
    [[0.5, 0.2], [0.6, 0.2], [0.6, 0.4], [0.5, 0.4]].forEach(([x, y]) => fireEvent.mouseDown(svg(), { clientX: x * 1000, clientY: y * 500, button: 0 }));
    fireEvent.keyDown(screen.getByTestId("draw-viewport"), { key: "Enter" });
    expect(screen.getAllByTestId("draw-slot").map((g) => g.querySelector("text")!.textContent)).toEqual(["Slot 1", "Slot 2"]);
    expect(screen.getByTestId("draw-count").textContent).toContain("2 slot(s)");
    fireEvent.click(screen.getByTestId("direct-run"));
    await screen.findByTestId("direct-result");
    const body = posts(net, "/direct").body!;
    expect(body.slots).toHaveLength(2);
    expect(body.slots[0]).toMatchObject({ id: 1, name: "Slot 1", poly: [[0.1, 0.2], [0.2, 0.2], [0.2, 0.4], [0.1, 0.4]] });
    expect(body.slots[1].poly).toHaveLength(4);
  });

  it("selects, moves, reshapes and deletes a slot", async () => {
    await open();
    drag(0.1, 0.2, 0.2, 0.4);
    fireEvent.click(screen.getByText("Select / edit"));
    const poly = () => screen.getByTestId("draw-slot").querySelector("polygon")!;
    fireEvent.mouseDown(poly(), { clientX: 150, clientY: 150, button: 0 });
    fireEvent.mouseMove(svg(), { clientX: 250, clientY: 150 }); // 100 px right = 0.1 of the width
    fireEvent.mouseUp(svg());
    expect(poly().getAttribute("points")!.startsWith("400,200")).toBe(true); // x 0.1 -> 0.2 of 2000 px, y 0.2 of 1000 px
    fireEvent.mouseDown(screen.getAllByTestId("draw-vertex")[0], { clientX: 200, clientY: 100, button: 0 });
    fireEvent.mouseMove(svg(), { clientX: 100, clientY: 50 });
    fireEvent.mouseUp(svg());
    expect(poly().getAttribute("points")!.startsWith("200,100")).toBe(true);
    fireEvent.click(screen.getByTestId("draw-delete"));
    expect(screen.queryAllByTestId("draw-slot")).toHaveLength(0);
  });

  it("zooms with the buttons and the wheel and fits again", async () => {
    await open();
    expect(screen.getByTestId("draw-zoom").textContent).toBe("100 %");
    expect((screen.getByTestId("draw-zoom-out") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByTestId("draw-zoom-in"));
    expect(screen.getByTestId("draw-zoom").textContent).toBe("150 %");
    fireEvent.wheel(screen.getByTestId("draw-viewport"), { deltaY: -100, clientX: 500, clientY: 250 });
    expect(Number.parseInt(screen.getByTestId("draw-zoom").textContent!, 10)).toBeGreaterThan(150);
    fireEvent.click(screen.getByTestId("draw-zoom-reset"));
    expect(screen.getByTestId("draw-zoom").textContent).toBe("100 %");
  });

  it("pans with the Pan tool without drawing a slot", async () => {
    await open();
    fireEvent.click(screen.getByTestId("draw-zoom-in"));
    fireEvent.click(within(screen.getByTestId("draw-canvas")).getByText("Pan"));
    const before = screen.getByTestId("draw-inner").getAttribute("style");
    fireEvent.mouseDown(svg(), { clientX: 500, clientY: 250, button: 0 });
    fireEvent.mouseMove(svg(), { clientX: 400, clientY: 200 });
    fireEvent.mouseUp(svg());
    expect(screen.getByTestId("draw-inner").getAttribute("style")).not.toBe(before);
    expect(screen.queryAllByTestId("draw-slot")).toHaveLength(0);
  });

  it("clears all, loads a starter preset, and saves and reloads the slots of a picture size", async () => {
    localStorage.clear();
    await open();
    fireEvent.click(screen.getByTestId("draw-preset"));
    expect(screen.getAllByTestId("draw-slot")).toHaveLength(12);
    drag(0.9, 0.05, 0.95, 0.1);
    expect(screen.getAllByTestId("draw-slot")).toHaveLength(13);
    fireEvent.click(screen.getByTestId("draw-save"));
    expect(screen.getByTestId("draw-note").textContent).toContain("Saved 13 slot(s)");
    fireEvent.click(screen.getByTestId("draw-clear"));
    expect(screen.queryAllByTestId("draw-slot")).toHaveLength(0);
    fireEvent.click(screen.getByTestId("draw-preset"));
    expect(screen.getAllByTestId("draw-slot")).toHaveLength(13); // the saved ones, not the starter
    expect(screen.getByTestId("draw-note").textContent).toContain("Loaded your saved 13");
  });

  it("offers the vehicle detectors too once slots are drawn, sends the chosen model, and goes back to bay models only when switched off", async () => {
    const net = await open();
    const pick = screen.getByTestId("direct-model");
    expect(within(pick).getAllByRole("option").map((o) => o.textContent)).toEqual(["This camera's own detector", "COCO · yolov8x.pt", "PKLot bay model · parking_yolo.pt"]);
    expect((pick as HTMLSelectElement).value).toBe("models/parking_yolo.pt"); // the bay model stays the default
    fireEvent.change(pick, { target: { value: "default" } });
    drag(0.1, 0.2, 0.2, 0.4);
    fireEvent.click(screen.getByTestId("direct-run"));
    await screen.findByTestId("direct-result");
    expect(posts(net, "/direct").body!.model).toBe("default");
    fireEvent.click(screen.getByTestId("direct-draw-toggle"));
    expect(within(screen.getByTestId("direct-model")).getAllByRole("option")).toHaveLength(1);
    expect((screen.getByTestId("direct-model") as HTMLSelectElement).value).toBe("models/parking_yolo.pt");
  });

  it("shows the drawn-slot result: busy, free and unknown counts and no encroached flag", async () => {
    vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (url === "/api/sample-sessions" && method === "POST") return json(SESSION);
      if (url.endsWith("/direct")) {
        return json({ ...DIRECT_RESULT, direct: { conf: 0.25, iou: 0.45, drawn: true, by: { "bay model, slot crop": 2, "vehicle detector": 1 } }, detected: 5, slot_count: 3,
                      counts: { free: 1, busy: 1, unknown: 1, hidden: 0, phantom: 0, duplicate: 0, slots: 3, learning: 0 },
                      slots: [{ id: 1, name: "Slot 1", state: "BUSY", poly: sq(0.3) }, { id: 2, name: "Slot 2", state: "FREE", poly: sq(0.5) }, { id: 3, name: "Slot 3", state: "UNKNOWN", poly: sq(0.7) }] });
      }
      if (url === "/api/sample-sessions/models") return json({ defaults: DEFAULTS, models: MODELS });
      if (url === "/api/sample-sessions/service") return json({ service: "up", loaded: null, command: "" });
      if (url === "/api/sample-sessions/images") return json({ folder: "data/samples", images: [] });
      return json({});
    });
    show();
    await upload();
    fireEvent.click(await screen.findByTestId("direct-draw-toggle"));
    drag(0.1, 0.2, 0.2, 0.4);
    fireEvent.click(screen.getByTestId("direct-run"));
    const res = await screen.findByTestId("direct-result");
    expect(within(res).getByTestId("sample-summary").textContent).toContain("3 drawn slot(s): 1 busy, 1 free, 1 unknown; decided by 2 bay model, slot crop, 1 vehicle detector");
    expect(within(res).getAllByTestId("sample-slot-row").map((r) => r.getAttribute("data-display"))).toEqual(["BUSY", "FREE", "UNKNOWN"]);
    expect(within(res).queryByText(/encroached by/)).toBeNull();
  });
});

describe("the two test panels and editing drawn slots", () => {
  const svg = () => screen.getByTestId("draw-svg");
  const drag = (x1: number, y1: number, x2: number, y2: number) => {
    fireEvent.mouseDown(svg(), { clientX: x1 * 1000, clientY: y1 * 500, button: 0 });
    fireEvent.mouseMove(svg(), { clientX: x2 * 1000, clientY: y2 * 500 });
    fireEvent.mouseUp(svg(), { clientX: x2 * 1000, clientY: y2 * 500 });
  };
  const opened = async () => {
    serve();
    show();
    await upload();
    fireEvent.click(screen.getByTestId("fold-direct-toggle"));
    fireEvent.click(await screen.findByTestId("direct-draw-toggle"));
  };

  it("starts with the Direct bay model and the GeoCalib + SAM panels closed, and each opens on its own", async () => {
    serve();
    show();
    await upload();
    for (const id of ["direct", "calibrated"]) {
      expect(screen.getByTestId(`fold-${id}-body`).hasAttribute("hidden")).toBe(true);
      expect(screen.getByTestId(`fold-${id}-toggle`).getAttribute("aria-expanded")).toBe("false");
    }
    fireEvent.click(screen.getByTestId("fold-calibrated-toggle"));
    expect(screen.getByTestId("fold-calibrated-body").hasAttribute("hidden")).toBe(false);
    expect(screen.getByTestId("fold-direct-body").hasAttribute("hidden")).toBe(true); // the other stays closed
    fireEvent.click(screen.getByTestId("fold-calibrated-toggle"));
    expect(screen.getByTestId("fold-calibrated-body").hasAttribute("hidden")).toBe(true);
  });

  it("closing a panel keeps what was drawn in it", async () => {
    await opened();
    drag(0.1, 0.2, 0.2, 0.4);
    fireEvent.click(screen.getByTestId("fold-direct-toggle"));
    fireEvent.click(screen.getByTestId("fold-direct-toggle"));
    expect(screen.getAllByTestId("draw-slot")).toHaveLength(1);
  });

  it("lists the drawn slots with an Edit and a Delete button each", async () => {
    await opened();
    drag(0.1, 0.2, 0.2, 0.4);
    drag(0.5, 0.2, 0.6, 0.4);
    expect(screen.getAllByTestId("draw-row")).toHaveLength(2);
    expect(screen.getAllByTestId("draw-row")[0].textContent).toContain("rectangle");
    fireEvent.click(screen.getAllByTestId("draw-row-edit")[1]); // selects it and switches to the edit tool
    expect(screen.getAllByTestId("draw-vertex")).toHaveLength(4);
    fireEvent.click(screen.getAllByTestId("draw-row-delete")[0]);
    expect(screen.getAllByTestId("draw-row")).toHaveLength(1);
    expect(screen.getByTestId("draw-slot").getAttribute("data-id")).toBe("2");
  });

  it("resizes a rectangle by a corner and keeps it a rectangle, even dragged past the opposite corner", async () => {
    await opened();
    drag(0.1, 0.2, 0.2, 0.4); // corners (0.1,0.2) (0.2,0.2) (0.2,0.4) (0.1,0.4)
    fireEvent.click(screen.getAllByTestId("draw-row-edit")[0]);
    fireEvent.mouseDown(screen.getAllByTestId("draw-vertex")[2], { clientX: 200, clientY: 200, button: 0 }); // the bottom right corner
    fireEvent.mouseMove(svg(), { clientX: 400, clientY: 300 }); // to (0.4, 0.6)
    fireEvent.mouseUp(svg());
    expect(screen.getByTestId("draw-slot").querySelector("polygon")!.getAttribute("points")).toBe("200,200 800,200 800,600 200,600");
    fireEvent.mouseDown(screen.getAllByTestId("draw-vertex")[2], { clientX: 400, clientY: 300, button: 0 });
    fireEvent.mouseMove(svg(), { clientX: 50, clientY: 50 }); // past the opposite corner (0.1, 0.1)
    fireEvent.mouseUp(svg());
    expect(screen.getByTestId("draw-slot").querySelector("polygon")!.getAttribute("points")).toBe("100,100 200,100 200,200 100,200"); // (0.05..0.1, 0.1..0.2): still a rectangle, flipped about the fixed corner
  });
});

describe("zoom and pan on the calibration picture", () => {
  const layer = () => screen.getByTestId("sess-layer").getAttribute("style");

  it("zooms with the buttons and the wheel, fits again, and keeps the marks the same size on screen", async () => {
    serve();
    show();
    await upload();
    clickAt(0.3, 0.8); // a ground point
    const r0 = Number(screen.getByTestId("sess-ref").getAttribute("r"));
    expect(screen.getByTestId("sess-zoom").textContent).toBe("100 %");
    expect((screen.getByTestId("sess-zoom-out") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByTestId("sess-zoom-in"));
    expect(screen.getByTestId("sess-zoom").textContent).toBe("150 %");
    fireEvent.wheel(screen.getByTestId("sess-stage"), { deltaY: -100, clientX: 500, clientY: 250 });
    const zoom = Number.parseInt(screen.getByTestId("sess-zoom").textContent!, 10);
    expect(zoom).toBeGreaterThan(150);
    expect(Number(screen.getByTestId("sess-ref").getAttribute("r"))).toBeCloseTo(r0 / (zoom / 100), 1); // zoomed in: the mark does not grow with the picture
    fireEvent.click(screen.getByTestId("sess-zoom-reset"));
    expect(screen.getByTestId("sess-zoom").textContent).toBe("100 %");
    expect(layer()).toContain("scale(1)");
  });

  it("pans with the Pan tool or the middle button without placing a point, and a plain click still places one", async () => {
    serve();
    show();
    await upload();
    fireEvent.click(screen.getByTestId("sess-zoom-in"));
    const canvas = screen.getByTestId("sess-canvas");
    const before = layer();
    fireEvent.click(screen.getByTitle(/Drag the picture instead of placing points/));
    fireEvent.mouseDown(canvas, { clientX: 500, clientY: 250, button: 0 });
    fireEvent.mouseMove(canvas, { clientX: 400, clientY: 200 });
    fireEvent.mouseUp(canvas);
    fireEvent.click(canvas, { clientX: 400, clientY: 200 });
    expect(layer()).not.toBe(before);
    expect(screen.queryAllByTestId("sess-ref")).toHaveLength(0); // nothing was placed while panning
    fireEvent.click(screen.getByTitle(/Drag the picture instead of placing points/)); // tool off
    const mid = layer();
    fireEvent.mouseDown(canvas, { clientX: 400, clientY: 200, button: 1 });
    fireEvent.mouseMove(canvas, { clientX: 450, clientY: 230 });
    fireEvent.mouseUp(canvas);
    expect(layer()).not.toBe(mid);
    expect(screen.queryAllByTestId("sess-ref")).toHaveLength(0);
    clickAt(0.3, 0.8);
    expect(screen.getAllByTestId("sess-ref")).toHaveLength(1); // a normal click places a point again
  });
});

describe("zones: SAM + GeoCalib per zone", () => {
  const outline = (x: number) => [[x, 0.9], [x + 0.3, 0.9], [x + 0.3, 0.4], [x, 0.4]].forEach(([a, b]) => clickAt(a, b));
  const geoScale = () => {
    clickAt(0.2, 0.8);
    clickAt(0.4, 0.8);
    fireEvent.change(screen.getByTestId("sess-ref-metres"), { target: { value: "5" } });
  };

  it("each zone gets its own SAM and GeoCalib run on its own outline; the bays of all zones are merged once", async () => {
    const net = serve();
    show();
    await upload();
    geoScale();
    fireEvent.click(screen.getByTestId("zone-draw"));
    expect(screen.getByTestId("zone-reason").textContent).toContain("draw Zone A's outline");
    outline(0.1);
    expect(screen.getByTestId("zone-outline")).toBeTruthy();
    fireEvent.click(screen.getByTestId("zone-sam-mode"));
    clickAt(0.2, 0.6);
    fireEvent.click(screen.getByTestId("zone-sam"));
    await screen.findByTestId("zone-surface", {}, { timeout: 4000 });
    const sam = posts(net, "/surface").body!;
    expect(sam.zone).toBe(1);
    expect(sam.bound).toHaveLength(4);
    expect(sam.clicks).toEqual([{ x: 400, y: 600, label: 1 }]);
    fireEvent.click(screen.getByTestId("zone-calibrate"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay")).toHaveLength(2), { timeout: 4000 });
    const cal = posts(net, "/calibrate").body!;
    expect([cal.method, cal.zone, cal.zone_name]).toEqual(["geocalib", 1, "Zone A"]);
    expect(cal.polygon).toEqual(sq(0.12, 0.55)); // SAM's ground inside the zone, not the drawn outline
    expect(screen.getByTestId("zone-calibration").textContent).toContain("HIGH");

    fireEvent.click(screen.getByTestId("zone-add"));
    outline(0.5);
    fireEvent.click(screen.getByTestId("zone-calibrate")); // no SAM: the outline is the ground
    await waitFor(() => expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(3), { timeout: 4000 });
    const second = net.calls.filter((c) => c.url.endsWith("/calibrate"))[1].body!;
    expect([second.zone, second.polygon.length]).toEqual([2, 4]);
    expect(screen.getAllByTestId("zone-bay-row").map((r) => r.textContent?.replace("Edit", ""))).toEqual(["A1 · Zone A", "A2 · Zone A", "B2 · Zone B"]);
    expect(screen.getByTestId("zone-dropped").textContent).toContain("B1 = A2");
    expect(screen.getAllByTestId("zone-layer").map((l) => [l.getAttribute("data-zone"), l.getAttribute("data-active")])).toEqual([["1", "0"], ["2", "1"]]);
    expect(screen.queryAllByTestId("sess-slot")).toHaveLength(0); // the whole-picture grid is not drawn in zones
  });

  it("runs every zone's bays as one list without a removed bay; deleting a zone asks the server", async () => {
    const net = serve();
    show();
    await upload();
    geoScale();
    fireEvent.click(screen.getByTestId("zone-draw"));
    outline(0.1);
    fireEvent.click(screen.getByTestId("zone-calibrate"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(2), { timeout: 4000 });
    fireEvent.click(screen.getAllByTestId("zone-bay-delete")[0]);
    fireEvent.click(screen.getByTestId("zone-run"));
    await screen.findByTestId("sample-summary");
    expect(posts(net, "/run").body).toMatchObject({ layout: "zones", skip: [101], dynamic: false }); // dynamic slots stay off unless switched on
    fireEvent.click(screen.getByTestId("zone-delete"));
    await waitFor(() => expect(net.calls.some((c) => c.method === "DELETE" && c.url.endsWith("/zones/1"))).toBe(true));
  });
});

describe("zones: add next bays and edit them", () => {
  const ready = async (net: Net) => {
    show();
    await upload();
    clickAt(0.2, 0.8);
    clickAt(0.4, 0.8);
    fireEvent.change(screen.getByTestId("sess-ref-metres"), { target: { value: "5" } });
    fireEvent.click(screen.getByTestId("zone-draw"));
    [[0.1, 0.9], [0.4, 0.9], [0.4, 0.4], [0.1, 0.4]].forEach(([a, b]) => clickAt(a, b));
    fireEvent.click(screen.getByTestId("zone-calibrate"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(2), { timeout: 4000 });
    return net;
  };

  it("sends the drawn outline with the calibration and adds the next 1, 2 or N bays", async () => {
    const net = await ready(serve());
    expect(posts(net, "/calibrate").body!.bound).toEqual([[0.1, 0.9], [0.4, 0.9], [0.4, 0.4], [0.1, 0.4]]); // the zone's own cars and row direction
    fireEvent.click(screen.getByTestId("zone-next-2"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(4));
    expect(posts(net, "/zones/1/extend").body).toEqual({ count: 2 });
    expect(screen.getAllByTestId("zone-bay-row").map((r) => r.textContent?.split(" ·")[0])).toEqual(["A1", "A2", "A3", "A4"]);
    fireEvent.change(screen.getByTestId("zone-next-n-value"), { target: { value: "5" } });
    fireEvent.click(screen.getByTestId("zone-next-n"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(7));
  });

  it("a bay can be selected, moved and reshaped; the edit is saved before the next bays and the run", async () => {
    const net = await ready(serve());
    fireEvent.click(screen.getAllByTestId("zone-bay-edit")[0]);
    expect(screen.getAllByTestId("zone-vertex")).toHaveLength(4);
    const bay = screen.getAllByTestId("zone-bay")[0].querySelector("polygon")!;
    fireEvent.mouseDown(bay, { button: 0, clientX: 250, clientY: 275 });
    fireEvent.mouseMove(screen.getByTestId("sess-canvas"), { clientX: 300, clientY: 275 }); // 5 % to the right
    fireEvent.mouseUp(screen.getByTestId("sess-canvas"));
    fireEvent.mouseDown(screen.getAllByTestId("zone-vertex")[0], { button: 0 });
    fireEvent.mouseMove(screen.getByTestId("sess-canvas"), { clientX: 200, clientY: 200 });
    fireEvent.mouseUp(screen.getByTestId("sess-canvas"));
    fireEvent.click(screen.getByTestId("zone-run"));
    await screen.findByTestId("sample-summary");
    const put = net.calls.find((c) => c.method === "PUT" && c.url.endsWith("/zones/1/slots"))!;
    const a1 = put.body!.slots.find((b: { id: number }) => b.id === 101);
    expect(a1.poly[0]).toEqual([0.2, 0.4]); // the dragged corner
    expect(a1.poly[1][0]).toBeCloseTo(0.35, 4); // moved by 5 %
    const order = net.calls.filter((c) => /zones\/1\/slots$|\/run$/.test(c.url)).map((c) => c.url.split("/").pop());
    expect(order).toEqual(["slots", "run"]);
    fireEvent.click(screen.getByTestId("zone-next-1"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(3));
    expect(net.calls.filter((c) => c.method === "PUT" && c.url.endsWith("/zones/1/slots"))).toHaveLength(1); // nothing new to save
  });

  it("Select zone & slot adjustment: arrow keys and buttons move the selected bay or the whole zone, size and turn buttons tune it; saved before the run", async () => {
    const net = await ready(serve());
    expect(screen.getByTestId("zone-fine")).toBeTruthy(); // Edit bays is on after a calibration
    fireEvent.click(screen.getByTestId("zone-draw"));
    expect(screen.queryByTestId("zone-fine")).toBeNull(); // and only there
    fireEvent.click(screen.getAllByTestId("zone-bay-edit")[0]);
    expect(screen.getByTestId("zone-fine").textContent).toContain("Select zone & slot adjustment");
    fireEvent.change(screen.getByTestId("zone-fine-step"), { target: { value: "10" } });
    fireEvent.keyDown(window, { key: "ArrowRight" }); // 10 px of 2000 = 0.005
    fireEvent.keyDown(window, { key: "ArrowUp", shiftKey: true }); // 50 px of 1000 = 0.05
    fireEvent.keyDown(screen.getByTestId("zone-next-n-value"), { key: "ArrowLeft" }); // typing in a box is not nudging
    fireEvent.click(screen.getByTestId("zone-fine-wider"));
    fireEvent.click(screen.getByTestId("zone-fine-cw"));
    fireEvent.click(screen.getByText(/Whole Zone A/)); // now every bay of the zone
    fireEvent.click(screen.getByTestId("zone-fine-down")); // 10 px of 1000 = 0.01
    fireEvent.click(screen.getByTestId("zone-fine-shrink"));
    fireEvent.click(screen.getByTestId("zone-run"));
    await screen.findByTestId("sample-summary");
    const put = net.calls.find((c) => c.method === "PUT" && c.url.endsWith("/zones/1/slots"))!;
    const [a1, a2] = [101, 102].map((id) => put.body!.slots.find((b: { id: number }) => b.id === id));
    // the mock bays are 0.1 x 0.1 squares at y 0.5: A2 only went down 0.01 (whole zone) and shrank 2 % about its centre
    expect(a2.poly[0][1]).toBeGreaterThan(0.505);
    expect(a2.poly[0][1]).toBeLessThan(0.52);
    // A1 went right 0.005 and up 0.05 (selected bay), widened and turned 0.5°, then down 0.01 with the zone and shrank: about (0.205, 0.46)
    expect(a1.poly[0][0]).toBeGreaterThan(0.195);
    expect(a1.poly[0][0]).toBeLessThan(0.22);
    expect(a1.poly[0][1]).toBeGreaterThan(0.44);
    expect(a1.poly[0][1]).toBeLessThan(0.475);
    expect(a1.poly[1][0] - a1.poly[0][0]).toBeGreaterThan(0.1 * 0.97); // widened 2 %, then shrunk 2 %: about as wide as before
    const order = net.calls.filter((c) => /zones\/1\/slots$|\/run$/.test(c.url)).map((c) => c.url.split("/").pop());
    expect(order).toEqual(["slots", "run"]);
  });

  it("splits the selected bay into two (left | right or front / back), the second with the zone's next free id, saved before the run", async () => {
    const net = await ready(serve());
    expect((screen.getByTestId("zone-split-width") as HTMLButtonElement).disabled).toBe(true); // nothing selected
    fireEvent.click(screen.getAllByTestId("zone-bay-edit")[0]);
    fireEvent.click(screen.getByTestId("zone-split-width"));
    expect(screen.getAllByTestId("zone-bay-row").map((r) => r.textContent?.split(" ·")[0])).toEqual(["A1a", "A1b", "A2"].map((n) => n)); // A1 became two, next to each other
    fireEvent.click(screen.getAllByTestId("zone-bay-edit")[2]); // A2
    fireEvent.click(screen.getByTestId("zone-split-length"));
    expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(4);
    fireEvent.click(screen.getByTestId("zone-run"));
    await screen.findByTestId("sample-summary", {}, { timeout: 4000 });
    const put = net.calls.find((c) => c.method === "PUT" && c.url.endsWith("/zones/1/slots"))!;
    const ids = put.body!.slots.map((b: { id: number }) => b.id).sort((x: number, y: number) => x - y);
    expect(ids).toEqual([101, 102, 103, 104]); // the originals 101, 102 and the two new ones, 103 and 104
    const a1a = put.body!.slots.find((b: { id: number }) => b.id === 101);
    const a1b = put.body!.slots.find((b: { id: number }) => b.id === 103);
    expect([a1a.name, a1b.name]).toEqual(["A1a", "A1b"]);
    expect(a1a.poly[1][0]).toBeCloseTo(a1b.poly[0][0], 3); // they share the cut
    expect(a1a.poly[0][0]).toBeCloseTo(0.2, 3); // and together fill the old bay (it spanned x 0.2 to 0.3)
    expect(a1b.poly[1][0]).toBeCloseTo(0.3, 3);
  });

  it("Undo takes back a move, a size change and a split one at a time (also with Ctrl+Z), and the restored bays are saved before the run", async () => {
    const net = await ready(serve());
    const undo = screen.getByTestId("zone-undo-edit") as HTMLButtonElement;
    expect(undo.disabled).toBe(true); // nothing to take back yet
    fireEvent.click(screen.getAllByTestId("zone-bay-edit")[0]);
    fireEvent.change(screen.getByTestId("zone-fine-step"), { target: { value: "10" } });
    fireEvent.click(screen.getByTestId("zone-fine-right")); // 10 px of 2000 = 0.005
    fireEvent.click(screen.getByTestId("zone-fine-wider"));
    fireEvent.click(screen.getByTestId("zone-split-width"));
    expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(3);
    expect(undo.textContent).toContain("(3)");
    fireEvent.click(undo); // the split
    expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(2);
    fireEvent.keyDown(window, { key: "z", ctrlKey: true }); // the wider
    fireEvent.click(undo); // the move
    expect(undo.disabled).toBe(true);
    fireEvent.click(screen.getByTestId("zone-run"));
    await screen.findByTestId("sample-summary");
    const put = net.calls.find((c) => c.method === "PUT" && c.url.endsWith("/zones/1/slots"))!;
    const a1 = put.body!.slots.find((b: { id: number }) => b.id === 101);
    expect(a1.poly[0][0]).toBeCloseTo(0.2, 4); // back where it was
    expect(a1.poly[1][0]).toBeCloseTo(0.3, 4);
    expect(put.body!.slots).toHaveLength(2);
  });

  it("Row / Back / Front buttons move the bay along its own directions", async () => {
    const net = await ready(serve());
    fireEvent.click(screen.getAllByTestId("zone-bay-edit")[0]);
    fireEvent.change(screen.getByTestId("zone-fine-step"), { target: { value: "10" } });
    fireEvent.click(screen.getByTestId("zone-fine-back")); // the mock bay: front edge at y 0.5, back edge above it: back = up, 10 px of 1000 = 0.01
    fireEvent.click(screen.getByTestId("zone-fine-row-right"));
    fireEvent.click(screen.getByTestId("zone-run"));
    await screen.findByTestId("sample-summary");
    const put = net.calls.find((c) => c.method === "PUT" && c.url.endsWith("/zones/1/slots"))!;
    const a1 = put.body!.slots.find((b: { id: number }) => b.id === 101);
    expect(a1.poly[0][0]).toBeCloseTo(0.205, 4);
    expect(Math.abs(a1.poly[0][1] - 0.5)).toBeCloseTo(0.01, 3);
  });
});

describe("the Top-down (aerial) scale method", () => {
  it("is a third scale method that sends the typed distance without a lens and needs no setup service", async () => {
    const net = serve({ down: true });
    show();
    await upload();
    fireEvent.click(screen.getByText("Top-down (aerial)"));
    expect(screen.queryByTestId("sess-lens")).toBeNull(); // GeoCalib's lens choice is not part of it
    zoneReady();
    expect(screen.getByTestId("zone-calibrate").textContent).toContain("top-down");
    fireEvent.click(screen.getByTestId("zone-calibrate"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay")).toHaveLength(2), { timeout: 4000 });
    const body = posts(net, "/calibrate").body!;
    expect([body.method, body.zone, "lens" in body]).toEqual(["topdown", 1, false]);
    expect(body.reference.metres).toBe(5);
  });

  it("GeoCalib stays the default and still sends its lens", async () => {
    const net = serve();
    show();
    await upload();
    expect(screen.getByTestId("sess-lens")).toBeTruthy();
    zoneReady();
    fireEvent.click(screen.getByTestId("zone-calibrate"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay")).toHaveLength(2), { timeout: 4000 });
    expect(posts(net, "/calibrate").body).toMatchObject({ method: "geocalib", lens: "simple_radial", rows: 0 }); // rows are Auto unless chosen: the zone's depth decides
  });
});

describe("dynamic slots and zone rows", () => {
  it("dynamic slots are off by default and sent only when switched on; a zone's rows and bay size are shown", async () => {
    const net = serve();
    show();
    await upload();
    clickAt(0.2, 0.8);
    clickAt(0.4, 0.8);
    fireEvent.change(screen.getByTestId("sess-ref-metres"), { target: { value: "5" } });
    fireEvent.click(screen.getByTestId("zone-draw"));
    [[0.1, 0.9], [0.4, 0.9], [0.4, 0.4], [0.1, 0.4]].forEach(([a, b]) => clickAt(a, b));
    fireEvent.change(screen.getByTestId("zone-rows"), { target: { value: "2" } });
    fireEvent.click(screen.getByTestId("zone-calibrate"));
    await waitFor(() => expect(screen.getAllByTestId("zone-bay-row")).toHaveLength(2), { timeout: 4000 });
    expect(posts(net, "/calibrate").body!.rows).toBe(2);
    const box = screen.getAllByTestId("sess-dynamic")[0] as HTMLInputElement;
    expect(box.checked).toBe(false);
    fireEvent.click(screen.getByTestId("zone-run"));
    await screen.findByTestId("sample-summary");
    expect(posts(net, "/run").body!.dynamic).toBe(false);
    fireEvent.click(box);
    fireEvent.click(screen.getByTestId("zone-run"));
    await waitFor(() => expect(net.calls.filter((c) => c.url.endsWith("/run"))).toHaveLength(2));
    expect(net.calls.filter((c) => c.url.endsWith("/run"))[1].body!.dynamic).toBe(true);
  });
});
