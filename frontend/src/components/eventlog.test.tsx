import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { eventInfo, eventSlot, isMovement, mergeEvents, perHour } from "../lib";
import type { CameraInfo, ParkingEvent, Snapshot } from "../types";
import { LiveView } from "./LiveView";
import { HistoryPage } from "./Pages";
import { EventTicker } from "./SidePanel";

/** Notices in the event log (encroached / encroach_cleared / rejected) next to entries and exits: categorised everywhere the log is
 * shown (never a false "vehicle out"), and the boxes the non-vehicle filter dropped drawn apart on the video. */
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

const ev = (id: number, event: ParkingEvent["event"], slot_id: number, note = "", session_id: number | null = null): ParkingEvent =>
  ({ id, slot_id, event, time: "2026-10-30T10:%MM:00+05:00".replace("%MM", String(10 + id)), ts: 1000 + id, session_id, note });
const LOG: ParkingEvent[] = [
  ev(5, "encroach_cleared", 2, "cover below the leave share"),
  ev(4, "exit", 1, "", 7),
  ev(3, "rejected", 0, "too_small: too small for the bay outside the bays, 40x70 px, not counted as a vehicle"),
  ev(2, "encroached", 2, "by A1, 42 % of the bay: free but not usable"),
  ev(1, "entry", 1, "", 7),
];

describe("helpers", () => {
  it("tell movements from notices and never turn a notice into a vehicle out", () => {
    expect(LOG.map((e) => isMovement(e))).toEqual([false, true, false, false, true]);
    expect(eventInfo(ev(1, "entry", 1)).text).toBe("vehicle in");
    expect(eventInfo(ev(1, "exit", 1)).text).toBe("vehicle out");
    expect(eventInfo(ev(1, "encroached", 2)).group).toBe("notice");
    expect(eventInfo(ev(1, "rejected", 0)).label).toBe("🚫 IGNORED");
    const future = { event: "something_new" } as unknown as ParkingEvent;
    expect(eventInfo(future)).toMatchObject({ group: "notice", label: "something_new" }); // a kind this build does not know
    expect(isMovement(future)).toBe(false);
  });

  it("name the slot, or say the notice concerns no bay", () => {
    expect(eventSlot(ev(1, "entry", 3))).toBe("Slot 3");
    expect(eventSlot(ev(1, "rejected", 0))).toBe("outside the bays");
  });

  it("the entries / exits per hour of Analytics count movements only", () => {
    const rows = perHour(LOG);
    expect(rows.reduce((a, r) => a + r.entries, 0)).toBe(1);
    expect(rows.reduce((a, r) => a + r.exits, 0)).toBe(1);
  });

  it("a notice merges into the live feed like any event (newest first, no duplicates)", () => {
    const merged = mergeEvents([LOG[1]], [LOG[0], LOG[1]]);
    expect(merged.map((e) => e.id)).toEqual([5, 4]);
  });
});

describe("the live event ticker", () => {
  it("files each kind under its own wording and icon", () => {
    render(<EventTicker events={LOG} />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(5);
    const text = items.map((li) => li.textContent ?? "");
    expect(text[0]).toContain("Slot 2 · encroachment cleared");
    expect(text[1]).toContain("Slot 1 · vehicle out");
    expect(text[2]).toContain("outside the bays · object ignored (not a vehicle)");
    expect(text[2]).toContain("too_small: too small for the bay");
    expect(text[3]).toContain("Slot 2 · bay encroached by a neighbour");
    expect(text[3]).toContain("by A1, 42 % of the bay");
    expect(text[4]).toContain("Slot 1 · vehicle in");
    expect(screen.getAllByTestId("ticker-icon").map((i) => i.getAttribute("data-kind"))).toEqual(["encroach_cleared", "exit", "rejected", "encroached", "entry"]);
    expect(screen.getAllByText(/vehicle out/)).toHaveLength(1); // only the real exit
  });

  it("says so when nothing was logged", () => {
    render(<EventTicker events={[]} />);
    expect(screen.getByText("No events logged yet.")).toBeTruthy();
  });
});

describe("the Event log on the History page", () => {
  const serve = (urls: string[]) =>
    vi.stubGlobal("fetch", (url: string) => {
      urls.push(url);
      const m = url.match(/^\/api\/events\?limit=200&kinds=(\w+)/);
      if (m) {
        const keep = { all: LOG, moves: LOG.filter(isMovement), notices: LOG.filter((e) => !isMovement(e)) }[m[1] as "all" | "moves" | "notices"];
        return Promise.resolve(new Response(JSON.stringify(keep), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      return Promise.resolve(new Response(JSON.stringify({ backend: "sqlite", error: null, cameras: [] }), { status: 200, headers: { "Content-Type": "application/json" } }));
    });
  const rows = () => screen.getAllByTestId("event-row").map((r) => [r.getAttribute("data-kind"), r.textContent ?? ""] as const);

  it("lists every kind by default, each with its own label, a dash for no bay and no session for a notice", async () => {
    const urls: string[] = [];
    serve(urls);
    render(<HistoryPage tokenKey="" cam="front" />);
    fireEvent.click(screen.getByText("Event log"));
    await waitFor(() => expect(screen.getAllByTestId("event-row")).toHaveLength(5));
    expect(urls.some((u) => u.startsWith("/api/events?limit=200&kinds=all&cam=front"))).toBe(true);
    expect(rows().map(([k]) => k)).toEqual(["encroach_cleared", "exit", "rejected", "encroached", "entry"]);
    const byKind = Object.fromEntries(rows());
    expect(byKind.exit).toContain("🅿️ OUT");
    expect(byKind.entry).toContain("🚗 IN");
    expect(byKind.encroached).toContain("⚠️ ENCROACHED");
    expect(byKind.encroach_cleared).toContain("✅ CLEARED");
    expect(byKind.rejected).toContain("🚫 IGNORED");
    expect(byKind.rejected).toContain("–"); // slot 0: no bay
    expect(byKind.rejected).not.toContain("OUT");
    expect(screen.getAllByText("7")).toHaveLength(2); // the sessions of the entry and the exit only
  });

  it("the filter asks the server for movements or notices and the CSV follows it", async () => {
    const urls: string[] = [];
    serve(urls);
    render(<HistoryPage tokenKey="" cam="front" />);
    fireEvent.click(screen.getByText("Event log"));
    await waitFor(() => expect(screen.getAllByTestId("event-row")).toHaveLength(5));
    fireEvent.click(screen.getByText("Entries & exits"));
    await waitFor(() => expect(screen.getAllByTestId("event-row")).toHaveLength(2));
    expect(rows().map(([k]) => k)).toEqual(["exit", "entry"]);
    expect(urls.some((u) => u.startsWith("/api/events?limit=200&kinds=moves"))).toBe(true);
    expect((screen.getByText("CSV").closest("a") as HTMLAnchorElement).href).toContain("/api/events.csv?kinds=moves");
    fireEvent.click(screen.getByText("Spillage & obstacles"));
    await waitFor(() => expect(screen.getAllByTestId("event-row")).toHaveLength(3));
    expect(rows().map(([k]) => k)).toEqual(["encroach_cleared", "rejected", "encroached"]);
    expect((screen.getByText("CSV").closest("a") as HTMLAnchorElement).href).toContain("kinds=notices");
    fireEvent.click(screen.getByText("All"));
    await waitFor(() => expect(screen.getAllByTestId("event-row")).toHaveLength(5));
  });

  it("shows an empty state, not a stale row, when the log was cleared", async () => {
    vi.stubGlobal("fetch", (url: string) => Promise.resolve(new Response(
      url.startsWith("/api/events") ? "[]" : JSON.stringify({ backend: "sqlite", error: null, cameras: [] }), { status: 200, headers: { "Content-Type": "application/json" } })));
    render(<HistoryPage tokenKey="" cam="front" />);
    fireEvent.click(screen.getByText("Event log"));
    await waitFor(() => expect(screen.getByTestId("event-empty").textContent).toContain("No events logged yet."));
    expect(screen.queryAllByTestId("event-row")).toHaveLength(0);
  });

  it("the kind filter is only shown on the Event log tab", async () => {
    serve([]);
    render(<HistoryPage tokenKey="" cam="front" />);
    expect(screen.queryByTestId("event-kinds")).toBeNull();
    fireEvent.click(screen.getByText("Event log"));
    expect(screen.getByTestId("event-kinds")).toBeTruthy();
  });
});

describe("boxes the non-vehicle filter dropped, on the video", () => {
  const camera = { id: "front", display: "Front", streams: { hd: "front_hd", sd: "front_sd", native: "front_native" } } as CameraInfo;
  const snap = {
    shape: [2560, 1440], frame_ts: 1, counts: { free: 1, busy: 1, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 2, learning: 0 },
    slots: [],
    detections: [
      { label: "car", conf: 0.9, src: "", kind: "vehicle", box: [0.1, 0.4, 0.3, 0.6] },
      { label: "car", conf: 0.6, src: "", kind: "vehicle", box: [0.6, 0.5, 0.62, 0.55], rejected: "too_small" },
      { label: "person", conf: 0.8, src: "", kind: "other", box: [0.7, 0.3, 0.72, 0.4] },
    ],
    area: { top: 0, bottom: 0, left: 0, right: 0, roi: null },
  } as unknown as Snapshot;
  const view = (s: Snapshot) => render(<LiveView snapshot={s} frameUrl={null} onNeedFrames={() => {}} camera={camera} tokenKey="" />);
  const boxes = () => screen.getAllByTestId("overlay-box");

  it("are grey and dashed with their reason, a vehicle stays yellow and solid, another object magenta", () => {
    view(snap);
    const [car, dropped, other] = boxes();
    expect(car.getAttribute("stroke")).toBe("#f9b115");
    expect(car.getAttribute("stroke-dasharray")).toBeNull();
    expect(car.getAttribute("data-rejected")).toBeNull();
    expect(dropped.getAttribute("stroke")).toBe("#9da5b1");
    expect(dropped.getAttribute("stroke-dasharray")).toBe("6 4");
    expect(dropped.getAttribute("data-rejected")).toBe("too_small");
    expect(other.getAttribute("stroke")).toBe("#d63384");
    expect(screen.getByText("ignored: too small")).toBeTruthy(); // the tag
    expect(screen.getAllByText(/ignored:/)).toHaveLength(1);
  });

  it("the hover says it is not counted as a vehicle and why", () => {
    view(snap);
    fireEvent.mouseMove(boxes()[1], { clientX: 10, clientY: 10 });
    expect(screen.getByText("ignored: not counted as a vehicle")).toBeTruthy();
    expect(screen.getByText("too small for the bay")).toBeTruthy();
    expect(screen.getByText(/car 0.60 · ignored/)).toBeTruthy();
  });

  it("every box is a direct child of the overlay, as before, and without the filter nothing is marked", () => {
    const { unmount } = view(snap);
    expect(screen.getByTestId("overlay").querySelectorAll(":scope > rect")).toHaveLength(3);
    unmount();
    const plain = { ...snap, detections: snap.detections!.map(({ rejected: _r, ...d }) => d) } as unknown as Snapshot;
    view(plain);
    expect(boxes().every((b) => b.getAttribute("data-rejected") == null)).toBe(true);
    expect(screen.queryByText(/ignored:/)).toBeNull();
    expect(boxes()[1].getAttribute("stroke")).toBe("#f9b115");
  });
});
