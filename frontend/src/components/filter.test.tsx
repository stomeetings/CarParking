import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import type { CameraInfo, SlotFilter, Snapshot } from "../types";
import { LiveView } from "./LiveView";
import { Counters } from "./SidePanel";

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
afterEach(cleanup);

const box = (x: number): [number, number][] => [[x, 0.2], [x + 0.1, 0.2], [x + 0.1, 0.4], [x, 0.4]];
const snapshot = {
  shape: [2560, 1440], frame_ts: 1,
  counts: { free: 2, busy: 1, unknown: 1, hidden: 0, phantom: 0, duplicate: 0, slots: 4, learning: 0 },
  slots: [
    { id: 1, name: "1", state: "FREE", hidden: false, phantom: false, duplicate_of: null, poly: box(0.1) },
    { id: 2, name: "2", state: "FREE", hidden: false, phantom: false, duplicate_of: null, poly: box(0.3) },
    { id: 3, name: "3", state: "BUSY", hidden: false, phantom: false, duplicate_of: null, poly: box(0.5) },
    { id: 4, name: "4", state: "UNKNOWN", hidden: true, phantom: false, duplicate_of: null, poly: box(0.7) },
  ],
  detections: [],
} as unknown as Snapshot;
const camera = { id: "front", display: "Front", streams: { hd: "front_hd", sd: "front_sd", native: "front_native" } } as CameraInfo;

function Dashboard() {
  const [filter, setFilter] = useState<SlotFilter>(null);
  return (
    <>
      <Counters snapshot={snapshot} filter={filter} onFilter={setFilter} />
      <LiveView snapshot={snapshot} frameUrl={null} onNeedFrames={() => {}} camera={camera} tokenKey=""
                slotFilter={filter} onClearFilter={() => setFilter(null)} />
    </>
  );
}

const faded = () => screen.getAllByTestId("overlay-slot").filter((g) => g.getAttribute("data-faded") === "1").map((g) => g.getAttribute("data-state"));

describe("counter cards filter the video's slot overlay", () => {
  it("Free / Busy / Unknown show only those slots; the active card again or Total slots shows all", () => {
    render(<Dashboard />);
    expect(faded()).toEqual([]); // all slots by default
    fireEvent.click(screen.getByTestId("filter-free"));
    expect(screen.getByTestId("filter-free").getAttribute("aria-pressed")).toBe("true");
    expect(faded()).toEqual(["BUSY", "UNKNOWN"]);
    expect(screen.getByTestId("filter-chip").textContent).toContain("Showing FREE only · 2 of 4");
    fireEvent.click(screen.getByTestId("filter-busy"));
    expect(faded()).toEqual(["FREE", "FREE", "UNKNOWN"]);
    expect(screen.getByTestId("filter-free").getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(screen.getByTestId("filter-busy")); // the active card again: all
    expect(faded()).toEqual([]);
    expect(screen.queryByTestId("filter-chip")).toBeNull();
    fireEvent.click(screen.getByTestId("filter-unknown"));
    expect(faded()).toEqual(["FREE", "FREE", "BUSY"]);
    fireEvent.click(screen.getByTestId("filter-all")); // Total slots: all
    expect(faded()).toEqual([]);
    fireEvent.click(screen.getByTestId("filter-free"));
    fireEvent.click(screen.getByTestId("filter-chip")); // the chip on the video clears it too
    expect(faded()).toEqual([]);
  });

  it("stays plain widgets without a filter handler", () => {
    render(<Counters snapshot={snapshot} />);
    expect(screen.queryByTestId("filter-free")).toBeNull();
  });
});
