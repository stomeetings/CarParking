import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { withCam } from "../api";
import type { CameraInfo } from "../types";
import { CameraGrid } from "./Grid";

beforeAll(() => {
  // no network in the test DOM: the tiles' video and live state connect to nothing
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
  vi.stubGlobal("WebSocket", Silent);
  vi.stubGlobal("RTCPeerConnection", Silent);
});
afterEach(cleanup);

const cam = (id: string, primary = false): CameraInfo => ({
  id, name: id.toUpperCase(), label: "", display: id.toUpperCase(), primary, sort_order: 0, metadata: {}, rtsp_url: "rtsp://***@h/101", rtsp_sub_url: "",
  streams: { hd: `${id}_hd`, sd: `${id}_sd`, native: `${id}_native` },
  status: { connected: true, heartbeat_age: 1, enabled: false, running: false, slot_mode: "dynamic", counts: null, state: null },
});

describe("CCTV grid", () => {
  it("shows a tile per camera and empty cells; click selects, double-click opens fullscreen", () => {
    const onSelect = vi.fn();
    const onOpen = vi.fn();
    render(<CameraGrid cameras={[cam("front", true), cam("gate"), cam("yard")]} size={2} page={0} onPage={() => {}}
                       selected="front" tokenKey="" onSelect={onSelect} onOpen={onOpen} />);
    expect(screen.getAllByText("Add a camera")).toHaveLength(1); // 2x2 = 4 cells, 3 cameras
    fireEvent.click(screen.getByTestId("tile-gate"));
    expect(onSelect).toHaveBeenCalledWith("gate");
    fireEvent.doubleClick(screen.getByTestId("tile-yard"));
    expect(onOpen).toHaveBeenCalledWith("yard");
    expect(screen.getByTestId("tile-front").className).toContain("ring-primary");
    expect(screen.getByTestId("tile-gate").textContent).toContain("Monitoring off");
  });
  it("pages when there are more cameras than cells", () => {
    const onPage = vi.fn();
    const many = ["a", "b", "c", "d", "e"].map((x) => cam(x));
    render(<CameraGrid cameras={many} size={2} page={1} onPage={onPage} selected={null} tokenKey="" onSelect={() => {}} onOpen={() => {}} />);
    expect(screen.getByTestId("tile-e")).toBeTruthy();
    expect(screen.queryByTestId("tile-a")).toBeNull();
    expect(screen.getByText(/page 2 \/ 2/)).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Previous page"));
    expect(onPage).toHaveBeenCalledWith(0);
  });
  it("overlays counts and health per tile (from the camera list until the tile's live state arrives)", () => {
    const busy = { free: 2, busy: 3, unknown: 1, hidden: 0, phantom: 0, duplicate: 0, slots: 6, learning: 0 };
    const tracking = { ...cam("front"), status: { ...cam("front").status, enabled: true, running: true, counts: busy } };
    const down = { ...cam("gate"), streams: { hd: "gate_hd", sd: null, native: "gate_native" },
                   status: { ...cam("gate").status, connected: false, heartbeat_age: 95 } };
    render(<CameraGrid cameras={[tracking, down]} size={2} page={0} onPage={() => {}} selected={null} tokenKey=""
                       onSelect={() => {}} onOpen={() => {}} />);
    const counts = screen.getByTestId("tile-counts-front").textContent;
    expect(counts).toContain("FREE 2");
    expect(counts).toContain("BUSY 3");
    expect(counts).toContain("? 1");
    expect(counts).toContain("TOTAL 6");
    expect(screen.getByTestId("tile-front").dataset.health).toBe("ok");
    expect(screen.getByTestId("tile-health-gate").textContent).toBe("Worker offline 95 s");
    expect(screen.getByTestId("tile-gate").dataset.health).toBe("down");
    expect(screen.getByTestId("tile-stream-front").textContent).toBe("SD"); // grid tiles: the sub-stream
    expect(screen.getByTestId("tile-stream-gate").textContent).toBe("HD"); // no sub-stream: flagged
  });
  it("keyboard: arrows select, Enter opens, PgDn pages; a camera picked elsewhere brings its page", () => {
    const onSelect = vi.fn();
    const onOpen = vi.fn();
    const onPage = vi.fn();
    const many = ["a", "b", "c", "d", "e"].map((x) => cam(x));
    const props = { cameras: many, size: 2 as const, page: 0, onPage, tokenKey: "", onSelect, onOpen };
    const { rerender } = render(<CameraGrid {...props} selected="a" />);
    const grid = screen.getByLabelText(/Camera grid/);
    fireEvent.keyDown(grid, { key: "ArrowRight" });
    expect(onSelect).toHaveBeenLastCalledWith("b");
    fireEvent.keyDown(grid, { key: "Enter" });
    expect(onOpen).toHaveBeenCalledWith("a");
    fireEvent.keyDown(grid, { key: "PageDown" });
    expect(onPage).toHaveBeenLastCalledWith(1);
    onPage.mockClear();
    rerender(<CameraGrid {...props} selected="e" />); // e.g. chosen in the header
    expect(onPage).toHaveBeenCalledWith(1);
    expect(screen.getByLabelText("Page 2")).toBeTruthy();
    expect(screen.getByTestId("grid-tour").textContent).toContain("Tour");
  });
  it("addresses one camera's endpoints with ?cam=", () => {
    expect(withCam("/api/status", "gate")).toBe("/api/status?cam=gate");
    expect(withCam("/api/events?limit=30", "gate 2")).toBe("/api/events?limit=30&cam=gate%202");
    expect(withCam("/api/status", null)).toBe("/api/status");
  });
});

describe("video fullscreen", () => {
  it("puts only that camera's picture into native fullscreen, and keeps its aspect ratio", async () => {
    const { stageStyle } = await import("../fullscreen");
    const requested: Element[] = [];
    Element.prototype.requestFullscreen = function (this: Element) {
      requested.push(this);
      Object.defineProperty(document, "fullscreenElement", { configurable: true, value: this });
      document.dispatchEvent(new Event("fullscreenchange"));
      return Promise.resolve();
    };
    document.exitFullscreen = () => {
      Object.defineProperty(document, "fullscreenElement", { configurable: true, value: null });
      document.dispatchEvent(new Event("fullscreenchange"));
      return Promise.resolve();
    };
    const onOpen = vi.fn();
    render(<CameraGrid cameras={[{ ...cam("gate"), label: "East Gate", display: "East Gate" }]} size={2} page={0}
                       onPage={() => {}} selected={null} tokenKey="" onSelect={() => {}} onOpen={onOpen} />);
    const tile = screen.getByTestId("tile-gate");
    expect(tile.textContent).toContain("East Gate"); // the display name
    fireEvent.click(screen.getByTestId("video-fullscreen"));
    await Promise.resolve();
    expect(requested).toHaveLength(1);
    expect(requested[0].contains(tile)).toBe(true); // the tile's own box, not the page
    expect(requested[0]).not.toBe(document.documentElement);
    expect(onOpen).not.toHaveBeenCalled(); // not the dashboard's 1x1 view
    expect(screen.getByLabelText("Exit video fullscreen")).toBeTruthy();
    fireEvent.doubleClick(screen.getByTestId("tile-gate")); // double-click in fullscreen leaves it
    await Promise.resolve();
    expect(document.fullscreenElement).toBeNull();
    expect(onOpen).not.toHaveBeenCalled();
    expect(stageStyle(true, 2560, 1440)).toEqual({ aspectRatio: "2560 / 1440", width: "min(100vw, calc(100vh * 2560 / 1440))" });
  });
});
