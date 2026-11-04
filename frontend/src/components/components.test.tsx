import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { StateMessage } from "../types";
import { WorkerBadge } from "./Layout";
import { Counters } from "./SidePanel";

afterEach(cleanup);

const state = (connected: boolean, enabled: boolean): StateMessage => ({
  type: "state", connected, heartbeat_age: connected ? 0.8 : 120, settings_version: 3, time: 0,
  snapshot: { enabled, status: "connected" },
});

describe("header badge", () => {
  it("shows the worker connected, idle or down", () => {
    render(<WorkerBadge live={state(true, true)} link="open" />);
    expect(screen.getByTestId("worker-badge").textContent).toBe("Worker Connected");
    cleanup();
    render(<WorkerBadge live={state(true, false)} link="open" />);
    expect(screen.getByTestId("worker-badge").textContent).toBe("Monitoring off");
    cleanup();
    render(<WorkerBadge live={state(false, true)} link="open" />);
    expect(screen.getByTestId("worker-badge").textContent).toBe("Worker Down");
    cleanup();
    render(<WorkerBadge live={null} link="closed" />);
    expect(screen.getByTestId("worker-badge").textContent).toBe("Backend offline");
  });
});

describe("counters", () => {
  it("shows the worker's counts, dashes without a worker", () => {
    render(<Counters snapshot={{ counts: { free: 1, busy: 4, unknown: 0, hidden: 1, phantom: 0, duplicate: 0, slots: 5, learning: 0 } }} />);
    const text = screen.getByTestId("counters").textContent ?? "";
    expect(text).toContain("1Free");
    expect(text).toContain("4Busy");
    expect(text).toContain("5Total slots");
    cleanup();
    render(<Counters snapshot={undefined} />);
    expect(screen.getByTestId("counters").textContent).toContain("–Free");
  });
});

describe("model status", () => {
  it("shows the active model and the auto-selection with its reason", async () => {
    const { ModelStatus } = await import("./SettingsDrawer");
    const hw = { cuda: false, gpu: null, cores: 6, ram_gb: 15.5 };
    render(<ModelStatus model={{ setting: "auto", active: "default", auto: true, auto_choice: "default", swappable: true,
                                 reason: "no CUDA GPU, 6 CPU cores, 15.5 GB RAM", weights: "yolov8s.pt", hardware: hw }} />);
    const text = screen.getByTestId("model-status").textContent ?? "";
    expect(text).toContain("Default / Light");
    expect(text).toContain("Auto-selected");
    expect(text).toContain("Picked at worker start: no CUDA GPU, 6 CPU cores");
    cleanup();
    render(<ModelStatus model={{ setting: "yolo11m", active: "yolo11m", auto: false, auto_choice: "default", swappable: true,
                                 reason: "no CUDA GPU", weights: "yolo11m.pt", hardware: hw }} />);
    const manual = screen.getByTestId("model-status").textContent ?? "";
    expect(manual).toContain("YOLO11m (Medium)");
    expect(manual).not.toContain("Auto-selected");
    expect(manual).toContain("Auto would pick Default / Light");
  });
});

describe("monitoring state", () => {
  it("says why there are no FREE / BUSY states instead of showing bare dashes", async () => {
    const { monitorState } = await import("./Monitor");
    const s = (connected: boolean, enabled: boolean, counts = false): StateMessage => ({
      type: "state", connected, heartbeat_age: connected ? 1 : 95, settings_version: 4, time: 0,
      snapshot: { enabled, status: "idle: monitoring is off",
                  ...(counts ? { counts: { free: 1, busy: 4, unknown: 0, hidden: 0, phantom: 0, duplicate: 0, slots: 5, learning: 0 } } : {}) },
    });
    expect(monitorState(s(true, false), "open").kind).toBe("off");
    expect(monitorState(s(true, true), "open").kind).toBe("starting");
    expect(monitorState(s(true, true, true), "open").kind).toBe("ok");
    expect(monitorState(s(false, true), "open")).toMatchObject({ kind: "down", text: expect.stringContaining("95 s") });
    expect(monitorState(null, "closed").kind).toBe("offline");
  });
  it("offers to turn monitoring on, and the counters explain the dashes", async () => {
    const { MonitorBanner } = await import("./Monitor");
    const off: StateMessage = { type: "state", connected: true, heartbeat_age: 1, settings_version: 4, time: 0, snapshot: { enabled: false } };
    render(<MonitorBanner live={off} link="open" onNeedAdmin={() => {}} cam={null} />);
    expect(screen.getByTestId("monitor-banner").textContent).toContain("Monitoring is off");
    expect(screen.getByRole("button", { name: /Turn monitoring on/ })).toBeTruthy();
    cleanup();
    render(<Counters snapshot={off.snapshot} note="Monitoring off: no FREE / BUSY states" />);
    expect(screen.getByTestId("counters-note").textContent).toBe("Monitoring off: no FREE / BUSY states");
  });
});
