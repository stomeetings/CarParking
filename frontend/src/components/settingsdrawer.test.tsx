import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ModelInfo, Options, SettingsValues } from "../types";
import { SettingsDrawer } from "./SettingsDrawer";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

const values: SettingsValues = {
  enabled: true, camera: "Front", stream: "main", custom_url: "", detector: "yolov8s", bay_model: "auto", inference: "1280", conf: 0.3,
  classes: ["car"], extra: [], slot_mode: "dynamic", match_rule: "ground", match_iou: 0.15, save_every: 60, hold: 20, exit_hold: 20,
  exit_frames: 0, occlusion: 0.8, hidden_max: 600, learn_hold: 20, learn: true, absent_free: true, cut_top: 0, cut_bottom: 0,
  cut_left: 0, cut_right: 0, roi: "", model_variant: "auto", hidden_ground: 0.8, ground_share: 0.4, vehicle_iou: 0.2, bay_share: 0.5,
  pin_reach: 0.25,
};
const options: Options = {
  stream: ["main", "sub"], detector: ["yolov8s"], bay_model: ["auto"], inference: ["1280"], slot_mode: ["dynamic", "fixed"],
  match_rule: ["ground"], classes: ["car"], extra: [], detector_selected: "yolov8s",
  ranges: { hidden_ground: [0.5, 1], ground_share: [0.1, 0.9], vehicle_iou: [0.05, 0.9], bay_share: [0.1, 0.95], pin_reach: [0, 0.5] },
  tuning_defaults: { hidden_ground: 0.8, ground_share: 0.4, vehicle_iou: 0.2, bay_share: 0.5, pin_reach: 0.25 },
};

function serve(puts: unknown[], overflow: unknown[] = [], stored = { auto_dynamic: false, slot_split: false }) {
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    if (String(url).includes("/api/settings/options")) return json(options);
    if (String(url).includes("/api/overflow")) {
      if (init?.method === "PUT") {
        const { enabled } = JSON.parse(String(init.body));
        overflow.push({ enabled });
        stored = { auto_dynamic: enabled, slot_split: enabled };
      }
      return json({ enabled: stored.auto_dynamic && stored.slot_split, ...stored });
    }
    if (init?.method === "PUT") {
      const body = JSON.parse(String(init.body));
      puts.push(body);
      const { version: _v, ...changed } = body;
      return json({ version: 4, values: { ...values, ...changed } });
    }
    return json({ version: 3, values });
  });
}

const LABEL = "Pin Boundary Grace Margin";
const drawer = () => render(<SettingsDrawer open onClose={() => {}} onNeedAdmin={() => {}} tokenKey="" codeStale={false}
                                            model={undefined as unknown as ModelInfo} cam={null} />);
const slider = () => screen.getByText(LABEL).closest("label")!.querySelector("input[type=range]") as HTMLInputElement;

describe("Pin grace (pin_reach) in the Settings drawer", () => {
  it("shows the field with its auto-grid-only hint, at the stored value", async () => {
    serve([]);
    drawer();
    await waitFor(() => expect(screen.getByText(LABEL)).toBeTruthy());
    const label = screen.getByText(LABEL).closest("label")!;
    expect(label.textContent).toContain("Allowed tolerance margin for Auto-Grid capacity pins.");
    expect(label.title).toContain("still be counted inside the capacity pin");
    expect(slider().value).toBe("0.25");
    expect(slider().min).toBe("0");
    expect(slider().max).toBe("0.5");
    expect(label.textContent).toContain("25 %");
  });

  it("moving it applies only pin_reach (with the version), and Reset to defaults brings it back to 25 %", async () => {
    const puts: unknown[] = [];
    serve(puts);
    drawer();
    await waitFor(() => expect(screen.getByText(LABEL)).toBeTruthy());
    expect(((await screen.findByTestId("tuning-reset")) as HTMLButtonElement).disabled).toBe(true); // all at the defaults
    fireEvent.change(slider(), { target: { value: "0.4" } });
    expect(slider().value).toBe("0.4");
    expect((screen.getByTestId("tuning-reset") as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(screen.getByText("Apply"));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ pin_reach: 0.4, version: 3 });
    await waitFor(() => expect(screen.getByText(/Applied \(version 4\)/)).toBeTruthy());
    fireEvent.click(screen.getByTestId("tuning-reset"));
    expect(slider().value).toBe("0.25");
    expect(screen.getByTestId("tuning-reset").textContent).toContain("25 %");
  });
});

describe("Dynamic Overflow & Sub-Bays switch", () => {
  const toggle = () => screen.getByTestId("overflow-switch").querySelector("[role=switch]") as HTMLButtonElement;

  it("shows the worker's stored state (off) and switching sends one PUT for both options, not a settings Apply", async () => {
    const puts: unknown[] = [];
    const overflow: unknown[] = [];
    serve(puts, overflow);
    drawer();
    await screen.findByTestId("overflow-switch");
    expect(screen.getByText("Dynamic Overflow & Sub-Bays")).toBeTruthy();
    expect(toggle().getAttribute("aria-checked")).toBe("false");
    fireEvent.click(toggle());
    await waitFor(() => expect(toggle().getAttribute("aria-checked")).toBe("true"));
    expect(overflow).toEqual([{ enabled: true }]);
    expect(puts).toEqual([]); // no settings version, no restart
    fireEvent.click(toggle());
    await waitFor(() => expect(toggle().getAttribute("aria-checked")).toBe("false"));
    expect(overflow).toEqual([{ enabled: true }, { enabled: false }]);
  });

  it("only one of the two on reads as off, with a note that switching sets both", async () => {
    serve([], [], { auto_dynamic: false, slot_split: true });
    drawer();
    await screen.findByTestId("overflow-switch");
    expect(toggle().getAttribute("aria-checked")).toBe("false");
    expect(screen.getByText(/only one of the two is on/)).toBeTruthy();
  });
});
