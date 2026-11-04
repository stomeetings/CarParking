import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SettingsDrawer } from "./SettingsDrawer";

/** The grouped Settings drawer: five named cards, the everyday two unfolded, folded bodies stay mounted, changes badge their group. */
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const VALUES = {
  enabled: true, camera: "front", stream: "main", custom_url: "", model_variant: "auto", detector: "coco", bay_model: "auto", inference: "1280",
  conf: 0.3, hold: 30, exit_hold: 60, exit_frames: 3, occlusion: 0.5, hidden_max: 300, hidden_ground: 0.8, ground_share: 0.5, vehicle_iou: 0.3,
  bay_share: 0.5, pin_reach: 0.2, spillage: false, obstacle_filter: false, cut_top: 0, cut_bottom: 0, cut_left: 0, cut_right: 0, roi: "",
  classes: [], extra: [], slot_mode: "fixed", match_rule: "ground", match_iou: 0.3, learn: false, learn_hold: 30, absent_free: false, save_every: 0,
};

function serve(puts: unknown[] = []) {
  vi.stubGlobal("fetch", (url: string, init?: RequestInit) => {
    const ok = (b: unknown) => Promise.resolve(new Response(JSON.stringify(b), { status: 200, headers: { "Content-Type": "application/json" } }));
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)));
      return ok({ values: { ...VALUES, ...puts[puts.length - 1] as object }, version: 2 });
    }
    if (url.startsWith("/api/settings/options")) return ok({ detector: ["coco"], classes: [], extra: [], ranges: {} });
    if (url.startsWith("/api/overflow")) return ok({ enabled: false, auto_dynamic: false, slot_split: false });
    return ok({ values: VALUES, version: 1 });
  });
}
const open = () => render(<SettingsDrawer open onClose={() => {}} onNeedAdmin={() => {}} tokenKey="" codeStale={false} model={undefined} cam="front" />);

describe("grouped settings", () => {
  it("shows the five groups with the everyday ones unfolded", async () => {
    serve();
    open();
    await screen.findByTestId("group-general");
    for (const [id, title] of [["general", "General & Monitoring"], ["ai", "AI & Detection Engine"], ["timing", "Timing & Hysteresis"],
      ["protect", "Advanced Smart Protections"], ["counting", "Dynamic Counting & Thresholds"]]) {
      expect(screen.getByTestId(`group-toggle-${id}`).textContent).toContain(title);
    }
    const state = (id: string) => screen.getByTestId(`group-${id}`).getAttribute("data-open");
    expect([state("general"), state("ai"), state("timing"), state("protect"), state("counting")]).toEqual(["true", "true", "false", "false", "false"]);
    fireEvent.click(screen.getByTestId("group-toggle-protect"));
    expect(state("protect")).toBe("true");
    expect(screen.getByTestId("group-toggle-protect").getAttribute("aria-expanded")).toBe("true");
  });

  it("a folded group stays mounted, and a change in it badges the header and is sent unchanged", async () => {
    const puts: unknown[] = [];
    serve(puts);
    open();
    await screen.findByTestId("spillage-switch"); // inside a folded group
    expect(screen.getByTestId("group-protect").getAttribute("data-open")).toBe("false");
    expect(screen.queryByTestId("group-pending-protect")).toBeNull();
    fireEvent.click(screen.getByTestId("spillage-switch").querySelector("[role=switch]")!);
    expect(screen.getByTestId("group-pending-protect").textContent).toBe("1 changed");
    fireEvent.click(screen.getByText("Apply"));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ spillage: true, version: 1 }); // the payload is what it always was
  });
});
