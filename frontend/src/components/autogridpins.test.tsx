import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Snapshot } from "../types";
import { AutoGridPins } from "./AutoGridPins";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const snapshot = {
  shape: [2000, 1000], mode: "fixed",
  auto_slots: { version: 3, count: 2, measured: 1, extrapolated: 1, manual: 0, expected: 4, short: 2, sources: { "1": "measured", "2": "extrapolated" } },
  slots: [
    { id: 1, name: "B1", state: "FREE", hidden: false, phantom: false, duplicate_of: null, poly: [[0.1, 0.6], [0.2, 0.6], [0.2, 0.7], [0.1, 0.7]] },
    { id: 2, name: "B2", state: "FREE", hidden: false, phantom: false, duplicate_of: null, poly: [[0.1, 0.4], [0.2, 0.4], [0.2, 0.5], [0.1, 0.5]] },
  ],
} as unknown as Snapshot;

function serve(puts: unknown[], status = 200) {
  vi.stubGlobal("fetch", (_url: string, init?: RequestInit) => {
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)));
      return status === 200 ? json({ ...JSON.parse(String(init.body)), camera: "Front", updated: "x", max_extrapolated: 3 }) : json({ detail: "no" }, status);
    }
    return json({ camera: "Front", updated: null, expected: null, pins: [], max_extrapolated: 3 });
  });
}
const draw = (surface: HTMLElement, a: [number, number], b: [number, number]) => {
  // the surface is 0x0 in jsdom: a fixed box makes clientX/Y map to fractions
  surface.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0, toJSON: () => ({}) });
  const svg = screen.getByTestId("pins-svg");
  fireEvent.mouseDown(svg, { clientX: a[0] * 1000, clientY: a[1] * 500, button: 0 });
  fireEvent.mouseMove(svg, { clientX: b[0] * 1000, clientY: b[1] * 500 });
  fireEvent.mouseUp(svg, { clientX: b[0] * 1000, clientY: b[1] * 500 });
};
const editor = () => render(<AutoGridPins cam="front" snapshot={snapshot} image="data:image/jpeg;base64,X" tokenKey="" onNeedAdmin={() => {}} />);

describe("auto grid pins", () => {
  it("shows the grid as a guide, says what is missing, and saves the count with a drawn pin", async () => {
    const puts: unknown[] = [];
    serve(puts);
    editor();
    await waitFor(() => expect(screen.getByTestId("expected-bays")).toBeTruthy());
    expect(screen.getAllByTestId("guide-bay").map((g) => g.getAttribute("data-source"))).toEqual(["measured", "extrapolated"]);
    expect(screen.getByTestId("pins-short").textContent).toContain("2 of 4 expected bays could not be placed");
    expect(screen.getByText("Save").closest("button")!.disabled).toBe(true); // nothing changed yet
    fireEvent.change(screen.getByTestId("expected-bays"), { target: { value: "4" } });
    draw(screen.getByTestId("pins-surface"), [0.3, 0.2], [0.4, 0.3]);
    expect(screen.getAllByTestId("pin")).toHaveLength(1);
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({ expected: 4, pins: [{ id: 1, name: "P1", poly: [[0.3, 0.2], [0.4, 0.2], [0.4, 0.3], [0.3, 0.3]] }] });
  });

  it("refuses a bad count before the server does, and ignores a tiny drag", async () => {
    const puts: unknown[] = [];
    serve(puts);
    editor();
    await waitFor(() => expect(screen.getByTestId("expected-bays")).toBeTruthy());
    fireEvent.change(screen.getByTestId("expected-bays"), { target: { value: "0" } });
    expect(screen.getByText(/a whole number from 1 to 64/)).toBeTruthy();
    expect(screen.getByText("Save").closest("button")!.disabled).toBe(true);
    draw(screen.getByTestId("pins-surface"), [0.3, 0.2], [0.3005, 0.2005]);
    expect(screen.queryAllByTestId("pin")).toHaveLength(0);
  });

  it("asks for the admin token when the save is refused with 401, and retries after", async () => {
    const puts: unknown[] = [];
    serve(puts, 401);
    const asked: string[] = [];
    render(<AutoGridPins cam="front" snapshot={snapshot} image={null} tokenKey="" onNeedAdmin={(r) => asked.push(r)} />);
    await waitFor(() => expect(screen.getByTestId("expected-bays")).toBeTruthy());
    fireEvent.change(screen.getByTestId("expected-bays"), { target: { value: "4" } });
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(asked).toHaveLength(1));
    expect(asked[0]).toContain("admin token");
  });
});
