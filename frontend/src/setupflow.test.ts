import { afterEach, describe, expect, it, vi } from "vitest";
import {
  EMPTY_REFERENCE, type DriftStatus, type GeometryEvent, type GeometrySummary, type HistoryPoint, type Reference, calibrateGate, chartLines, clickPoint, confidenceNote,
  defaultParent, driftNote, eventText, needsUnverifiedOk, pixelDistance, reanchorGate, reanchorLines, reanchorParents, referenceBody, referenceProblem,
  staleBanner, waitForJob,
} from "./setupflow";

afterEach(() => vi.unstubAllGlobals());

const GOOD: Reference = { a: [0.2, 0.7], b: [0.4, 0.75], metres: "2.61" };
const ready = { still: true, polygon: [[0.1, 0.5], [0.9, 0.5], [0.9, 0.9]] as [number, number][], reference: GOOD, withoutReference: false, service: "up" as const, busy: false };

describe("clicks on the still", () => {
  it("become shares of the image, clamped; a box without size gives nothing", () => {
    const rect = { left: 100, top: 50, width: 400, height: 200 };
    expect(clickPoint(rect, 300, 150)).toEqual([0.5, 0.5]);
    expect(clickPoint(rect, 0, 9999)).toEqual([0, 1]);
    expect(clickPoint({ left: 0, top: 0, width: 0, height: 0 }, 5, 5)).toBeNull();
    expect(pixelDistance([0, 0], [0.5, 0.5], [1920, 1080])).toBeCloseTo(Math.hypot(960, 540));
  });
});

describe("the known distance, asked for up front", () => {
  it("says what is missing, in order", () => {
    expect(referenceProblem(EMPTY_REFERENCE)).toContain("Click the two end points");
    expect(referenceProblem({ ...GOOD, b: null })).toContain("Click the two end points");
    expect(referenceProblem({ ...GOOD, b: [0.2, 0.7] })).toContain("too close");
    expect(referenceProblem({ ...GOOD, b: [0.205, 0.7] })).toContain("too close");
    expect(referenceProblem({ ...GOOD, metres: "" })).toContain("Type the distance");
    expect(referenceProblem({ ...GOOD, metres: "abc" })).toContain("Type the distance");
    expect(referenceProblem({ ...GOOD, metres: "0.1" })).toContain("between 0.3 and 200");
    expect(referenceProblem({ ...GOOD, metres: "900" })).toContain("between 0.3 and 200");
    expect(referenceProblem(GOOD)).toBeNull();
  });

  it("becomes the API's body only when complete; a decimal comma is fine", () => {
    expect(referenceBody(EMPTY_REFERENCE)).toBeNull();
    expect(referenceBody({ ...GOOD, metres: "x" })).toBeNull();
    expect(referenceBody(GOOD)).toEqual({ a: [0.2, 0.7], b: [0.4, 0.75], metres: 2.61 });
    expect(referenceBody({ ...GOOD, metres: "2,425" })?.metres).toBe(2.425);
  });
});

describe("the calibration gate", () => {
  it("opens only with a still, a surface and a known distance (or the explicit decision to go without)", () => {
    expect(calibrateGate(ready)).toEqual({ ok: true, why: null });
    expect(calibrateGate({ ...ready, still: false }).why).toContain("still");
    const noRef = { ...ready, reference: EMPTY_REFERENCE };
    expect(calibrateGate(noRef)).toMatchObject({ ok: false });
    expect(calibrateGate(noRef).why).toContain("distance you know");
    expect(calibrateGate({ ...noRef, withoutReference: true })).toEqual({ ok: true, why: null }); // the explicit choice
    expect(calibrateGate({ ...ready, polygon: null }).why).toContain("surface");
    expect(calibrateGate({ ...ready, polygon: [[0, 0], [1, 1]] }).ok).toBe(false);
    expect(calibrateGate({ ...ready, service: "down" }).why).toContain("not running");
    expect(calibrateGate({ ...ready, busy: true }).why).toContain("running");
  });

  it("accepting needs an extra yes unless the scale was anchored", () => {
    expect(needsUnverifiedOk("HIGH")).toBe(false);
    for (const c of ["MEDIUM", "LOW", "NONE", null, undefined] as const) expect(needsUnverifiedOk(c)).toBe(true);
    expect(confidenceNote("HIGH", 0.05)).toContain("±5 %");
    expect(confidenceNote("MEDIUM", 0.25)).toContain("±25 %");
    expect(confidenceNote("LOW", 0.35)).toContain("known distance");
    expect(confidenceNote(null, null)).toBe("Not calibrated");
  });
});

describe("waiting for a job", () => {
  const job = (state: string, progress = 0) => ({ id: "j", kind: "calibrate", cam: "c", state, stage: state, progress, result: null, error: null });

  it("polls until it is done, reporting every look", async () => {
    const seq = [job("queued"), job("running", 0.5), job("done", 1)];
    vi.stubGlobal("fetch", () => Promise.resolve(new Response(JSON.stringify(seq.shift()), { headers: { "Content-Type": "application/json" } })));
    const seen: string[] = [];
    const done = await waitForJob("j", (j) => seen.push(j.state), { cancelled: false }, 1);
    expect(done.state).toBe("done");
    expect(seen).toEqual(["queued", "running", "done"]);
  });

  it("returns a failed job as it is, and stops when cancelled", async () => {
    vi.stubGlobal("fetch", () => Promise.resolve(new Response(JSON.stringify({ ...job("error"), error: "boom" }), { headers: { "Content-Type": "application/json" } })));
    expect((await waitForJob("j", () => {}, { cancelled: false }, 1)).error).toBe("boom");
    vi.stubGlobal("fetch", () => Promise.resolve(new Response(JSON.stringify(job("running")), { headers: { "Content-Type": "application/json" } })));
    await expect(waitForJob("j", () => {}, { cancelled: true }, 1)).rejects.toThrow("cancelled");
  });
});

describe("history and the drift guard", () => {
  const ev = (kind: string, detail: Record<string, unknown> = {}, ver: number | null = 3, t = 1792650000): GeometryEvent => ({ id: 1, t, kind, ver, detail });

  it("tells the calibration's story in words", () => {
    expect(eventText(ev("draft"))).toBe("Calibration v3 made (a draft)");
    expect(eventText(ev("activated", { previous: 2, rollback: false }))).toBe("v3 accepted, replacing v2");
    expect(eventText(ev("activated", { previous: null, rollback: false }))).toBe("v3 accepted");
    expect(eventText(ev("activated", { previous: 4, rollback: true }))).toBe("Rolled back to v3 (from v4)");
    expect(eventText(ev("stale", { reason: "drift: the background moved 31 px" }))).toBe("v3 marked stale: drift: the background moved 31 px");
    expect(eventText(ev("discarded", { versions: [5, 6] }, null))).toBe("Drafts v5, v6 discarded");
    expect(eventText(ev("discarded", { versions: [5] }, 5))).toBe("Draft v5 discarded");
    expect(eventText(ev("drift_suspected", { shift_px: 13.6 }))).toBe("The camera may have moved (14 px): checking again");
    expect(eventText(ev("drift_cleared"))).toBe("The camera is back where it was");
    expect(eventText(ev("something_new"))).toBe("something_new");
  });

  it("the drift line says what the guard last saw, and how old that is", () => {
    const at = 1000;
    expect(driftNote(null, true, at)).toEqual({ text: "Drift guard: waiting for the first check", tone: "muted" });
    expect(driftNote(undefined, false, at).text).toContain("no stored reference");
    const d = (over: object): DriftStatus => ({ status: "ok", shift_px: 2.4, at: 980, version: 1, baseline: "calibration", ...over }) as DriftStatus;
    expect(driftNote(d({}), true, at)).toEqual({ text: "Camera position unchanged (2 px) · checked 20 s ago", tone: "ok" });
    expect(driftNote(d({ baseline: "worker" }), false, at).text).toContain("baseline: first frame seen");
    expect(driftNote(d({ status: "drift", shift_px: 14.2 }), true, at)).toEqual({ text: "The camera may have moved (14 px): confirming · checked 20 s ago", tone: "warn" });
    expect(driftNote(d({ status: "inconclusive", shift_px: null, why: "only 8 matches" }), true, at)).toEqual({
      text: "Camera position can't be checked now (only 8 matches) · checked 20 s ago", tone: "muted" });
    expect(driftNote(d({ status: "stale" }), true, at).tone).toBe("warn");
    expect(driftNote(d({ at: 2000 }), true, at).text).toContain("checked 0 s ago"); // a clock a little behind: never negative
  });

  it("the banner appears only when the camera moved and nothing has been accepted since", () => {
    const events = [ev("stale", {}, 3, 300), ev("activated", {}, 3, 200), ev("draft", {}, 3, 100)];
    expect(staleBanner(events, null)?.kind).toBe("stale");
    expect(staleBanner(events, 4)).toBeNull(); // something is active again
    expect(staleBanner([ev("draft", {}, 4, 500), ...events], null)).toBeNull(); // a new draft is newer than the stale: the Admin is on it
    expect(staleBanner([ev("drift_suspected", {}, 3, 400), ...events], null)?.kind).toBe("stale"); // suspicion noise does not hide it
    expect(staleBanner([], null)).toBeNull();
  });

  it("scales the chart lines to the largest value", () => {
    const p = (t: number, avg: number, max: number, cap: number): HistoryPoint => ({ t, samples: 5, occupied_avg: avg, occupied_max: max, occupied_min: 0, capacity_avg: cap, capacity_max: cap, vehicles_avg: avg, ver: 1 });
    const lines = chartLines([p(0, 1, 2, 4), p(50, 2, 4, 4), p(100, 0, 1, 4)], 0, 100, 200, 100);
    expect(lines.top).toBe(4);
    expect(lines.occupiedMax).toBe("0.0,50.0 100.0,0.0 200.0,75.0"); // x by time, y by value (up is more)
    expect(lines.capacity).toBe("0.0,0.0 100.0,0.0 200.0,0.0");
    expect(lines.area.startsWith("0.0,100 ") && lines.area.endsWith(" 200.0,100")).toBe(true);
    expect(chartLines([], 0, 1, 10, 10)).toEqual({ top: 1, occupiedMax: "", area: "", capacity: "" });
  });
});

describe("re-anchoring helpers", () => {
  const v = (ver: number, state: GeometrySummary["state"], activated: number | null, guard: boolean) =>
    ({ ver, state, activated_at: activated, drift_guard: guard } as GeometrySummary);
  it("only calibrations that were in service and carry a reference can be a parent", () => {
    const all = [v(4, "draft", null, true), v(3, "stale", 300, true), v(2, "superseded", 200, false), v(1, "superseded", 100, true)];
    expect(reanchorParents(all).map((x) => x.ver)).toEqual([3, 1]);
    expect(defaultParent(all)?.ver).toBe(3); // the one last in service
    expect(defaultParent([v(1, "draft", null, true)])).toBeNull();
  });
  it("the gate says what is missing, in order", () => {
    expect(reanchorGate({ still: true, parents: 0, busy: false }).why).toContain("full setup");
    expect(reanchorGate({ still: false, parents: 1, busy: false }).why).toContain("fresh still");
    expect(reanchorGate({ still: true, parents: 1, busy: true }).why).toContain("running");
    expect(reanchorGate({ still: true, parents: 1, busy: false })).toEqual({ ok: true, why: null });
  });
  it("the report reads as lines, and the events name the parent", () => {
    const r = { parent: 1, method: "", matches: 400, inliers: 310, rms_px: 0.84, rotation_deg: 4.6, yaw_deg: -4, pitch_before_deg: 32, pitch_after_deg: 30.5,
                roll_before_deg: 0.5, roll_after_deg: 0.9, bays: 12, bays_in_view: 11 };
    expect(reanchorLines(r)).toEqual([
      "The camera turned 4.6° (heading -4.0°, tilt -1.5°, roll +0.4°)",
      "310 of 400 background points agree, residual 0.8 px",
      "11 of 12 bays are still in view",
    ]);
    const ev = (kind: string, detail: Record<string, unknown>) => ({ id: 1, t: 1, kind, ver: 2, detail });
    expect(eventText(ev("reanchored", { parent: 1, rotation_deg: 4.6 }))).toBe("v2 re-anchored from v1 after a 4.6° turn");
    expect(eventText(ev("activated", { previous: null, parent: 1 }))).toBe("v2 accepted (re-anchored from v1)");
    expect(eventText(ev("draft", { parent: 1 }))).toBe("Calibration v2 made (a draft, from v1)");
  });
});
