import { describe, expect, it } from "vitest";
import { tokens, withToken, wsUrl } from "./api";
import { ago, autoGridText, baySource, bucketLabel, changedSettings, clock, duration, durationBuckets, mergeEvents, moveSelection, outsidePath, pageList, pageOf, perHour, points, slotFlags,
         sourceLabel, stalled, tileHealth } from "./lib";
import type { ParkingEvent, SettingsValues, Slot } from "./types";

const ev = (id: number, event: "entry" | "exit" = "entry", time = "2026-10-20T14:03:27+05:00"): ParkingEvent =>
  ({ id, slot_id: 1, event, time, ts: id, session_id: 1, note: "" });

describe("overlay geometry", () => {
  it("maps fractions onto the frame's pixel canvas", () => {
    expect(points([[0.1, 0.3], [0.3, 0.3], [0.3, 0.5]], 2560, 1440)).toBe("256,432 768,432 768,720");
  });
  it("dims everything outside the hard walls (even-odd: frame minus area)", () => {
    const d = outsidePath({ top: 0.1, bottom: 0, left: 0.12, right: 0.42, roi: null }, 1000, 500);
    expect(d).toBe("M0,0 H1000 V500 H0 Z M120,50 H580 V500 H120 Z");
    expect(outsidePath(undefined, 1000, 500)).toBe("");
  });
  it("clips a polygon area to the walls", () => {
    const d = outsidePath({ top: 0, bottom: 0, left: 0, right: 0.5, roi: [[0.4, 0.1], [0.9, 0.1], [0.9, 1]] }, 100, 100);
    expect(d).toBe("M0,0 H100 V100 H0 Z M40,10 L50,10 L50,100 Z");
  });
  it("labels hidden, phantom and duplicate slots like the Streamlit overlay", () => {
    const s: Slot = { id: 1, name: "1", state: "BUSY", hidden: true, phantom: false, duplicate_of: 11, poly: [] };
    expect(slotFlags(s)).toEqual(["hidden (kept BUSY)", "duplicate of 11"]);
  });
});

describe("event ticker", () => {
  it("merges newest first without duplicates and caps the list", () => {
    const merged = mergeEvents([ev(3), ev(2)], [ev(4), ev(3)], 3);
    expect(merged.map((e) => e.id)).toEqual([4, 3, 2]);
    expect(mergeEvents([ev(1)], [ev(2), ev(3)], 2).map((e) => e.id)).toEqual([3, 2]);
  });
  it("shows the camera's wall-clock time, with the date on other days", () => {
    expect(clock("2026-10-20T14:03:27+05:00", new Date(2026, 9, 20, 18))).toBe("14:03:27");
    expect(clock("2026-10-19T09:15:00+05:00", new Date(2026, 9, 20, 18))).toBe("Oct 19 09:15");
    expect(clock(null)).toBe("");
  });
});

describe("settings", () => {
  const base = { hold: 15, classes: ["car", "truck"], enabled: false, roi: "" } as unknown as SettingsValues;
  it("sends only what changed", () => {
    expect(changedSettings(base, { ...base, hold: 20 })).toEqual({ hold: 20 });
    expect(changedSettings(base, { ...base, classes: ["car"] })).toEqual({ classes: ["car"] });
    expect(changedSettings(base, { ...base })).toEqual({});
  });
});

describe("tokens in URLs", () => {
  it("adds the read token where headers can't be sent (img, a, WebSocket)", () => {
    tokens.setAdmin("");
    tokens.setView("");
    expect(withToken("/api/events.csv")).toBe("/api/events.csv");
    tokens.setView("v t");
    expect(withToken("/ws/stream?frames=1")).toBe("/ws/stream?frames=1&token=v%20t");
    expect(wsUrl("/ws/stream", { protocol: "https:", host: "park.lan" })).toBe("wss://park.lan/ws/stream?token=v%20t");
    tokens.setAdmin("adm"); // the admin token also allows viewing
    expect(withToken("/api/frame.jpg")).toBe("/api/frame.jpg?token=adm");
    tokens.setAdmin("");
    tokens.setView("");
  });
});

describe("analytics", () => {
  it("counts entries and exits per hour of the camera's day", () => {
    const rows = perHour([ev(1, "entry", "2026-10-20T08:10:00+05:00"), ev(2, "exit", "2026-10-20T08:50:00+05:00"),
                          ev(3, "entry", "2026-10-20T17:00:00+05:00")]);
    expect(rows[8]).toEqual({ hour: "08", entries: 1, exits: 1 });
    expect(rows[17].entries).toBe(1);
    expect(rows.reduce((n, r) => n + r.entries + r.exits, 0)).toBe(3);
  });
  it("buckets stay lengths", () => {
    expect(durationBuckets([5, 30, 61, 500]).map((b) => b.sessions)).toEqual([1, 1, 1, 0, 0, 1]);
    expect(duration(75)).toBe("1 h 15 min");
    expect(duration(12.4)).toBe("12 min");
  });
});


describe("CCTV grid logic", () => {
  const w = { connected: true, heartbeatAge: 1, enabled: true, running: true, hasCounts: true };
  it("tile health, most severe first", () => {
    expect(tileHealth(w, "playing")).toEqual({ tone: "ok", label: "Tracking" });
    expect(tileHealth({ ...w, connected: false, heartbeatAge: 95 }, "playing")).toEqual({ tone: "down", label: "Worker offline 95 s" });
    expect(tileHealth({ ...w, connected: false, heartbeatAge: 600 }, "playing").label).toBe("Worker offline 10 min");
    expect(tileHealth(w, "reconnecting")).toEqual({ tone: "stream", label: "Stream reconnecting…" });
    expect(tileHealth({ ...w, enabled: false }, "playing").tone).toBe("idle");
    expect(tileHealth({ ...w, hasCounts: false }, "playing").tone).toBe("starting");
  });
  it("arrow keys move through the visible tiles", () => {
    const ids = ["a", "b", "c", "d"];
    expect(moveSelection(ids, "a", "ArrowRight", 2)).toBe("b");
    expect(moveSelection(ids, "a", "ArrowDown", 2)).toBe("c");
    expect(moveSelection(ids, "d", "ArrowUp", 2)).toBe("b");
    expect(moveSelection(ids, "d", "ArrowRight", 2)).toBe("d"); // the edge: stays
    expect(moveSelection(ids, null, "ArrowRight", 2)).toBe("b");
    expect(moveSelection(ids, "a", "Enter", 2)).toBe("a");
    expect(moveSelection([], "a", "ArrowRight", 2)).toBeNull();
  });
  it("finds a camera's page, detects a stalled video, formats ages", () => {
    expect(pageOf(["a", "b", "c", "d", "e"], "e", 4)).toBe(1);
    expect(pageOf(["a"], "zz", 4)).toBeNull();
    expect(stalled(5.0, 1000, 5.0, 9500)).toBe(true);
    expect(stalled(5.0, 1000, 5.2, 9500)).toBe(false);
    expect(stalled(5.0, 1000, 5.0, 5000)).toBe(false);
    expect(ago(1.23)).toBe("1.2 s");
    expect(ago(45)).toBe("45 s");
    expect(ago(300)).toBe("5 min");
  });
});

describe("central history", () => {
  it("labels occupancy buckets and the data source", () => {
    expect(bucketLabel("2026-10-21T14:05", "5 min")).toBe("14:05");
    expect(bucketLabel("2026-10-21T14:00", "1 h")).toBe("Oct 21 14h");
    expect(bucketLabel("2026-10-21T00:00", "1 day")).toBe("Oct 21");
    expect(sourceLabel("postgres", 1.5)).toBe("PostgreSQL · synced 1.5 s ago");
    expect(sourceLabel("sqlite")).toBe("SQLite (fallback)");
  });
});

describe("slot notes", () => {
  it("say why a slot is shown UNKNOWN", () => {
    const s: Slot = { id: 5, name: "5", state: "UNKNOWN", hidden: false, phantom: false, duplicate_of: null, poly: [],
                      note: "on the vehicle of slot 6" };
    expect(slotFlags(s)).toEqual(["on the vehicle of slot 6"]);
  });
});

describe("pageList", () => {
  it("shows the first, the last and the pages around the current one", () => {
    expect(pageList(1, 1)).toEqual([1]);
    expect(pageList(1, 4)).toEqual([1, 2, null, 4]);
    expect(pageList(5, 9)).toEqual([1, null, 4, 5, 6, null, 9]);
    expect(pageList(9, 9)).toEqual([1, null, 8, 9]);
    expect(pageList(2, 3)).toEqual([1, 2, 3]);
  });
});

describe("auto grid", () => {
  it("says what the grid is made of and what could not be placed", () => {
    expect(autoGridText({ version: 3, count: 11, measured: 11, extrapolated: 0, manual: 0, expected: null })).toBe("auto grid · 11 bays from calibration v3");
    expect(autoGridText({ version: 3, count: 11, measured: 9, extrapolated: 1, manual: 1, expected: 11 }))
      .toBe("auto grid · 11 bays (9 measured · 1 extrapolated · 1 pinned) from calibration v3");
    expect(autoGridText({ version: 3, count: 10, measured: 10, expected: 12, short: 2 }))
      .toBe("auto grid · 10 bays from calibration v3 · 2 of 12 expected bays could not be placed: pin them");
    expect(autoGridText({ version: 1, count: 4 })).toBe("auto grid · 4 bays from calibration v1"); // an older worker's status
    expect(autoGridText({ version: 3, count: 11, measured: 11, dynamic: 2 })).toBe("auto grid · 11 bays + 2 vehicle-driven from calibration v3");
    expect(autoGridText({ version: 3, count: 11, measured: 11, dynamic: 3, sub: 1 })).toBe("auto grid · 11 bays + 3 vehicle-driven (1 sub-bay) from calibration v3");
  });
  it("reads a bay's source from the status (JSON keys are strings), measured when it says nothing", () => {
    const a = { version: 1, count: 3, sources: { "2": "extrapolated", "3": "manual" } } as const;
    expect([1, 2, 3].map((id) => baySource(a, id))).toEqual(["measured", "extrapolated", "manual"]);
    expect(baySource(undefined, 1)).toBeNull();
  });
});
