import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { HistoryPage } from "./Pages";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const json = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } }));

describe("history page (central history)", () => {
  it("says where the history comes from and lists the camera's state changes", async () => {
    const urls: string[] = [];
    vi.stubGlobal("fetch", (url: string) => {
      urls.push(url);
      if (url.startsWith("/api/history/sync")) {
        return json({ backend: "postgres", error: null, cameras: [{ id: "front", display: "Front", sqlite: { sessions: 5, events: 9 },
          behind: { sessions: 0, events: 0 }, rows: { sessions: 5, events: 9, samples: 3 }, age_s: 1.2, error: "", source: "postgres" }] });
      }
      if (url.startsWith("/api/camera-events")) {
        return json([{ id: 2, time: "2026-10-21T09:00:05+05:00", ts: 1, kind: "worker_offline", detail: "last heartbeat 20 s ago" },
                     { id: 1, time: "2026-10-21T08:00:00+05:00", ts: 0, kind: "first_seen", detail: "" }]);
      }
      return json([]);
    });
    render(<HistoryPage tokenKey="" cam="front" />);
    await waitFor(() => expect(screen.getByTestId("history-source").textContent).toBe("PostgreSQL · synced 1.2 s ago"));
    fireEvent.click(screen.getByText("Camera status"));
    await waitFor(() => expect(screen.getByTestId("camera-events").textContent).toContain("Worker offline"));
    expect(screen.getByTestId("camera-events").textContent).toContain("last heartbeat 20 s ago");
    expect(urls.some((u) => u.startsWith("/api/camera-events?limit=200&cam=front"))).toBe(true);
    expect(screen.queryByText("CSV")).toBeNull(); // no download for state changes
  });

  it("pages the sessions 20 at a time, parked cars first, with a pager under the table", async () => {
    const urls: string[] = [];
    const row = (id: number, open = false) => ({ id, slot_id: id % 7, in_time: "2026-10-22T08:00:00+05:00", out_time: open ? null : "2026-10-22T09:00:00+05:00",
      in_ts: 1, out_ts: open ? null : 2, duration_min: open ? null : 60, status: open ? "Occupied" : "Available", note: "" });
    vi.stubGlobal("fetch", (url: string) => {
      urls.push(url);
      const m = url.match(/^\/api\/sessions\/page\?page=(\d+)&size=(\d+)/);
      if (m) {
        const page = Number(m[1]);
        const items = Array.from({ length: page < 3 ? 20 : 9 }, (_, i) => row(200 - (page - 1) * 20 - i, page === 1 && i < 5));
        return json({ items, total: 49, open: 5, page, size: 20, pages: 3 });
      }
      return json({ backend: "sqlite", error: null, cameras: [] });
    });
    render(<HistoryPage tokenKey="" cam="front" />);
    await waitFor(() => expect(screen.getByTestId("page-info").textContent).toContain("Showing 1–20 of 49 sessions"));
    expect(screen.getByTestId("page-info").textContent).toContain("5 parked now");
    expect(screen.getAllByText("parked")).toHaveLength(5);
    expect(urls.some((u) => u.startsWith("/api/sessions/page?page=1&size=20&cam=front"))).toBe(true);
    expect((screen.getByLabelText("Previous page") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByLabelText("Next page"));
    await waitFor(() => expect(screen.getByTestId("page-info").textContent).toContain("Showing 21–40 of 49"));
    expect(screen.getByRole("button", { current: "page" }).textContent).toBe("2");
    fireEvent.click(screen.getByRole("button", { name: "3" }));
    await waitFor(() => expect(screen.getByTestId("page-info").textContent).toContain("Showing 41–49 of 49"));
    expect((screen.getByLabelText("Next page") as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("history page: This grid / All versions, and Refresh", () => {
  const row = (id: number, scope?: string) => ({ id, slot_id: id + 50, in_time: "2026-10-22T08:00:00+05:00", out_time: null, in_ts: 1, out_ts: null,
    duration_min: null, status: "Occupied", note: "", ...(scope ? { scope } : {}) });

  it("asks for all=1 and shows a Scope column only in All versions, going back to page 1", async () => {
    const urls: string[] = [];
    vi.stubGlobal("fetch", (url: string) => {
      urls.push(url);
      if (url.startsWith("/api/sessions/page")) {
        const all = url.includes("&all=1");
        return json({ items: all ? [row(5, "Front | yolov8s | auto v10"), row(4, "Front | yolov8s | auto v3")] : [row(5)],
                      total: all ? 2 : 1, open: all ? 2 : 1, page: 1, size: 20, pages: 1 });
      }
      return json({ backend: "sqlite", error: null, cameras: [] });
    });
    render(<HistoryPage tokenKey="" cam="front" />);
    await waitFor(() => expect(screen.getByTestId("page-info").textContent).toContain("of 1 sessions"));
    expect(screen.queryByText("Scope")).toBeNull();
    expect(screen.queryAllByTestId("session-scope")).toHaveLength(0);
    expect(urls.some((u) => u.includes("&all=1"))).toBe(false);
    fireEvent.click(screen.getByText("All versions"));
    await waitFor(() => expect(screen.getAllByTestId("session-scope")).toHaveLength(2));
    expect(screen.getByText("Scope")).toBeTruthy();
    expect(screen.getAllByTestId("session-scope").map((c) => c.textContent)).toEqual(["yolov8s | auto v10", "yolov8s | auto v3"]);
    expect(urls.some((u) => u.startsWith("/api/sessions/page?page=1&size=20&all=1&cam=front"))).toBe(true);
    fireEvent.click(screen.getByText("This grid"));
    await waitFor(() => expect(screen.queryAllByTestId("session-scope")).toHaveLength(0));
  });

  it("the Refresh button fetches the table again at once", async () => {
    let calls = 0;
    vi.stubGlobal("fetch", (url: string) => {
      if (url.startsWith("/api/sessions/page")) {
        calls += 1;
        return json({ items: [row(calls)], total: 1, open: 1, page: 1, size: 20, pages: 1 });
      }
      return json({ backend: "sqlite", error: null, cameras: [] });
    });
    render(<HistoryPage tokenKey="" cam="front" />);
    await waitFor(() => expect(calls).toBe(1));
    fireEvent.click(screen.getByTitle("Refresh"));
    await waitFor(() => expect(calls).toBe(2));
    await waitFor(() => expect(screen.getByText("2")).toBeTruthy()); // the new row (#2) is shown
  });
});


describe("history page: the log against the live count", () => {
  const sync = (live: unknown) => ({ backend: "postgres", error: null, cameras: [{ id: "front", display: "Front", sqlite: { sessions: 5, events: 9 },
    behind: { sessions: 0, events: 0 }, rows: { sessions: 5, events: 9, samples: 3 }, age_s: 1, error: "", source: "postgres", live }] });
  const page = { items: [], total: 0, open: 0, page: 1, size: 20, pages: 1 };
  const serve = (live: unknown) => vi.stubGlobal("fetch", (url: string) => json(url.startsWith("/api/history/sync") ? sync(live) : page));

  it("warns when the log has differed from the live count for 30 s or more, with the error", async () => {
    serve({ expected: 4, open: 3, mismatch_s: 45.4, error: "OperationalError: database is locked" });
    render(<HistoryPage tokenKey="" cam="front" />);
    await waitFor(() => expect(screen.getByTestId("history-mismatch")).toBeTruthy());
    const text = screen.getByTestId("history-mismatch").textContent ?? "";
    expect(text).toContain("3 open sessions against 4 vehicles counted");
    expect(text).toContain("for 45 s");
    expect(text).toContain("database is locked");
  });

  it("stays quiet for a short difference, an equal count, or no worker", async () => {
    for (const live of [{ expected: 4, open: 3, mismatch_s: 12, error: "" }, { expected: 4, open: 4, mismatch_s: 0, error: "" }, null]) {
      serve(live);
      const view = render(<HistoryPage tokenKey="" cam="front" />);
      await waitFor(() => expect(screen.getByTestId("history-source").textContent).toContain("PostgreSQL"));
      expect(screen.queryByTestId("history-mismatch")).toBeNull();
      view.unmount();
    }
  });
});
