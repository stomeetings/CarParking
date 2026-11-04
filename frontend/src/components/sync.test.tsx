import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RefreshData, SyncStatus, savedSyncEvery } from "./Sync";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  localStorage.clear();
});

describe("sync indicator and Refresh data", () => {
  it("auto-syncs every 20 s by default, counts up and down, and warns when updates stop", () => {
    expect(savedSyncEvery()).toBe(20);
    vi.useFakeTimers();
    const t0 = Date.now();
    render(<SyncStatus at={t0} link="open" every={20} onEvery={() => {}} />);
    expect(screen.getByTestId("sync-text").textContent).toBe("· updated 0.0 s ago · next ≤ 20 s");
    act(() => vi.advanceTimersByTime(4500));
    expect(screen.getByTestId("sync-text").textContent).toBe("· updated 4.5 s ago · next ≤ 16 s");
    act(() => vi.advanceTimersByTime(26000));
    expect(screen.getByTestId("sync-text").textContent).toContain("No update for 31 s");
    cleanup();
    render(<SyncStatus at={t0} link="closed" every={20} onEvery={() => {}} />);
    expect(screen.getByTestId("sync-text").textContent).toContain("Reconnecting");
  });

  it("the interval picker offers 20 / 30 / 5 s and live", () => {
    const onEvery = vi.fn();
    render(<SyncStatus at={Date.now()} link="open" every={20} onEvery={onEvery} />);
    const pick = screen.getByTestId("sync-every") as HTMLSelectElement;
    expect([...pick.options].map((o) => o.textContent)).toEqual(["20 s", "30 s", "5 s", "live (1 s)"]);
    fireEvent.change(pick, { target: { value: "30" } });
    expect(onEvery).toHaveBeenCalledWith(30);
  });

  it("refreshes without a reload and spins until fresh data arrives", () => {
    const onRefresh = vi.fn();
    const reload = vi.fn();
    vi.stubGlobal("location", { ...globalThis.location, reload }); // (globalThis: no ReferenceError where there is no window)
    const { rerender } = render(<RefreshData onRefresh={onRefresh} at={1} />);
    fireEvent.click(screen.getByTestId("refresh-data"));
    expect(onRefresh).toHaveBeenCalledTimes(1);
    expect(reload).not.toHaveBeenCalled();
    expect(screen.getByTestId("refresh-data").textContent).toContain("Refreshing");
    rerender(<RefreshData onRefresh={onRefresh} at={Date.now() + 10} />);
    expect(screen.getByTestId("refresh-data").textContent).toContain("Refresh data");
    vi.unstubAllGlobals();
  });
});
