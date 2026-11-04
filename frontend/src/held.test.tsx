import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useHeld } from "./hooks";

afterEach(() => vi.useRealTimers());

describe("useHeld: the counter cards don't blink to '–'", () => {
  it("keeps the last value through a gap, drops it after the hold, at once for another key", () => {
    vi.useFakeTimers();
    const { result, rerender } = renderHook(({ v, key }: { v: number | undefined; key: string }) => useHeld(v, key, 30000),
                                            { initialProps: { v: 5 as number | undefined, key: "camera-1|false" } });
    expect(result.current).toBe(5);
    rerender({ v: undefined, key: "camera-1|false" }); // a "loading the model" message without counts
    expect(result.current).toBe(5);
    act(() => vi.advanceTimersByTime(20000));
    rerender({ v: undefined, key: "camera-1|false" });
    expect(result.current).toBe(5);
    rerender({ v: 6, key: "camera-1|false" }); // counts again: the new value at once
    expect(result.current).toBe(6);
    rerender({ v: undefined, key: "camera-1|false" });
    act(() => vi.advanceTimersByTime(30001)); // a real outage: the gap shows after the hold
    expect(result.current).toBeUndefined();
    rerender({ v: 4, key: "camera-1|false" });
    rerender({ v: undefined, key: "camera-2|false" }); // another camera: nothing of the previous one's
    expect(result.current).toBeUndefined();
    rerender({ v: 4, key: "camera-1|false" });
    rerender({ v: undefined, key: "camera-1|true" }); // monitoring switched off: no stale counts
    expect(result.current).toBeUndefined();
  });
});
