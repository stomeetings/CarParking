import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useLive } from "./hooks";

class FakeSocket {
  static all: FakeSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((m: { data: unknown }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  binaryType = "";
  closed = false;
  constructor(public url: string) {
    FakeSocket.all.push(this);
  }
  close() {
    this.closed = true;
  }
}

let fetches = 0;
beforeEach(() => {
  FakeSocket.all = [];
  fetches = 0;
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.stubGlobal("fetch", () => {
    fetches += 1;
    return Promise.resolve(new Response("[]", { status: 200, headers: { "Content-Type": "application/json" } }));
  });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

let seen: ReturnType<typeof useLive>;
function Probe({ frames, cam, refresh = 0 }: { frames: boolean; cam: string; refresh?: number }) {
  seen = useLive(frames, "", cam, true, refresh);
  return null;
}

describe("useLive (the dashboard's WebSocket)", () => {
  it("switching the frame images on and off keeps the live state: no blank dashboard, no refetch", async () => {
    const { rerender } = render(<Probe frames={false} cam="front" />);
    const first = FakeSocket.all[0];
    act(() => {
      first.onopen?.();
      first.onmessage?.({ data: JSON.stringify({ type: "state", connected: true, snapshot: { counts: { free: 3 } } }) });
    });
    expect(seen.link).toBe("open");
    const fetched = fetches;

    rerender(<Probe frames={true} cam="front" />); // the video failed: frames on
    expect(first.closed).toBe(true);
    expect(FakeSocket.all).toHaveLength(2);
    expect(FakeSocket.all[1].url).toContain("frames=1");
    expect(seen.state?.connected).toBe(true); // still the last state while the new socket connects
    expect(seen.link).toBe("open");
    expect(fetches).toBe(fetched); // the ticker isn't reloaded

    rerender(<Probe frames={false} cam="front" />);
    expect(seen.state?.connected).toBe(true);
    await act(async () => {});
    expect(fetches).toBe(fetched);
  });

  it("another camera starts from nothing", async () => {
    const { rerender } = render(<Probe frames={false} cam="front" />);
    act(() => FakeSocket.all[0].onmessage?.({ data: JSON.stringify({ type: "state", connected: true }) }));
    expect(seen.state).not.toBeNull();
    rerender(<Probe frames={false} cam="gate" />);
    expect(seen.state).toBeNull();
    expect(seen.link).toBe("connecting");
    expect(FakeSocket.all.at(-1)?.url).toContain("cam=gate");
    await act(async () => {});
  });
});

describe("Refresh data", () => {
  it("reconnects and refetches in the background: the state stays on screen, nothing reloads", async () => {
    const { rerender } = render(<Probe frames={false} cam="front" />);
    act(() => {
      FakeSocket.all[0].onopen?.();
      FakeSocket.all[0].onmessage?.({ data: JSON.stringify({ type: "state", connected: true }) });
    });
    const at = seen.at;
    expect(at).not.toBeNull();
    await act(async () => {});
    const fetched = fetches;
    rerender(<Probe frames={false} cam="front" refresh={1} />);
    expect(FakeSocket.all[0].closed).toBe(true);
    expect(FakeSocket.all).toHaveLength(2);
    expect(seen.state?.connected).toBe(true); // kept while the new connection comes up
    await act(async () => {});
    expect(fetches).toBe(fetched + 1); // the ticker refetched
    act(() => FakeSocket.all[1].onmessage?.({ data: JSON.stringify({ type: "state", connected: true }) }));
    expect(seen.at).toBeGreaterThanOrEqual(at!);
  });
});

describe("auto-sync interval", () => {
  it("asks the server for routine updates every N seconds", async () => {
    function P() {
      useLive(false, "", "front", false, 0, 20);
      return null;
    }
    render(<P />);
    expect(FakeSocket.all[0].url).toContain("every=20");
    await act(async () => {});
  });
});
