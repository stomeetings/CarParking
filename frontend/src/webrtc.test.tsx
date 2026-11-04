import { act, cleanup, render } from "@testing-library/react";
import { useRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useWebRtc } from "./hooks";

class FakePc {
  static all: FakePc[] = [];
  connectionState = "new";
  iceGatheringState = "complete";
  localDescription = { sdp: "offer" };
  frames = 0;
  closed = false;
  onconnectionstatechange: (() => void) | null = null;
  ontrack: unknown = null;
  constructor() {
    FakePc.all.push(this);
  }
  addTransceiver() {}
  async createOffer() {
    return { type: "offer", sdp: "offer" };
  }
  async setLocalDescription() {}
  async setRemoteDescription() {}
  addEventListener() {}
  removeEventListener() {}
  close() {
    this.closed = true;
  }
  connect() {
    this.connectionState = "connected";
    this.onconnectionstatechange?.();
  }
  async getStats() {
    return new Map([["v", { type: "inbound-rtp", kind: "video", framesDecoded: this.frames }]]);
  }
}

beforeEach(() => {
  vi.useFakeTimers();
  FakePc.all = [];
  vi.stubGlobal("RTCPeerConnection", FakePc);
  vi.stubGlobal("fetch", () => Promise.resolve(new Response("answer", { status: 201 })));
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function Player({ refresh = 0 }: { refresh?: number }) {
  const video = useRef<HTMLVideoElement>(null);
  useWebRtc(video, "parking_hd", true, "", refresh);
  return <video ref={video} />;
}

const settle = () => act(async () => { await vi.advanceTimersByTimeAsync(0); });

describe("video stall watchdog and Refresh data", () => {
  it("reconnects when no new frame is decoded for over 5 s, never while frames flow", async () => {
    const { rerender } = render(<Player />);
    await settle();
    const first = FakePc.all[0];
    first.connect();
    for (let i = 0; i < 10; i++) {  // 10 s of live video: frames keep coming
      first.frames += 25;
      await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    }
    expect(FakePc.all).toHaveLength(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(7000); }); // frozen: no new frames
    expect(first.closed).toBe(true);
    await act(async () => { await vi.advanceTimersByTimeAsync(2500); }); // the retry after a failure
    expect(FakePc.all.length).toBeGreaterThanOrEqual(2);
    const before = FakePc.all.length;
    rerender(<Player refresh={1} />); // Refresh data
    await settle();
    expect(FakePc.all.length).toBe(before + 1);
    expect(FakePc.all[before - 1].closed).toBe(true);
  });
});
