import { useEffect, useRef, useState } from "react";
import { api, withCam, withToken, wsUrl } from "./api";
import { mergeEvents, stalled } from "./lib";
import type { ParkingEvent, SlotFilter, StateMessage } from "./types";

export type Link = "connecting" | "open" | "closed" | "unauthorized";

/** The backend's WebSocket (/ws/stream): the worker's state (about once a second), new entries/exits for the
 * ticker and, with frames, the worker's overlay image per processed frame (the fallback when video can't play).
 * Reconnects with backoff; a 4401 close means the viewer token is missing or wrong.
 * refresh: a new value = "Refresh data": a fresh connection (the server sends the current state at once) and the
 * ticker refetched, in the background (nothing on screen is cleared, no page reload). .at = when the last state came.
 * every: seconds between routine state updates (the server's ?every=; 0 = about once a second). A car parking or
 * leaving, a slot's state, the worker's connection and setting changes still arrive at once. */
export function useLive(frames: boolean, token: string, cam: string | null = null, withEvents = true, refresh = 0,
                        every = 0) {
  const [state, setState] = useState<StateMessage | null>(null);
  const [at, setAt] = useState<number | null>(null);
  const [events, setEvents] = useState<ParkingEvent[]>([]);
  const [link, setLink] = useState<Link>("connecting");
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const lastUrl = useRef<string | null>(null);

  // another camera (or token): nothing of the previous one's. Only here: switching the frame images on or off below
  // reconnects in the background and keeps the counts, the ticker and the last frame on screen (clearing them on
  // every switch made the whole dashboard blank and refill like a page reload whenever the video came and went)
  useEffect(() => {
    let alive = true;
    setLink("connecting");
    setState(null);
    setAt(null);
    setEvents([]);
    setFrameUrl(null);
    if (withEvents) {
      api.get<ParkingEvent[]>(withCam("/api/events?limit=30&kinds=all", cam)).then((e) => alive && setEvents((cur) => mergeEvents(cur, e))).catch(() => {});
    }
    return () => {
      alive = false;
      if (lastUrl.current) URL.revokeObjectURL(lastUrl.current);
      lastUrl.current = null;
    };
  }, [token, cam, withEvents]);

  // Refresh data: the ticker's recent events again (merged: nothing disappears meanwhile)
  useEffect(() => {
    if (!refresh || !withEvents) return;
    let alive = true;
    api.get<ParkingEvent[]>(withCam("/api/events?limit=30&kinds=all", cam)).then((e) => alive && setEvents((cur) => mergeEvents(cur, e))).catch(() => {});
    return () => {
      alive = false;
    };
  }, [refresh]); // only on a click (cam / token changes refetch above)

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let delay = 1000;
    let stopped = false;

    const open = () => {
      const query = [frames && "frames=1", every > 0 && `every=${every}`].filter(Boolean).join("&");
      ws = new WebSocket(wsUrl(withCam(`/ws/stream${query ? `?${query}` : ""}`, cam)));
      ws.binaryType = "blob";
      ws.onopen = () => {
        delay = 1000;
        setLink("open");
      };
      ws.onmessage = (m) => {
        if (typeof m.data !== "string") {
          const url = URL.createObjectURL(m.data as Blob);
          setFrameUrl(url);
          if (lastUrl.current) URL.revokeObjectURL(lastUrl.current);
          lastUrl.current = url;
          return;
        }
        const msg = JSON.parse(m.data);
        if (msg.type === "state") {
          setState(msg as StateMessage);
          setAt(Date.now());
        }
        else if (msg.type === "event") setEvents((cur) => mergeEvents(cur, [msg as ParkingEvent]));
      };
      ws.onclose = (e) => {
        if (stopped) return;
        if (e.code === 4401) {
          setLink("unauthorized");
          return;
        }
        setLink("closed");
        timer = setTimeout(() => (setLink("connecting"), open()), delay);
        delay = Math.min(delay * 2, 10000);
      };
    };
    open();
    return () => {
      stopped = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, [frames, token, cam, refresh, every]);

  return { state, events, link, frameUrl, at };
}

/** The last value that was there, kept for up to `hold` ms while the new one is missing: the counter cards don't
 * blink to "–" while the worker reloads its model after a settings save or a slot map reset (a few "loading" state
 * messages without counts) or during a short heartbeat gap. Another key (camera, monitoring switched off): dropped
 * at once. After `hold` the missing value shows (a real outage). */
export function useHeld<T>(value: T | undefined, key: string, hold = 30000): T | undefined {
  const last = useRef<{ v: T; at: number; key: string } | null>(null);
  const [, expire] = useState(0);
  if (value !== undefined) last.current = { v: value, at: Date.now(), key };
  else if (last.current && last.current.key !== key) last.current = null;
  const held = value === undefined ? last.current : null;
  useEffect(() => {
    if (!held) return;
    const t = setTimeout(() => expire((n) => n + 1), Math.max(0, held.at + hold - Date.now()));
    return () => clearTimeout(t);
  }, [held, hold]);
  if (value !== undefined) return value;
  return held && Date.now() - held.at < hold ? held.v : undefined;
}

/** The counter cards' filter ("show BUSY only"), kept per camera for this browser tab: the live updates never touch it, and a
 * remount or reload of the page brings it back. Only the viewer changes it (another camera has its own). */
export const FILTER_KEY = "carparking.slotFilter";
const filterOf = (cam: string | null): SlotFilter => {
  try {
    const v = JSON.parse(sessionStorage.getItem(FILTER_KEY) ?? "{}")?.[cam ?? ""];
    return v === "FREE" || v === "BUSY" || v === "UNKNOWN" || v === "ENCROACHED" || v === "BLOCKED" ? v : null;
  } catch {
    return null;
  }
};
export function useSlotFilter(cam: string | null): [SlotFilter, (f: SlotFilter) => void] {
  const [filter, setFilter] = useState<SlotFilter>(() => filterOf(cam));
  useEffect(() => setFilter(filterOf(cam)), [cam]);
  const choose = (f: SlotFilter) => {
    setFilter(f);
    try {
      const all = JSON.parse(sessionStorage.getItem(FILTER_KEY) ?? "{}") ?? {};
      sessionStorage.setItem(FILTER_KEY, JSON.stringify({ ...all, [cam ?? ""]: f }));
    } catch {
      /* not remembered in a private window: kept for this view only */
    }
  };
  return [filter, choose];
}

/** No newly decoded video frame for this long while connected = a frozen picture: reconnect. */
export const STALL_MS = 5000;

export type VideoStatus = "off" | "connecting" | "playing" | "failed" | "reconnecting";

/** WebRTC from go2rtc, signalled through the backend (POST /api/webrtc?src=, SDP offer -> answer; go2rtc itself is
 * never reachable from the LAN). Retries after a failure or a dropped connection: 2 s, then up to every 10 s.
 * "reconnecting" = it had played and is coming back. A watchdog restarts a stream whose picture froze: no newly
 * decoded frame (WebRTC stats, framesDecoded) for STALL_MS while connected, a camera or go2rtc hiccup the connection
 * itself doesn't report. Not video.currentTime: on a live WebRTC stream Chrome advances it with the clock even while
 * no frame arrives, so a frozen picture was never noticed. refresh: a new value = reconnect now (Refresh data).
 * enabled=false (tile off screen, tab hidden) closes the connection, so go2rtc stops converting nobody's video. */
export function useWebRtc(video: React.RefObject<HTMLVideoElement | null>, src: string, enabled: boolean, token: string,
                          refresh = 0) {
  const [status, setStatus] = useState<VideoStatus>(enabled ? "connecting" : "off"); // no "off" flash on the first render
  const attempt = useRef(0);
  const played = useRef(false);

  useEffect(() => {
    if (!enabled) {
      setStatus("off");
      return;
    }
    let pc: RTCPeerConnection | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;
    let wait = 2000; // a cold stream (go2rtc starting ffmpeg) often fails once: retry soon, then back off

    const fail = () => {
      if (stopped) return;
      setStatus("failed");
      pc?.close();
      retry = setTimeout(start, wait);
      wait = Math.min(wait * 2, 10000);
    };

    async function start() {
      if (stopped) return;
      const mine = ++attempt.current;
      setStatus(played.current ? "reconnecting" : "connecting");
      try {
        pc = new RTCPeerConnection();
        pc.addTransceiver("video", { direction: "recvonly" });
        pc.ontrack = (e) => {
          const el = video.current;
          if (!el) return;
          el.srcObject = e.streams[0] ?? new MediaStream([e.track]);
          el.play().catch(() => {});
        };
        pc.onconnectionstatechange = () => {
          if (!pc || mine !== attempt.current) return;
          if (pc.connectionState === "connected") {
            played.current = true;
            setStatus("playing");
            wait = 2000;
          }
          if (pc.connectionState === "failed" || pc.connectionState === "disconnected") fail();
        };
        await pc.setLocalDescription(await pc.createOffer());
        await iceGathered(pc, 2000); // non-trickle: our candidates go inside the one offer
        const r = await fetch(withToken(`/api/webrtc?src=${encodeURIComponent(src)}`), {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: pc.localDescription?.sdp,
        });
        if (!r.ok) throw new Error(`signalling ${r.status}`);
        await pc.setRemoteDescription({ type: "answer", sdp: await r.text() });
        setTimeout(() => {
          if (mine === attempt.current && pc && pc.connectionState !== "connected") fail();
        }, 10000);
      } catch {
        fail();
      }
    }

    // the frozen-picture watchdog: decoded frames (stats); currentTime only where stats have no frame count
    let lastFrames = -1;
    let lastChange = Date.now();
    const watchdog = setInterval(async () => {
      const el = video.current;
      const conn = pc;
      if (!el || !conn || conn.connectionState !== "connected") return;
      let frames = -1;
      try {
        (await conn.getStats()).forEach((r) => {
          if (r.type === "inbound-rtp" && r.kind === "video" && typeof r.framesDecoded === "number") frames = r.framesDecoded;
        });
      } catch {
        /* no stats: below */
      }
      if (stopped || conn !== pc) return;
      const now = Date.now();
      const count = frames >= 0 ? frames : el.currentTime;
      if (count !== lastFrames) {
        lastFrames = count;
        lastChange = now;
      } else if (stalled(lastFrames, lastChange, count, now, STALL_MS)) {
        lastChange = now;
        fail(); // reconnect: the picture froze
      }
    }, 1000);

    start();
    return () => {
      stopped = true;
      clearTimeout(retry);
      clearInterval(watchdog);
      pc?.close();
      if (video.current) video.current.srcObject = null;
    };
  }, [src, enabled, token, video, refresh]);

  return status;
}

function iceGathered(pc: RTCPeerConnection, timeout: number): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      pc.removeEventListener("icegatheringstatechange", check);
      resolve();
    };
    const check = () => pc.iceGatheringState === "complete" && done();
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(done, timeout);
  });
}

/** Poll a JSON endpoint every `every` ms (pages that don't need the WebSocket). */
export function usePoll<T>(path: string | null, every: number, token: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!path) return;
    let alive = true;
    const load = () =>
      api
        .get<T>(path)
        .then((d) => alive && (setData(d), setError(null)))
        .catch((e) => alive && setError(String(e.message ?? e)));
    load();
    const t = setInterval(load, every);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [path, every, token, tick]);
  return { data, error, reload: () => setTick((t) => t + 1) };
}

/** Is the browser tab visible? (A hidden tab plays no video: its streams close.) */
export function usePageVisible(): boolean {
  const [visible, setVisible] = useState(() => typeof document === "undefined" || document.visibilityState !== "hidden");
  useEffect(() => {
    const on = () => setVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", on);
    return () => document.removeEventListener("visibilitychange", on);
  }, []);
  return visible;
}

/** Is the element on screen (or near it)? Off-screen tiles (scrolled away on a phone) close their streams. */
export function useInView(el: React.RefObject<HTMLElement | null>, margin = "100px"): boolean {
  const [inView, setInView] = useState(true);
  useEffect(() => {
    if (!el.current || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { rootMargin: margin });
    io.observe(el.current);
    return () => io.disconnect();
  }, [el, margin]);
  return inView;
}
