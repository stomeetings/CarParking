import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Loader2, Maximize2, Pause, Play, Plus } from "lucide-react";
import { memo, useEffect, useRef, useState } from "react";
import { FullscreenButton, stageStyle, useFullscreen } from "../fullscreen";
import { useInView, useLive, usePageVisible, useWebRtc } from "../hooks";
import { SLOT_COLORS, ago, moveSelection, pageOf, points, slotColorKey, tileHealth, type TileTone } from "../lib";
import type { CameraInfo } from "../types";

export type GridSize = 1 | 2 | 3;

const TOUR_SECONDS = 10;

/** CCTV multi-view: size x size tiles of live cameras, paged, sized to fit the screen (no scrolling on a desktop).
 * Click a tile to select its camera (the header follows; choosing one in the header brings its page), double-click
 * or Enter to open it (the full dashboard, HD), arrows move the selection, PgUp/PgDn change page; "Tour" cycles the
 * pages every 10 s like a DVR sequence. Tiles play the sub-stream (cheap) and only while they can be seen. */
export function CameraGrid({ cameras, size, page, onPage, selected, tokenKey, onSelect, onOpen, refresh = 0, every = 0 }: {
  refresh?: number; // Refresh data: every tile's live state reconnects in the background
  every?: number; // auto-sync interval (s) of the tiles' live state
  cameras: CameraInfo[];
  size: GridSize;
  page: number;
  onPage: (p: number) => void;
  selected: string | null;
  tokenKey: string;
  onSelect: (id: string) => void;
  onOpen: (id: string) => void;
}) {
  const per = size * size;
  const ids = cameras.map((c) => c.id);
  const pages = Math.max(1, Math.ceil(cameras.length / per));
  const p = Math.min(page, pages - 1);
  const shown = cameras.slice(p * per, p * per + per);
  const cols = { 1: "grid-cols-1", 2: "md:grid-cols-2", 3: "md:grid-cols-2 xl:grid-cols-3" }[size];
  const [tour, setTour] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  // the header (or the Cameras page) picked a camera on another page: show that page
  useEffect(() => {
    const target = pageOf(ids, selected, per);
    if (target != null && target !== p) onPage(target);
  }, [selected, per, ids.join()]); // eslint-disable-line react-hooks/exhaustive-deps

  // DVR "sequence": the next page every TOUR_SECONDS
  useEffect(() => {
    if (!tour || pages < 2) return;
    const t = setInterval(() => onPage((p + 1) % pages), TOUR_SECONDS * 1000);
    return () => clearInterval(t);
  }, [tour, pages, p, onPage]);

  useEffect(() => box.current?.focus({ preventScroll: true }), []); // keys work at once

  const onKey = (e: React.KeyboardEvent) => {
    const visibleIds = shown.map((c) => c.id);
    const width = window.innerWidth;
    const colsNow = width < 768 ? 1 : size === 3 && width < 1280 ? 2 : size;
    if (e.key.startsWith("Arrow")) {
      const next = moveSelection(visibleIds, selected && visibleIds.includes(selected) ? selected : null, e.key, colsNow);
      if (next) onSelect(next);
      e.preventDefault();
    } else if (e.key === "Enter" && selected) onOpen(selected);
    else if (e.key === "PageDown" && p < pages - 1) goPage(p + 1);
    else if (e.key === "PageUp" && p > 0) goPage(p - 1);
  };

  // a tour step moves the selection onto the new page (so the page-follows-selection effect doesn't undo it)
  const goPage = (n: number) => {
    onPage(n);
    const first = cameras[n * per];
    if (first && pageOf(ids, selected, per) !== n) onSelect(first.id);
  };
  useEffect(() => {
    if (tour && pageOf(ids, selected, per) !== p && cameras[p * per]) onSelect(cameras[p * per].id);
  }, [p]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-3">
      <div ref={box} tabIndex={0} onKeyDown={onKey} aria-label="Camera grid: arrows select, Enter opens, PgUp/PgDn page"
           className="mx-auto rounded-lg outline-none md:max-w-[var(--fit)]"
           style={{ "--fit": `calc((100vh - 13rem) * 16 / 9)` } as React.CSSProperties}>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={`${size}-${p}`} initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -24 }}
                      transition={{ duration: 0.18 }} className={`grid gap-3 ${cols}`} data-testid="camera-grid">
            {shown.map((c) => (
              <CameraTile key={c.id} camera={c} tokenKey={tokenKey} selected={c.id === selected} refresh={refresh} every={every}
                          onSelect={onSelect} onOpen={onOpen} />
            ))}
            {Array.from({ length: per - shown.length }, (_, i) => (
              <a key={`empty-${i}`} href="#/cameras"
                 className="flex aspect-video items-center justify-center gap-2 rounded-lg border-2 border-dashed border-line text-sm text-muted hover:border-primary hover:text-primary">
                <Plus size={16} /> Add a camera
              </a>
            ))}
          </motion.div>
        </AnimatePresence>
      </div>
      {pages > 1 && (
        <div className="flex flex-wrap items-center justify-center gap-3 text-sm text-muted" data-testid="grid-pager">
          <button type="button" disabled={p === 0} onClick={() => goPage(p - 1)} className="rounded p-1 hover:bg-surface disabled:opacity-40" aria-label="Previous page">
            <ChevronLeft size={18} />
          </button>
          <div className="flex items-center gap-1.5">
            {Array.from({ length: pages }, (_, i) => (
              <button key={i} type="button" onClick={() => goPage(i)} aria-label={`Page ${i + 1}`}
                      className={`h-2 rounded-full transition-all ${i === p ? "w-6 bg-primary" : "w-2 bg-line hover:bg-muted"}`} />
            ))}
          </div>
          <span className="tabular-nums">page {p + 1} / {pages}</span>
          <button type="button" disabled={p >= pages - 1} onClick={() => goPage(p + 1)} className="rounded p-1 hover:bg-surface disabled:opacity-40" aria-label="Next page">
            <ChevronRight size={18} />
          </button>
          <button type="button" onClick={() => setTour((t) => !t)} data-testid="grid-tour"
                  title={`Cycle the pages every ${TOUR_SECONDS} s`}
                  className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs font-semibold ${tour ? "border-primary bg-primary text-white" : "border-line bg-surface"}`}>
            {tour ? <Pause size={12} /> : <Play size={12} />} Tour
          </button>
        </div>
      )}
    </div>
  );
}

const TONES: Record<TileTone, { dot: string; pill: string }> = {
  ok: { dot: "bg-success", pill: "bg-success/90" },
  starting: { dot: "bg-info animate-pulse", pill: "bg-info/90" },
  idle: { dot: "bg-warning", pill: "bg-warning/90 text-ink" },
  stream: { dot: "bg-warning animate-pulse", pill: "bg-warning/90 text-ink" },
  down: { dot: "bg-danger animate-pulse", pill: "bg-danger/90" },
};

export const CameraTile = memo(function CameraTile({ camera, tokenKey, selected, onSelect, onOpen, refresh = 0, every = 0 }: {
  refresh?: number;
  every?: number;
  camera: CameraInfo;
  tokenKey: string;
  selected: boolean;
  onSelect: (id: string) => void;
  onOpen: (id: string) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const fsBox = useRef<HTMLDivElement>(null);
  const fs = useFullscreen(fsBox, video);
  const visible = usePageVisible();
  const inView = useInView(fsBox);
  const hdOnly = !camera.streams.sd;
  // video only while someone can see it: off screen or a hidden tab closes the stream (go2rtc stops converting)
  const status = useWebRtc(video, camera.streams.sd ?? camera.streams.hd, visible && (inView || fs.active), tokenKey, refresh);
  const live = useLive(false, tokenKey, camera.id, false, refresh, every); // state only: slots and counts of this camera
  const snap = live.state?.snapshot;
  const [w, h] = snap?.shape ?? [2560, 1440];
  // until this tile's own live state arrives, the camera list's status (no false "worker down" on load)
  const connected = live.state ? live.state.connected : camera.status.connected;
  const enabled = live.state ? !!snap?.enabled : camera.status.enabled;
  const running = live.state ? !!snap?.enabled && !!snap?.counts : camera.status.running;
  const counts = (live.state ? snap?.counts : camera.status.counts) ?? null;
  const health = tileHealth({ connected, heartbeatAge: live.state?.heartbeat_age ?? camera.status.heartbeat_age, enabled,
                              running, hasCounts: !!counts }, status);
  const tracking = connected && enabled && running && !!counts;
  const aiAge = live.state && snap?.frame_ts ? live.state.time - snap.frame_ts : null;
  const tone = TONES[health.tone];

  return (
    <div ref={fsBox} className={fs.active ? "flex items-center justify-center bg-black" : ""}>
      <div data-testid={`tile-${camera.id}`} data-health={health.tone} onClick={() => onSelect(camera.id)}
           onDoubleClick={fs.active ? fs.toggle : () => onOpen(camera.id)}
           title={fs.active ? "Double-click or Esc: exit fullscreen" : "Double-click (or Enter): open this camera"}
           className={`group relative cursor-pointer overflow-hidden bg-stage shadow-sm transition ${fs.active ? "" : `rounded-lg ring-2 ${selected ? "ring-primary shadow-lg" : "ring-transparent hover:ring-primary/40"}`}`}
           style={stageStyle(fs.active, w, h)}>
        <video ref={video} muted autoPlay playsInline className={`absolute inset-0 h-full w-full object-fill ${health.tone === "down" ? "opacity-50 grayscale" : ""}`} />
        {status !== "playing" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-xs text-white/75" data-testid={`tile-video-${camera.id}`}>
            {status === "failed" ? <span>Stream unavailable: retrying</span> : status === "off" ? <span>Paused (not visible)</span> : (
              <><Loader2 size={22} className="animate-spin" />{status === "reconnecting" ? "Reconnecting stream…" : "Connecting stream…"}</>
            )}
          </div>
        )}
        {tracking && (
          <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
            {(snap?.slots ?? []).map((s) => {
              const col = SLOT_COLORS[slotColorKey(s)] ?? SLOT_COLORS.UNKNOWN;
              return <polygon key={s.id} points={points(s.poly, w, h)} fill={col.fill} stroke={col.stroke} strokeWidth={2}
                              vectorEffect="non-scaling-stroke" />;
            })}
          </svg>
        )}
        {health.tone === "down" && <div className="pointer-events-none absolute inset-0 bg-danger/15" />}

        <div className="absolute inset-x-0 top-0 flex items-center gap-2 bg-gradient-to-b from-black/75 to-transparent px-3 py-2 text-white">
          <span className={`h-2 w-2 shrink-0 rounded-full ${tone.dot}`} />
          <span className="truncate text-sm font-semibold">{camera.display}</span>
          <span className={`hidden shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold sm:inline ${tone.pill}`} data-testid={`tile-health-${camera.id}`}>
            {health.label}
          </span>
          <span className="ml-auto shrink-0 rounded bg-white/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase">{camera.status.slot_mode}</span>
          <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold ${hdOnly ? "bg-warning/90 text-ink" : "bg-white/15"}`}
                data-testid={`tile-stream-${camera.id}`}
                title={hdOnly ? "No sub-stream: this tile plays HD (more CPU). Add a sub-stream URL on the Cameras page." : "Sub-stream (low bandwidth); HD when opened"}>
            {hdOnly ? "HD" : "SD"}
          </span>
          {!fs.active && (
            <button type="button" onClick={(e) => (e.stopPropagation(), onOpen(camera.id))} aria-label="Open in the dashboard"
                    title="Open this camera's dashboard (HD)" className="shrink-0 rounded p-0.5 opacity-70 hover:bg-white/20 hover:opacity-100">
              <Maximize2 size={14} />
            </button>
          )}
          <FullscreenButton active={fs.active} onClick={fs.toggle} className="!p-1 shrink-0" />
        </div>

        <div className="absolute inset-x-0 bottom-0 flex flex-wrap items-center gap-1.5 bg-gradient-to-t from-black/75 to-transparent px-3 py-2 text-[11px] font-semibold text-white"
             data-testid={`tile-counts-${camera.id}`}>
          {tracking && counts ? (
            <>
              <span className="rounded bg-success px-1.5 py-0.5">FREE {counts.free}</span>
              <span className="rounded bg-danger px-1.5 py-0.5">BUSY {counts.busy}</span>
              {counts.unknown > 0 && <span className="rounded bg-warning px-1.5 py-0.5 text-ink" title="Slots not decided yet">? {counts.unknown}</span>}
              <span className="rounded bg-white/20 px-1.5 py-0.5">TOTAL {counts.slots}</span>
              {aiAge != null && <span className="ml-auto font-normal text-white/70" title="Age of the worker's latest detections">AI {ago(aiAge)} ago</span>}
            </>
          ) : (
            <span className={`rounded px-1.5 py-0.5 ${tone.pill}`}>{health.label}</span>
          )}
        </div>
      </div>
    </div>
  );
});
