import { Box, Eye, Layers, Radio, Ruler, ScanLine, Video, VideoOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { withCam, withToken } from "../api";
import { type VideoStatus, usePageVisible, useWebRtc } from "../hooks";
import { BLOCK_HATCH, ENCROACH_HATCH, REJECT_TEXT, SLOT_COLORS, baySource, detectionTitle, outsidePath, points, slotColorKey, slotFlags, slotMatches, slotTitle } from "../lib";
import type { BaySource, CameraInfo, Detection, Slot, SlotFilter, Snapshot } from "../types";
import { FullscreenButton, stageStyle, useFullscreen } from "../fullscreen";
import { MetricOverlay, useMetricsLayer } from "./Metrics";
import { BlockHatch, CanvasFrame, EncroachHatch, Pill } from "./ui";

type Hover = { x: number; y: number; title: string; lines: string[]; color: string };

// a camera's go2rtc streams by kind (GET /api/cameras): hd = the main stream as 1080p H.264 (sharp, every browser),
// sd = the sub-stream as H.264 (cheap: grid tiles, slow links), native = as the camera sends it (HEVC)
type Kind = "hd" | "sd" | "native";
const KINDS: Record<Kind, { label: string; title: string }> = {
  hd: { label: "HD · 1080p", title: "Main stream as 1920×1080 H.264: sharp, plays in every browser" },
  sd: { label: "SD", title: "The sub-stream as H.264: for slow links" },
  native: { label: "Native · HEVC", title: "As the camera sends it (HEVC): Safari, or Edge with hardware decoding" },
};
const STREAM_KEY = "carparking.streamKind";
const savedKind = (): Kind => {
  try {
    const s = localStorage.getItem(STREAM_KEY) ?? "";
    return s === "sd" || s === "native" ? s : "hd";
  } catch {
    return "hd";
  }
};

/** The overlay layers the viewer switched on / off (Slots, Boxes, Area): remembered, so a refresh, a reconnect or
 * the player mounting again never turns a hidden layer back on. Missing / unreadable = all on. */
export type OverlayLayers = { slots: boolean; boxes: boolean; area: boolean };
export const LAYERS_KEY = "carparking.layers";
export const savedLayers = (): OverlayLayers => {
  try {
    const v = JSON.parse(localStorage.getItem(LAYERS_KEY) ?? "{}") ?? {};
    return { slots: v.slots !== false, boxes: v.boxes !== false, area: v.area !== false };
  } catch {
    return { slots: true, boxes: true, area: true };
  }
};

/** Native size of the picture vs the device pixels it is shown on: true when the browser has to upscale (a low-res
 * stream on a big screen), where crisp (pixelated) scaling beats a blur. HD shown smaller stays smoothly filtered. */
function useUpscaled(el: React.RefObject<HTMLDivElement | null>, native: [number, number] | null) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!el.current) return;
    const ro = new ResizeObserver(([e]) => setShown(e.contentRect.width * window.devicePixelRatio));
    ro.observe(el.current);
    return () => ro.disconnect();
  }, [el]);
  return !!native && native[0] > 0 && native[0] < shown * 0.9;
}

/** The camera (WebRTC video) with the worker's slots and detections drawn over it as SVG, hover tooltips on both.
 * When the video can't play (no WebRTC path, blocked UDP, HEVC on a browser without it), the worker's own overlay
 * images (WebSocket frames) take its place: onNeedFrames switches them on. */
export function LiveView({ snapshot, frameUrl, onNeedFrames, camera, onExit, tokenKey, refresh = 0, slotFilter = null, onClearFilter }: {
  refresh?: number; // Refresh data: the video reconnects too
  slotFilter?: SlotFilter; // only these slots drawn (the counter cards); the others faded
  onClearFilter?: () => void;
  snapshot: Snapshot | undefined;
  frameUrl: string | null;
  onNeedFrames: (need: boolean) => void;
  camera: CameraInfo | undefined;
  onExit?: () => void; // fullscreen from the grid: a double-click on the picture goes back
  tokenKey: string;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [kind, setKindState] = useState<Kind>(savedKind);
  const setKind = (k: Kind) => {
    setKindState(k);
    try {
      localStorage.setItem(STREAM_KEY, k);
    } catch {
      /* not remembered in a private window */
    }
  };
  const streams = camera?.streams ?? { hd: "parking_hd", sd: "parking", native: "parking_native" };
  const stream = streams[kind] ?? streams.hd;
  const [native, setNative] = useState<[number, number] | null>(null);
  const [frameNative, setFrameNative] = useState<[number, number] | null>(null);
  const [videoOn, setVideoOn] = useState(true);
  const [layers, setLayersState] = useState<OverlayLayers>(savedLayers);
  const setLayers = (change: (l: OverlayLayers) => OverlayLayers) => setLayersState((l) => {
    const next = change(l);
    try {
      localStorage.setItem(LAYERS_KEY, JSON.stringify(next));
    } catch {
      /* not remembered in a private window: kept for this view only */
    }
    return next;
  });
  const [metricsOn, toggleMetrics] = useMetricsLayer(); // Phase 2: only offered while the snapshot carries geometry
  const [hover, setHover] = useState<Hover | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const fsBox = useRef<HTMLDivElement>(null);
  const fs = useFullscreen(fsBox, video);
  const visible = usePageVisible(); // a hidden tab plays no video (go2rtc stops converting)
  const status = useWebRtc(video, stream, videoOn && visible, tokenKey, refresh);
  // after a failure the frame images stay until the video really plays again: a stream that keeps retrying
  // (connecting -> failed -> connecting) must not flip the picture and the frame feed every few seconds
  const [lost, setLost] = useState(false);
  useEffect(() => setLost(false), [stream]);
  useEffect(() => {
    if (status === "failed") setLost(true);
    if (status === "playing") setLost(false);
  }, [status]);
  const fallback = !videoOn || lost || status === "failed" || (status === "off" && visible);
  const baked = fallback && snapshot?.clean_frames !== true; // the frame images have the worker's drawing in them
  const crisp = useUpscaled(frame, fallback ? frameNative : native);

  useEffect(() => onNeedFrames(fallback && visible), [fallback, visible, onNeedFrames]);

  const [w, h] = snapshot?.shape ?? [2560, 1440];
  const slots = snapshot?.slots ?? [];
  const dets = snapshot?.detections ?? [];
  const title = camera?.display ?? snapshot?.camera ?? "Camera";

  const show = (e: React.MouseEvent, title: string, lines: string[], color: string) => {
    const box = frame.current?.getBoundingClientRect();
    if (box) setHover({ x: e.clientX - box.left, y: e.clientY - box.top, title, lines, color });
  };

  return (
    <CanvasFrame
      title={<span className="flex items-center gap-2"><Radio size={16} className="text-danger" /> {title}</span>}
      actions={
        <>
          {(Object.keys(KINDS) as Kind[]).filter((k) => streams[k]).map((k) => (
            <Pill key={k} active={kind === k && videoOn} onClick={() => (setKind(k), setVideoOn(true))} title={KINDS[k].title}>
              <Video size={12} /> {KINDS[k].label}
            </Pill>
          ))}
          <Pill active={!videoOn} onClick={() => setVideoOn((v) => !v)} title="The worker's processed frames instead of video">
            <VideoOff size={12} /> Frames
          </Pill>
          <span className="mx-1 h-4 w-px bg-line" />
          <Pill active={layers.slots} onClick={() => setLayers((l) => ({ ...l, slots: !l.slots }))} title="Slot outlines">
            <Layers size={12} /> Slots
          </Pill>
          <Pill active={layers.boxes} onClick={() => setLayers((l) => ({ ...l, boxes: !l.boxes }))} title="Detected vehicles">
            <Box size={12} /> Boxes
          </Pill>
          <Pill active={layers.area} onClick={() => setLayers((l) => ({ ...l, area: !l.area }))} title="Dim outside the parking area">
            <ScanLine size={12} /> Area
          </Pill>
          {snapshot?.geometry && (
            <Pill active={metricsOn} onClick={toggleMetrics} title="Measured vehicle footprints, distances and bays (from the camera's calibration)">
              <Ruler size={12} /> Metrics
            </Pill>
          )}
        </>
      }
    >
      <div ref={fsBox} className={fs.active ? "flex items-center justify-center bg-black" : ""} data-testid="video-fs-box">
      <div ref={frame} className="relative w-full bg-stage" style={stageStyle(fs.active, w, h)} onMouseLeave={() => setHover(null)}
           onDoubleClick={fs.active ? fs.toggle : onExit}
           title={fs.active ? "Double-click or Esc: exit fullscreen" : onExit ? "Double-click: back to the grid" : undefined}>
        <FullscreenButton active={fs.active} onClick={fs.toggle} className="absolute top-2 right-2 z-20" />
        {/* every stream and the overlay share the camera's 16:9 frame, so fill = no letterbox, no distortion; the
            browser scales the picture once, to the box's device pixels (crisp scaling only when it must upscale) */}
        {/* poster: the camera's latest still frame while the HD stream connects (no black gap after a double-click) */}
        <video ref={video} muted autoPlay playsInline data-testid="live-video"
               poster={camera && snapshot?.frame_ts ? withToken(withCam(`/api/frame.jpg?raw=1&r=${refresh}`, camera.id)) : undefined}
               onLoadedMetadata={(e) => setNative([e.currentTarget.videoWidth, e.currentTarget.videoHeight])}
               onResize={(e) => setNative([e.currentTarget.videoWidth, e.currentTarget.videoHeight])}
               className={`absolute inset-0 h-full w-full object-fill ${crisp ? "[image-rendering:pixelated]" : ""} ${fallback ? "invisible" : ""}`} />
        {fallback && frameUrl && (
          <img src={frameUrl} alt="Latest processed frame" data-testid="frame-image"
               onLoad={(e) => setFrameNative([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
               className={`absolute inset-0 h-full w-full object-fill ${crisp ? "[image-rendering:pixelated]" : ""}`} />
        )}
        {fallback && !frameUrl && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-white/70">Waiting for the first frame…</div>
        )}
        {/* the overlay in the camera's own pixels (the worker's full frame, e.g. 2560×1440): resolution-independent
            vectors, lines a fixed screen width (non-scaling-stroke), so they stay sharp at any size */}
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full"
             shapeRendering="geometricPrecision" textRendering="geometricPrecision" data-testid="overlay">
          {/* an older worker's (or clean_frames off) images carry its own drawing: then the SVG is only for hovering. A clean
              frame (the video, or the worker's clean frame image) has nothing drawn: every layer is the SVG, and only the
              toggles decide what is drawn */}
          <EncroachHatch size={Math.max(6, w / 140)} />
          <BlockHatch size={Math.max(8, w / 120)} />
          {layers.area && !baked && <path d={outsidePath(snapshot?.area, w, h)} fill="rgba(0,0,0,0.45)" fillRule="evenodd" />}
          {/* the slot filter: the others faded to a thin outline; in frame mode (the image has every slot drawn already)
              the matching slots are outlined on top instead */}
          {layers.slots && slots.map((s) => {
            const match = slotMatches(s, slotFilter);
            return <SlotShape key={s.id} slot={s} w={w} h={h} ghost={baked && !(slotFilter != null && match)}
                              faded={!match} onHover={show} source={baySource(snapshot?.auto_slots, s.id)} />;
          })}
          {layers.boxes && dets.map((d, i) => <BoxShape key={i} det={d} w={w} h={h} ghost={baked} onHover={show} />)}
          {snapshot?.geometry && metricsOn && <MetricOverlay geometry={snapshot.geometry} w={w} h={h} />}
        </svg>
        {slotFilter != null && (
          <button type="button" onClick={onClearFilter} data-testid="filter-chip"
                  className="absolute top-2 left-2 z-20 inline-flex items-center gap-1.5 rounded-full bg-stage/80 px-3 py-1 text-xs font-semibold text-white shadow hover:bg-stage">
            <span className={`h-2 w-2 rounded-full ${{ FREE: "bg-success", BUSY: "bg-danger", UNKNOWN: "bg-warning", ENCROACHED: "bg-[#f08c00]", BLOCKED: "bg-[#5b6475]" }[slotFilter]}`} />
            Showing {slotFilter} only · {slots.filter((s) => slotMatches(s, slotFilter)).length} of {slots.length}
            <span className="text-white/70">✕</span>
          </button>
        )}
        {hover && (
          <div className="pointer-events-none absolute z-10 min-w-36 rounded-md border border-line bg-surface/95 px-3 py-2 text-xs shadow-lg"
               style={{ left: hover.x + 14, top: hover.y + 14, borderLeft: `3px solid ${hover.color}` }}>
            <p className="font-semibold text-ink">{hover.title}</p>
            {hover.lines.map((l) => <p key={l} className="text-muted">{l}</p>)}
          </div>
        )}
        <VideoChip status={status} fallback={fallback} snapshot={snapshot} size={fallback ? frameNative : native} />
        <StreamBanner stream={snapshot?.stream} />
      </div>
      </div>
    </CanvasFrame>
  );
}

const SOURCE_NOTE: Record<BaySource, string> = {
  measured: "",
  extrapolated: "extrapolated: the lane continued past the measured ground, not measured",
  manual: "pinned by the operator",
  dynamic: "vehicle-driven: made for a vehicle that stands in no bay, gone when it leaves",
  sub: "sub-bay: a second vehicle in a slot another one holds, gone when it leaves",
};

export function SlotShape({ slot, w, h, ghost, faded = false, onHover, source = null, line = 1 }: {
  line?: number; // multiplies the outline width (a zoomed picture draws thinner lines)
  source?: BaySource | null; // the auto grid's: extrapolated bays are dotted, pinned ones carry a P
  slot: Slot;
  w: number;
  h: number;
  ghost: boolean;
  faded?: boolean; // filtered out: a faint outline, no fill, no label
  onHover: (e: React.MouseEvent, title: string, lines: string[], color: string) => void;
}) {
  const encroached = slot.avail === "ENCROACHED";
  const blocked = slot.avail === "BLOCKED";
  const c = SLOT_COLORS[slotColorKey(slot)] ?? SLOT_COLORS.UNKNOWN;
  const [lx, ly] = slot.poly.reduce(([x, y], [px, py]) => [Math.min(x, px), Math.min(y, py)], [1, 1]);
  const label = `${slot.id}${source === "manual" ? "P" : source === "dynamic" ? "D" : source === "sub" ? "S" : ""} ${blocked ? "BLOCKED" : encroached ? "ENCROACHED" : slot.state}`;
  const note = source ? SOURCE_NOTE[source] : "";
  return (
    <g onMouseMove={(e) => onHover(e, slotTitle(slot), [encroached || blocked ? "FREE, not usable" : slot.state, ...slotFlags(slot), ...(note ? [note] : [])], c.stroke)} className="cursor-help"
       opacity={faded ? 0.18 : 1} data-testid="overlay-slot" data-state={slot.state} data-avail={slot.avail ?? undefined} data-faded={faded ? "1" : "0"}
       data-source={source ?? undefined}>
      <polygon points={points(slot.poly, w, h)} fill={ghost || faded ? "transparent" : blocked ? `url(#${BLOCK_HATCH})` : encroached ? `url(#${ENCROACH_HATCH})` : c.fill} stroke={ghost ? "transparent" : c.stroke}
               strokeWidth={2 * line} vectorEffect="non-scaling-stroke"
               strokeDasharray={slot.hidden || slot.blocked || slot.duplicate_of != null ? "7 4" : source === "extrapolated" ? "2 4" : source === "dynamic" || source === "sub" ? "8 3 2 3" : undefined} />
      {!ghost && !faded && (
        <g transform={`translate(${lx * w + w / 300}, ${ly * h + w / 300})`}>
          <rect width={label.length * w / 150 + w / 120} height={w / 55} rx={w / 400} fill={c.stroke} />
          <text x={w / 240} y={w / 75} fontSize={w / 90} fontWeight={700} fill="white">{label}</text>
        </g>
      )}
    </g>
  );
}

export function BoxShape({ det, w, h, ghost, onHover, line = 1 }: {
  line?: number; // multiplies the outline width (a zoomed picture draws thinner lines)
  det: Detection;
  w: number;
  h: number;
  ghost: boolean;
  onHover: (e: React.MouseEvent, title: string, lines: string[], color: string) => void;
}) {
  const [x1, y1, x2, y2] = det.box;
  // a box the non-vehicle filter dropped (cone, person, shadow ...): grey and dashed with its reason, not the yellow of a vehicle
  const dropped = det.kind === "vehicle" && !!det.rejected;
  const color = dropped ? "#9da5b1" : det.kind === "vehicle" ? "#f9b115" : "#d63384";
  const tag = dropped ? `ignored: ${det.rejected!.replace("_", " ")}` : "";
  if (det.poly && det.poly.length === 4 && !dropped) {  // an oriented box (YOLO-OBB): its tilted outline, the same yellow
    const pts = det.poly.map(([x, y]) => `${x * w},${y * h}`).join(" ");
    return (
      <polygon data-testid="overlay-obb" points={pts} fill="transparent" stroke={ghost ? "transparent" : color} strokeWidth={1.5 * line} vectorEffect="non-scaling-stroke" className="cursor-help"
               style={{ pointerEvents: "all" }}
               onMouseMove={(e) => onHover(e, detectionTitle(det), ["oriented box (rotated)", det.src === "zoom" ? "found by the zoomed re-check" : "whole-frame detection"], color)} />
    );
  }
  return (
    <>
      <rect data-testid="overlay-box" data-rejected={dropped ? det.rejected : undefined} x={x1 * w} y={y1 * h} width={(x2 - x1) * w} height={(y2 - y1) * h} fill="transparent"
            stroke={ghost ? "transparent" : color} strokeWidth={1.5 * line} vectorEffect="non-scaling-stroke" className="cursor-help"
            strokeDasharray={dropped ? "6 4" : undefined} style={{ pointerEvents: "all" }}
            onMouseMove={(e) => onHover(e, detectionTitle(det),
              dropped ? ["ignored: not counted as a vehicle", REJECT_TEXT[det.rejected!] ?? det.rejected!]
                : [det.kind === "vehicle" ? "vehicle (counts for slots)" : "shown only, never a slot",
                   det.src === "zoom" ? "found by the zoomed re-check" : "whole-frame detection"], color)} />
      {dropped && !ghost && (
        <g transform={`translate(${x1 * w}, ${Math.max(0, y1 * h - w / 55)})`} pointerEvents="none">
          <rect width={tag.length * w / 150 + w / 120} height={w / 55} rx={w / 400} fill={color} />
          <text x={w / 240} y={w / 75} fontSize={w / 90} fontWeight={700} fill="white">{tag}</text>
        </g>
      )}
    </>
  );
}

/** A stale or frozen camera stream: its slots are shown UNKNOWN (grey) and nothing is counted or logged until it is live again. */
function StreamBanner({ stream }: { stream: Snapshot["stream"] }) {
  if (!stream || stream.state === "active") return null;
  const why = stream.state === "stale" ? `no new frame for ${Math.round(stream.age_s)} s` : `the same picture for ${Math.round(stream.same_s)} s`;
  return (
    <p className="absolute left-2 top-2 rounded-md bg-warning/90 px-3 py-1.5 text-xs font-medium text-white" role="status" data-testid="stream-banner">
      Camera stream {stream.state}: {why}. Slots are shown UNKNOWN and open sessions are kept until it is live again.
    </p>
  );
}

function VideoChip({ status, fallback, snapshot, size }: {
  status: VideoStatus;
  fallback: boolean;
  snapshot: Snapshot | undefined;
  size: [number, number] | null;
}) {
  const res = size && size[0] ? ` · ${size[0]}×${size[1]}` : "";
  const text = fallback
    ? `Frames${res} · ${snapshot?.rate ? `${snapshot.rate.toFixed(2)} /s` : "waiting"}`
    : { off: "Video off", connecting: "Connecting video…", playing: `Live video${res}`, failed: "Video unavailable",
        reconnecting: "Reconnecting video…" }[status];
  const dot = !fallback && status === "playing" ? "bg-success animate-pulse" : fallback ? "bg-info" : "bg-warning";
  return (
    <span className="absolute bottom-2 left-2 inline-flex items-center gap-1.5 rounded-full bg-stage/70 px-2.5 py-1 text-[11px] font-medium text-white"
          data-testid="video-status">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {text}
      {!fallback && status === "playing" && <span className="text-white/60"><Eye size={11} className="inline" /> overlay ≈ 1–3 s behind</span>}
    </span>
  );
}
