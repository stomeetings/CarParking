import { ArrowLeft, Check, ClipboardCopy, Download, FlaskConical, Hexagon, ImageIcon, Lock, MousePointer2, RectangleHorizontal, RefreshCw, Save, Trash2, Undo2, Upload, Video, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, withCam } from "../api";
import { useWebRtc } from "../hooks";
import { type DrawnSlot, type Pt, addSlot, mapProblems, moveSlot, moveVertex, nearFirst, pt, rectPoly, setCapacity, toJson, fromJson } from "../slotdraw";
import { CapacityField } from "./CapacityField";
import type { CameraInfo, SettingsResponse, SlotState, Snapshot, StateMessage } from "../types";
import { AutoGridPins } from "./AutoGridPins";
import { SuggestSlots } from "./SuggestSlots";
import { BayAdjust, useBayHistory } from "./BayAdjust";
import { Button, Card, CanvasFrame, IconButton, Pill, inputClass } from "./ui";

type Mode = "polygon" | "rect" | "edit";
type Drag = { kind: "vertex"; id: number; index: number } | { kind: "slot"; id: number; last: Pt } | { kind: "rect"; start: Pt };

interface Saved {
  camera: string;
  updated: string | null;
  slots: DrawnSlot[];
  cameras: string[];
  in_use: boolean;
  mode?: "dynamic" | "fixed"; // the camera's slot mapping mode: slots are drawn / edited only in Fixed mode
  editable?: boolean;
  locked?: string | null; // why not (Dynamic mode)
}

/** Test slots: the server's read-only answer for the drawn slots (POST /api/fixed-slots/preview). */
interface PreviewResult {
  id: number;
  name: string;
  state: "FREE" | "BUSY";
  score: number;
  vehicle: { label: string; conf: number; box: [number, number, number, number] } | null;
}
interface Preview {
  scope: "all" | "selected";
  results: PreviewResult[];
  vehicles: [number, number, number, number][];
  frame_age: number | null;
  rule: string;
  source?: "worker" | "on-demand" | "snapshot"; // the worker's live detections, a frame grabbed for this test, or the editor's snapshot
  image?: string | null; // the frame that was tested (on-demand): shown under the result
}

/** The editor's frozen still (POST /api/fixed-slots/snapshot): the canvas for drawing and Test slots. */
export interface Still {
  id: string;
  ts: number;
  source: "camera" | "go2rtc" | "worker";
  width: number;
  height: number;
  image: string; // data URL
}
const STILL_SOURCE: Record<Still["source"], string> = { camera: "camera snapshot", go2rtc: "video service", worker: "worker's last frame" };

export const DYNAMIC_LOCKED = "Dynamic Mode automatically detects parking spaces. Switch to Fixed Mode to manually draw or edit slots.";
const STATE_COLOR: Record<SlotState, string> = { FREE: "#2eb85c", BUSY: "#e55353", UNKNOWN: "#f9b115" };

/** Fixed coordinates drawing tool: draw parking slots with the mouse on a camera frame (or the live video), check
 * the coordinates as JSON, export / import them, and save them as the camera's fixed slot map (the same file the
 * Streamlit editor writes; the worker uses it when the slot mapping mode is Fixed). In Dynamic mode it is read-only:
 * the AI finds the slots itself, so the page shows them (live, FREE / BUSY) and the drawing tools are gone; the API
 * refuses a save too (409). */
export function SlotEditor({ snapshot, tokenKey, onNeedAdmin, live, cam, cameras, onCam }: {
  cam: string | null; // the registry camera being edited (the selected camera)
  cameras: CameraInfo[];
  onCam: (id: string) => void;
  snapshot: Snapshot | undefined;
  tokenKey: string;
  onNeedAdmin: (reason: string, retry?: () => void) => void; // retry: run again once the admin token is accepted
  live: StateMessage | null;
}) {
  const [camera, setCamera] = useState("");
  const [saved, setSaved] = useState<Saved | null>(null);
  const [slots, setSlots] = useState<DrawnSlot[]>([]);
  const [mode, setMode] = useState<Mode>("polygon");
  const [draft, setDraft] = useState<Pt[]>([]);
  const [cursor, setCursor] = useState<Pt | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [background, setBackground] = useState<"frame" | "video">("frame");
  const [still, setStill] = useState<Still | null>(null); // the frozen snapshot under the drawing
  const [stillState, setStillState] = useState<"loading" | "ok" | "error">("loading");
  const [stillError, setStillError] = useState("");
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [importText, setImportText] = useState<string | null>(null);
  const [pixels, setPixels] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null); // Test slots: read-only FREE / BUSY overlay
  const previewOn = useRef(false); // the keyboard handler reads it, so Esc works on the very first key press
  previewOn.current = preview != null;
  const [testing, setTesting] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const [slotMode, setSlotMode] = useState<"dynamic" | "fixed" | null>(null); // from the Parking mode card (settings)
  const camInfo = cameras.find((c) => c.id === cam);
  const camId = cam ?? cameras.find((c) => c.primary)?.id ?? null; // (no cam = the primary camera; the suggest endpoints want its id)
  const status = useWebRtc(video, camInfo?.streams.hd ?? "parking_hd", background === "video", tokenKey);

  // the camera's native frame (pixel export): the worker's, else the snapshot's
  const size: [number, number] = snapshot?.shape ?? (still ? [still.width, still.height] : [2560, 1440]);
  const [w, h] = size;
  const history = useBayHistory(slots, setSlots); // Undo / Redo of the edits made here (drag, panel, split, delete), 50 deep

  const load = useCallback(() => {
    api.get<Saved>(withCam("/api/fixed-slots", cam)).then((r) => {
      if (r.mode) setSlotMode(r.mode);
      setSaved(r);
      setCamera(r.camera);
      setSlots(r.slots);
      history.clear();
      setDraft([]);
      setSelected(null);
    }).catch((e) => setMessage({ tone: "error", text: e.message }));
  }, [cam]);
  useEffect(() => load(), [load, tokenKey]);
  useEffect(() => setSlotMode(null), [cam]); // another camera: its own mode

  // Snapshot: one still captured when the editor opens (and on a camera change / Refresh snapshot), frozen as the
  // canvas; Test slots runs on it, so neither drawing nor testing depends on the video stream
  const stillSeq = useRef(0);
  const takeStill = useCallback(async (): Promise<Still | null> => {
    const seq = ++stillSeq.current;
    setStillState("loading");
    setPreview(null);
    try {
      const r = await api.query<Still>(withCam("/api/fixed-slots/snapshot", cam), {});
      if (seq !== stillSeq.current) return null; // a newer request (another camera) won
      setStill(r);
      setStillState("ok");
      return r;
    } catch (e) {
      if (seq !== stillSeq.current) return null;
      setStillError((e as Error).message);
      setStillState("error");
      return null;
    }
  }, [cam]);
  useEffect(() => {
    setStill(null);
    void takeStill();
  }, [takeStill, tokenKey]);

  const locked = (slotMode ?? saved?.mode) === "dynamic";
  const lockText = saved?.locked || DYNAMIC_LOCKED;
  // Dynamic mode: nothing half-drawn survives, the list shows the saved map
  useEffect(() => {
    if (!locked) return;
    setDraft([]);
    setDrag(null);
    setSelected(null);
    setImportText(null);
    if (saved) setSlots(saved.slots);
  }, [locked, saved]);
  const detected = useMemo(() => (snapshot?.slots ?? []).filter((s) => !s.hidden && !s.phantom && s.duplicate_of == null), [snapshot]);

  const dirty = useMemo(() => JSON.stringify(slots) !== JSON.stringify(saved?.slots ?? []), [slots, saved]);
  const problems = useMemo(() => mapProblems(slots), [slots]);
  const json = useMemo(() => toJson(camera, slots, size), [camera, slots, w, h]);

  // mouse position -> shares of the frame
  const at = (e: { clientX: number; clientY: number }): Pt => {
    const r = box.current!.getBoundingClientRect();
    return pt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
  };
  const screen = (): [number, number] => {
    const r = box.current?.getBoundingClientRect();
    return r ? [r.width, r.height] : [w, h];
  };

  const finish = (poly: Pt[]) => {
    const r = addSlot(slots, poly);
    if (r.error) setMessage({ tone: "error", text: `Not added: ${r.error}.` });
    else {
      history.push();
      setSlots(r.slots);
      setSelected(r.slots[r.slots.length - 1].id);
      setMessage(null);
    }
    setDraft([]);
  };

  const onDown = (e: React.MouseEvent) => {
    if (e.button !== 0 || locked || preview) return;
    const p = at(e);
    if (mode === "rect") setDrag({ kind: "rect", start: p });
    if (mode === "polygon") {
      if (nearFirst(draft, p, screen())) finish(draft);
      else setDraft((d) => [...d, p]);
    }
  };
  const onMove = (e: React.MouseEvent) => {
    const p = at(e);
    setCursor(p);
    if (locked || preview) return;
    if (drag?.kind === "vertex") setSlots((s) => moveVertex(s, drag.id, drag.index, p));
    if (drag?.kind === "slot") {
      setSlots((s) => moveSlot(s, drag.id, p[0] - drag.last[0], p[1] - drag.last[1]));
      setDrag({ ...drag, last: p });
    }
  };
  const onUp = (e: React.MouseEvent) => {
    if (drag?.kind === "rect") {
      const end = at(e);
      if (Math.abs(end[0] - drag.start[0]) * screen()[0] > 4 && Math.abs(end[1] - drag.start[1]) * screen()[1] > 4) {
        finish(rectPoly(drag.start, end));
      }
    }
    setDrag(null);
  };

  // keyboard: Enter finishes, Esc cancels, Backspace / Ctrl+Z undoes the last corner, Delete removes the selection
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (previewOn.current && e.key === "Escape") return setPreview(null); // Esc leaves the preview
      if (locked || previewOn.current || (e.target as HTMLElement)?.closest("input, textarea, select")) return;
      if (e.key === "Enter" && draft.length >= 3) finish(draft);
      else if (e.key === "Escape") setDraft([]);
      else if ((e.key === "Backspace" || (e.key === "z" && (e.ctrlKey || e.metaKey))) && draft.length) {
        e.preventDefault();
        setDraft((d) => d.slice(0, -1));
      } else if (e.key === "Delete" && selected != null) {
        history.push();
        setSlots((s) => s.filter((x) => x.id !== selected));
        setSelected(null);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });

  // Test slots (read-only): the drawn slots, saved or not, against the vehicles detected on the snapshot (the canvas).
  // Nothing is saved or changed; Back to edit (or Esc) returns to the editor exactly as it was.
  const runTest = async (scope: "all" | "selected") => {
    if (problems.length) return setMessage({ tone: "error", text: `Can't test: ${problems[0]}` });
    if (scope === "selected" && selected == null) return;
    setTesting(true);
    setDraft([]);
    setDrag(null);
    setBackground("frame"); // the result is drawn over the image it was found on
    const ask = (id: string | undefined) => api.query<Omit<Preview, "scope">>(withCam("/api/fixed-slots/preview", cam),
      { slots, only: scope === "selected" ? [selected] : undefined, snapshot: id });
    try {
      let r: Omit<Preview, "scope">;
      try {
        r = await ask(still?.id);
      } catch (e) {
        if ((e as ApiError).status !== 410) throw e;
        const fresh = await takeStill(); // the server no longer has it (restarted): a new still, then test that
        r = await ask(fresh?.id);
      }
      setPreview({ ...r, scope });
      setMessage(null);
    } catch (e) {
      setMessage({ tone: "error", text: `Test failed: ${(e as Error).message}` });
    } finally {
      setTesting(false);
    }
  };
  // an edit in the slot list (rename, delete) makes the answer stale: back to edit
  useEffect(() => setPreview(null), [slots, cam]);
  const tested = useMemo(() => new Map((preview?.results ?? []).map((r) => [r.id, r])), [preview]);

  const save = async () => {
    if (locked) return;
    try {
      const r = await api.put<{ slots: DrawnSlot[]; settings_version: number }>(withCam("/api/fixed-slots", cam), { camera, slots });
      setMessage({ tone: "ok", text: `Saved ${r.slots.length} slot(s) for "${camera}".${saved?.in_use ? " The worker uses them now." : " Used when Settings > Slot mapping mode = Fixed."}` });
      load();
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 401 || err.status === 503) onNeedAdmin("Saving the slot map needs the admin token.", save);
      else if (err.status === 409) (setMessage({ tone: "error", text: err.message }), load()); // switched to Dynamic meanwhile
      else setMessage({ tone: "error", text: `Not saved: ${err.message}` });
    }
  };

  const download = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    a.download = `fixed_slots_${camera.replace(/[^\w-]+/g, "_").toLowerCase() || "camera"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const copy = () => navigator.clipboard?.writeText(json).then(() => setMessage({ tone: "ok", text: "JSON copied." }),
    () => setMessage({ tone: "error", text: "The browser blocked the clipboard: select the JSON and copy it." }));
  const doImport = (text: string) => {
    if (locked) return;
    try {
      const got = fromJson(text, size);
      history.push();
      setSlots(got);
      setImportText(null);
      setMessage({ tone: "ok", text: `Imported ${got.length} slot(s): check them on the image, then Save.` });
    } catch (e) {
      setMessage({ tone: "error", text: `Import failed: ${(e as Error).message}` });
    }
  };

  const stillTime = still ? new Date(still.ts * 1000).toLocaleTimeString() : "";
  const sw = (n: number) => n; // stroke widths in screen px (non-scaling)
  const pts = (poly: Pt[]) => poly.map(([x, y]) => `${x * w},${y * h}`).join(" ");
  const rect = drag?.kind === "rect" && cursor ? rectPoly(drag.start, cursor) : null;

  return (
    <div className="grid gap-4 xl:grid-cols-10">
      <CanvasFrame className="xl:col-span-7"
            title={locked ? "Slot map · read-only (Dynamic mode)" : "Fixed coordinates drawing tool"}
            actions={
              <>
                {!locked && <>
                <Pill active={mode === "polygon"} onClick={() => (setMode("polygon"), setDraft([]))} title="Click each corner; click the first one (or Enter) to finish">
                  <Hexagon size={12} /> Polygon
                </Pill>
                <Pill active={mode === "rect"} onClick={() => (setMode("rect"), setDraft([]))} title="Drag a rectangle">
                  <RectangleHorizontal size={12} /> Rectangle
                </Pill>
                <Pill active={mode === "edit"} onClick={() => (setMode("edit"), setDraft([]))} title="Drag corners or whole slots; click a slot to select it">
                  <MousePointer2 size={12} /> Edit
                </Pill>
                <span className="mx-1 h-4 w-px bg-line" />
                </>}
                <Pill active={background === "frame"} onClick={() => setBackground("frame")} title="The snapshot: a frozen camera still (drawing and Test slots use it)">
                  <ImageIcon size={12} /> Snapshot
                </Pill>
                <Pill active={background === "video"} onClick={() => setBackground("video")} title="The live HD video (for reference; Test slots still uses the snapshot)">
                  <Video size={12} /> Live video
                </Pill>
                <Button variant="outline" onClick={() => void takeStill()} disabled={stillState === "loading" || testing}
                        title="Capture a new still from the camera now">
                  <RefreshCw size={14} className={stillState === "loading" ? "animate-spin" : ""} /> Refresh snapshot
                </Button>
              </>
            }>
        <div ref={box} className="relative w-full select-none bg-stage" style={{ aspectRatio: `${w} / ${h}` }} data-testid="draw-surface">
          {preview?.image ? (
            <img src={preview.image} alt="The frame that was tested" className="absolute inset-0 h-full w-full object-fill"
                 draggable={false} data-testid="preview-image" />
          ) : background === "frame" ? (
            still ? (
              <img src={still.image} alt="Camera snapshot" className="absolute inset-0 h-full w-full object-fill" draggable={false}
                   data-testid="still-image" />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-sm text-white/70" data-testid="still-status">
                {stillState === "error" ? (
                  <>
                    <span>No snapshot: {stillError}</span>
                    <button type="button" onClick={() => void takeStill()}
                            className="inline-flex items-center gap-1 rounded bg-white px-3 py-1 text-xs font-semibold text-ink hover:bg-white/90">
                      <RefreshCw size={12} /> Try again
                    </button>
                  </>
                ) : "Capturing a snapshot from the camera…"}
              </div>
            )
          ) : (
            <video ref={video} muted autoPlay playsInline className="absolute inset-0 h-full w-full object-fill" />
          )}
          <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className={`absolute inset-0 h-full w-full ${locked || mode === "edit" ? "cursor-default" : "cursor-crosshair"}`}
               onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={() => (setCursor(null), setDrag(null))}
               onDoubleClick={() => !locked && mode === "polygon" && draft.length >= 3 && finish(draft)} data-testid="draw-svg">
            {locked && detected.map((s) => {
              const [lx, ly] = s.poly.reduce(([x, y], [px, py]) => [Math.min(x, px), Math.min(y, py)], [1, 1]);
              return (
                <g key={s.id} pointerEvents="none" data-testid="detected-slot">
                  <polygon points={pts(s.poly)} fill={`${STATE_COLOR[s.state]}38`} stroke={STATE_COLOR[s.state]}
                           strokeWidth={sw(2)} vectorEffect="non-scaling-stroke" />
                  <text x={lx * w + w / 250} y={ly * h + w / 70} fontSize={w / 70} fontWeight={700} fill="white"
                        stroke="black" strokeWidth={w / 900} paintOrder="stroke">{s.name} · {s.state}</text>
                </g>
              );
            })}
            {!locked && preview && slots.map((s) => {
              const r = tested.get(s.id);
              const [lx, ly] = s.poly.reduce(([x, y], [px, py]) => [Math.min(x, px), Math.min(y, py)], [1, 1]);
              const color = r ? STATE_COLOR[r.state] : "#8a93a2";
              return (
                <g key={s.id} pointerEvents="none" opacity={r ? 1 : 0.35} data-testid="preview-slot" data-state={r?.state ?? "untested"}>
                  <polygon points={pts(s.poly)} fill={r ? `${color}55` : "transparent"} stroke={color} strokeWidth={sw(r ? 3 : 1.5)}
                           vectorEffect="non-scaling-stroke" strokeDasharray={r ? undefined : "6 4"} />
                  <text x={lx * w + w / 250} y={ly * h + w / 70} fontSize={w / 70} fontWeight={700} fill="white"
                        stroke="black" strokeWidth={w / 900} paintOrder="stroke">{s.name}{r ? ` · ${r.state}` : ""}</text>
                </g>
              );
            })}
            {!locked && preview && preview.vehicles.map(([x1, y1, x2, y2], i) => (
              <rect key={`v${i}`} x={x1 * w} y={y1 * h} width={(x2 - x1) * w} height={(y2 - y1) * h} fill="none"
                    stroke="#ffd600" strokeWidth={sw(1.5)} vectorEffect="non-scaling-stroke" strokeDasharray="4 3"
                    pointerEvents="none" data-testid="preview-vehicle" />
            ))}
            {!locked && !preview && slots.map((s) => {
              const sel = s.id === selected;
              const [lx, ly] = s.poly.reduce(([x, y], [px, py]) => [Math.min(x, px), Math.min(y, py)], [1, 1]);
              return (
                <g key={s.id}>
                  <polygon points={pts(s.poly)} fill={sel ? "rgba(249,177,21,0.28)" : "rgba(46,184,92,0.22)"}
                           stroke={sel ? "#f9b115" : "#2eb85c"} strokeWidth={sw(2)} vectorEffect="non-scaling-stroke"
                           className={mode === "edit" ? "cursor-move" : ""}
                           onMouseDown={(e) => {
                             if (mode !== "edit") return;
                             e.stopPropagation();
                             setSelected(s.id);
                             history.push();
                             setDrag({ kind: "slot", id: s.id, last: at(e) });
                           }} />
                  <text x={lx * w + w / 250} y={ly * h + w / 70} fontSize={w / 70} fontWeight={700} fill="white"
                        stroke="black" strokeWidth={w / 900} paintOrder="stroke" pointerEvents="none">{s.name}</text>
                  {mode === "edit" && s.poly.map(([x, y], i) => (
                    <circle key={i} cx={x * w} cy={y * h} r={w / 220} fill="white" stroke={sel ? "#f9b115" : "#2eb85c"}
                            strokeWidth={sw(2)} vectorEffect="non-scaling-stroke" className="cursor-grab"
                            onMouseDown={(e) => {
                              e.stopPropagation();
                              setSelected(s.id);
                              history.push();
                              setDrag({ kind: "vertex", id: s.id, index: i });
                            }} />
                  ))}
                </g>
              );
            })}
            {draft.length > 0 && (
              <g pointerEvents="none">
                <polyline points={pts(cursor ? [...draft, cursor] : draft)} fill="rgba(88,86,214,0.15)" stroke="#5856d6"
                          strokeWidth={sw(2)} vectorEffect="non-scaling-stroke" strokeDasharray="6 4" />
                {draft.map(([x, y], i) => (
                  <circle key={i} cx={x * w} cy={y * h} r={i === 0 && draft.length >= 3 ? w / 160 : w / 260}
                          fill={i === 0 ? "#5856d6" : "white"} stroke="#5856d6" strokeWidth={sw(2)} vectorEffect="non-scaling-stroke" />
                ))}
              </g>
            )}
            {rect && <polygon points={pts(rect)} fill="rgba(88,86,214,0.15)" stroke="#5856d6" strokeWidth={sw(2)}
                              vectorEffect="non-scaling-stroke" strokeDasharray="6 4" pointerEvents="none" />}
          </svg>
          {preview && (
            <div className="absolute inset-x-0 top-0 flex justify-center p-3">
              <div className="flex max-w-2xl flex-wrap items-center gap-2 rounded-md bg-stage/85 px-3 py-2 text-sm text-white shadow" data-testid="preview-banner">
                <FlaskConical size={15} className="shrink-0 text-warning" />
                <span className="font-semibold">Preview mode – active</span>
                <span className="text-white/80">
                  {preview.scope === "all" ? "all slots" : `slot ${preview.results[0]?.name ?? ""}`}:
                  {" "}{preview.results.filter((r) => r.state === "FREE").length} FREE · {preview.results.filter((r) => r.state === "BUSY").length} BUSY
                  {preview.source === "snapshot" ? ` · snapshot of ${stillTime}` : preview.source === "on-demand" ? " · frame grabbed now" : preview.frame_age != null ? ` · live monitoring, ${Math.max(0, preview.frame_age).toFixed(0)} s ago` : ""} · rule {preview.rule}
                </span>
                <button type="button" onClick={() => setPreview(null)} data-testid="preview-back"
                        className="ml-1 inline-flex items-center gap-1 rounded bg-white px-2 py-0.5 text-xs font-semibold text-ink hover:bg-white/90">
                  <ArrowLeft size={12} /> Back to edit
                </button>
              </div>
            </div>
          )}
          {locked && (
            <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-3">
              <p className="flex max-w-xl items-center gap-2 rounded-md bg-stage/85 px-3 py-2 text-sm text-white shadow" data-testid="dynamic-lock">
                <Lock size={15} className="shrink-0 text-warning" /> {lockText}
              </p>
            </div>
          )}
          {cursor && (
            <span className="pointer-events-none absolute right-2 bottom-2 rounded bg-stage/75 px-2 py-0.5 font-mono text-[11px] text-white"
                  data-testid="cursor-readout">
              x {cursor[0].toFixed(4)} · y {cursor[1].toFixed(4)} · {Math.round(cursor[0] * w)}, {Math.round(cursor[1] * h)} px
            </span>
          )}
          {background === "video" && status !== "playing" && (
            <span className="absolute top-2 left-2 rounded bg-stage/75 px-2 py-0.5 text-[11px] text-white">Video: {status}</span>
          )}
          {background === "frame" && still && !preview && (
            <span className="pointer-events-none absolute top-2 left-2 rounded bg-stage/75 px-2 py-0.5 text-[11px] text-white" data-testid="still-info">
              {stillState === "loading" ? "Refreshing snapshot…" : `Snapshot ${stillTime} · ${STILL_SOURCE[still.source]}`}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-4 py-2.5 text-xs text-muted">
          {locked && (
            <span data-testid="detected-summary">
              Read-only. The AI finds the slots from where vehicles stand
              {snapshot?.counts ? `: ${snapshot.counts.slots} slot(s) now · ${snapshot.counts.free} FREE · ${snapshot.counts.busy} BUSY · ${snapshot.counts.unknown} UNKNOWN.` : " (none yet: switch Monitoring on)."}
            </span>
          )}
          {preview && <span data-testid="preview-hint">Preview only: nothing is saved or changed. The vehicles found on the {preview.source === "snapshot" ? "snapshot" : "frame"} (dashed yellow) against the drawn slots, with this camera's match rule, no hold time. Back to edit or Esc returns to editing.</span>}
          {!preview && !locked && mode === "polygon" && <span>Click the corners of a slot. Finish: click the first corner, double-click, or Enter. Esc cancels, Backspace / Ctrl+Z removes the last corner.</span>}
          {!preview && !locked && mode === "rect" && <span>Drag a rectangle over a slot.</span>}
          {!preview && !locked && mode === "edit" && <span>Drag a corner to reshape, drag a slot to move it. Delete removes the selected slot.</span>}
          {draft.length > 0 && (
            <>
              <span className="font-medium text-ink">{draft.length} corner(s)</span>
              <Button variant="outline" onClick={() => setDraft((d) => d.slice(0, -1))}><Undo2 size={13} /> Undo</Button>
              <Button variant="outline" disabled={draft.length < 3} onClick={() => finish(draft)}><Check size={13} /> Finish slot</Button>
              <Button variant="ghost" onClick={() => setDraft([])}><X size={13} /> Cancel</Button>
            </>
          )}
        </div>
      </CanvasFrame>

      <div className="space-y-4 xl:col-span-3">
        <ParkingMode tokenKey={tokenKey} live={live} onNeedAdmin={onNeedAdmin} savedVersion={saved?.updated ?? ""} cam={cam}
                     dirty={dirty} onMode={setSlotMode} />
        <Card title={locked ? "Slots (read-only)" : "Slots"}
              actions={<span className="text-xs text-muted">{locked ? `${detected.length} detected` : `${slots.length} drawn · ${dirty ? "unsaved" : "saved"}`}</span>}>
          <label className="mb-3 block">
            <span className="mb-1 block text-xs font-semibold tracking-wide text-muted uppercase">Camera</span>
            <select className={inputClass} value={cam ?? ""} data-testid="editor-camera"
                    onChange={(e) => (dirty ? setMessage({ tone: "error", text: "Save or Revert the changes first." }) : onCam(e.target.value))}>
              {cameras.map((c) => <option key={c.id} value={c.id}>{c.display}</option>)}
            </select>
          </label>
          {locked && (
            <p className="mb-3 flex gap-2 rounded-md bg-warning/15 px-3 py-2 text-xs text-ink" data-testid="dynamic-lock-note">
              <Lock size={14} className="shrink-0" /> {lockText}
            </p>
          )}
          {locked ? (
            <ul className="max-h-64 divide-y divide-line overflow-y-auto rounded-md border border-line" data-testid="detected-list">
              {detected.map((s) => (
                <li key={s.id} className="flex items-center gap-2 px-2.5 py-1.5 text-sm">
                  <span className="w-6 text-xs font-semibold text-muted">{s.id}</span>
                  <span className="flex-1">{s.name}</span>
                  <span className="rounded-full px-2 py-0.5 text-xs font-semibold text-white" style={{ background: STATE_COLOR[s.state] }}>{s.state}</span>
                </li>
              ))}
              {!detected.length && <li className="px-3 py-4 text-center text-sm text-muted">No slots detected yet: the AI learns them from parked vehicles while Monitoring is on.</li>}
            </ul>
          ) : (<>
          {saved && (
            <p className={`mb-3 rounded-md px-3 py-2 text-xs ${saved.in_use ? "bg-success/10 text-success-ink" : "bg-body text-muted"}`}>
              {saved.in_use ? "The worker runs this camera in Fixed mode: a save applies at once."
                : "Saved maps are used when Settings › Advanced › Slot mapping mode = Fixed."}
              {saved.updated && <> Last saved {saved.updated.replace("T", " ").slice(0, 19)}.</>}
            </p>
          )}
          <ul className="max-h-64 divide-y divide-line overflow-y-auto rounded-md border border-line" data-testid="slot-list">
            {slots.map((s) => (
              <li key={s.id} className={`flex items-center gap-2 px-2.5 py-1.5 ${s.id === selected ? "bg-warning/10" : ""}`}
                  onClick={() => setSelected(s.id)}>
                <span className="w-6 text-xs font-semibold text-muted">{s.id}</span>
                <input className="min-w-0 flex-1 rounded border border-transparent bg-transparent px-1 py-0.5 text-sm hover:border-line focus:border-primary focus:outline-none"
                       value={s.name} onChange={(e) => setSlots((all) => all.map((x) => (x.id === s.id ? { ...x, name: e.target.value } : x)))} />
                <CapacityField value={s.capacity} onChange={(t) => setSlots((all) => setCapacity(all, s.id, t))} />
                <span className="text-xs text-muted">{s.poly.length} pts</span>
                <IconButton icon={Trash2} label="Delete" size="sm" variant="danger"
                            onClick={(e) => (e.stopPropagation(), history.push(), setSlots((all) => all.filter((x) => x.id !== s.id)))} />
              </li>
            ))}
            {!slots.length && <li className="px-3 py-4 text-center text-sm text-muted">No slots: draw one on the image.</li>}
          </ul>
          {problems.length > 0 && <p className="mt-2 text-xs text-danger">{problems[0]}</p>}
          {!locked && (slots.length > 0 || history.undoN > 0) && (
            <div className="mt-3"><BayAdjust slots={slots} selected={selected} group={{ ids: slots.map((s) => s.id), label: "Whole map" }} w={w} h={h} onChange={setSlots} history={history} ns="slot"
                                  active={!preview && draft.length === 0} title="Edit slots: adjustment" /></div>
          )}
          </>)}
          {message && (
            <p className={`mt-3 rounded-md px-3 py-2 text-sm ${message.tone === "ok" ? "bg-success/10 text-success-ink" : "bg-danger/10 text-danger"}`}>{message.text}</p>
          )}
          {testing && (
            <p className="mt-3 rounded-md bg-body px-3 py-2 text-xs text-muted" data-testid="test-busy">
              Detecting vehicles on the snapshot… No live video or monitoring needed. The first test of a snapshot loads the
              model and can take up to ~20 s; the next tests of the same snapshot are instant.
            </p>
          )}
          {!locked && <div className="mt-3 flex flex-wrap gap-2" data-testid="test-actions">
            {preview ? (
              <Button variant="outline" onClick={() => setPreview(null)}><ArrowLeft size={14} /> Back to edit</Button>
            ) : (
              <>
                <Button variant="outline" onClick={() => runTest("all")} disabled={testing || stillState === "loading" || !slots.length || problems.length > 0}
                        title="Preview FREE / BUSY of every drawn slot on the snapshot (nothing is saved)">
                  <FlaskConical size={14} /> {testing ? "Detecting…" : "Test all slots"}
                </Button>
                <Button variant="outline" onClick={() => runTest("selected")} disabled={testing || stillState === "loading" || selected == null || problems.length > 0}
                        title={selected == null ? "Select a slot first (click it on the image or in the list)" : "Preview only the selected slot"}>
                  <FlaskConical size={14} /> Test selected slot
                </Button>
              </>
            )}
          </div>}
          {!locked && <div className="mt-3 flex flex-wrap gap-2">
            <Button onClick={save} disabled={!dirty || problems.length > 0 || !camera.trim()}><Save size={14} /> Save to server</Button>
            <Button variant="ghost" disabled={!dirty} onClick={() => load()}>Revert</Button>
            <Button variant="ghost" disabled={!slots.length} onClick={() => (history.push(), setSlots([]), setSelected(null))}>Clear all</Button>
          </div>}
        </Card>

        <Card title={locked ? "Saved fixed map (JSON)" : "Coordinates (JSON)"}
              actions={
                <>
                  <Pill active={pixels} onClick={() => setPixels((p) => !p)} title="Show the corners in camera pixels">px</Pill>
                  <Button variant="ghost" onClick={copy} title="Copy JSON"><ClipboardCopy size={14} /></Button>
                  <Button variant="ghost" onClick={download} title="Download JSON"><Download size={14} /></Button>
                  {!locked && <Button variant="ghost" onClick={() => setImportText((t) => (t == null ? "" : null))} title="Import JSON"><Upload size={14} /></Button>}
                </>
              }>
          {importText != null ? (
            <div className="space-y-2">
              <textarea className={`${inputClass} h-48 font-mono text-xs`} placeholder='Paste an export, or {"slots": [...]} from config/fixed_slots.json'
                        value={importText} onChange={(e) => setImportText(e.target.value)} />
              <div className="flex items-center gap-2">
                <Button onClick={() => doImport(importText)} disabled={!importText.trim()}>Import</Button>
                <label className="cursor-pointer text-xs font-medium text-primary">
                  or a file…
                  <input type="file" accept="application/json,.json" className="hidden"
                         onChange={(e) => e.target.files?.[0]?.text().then(doImport)} />
                </label>
              </div>
            </div>
          ) : (
            <pre className="max-h-80 overflow-auto rounded-md bg-stage p-3 text-[11px] leading-relaxed text-white/90" data-testid="json-preview">
              {pixels
                ? JSON.stringify(JSON.parse(json).slots.map((s: { id: number; name: string; points_px: number[][] }) => ({ id: s.id, name: s.name, points_px: s.points_px })), null, 2)
                : JSON.stringify({ camera, slots }, null, 2)}
            </pre>
          )}
          <p className="mt-2 text-xs text-muted">
            Corners are shares of the frame (0–1), the format of config/fixed_slots.json; px = the camera's {w}×{h} frame.
          </p>
        </Card>
      </div>
      {locked && snapshot?.auto_slots && (
        <div className="xl:col-span-10" data-testid="auto-grid-pins">
          <AutoGridPins cam={cam} snapshot={snapshot} image={still?.image ?? null} tokenKey={tokenKey} onNeedAdmin={onNeedAdmin} />
        </div>
      )}
      <div className="xl:col-span-10" data-testid="suggest-slots">
        <SuggestSlots camId={camId} cam={cam} snapshot={snapshot} image={still?.image ?? null} stillId={still?.id ?? null}
                      tokenKey={tokenKey} onNeedAdmin={onNeedAdmin} />
      </div>
    </div>
  );
}

/** Fixed or Dynamic for the camera the worker runs (the shared "Slot mapping mode" setting, the same one as in
 * Streamlit and the Settings drawer). Fixed needs a saved slot map: without one the worker would idle. */
export function ParkingMode({ tokenKey, live, onNeedAdmin, savedVersion, cam, dirty = false, onMode }: {
  cam: string | null;
  dirty?: boolean; // unsaved drawing in the editor: not thrown away by a switch to Dynamic
  onMode?: (mode: "dynamic" | "fixed") => void; // the editor locks / unlocks with it
  tokenKey: string;
  live: StateMessage | null;
  onNeedAdmin: (reason: string, retry?: () => void) => void; // retry: run again once the admin token is accepted
  savedVersion: string;
}) {
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [fixedCount, setFixedCount] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const version = live?.settings_version;

  useEffect(() => {
    api.get<SettingsResponse>(withCam("/api/settings", cam)).then((s) => {
      setSettings(s);
      return api.get<{ slots: unknown[] }>(withCam("/api/fixed-slots", cam));
    }).then((f) => setFixedCount(f.slots.length)).catch(() => {});
  }, [tokenKey, version, savedVersion, cam]);

  const current = settings?.values.slot_mode;
  useEffect(() => {
    if (current) onMode?.(current);
  }, [current, onMode]);

  if (!settings) return null;
  const { camera, slot_mode: mode, enabled } = settings.values;
  // Fixed is allowed before a map exists (drawing is only possible in Fixed mode; the worker waits for a saved map)
  const choose = async (next: "dynamic" | "fixed") => {
    if (next === mode) return;
    if (next === "dynamic" && dirty) {
      setNote({ tone: "error", text: "Save or Revert the drawn slots first: in Dynamic mode the editor is read-only." });
      return;
    }
    setBusy(true);
    setNote(null);
    try {
      const r = await api.put<SettingsResponse>(withCam("/api/settings", cam), { slot_mode: next });
      setSettings(r);
      setNote({ tone: "ok", text: next === "fixed"
        ? fixedCount ? `Fixed: the worker now watches the ${fixedCount} drawn slot(s). Drawing is unlocked.`
          : "Fixed: drawing is unlocked. Draw the slots and Save: the worker waits for a saved map."
        : "Dynamic: the worker learns slots from where vehicles stand. The editor is read-only." });
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 401 || err.status === 503) onNeedAdmin("Changing the parking mode needs the admin token.", () => choose(next));
      else setNote({ tone: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  };
  const option = (value: "dynamic" | "fixed", title: string, text: string) => (
    <button type="button" disabled={busy} onClick={() => choose(value)} data-testid={`mode-${value}`}
            className={`flex-1 rounded-md border px-3 py-2 text-left transition-colors ${mode === value
              ? "border-primary bg-primary/10 ring-1 ring-primary" : "border-line bg-surface hover:border-primary"}`}>
      <span className={`block text-sm font-semibold ${mode === value ? "text-primary" : "text-ink"}`}>{title}</span>
      <span className="block text-xs text-muted">{text}</span>
    </button>
  );
  return (
    <Card title="Parking mode" actions={<span className="text-xs text-muted">{camera}</span>}>
      <div className="flex gap-2" data-testid="parking-mode">
        {option("dynamic", "Dynamic", "learned from where vehicles stand")}
        {option("fixed", "Fixed", `the drawn slots (${fixedCount ?? 0} saved)`)}
      </div>
      <p className="mt-2 text-xs text-muted">
        This camera only; each camera has its own mode. Shared with Settings.
        {!enabled && " Monitoring is off: turn it on to see FREE / BUSY."}
      </p>
      {note && (
        <p className={`mt-2 rounded-md px-3 py-2 text-xs ${note.tone === "ok" ? "bg-success/10 text-success-ink" : "bg-danger/10 text-danger"}`}>{note.text}</p>
      )}
    </Card>
  );
}
