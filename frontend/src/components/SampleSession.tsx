import { ChevronDown, ChevronRight, Crosshair, FlaskConical, Hand, ImageUp, Loader2, RefreshCw, SlidersHorizontal, Sparkles, Trash2, ZoomIn, ZoomOut } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { type ApiError, api } from "../api";
import { points } from "../lib";
import { resizeBay, rotateBay, shiftAlong, shiftBays, splitBay, tuneBays } from "../bayfine";
import { type DrawnSlot, type Pt, area, moveSlot, moveVertex, pt } from "../slotdraw";
import { type Expected, ExpectedInputs, type Layer, LayerBoxes, MAX_BYTES, ResultPanel, type SampleResult, readFile } from "./SampleResult";
import { DirectSlotCanvas } from "./DirectSlotCanvas";
import { Button, Card, Pill } from "./ui";
import { MAX_ZOOM, MIN_ZOOM, useZoomPan } from "./useZoomPan";

/** The Sample test page. A parking picture from anywhere is its own throw-away camera: (1) the scale from a few clicks, (2) the parking surface, (3) a bay grid laid on it, which you
 * review and fix, (4) the picture through the detector, the protections and these bays. Nothing is saved anywhere: the server keeps the session in
 * memory for half an hour. */
interface Calibration {
  method: "rectangle" | "geocalib";
  confidence: string;
  h: number;
  f: number;
  hfov_deg: number | null;
  area_m2: number;
  warnings: string[];
  pins_fit: { reprojection_px: number } | null;
  sizing?: { width_m: number | null; length_m: number | null; source: string; bays_along: number; rows: number }; // a zone's bays: their size and where it came from
  bays: number;
}
interface SessionJob { id: string; state: "queued" | "running" | "done" | "error"; stage: string; progress: number; result: Record<string, unknown> | null; error: string | null }
interface Session {
  id: string;
  width: number;
  height: number;
  image: string;
  layers: Record<Layer, boolean>;
  tune?: Tune;
  calibration: Calibration | null;
  polygon: Pt[] | null;
  slots: DrawnSlot[];
  job: SessionJob | null;
}
/** The detector and rule settings sent with every model step of a session (the server keeps the last ones). Ranges mirror the server's. */
export interface Tune { model: string; conf: number; iou: number; footprint: number; encroach: number }
export const TUNE_DEFAULTS: Tune = { model: "default", conf: 0.2, iou: 0.45, footprint: 0.5, encroach: 0.3 };
interface ModelOption { id: string; label: string; note: string; kind: "vehicle" | "bay"; ready: boolean }
const SLIDERS: { key: Exclude<keyof Tune, "model">; label: string; min: number; max: number; step: number; pct?: boolean; help: string }[] = [
  { key: "conf", label: "Confidence", min: 0.1, max: 0.75, step: 0.01, help: "Lower finds more distant and half-hidden cars (and a few false ones). A live camera runs at 0.30." },
  { key: "iou", label: "NMS overlap (IoU)", min: 0.2, max: 0.7, step: 0.01, help: "Two boxes overlapping more than this are one car. Lower merges neighbours; higher keeps cars that stand close together." },
  { key: "footprint", label: "Blocked footprint", min: 0.2, max: 0.8, step: 0.01, pct: true, help: "Share of an object's footprint inside a bay that makes it Blocked by object." },
  { key: "encroach", label: "Encroach cover", min: 0.15, max: 0.6, step: 0.01, pct: true, help: "Share of a free bay one vehicle covers that makes it Encroached (free, not usable)." },
];

interface Service { service: "up" | "down"; loaded: unknown; command: string }
type Mode = "pins" | "ref" | "polygon" | "sam" | "edit" | "zone-outline" | "zone-sam";
/** A zone being set up on the page: its outline (drawn), SAM's clicks in it and SAM's surface inside it (or none: the outline is the surface). */
interface ZoneDraft { id: number; name: string; outline: Pt[]; sam: { p: Pt; keep: boolean }[]; surface: Pt[] | null; rows: number }
/** A zone the server calibrated (its own GeoCalib / pins run and grid): GET session `zones`. */
interface ZoneCal { zone: number; name: string; polygon: Pt[]; calibration: Calibration; slots: DrawnSlot[] }
interface ZoneMerge { zones: ZoneCal[]; zone_slots: DrawnSlot[]; zone_dropped: { id: number; name: string; same_as: string }[] }
const ZONE_COLOURS = ["#0ea5b7", "#f97316", "#8b5cf6", "#16a34a", "#e11d48", "#ca8a04"];
const zoneColour = (id: number) => ZONE_COLOURS[(id - 1) % ZONE_COLOURS.length];
const zoneLetter = (id: number) => String.fromCharCode(65 + ((id - 1) % 26));
type Method = "rectangle" | "geocalib" | "topdown";

const BASE = "/api/sample-sessions";
const PIN_HELP = "Click the 4 corners of a rectangle on the ground whose sides you know (a parking bay, a painted box), in order around it: 1 → 2 is the first side, 2 → 3 the second.";

/** Dynamic slots on / off for a run. Off (the default): the run judges exactly the fixed / configured bays. On: the live auto grid's vehicle-driven
 * rules also run: a vehicle standing off the bays gets a bay of its own (D1, D2 …), two vehicles in one bay get sub-bays, a vehicle is given the bay its wheels stand on. */
function DynamicToggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="mt-2 flex items-center gap-1.5 text-sm" title="Off: only the fixed bays are judged, nothing is generated or adjusted. On: bays for vehicles off the layout, sub-bays and the wheel-footprint claim, as the live auto grid">
      <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked)} data-testid="sess-dynamic" />
      Dynamic slots <span className="text-xs text-muted">({on ? "on: bays may be added for vehicles off the layout" : "off: fixed bays only"})</span>
    </label>
  );
}

/** A section that opens and closes. Its content stays mounted while closed (hidden), so a half-done calibration or drawing is not lost. */
function Fold({ id, title, open, onToggle, children }: { id: string; title: React.ReactNode; open: boolean; onToggle: () => void; children: React.ReactNode }) {
  return (
    <section className="overflow-hidden rounded-lg border border-line bg-white shadow-sm" data-testid={`fold-${id}`}>
      <h2 className={`px-4 py-2.5 text-sm font-semibold text-ink ${open ? "border-b border-line" : ""}`}>
        <button type="button" className="flex w-full items-center gap-2 text-left" onClick={onToggle} aria-expanded={open} data-testid={`fold-${id}-toggle`}>
          {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}{title}
        </button>
      </h2>
      <div hidden={!open} className="p-4" data-testid={`fold-${id}-body`}>{children}</div>
    </section>
  );
}

export function SampleSession({ cam, tokenKey, onNeedAdmin }: {
  cam: string | null;
  tokenKey: string;
  onNeedAdmin: (reason: string, retry?: () => void) => void;
}) {
  const [session, setSession] = useState<Session | null>(null);
  const [method, setMethod] = useState<Method>("geocalib");
  const [mode, setMode] = useState<Mode>("ref");
  const [pins, setPins] = useState<Pt[]>([]);
  const [sides, setSides] = useState({ a: "2.5", b: "5" });
  const [ref, setRef] = useState<Pt[]>([]);
  const [refMetres, setRefMetres] = useState("");
  const [lens, setLens] = useState("simple_radial");
  const [samples, setSamples] = useState<{ folder: string; images: string[] } | null>(null);
  const [service, setService] = useState<Service | null>(null);
  const [tune, setTune] = useState<Tune>(TUNE_DEFAULTS);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [open, setOpen] = useState({ calibrated: false, direct: false }); // the two ways of testing start closed: open the one you work on
  const [directModel, setDirectModel] = useState("");
  const [directConf, setDirectConf] = useState(0.25);
  const [directIou, setDirectIou] = useState(0.45);
  const [direct, setDirect] = useState<SampleResult | null>(null);
  const [shown, setShown] = useState<"calibrated" | "direct" | null>(null); // whose result is on screen: the last test run, never both at once
  const [useDrawn, setUseDrawn] = useState(false); // direct bay model on slots drawn by hand
  const [drawn, setDrawn] = useState<DrawnSlot[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [drag, setDrag] = useState<{ kind: "vertex"; id: number; index: number } | { kind: "move"; id: number; last: Pt } | null>(null);
  const [job, setJob] = useState<SessionJob | null>(null);
  const [layers, setLayers] = useState<Record<Layer, boolean>>({ spillage: false, obstacle_filter: false, blocked_objects: false });
  const [expected, setExpected] = useState<Expected>({ free: "", busy: "", blocked: "" });
  const [result, setResult] = useState<SampleResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const stage = useRef<HTMLDivElement>(null); // the zoomed, panned layer holding the picture and its marks (its rect gives shares of the picture)
  const viewport = useRef<HTMLDivElement>(null); // the fixed box the layer is clipped to
  const { view, zoomCentre, panBy, fit, transform } = useZoomPan(viewport, !!session);
  const [panTool, setPanTool] = useState(false); // drag moves the picture instead of placing a point (the middle mouse button does it in every mode)
  const [panning, setPanning] = useState<{ x: number; y: number } | null>(null);
  // zones: every zone its own SAM surface and its own GeoCalib (or pins) run and grid on the server; the run merges every zone's bays
  const [zones, setZones] = useState<ZoneDraft[]>([]);
  const [activeZone, setActiveZone] = useState<number | null>(null);
  const [merge, setMerge] = useState<ZoneMerge>({ zones: [], zone_slots: [], zone_dropped: [] });
  const [skip, setSkip] = useState<number[]>([]); // merged bays the user removed
  const [zoneDirty, setZoneDirty] = useState<number[]>([]); // zones whose bays were moved / reshaped here and not saved yet
  const [nextN, setNextN] = useState("3");
  const [dynamic, setDynamic] = useState(false); // dynamic slots (bays for vehicles off the layout, sub-bays, footprint claim): only when switched on
  const [zoneNote, setZoneNote] = useState<string | null>(null);
  const [fineTarget, setFineTarget] = useState<"bay" | "zone">("bay"); // fine adjustment: the selected bay, or every bay of the active zone
  const [fineStep, setFineStep] = useState(1); // pixels per arrow press (Shift: five times)
  const [undo, setUndo] = useState<{ merge: ZoneMerge; dirty: number[] }[]>([]); // the bay edits made here (drag, fine adjustment, split), newest last

  const fail = useCallback((e: unknown, what: string, retry?: () => void) => {
    const err = e as ApiError;
    if (err.status === 401 || err.status === 503) onNeedAdmin(`${what} needs the admin token.`, retry);
    else setMessage(err.message);
  }, [onNeedAdmin]);

  const reset = () => {
    setZones([]); setActiveZone(null); setMerge({ zones: [], zone_slots: [], zone_dropped: [] }); setSkip([]); setZoneDirty([]); setZoneNote(null); setUndo([]);
    setDirect(null); setShown(null); setUseDrawn(false); setDrawn([]); setPins([]); setRef([]); setSel(null); setJob(null); setResult(null); setMessage(null);
    setMode(method === "rectangle" ? "pins" : "ref");
  };
  useEffect(() => { // a different camera / token: the session belongs to the old profile
    setSession(null);
    reset();
  }, [cam, tokenKey]);

  const check = useCallback(() => {
    api.get<Service>(`${BASE}/service`).then(setService).catch(() => setService(null));
  }, []);
  useEffect(() => {
    check();
    api.get<{ folder: string; images: string[] }>(`${BASE}/images`).then(setSamples).catch(() => setSamples(null));
    api.get<{ models: ModelOption[]; defaults: Tune }>(`${BASE}/models`).then((m) => (setModels(m.models), setTune(m.defaults))).catch(() => setModels([]));
  }, [tokenKey, check]);

  const start = async (src: { file?: File; sample?: string }) => {
    const f = src.file;
    if (!f && !src.sample) return;
    if (f && !/^image\/(jpeg|png)$/.test(f.type)) return setMessage("Pick a JPEG or PNG picture.");
    if (f && f.size > MAX_BYTES) return setMessage("That picture is over 12 MB.");
    setBusy(true);
    try {
      if (session) void api.delete(`${BASE}/${session.id}`).catch(() => {});
      const s = await api.post<Session>(BASE, { ...(f ? { image: await readFile(f) } : { sample: src.sample }), ...(cam ? { profile_cam: cam } : {}) });
      reset();
      setSession(s);
      setLayers(s.layers);
      setZones([{ id: 1, name: "Zone A", outline: [], sam: [], surface: null, rows: 0 }]); // the picture is laid out by zones only: the first one is ready to draw
      setActiveZone(1);
    } catch (e) {
      fail(e, "Starting a sample session");
    } finally {
      setBusy(false);
    }
  };

  // the running job: polled until it ends
  const follow = useCallback((j: SessionJob) => {
    setJob(j);
    if (j.state === "done" && j.result) {
      const r = j.result as { polygon?: Pt[]; slots?: DrawnSlot[]; zone?: number | null } & Partial<ZoneMerge>;
      if (r.zone != null && r.slots) {
        const zid = r.zone;
        setMerge({ zones: r.zones ?? [], zone_slots: r.zone_slots ?? [], zone_dropped: r.zone_dropped ?? [] });
        setUndo([]);
        setZoneDirty((d) => d.filter((x) => x !== zid)); // a new calibration replaces the zone's edited bays
        setMode("edit");
      } else if (r.zone != null && r.polygon) {
        const zid = r.zone;
        setZones((all) => all.map((z) => (z.id === zid ? { ...z, surface: r.polygon! } : z)));
        setMode("edit");
      }
    }
    if (j.state === "error") {
      setMessage(j.error);
      check(); // a stopped setup service is the usual reason: refresh the banner
    }
  }, [check]);
  useEffect(() => {
    if (!session || !job || (job.state !== "queued" && job.state !== "running")) return;
    const t = setTimeout(() => {
      api.get<SessionJob>(`${BASE}/${session.id}/job`).then(follow).catch((e) => (setJob(null), fail(e, "Following the job")));
    }, 1000);
    return () => clearTimeout(t);
  }, [job, session, follow, fail]);
  const running = job?.state === "queued" || job?.state === "running";

  const at = (e: { clientX: number; clientY: number }): Pt => {
    const r = stage.current!.getBoundingClientRect();
    return pt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
  };
  const click = (e: React.MouseEvent) => {
    if (!session || running || drag || panTool) return;
    const p = at(e);
    if (mode === "zone-outline" || mode === "zone-sam") {
      const z = zones.find((x) => x.id === activeZone);
      if (!z) return;
      if (mode === "zone-sam") return setZone(z.id, { sam: [...z.sam, { p, keep: !e.shiftKey }] });
      const last = z.outline[z.outline.length - 1];
      if (last && Math.hypot((p[0] - last[0]) * w, (p[1] - last[1]) * h) < w / 200) return; // the second click of a double-click
      return setZone(z.id, { outline: [...z.outline, p], surface: null });
    }
    if (mode === "pins" && pins.length < 4) setPins([...pins, p]);
    else if (mode === "ref" && ref.length < 2) setRef([...ref, p]);
  };

  const sendJob = async (path: string, body: unknown, what: string) => {
    if (!session) return;
    setMessage(null);
    setBusy(true);
    try {
      follow(await api.post<SessionJob>(`${BASE}/${session.id}/${path}`, body));
    } catch (e) {
      fail(e, what, () => void sendJob(path, body, what));
    } finally {
      setBusy(false);
    }
  };
  const run = async () => {
    if (!session) return;
    setBusy(true);
    setMessage(null);
    try {
      if (!(await saveZones())) return;
      setResult(await api.post<SampleResult>(`${BASE}/${session.id}/run`, { layers, tune, layout: "zones", skip, dynamic }));
      setShown("calibrated");
    } catch (e) {
      fail(e, "Running the test", () => void run());
    } finally {
      setBusy(false);
    }
  };
  // the bay model straight on the picture: a comparison next to the calibrated pipeline, which it does not touch
  const bayModels = models.filter((m) => m.kind === "bay");
  const directModels = useDrawn ? models.filter((m) => m.ready) : bayModels; // hand-drawn slots can be judged by a vehicle detector too
  const runDirect = async () => {
    if (!session) return;
    setBusy(true);
    setMessage(null);
    try {
      setDirect(await api.post<SampleResult>(`${BASE}/${session.id}/direct`, { ...(directModel || bayModels[0] ? { model: directModel || bayModels[0].id } : {}), conf: directConf, iou: directIou, ...(useDrawn ? { slots: drawn } : {}) }));
      setShown("direct");
    } catch (e) {
      fail(e, "Running the bay model", () => void runDirect());
    } finally {
      setBusy(false);
    }
  };
  const discard = async () => {
    if (session) await api.delete(`${BASE}/${session.id}`).catch(() => {});
    setSession(null);
    reset();
  };

  // ---- zones --------------------------------------------------------------------------------------------------------------------------------
  const setZone = (id: number, change: Partial<ZoneDraft>) => setZones((all) => all.map((z) => (z.id === id ? { ...z, ...change } : z)));
  const addZone = () => {
    const id = Math.max(0, ...zones.map((z) => z.id), ...merge.zones.map((z) => z.zone)) + 1;
    setZones([...zones, { id, name: `Zone ${zoneLetter(id)}`, outline: [], sam: [], surface: null, rows: 0 }]);
    setActiveZone(id);
    setMode("zone-outline");
  };
  const dropZone = async (id: number) => {
    if (session && merge.zones.some((z) => z.zone === id)) {
      try {
        const m = await api.delete<ZoneMerge>(`${BASE}/${session.id}/zones/${id}`);
        setMerge(m);
        setUndo([]); // (the zone is gone: older snapshots would bring it back)
      } catch (e) {
        return fail(e, "Removing the zone");
      }
    }
    const rest = zones.filter((z) => z.id !== id);
    setZones(rest);
    if (activeZone === id) setActiveZone(rest.length ? rest[rest.length - 1].id : null);
  };
  const scaleReason = (): string | null => {
    if (method === "rectangle") {
      if (pins.length < 4) return "step 1: click the 4 corners of the ground rectangle";
      if (!(Number(sides.a) > 0 && Number(sides.b) > 0)) return "step 1: type the rectangle's two sides in metres";
    } else {
      if (ref.length < 2) return "step 1: click two ground points a known distance apart";
      if (!(Number(refMetres) > 0)) return "step 1: type the distance between them in metres";
    }
    return null;
  };
  const zoneReason = (z: ZoneDraft): string | null => {
    if (z.outline.length < 3) return `draw ${z.name}'s outline (3 or more corners around that part of the lot)`;
    if (area(z.outline) < 0.002) return `${z.name}'s corners enclose almost no area`;
    return scaleReason();
  };
  const outlineZone = (z: ZoneDraft) => session && sendJob("surface", {
    tune, zone: z.id, bound: z.outline, clicks: z.sam.map(({ p: [x, y], keep }) => ({ x: x * session.width, y: y * session.height, label: keep ? 1 : 0 })) }, `Outlining ${z.name} with SAM`);
  const calibrateZone = (z: ZoneDraft) => {
    const polygon = z.surface ?? z.outline;
    return sendJob("calibrate", method === "rectangle"
      ? { method, tune, polygon, pins, sides: [Number(sides.a), Number(sides.b)], zone: z.id, zone_name: z.name, bound: z.outline, rows: z.rows }
      : { method, tune, polygon, ...(method === "geocalib" ? { lens } : {}), reference: { a: ref[0], b: ref[1], metres: Number(refMetres) }, zone: z.id, zone_name: z.name, bound: z.outline, rows: z.rows }, `Calibrating ${z.name}`);
  };
  // a zone's bays edited on the picture: kept here (in the zone and in the merged list) and saved to the server before an extension or a run
  const editZoneBays = (next: DrawnSlot[], id: number) => {
    const zid = Math.floor(id / 100);
    const changed = next.find((b) => b.id === id)!;
    setMerge((m) => ({ ...m, zone_slots: m.zone_slots.map((b) => (b.id === id ? changed : b)),
                       zones: m.zones.map((z) => (z.zone === zid ? { ...z, slots: z.slots.map((b) => (b.id === id ? changed : b)) } : z)) }));
    setZoneDirty((d) => (d.includes(zid) ? d : [...d, zid]));
  };
  const saveZones = async (): Promise<boolean> => {
    if (!session) return false;
    for (const zid of zoneDirty) {
      const z = merge.zones.find((x) => x.zone === zid);
      if (!z) continue;
      try {
        setMerge(await api.put<ZoneMerge>(`${BASE}/${session.id}/zones/${zid}/slots`, { slots: z.slots }));
      } catch (e) {
        fail(e, `Saving ${z.name}'s bays`);
        return false;
      }
    }
    setZoneDirty([]);
    return true;
  };
  const extendZone = async (zid: number, count: number) => {
    if (!session) return;
    setBusy(true);
    setMessage(null);
    setZoneNote(null);
    try {
      if (!(await saveZones())) return;
      const r = await api.post<ZoneMerge & { added: number; note: string | null }>(`${BASE}/${session.id}/zones/${zid}/extend`, { count });
      setMerge({ zones: r.zones, zone_slots: r.zone_slots, zone_dropped: r.zone_dropped });
      setUndo([]); // (the server added bays: an older snapshot would drop them)
      setZoneNote(r.note);
    } catch (e) {
      fail(e, "Adding the next bays");
    } finally {
      setBusy(false);
    }
  };

  const startPan = (e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && panTool)) {
      e.preventDefault();
      setPanning({ x: e.clientX, y: e.clientY });
    }
  };
  const onMove = (e: React.MouseEvent) => {
    if (panning) {
      panBy(e.clientX - panning.x, e.clientY - panning.y);
      return setPanning({ x: e.clientX, y: e.clientY });
    }
    if (!drag) return;
    const p = at(e);
    const zb = merge.zone_slots;
    if (drag.kind === "vertex") editZoneBays(moveVertex(zb, drag.id, drag.index, p), drag.id);
    else {
      editZoneBays(moveSlot(zb, drag.id, p[0] - drag.last[0], p[1] - drag.last[1]), drag.id);
      setDrag({ ...drag, last: p });
    }
  };

  const [w, h] = session ? [session.width, session.height] : [1920, 1080];
  const px = (p: Pt): [number, number] => [p[0] * w, p[1] * h];
  const dot = w / 220 / view.s; // marks keep their size on screen when the picture is zoomed
  const font = w / 70 / view.s;
  const az = zones.find((z) => z.id === activeZone) ?? null;
  const calOf = (id: number) => merge.zones.find((z) => z.zone === id) ?? null;
  const bays = merge.zone_slots.filter((b) => !skip.includes(b.id));
  // ---- fine adjustment of laid bays (the generation is not touched): arrow keys / buttons shift, the size and turn buttons tune the selected bay or the whole active zone
  const fineIds = (): number[] => (fineTarget === "zone" ? bays.filter((b) => Math.floor(b.id / 100) === activeZone).map((b) => b.id) : bays.filter((b) => b.id === sel).map((b) => b.id));
  const pushUndo = () => setUndo((u) => [...u.slice(-49), { merge, dirty: zoneDirty }]);
  const undoEdit = () => {
    const last = undo[undo.length - 1];
    if (!last) return;
    setUndo((u) => u.slice(0, -1));
    setMerge(last.merge);
    setZoneDirty((d) => [...new Set([...d, ...last.dirty, ...last.merge.zones.map((z) => z.zone)])]); // (the server holds the edited bays: save the restored ones over them)
  };
  const applyFine = (next: DrawnSlot[], ids: number[]) => {
    pushUndo();
    const changed = next.filter((b) => ids.includes(b.id));
    const zs = [...new Set(changed.map((b) => Math.floor(b.id / 100)))];
    setMerge((m) => ({ ...m, zone_slots: m.zone_slots.map((b) => changed.find((c) => c.id === b.id) ?? b),
                       zones: m.zones.map((z) => (zs.includes(z.zone) ? { ...z, slots: z.slots.map((b) => changed.find((c) => c.id === b.id) ?? b) } : z)) }));
    setZoneDirty((d) => [...d, ...zs.filter((z) => !d.includes(z))]);
  };
  const fineShift = (dxPx: number, dyPx: number) => {
    const ids = fineIds();
    if (ids.length) applyFine(shiftBays(merge.zone_slots, ids, dxPx / w, dyPx / h), ids);
  };
  const fineAlong = (across: number, along: number) => {
    const ids = fineIds();
    if (ids.length) applyFine(shiftAlong(merge.zone_slots, ids, w, h, across, along), ids);
  };
  const fineTune = (fn: (poly: Pt[]) => Pt[]) => {
    const ids = fineIds();
    if (ids.length) applyFine(tuneBays(merge.zone_slots, ids, fn), ids);
  };
  // split the selected bay in two (a bay laid across two cars): side by side, or front to back; the second part gets the zone's next free id
  const splitSelected = (axis: "width" | "length") => {
    const b = merge.zone_slots.find((x) => x.id === sel);
    if (!b) return;
    const zid = Math.floor(b.id / 100);
    const used = new Set(merge.zone_slots.filter((x) => Math.floor(x.id / 100) === zid).map((x) => x.id));
    const free = Array.from({ length: 99 }, (_, i) => zid * 100 + i + 1).find((id) => !used.has(id));
    if (free === undefined) return setZoneNote("This zone already has 99 bays: remove one first.");
    const [p1, p2] = splitBay(b.poly, axis, 2);
    if (!p2) return setZoneNote("This bay cannot be split: it is not a four-cornered bay.");
    setZoneNote(null);
    const first = { ...b, name: `${b.name}a`, poly: p1 };
    const second = { ...b, id: free, name: `${b.name}b`, poly: p2 };
    pushUndo();
    const cut = (all: DrawnSlot[]) => all.flatMap((x) => (x.id === b.id ? [first, second] : [x]));
    setMerge((m) => ({ ...m, zone_slots: cut(m.zone_slots), zones: m.zones.map((z) => (z.zone === zid ? { ...z, slots: cut(z.slots) } : z)) }));
    setZoneDirty((d) => (d.includes(zid) ? d : [...d, zid]));
  };
  useEffect(() => {
    if (mode !== "edit") return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")) return; // typing is not nudging
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "z") {
        if (undo.length) (e.preventDefault(), undoEdit());
        return;
      }
      const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
      if (!d || e.ctrlKey || e.metaKey || e.altKey) return;
      if (!fineIds().length) return;
      e.preventDefault();
      const k = fineStep * (e.shiftKey ? 5 : 1);
      fineShift(d[0] * k, d[1] * k);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  const zonesCard = () => (
    <>
      <Card title={`2 · Zones (${zones.length})`}>
        <p className="text-xs text-muted">
          Outline each part of the lot that lies at its own angle or depth (a row, a block) as a zone. SAM finds the parking ground inside it and GeoCalib (or the ground
          pins of step 1) calibrates that zone on its own, so its bay grid follows its own perspective. Every zone's bays are then run together as one list.
        </p>
        <ul className="mt-2 divide-y divide-line rounded-md border border-line" data-testid="zone-list">
          {zones.map((z) => {
            const cal = calOf(z.id);
            return (
              <li key={z.id} className={`flex items-center gap-2 px-2.5 py-1 text-sm ${z.id === activeZone ? "bg-primary/5" : ""}`} data-testid="zone-row">
                <span className="inline-block h-3 w-3 rounded-full" style={{ background: zoneColour(z.id) }} />
                <button type="button" className="flex-1 text-left" onClick={() => (setActiveZone(z.id), setMode("zone-outline"))} data-testid="zone-select">
                  {z.name} <span className="text-xs text-muted">· {cal ? `${cal.calibration.bays} bay(s), ${cal.calibration.confidence}` : z.outline.length < 3 ? "outline not drawn" : "not calibrated"}</span>
                </button>
                <button type="button" title="Delete this zone" className="rounded p-1 text-muted hover:bg-danger/10 hover:text-danger" onClick={() => void dropZone(z.id)} data-testid="zone-delete">
                  <Trash2 size={14} />
                </button>
              </li>
            );
          })}
        </ul>
        <div className="mt-2"><Button variant="outline" onClick={addZone} disabled={zones.length >= 20} data-testid="zone-add">+ Add a zone</Button></div>
        {az && (() => {
          const cal = calOf(az.id);
          const why = zoneReason(az);
          return (
            <div key={az.id} className="mt-3 space-y-2 border-t border-line pt-3" data-testid="zone-editor">
              <label className="block text-xs text-muted">Name
                <input className="ml-1 w-32 rounded-md border border-line px-1.5 py-1 text-sm text-ink" value={az.name} data-testid="zone-name" onChange={(e) => setZone(az.id, { name: e.target.value })} />
              </label>
              <div className="flex flex-wrap gap-2">
                <Button variant={mode === "zone-outline" ? "primary" : "outline"} onClick={() => setMode("zone-outline")} data-testid="zone-draw"><Crosshair size={14} /> Draw outline ({az.outline.length})</Button>
                <Button variant="ghost" onClick={() => setZone(az.id, { outline: az.outline.slice(0, -1), surface: null })} disabled={!az.outline.length} data-testid="zone-undo">Undo corner</Button>
                <Button variant="ghost" onClick={() => setZone(az.id, { outline: [], sam: [], surface: null })} disabled={!az.outline.length && !az.sam.length} data-testid="zone-clear">Clear</Button>
                <label className="block text-xs text-muted" title="Rows of bays in this zone, from its front edge (the one nearest the camera). Auto: the zone's depth over a bay's length, so a zone drawn over two cars side by side gets two rows">Rows
                  <select className="ml-1 rounded-md border border-line bg-white px-1.5 py-1 text-sm text-ink" value={az.rows} data-testid="zone-rows"
                          onChange={(e) => setZone(az.id, { rows: Number(e.target.value) })}>
                    <option value={0}>Auto</option>
                    {[1, 2, 3, 4, 5, 6].map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant={mode === "zone-sam" ? "primary" : "outline"} onClick={() => setMode("zone-sam")} disabled={az.outline.length < 3} data-testid="zone-sam-mode"><Sparkles size={14} /> Click for SAM ({az.sam.length})</Button>
                <Button variant="outline" onClick={() => void outlineZone(az)} disabled={!az.sam.length || az.outline.length < 3 || running || busy} data-testid="zone-sam">
                  <Sparkles size={14} /> Find the ground with SAM
                </Button>
              </div>
              <p className="text-xs text-muted">{az.surface ? "SAM's ground inside the zone is shown filled: it is what gets calibrated." : "Without SAM the outline itself is the parking ground. In SAM mode click the open ground (Shift-click what is not parking)."}</p>
              <Button onClick={() => void calibrateZone(az)} disabled={!!why || running || busy} title={why ?? undefined} data-testid="zone-calibrate">
                {running ? <Loader2 size={14} className="animate-spin" /> : <FlaskConical size={14} />} {cal ? "Calibrate again" : "Calibrate and lay bays"} ({method === "geocalib" ? "GeoCalib" : method === "topdown" ? "top-down" : "ground pins"})
              </Button>
              {why && <p className="rounded-md bg-warning/10 px-2.5 py-1.5 text-xs text-ink" data-testid="zone-reason">The button wakes up when this is done: {why}.</p>}
              {job && running && <p className="text-xs text-ink" data-testid="zone-job">{job.stage} · {Math.round(job.progress * 100)} %</p>}
              {cal && (
                <div className="space-y-1 text-xs" data-testid="zone-calibration">
                  <p><span className="rounded bg-success/15 px-1.5 py-0.5 font-semibold text-[#1b7a3e]">{cal.calibration.confidence}</span> camera {cal.calibration.h.toFixed(1)} m up · focal {Math.round(cal.calibration.f)} px
                    {cal.calibration.hfov_deg != null ? ` · ${cal.calibration.hfov_deg}° wide` : ""} · {cal.calibration.area_m2} m² · {cal.calibration.bays} bays</p>
                  {cal.calibration.sizing && (
                    <p data-testid="zone-sizing">Bays {cal.calibration.sizing.length_m != null ? `${cal.calibration.sizing.width_m} × ${cal.calibration.sizing.length_m} m` : `about ${cal.calibration.sizing.width_m ?? "?"} m wide`} ({cal.calibration.sizing.source}) · {cal.calibration.sizing.bays_along} along the row
                      {cal.calibration.sizing.rows > 1 ? ` × ${cal.calibration.sizing.rows} rows` : ""}</p>
                  )}
                  {cal.calibration.warnings.map((t) => <p key={t} className="text-warning">{t}</p>)}
                </div>
              )}
              {cal && (
                <div className="space-y-2 border-t border-line pt-2" data-testid="zone-tools">
                  <div className="flex flex-wrap items-center gap-2">
                    <Button variant="outline" onClick={() => void extendZone(az.id, 1)} disabled={busy || running} data-testid="zone-next-1"
                            title="One more bay after the last one, at the same spacing on the ground">Add next 1</Button>
                    <Button variant="outline" onClick={() => void extendZone(az.id, 2)} disabled={busy || running} data-testid="zone-next-2">Add next 2</Button>
                    <input className="w-12 rounded-md border border-line px-1.5 py-1 text-sm" inputMode="numeric" value={nextN} data-testid="zone-next-n-value"
                           onChange={(e) => setNextN(e.target.value.replace(/[^0-9]/g, ""))} />
                    <Button variant="outline" onClick={() => void extendZone(az.id, Math.min(50, Math.max(1, Number(nextN) || 1)))} disabled={busy || running} data-testid="zone-next-n">
                      Add next {Math.min(50, Math.max(1, Number(nextN) || 1))}
                    </Button>
                  </div>
                  <Button variant={mode === "edit" ? "primary" : "outline"} onClick={() => setMode("edit")} data-testid="zone-edit-mode">Edit bays</Button>
                  <p className="text-xs text-muted">Edit bays: click a bay to select it, drag it to move it, drag its corners to reshape it. Edits are saved before the next bays are added and before the run.</p>
                  {mode === "edit" && (
                    <div className="space-y-1.5 rounded-md border border-line p-2" data-testid="zone-fine">
                      <p className="text-xs font-semibold">Select zone &amp; slot adjustment</p>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <Pill active={fineTarget === "bay"} onClick={() => setFineTarget("bay")} title="The bay selected in the list or on the picture">Selected bay{sel != null ? ` (${bays.find((b) => b.id === sel)?.name ?? "—"})` : ""}</Pill>
                        <Pill active={fineTarget === "zone"} onClick={() => setFineTarget("zone")} title="Every bay of the active zone, moved or tuned together">Whole {az.name}</Pill>
                        <label className="ml-auto flex items-center gap-1 text-xs">Step
                          <select className="rounded-md border border-line px-1 py-0.5" value={fineStep} onChange={(e) => setFineStep(Number(e.target.value))} data-testid="zone-fine-step">
                            {[1, 2, 5, 10].map((n) => <option key={n} value={n}>{n} px</option>)}
                          </select>
                        </label>
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5">
                        {([["←", -1, 0, "left"], ["↑", 0, -1, "up"], ["↓", 0, 1, "down"], ["→", 1, 0, "right"]] as const).map(([t, dx, dy, id]) => (
                          <Button key={id} variant="outline" onClick={() => fineShift(dx * fineStep, dy * fineStep)} disabled={fineIds().length === 0} data-testid={`zone-fine-${id}`} title={`Move ${id} by the step`}>{t}</Button>
                        ))}
                        <span className="text-xs text-muted">or the arrow keys (Shift = 5×)</span>
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-xs text-muted">Along the bay:</span>
                        {([["Row ◂", -1, 0, "row-left", "along its front, to its left"], ["Row ▸", 1, 0, "row-right", "along its front, to its right"],
                           ["Back ↥", 0, 1, "back", "towards its back (away from the camera)"], ["Front ↧", 0, -1, "front", "towards its front (nearer the camera)"]] as const).map(([t, a, b2, id, tip]) => (
                          <Button key={id} variant="outline" onClick={() => fineAlong(a * fineStep, b2 * fineStep)} disabled={fineIds().length === 0} data-testid={`zone-fine-${id}`} title={`Move ${tip}`}>{t}</Button>
                        ))}
                        <Button variant="outline" onClick={undoEdit} disabled={undo.length === 0} data-testid="zone-undo-edit" title="Take back the last bay edit (move, reshape, size, turn, split)">Undo{undo.length ? ` (${undo.length})` : ""}</Button>
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <Button variant="outline" onClick={() => fineTune((p) => resizeBay(p, w, h, 0.98, 1))} disabled={fineIds().length === 0} data-testid="zone-fine-narrower" title="2 % narrower along its front">Narrower</Button>
                        <Button variant="outline" onClick={() => fineTune((p) => resizeBay(p, w, h, 1.02, 1))} disabled={fineIds().length === 0} data-testid="zone-fine-wider" title="2 % wider along its front">Wider</Button>
                        <Button variant="outline" onClick={() => fineTune((p) => resizeBay(p, w, h, 1, 0.98))} disabled={fineIds().length === 0} data-testid="zone-fine-shorter" title="2 % shorter towards its back">Shorter</Button>
                        <Button variant="outline" onClick={() => fineTune((p) => resizeBay(p, w, h, 1, 1.02))} disabled={fineIds().length === 0} data-testid="zone-fine-longer" title="2 % longer towards its back">Longer</Button>
                        <Button variant="outline" onClick={() => fineTune((p) => resizeBay(p, w, h, 0.98, 0.98))} disabled={fineIds().length === 0} data-testid="zone-fine-shrink" title="2 % smaller all round: keeps a bay off its neighbour's car">Shrink 2 %</Button>
                        <Button variant="outline" onClick={() => fineTune((p) => rotateBay(p, w, h, -0.5))} disabled={fineIds().length === 0} data-testid="zone-fine-ccw" title="Turn 0.5° anticlockwise">↺ 0.5°</Button>
                        <Button variant="outline" onClick={() => fineTune((p) => rotateBay(p, w, h, 0.5))} disabled={fineIds().length === 0} data-testid="zone-fine-cw" title="Turn 0.5° clockwise">↻ 0.5°</Button>
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5 border-t border-line pt-1.5">
                        <span className="text-xs font-semibold">Split the selected bay</span>
                        <Button variant="outline" onClick={() => splitSelected("width")} disabled={!bays.some((b) => b.id === sel)} data-testid="zone-split-width"
                                title="Two bays side by side, left and right (equal on the ground)">Left | right</Button>
                        <Button variant="outline" onClick={() => splitSelected("length")} disabled={!bays.some((b) => b.id === sel)} data-testid="zone-split-length"
                                title="Two bays one behind the other, front and back (a bay laid across two cars)">Front / back</Button>
                      </div>
                      <p className="text-xs text-muted">Changes only the bays drawn here; the generated layout is not recalculated. They are saved before the next bays are added and before the run.</p>
                    </div>
                  )}
                  {zoneNote && <p className="text-xs text-warning" data-testid="zone-note">{zoneNote}</p>}
                </div>
              )}
            </div>
          );
        })()}
      </Card>
      <Card title={`3 · Review and run (${bays.length} bays)`}>
        {merge.zone_dropped.length > 0 && (
          <p className="mb-2 text-xs text-muted" data-testid="zone-dropped">
            {merge.zone_dropped.length} bay(s) of overlapping zones counted once: {merge.zone_dropped.slice(0, 4).map((d) => `${d.name} = ${d.same_as}`).join(", ")}{merge.zone_dropped.length > 4 ? " …" : ""}
          </p>
        )}
        <ul className="max-h-48 divide-y divide-line overflow-y-auto rounded-md border border-line" data-testid="zone-bays">
          {bays.map((b) => (
            <li key={b.id} className={`flex items-center gap-2 px-2.5 py-1 text-sm ${b.id === sel ? "bg-primary/5" : ""}`} data-testid="zone-bay-row">
              <span className="flex-1">{b.name} <span className="text-xs text-muted">· {merge.zones.find((z) => z.zone === Math.floor(b.id / 100))?.name}</span></span>
              <button type="button" className="rounded px-1.5 py-0.5 text-xs font-medium text-primary hover:bg-primary/10" data-testid="zone-bay-edit"
                      title="Select it to move it or drag its corners" onClick={() => (setActiveZone(Math.floor(b.id / 100)), setSel(b.id), setMode("edit"))}>Edit</button>
              <button type="button" title="Remove this bay from the run" className="rounded p-1 text-muted hover:bg-danger/10 hover:text-danger" data-testid="zone-bay-delete"
                      onClick={() => setSkip([...skip, b.id])}>
                <Trash2 size={14} />
              </button>
            </li>
          ))}
        </ul>
        {skip.length > 0 && <Button variant="ghost" onClick={() => setSkip([])} data-testid="zone-restore">Restore {skip.length} removed</Button>}
        <LayerBoxes layers={layers} onChange={setLayers} />
        <DynamicToggle on={dynamic} onChange={setDynamic} />
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Button onClick={() => void run()} disabled={busy || bays.length === 0} data-testid="zone-run">
            {busy ? <Loader2 size={14} className="animate-spin" /> : <FlaskConical size={14} />} Run test on all zones
          </Button>
          <ExpectedInputs value={expected} onChange={setExpected} />
        </div>
      </Card>
    </>
  );

  return (
    <div className="space-y-4" data-testid="sample-session">
      <Card title={<span className="flex items-center gap-2"><FlaskConical size={16} className="text-primary" /> Sample test</span>}>
        <p className="mb-3 text-xs text-muted">
          Any parking picture, set up like Camera 1: GeoCalib and one typed ground distance give the scale, SAM outlines the parking surface, a bay grid is laid on it
          for you to review, then the picture runs through the detector, spillage, obstacle filter and Blocked by object. Nothing is saved: the session lives in the
          server's memory for half an hour and no live camera is touched.
        </p>
        <p className={`mb-3 rounded-md px-3 py-2 text-xs ${service?.service === "up" ? "bg-success/10 text-[#1b7a3e]" : "bg-warning/10 text-warning"}`} data-testid="sess-service">
          {service == null ? "Checking the setup service…"
            : service.service === "up" ? `Setup service is running${service.loaded ? " (models loaded)" : " (the first SAM / GeoCalib request loads its models: a few minutes on a CPU)"}.`
            : <>The setup service (SAM, GeoCalib) is not running. Start it with <code>{service.command}</code>, or scale the picture with ground pins, which needs no service. </>}
          <button type="button" className="ml-2 inline-flex items-center gap-1 underline" onClick={check} data-testid="sess-service-check"><RefreshCw size={11} /> Check</button>
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <input ref={input} type="file" accept="image/jpeg,image/png" className="hidden" data-testid="sess-file" onChange={(e) => void start({ file: e.target.files?.[0] })} />
          <Button variant="outline" onClick={() => input.current?.click()} disabled={busy}><ImageUp size={14} /> {session ? "Another picture" : "Upload a picture"}</Button>
          {samples && samples.images.length > 0 && (
            <select className="rounded-md border border-line bg-white px-2 py-1.5 text-sm" data-testid="sess-sample" value="" disabled={busy}
                    onChange={(e) => e.target.value && void start({ sample: e.target.value })}>
              <option value="">…or pick from {samples.folder}/</option>
              {samples.images.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          )}
          {session && <Button variant="ghost" onClick={() => void discard()}><Trash2 size={14} /> Start over</Button>}
        </div>
        {message && <p className="mt-3 rounded-md bg-danger/10 px-3 py-2 text-sm text-danger" role="alert" data-testid="sess-error">{message}</p>}
      </Card>

      <Card title={
        <button type="button" className="flex items-center gap-2" onClick={() => setSettingsOpen(!settingsOpen)} aria-expanded={settingsOpen} data-testid="sess-settings-toggle">
          {settingsOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}<SlidersHorizontal size={16} className="text-primary" /> Detection settings
          <span className="text-xs font-normal text-muted" data-testid="sess-settings-summary">
            {(models.find((m) => m.id === tune.model)?.label ?? "This camera's own detector")} · conf {tune.conf.toFixed(2)} · IoU {tune.iou.toFixed(2)} · blocked {Math.round(tune.footprint * 100)} % · encroach {Math.round(tune.encroach * 100)} %
          </span>
        </button>
      } actions={settingsOpen ? <Button variant="ghost" onClick={() => setTune({ ...TUNE_DEFAULTS })} data-testid="sess-settings-reset">Reset to defaults</Button> : undefined}>
        {settingsOpen ? (
          <div className="space-y-3" data-testid="sess-settings">
            <label className="block text-xs text-muted">YOLO model
              <select className="mt-1 block w-full rounded-md border border-line bg-white px-2 py-1.5 text-sm text-ink" value={tune.model} data-testid="sess-model"
                      onChange={(e) => setTune({ ...tune, model: e.target.value })}>
                <option value="default">This camera's own detector</option>
                {models.filter((m) => m.id !== "default").map((m) => <option key={m.id} value={m.id}>{m.label}{m.ready ? "" : " …"}</option>)}
              </select>
              <span className="mt-1 block" data-testid="sess-model-note">{models.find((m) => m.id === tune.model)?.note}</span>
            </label>
            <div className="grid gap-3 sm:grid-cols-2">
              {SLIDERS.map((d) => {
                const shown = d.pct ? Math.round(tune[d.key] * 100) : tune[d.key];
                const set = (v: number) => setTune({ ...tune, [d.key]: Math.min(d.max, Math.max(d.min, Math.round((d.pct ? v / 100 : v) * 100) / 100)) });
                return (
                  <label key={d.key} className="block text-xs text-muted" title={d.help}>{d.label}{d.pct ? " (%)" : ""}
                    <span className="mt-1 flex items-center gap-2">
                      <input type="range" className="flex-1" min={d.pct ? d.min * 100 : d.min} max={d.pct ? d.max * 100 : d.max} step={d.pct ? d.step * 100 : d.step} value={shown}
                             onChange={(e) => set(Number(e.target.value))} data-testid={`sess-${d.key}`} />
                      <input type="number" className="w-16 rounded-md border border-line px-1.5 py-1 text-sm text-ink" min={d.pct ? d.min * 100 : d.min} max={d.pct ? d.max * 100 : d.max}
                             step={d.pct ? d.step * 100 : d.step} value={shown} onChange={(e) => e.target.value !== "" && set(Number(e.target.value))} data-testid={`sess-${d.key}-n`} />
                    </span>
                    <span className="block">{d.help}</span>
                  </label>
                );
              })}
            </div>
            <p className="text-xs text-muted">Applied to the next SAM, Calibrate or Run step; a Run reuses the last detector pass when only the layers changed.</p>
          </div>
        ) : null}
      </Card>

      {session && (
        <Fold id="direct" open={open.direct} onToggle={() => setOpen({ ...open, direct: !open.direct })}
              title={<span className="flex items-center gap-2"><FlaskConical size={16} className="text-primary" /> Direct bay model (no calibration)</span>}>
          <p className="mb-3 text-xs text-muted">
            For comparison: the trained PKLot bay model run straight on this picture. It finds and colours bays itself (red = busy, green = free); it needs no scale, surface or grid and does not
            change the calibrated pipeline below. It was trained on top-down lots, so expect little on an angled view.
          </p>
          <label className="mb-3 flex items-center gap-2 text-sm" title="Draw the bays yourself; the model then only says busy or free for each of them">
            <input type="checkbox" checked={useDrawn} onChange={(e) => (setUseDrawn(e.target.checked), setDirectModel(""))} data-testid="direct-draw-toggle" />
            Draw fixed slots (the model classifies my slots, not its own boxes)
          </label>
          {useDrawn && (
            <div className="mb-3" data-testid="direct-draw">
              <DirectSlotCanvas image={session.image} width={session.width} height={session.height} slots={drawn} onChange={setDrawn} />
              <p className="mt-2 text-xs text-muted">
                No SAM surface, GeoCalib grid, spillage or encroachment here. A bay model looks at each slot's own crop, then at the whole picture; a slot it cannot see is judged by the camera's vehicle
                detector (a vehicle in the slot = BUSY, none = FREE). A vehicle detector chosen above judges every slot that way. Each slot says which one decided it.
              </p>
            </div>
          )}
          <div className="flex flex-wrap items-end gap-3">
            <label className="block text-xs text-muted">Bay model
              <select className="mt-1 block rounded-md border border-line bg-white px-2 py-1.5 text-sm text-ink" value={directModel || bayModels[0]?.id || ""} data-testid="direct-model"
                      onChange={(e) => setDirectModel(e.target.value)} disabled={bayModels.length === 0}>
                {directModels.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </label>
            <label className="block text-xs text-muted">Confidence
              <input type="number" className="ml-1 w-16 rounded-md border border-line px-1.5 py-1 text-sm text-ink" min={0.05} max={0.75} step={0.05} value={directConf}
                     data-testid="direct-conf" onChange={(e) => e.target.value !== "" && setDirectConf(Math.min(0.75, Math.max(0.05, Number(e.target.value))))} />
            </label>
            <label className="block text-xs text-muted">IoU
              <input type="number" className="ml-1 w-16 rounded-md border border-line px-1.5 py-1 text-sm text-ink" min={0.2} max={0.7} step={0.05} value={directIou}
                     data-testid="direct-iou" onChange={(e) => e.target.value !== "" && setDirectIou(Math.min(0.7, Math.max(0.2, Number(e.target.value))))} />
            </label>
            <Button variant="outline" onClick={() => void runDirect()} disabled={busy || running || bayModels.length === 0 || (useDrawn && drawn.length === 0)} data-testid="direct-run">
              {busy ? <Loader2 size={14} className="animate-spin" /> : <FlaskConical size={14} />} Run direct bay model
            </Button>
          </div>
        </Fold>
      )}

      {session && (
        <Fold id="calibrated" open={open.calibrated} onToggle={() => setOpen({ ...open, calibrated: !open.calibrated })}
              title={<span className="flex items-center gap-2"><Sparkles size={16} className="text-primary" /> GeoCalib + SAM (calibrated grid)</span>}>
        <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,1.4fr)]">
          <Card title={<span>Picture · {w}×{h}</span>} bodyClass="p-0" actions={
            <>
              <Pill active={panTool} onClick={() => setPanTool(!panTool)} title="Drag the picture instead of placing points (the middle mouse button does this in every mode; the wheel zooms)"><Hand size={12} /> Pan</Pill>
              <Button variant="outline" onClick={() => zoomCentre(1.5)} disabled={view.s >= MAX_ZOOM} data-testid="sess-zoom-in" title="Zoom in (or the mouse wheel)"><ZoomIn size={13} /></Button>
              <Button variant="outline" onClick={() => zoomCentre(1 / 1.5)} disabled={view.s <= MIN_ZOOM} data-testid="sess-zoom-out" title="Zoom out"><ZoomOut size={13} /></Button>
              <Button variant="ghost" onClick={fit} disabled={view.s === 1} data-testid="sess-zoom-reset">Fit</Button>
              <span className="text-xs text-muted" data-testid="sess-zoom">{Math.round(view.s * 100)} %</span>
            </>
          }>
            <div ref={viewport} className="relative w-full select-none overflow-hidden bg-ink" style={{ aspectRatio: `${w} / ${h}` }} data-testid="sess-stage">
             <div ref={stage} className="absolute top-0 left-0 h-full w-full" style={{ transform, transformOrigin: "0 0" }} data-testid="sess-layer">
              <img src={session.image} alt="Sample picture" className="absolute inset-0 h-full w-full object-fill" draggable={false} />
              <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full"
                   style={{ cursor: panning ? "grabbing" : panTool ? "grab" : mode !== "edit" ? "crosshair" : undefined }}
                   onMouseDown={startPan} onClick={click} onMouseMove={onMove} onMouseUp={() => (setDrag(null), setPanning(null))}
                   onMouseLeave={() => (setDrag(null), setPanning(null))}
                   data-testid="sess-canvas">
                {<>
                  {[...zones.filter((z) => z.id !== activeZone), ...(az ? [az] : [])].map((z) => {
                    const on = z.id === activeZone;
                    const colour = zoneColour(z.id);
                    return (
                      <g key={`z${z.id}`} opacity={on ? 1 : 0.5} data-testid="zone-layer" data-zone={z.id} data-active={on ? "1" : "0"}>
                        {z.outline.length > 1 && <polygon points={points(z.outline, w, h)} fill="none" stroke={colour} strokeWidth={2} strokeDasharray="8 5" vectorEffect="non-scaling-stroke"
                                                          pointerEvents="none" data-testid="zone-outline" />}
                        {z.surface && <polygon points={points(z.surface, w, h)} fill={`${colour}22`} stroke={colour} strokeWidth={1} vectorEffect="non-scaling-stroke" pointerEvents="none" data-testid="zone-surface" />}
                        {on && z.outline.map((p, i) => <circle key={`o${i}`} cx={px(p)[0]} cy={px(p)[1]} r={dot * 0.8} fill={colour} pointerEvents="none" />)}
                        {on && z.sam.map(({ p, keep }, i) => <circle key={`s${i}`} cx={px(p)[0]} cy={px(p)[1]} r={dot} fill={keep ? "#0ea5b7" : "#dc2626"} pointerEvents="none" data-testid="zone-sam-click" />)}
                        {bays.filter((b) => Math.floor(b.id / 100) === z.id).map((b) => (
                          <g key={b.id} data-testid="zone-bay" data-id={b.id}>
                            <polygon points={points(b.poly, w, h)} fill={b.id === sel ? "rgba(79,70,229,0.25)" : `${colour}26`} stroke={b.id === sel ? "#4f46e5" : colour}
                                     strokeWidth={b.id === sel ? 2.5 : 2} vectorEffect="non-scaling-stroke" style={{ cursor: on ? (mode === "edit" ? "move" : "default") : "pointer" }}
                                     onClick={(e) => { if (!on && !panTool) (e.stopPropagation(), setActiveZone(z.id)); }}
                                     onMouseDown={(e) => { if (on && mode === "edit" && !panTool && e.button === 0) (e.stopPropagation(), setSel(b.id), pushUndo(), setDrag({ kind: "move", id: b.id, last: at(e) })); }} />
                            <text x={b.poly.reduce((m, p) => m + p[0], 0) / b.poly.length * w} y={b.poly.reduce((m, p) => m + p[1], 0) / b.poly.length * h} fontSize={font * 0.8}
                                  textAnchor="middle" dominantBaseline="middle" fontWeight={700} fill="white" stroke="black" strokeWidth={font / 14} paintOrder="stroke" pointerEvents="none">{b.name}</text>
                          </g>
                        ))}
                        {on && mode === "edit" && bays.filter((b) => b.id === sel && Math.floor(b.id / 100) === z.id).flatMap((b) => b.poly.map(([x, y], i) => (
                          <circle key={`v${b.id}-${i}`} cx={x * w} cy={y * h} r={dot} fill="white" stroke="#4f46e5" strokeWidth={2} vectorEffect="non-scaling-stroke" style={{ cursor: "grab" }}
                                  data-testid="zone-vertex" onMouseDown={(e) => { if (e.button === 0 && !panTool) (e.stopPropagation(), pushUndo(), setDrag({ kind: "vertex", id: b.id, index: i })); }} />
                        )))}
                      </g>
                    );
                  })}
                </>}
                {pins.length > 1 && <polygon points={points(pins, w, h)} fill="rgba(88,86,214,0.12)" stroke="#5856d6" strokeWidth={2} vectorEffect="non-scaling-stroke" pointerEvents="none" />}
                {pins.map((p, i) => (
                  <g key={`p${i}`} pointerEvents="none" data-testid="sess-pin">
                    <circle cx={px(p)[0]} cy={px(p)[1]} r={dot * 1.3} fill="#5856d6" />
                    <text x={px(p)[0] + dot * 1.8} y={px(p)[1] - dot * 1.2} fontSize={font} fontWeight={700} fill="white" stroke="black" strokeWidth={font / 14} paintOrder="stroke">{i + 1}</text>
                  </g>
                ))}
                {ref.length === 2 && <line x1={px(ref[0])[0]} y1={px(ref[0])[1]} x2={px(ref[1])[0]} y2={px(ref[1])[1]} stroke="#f9b115" strokeWidth={2} vectorEffect="non-scaling-stroke" pointerEvents="none" />}
                {ref.map((p, i) => <circle key={`r${i}`} cx={px(p)[0]} cy={px(p)[1]} r={dot * 1.3} fill="#f9b115" pointerEvents="none" data-testid="sess-ref" />)}
              </svg>
             </div>
            </div>
          </Card>

          <div className="space-y-4">
            <Card title="1 · Scale">
              <div className="flex gap-1.5" data-testid="sess-method">
                <Pill active={method === "geocalib"} onClick={() => (setMethod("geocalib"), setMode("ref"))} title="GeoCalib in the setup service for the lens and tilt; one typed distance for the scale">GeoCalib + distance</Pill>
                <Pill active={method === "topdown"} onClick={() => (setMethod("topdown"), setMode("ref"))}
                      title="For an aerial / drone picture looking straight down (GeoCalib reads those as nearly level): the ground scale is the same everywhere, from one typed distance; no setup service">Top-down (aerial)</Pill>
                <Pill active={method === "rectangle"} onClick={() => (setMethod("rectangle"), setMode("pins"))} title="Fallback without the setup service: four clicked corners of a ground rectangle of known size">Ground pins (no service)</Pill>
              </div>
              {method === "rectangle" ? (
                <div className="mt-3 space-y-2">
                  <p className="text-xs text-muted">{PIN_HELP}</p>
                  <div className="flex flex-wrap items-end gap-2">
                    <Button variant={mode === "pins" ? "primary" : "outline"} onClick={() => setMode("pins")}><Crosshair size={14} /> Place pins ({pins.length}/4)</Button>
                    <label className="block text-xs text-muted">Side 1→2 (m)
                      <input className="ml-1 w-16 rounded-md border border-line px-1.5 py-1 text-sm text-ink" inputMode="decimal" value={sides.a} data-testid="sess-side-a"
                             onChange={(e) => setSides({ ...sides, a: e.target.value })} />
                    </label>
                    <label className="block text-xs text-muted">Side 2→3 (m)
                      <input className="ml-1 w-16 rounded-md border border-line px-1.5 py-1 text-sm text-ink" inputMode="decimal" value={sides.b} data-testid="sess-side-b"
                             onChange={(e) => setSides({ ...sides, b: e.target.value })} />
                    </label>
                    <Button variant="ghost" onClick={() => setPins([])} disabled={!pins.length}>Clear pins</Button>
                  </div>
                </div>
              ) : (
                <div className="mt-3 space-y-2">
                  <p className="text-xs text-muted">Click two points on the ground a known distance apart (a bay's width, a wheelbase), then type that distance. {method === "topdown" ? "The picture is taken as looking straight down: that distance is the whole scale, and it needs no setup service." : "Needs the setup service."}</p>
                  <div className="flex flex-wrap items-end gap-2">
                    <Button variant={mode === "ref" ? "primary" : "outline"} onClick={() => setMode("ref")}><Crosshair size={14} /> Place points ({ref.length}/2)</Button>
                    <label className="block text-xs text-muted">Distance (m)
                      <input className="ml-1 w-16 rounded-md border border-line px-1.5 py-1 text-sm text-ink" inputMode="decimal" value={refMetres} data-testid="sess-ref-metres"
                             onChange={(e) => setRefMetres(e.target.value)} />
                    </label>
                    {method === "geocalib" && <select className="rounded-md border border-line bg-white px-2 py-1 text-sm" value={lens} onChange={(e) => setLens(e.target.value)} data-testid="sess-lens"
                            title="GeoCalib's lens model: a wide CCTV lens needs a radial one">
                      <option value="simple_radial">Lens: radial (CCTV)</option>
                      <option value="pinhole">Lens: none (pinhole)</option>
                      <option value="radial">Lens: strong radial</option>
                    </select>}
                    <Button variant="ghost" onClick={() => setRef([])} disabled={!ref.length}>Clear</Button>
                  </div>
                </div>
              )}
            </Card>

            {zonesCard()}
          </div>
        </div>
        </Fold>
      )}

      {shown === "calibrated" && result && <ResultPanel result={result} label="zones (SAM + GeoCalib per zone)" expected={expected} />}
      {shown === "direct" && direct && <div data-testid="direct-result"><ResultPanel result={direct} label="direct bay model (no calibration)" expected={expected} /></div>}
    </div>
  );
}
