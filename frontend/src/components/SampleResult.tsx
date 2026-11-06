import { useRef, useState } from "react";
import { Hand, ZoomIn, ZoomOut } from "lucide-react";
import { BLOCK_HATCH, ENCROACH_HATCH, SLOT_COLORS, baySource, outsidePath } from "../lib";
import type { Detection, ParkingEvent, Slot, Snapshot } from "../types";
import { BoxShape, SlotShape } from "./LiveView";
import { Counters, EventTicker } from "./SidePanel";
import { BlockHatch, Button, Card, EncroachHatch } from "./ui";
import { MAX_ZOOM, MIN_ZOOM, useZoomPan } from "./useZoomPan";

/** The server's answer to a sample run: the dashboard's own data for one picture, plus what the run logged (in a throw-away log). */
export interface SampleResult {
  shape: [number, number];
  counts: NonNullable<Snapshot["counts"]>;
  slots: Slot[];
  detections: Detection[];
  area: Snapshot["area"];
  vehicles: number;
  others: number;
  obstacle?: { rejected: number; reasons: Record<string, number> };
  events: ParkingEvent[]; // oldest first
  layers: { spillage: boolean; obstacle_filter: boolean; blocked_objects: boolean; ground: boolean };
  frames: number;
  step_s: number;
  seconds: number;
  kind: "fixed" | "auto" | "session" | "direct"; // "direct": the bay model straight on the picture, no calibration
  slot_count: number;
  source: "upload" | "sample" | "session";
  name: string | null;
  image: string; // data URL of the picture that was tested
  tune?: { model: string; conf: number; iou: number; footprint: number; encroach: number }; // the settings this run used (a session run)
  detector?: string;
  detected?: number; // vehicles the detector found, before the parking surface cut some off
  unplaced?: { conf: number; why: string }[]; // vehicles that hold no bay (and why), a session run
  direct?: { conf: number; iou: number; drawn?: boolean; by?: Record<string, number> }; // a direct bay-model run: the settings it used
  layout?: "auto" | "zones"; // a session run: the auto grid, or the bays of its row zones
}

export type Layer = "spillage" | "obstacle_filter" | "blocked_objects";
export const LAYERS: [Layer, string, string][] = [
  ["spillage", "Spillage protection", "A free bay a neighbour's vehicle stands on is ENCROACHED"],
  ["obstacle_filter", "Obstacle filter", "Cones, shadows, other non-vehicle boxes do not make a bay BUSY"],
  ["blocked_objects", "Blocked by object", "A free bay with a pot, vase, box ... in it is BLOCKED"],
];
export const MAX_BYTES = 12_000_000;
export type Expected = { free: string; busy: string; blocked: string };
export const EXPECT: [keyof Expected, string][] = [["free", "Free"], ["busy", "Busy"], ["blocked", "Blocked"]];

export const readFile = (f: File) => new Promise<string>((ok, bad) => {
  const r = new FileReader();
  r.onload = () => ok(String(r.result));
  r.onerror = () => bad(new Error("could not read the file"));
  r.readAsDataURL(f);
});

/** What the cards say for a result, to compare with what the picture shows: usable free bays, busy bays, bays blocked by an object. */
export const observed = (c: SampleResult["counts"]) => ({ free: c.available ?? c.free, busy: c.busy, blocked: c.blocked_objects ?? 0 });

/** The three protection switches of a test run. */
export function LayerBoxes({ layers, onChange }: { layers: Record<Layer, boolean>; onChange: (l: Record<Layer, boolean>) => void }) {
  return (
    <div className="mt-3 flex flex-wrap gap-4" data-testid="sample-layers">
      {LAYERS.map(([k, label, hint]) => (
        <label key={k} className="flex items-center gap-1.5 text-sm" title={hint}>
          <input type="checkbox" checked={layers[k]} onChange={(e) => onChange({ ...layers, [k]: e.target.checked })} data-testid={`sample-${k}`} />
          {label}
        </label>
      ))}
    </div>
  );
}

/** Optional expected counts: what the picture is known to show, checked against the cards. */
export function ExpectedInputs({ value, onChange }: { value: Expected; onChange: (e: Expected) => void }) {
  return (
    <div className="flex items-end gap-2" title="Optional: what you know the picture to show; the result is checked against it">
      <span className="pb-1.5 text-xs font-semibold tracking-wide text-muted uppercase">Expected</span>
      {EXPECT.map(([k, label]) => (
        <label key={k} className="block text-xs text-muted">
          {label}
          <input className="ml-1 w-14 rounded-md border border-line px-1.5 py-1 text-sm text-ink" inputMode="numeric" value={value[k]}
                 onChange={(e) => onChange({ ...value, [k]: e.target.value.replace(/\D/g, "") })} data-testid={`expect-${k}`} />
        </label>
      ))}
    </div>
  );
}

/** The result of a run, drawn like the live dashboard: the picture with the slots coloured, the cards, the notices and a slot table; `expected` (optional
 * counts the picture is known to show) is checked against the cards. Shared by the camera-slots test and the picture sessions. */
export function ResultPanel({ result, label, expected }: { result: SampleResult; label: string; expected: Expected }) {
  const [hover, setHover] = useState<{ x: number; y: number; title: string; lines: string[] } | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const { view, zoomCentre, panBy, fit, transform } = useZoomPan(stage); // the wheel zooms about the pointer, a drag moves the picture
  const [grab, setGrab] = useState(false);
  const [w, h] = result.shape;
  const show = (e: React.MouseEvent, title: string, lines: string[]) => {
    const box = stage.current?.getBoundingClientRect();
    if (box) setHover({ x: e.clientX - box.left, y: e.clientY - box.top, title, lines });
  };
  const snapshot = { counts: result.counts, slots: result.slots, shape: result.shape } as unknown as Snapshot;
  const seen = observed(result.counts);
  const checks = EXPECT.filter(([k]) => expected[k].trim() !== "").map(([k, name]) => ({ k, label: name, want: Number(expected[k]), got: seen[k] }));
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,1.4fr)]">
      <Card title={<span>Result · {label}</span>} bodyClass="p-0" actions={
        <>
          <Button variant="outline" onClick={() => zoomCentre(1.5)} disabled={view.s >= MAX_ZOOM} data-testid="res-zoom-in" title="Zoom in (or the mouse wheel)"><ZoomIn size={13} /></Button>
          <Button variant="outline" onClick={() => zoomCentre(1 / 1.5)} disabled={view.s <= MIN_ZOOM} data-testid="res-zoom-out" title="Zoom out"><ZoomOut size={13} /></Button>
          <Button variant="ghost" onClick={fit} disabled={view.s === 1} data-testid="res-zoom-reset">Fit</Button>
          <span className="text-xs text-muted" data-testid="res-zoom">{Math.round(view.s * 100)} %</span>
          <span className="flex items-center gap-1 text-xs text-muted"><Hand size={12} /> drag to move</span>
        </>
      }>
        <div ref={stage} className="relative w-full select-none overflow-hidden bg-ink" style={{ aspectRatio: `${w} / ${h}`, cursor: view.s > 1 ? (grab ? "grabbing" : "grab") : undefined }}
             onMouseLeave={() => (setHover(null), setGrab(false))} onMouseDown={(e) => e.button === 0 && setGrab(true)} onMouseUp={() => setGrab(false)}
             onMouseMove={(e) => grab && view.s > 1 && panBy(e.movementX, e.movementY)} data-testid="sample-stage">
         <div className="absolute top-0 left-0 h-full w-full" style={{ transform, transformOrigin: "0 0" }} data-testid="res-layer">
          <img src={result.image} alt="Sample picture" className="absolute inset-0 h-full w-full object-fill" draggable={false} />
          <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full" shapeRendering="geometricPrecision"
               data-testid="sample-overlay">
            <EncroachHatch size={Math.max(6, w / 140)} />
            <BlockHatch size={Math.max(8, w / 120)} />
            <path d={outsidePath(result.area, w, h)} fill="rgba(0,0,0,0.45)" fillRule="evenodd" />
            {result.slots.map((s) => <SlotShape key={s.id} slot={s} w={w} h={h} ghost={false} onHover={(e, t, l) => show(e, t, l)} source={baySource(undefined, s.id)} />)}
            {result.detections.map((d, i) => <BoxShape key={i} det={d} w={w} h={h} ghost={false} onHover={(e, t, l) => show(e, t, l)} />)}
          </svg>
         </div>
          {hover && (
            <div className="pointer-events-none absolute z-20 max-w-xs rounded-md bg-ink/90 px-2.5 py-1.5 text-xs text-white shadow"
                 style={{ left: hover.x + 12, top: hover.y + 12 }}>
              <p className="font-semibold">{hover.title}</p>
              {hover.lines.map((l) => <p key={l} className="text-white/80">{l}</p>)}
            </div>
          )}
        </div>
        <p className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2 text-xs text-muted" data-testid="sample-legend">
          <Swatch color={SLOT_COLORS.FREE.stroke} label="Free" />
          <Swatch color={SLOT_COLORS.BUSY.stroke} label="Busy" />
          <Swatch color={SLOT_COLORS.BLOCKED.stroke} label="Blocked by object (cross-hatched)" hatch={BLOCK_HATCH} />
          <Swatch color={SLOT_COLORS.ENCROACHED.stroke} label="Encroached" hatch={ENCROACH_HATCH} />
          <span>yellow = vehicle · magenta = other object · grey dashed = ignored</span>
        </p>
      </Card>
      <div className="space-y-4">
        <Counters snapshot={snapshot} />
        {checks.length > 0 && (
          <div className="flex flex-wrap gap-2" data-testid="sample-checks">
            {checks.map((c) => (
              <span key={c.k} data-ok={c.want === c.got ? "1" : "0"}
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${c.want === c.got ? "bg-success/15 text-[#1b7a3e]" : "bg-danger/10 text-danger"}`}>
                {c.want === c.got ? "✓" : "✗"} {c.label}: expected {c.want}, got {c.got}
              </span>
            ))}
          </div>
        )}
        {result.kind === "direct" ? (
          <p className="text-xs text-muted" data-testid="sample-summary">
            {result.detector} at conf {result.direct?.conf}, IoU {result.direct?.iou} in {result.seconds} s ·{" "}
            {result.direct?.drawn
              ? `${result.slot_count} drawn slot(s): ${result.counts.busy} busy, ${result.counts.free} free, ${result.counts.unknown} unknown; decided by ${Object.entries(result.direct.by ?? {}).map(([k, n]) => `${n} ${k}`).join(", ") || "nothing"}. No calibration, grid or layers.`
              : `${result.slot_count} bays found by the model itself (no calibration, no slot grid, no layers): its boxes, not your slots. Compare with the calibrated result.`}
          </p>
        ) : (
        <>
        <p className="text-xs text-muted" data-testid="sample-summary">
          {result.vehicles} vehicle(s){result.detected != null && result.detected > result.vehicles ? ` counted (${result.detected} found, ${result.detected - result.vehicles} outside the parking surface or dropped by the filter)` : ""},
          {" "}{result.others} other object(s) in {result.seconds} s{result.tune ? ` · ${result.detector} at conf ${result.tune.conf}, IoU ${result.tune.iou}, blocked ${Math.round(result.tune.footprint * 100)} %, encroach ${Math.round(result.tune.encroach * 100)} %` : ""} · {result.slot_count} slots ({result.kind === "session" ? (result.layout === "zones" ? "zones, merged" : "calibrated picture") : result.kind === "auto" ? "auto grid" : "fixed map"}) ·
          layers: {LAYERS.filter(([k]) => result.layers[k]).map(([, l]) => l).join(", ") || "none"}{result.layers.ground ? " (ground metric)" : ""} ·
          settled over {result.frames} replayed frames
          {result.obstacle ? ` · obstacle filter dropped ${result.obstacle.rejected}` : ""}
          {result.kind === "session" && result.slots.some((x) => x.id >= 1000)
            ? ` · ${result.slots.filter((x) => x.id >= 1000).length} bay(s) added for vehicles off the grid (D1, D2 …)` : ""}
        </p>
        {result.unplaced && result.unplaced.length > 0 && (
          <p className="text-xs text-muted" data-testid="sample-unplaced">
            {result.unplaced.length} vehicle(s) hold no bay: {Object.entries(result.unplaced.reduce<Record<string, number>>((m, u) => ({ ...m, [u.why]: (m[u.why] ?? 0) + 1 }), {}))
              .map(([why, n]) => `${n} ${why}`).join(", ")}
          </p>
        )}
        </>
        )}
        <EventTicker events={[...result.events].reverse()} />
        <Card title="Slots" bodyClass="p-0">
          <table className="w-full text-sm" data-testid="sample-slots">
            <tbody className="divide-y divide-line">
              {result.slots.map((s) => (
                <tr key={s.id} data-testid="sample-slot-row" data-display={s.display ?? s.state}>
                  <td className="px-3 py-1.5 font-medium">{s.name}{(s as { zone?: string }).zone ? <span className="ml-1 text-xs font-normal text-muted">· {(s as { zone?: string }).zone}</span> : null}</td>
                  <td className="px-3 py-1.5">{s.display ?? s.state}</td>
                  <td className="px-3 py-1.5 text-xs text-muted">{s.blocked_by?.length ? `blocked by ${s.blocked_by.join(", ")}` : s.encroached_by?.length ? `encroached by ${s.encroached_by.join(", ")}` : (s as { by?: string }).by ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  );
}

function Swatch({ color, label, hatch }: { color: string; label: string; hatch?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block h-3 w-5 rounded-sm border" style={{ borderColor: color, background: hatch ? `repeating-linear-gradient(45deg, ${color}66 0 2px, transparent 2px 5px)` : `${color}33` }} />
      {label}
    </span>
  );
}
