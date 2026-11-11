import { Check, Hexagon, Pin, RectangleHorizontal, Save, Trash2, Undo2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ApiError, api, withCam } from "../api";
import { baySource, points } from "../lib";
import { type DrawnSlot, type Pt, addSlot, mapProblems, nearFirst, nextId, pt, rectPoly, setCapacity } from "../slotdraw";
import { CapacityField } from "./CapacityField";
import type { Snapshot } from "../types";
import { Button, Card, Pill, inputClass } from "./ui";

interface Saved {
  camera: string;
  updated: string | null;
  expected: number | null;
  pins: DrawnSlot[];
  max_extrapolated: number;
}

const PIN = "#8e44ad";

/** The auto grid's operator input (Dynamic mode on a HIGH calibration): how many bays the lot has, and bays drawn on the image
 * where the calibration cannot see (a far end beyond the camera's resolution). The grid is drawn under the pins as a guide:
 * measured bays solid, extrapolated ones dotted. A pin replaces the grid bay it overlaps, or adds one; the worker applies a save
 * from its next frame (no restart). Corners are shares of the frame, like every slot map. */
export function AutoGridPins({ cam, snapshot, image, tokenKey, onNeedAdmin }: {
  cam: string | null;
  snapshot: Snapshot | undefined;
  image: string | null; // the editor's frozen still (null while it is being captured)
  tokenKey: string;
  onNeedAdmin: (reason: string, retry?: () => void) => void;
}) {
  const [saved, setSaved] = useState<Saved | null>(null);
  const [expected, setExpected] = useState("");
  const [pins, setPins] = useState<DrawnSlot[]>([]);
  const [mode, setMode] = useState<"rect" | "polygon">("rect");
  const [draft, setDraft] = useState<Pt[]>([]);
  const [start, setStart] = useState<Pt | null>(null);
  const [cursor, setCursor] = useState<Pt | null>(null);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const [w, h] = snapshot?.shape ?? [2560, 1440];
  const auto = snapshot?.auto_slots;

  const load = useCallback(() => {
    api.get<Saved>(withCam("/api/auto-pins", cam)).then((r) => {
      const pins = Array.isArray(r.pins) ? r.pins : []; // (an older backend has no such endpoint)
      setSaved({ ...r, pins, expected: r.expected ?? null });
      setExpected(r.expected == null ? "" : String(r.expected));
      setPins(pins);
      setDraft([]);
      setStart(null);
    }).catch((e) => setMessage({ tone: "error", text: e.message }));
  }, [cam]);
  useEffect(() => load(), [load, tokenKey]);

  const count = expected.trim() === "" ? null : Number(expected);
  const countProblem = count != null && (!Number.isInteger(count) || count < 1 || count > 64) ? "expected bays: a whole number from 1 to 64" : null;
  const problems = useMemo(() => [...(countProblem ? [countProblem] : []), ...mapProblems(pins)], [countProblem, pins]);
  const dirty = JSON.stringify({ e: count, p: pins }) !== JSON.stringify({ e: saved?.expected ?? null, p: saved?.pins ?? [] });

  const at = (e: { clientX: number; clientY: number }): Pt => {
    const r = box.current!.getBoundingClientRect();
    return pt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
  };
  const finish = (poly: Pt[]) => {
    const r = addSlot(pins, poly, `P${nextId(pins)}`);
    if (r.error) setMessage({ tone: "error", text: `Not added: ${r.error}.` });
    else (setPins(r.slots), setMessage(null));
    setDraft([]);
  };
  const onDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const p = at(e);
    if (mode === "rect") setStart(p);
    else if (nearFirst(draft, p, [box.current?.clientWidth ?? w, box.current?.clientHeight ?? h])) finish(draft);
    else setDraft((d) => [...d, p]);
  };
  const onUp = (e: React.MouseEvent) => {
    if (mode === "rect" && start) {
      const end = at(e);
      const r = box.current!.getBoundingClientRect();
      if (Math.abs(end[0] - start[0]) * r.width > 4 && Math.abs(end[1] - start[1]) * r.height > 4) finish(rectPoly(start, end));
    }
    setStart(null);
  };

  const save = async () => {
    try {
      const r = await api.put<Saved>(withCam("/api/auto-pins", cam), { expected: count, pins });
      setMessage({ tone: "ok", text: `Saved: ${r.expected ?? "no"} expected bays, ${r.pins.length} pin(s). The grid is rebuilt from the next frame.` });
      load();
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 401 || err.status === 503) onNeedAdmin("Saving the auto grid's bays needs the admin token.", save);
      else setMessage({ tone: "error", text: `Not saved: ${err.message}` });
    }
  };

  const rect = mode === "rect" && start && cursor ? rectPoly(start, cursor) : null;
  const guide = snapshot?.slots ?? [];
  return (
    <Card title={<span className="flex items-center gap-2"><Pin size={16} className="text-primary" /> Auto grid · expected bays and pins</span>}
          actions={
            <>
              <Pill active={mode === "rect"} onClick={() => (setMode("rect"), setDraft([]))} title="Drag a rectangle over a bay"><RectangleHorizontal size={12} /> Rectangle</Pill>
              <Pill active={mode === "polygon"} onClick={() => (setMode("polygon"), setDraft([]))} title="Click each corner; click the first one to finish"><Hexagon size={12} /> Polygon</Pill>
            </>
          } bodyClass="p-0">
      <div ref={box} className="relative w-full select-none bg-ink" style={{ aspectRatio: `${w} / ${h}` }} data-testid="pins-surface">
        {image ? <img src={image} alt="Camera snapshot" className="absolute inset-0 h-full w-full object-fill" draggable={false} />
          : <div className="absolute inset-0 flex items-center justify-center text-sm text-white/70">Capturing a snapshot from the camera…</div>}
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full cursor-crosshair"
             onMouseDown={onDown} onMouseMove={(e) => setCursor(at(e))} onMouseUp={onUp}
             onMouseLeave={() => (setCursor(null), setStart(null))}
             onDoubleClick={() => mode === "polygon" && draft.length >= 3 && finish(draft)} data-testid="pins-svg">
          {guide.map((s) => {
            const src = baySource(auto, s.id);
            return src === "manual" ? null : (
              <polygon key={s.id} points={points(s.poly, w, h)} fill="rgba(255,255,255,0.06)" stroke="#ffffff" strokeOpacity={0.8}
                       strokeWidth={1.5} vectorEffect="non-scaling-stroke" strokeDasharray={src === "extrapolated" ? "2 4" : undefined}
                       pointerEvents="none" data-testid="guide-bay" data-source={src ?? undefined} />
            );
          })}
          {pins.map((p) => (
            <g key={p.id} pointerEvents="none" data-testid="pin">
              <polygon points={points(p.poly, w, h)} fill={`${PIN}40`} stroke={PIN} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
              <text x={Math.min(...p.poly.map((q) => q[0])) * w + w / 250} y={Math.min(...p.poly.map((q) => q[1])) * h + w / 70}
                    fontSize={w / 70} fontWeight={700} fill="white" stroke="black" strokeWidth={w / 900} paintOrder="stroke">{p.name}</text>
            </g>
          ))}
          {draft.length > 0 && (
            <polyline points={points(cursor ? [...draft, cursor] : draft, w, h)} fill="rgba(142,68,173,0.15)" stroke={PIN}
                      strokeWidth={2} vectorEffect="non-scaling-stroke" strokeDasharray="6 4" pointerEvents="none" />
          )}
          {rect && <polygon points={points(rect, w, h)} fill="rgba(142,68,173,0.15)" stroke={PIN} strokeWidth={2}
                            vectorEffect="non-scaling-stroke" strokeDasharray="6 4" pointerEvents="none" />}
        </svg>
      </div>
      <div className="space-y-3 p-4">
        <p className="text-xs text-muted" data-testid="pins-hint">
          White outlines: the grid now ({auto ? `${auto.count} bays${auto.extrapolated ? `, ${auto.extrapolated} dotted = extrapolated` : ""}` : "no auto grid: needs a HIGH-confidence calibration in Dynamic mode"}).
          Draw a bay where a vehicle stands that the grid misses; it replaces the grid bay it overlaps, or adds one.
        </p>
        <label className="block">
          <span className="mb-1 block text-xs font-semibold tracking-wide text-muted uppercase">Expected bays (this camera's lot)</span>
          <input className={inputClass} inputMode="numeric" placeholder="not set (the lane gets one extra bay)" value={expected}
                 onChange={(e) => setExpected(e.target.value)} data-testid="expected-bays" />
          <span className="mt-1 block text-xs text-muted">The lane is continued past the last measured bay until this many exist (at most {saved?.max_extrapolated ?? 3} added).</span>
        </label>
        {auto?.short ? (
          <p className="rounded-md bg-warning/15 px-3 py-2 text-xs text-ink" data-testid="pins-short">
            {auto.short} of {auto.expected} expected bays could not be placed: draw them as pins.
          </p>
        ) : null}
        {draft.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="font-medium text-ink">{draft.length} corner(s)</span>
            <Button variant="outline" onClick={() => setDraft((d) => d.slice(0, -1))}><Undo2 size={13} /> Undo</Button>
            <Button variant="outline" disabled={draft.length < 3} onClick={() => finish(draft)}><Check size={13} /> Finish pin</Button>
            <Button variant="ghost" onClick={() => setDraft([])}><X size={13} /> Cancel</Button>
          </div>
        )}
        <ul className="max-h-48 divide-y divide-line overflow-y-auto rounded-md border border-line" data-testid="pin-list">
          {pins.map((p) => (
            <li key={p.id} className="flex items-center gap-2 px-2.5 py-1.5">
              <span className="w-6 text-xs font-semibold text-muted">{p.id}</span>
              <input className="min-w-0 flex-1 rounded border border-transparent bg-transparent px-1 py-0.5 text-sm hover:border-line focus:border-primary focus:outline-none"
                     value={p.name} onChange={(e) => setPins((all) => all.map((x) => (x.id === p.id ? { ...x, name: e.target.value } : x)))} />
              <CapacityField value={p.capacity} onChange={(t) => setPins((all) => setCapacity(all, p.id, t))} />
              <button type="button" title="Delete" className="rounded p-1 text-muted hover:bg-danger/10 hover:text-danger"
                      onClick={() => setPins((all) => all.filter((x) => x.id !== p.id))}><Trash2 size={14} /></button>
            </li>
          ))}
          {!pins.length && <li className="px-3 py-4 text-center text-sm text-muted">No pins: the grid is the calibration's.</li>}
        </ul>
        {problems.length > 0 && <p className="text-xs text-danger">{problems[0]}</p>}
        {message && <p className={`rounded-md px-3 py-2 text-sm ${message.tone === "ok" ? "bg-success/10 text-success-ink" : "bg-danger/10 text-danger"}`}>{message.text}</p>}
        <div className="flex flex-wrap gap-2">
          <Button onClick={save} disabled={!dirty || problems.length > 0}><Save size={14} /> Save</Button>
          <Button variant="ghost" disabled={!dirty} onClick={load}>Revert</Button>
        </div>
      </div>
    </Card>
  );
}
