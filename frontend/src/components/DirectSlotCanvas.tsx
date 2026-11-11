import { Check, Hand, Hexagon, MousePointer2, RectangleHorizontal, Save, Trash2, X, ZoomIn, ZoomOut } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { points } from "../lib";
import { type DrawnSlot, type Pt, addSlot, mapProblems, moveSlot, moveVertex, nearFirst, nextId, pt, rectPoly } from "../slotdraw";
import { BayAdjust, useBayHistory } from "./BayAdjust";
import { Button, Pill } from "./ui";
import { MAX_ZOOM, MIN_ZOOM, useZoomPan } from "./useZoomPan";

/** Draw fixed slots by hand on a sample picture, for the Direct bay model: rectangles and polygons you can select, move, reshape and delete, on a
 * canvas you can zoom (wheel or buttons) and pan (Pan tool, or the middle mouse button). Slots are shares of the picture (0..1), like every slot map.
 * Pure drawing: it keeps no state of its own about the pipeline, the parent sends the slots to the server when the model runs. */
type Tool = "rect" | "polygon" | "edit" | "pan";
type Drag = { kind: "vertex"; id: number; index: number; anchor?: Pt } | { kind: "slot"; id: number; last: Pt } | { kind: "rect"; start: Pt } | { kind: "pan"; x: number; y: number };

export const presetKey = (w: number, h: number) => `sample-direct-slots:${w}x${h}`;

/** A starter layout: 2 rows of 6 rectangles over the lower middle of the picture, to be moved onto the real bays. */
export function starterSlots(): DrawnSlot[] {
  const out: DrawnSlot[] = [];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 6; c++) {
      const x = 0.14 + c * 0.12;
      const y = 0.45 + r * 0.2;
      out.push({ id: r * 6 + c + 1, name: `Slot ${r * 6 + c + 1}`, poly: [pt(x, y), pt(x + 0.1, y), pt(x + 0.1, y + 0.16), pt(x, y + 0.16)] });
    }
  }
  return out;
}

/** An axis-aligned rectangle as rectPoly draws it (or its transpose): its corners are resized as a rectangle, not bent into a quadrilateral. */
export function isRect(poly: Pt[]): boolean {
  if (poly.length !== 4) return false;
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-4;
  const [a, b, c, d] = poly;
  return (near(a[1], b[1]) && near(b[0], c[0]) && near(c[1], d[1]) && near(d[0], a[0])) || (near(a[0], b[0]) && near(b[1], c[1]) && near(c[0], d[0]) && near(d[1], a[1]));
}

export function DirectSlotCanvas({ image, width, height, slots, onChange }: {
  image: string;
  width: number;
  height: number;
  slots: DrawnSlot[];
  onChange: (slots: DrawnSlot[]) => void;
}) {
  const [tool, setTool] = useState<Tool>("rect");
  const [draft, setDraft] = useState<Pt[]>([]);
  const [cursor, setCursor] = useState<Pt | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [sel, setSel] = useState<number | null>(null);
    const [note, setNote] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const outer = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  const { view, zoomCentre, panBy, fit, transform } = useZoomPan(outer);
  const history = useBayHistory(slots, onChange); // Undo / Redo of every edit made here, the panel's included (50 deep)
  const problems = useMemo(() => mapProblems(slots), [slots]);

  const at = (e: { clientX: number; clientY: number }): Pt => {
    const r = inner.current!.getBoundingClientRect(); // includes the zoom and pan, so this is a share of the picture
    return pt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
  };
  const screenSize = (): [number, number] => {
    const r = inner.current?.getBoundingClientRect();
    return r ? [r.width, r.height] : [width, height];
  };

  const finish = (poly: Pt[]) => {
    const r = addSlot(slots, poly, `Slot ${nextId(slots)}`);
    if (r.error) setNote({ tone: "error", text: `Not added: ${r.error}.` });
    else {
      history.push();
      onChange(r.slots);
      setSel(r.slots[r.slots.length - 1].id);
      setNote(null);
    }
    setDraft([]);
  };

  const onDown = (e: React.MouseEvent) => {
    outer.current?.focus();
    if (e.button === 1 || (e.button === 0 && tool === "pan")) {
      e.preventDefault();
      return setDrag({ kind: "pan", x: e.clientX, y: e.clientY });
    }
    if (e.button !== 0) return;
    const p = at(e);
    if (tool === "rect") setDrag({ kind: "rect", start: p });
    if (tool === "polygon") {
      if (nearFirst(draft, p, screenSize())) finish(draft);
      else setDraft((d) => [...d, p]);
    }
    if (tool === "edit") setSel(null);
  };
  const onMove = (e: React.MouseEvent) => {
    setCursor(at(e));
    if (drag?.kind === "pan") {
      panBy(e.clientX - drag.x, e.clientY - drag.y);
      return setDrag({ ...drag, x: e.clientX, y: e.clientY });
    }
    const p = at(e);
    if (drag?.kind === "vertex") {
      const a = drag.anchor;
      onChange(a ? slots.map((s) => (s.id === drag.id ? { ...s, poly: rectPoly(a, p) } : s)) : moveVertex(slots, drag.id, drag.index, p));
    }
    if (drag?.kind === "slot") {
      onChange(moveSlot(slots, drag.id, p[0] - drag.last[0], p[1] - drag.last[1]));
      setDrag({ ...drag, last: p });
    }
  };
  const onUp = (e: React.MouseEvent) => {
    if (drag?.kind === "rect") {
      const end = at(e);
      const [sw, sh] = screenSize();
      if (Math.abs(end[0] - drag.start[0]) * sw > 4 && Math.abs(end[1] - drag.start[1]) * sh > 4) finish(rectPoly(drag.start, end));
    }
    setDrag(null);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && draft.length >= 3) finish(draft);
    else if (e.key === "Escape") setDraft([]);
    else if ((e.key === "Backspace" || (e.key === "z" && (e.ctrlKey || e.metaKey))) && draft.length) {
      e.preventDefault();
      setDraft((d) => d.slice(0, -1));
    } else if ((e.key === "Delete" || e.key === "Backspace") && sel != null && !draft.length) {
      e.preventDefault();
      remove(sel);
    }
  };
  const remove = (id: number) => { history.push(); onChange(slots.filter((s) => s.id !== id)); setSel(null); };

  const save = () => {
    if (problems.length) return setNote({ tone: "error", text: `Not saved: ${problems[0]}` });
    try {
      localStorage.setItem(presetKey(width, height), JSON.stringify(slots));
      setNote({ tone: "ok", text: `Saved ${slots.length} slot(s) in this browser for ${width}×${height} pictures.` });
    } catch {
      setNote({ tone: "error", text: "The browser would not store them." });
    }
  };
  const loadPreset = () => {
    let got: DrawnSlot[] | null = null;
    try {
      const raw = localStorage.getItem(presetKey(width, height));
      if (raw) got = JSON.parse(raw) as DrawnSlot[];
    } catch { /* a damaged entry: fall back to the starter layout */ }
    const used = got && got.length > 0 && !mapProblems(got).length ? got : null;
    history.push();
    onChange(used ?? starterSlots());
    setSel(null);
    setDraft([]);
    setNote({ tone: "ok", text: used ? `Loaded your saved ${used.length} slot(s).` : "Loaded a starter layout of 12 slots: move them onto the real bays (Edit tool)." });
  };

  const w = width;
  const h = height;
  const dot = w / 220 / view.s;
  const font = w / 70 / view.s;
  const rect = drag?.kind === "rect" && cursor ? rectPoly(drag.start, cursor) : null;
  const cur = tool === "pan" || drag?.kind === "pan" ? (drag?.kind === "pan" ? "grabbing" : "grab") : tool === "edit" ? "default" : "crosshair";

  return (
    <div className="space-y-2" data-testid="draw-canvas">
      <div className="flex flex-wrap items-center gap-1.5">
        <Pill active={tool === "rect"} onClick={() => (setTool("rect"), setDraft([]))} title="Drag a rectangle over a bay"><RectangleHorizontal size={12} /> Rectangle</Pill>
        <Pill active={tool === "polygon"} onClick={() => (setTool("polygon"), setDraft([]))} title="Click each corner; click the first one, double-click or press Enter to finish"><Hexagon size={12} /> Polygon</Pill>
        <Pill active={tool === "edit"} onClick={() => (setTool("edit"), setDraft([]))} title="Click a slot to select it; drag it to move it or its corners to reshape it"><MousePointer2 size={12} /> Select / edit</Pill>
        <Pill active={tool === "pan"} onClick={() => (setTool("pan"), setDraft([]))} title="Drag the picture (the middle mouse button does this in every tool)"><Hand size={12} /> Pan</Pill>
        <span className="mx-1 h-4 w-px bg-line" />
        <Button variant="outline" onClick={() => zoomCentre(1.5)} disabled={view.s >= MAX_ZOOM} data-testid="draw-zoom-in" title="Zoom in (or the mouse wheel)"><ZoomIn size={13} /></Button>
        <Button variant="outline" onClick={() => zoomCentre(1 / 1.5)} disabled={view.s <= MIN_ZOOM} data-testid="draw-zoom-out" title="Zoom out"><ZoomOut size={13} /></Button>
        <Button variant="ghost" onClick={fit} disabled={view.s === 1} data-testid="draw-zoom-reset">Fit</Button>
        <span className="text-xs text-muted" data-testid="draw-zoom">{Math.round(view.s * 100)} %</span>
      </div>

      <div ref={outer} tabIndex={0} onKeyDown={onKey} className="relative w-full overflow-hidden bg-stage outline-none select-none focus:ring-2 focus:ring-primary/40"
           style={{ aspectRatio: `${w} / ${h}` }} data-testid="draw-viewport">
        <div ref={inner} className="absolute top-0 left-0 h-full w-full" style={{ transform, transformOrigin: "0 0" }}
             data-testid="draw-inner">
          <img src={image} alt="Picture to draw the slots on" className="absolute inset-0 h-full w-full object-fill" draggable={false} />
          <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full" style={{ cursor: cur }}
               onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={() => (setCursor(null), setDrag(null))}
               onDoubleClick={() => tool === "polygon" && draft.length >= 3 && finish(draft)} onContextMenu={(e) => e.preventDefault()} data-testid="draw-svg">
            {slots.map((s) => {
              const on = s.id === sel;
              const lx = Math.min(...s.poly.map((p) => p[0])) * w;
              const ly = Math.min(...s.poly.map((p) => p[1])) * h;
              return (
                <g key={s.id} data-testid="draw-slot" data-id={s.id}>
                  <polygon points={points(s.poly, w, h)} fill={on ? "rgba(249,177,21,0.28)" : "rgba(79,70,229,0.18)"} stroke={on ? "#f9b115" : "#4f46e5"} strokeWidth={2}
                           vectorEffect="non-scaling-stroke" style={{ cursor: tool === "edit" ? "move" : undefined }}
                           onMouseDown={(e) => { if (tool === "edit" && e.button === 0) (e.stopPropagation(), outer.current?.focus(), setSel(s.id), history.push(), setDrag({ kind: "slot", id: s.id, last: at(e) })); }} />
                  <text x={lx + font * 0.3} y={ly + font * 1.1} fontSize={font} fontWeight={700} fill="white" stroke="black" strokeWidth={font / 12} paintOrder="stroke"
                        pointerEvents="none">{s.name}</text>
                  {tool !== "pan" && on && s.poly.map(([x, y], i) => (
                    <circle key={i} cx={x * w} cy={y * h} r={dot} fill="white" stroke="#f9b115" strokeWidth={2} vectorEffect="non-scaling-stroke" style={{ cursor: "grab" }}
                            data-testid="draw-vertex" onMouseDown={(e) => { if (e.button === 0) (e.stopPropagation(), history.push(), setDrag({ kind: "vertex", id: s.id, index: i, ...(isRect(s.poly) ? { anchor: s.poly[(i + 2) % 4] } : {}) })); }} />
                  ))}
                </g>
              );
            })}
            {draft.length > 0 && (
              <g pointerEvents="none">
                <polyline points={points(cursor ? [...draft, cursor] : draft, w, h)} fill="rgba(88,86,214,0.15)" stroke="#5856d6" strokeWidth={2} vectorEffect="non-scaling-stroke" strokeDasharray="6 4" />
                {draft.map(([x, y], i) => <circle key={i} cx={x * w} cy={y * h} r={dot * (i === 0 && draft.length >= 3 ? 1.4 : 1)} fill={i === 0 ? "#5856d6" : "white"} stroke="#5856d6" strokeWidth={2} vectorEffect="non-scaling-stroke" />)}
              </g>
            )}
            {rect && <polygon points={points(rect, w, h)} fill="rgba(88,86,214,0.15)" stroke="#5856d6" strokeWidth={2} vectorEffect="non-scaling-stroke" strokeDasharray="6 4" pointerEvents="none" />}
          </svg>
        </div>
        {cursor && (
          <span className="pointer-events-none absolute right-2 bottom-2 rounded bg-stage/75 px-2 py-0.5 font-mono text-[11px] text-white" data-testid="draw-readout">
            {Math.round(cursor[0] * w)}, {Math.round(cursor[1] * h)} px
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
        {tool === "rect" && <span>Drag a rectangle over a bay.</span>}
        {tool === "polygon" && <span>Click the corners; finish with the first corner, a double-click or Enter. Esc cancels, Backspace removes the last corner.</span>}
        {tool === "edit" && <span>Click a slot to select it; drag it to move it, drag a corner to resize or reshape it (a rectangle stays a rectangle). Delete removes it.</span>}
        {tool === "pan" && <span>Drag the picture. Wheel zooms about the pointer.</span>}
        {draft.length > 0 && (
          <>
            <span className="font-medium text-ink">{draft.length} corner(s)</span>
            <Button variant="outline" disabled={draft.length < 3} onClick={() => finish(draft)}><Check size={13} /> Finish slot</Button>
            <Button variant="ghost" onClick={() => setDraft([])}><X size={13} /> Cancel</Button>
          </>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-ink" data-testid="draw-count">{slots.length} slot(s) drawn</span>
        <Button variant="outline" onClick={save} disabled={!slots.length} data-testid="draw-save" title="Remember these slots in this browser for pictures of the same size"><Save size={13} /> Save slots</Button>
        <Button variant="outline" onClick={loadPreset} data-testid="draw-preset" title="Your saved slots for this picture size, or a starter layout to move onto the bays">Load default slot preset</Button>
        <Button variant="ghost" onClick={() => remove(sel!)} disabled={sel == null} data-testid="draw-delete"><Trash2 size={13} /> Delete selected</Button>
        <Button variant="ghost" onClick={() => (history.push(), onChange([]), setSel(null), setDraft([]), setNote(null))} disabled={!slots.length} data-testid="draw-clear">Clear all</Button>
      </div>
      {slots.length > 0 && (
        <ul className="max-h-40 divide-y divide-line overflow-y-auto rounded-md border border-line" data-testid="draw-list">
          {slots.map((s) => (
            <li key={s.id} className={`flex items-center gap-2 px-2.5 py-1 text-sm ${s.id === sel ? "bg-warning/10" : ""}`} data-testid="draw-row">
              <span className="flex-1">{s.name} <span className="text-xs text-muted">· {isRect(s.poly) ? "rectangle" : `${s.poly.length} corners`}</span></span>
              <button type="button" className="rounded px-1.5 py-0.5 text-xs font-medium text-primary hover:bg-primary/10" data-testid="draw-row-edit"
                      title="Select it and switch to the edit tool: move it, or drag its corners" onClick={() => (setSel(s.id), setTool("edit"), setDraft([]))}>Edit</button>
              <button type="button" title="Delete this slot" className="rounded p-1 text-muted hover:bg-danger/10 hover:text-danger" data-testid="draw-row-delete" onClick={() => remove(s.id)}>
                <Trash2 size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}
      {(slots.length > 0 || history.undoN > 0) && (
        <BayAdjust slots={slots} selected={sel} group={{ ids: slots.map((s) => s.id), label: "Whole map" }} w={w} h={h} onChange={onChange} history={history} ns="draw"
                   active={draft.length === 0} title="Edit bays: slot adjustment" />
      )}
      {problems.length > 0 && <p className="text-xs text-danger" data-testid="draw-problem">{problems[0]}</p>}
      {note && <p className={`rounded-md px-3 py-1.5 text-xs ${note.tone === "ok" ? "bg-success/10 text-success-ink" : "bg-danger/10 text-danger"}`} data-testid="draw-note">{note.text}</p>}
    </div>
  );
}
