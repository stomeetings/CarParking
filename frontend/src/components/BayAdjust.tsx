import { ChevronDown, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { type DepthSpec, depthGroup, resizeBay, rotateBay, rotateGroup, shiftAlong, shiftBays, splitBay, tuneBays } from "../bayfine";
import { type DrawnSlot, type Pt, nextId } from "../slotdraw";
import { Button, Pill } from "./ui";

/** Undo / redo of bay edits: `current` is what is on screen, `restore` puts a snapshot back. push() is called BEFORE an edit (it keeps the picture as it
 * is), a new edit drops the redo list. The newest `limit` snapshots are kept. */
export function useBayHistory<T>(current: T, restore: (v: T) => void, limit = 50) {
  const [past, setPast] = useState<T[]>([]);
  const [future, setFuture] = useState<T[]>([]);
  const push = () => {
    setPast((p) => [...p.slice(-(limit - 1)), current]);
    setFuture([]);
  };
  const undo = () => {
    if (!past.length) return;
    setFuture((f) => [...f, current]);
    setPast((p) => p.slice(0, -1));
    restore(past[past.length - 1]);
  };
  const redo = () => {
    if (!future.length) return;
    setPast((p) => [...p.slice(-(limit - 1)), current]);
    setFuture((f) => f.slice(0, -1));
    restore(future[future.length - 1]);
  };
  const clear = () => {
    setPast([]);
    setFuture([]);
  };
  return { push, undo, redo, clear, undoN: past.length, redoN: future.length };
}
export type BayHistory = ReturnType<typeof useBayHistory>;

/** Edit bays: the fine-adjustment panel shared by the Sample test's zones, its hand-drawn fixed slots and the Fixed slots editor. It works on `slots` (shares of
 * the picture) and hands every edited list to `onChange`; the history is the parent's (`history`), so a drag made on the canvas is undone the same way.
 *  - Move: screen arrows (also the arrow keys, Shift = 5x) and "along the bay" (its own row / depth directions, right for a slanted row).
 *  - Size and turn: each bay about its own centre (single bay, or every bay of the group one by one).
 *  - Whole group: the grid turns as one block about its middle (`rotateZone` can turn more than the bays, e.g. the zone's outline).
 *  - Split: the selected bay into two, left | right or front / back.
 *  - Undo / redo (Ctrl+Z, Ctrl+Y). */
export function BayAdjust({ slots, selected, group, w, h, onChange, history, ns, active = true, newId, rotateZone, depthZone, title = "Select zone & slot adjustment" }: {
  slots: DrawnSlot[];
  selected: number | null;
  group: { ids: number[]; label: string };
  w: number;
  h: number;
  onChange: (next: DrawnSlot[]) => void;
  history: BayHistory;
  ns: string;
  active?: boolean;
  /** The id of the second half of a split bay, or null when there is none to give (default: the next id of the list). */
  newId?: (b: DrawnSlot, all: DrawnSlot[]) => number | null;
  /** Turns the whole group by deg; false = could not (the panel then changes nothing). Default: the group's bays as one rigid block. */
  rotateZone?: (deg: number) => boolean;
  /** Changes the depth of the whole group (its back edge, the front line stays); false = could not. Default: the group's bays. */
  depthZone?: (spec: DepthSpec) => boolean;
  title?: string;
}) {
  const [target, setTarget] = useState<"bay" | "zone">("bay");
  const [step, setStep] = useState(1); // pixels per press
  const [turn, setTurn] = useState(0.5); // degrees per press
  const [note, setNote] = useState<string | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({ panel: true, move: true, turn: true, depth: false, size: false, split: false }); // folded sections stay mounted

  const ids = target === "zone" ? slots.filter((s) => group.ids.includes(s.id)).map((s) => s.id) : slots.filter((s) => s.id === selected).map((s) => s.id);
  const none = ids.length === 0;
  const edit = (next: DrawnSlot[]) => {
    history.push();
    onChange(next);
    setNote(null);
  };
  const shift = (dx: number, dy: number) => ids.length && edit(shiftBays(slots, ids, dx / w, dy / h));
  const along = (a: number, b: number) => ids.length && edit(shiftAlong(slots, ids, w, h, a, b));
  const tune = (fn: (poly: Pt[]) => Pt[]) => ids.length && edit(tuneBays(slots, ids, fn));
  const turnZone = (deg: number) => {
    if (rotateZone) {
      if (rotateZone(deg) === false) setNote("The zone cannot turn further: a corner would leave the picture.");
      else (history.push(), setNote(null));
      return;
    }
    const next = rotateGroup(slots, group.ids, w, h, deg);
    if (!next) setNote("The group cannot turn further: a corner would leave the picture.");
    else edit(next);
  };
  const depth = (spec: DepthSpec) => {
    if (depthZone) {
      if (depthZone(spec) === false) setNote("The back edge cannot move further: a corner would leave the picture, or the zone has no depth.");
      else (history.push(), setNote(null));
      return;
    }
    const r = depthGroup(slots, group.ids, w, h, spec);
    if (!r) setNote("The back edge cannot move further: a corner would leave the picture, or the bays have no depth.");
    else edit(r.slots);
  };
  const split = (axis: "width" | "length") => {
    const b = slots.find((x) => x.id === selected);
    if (!b) return;
    const id = newId ? newId(b, slots) : nextId(slots);
    if (id == null) return setNote("No free id for the second half: remove a bay first.");
    const [p1, p2] = splitBay(b.poly, axis, 2);
    if (!p2) return setNote("This bay cannot be split: it is not a four-cornered bay.");
    edit(slots.flatMap((x) => (x.id === b.id ? [{ ...b, name: `${b.name}a`, poly: p1 }, { ...b, id, name: `${b.name}b`, poly: p2 }] : [x])));
  };

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")) return; // typing is not nudging
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && !e.altKey && (k === "z" || k === "y")) {
        const redo = k === "y" || e.shiftKey;
        if (redo ? history.redoN : history.undoN) (e.preventDefault(), redo ? history.redo() : history.undo());
        return;
      }
      const d = { arrowleft: [-1, 0], arrowright: [1, 0], arrowup: [0, -1], arrowdown: [0, 1] }[k];
      if (!d || e.ctrlKey || e.metaKey || e.altKey || none) return;
      e.preventDefault();
      const n = step * (e.shiftKey ? 5 : 1);
      shift(d[0] * n, d[1] * n);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const b = (label: string, fn: () => unknown, id: string, tip: string, off = none) => (
    <Button key={id} variant="outline" onClick={() => void fn()} disabled={off} data-testid={`${ns}-${id}`} title={tip}>{label}</Button>
  );
  // sections open and close so the panel stays short beside the picture; closed ones stay mounted (hidden), so their buttons and keys keep working
  const sec = (id: string, label: string, body: React.ReactNode) => (
    <div className="border-t border-line pt-1" key={id}>
      <button type="button" className="flex w-full items-center gap-1 text-left text-xs font-semibold" onClick={() => setOpen({ ...open, [id]: !open[id] })}
              aria-expanded={!!open[id]} data-testid={`${ns}-sec-${id}`}>
        {open[id] ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{label}
      </button>
      <div hidden={!open[id]} className="mt-1 space-y-1.5" data-testid={`${ns}-sec-${id}-body`}>{body}</div>
    </div>
  );
  const row = "flex flex-wrap items-center gap-1.5";
  const setAll = (on: boolean) => setOpen(Object.fromEntries(Object.keys(open).map((k) => [k, k === "panel" ? true : on])));
  return (
    <div className="space-y-1.5 rounded-md border border-line p-2" data-testid={`${ns}-fine`}>
      <button type="button" className="flex w-full items-center gap-1 text-left text-xs font-semibold" onClick={() => setOpen({ ...open, panel: !open.panel })}
              aria-expanded={!!open.panel} data-testid={`${ns}-fine-toggle`} title="Fold the whole panel away">
        {open.panel ? <ChevronDown size={14} /> : <ChevronRight size={14} />}{title}
        {!open.panel && history.undoN > 0 && <span className="ml-auto font-normal text-muted">{history.undoN} edit(s)</span>}
      </button>
      <div hidden={!open.panel} className="space-y-1.5" data-testid={`${ns}-fine-body`}>
        <div className={row}>
          <Pill active={target === "bay"} onClick={() => setTarget("bay")} title="The bay selected in the list or on the picture">Selected bay{selected != null ? ` (${slots.find((s) => s.id === selected)?.name ?? "—"})` : ""}</Pill>
          <Pill active={target === "zone"} onClick={() => setTarget("zone")} title="Every bay of the group, moved or tuned together">{group.label}</Pill>
          <label className="ml-auto flex items-center gap-1 text-xs">Step
            <select className="rounded-md border border-line px-1 py-0.5" value={step} onChange={(e) => setStep(Number(e.target.value))} data-testid={`${ns}-fine-step`}>
              {[1, 2, 5, 10].map((n) => <option key={n} value={n}>{n} px</option>)}
            </select>
          </label>
        </div>
        <div className={row}>
          {b(`Undo${history.undoN ? ` (${history.undoN})` : ""}`, history.undo, "undo-edit", "Take back the last bay edit (Ctrl+Z)", history.undoN === 0)}
          {b(`Redo${history.redoN ? ` (${history.redoN})` : ""}`, history.redo, "redo-edit", "Do the edit again that Undo took back (Ctrl+Y)", history.redoN === 0)}
          <Button variant="ghost" onClick={() => setAll(false)} data-testid={`${ns}-fold-all`} title="Fold every section">Fold all</Button>
          <Button variant="ghost" onClick={() => setAll(true)} data-testid={`${ns}-open-all`} title="Open every section">Open all</Button>
        </div>
        {sec("move", "Move", <>
          <div className={row}>
            {([["←", -1, 0, "left"], ["↑", 0, -1, "up"], ["↓", 0, 1, "down"], ["→", 1, 0, "right"]] as const).map(([t, dx, dy, id]) => b(t, () => shift(dx * step, dy * step), `fine-${id}`, `Move ${id} by the step`))}
            <span className="text-xs text-muted">or the arrow keys (Shift = 5×)</span>
          </div>
          <div className={row}>
            <span className="text-xs text-muted">Along the bay:</span>
            {([["Row ◂", -1, 0, "row-left", "along its front, to its left"], ["Row ▸", 1, 0, "row-right", "along its front, to its right"],
               ["Back ↥", 0, 1, "back", "towards its back (away from the camera)"], ["Front ↧", 0, -1, "front", "towards its front (nearer the camera)"]] as const)
              .map(([t, a, c, id, tip]) => b(t, () => along(a * step, c * step), `fine-${id}`, `Move ${tip}`))}
          </div>
        </>)}
        {sec("turn", "Rotate", <div className={row} data-testid={`${ns}-turn`}>
          <label className="flex items-center gap-1 text-xs">by
            <select className="rounded-md border border-line px-1 py-0.5" value={turn} onChange={(e) => setTurn(Number(e.target.value))} data-testid={`${ns}-turn-step`}>
              {[0.1, 0.5, 1, 2, 5].map((n) => <option key={n} value={n}>{n}°</option>)}
            </select>
          </label>
          <span className="text-xs text-muted">each bay about its own centre:</span>
          {b("↺", () => tune((p) => rotateBay(p, w, h, -turn)), "fine-ccw", `Turn ${turn}° anticlockwise`)}
          {b("↻", () => tune((p) => rotateBay(p, w, h, turn)), "fine-cw", `Turn ${turn}° clockwise`)}
          <span className="text-xs text-muted">the whole group as one block:</span>
          {b("↺", () => turnZone(-turn), "turn-ccw", `Turn the whole group ${turn}° anticlockwise: all its bays${rotateZone ? " and its outline" : ""} swing together`, group.ids.length === 0)}
          {b("↻", () => turnZone(turn), "turn-cw", `Turn the whole group ${turn}° clockwise`, group.ids.length === 0)}
        </div>)}
        {sec("depth", "Back edge (perspective depth)", <div className={row} data-testid={`${ns}-depth`}>
          <span className="text-xs text-muted">the whole group, the front line stays:</span>
          {b("Back edge ↥", () => depth({ px: step }), "depth-up", "Push the back (top) edge of the whole group up the picture by the step: deeper bays", group.ids.length === 0)}
          {b("Back edge ↧", () => depth({ px: -step }), "depth-down", "Pull the back (top) edge down the picture by the step: shallower bays", group.ids.length === 0)}
          {b("Depth − 2 %", () => depth({ k: 0.98 }), "depth-less", "Compress the depth of every bay (and the zone) by 2 % about the front line", group.ids.length === 0)}
          {b("Depth + 2 %", () => depth({ k: 1.02 }), "depth-more", "Expand the depth of every bay (and the zone) by 2 % about the front line", group.ids.length === 0)}
        </div>)}
        {sec("size", "Size", <div className={row}>
          {b("Narrower", () => tune((p) => resizeBay(p, w, h, 0.98, 1)), "fine-narrower", "2 % narrower along its front")}
          {b("Wider", () => tune((p) => resizeBay(p, w, h, 1.02, 1)), "fine-wider", "2 % wider along its front")}
          {b("Shorter", () => tune((p) => resizeBay(p, w, h, 1, 0.98)), "fine-shorter", "2 % shorter towards its back")}
          {b("Longer", () => tune((p) => resizeBay(p, w, h, 1, 1.02)), "fine-longer", "2 % longer towards its back")}
          {b("Shrink 2 %", () => tune((p) => resizeBay(p, w, h, 0.98, 0.98)), "fine-shrink", "2 % smaller all round: keeps a bay off its neighbour's car")}
        </div>)}
        {sec("split", "Split the selected bay", <div className={row}>
          {b("Left | right", () => split("width"), "split-width", "Two bays side by side, left and right (equal on the ground)", !slots.some((s) => s.id === selected))}
          {b("Front / back", () => split("length"), "split-length", "Two bays one behind the other, front and back (a bay laid across two cars)", !slots.some((s) => s.id === selected))}
        </div>)}
        {note && <p className="text-xs text-warning" data-testid={`${ns}-fine-note`}>{note}</p>}
      </div>
    </div>
  );
}
