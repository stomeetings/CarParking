import { useEffect, useRef, useState } from "react";
import { Crosshair, Magnet, Copy, Sparkles, Wand2 } from "lucide-react";
import { Sub } from "./Accordion";
import { Button } from "./ui";

type Pt = [number, number];
export type DotAt = "" | "front" | "mid" | "back"; // "" = the panel's "Dots are at the" choice
export interface LayoutDot { p: Pt; index: string; at: DotAt }
export interface DotNext { index: string; at: DotAt } // what the next dot you click gets: its line number and its place on the line
/** What POST /zones/{z}/layout sends (the server's LayoutIn). */
export interface LayoutRequest {
  mode: "dots" | "master" | "refit";
  count: number;
  depth: number;
  gap: number;
  rows: number;
  dot_at: "front" | "mid" | "back";
  snap: boolean;
  fit_zone?: boolean;
  stall_angle?: number; // degrees between the row and the bay separators (90 = square); empty = from a front / back dot pair on one line
  margin_m: number;
  width?: number;
  heading?: number;
  dots?: { x: number; y: number; index?: number; at?: "front" | "mid" | "back" }[];
  master?: Pt[];
  master_index?: number;
}
/** What POST /zones/{z}/auto-calibrate answers when it worked (the vision model's proposal, checked on the ground plane; nothing is applied until the user lays the bays). */
export interface AutoProposal {
  ok: boolean;
  reason?: string;
  model?: string | null;
  count: number;
  stall_angle: number;
  depth: number;
  step_m?: number;
  dots: { x: number; y: number; index: number; at: "front" | "mid" | "back" }[];
  warnings?: string[];
}
export interface LayoutAnswer { info?: Record<string, unknown>; snap?: { snapped: boolean; support: number; matched: number; du_m: number; dstep_m: number; reason: string } | null; warnings?: string[] }

const num = (s: string, fallback: number) => (s.trim() !== "" && Number.isFinite(Number(s)) ? Number(s) : fallback);

/**
 * Lay a calibrated zone's bays in ground metres from a few marks (carparking/layout.py): reference dots anywhere on the separator lines, one master bay to clone,
 * or a re-fit of the bays as they stand; with an optional setup-time snap to the painted lines and an inward margin in metres. Every action sends the whole
 * request; the margin slider re-sends the last one so the bays shrink / grow as it moves.
 */
export function LayoutTools({ zoneName, calibrated = true, next, onNext, bays, expected, master, dots, placing, busy, answer, onPropose, onPlace, onDotIndex, onDotAt, onDotRemove, onDotsClear, onApply }: {
  zoneName: string;
  next: DotNext;
  onNext: (n: DotNext) => void;
  calibrated?: boolean; // false: dots can still be placed and the layout is asked for; the page calibrates the zone first and then lays the bays from the dots
  bays: number; // the zone's bays now
  expected: string; // the zone's Expected bays ("" = auto)
  master: Pt[] | null; // the selected bay's corners (fractions), the master to clone
  dots: LayoutDot[];
  placing: boolean;
  busy: boolean;
  answer: LayoutAnswer | null;
  onPropose?: () => Promise<AutoProposal | null>; // ask the vision model for dots, angle and count (the page puts the dots in); undefined = not offered
  onPlace: () => void;
  onDotIndex: (i: number, v: string) => void;
  onDotAt: (i: number, v: DotAt) => void;
  onDotRemove: (i: number) => void;
  onDotsClear: () => void;
  onApply: (req: LayoutRequest) => void;
}) {
  const [count, setCount] = useState("");
  const [width, setWidth] = useState("");
  const [depth, setDepth] = useState("5");
  const [gap, setGap] = useState("0");
  const [rows, setRows] = useState("1");
  const [dotAt, setDotAt] = useState<"front" | "mid" | "back">("front");
  const [heading, setHeading] = useState("");
  const [angle, setAngle] = useState("");
  const [masterIndex, setMasterIndex] = useState("0");
  const [snap, setSnap] = useState(false);
  const [fitZone, setFitZone] = useState(false);
  const [margin, setMargin] = useState(0);
  const last = useRef<LayoutRequest | null>(null);
  const [proposing, setProposing] = useState(false);
  const [proposal, setProposal] = useState<AutoProposal | null>(null);
  const propose = async () => {
    if (!onPropose) return;
    setProposing(true);
    setProposal(null);
    try {
      const r = await onPropose();
      if (r) { // fill the form for review: nothing is laid until "Fit the row to the dots"
        setCount(String(r.count));
        setAngle(String(Math.round(r.stall_angle * 10) / 10));
        setDepth(String(Math.round(r.depth * 10) / 10));
        setOpen((o) => ({ ...o, dots: true }));
        setProposal(r);
      }
    } finally {
      setProposing(false);
    }
  };
  const [open, setOpen] = useState({ dots: true, master: false, finish: false }); // the dots are the main way in; the rest is folded until needed
  const flip = (k: keyof typeof open) => setOpen((o) => ({ ...o, [k]: !o[k] }));

  const base = (): Omit<LayoutRequest, "mode"> => ({
    count: Math.min(99, Math.max(1, Math.round(num(count, Number(expected) >= 1 ? Number(expected) : Math.max(1, bays))))),
    depth: num(depth, 5), gap: num(gap, 0), rows: Math.min(6, Math.max(1, Math.round(num(rows, 1)))), dot_at: dotAt, snap, margin_m: margin, ...(fitZone ? { fit_zone: true } : {}),
    ...(width.trim() !== "" ? { width: num(width, 2.5) } : {}), ...(heading.trim() !== "" ? { heading: num(heading, 0) } : {}),
    ...(angle.trim() !== "" ? { stall_angle: Math.min(160, Math.max(20, num(angle, 90))) } : {}),
  });
  const send = (req: LayoutRequest) => { last.current = req; onApply(req); };
  const fromDots = () => send({ ...base(), mode: "dots", dots: dots.map((d) => ({ x: d.p[0], y: d.p[1], ...(d.index.trim() !== "" ? { index: Math.max(0, Math.round(Number(d.index))) } : {}), ...(d.at ? { at: d.at } : {}) })) });
  const fromMaster = () => master && send({ ...base(), mode: "master", master, master_index: Math.max(0, Math.round(num(masterIndex, 0))) });
  const refit = () => send({ ...base(), mode: "refit" });

  // the margin slider re-sends the last request with the new margin (a short pause first), so the bays visibly shrink / grow
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  const moveMargin = (v: number) => {
    setMargin(v);
    clearTimeout(timer.current);
    if (last.current) { const req = { ...last.current, snap, margin_m: v }; timer.current = setTimeout(() => send(req), 350); }
  };

  const field = "ml-1 w-14 rounded-md border border-line px-1.5 py-1 text-sm text-ink";
  return (
    <div className="space-y-3 border-t border-line pt-2" data-testid="layout-tools">
      <p className="text-xs font-semibold tracking-wide text-muted uppercase">Lay {zoneName}'s bays in metres</p>

      <div className="flex flex-wrap items-end gap-2">
        <label className="block text-xs text-muted" title="How many bays: the zone's Expected bays when empty">Bays
          <input className={field} inputMode="numeric" placeholder={Number(expected) >= 1 ? expected : String(bays || 1)} value={count} data-testid="lay-count" onChange={(e) => setCount(e.target.value.replace(/[^0-9]/g, "").slice(0, 2))} />
        </label>
        <label className="block text-xs text-muted" title="Bay width in metres; empty = from the dots">Width (m)
          <input className={field} inputMode="decimal" value={width} data-testid="lay-width" onChange={(e) => setWidth(e.target.value)} />
        </label>
        <label className="block text-xs text-muted">Depth (m)
          <input className={field} inputMode="decimal" value={depth} data-testid="lay-depth" onChange={(e) => setDepth(e.target.value)} />
        </label>
        <label className="block text-xs text-muted" title="Space between bays in a row">Gap (m)
          <input className={field} inputMode="decimal" value={gap} data-testid="lay-gap" onChange={(e) => setGap(e.target.value)} />
        </label>
        <label className="block text-xs text-muted">Rows
          <input className={field} inputMode="numeric" value={rows} data-testid="lay-rows" onChange={(e) => setRows(e.target.value.replace(/[^0-9]/g, "").slice(0, 1))} />
        </label>
        <label className="block text-xs text-muted" title="Angle between the row and the bay lines, 20-160 (90 = square). Empty = taken from two dots on one separator line, one at its front and one at its back, so the bays lean like the cars">Bay angle (°)
          <input className={field} inputMode="decimal" placeholder="auto" value={angle} data-testid="lay-angle" onChange={(e) => setAngle(e.target.value.replace(/[^0-9.]/g, "").slice(0, 5))} />
        </label>
      </div>

      {onPropose && (
        <div className="space-y-1" data-testid="lay-ai-box">
          <Button variant="outline" onClick={() => void propose()} disabled={busy || proposing || !calibrated} data-testid="lay-ai"
                  title={calibrated ? "A vision model reads the painted lines on this zone and proposes the reference dots, the bay angle and the count. You review them, then lay the bays." : "Calibrate the zone first"}>
            <Sparkles size={14} /> {proposing ? "Asking the AI…" : "AI calibrate (propose dots, angle, count)"}
          </Button>
          {proposal && (
            <div className="text-xs" data-testid="lay-ai-result">
              <p className="text-[#1b7a3e]" data-testid="lay-ai-note">Proposed {proposal.count} bays at {Math.round(proposal.stall_angle)}° with {proposal.dots.length} dots{proposal.model ? ` (${proposal.model})` : ""}. Check them on the picture, then press "Fit the row to the dots".</p>
              {(proposal.warnings ?? []).map((t) => <p key={t} className="text-warning">{t}</p>)}
            </div>
          )}
        </div>
      )}

      <Sub id="lay-dots" title="Reference dots" hint={`${dots.length} placed`} open={open.dots} onToggle={() => flip("dots")}>
      <div className="space-y-1.5" data-testid="lay-dots-box">
        <div className="flex flex-wrap items-center gap-2">
          <Button variant={placing ? "primary" : "outline"} onClick={onPlace} disabled={busy} data-testid="lay-dot-mode"
                  title="Click on the painted separator lines, anywhere along the row (start, middle or end)"><Crosshair size={14} /> Place reference dots ({dots.length})</Button>
          <label className="block text-xs text-muted" title="Where on its line each dot sits: the near end, the middle or the far end">Dots are at the
            <select className="ml-1 rounded-md border border-line bg-white px-1.5 py-1 text-sm text-ink" value={dotAt} data-testid="lay-dot-at" onChange={(e) => setDotAt(e.target.value as "front" | "mid" | "back")}>
              <option value="front">front (near end)</option><option value="mid">middle</option><option value="back">back (far end)</option>
            </select>
          </label>
          {dots.length === 1 && (
            <label className="block text-xs text-muted" title="One dot needs the row's direction on the ground, in degrees (0 = left to right), and the bay width above">Heading (°)
              <input className={field} inputMode="decimal" value={heading} data-testid="lay-heading" onChange={(e) => setHeading(e.target.value)} />
            </label>
          )}
          <Button variant="ghost" onClick={onDotsClear} disabled={!dots.length} data-testid="lay-dots-clear">Clear dots</Button>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted" data-testid="lay-next-box">
          <span title="Dots on several slots (the first, the middle, the last) pin the whole row: choose the line, then click its dots">Next dot goes on line</span>
          <input className={field} inputMode="numeric" placeholder="auto" value={next.index} data-testid="lay-next-line" onChange={(e) => onNext({ ...next, index: e.target.value.replace(/[^0-9]/g, "").slice(0, 2) })} />
          <select className="rounded-md border border-line bg-white px-1.5 py-1 text-sm text-ink" value={next.at} data-testid="lay-next-place" onChange={(e) => onNext({ ...next, at: e.target.value as DotAt })}>
            <option value="">at the place above</option><option value="front">front</option><option value="mid">middle</option><option value="back">back</option>
          </select>
          <span>(line 0 = left of bay 1, line 1 = between bays 1 and 2, line N = right of bay N)</span>
        </div>
        {dots.length > 0 && (
          <ul className="flex flex-wrap gap-1.5" data-testid="lay-dot-list">
            {dots.map((d, i) => (
              <li key={i} className="flex items-center gap-1 rounded-md border border-line px-1.5 py-0.5 text-xs" data-testid="lay-dot">
                <span className="font-semibold">{i + 1}</span>
                <label className="text-muted" title="Which separator line it is on: 0 = the row's first line, 1 = between bays 1 and 2 ... (empty = worked out from the spacing)">line
                  <input className="ml-1 w-9 rounded border border-line px-1 py-0.5 text-ink" inputMode="numeric" placeholder="auto" value={d.index} data-testid="lay-dot-index"
                         onChange={(e) => onDotIndex(i, e.target.value.replace(/[^0-9]/g, "").slice(0, 2))} />
                </label>
                <select className="rounded border border-line bg-white px-0.5 py-0.5 text-ink" value={d.at} data-testid="lay-dot-place" aria-label={`Dot ${i + 1} place on its line`}
                        onChange={(e) => onDotAt(i, e.target.value as DotAt)}>
                  <option value="">as above</option><option value="front">front</option><option value="mid">middle</option><option value="back">back</option>
                </select>
                <button type="button" className="text-muted hover:text-danger" onClick={() => onDotRemove(i)} aria-label={`Remove dot ${i + 1}`} data-testid="lay-dot-remove">✕</button>
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-muted" data-testid="lay-angle-hint">Cars parked at an angle? Put two dots on ONE painted line, one at its front (near end) and one at its back (far end), one after the other, and set the second one's place to "back": the bays then lean at the cars' angle, in perspective. Or type the angle above.</p>
        <p className="text-xs text-muted">Dot 1 (the first dot along the row, or line 0 when you type line numbers) is where the first bay starts: nothing is drawn before it. The dots give the direction and spacing; the bays then run to the far end of the zone outline.</p>
        <label className="flex items-center gap-1.5 text-xs text-muted" title="Make every bay exactly the distance from Dot 1 to the zone's far end divided by the number of bays">
          <input type="checkbox" checked={fitZone} data-testid="lay-fit-zone" onChange={(e) => setFitZone(e.target.checked)} /> Stretch the bays from Dot 1 to the zone's far end
        </label>
        <Button onClick={fromDots} disabled={busy || dots.length < 1} data-testid="lay-fit-dots"
                title={dots.length === 1 ? "One dot needs the width and the heading" : "Fit the row to the dots and lay the bays"}><Wand2 size={14} /> {calibrated ? "Fit the row to the dots" : "Calibrate, then fit the row to the dots"}</Button>
        {!calibrated && <p className="text-xs text-muted" data-testid="lay-precal">Dots can be placed now. This zone is not calibrated yet: the button calibrates it first and then lays the bays from your dots.</p>}
      </div>
      </Sub>

      <Sub id="lay-master" title="Clone a master bay / re-fit" hint={master ? "a bay is selected" : "select a bay"} open={open.master} onToggle={() => flip("master")}>
      <div className="flex flex-wrap items-center gap-2" data-testid="lay-master-box">
        <label className="block text-xs text-muted" title="Where the master stands in the row: 0 = the first bay; the clones then run both ways">Master is bay no.
          <input className={field} inputMode="numeric" value={masterIndex} data-testid="lay-master-index" onChange={(e) => setMasterIndex(e.target.value.replace(/[^0-9]/g, "").slice(0, 2))} />
        </label>
        <Button variant="outline" onClick={fromMaster} disabled={busy || !master || !calibrated} data-testid="lay-clone"
                title={master ? "Measure the selected bay on the ground and repeat it along the zone" : "Select a bay first (Edit bays, click one)"}><Copy size={14} /> Clone the selected bay</Button>
        <Button variant="ghost" onClick={refit} disabled={busy || bays < 1 || !calibrated} data-testid="lay-refit"
                title="Re-express the bays as they stand now as a lattice in metres (after a recalibration)">Re-fit as metres</Button>
      </div>

      </Sub>

      <Sub id="lay-finish" title="Snap and inward margin" hint={`${snap ? "snap on" : "snap off"} · ${margin.toFixed(2)} m`} open={open.finish} onToggle={() => flip("finish")}>
      <div className="space-y-1" data-testid="lay-finish">
        <label className="flex items-center gap-1.5 text-sm" title="Pull the separators onto the painted lines the camera sees (setup only, never at run time)">
          <input type="checkbox" checked={snap} data-testid="lay-snap" onChange={(e) => setSnap(e.target.checked)} />
          <Magnet size={14} /> Snap to the painted lines <span className="text-xs text-muted">(applied with the next layout)</span>
        </label>
        <label className="block text-xs text-muted" title="Every bay pulled in by this many metres on the ground, so neighbours never touch">Inward margin: <b data-testid="lay-margin-value">{margin.toFixed(2)} m</b>
          <input type="range" className="ml-2 w-40 align-middle" min={0} max={0.5} step={0.05} value={margin} data-testid="lay-margin" onChange={(e) => moveMargin(Number(e.target.value))} />
          <span className="ml-2">{margin > 0 ? `each bay ${(margin * 200).toFixed(0)} cm narrower and shorter in all` : "off"}</span>
        </label>
      </div>
      </Sub>

      {answer && (
        <div className="space-y-0.5 text-xs" data-testid="lay-answer">
          {answer.info && Object.keys(answer.info).length > 0 && (
            <p data-testid="lay-info">{Object.entries(answer.info).filter(([, v]) => typeof v === "number" || typeof v === "string").map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ")}</p>
          )}
          {answer.snap && <p className={answer.snap.snapped ? "text-[#1b7a3e]" : "text-warning"} data-testid="lay-snap-result">
            {answer.snap.snapped ? `Snapped: ${Math.round(answer.snap.support * 100)} % of the lines found, moved ${(answer.snap.du_m * 100).toFixed(0)} cm, spacing ${answer.snap.dstep_m >= 0 ? "+" : ""}${(answer.snap.dstep_m * 100).toFixed(0)} cm`
              : `Not snapped: ${answer.snap.reason}`}</p>}
          {(answer.warnings ?? []).map((t) => <p key={t} className="text-warning">{t}</p>)}
        </div>
      )}
    </div>
  );
}
