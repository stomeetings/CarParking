import { Check, FlaskConical, Loader2, Save, Sparkles, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ApiError, api, withCam } from "../api";
import { points } from "../lib";
import { type Pt, moveSlot, moveVertex } from "../slotdraw";
import {
  type Bay, type BayStatus, type Job, STATUS_COLOR, type SuggestListItem, type SuggestView,
  acceptBlock, acceptText, diffText, gridBody, pinText, protectedText, sameGrid, statusOf,
} from "../suggest";
import type { Snapshot } from "../types";
import { Button, Card, Pill, inputClass } from "./ui";

type Layer = "diff" | "both" | "draft" | "baseline";
type Drag = { kind: "vertex"; id: number; index: number } | { kind: "move"; id: number; last: Pt };
const LAYER_KEY = "carparking.suggestLayer";
const LAYERS: [Layer, string, string][] = [
  ["diff", "Compare", "The draft coloured by what accepting would change: new, moved, edited, and the running bays that would go (red, dashed)"],
  ["both", "Both", "The running slots (solid white) and the draft (dotted) together"],
  ["draft", "Draft", "Only the AI draft"],
  ["baseline", "Running", "Only the slots that run now"],
];
const PIN = "#8e44ad";

const readLayer = (): Layer => {
  try {
    const v = localStorage.getItem(LAYER_KEY);
    return LAYERS.some(([l]) => l === v) ? (v as Layer) : "diff";
  } catch {
    return "diff";
  }
};

/** Suggest slots: an AI bay grid (SAM + GeoCalib calibration) as a DRAFT layer over the editor's snapshot, next to the slots that
 * run now. Compare, reshape a bay (drag it or its corners), test the draft on the snapshot, then Accept or Discard. Nothing the
 * worker reads changes until Accept: monitoring carries on with the running slots meanwhile. */
export function SuggestSlots({ camId, cam, snapshot, image, stillId, tokenKey, onNeedAdmin }: {
  camId: string | null; // the registry camera's id (the endpoints are per camera id)
  cam: string | null;
  snapshot: Snapshot | undefined;
  image: string | null; // the editor's frozen still
  stillId: string | null;
  tokenKey: string;
  onNeedAdmin: (reason: string, retry?: () => void) => void;
}) {
  const [gate, setGate] = useState<"ok" | "off" | "auth">("ok"); // off: the geometry feature is switched off on the server
  const [list, setList] = useState<SuggestListItem[]>([]);
  const [mode, setMode] = useState<"quick" | "full">("quick");
  const [expected, setExpected] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [view, setView] = useState<SuggestView | null>(null);
  const [work, setWork] = useState<Bay[]>([]); // the draft being fine-tuned
  const [sel, setSel] = useState<number | null>(null);
  const [take, setTake] = useState<Set<number>>(new Set()); // Fixed camera: the suggested bays to take
  const [tested, setTested] = useState<Record<number, string> | null>(null);
  const [layer, setLayer] = useState<Layer>(readLayer);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const [w, h] = view ? [view.width, view.height] : (snapshot?.shape ?? [2560, 1440]);
  const base = camId ? `/api/cameras/${encodeURIComponent(camId)}/suggest` : null;

  const fail = (e: unknown, what: string, retry?: () => void) => {
    const err = e as ApiError;
    if (err.status === 401 || err.status === 403) onNeedAdmin(`${what} needs the admin token.`, retry);
    else setMessage({ tone: "error", text: `${what} failed: ${err.message}` });
  };

  const loadList = useCallback(() => {
    if (!base) return;
    api.get<{ suggestions: SuggestListItem[] }>(base).then((r) => (setGate("ok"), setList(Array.isArray(r?.suggestions) ? r.suggestions : [])))
      .catch((e: ApiError) => setGate(e.status === 401 || e.status === 403 ? "auth" : "off"));
  }, [base]);
  useEffect(() => {
    setView(null);
    setJob(null);
    setMessage(null);
    loadList();
  }, [loadList, tokenKey]);

  const open = useCallback(async (version: number) => {
    if (!base) return;
    try {
      const v = await api.get<SuggestView>(`${base}/${version}`);
      setView(v);
      setWork(v.grid);
      setTake(new Set(v.grid.filter((b) => !b.protected).map((b) => b.id))); // (a protected running slot needs no merge)
      setSel(null);
      setTested(null);
      setConfirming(false);
    } catch (e) {
      fail(e, "Opening the suggestion");
    }
  }, [base]); // eslint-disable-line react-hooks/exhaustive-deps

  // the job as it stands: a finished one opens its draft
  const follow = (j: Job) => {
    setJob(j);
    if (j.state === "done" && j.result) (void open(j.result.version), loadList());
  };
  // the running job: polled until it ends
  useEffect(() => {
    if (!job || (job.state !== "queued" && job.state !== "running")) return;
    const t = setTimeout(() => {
      api.get<Job>(`/api/setup/jobs/${job.id}`).then(follow)
        .catch((e) => (setJob(null), fail(e, "Following the suggestion job")));
    }, 1000);
    return () => clearTimeout(t);
  }, [job]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    if (!base) return;
    const n = expected.trim() === "" ? null : Number(expected);
    if (n != null && (!Number.isInteger(n) || n < 1 || n > 64)) return setMessage({ tone: "error", text: "Expected bays: a whole number from 1 to 64." });
    setMessage(null);
    setView(null);
    try {
      follow(await api.post<Job>(base, { mode, ...(stillId ? { still: stillId } : {}), ...(n != null ? { expected: n } : {}) }));
    } catch (e) {
      fail(e, "Suggesting slots", start);
    }
  };

  const dirty = view != null && !sameGrid(work, view.grid);
  const status = (b: Bay): BayStatus => statusOf(b, view?.diff ?? null);
  const removed = useMemo(() => {
    const gone = new Set(view?.diff?.removed ?? []);
    return (view?.baseline.slots ?? []).filter((s) => gone.has(s.id));
  }, [view]);

  const save = async (): Promise<boolean> => {
    if (!view || !base) return false;
    try {
      await api.put(`${base}/${view.version}`, gridBody(work));
      await open(view.version);
      return true;
    } catch (e) {
      fail(e, "Saving the draft", save);
      return false;
    }
  };
  const accept = async () => {
    if (!view || !base) return;
    setBusy(true);
    try {
      if (dirty && !(await save())) return;
      const r = await api.post<{ applied: string }>(`${base}/${view.version}/accept`, view.meta.kind === "fixed" ? { select: [...take].sort((a, b) => a - b) } : {});
      setMessage({ tone: "ok", text: r.applied === "fixed" ? "Merged into the hand-drawn map; the worker applies it from its next frame." : "Accepted: the worker switches to the new grid within a few seconds." });
      setView(null);
      setJob(null);
      loadList();
    } catch (e) {
      fail(e, "Accepting", accept);
      void open(view.version); // (the draft may be stale now)
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  };
  const discard = async () => {
    if (!view || !base) return;
    try {
      await api.delete(`${base}/${view.version}`);
      setView(null);
      setJob(null);
      setMessage({ tone: "ok", text: "Draft discarded. The running slots were not touched." });
      loadList();
    } catch (e) {
      fail(e, "Discarding", discard);
    }
  };
  const test = async () => {
    try {
      const r = await api.query<{ results: { id: number; state: string }[] }>(withCam("/api/fixed-slots/preview", cam),
        { slots: work.map(({ id, name, poly }) => ({ id, name, poly })), ...(stillId ? { snapshot: stillId } : {}) });
      setTested(Object.fromEntries(r.results.map((x) => [x.id, x.state])));
    } catch (e) {
      fail(e, "Testing the draft");
    }
  };

  const at = (e: { clientX: number; clientY: number }): Pt => {
    const r = box.current!.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };
  const onMove = (e: React.MouseEvent) => {
    if (!drag) return;
    const p = at(e);
    if (drag.kind === "vertex") setWork((all) => moveVertex(all, drag.id, drag.index, p) as Bay[]);
    else {
      setWork((all) => moveSlot(all, drag.id, p[0] - drag.last[0], p[1] - drag.last[1]) as Bay[]);
      setDrag({ ...drag, last: p });
    }
  };
  const chooseLayer = (l: Layer) => {
    setLayer(l);
    try {
      localStorage.setItem(LAYER_KEY, l);
    } catch {
      /* private window: the choice lasts for this page only */
    }
  };

  if (gate === "off" || !base) return null; // geometry switched off, or no camera yet: the card does not exist
  if (gate === "auth") {
    return (
      <Card title={<span className="flex items-center gap-2"><Sparkles size={16} className="text-primary" /> Suggest slots</span>}>
        <p className="text-sm text-muted">Suggestions are an admin tool.</p>
        <div className="mt-2"><Button variant="outline" onClick={() => onNeedAdmin("Suggest slots needs the admin token.", loadList)}>Enter admin token</Button></div>
      </Card>
    );
  }

  const running = job?.state === "queued" || job?.state === "running";
  const showBase = layer === "baseline" || layer === "both";
  const showDraft = layer !== "baseline";
  const block = view ? acceptBlock(view, take.size) : null;
  const sel1 = work.find((b) => b.id === sel && !b.protected) ?? null; // (a protected bay has no corners to drag)
  const label = (b: Bay) => {
    const xs = b.poly.map((p) => p[0]), ys = b.poly.map((p) => p[1]);
    return { x: Math.min(...xs) * w + w / 250, y: Math.min(...ys) * h + w / 70 };
  };

  return (
    <Card title={<span className="flex items-center gap-2"><Sparkles size={16} className="text-primary" /> Suggest slots · AI draft</span>}
          actions={LAYERS.map(([l, text, title]) => <Pill key={l} active={layer === l} onClick={() => chooseLayer(l)} title={title}>{text}</Pill>)}
          bodyClass="p-0">
      <div ref={box} className="relative w-full select-none bg-ink" style={{ aspectRatio: `${w} / ${h}` }} data-testid="suggest-surface">
        {image ? <img src={image} alt="Camera snapshot" className="absolute inset-0 h-full w-full object-fill" draggable={false} />
          : <div className="absolute inset-0 flex items-center justify-center text-sm text-white/70">Capturing a snapshot from the camera…</div>}
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full"
             onMouseMove={onMove} onMouseUp={() => setDrag(null)} onMouseLeave={() => setDrag(null)} data-testid="suggest-svg">
          {view && (showBase || layer === "diff") && (layer === "diff" ? removed : view.baseline.slots.filter((s) => s.source !== "manual")).map((s) => (
            <polygon key={`b${s.id}`} points={points(s.poly, w, h)} fill={layer === "diff" ? "rgba(214,69,69,0.10)" : "rgba(255,255,255,0.06)"}
                     stroke={layer === "diff" ? STATUS_COLOR.removed : "#ffffff"} strokeWidth={2} vectorEffect="non-scaling-stroke"
                     strokeDasharray={layer === "diff" ? "8 5" : undefined} pointerEvents="none"
                     data-testid={layer === "diff" ? "removed-bay" : "baseline-bay"} />
          ))}
          {showDraft && work.map((b) => {
            const st = status(b);
            const colour = layer === "draft" || layer === "both" ? STATUS_COLOR.unchanged : STATUS_COLOR[st];
            const t = tested?.[b.id];
            return (
              <g key={b.id} data-testid="draft-bay" data-status={st} data-id={b.id} data-protected={b.protected ? "true" : undefined}>
                <polygon points={points(b.poly, w, h)} fill={t ? (t === "BUSY" ? "rgba(214,69,69,0.30)" : "rgba(31,157,85,0.30)") : `${colour}26`}
                         stroke={colour} strokeWidth={b.id === sel ? 3.5 : 2.5} vectorEffect="non-scaling-stroke"
                         strokeDasharray={b.protected ? undefined : "3 4"} style={{ cursor: b.protected ? "default" : "move" }}
                         onMouseDown={(e) => { if (e.button === 0) (e.stopPropagation(), setSel(b.id), !b.protected && setDrag({ kind: "move", id: b.id, last: at(e) })); }} />
                <text x={label(b).x} y={label(b).y} fontSize={w / 70} fontWeight={700} fill="white" stroke="black" strokeWidth={w / 900}
                      paintOrder="stroke" pointerEvents="none">{b.name}{t ? ` ${t}` : ""}</text>
              </g>
            );
          })}
          {view && showDraft && view.pins.map((p) => (
            <g key={`p${p.id}`} pointerEvents="none" data-testid="pin-bay">
              <polygon points={points(p.poly, w, h)} fill={`${PIN}40`} stroke={PIN} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
              <text x={label(p).x} y={label(p).y} fontSize={w / 70} fontWeight={700} fill="white" stroke="black" strokeWidth={w / 900} paintOrder="stroke">{p.name}</text>
            </g>
          ))}
          {showDraft && sel1 && sel1.poly.map(([x, y], i) => (
            <circle key={i} cx={x * w} cy={y * h} r={w / 220} fill="white" stroke={STATUS_COLOR.edited} strokeWidth={2} vectorEffect="non-scaling-stroke"
                    style={{ cursor: "grab" }} data-testid="vertex"
                    onMouseDown={(e) => { if (e.button === 0) (e.stopPropagation(), setDrag({ kind: "vertex", id: sel1.id, index: i })); }} />
          ))}
        </svg>
      </div>

      <div className="space-y-3 p-4">
        {!view && (
          <>
            <p className="text-xs text-muted" data-testid="suggest-hint">
              Asks the AI for a bay grid on the snapshot above. Quick (a few seconds) re-lays the active calibration around today's vehicles;
              Full (a minute or two) measures the lens and tilt again with SAM + GeoCalib and keeps the calibration's metres. The result is a
              draft: nothing runs differently until you accept it.
            </p>
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex gap-1.5">
                <Pill active={mode === "quick"} onClick={() => setMode("quick")} title="The active calibration re-tiled on a fresh still">Quick</Pill>
                <Pill active={mode === "full"} onClick={() => setMode("full")} title="SAM + GeoCalib again in the setup service (needs it running)">Full (SAM + GeoCalib)</Pill>
              </div>
              <label className="block">
                <span className="mb-1 block text-xs font-semibold tracking-wide text-muted uppercase">Expected bays</span>
                <input className={`${inputClass} w-28`} inputMode="numeric" placeholder="saved count" value={expected}
                       onChange={(e) => setExpected(e.target.value)} data-testid="suggest-expected" />
              </label>
              <Button onClick={start} disabled={running}>
                {running ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />} Auto-detect / suggest slots
              </Button>
            </div>
            {job && (
              <p className={`rounded-md px-3 py-2 text-sm ${job.state === "error" ? "bg-danger/10 text-danger" : "bg-primary/10 text-ink"}`} data-testid="suggest-job">
                {job.state === "error" ? `Suggestion failed: ${job.error}` : `${job.stage} · ${Math.round(job.progress * 100)} %`}
              </p>
            )}
            {job?.state === "done" && job.result?.warnings.map((t) => <p key={t} className="text-xs text-warning">{t}</p>)}
            {list.length > 0 && (
              <ul className="divide-y divide-line rounded-md border border-line" data-testid="suggest-list">
                {list.map((s) => (
                  <li key={s.version} className="flex items-center gap-2 px-3 py-1.5 text-sm">
                    <span className="font-medium">Draft v{s.version}</span>
                    <span className="text-muted">{s.meta.mode} · {s.bays} bays · {new Date(s.created_at * 1000).toLocaleString()}</span>
                    <span className="flex-1" />
                    <Button variant="outline" onClick={() => void open(s.version)}>Open</Button>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}

        {view && (
          <>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="font-semibold">Draft v{view.version}</span>
              <span className="rounded bg-primary/10 px-1.5 py-0.5 text-xs text-primary">{view.meta.mode}</span>
              <span className="text-muted" data-testid="suggest-diff">{diffText(view.diff)}</span>
            </div>
            <p className="rounded-md bg-primary/10 px-3 py-2 text-xs text-ink" data-testid="suggest-banner">
              Draft only: live monitoring is unchanged until you accept. Drag a bay or its corners to fine-tune it.
              {view.calibration.confidence ? ` Calibration v${view.calibration.version} (${view.calibration.confidence}, camera ${view.calibration.h?.toFixed(1)} m).` : ""}
              {view.meta.lined_up === "vehicles" ? ` Rows lined up with ${view.meta.vehicles ?? 0} vehicle(s) on the snapshot.` : ""}
            </p>
            {view.stale && (
              <p className="rounded-md bg-warning/15 px-3 py-2 text-xs text-ink" role="alert" data-testid="suggest-stale">
                The slots that run now changed since this suggestion was made: it can no longer be accepted. Discard it and ask again.
              </p>
            )}
            {protectedText(work) && (
              <p className="rounded-md border border-dashed border-primary/50 px-3 py-2 text-xs text-ink" data-testid="suggest-protected">{protectedText(work)}</p>
            )}
            {view.pins.length > 0 && <p className="text-xs text-muted" data-testid="suggest-pins">Pins kept as anchors: {view.pins.map(pinText).join(" · ")}</p>}
            <ul className="max-h-56 divide-y divide-line overflow-y-auto rounded-md border border-line" data-testid="suggest-bays">
              {work.map((b) => (
                <li key={b.id} className={`flex items-center gap-2 px-2.5 py-1.5 text-sm ${b.id === sel ? "bg-primary/5" : ""}`}>
                  {view.meta.kind === "fixed" && (
                    <input type="checkbox" aria-label={`take ${b.name}`} checked={take.has(b.id)}
                           onChange={() => setTake((t) => { const n = new Set(t); if (n.has(b.id)) n.delete(b.id); else n.add(b.id); return n; })} />
                  )}
                  <button type="button" className="flex flex-1 items-center gap-2 text-left" onClick={() => setSel(b.id)}>
                    <span className="w-8 text-xs font-semibold text-muted">{b.id}</span>
                    <span className="flex-1">{b.name}</span>
                    {b.capacity && b.capacity > 1 && <span className="text-xs text-muted">holds {b.capacity}</span>}
                    {b.protected
                      ? <span className="rounded bg-ink/70 px-1.5 py-0.5 text-xs text-white" title="A running slot: kept exactly as it runs" data-testid="protected-badge">running</span>
                      : <span className="rounded px-1.5 py-0.5 text-xs text-white" style={{ background: STATUS_COLOR[status(b)] }}>{status(b)}</span>}
                  </button>
                  {!b.protected && (
                    <button type="button" title="Remove this bay from the draft" className="rounded p-1 text-muted hover:bg-danger/10 hover:text-danger"
                            onClick={() => setWork((all) => all.filter((x) => x.id !== b.id))}><Trash2 size={14} /></button>
                  )}
                </li>
              ))}
            </ul>
            {confirming && (
              <div className="space-y-2 rounded-md border border-primary/40 bg-primary/5 p-3 text-sm" data-testid="suggest-confirm">
                <p>{acceptText(view, take.size)}</p>
                <div className="flex gap-2">
                  <Button onClick={accept} disabled={busy}><Check size={14} /> Confirm accept</Button>
                  <Button variant="ghost" onClick={() => setConfirming(false)}>Cancel</Button>
                </div>
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => setConfirming(true)} disabled={busy || block != null || work.length === 0} title={block ?? undefined}><Check size={14} /> Accept</Button>
              <Button variant="outline" onClick={save} disabled={!dirty}><Save size={14} /> Save draft</Button>
              <Button variant="outline" onClick={() => setWork(view.grid)} disabled={!dirty}>Revert edits</Button>
              <Button variant="outline" onClick={test} title="Preview FREE / BUSY of the draft's bays on the snapshot (nothing is saved)"><FlaskConical size={14} /> Test draft</Button>
              {tested && <Button variant="ghost" onClick={() => setTested(null)}>Clear test</Button>}
              <Button variant="danger" onClick={discard}><X size={14} /> Discard</Button>
            </div>
            {block && <p className="text-xs text-muted" data-testid="suggest-block">{block}</p>}
          </>
        )}
        {message && <p className={`rounded-md px-3 py-2 text-sm ${message.tone === "ok" ? "bg-success/10 text-[#1b7a3e]" : "bg-danger/10 text-danger"}`} data-testid="suggest-message">{message.text}</p>}
      </div>
    </Card>
  );
}
