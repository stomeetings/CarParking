import { Check, ImageIcon, Lock, RefreshCw, Ruler, Trash2, Undo2 } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api";
import {
  CONFIDENCE_TONE, EMPTY_REFERENCE, type CalibrateResult, type GeometryEvent, type GeometrySummary, type GeometryVersion, type HistoryResponse,
  type Pt, type Reference, type SegmentResult, type SetupJob, type SetupStatus, calibrateGate, clickPoint, confidenceNote, needsUnverifiedOk,
  referenceBody, referenceProblem, staleBanner, waitForJob,
} from "../setupflow";
import type { CameraInfo } from "../types";
import { EventsCard, HistoryCard, StaleBanner } from "./MetricHistory";
import { ReanchorCard } from "./Reanchor";
import type { Still } from "./SlotEditor";
import { Button, Card, Field, Pill, inputClass } from "./ui";

type Step = 1 | 2 | 3 | 4 | 5;
type LensModel = "simple_radial" | "radial" | "pinhole";
interface Click { x: number; y: number; label: 0 | 1 }
interface Versions { active: GeometrySummary | null; versions: GeometrySummary[] }

const STEPS: { n: Step; label: string }[] = [
  { n: 1, label: "Still" }, { n: 2, label: "Known distance" }, { n: 3, label: "Parking surface" }, { n: 4, label: "Calibrate" }, { n: 5, label: "Review & accept" },
];
const LENS: Record<LensModel, string> = {
  simple_radial: "Wide CCTV lens (one distortion term) - recommended",
  radial: "Strong distortion (two terms)",
  pinhole: "No lens distortion",
};
const STATE_TONE: Record<GeometrySummary["state"], string> = {
  draft: "bg-warning/15 text-ink", active: "bg-success/10 text-success", superseded: "bg-body text-muted", stale: "bg-danger/10 text-danger",
};

const pts = (poly: Pt[], w: number, h: number) => poly.map(([x, y]) => `${x * w},${y * h}`).join(" ");
const when = (t: number | null) => (t ? new Date(t * 1000).toLocaleString() : "");

/** Camera setup (Phase 2): a still, a distance the Admin knows (asked for up front: it is what makes the metres accurate),
 * the parking surface (one click for SAM), the calibration, and Accept. A draft changes nothing for the live system; Accept
 * makes it the camera's active calibration, which the rest of the system reads from the store on its next check: nothing is
 * restarted. Every endpoint needs the admin token and CARPARK_GEOMETRY_ENABLED on the server (api/geometry_routes.py). */
export function CameraSetup({ cam, cameras, onCam, tokenKey, onNeedAdmin }: {
  cam: string | null;
  cameras: CameraInfo[];
  onCam: (id: string) => void;
  tokenKey: string;
  onNeedAdmin: (reason: string, retry?: () => void) => void;
}) {
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [step, setStep] = useState<Step>(1);
  const [still, setStill] = useState<Still | null>(null);
  const [reference, setReference] = useState<Reference>(EMPTY_REFERENCE);
  const [without, setWithout] = useState(false);
  const [clicks, setClicks] = useState<Click[]>([]);
  const [label, setLabel] = useState<0 | 1>(1);
  const [seg, setSeg] = useState<SegmentResult | null>(null);
  const [chosen, setChosen] = useState(0);
  const [lens, setLens] = useState<LensModel>("simple_radial");
  const [job, setJob] = useState<SetupJob | null>(null);
  const [busy, setBusy] = useState<"still" | "segment" | "calibrate" | "accept" | null>(null);
  const [result, setResult] = useState<CalibrateResult | null>(null);
  const [draft, setDraft] = useState<GeometryVersion | null>(null);
  const [unverifiedOk, setUnverifiedOk] = useState(false);
  const [versions, setVersions] = useState<Versions | null>(null);
  const [events, setEvents] = useState<GeometryEvent[] | null>(null); // the calibration's story (drift guard included)
  const [history, setHistory] = useState<HistoryResponse | null>(null); // measured occupancy over time
  const [hours, setHours] = useState(24);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [anchor, setAnchor] = useState<{ from: number | null } | null>(null); // the re-anchoring card is open (from this version, or the server's default)
  const stop = useRef({ cancelled: false });

  const size: [number, number] = still ? [still.width, still.height] : [1920, 1080];
  const [w, h] = size;
  const enabled = status?.enabled === true;
  const candidate = seg ? (seg.candidates[chosen] ?? seg) : null;

  const fail = useCallback((e: unknown, retry?: () => void) => {
    if (e instanceof ApiError && e.status === 401) onNeedAdmin("Enter the admin token to run the camera setup.", retry);
    else setMessage({ tone: "error", text: (e as Error).message });
  }, [onNeedAdmin]);

  const loadVersions = useCallback(async () => {
    if (!cam) return;
    try {
      setVersions(await api.get<Versions>(`/api/cameras/${encodeURIComponent(cam)}/setup/geometry`));
    } catch (e) {
      if (!(e instanceof ApiError && (e.status === 503 || e.status === 401))) setMessage({ tone: "error", text: (e as Error).message });
    }
  }, [cam]);

  // history and events are secondary: a failure here is not worth an error banner (the setup itself still works)
  const loadEvents = useCallback(async () => {
    if (!cam) return;
    try {
      setEvents((await api.get<{ events: GeometryEvent[] }>(`/api/cameras/${encodeURIComponent(cam)}/geometry/events?limit=30`)).events);
    } catch {
      /* not shown */
    }
  }, [cam]);
  const loadHistory = useCallback(async () => {
    if (!cam) return;
    try {
      setHistory(await api.get<HistoryResponse>(`/api/cameras/${encodeURIComponent(cam)}/geometry/history?hours=${hours}&points=96`));
    } catch {
      /* not shown */
    }
  }, [cam, hours]);
  const refreshAll = useCallback(() => {
    void loadVersions();
    void loadEvents();
    void loadHistory();
  }, [loadVersions, loadEvents, loadHistory]);

  const loadStatus = useCallback(() => {
    api.get<SetupStatus>("/api/setup/status").then((s) => {
      setStatus(s);
      if (s.enabled) refreshAll();
    }).catch((e) => fail(e, loadStatus));
  }, [fail, refreshAll]);
  useEffect(() => loadStatus(), [loadStatus, tokenKey]);

  // another camera: a fresh setup (the still, the clicks and the draft belong to one camera)
  useEffect(() => {
    stop.current.cancelled = true;
    stop.current = { cancelled: false };
    setStill(null); setStep(1); setReference(EMPTY_REFERENCE); setWithout(false); setClicks([]); setSeg(null); setChosen(0);
    setJob(null); setBusy(null); setResult(null); setDraft(null); setUnverifiedOk(false); setMessage(null); setVersions(null);
    setEvents(null); setHistory(null); setAnchor(null);
  }, [cam]);
  useEffect(() => () => { stop.current.cancelled = true; }, []);

  const base = `/api/cameras/${encodeURIComponent(cam ?? "")}/setup`;

  const takeStill = async () => {
    setBusy("still");
    setMessage(null);
    try {
      const s = await api.post<Still>(`${base}/still`, {});
      setStill(s);
      setClicks([]); setSeg(null); setResult(null); setDraft(null); setReference(EMPTY_REFERENCE);
      setStep(2);
    } catch (e) {
      fail(e, () => void takeStill());
    } finally {
      setBusy(null);
    }
  };

  const run = async <T,>(kind: "segment" | "calibrate", path: string, body: unknown): Promise<SetupJob<T> | null> => {
    setBusy(kind);
    setMessage(null);
    setJob(null);
    try {
      const started = await api.post<SetupJob<T>>(`${base}/${path}`, body);
      const done = await waitForJob<T>(started.id, (j) => setJob(j as SetupJob), stop.current);
      if (done.state === "error") {
        setMessage({ tone: "error", text: done.error ?? "the job failed" });
        return null;
      }
      return done;
    } catch (e) {
      if ((e as Error).message !== "cancelled") fail(e);
      return null;
    } finally {
      setBusy(null);
    }
  };

  const findSurface = async () => {
    if (!still) return;
    const done = await run<SegmentResult>("segment", "segment", { still: still.id, clicks });
    if (done?.result) {
      const r = done.result;
      setSeg(r);
      const best = r.candidates.findIndex((c) => c.score === r.score && c.area_frac === r.area_frac); // the server's pick
      setChosen(best >= 0 ? best : 0);
    }
  };

  const calibrate = async () => {
    if (!still || !candidate) return;
    const ref = without ? null : referenceBody(reference, size);
    const done = await run<CalibrateResult>("calibrate", "calibrate", {
      still: still.id, polygon: candidate.polygon, camera_model: lens, ...(ref ? { reference: ref } : {}),
      meta: { clicks, score: candidate.score, model: seg?.model },
    });
    if (done?.result) {
      setResult(done.result);
      setUnverifiedOk(false);
      try {
        setDraft(await api.get<GeometryVersion>(`${base}/geometry/${done.result.version}`));
      } catch (e) {
        fail(e);
      }
      void loadVersions();
      void loadEvents();
      setStep(5);
    }
  };

  const activate = async (version: number, unverified: boolean) => {
    setBusy("accept");
    setMessage(null);
    try {
      const r = await api.post<{ active: GeometrySummary; previous: number | null }>(`${base}/activate`, { version, accept_unverified: unverified });
      setMessage({ tone: "ok", text: `Version ${r.active.ver} is now this camera's active calibration${r.previous ? ` (replacing ${r.previous}, kept for rollback)` : ""}. The live system reads it on its next check: nothing was restarted.` });
      await loadVersions();
      void loadEvents();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(null);
    }
  };

  const discard = async (version: number) => {
    setBusy("accept");
    try {
      await api.delete(`${base}/draft?version=${version}`);
      if (result?.version === version) { setResult(null); setDraft(null); setStep(4); }
      setMessage({ tone: "ok", text: `Draft ${version} discarded.` });
      await loadVersions();
      void loadEvents();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(null);
    }
  };

  // clicks on the still: the end points of the known distance (step 2) or ground / not-ground points (step 3)
  const onSurfaceClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!still || (step !== 2 && step !== 3) || busy) return;
    const p = clickPoint(e.currentTarget.getBoundingClientRect(), e.clientX, e.clientY);
    if (!p) return;
    if (step === 2) {
      setReference((r) => (!r.a || (r.a && r.b) ? { ...r, a: p, b: null } : { ...r, b: p }));
    } else {
      setClicks((c) => [...c, { x: p[0], y: p[1], label }]);
      setSeg(null);
    }
  };

  const refProblem = referenceProblem(reference, size);
  const refOk = refProblem === null;
  const gate = calibrateGate({ still: !!still, polygon: candidate?.polygon ?? null, reference, withoutReference: without,
                               service: status?.service ?? null, busy: busy !== null, size });
  const reach: Record<Step, boolean> = {
    1: true, 2: !!still, 3: !!still && (refOk || without), 4: !!candidate && !!still && (refOk || without), 5: !!result,
  };
  useEffect(() => {
    if (enabled) void loadHistory(); // another range
  }, [enabled, loadHistory]);
  const serviceDown = status?.service === "down";
  const isActive = !!result && versions?.active?.ver === result.version;
  const running = job && (job.state === "queued" || job.state === "running") ? job : null;

  const panel: Record<Step, ReactNode> = {
    1: (
      <div className="space-y-3">
        <p className="text-sm text-muted">Capture one frame of the camera now. Everything below works on this frozen still.</p>
        <Button onClick={() => void takeStill()} disabled={!enabled || busy !== null}>
          <RefreshCw size={14} className={busy === "still" ? "animate-spin" : ""} /> {still ? "Take a new still" : "Take a still"}
        </Button>
      </div>
    ),
    2: (
      <div className="space-y-3">
        <p className="rounded-md bg-primary/10 px-3 py-2 text-sm text-ink" data-testid="reference-hint">
          <strong>Type one distance you really know.</strong> Click its two end points on the still, then enter it in metres: a car's
          wheelbase from its spec sheet, a bay's width from the site plan, a length you measured with a tape. This anchors the scale
          (accuracy about ±5 %). Without it the metres come from typical car sizes and depth and can be about ±25 % off.
        </p>
        <ol className="list-decimal space-y-0.5 pl-5 text-xs text-muted">
          <li className={reference.a ? "text-success" : ""}>Click the first end point {reference.a && "✓"}</li>
          <li className={reference.b ? "text-success" : ""}>Click the second end point {reference.b && "✓"}</li>
          <li className={refOk ? "text-success" : ""}>Type the distance {refOk && "✓"}</li>
        </ol>
        <Field label="Distance between the two points (metres)">
          <input className={inputClass} inputMode="decimal" value={reference.metres} placeholder="e.g. 2.61" data-testid="reference-metres"
                 onChange={(e) => setReference((r) => ({ ...r, metres: e.target.value }))} />
        </Field>
        {refProblem && !without && <p className="text-xs text-danger" data-testid="reference-problem">{refProblem}</p>}
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => setReference(EMPTY_REFERENCE)} disabled={!reference.a}><Undo2 size={14} /> Clear points</Button>
        </div>
        <label className="flex items-start gap-2 text-xs text-muted">
          <input type="checkbox" checked={without} onChange={(e) => setWithout(e.target.checked)} data-testid="reference-skip" className="mt-0.5" />
          <span>Go on without a known distance. The calibration will be MEDIUM or LOW confidence (about ±25 % on every metre) and accepting it needs an extra confirmation.</span>
        </label>
        <Button onClick={() => setStep(3)} disabled={!(refOk || without)}>Next: parking surface</Button>
      </div>
    ),
    3: (
      <div className="space-y-3">
        <p className="text-sm text-muted">Click on the open parking ground (green). Add red clicks on what is not parking (road, grass, a wall). Then let the model outline it.</p>
        <div className="flex flex-wrap gap-2">
          <Pill active={label === 1} onClick={() => setLabel(1)}><span className="h-2 w-2 rounded-full bg-success" /> Ground</Pill>
          <Pill active={label === 0} onClick={() => setLabel(0)}><span className="h-2 w-2 rounded-full bg-danger" /> Not ground</Pill>
          <Button variant="ghost" onClick={() => (setClicks((c) => c.slice(0, -1)), setSeg(null))} disabled={!clicks.length}><Undo2 size={14} /> Undo</Button>
          <Button variant="ghost" onClick={() => (setClicks([]), setSeg(null))} disabled={!clicks.length}><Trash2 size={14} /> Clear</Button>
        </div>
        <p className="text-xs text-muted" data-testid="clicks-count">{clicks.filter((c) => c.label === 1).length} ground · {clicks.filter((c) => c.label === 0).length} not ground</p>
        <Button onClick={() => void findSurface()} disabled={!clicks.some((c) => c.label === 1) || busy !== null || serviceDown || !enabled}
                title={serviceDown ? "The setup service is not running" : undefined}>
          Find the surface
        </Button>
        {seg && (
          <div className="space-y-2" data-testid="segment-result">
            <p className="text-xs text-muted">The outline is the green polygon. Pick another candidate if it is better; add clicks and search again to refine.</p>
            <div className="flex flex-wrap gap-1.5">
              {seg.candidates.map((c, i) => (
                <Pill key={i} active={i === chosen} onClick={() => setChosen(i)} title={`SAM score ${c.score.toFixed(2)}`}>
                  <span data-testid={`candidate-${i}`}>{Math.round(c.area_frac * 100)} % of the image</span>
                </Pill>
              ))}
            </div>
            {seg.warnings.map((m) => <p key={m} className="text-xs text-warning">{m}</p>)}
            <Button onClick={() => setStep(4)} disabled={!reach[4]}>Next: calibrate</Button>
          </div>
        )}
      </div>
    ),
    4: (
      <div className="space-y-3">
        <ul className="space-y-1 text-sm" data-testid="inputs-summary">
          <li>{reference.a && refOk ? `✓ Known distance ${reference.metres} m` : without ? "• No known distance (±25 %)" : "✗ No known distance"}</li>
          <li>{candidate ? `✓ Surface outlined (${Math.round(candidate.area_frac * 100)} % of the image)` : "✗ No surface yet"}</li>
        </ul>
        <Field label="Camera lens">
          <select className={inputClass} value={lens} onChange={(e) => setLens(e.target.value as LensModel)} data-testid="camera-model">
            {(Object.keys(LENS) as LensModel[]).map((k) => <option key={k} value={k}>{LENS[k]}</option>)}
          </select>
        </Field>
        <Button onClick={() => void calibrate()} disabled={!gate.ok}>Calibrate</Button>
        {!gate.ok && gate.why && <p className="text-xs text-muted" data-testid="calibrate-gate">{gate.why}</p>}
      </div>
    ),
    5: result ? (
      <div className="space-y-3" data-testid="review">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ring-1 ${CONFIDENCE_TONE[result.confidence] ?? CONFIDENCE_TONE.NONE}`} data-testid="result-confidence">
            {result.confidence}
          </span>
          <span className="text-xs text-muted">{confidenceNote(result.confidence, result.scale_uncertainty)}</span>
        </div>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted">Camera height</dt><dd>{result.h.toFixed(2)} m</dd>
          <dt className="text-muted">Tilt</dt><dd>{result.pitch_deg.toFixed(1)}° down</dd>
          <dt className="text-muted">Surface</dt><dd>{result.area_m2.toFixed(0)} m²</dd>
          <dt className="text-muted">Bays</dt><dd>{result.capacity} ({result.bays_with_cars} on parked cars)</dd>
        </dl>
        {result.warnings.length > 0 && (
          <ul className="space-y-1 rounded-md bg-warning/15 px-3 py-2 text-xs text-ink" data-testid="result-warnings">
            {result.warnings.map((m) => <li key={m}>{m}</li>)}
          </ul>
        )}
        {needsUnverifiedOk(result.confidence) && (
          <label className="flex items-start gap-2 text-xs text-ink">
            <input type="checkbox" checked={unverifiedOk} onChange={(e) => setUnverifiedOk(e.target.checked)} data-testid="unverified-ok" className="mt-0.5" />
            <span>I accept this calibration without a typed reference: its metres are about ±{Math.round((result.scale_uncertainty ?? 0.25) * 100)} % off.
              (Better: go back to step 2 and give a known distance.)</span>
          </label>
        )}
        {isActive ? (
          <p className="rounded-md bg-success/10 px-3 py-2 text-sm text-success" data-testid="result-active"><Check size={14} className="mr-1 inline" /> Version {result.version} is this camera's active calibration.</p>
        ) : (
          <>
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => void activate(result.version, unverifiedOk)} disabled={busy !== null || (needsUnverifiedOk(result.confidence) && !unverifiedOk)}>
                <Check size={14} /> Accept and activate
              </Button>
              <Button variant="outline" onClick={() => void discard(result.version)} disabled={busy !== null}><Trash2 size={14} /> Discard draft</Button>
            </div>
            <p className="text-xs text-muted">A draft changes nothing. Accept makes it the active calibration; the live system picks it up on its next check, with no restart.</p>
          </>
        )}
      </div>
    ) : <p className="text-sm text-muted">Run the calibration first.</p>,
  };

  return (
    <div className="grid gap-4 xl:grid-cols-12">
      <div className="space-y-4 xl:col-span-7">
        <Card title={<span className="inline-flex items-center gap-2"><Ruler size={15} /> Camera setup</span>}
              actions={cameras.length > 1 && (
                <select className={`${inputClass} w-auto`} value={cam ?? ""} onChange={(e) => onCam(e.target.value)} data-testid="setup-camera">
                  {cameras.map((c) => <option key={c.id} value={c.id}>{c.display}</option>)}
                </select>
              )} bodyClass="p-0">
          <div className="relative w-full select-none bg-stage" style={{ aspectRatio: `${w} / ${h}` }} data-testid="setup-surface">
            {still ? (
              <>
                <img src={still.image} alt="Camera still" className="absolute inset-0 h-full w-full object-fill" draggable={false} data-testid="setup-still" />
                <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className={`absolute inset-0 h-full w-full ${step === 2 || step === 3 ? "cursor-crosshair" : ""}`}
                     onClick={onSurfaceClick} data-testid="setup-svg">
                  {step === 5 && draft && (
                    <>
                      <polygon points={pts(draft.record.surface.polygon, w, h)} fill="rgba(46,184,92,0.12)" stroke="#2eb85c" strokeWidth={w / 500} />
                      {draft.record.capacity.slots.map((s) => (
                        <polygon key={s.id} points={pts(s.poly, w, h)} fill={s.car ? "rgba(249,177,21,0.18)" : "rgba(46,184,92,0.2)"}
                                 stroke={s.car ? "#f9b115" : "#2eb85c"} strokeWidth={w / 600} data-testid={s.car ? "bay-car" : "bay-free"} />
                      ))}
                    </>
                  )}
                  {step !== 5 && candidate && (
                    <polygon points={pts(candidate.polygon, w, h)} fill="rgba(46,184,92,0.15)" stroke="#2eb85c" strokeWidth={w / 400} data-testid="surface-polygon" />
                  )}
                  {reference.a && reference.b && <line x1={reference.a[0] * w} y1={reference.a[1] * h} x2={reference.b[0] * w} y2={reference.b[1] * h} stroke="#39f" strokeWidth={w / 500} />}
                  {[reference.a, reference.b].map((p, i) => p && <circle key={i} cx={p[0] * w} cy={p[1] * h} r={w / 140} fill="#39f" stroke="white" strokeWidth={w / 700} data-testid="reference-point" />)}
                  {step !== 5 && clicks.map((c, i) => <circle key={i} cx={c.x * w} cy={c.y * h} r={w / 150} fill={c.label ? "#2eb85c" : "#e55353"} stroke="white" strokeWidth={w / 700} data-testid="surface-click" />)}
                </svg>
                <span className="pointer-events-none absolute top-2 left-2 rounded bg-stage/75 px-2 py-0.5 text-[11px] text-white">Still · {w}×{h}</span>
              </>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center gap-2 p-6 text-center text-sm text-white/70" data-testid="setup-empty">
                <ImageIcon size={18} /> {enabled ? "Take a still to begin." : "Camera setup is not available."}
              </div>
            )}
          </div>
        </Card>
        <StaleBanner event={staleBanner(events ?? [], versions?.active?.ver)} onReanchor={enabled ? () => setAnchor({ from: null }) : undefined} />
        {enabled && anchor && versions && (
          <ReanchorCard base={base} versions={versions.versions} from={anchor.from} stale={!!staleBanner(events ?? [], versions.active?.ver)} stop={stop}
                        fail={fail} onChanged={refreshAll} onFull={() => (setAnchor(null), setStep(1))} onClose={() => setAnchor(null)} />
        )}
        <VersionsCard versions={versions} busy={busy !== null} onActivate={(v, unverified) => void activate(v.ver, unverified)}
                      onDiscard={(v) => void discard(v.ver)} onReanchor={(v) => setAnchor({ from: v.ver })} />
        <HistoryCard history={history} hours={hours} onHours={setHours} />
        <EventsCard events={events} />
      </div>

      <div className="space-y-4 xl:col-span-5">
        <StatusCard status={status} />
        {enabled && (
          <Card title="Setup">
            <ol className="mb-4 flex flex-wrap gap-1.5" data-testid="steps">
              {STEPS.map((s) => (
                <li key={s.n}>
                  <button type="button" disabled={!reach[s.n]} onClick={() => setStep(s.n)} data-testid={`step-${s.n}`} aria-current={step === s.n ? "step" : undefined}
                          className={`rounded-full px-2.5 py-1 text-xs font-medium ${step === s.n ? "bg-primary text-white" : "border border-line text-muted hover:text-primary"} disabled:opacity-40`}>
                    {s.n} · {s.label}
                  </button>
                </li>
              ))}
            </ol>
            {panel[step]}
            {running && (
              <div className="mt-4" data-testid="job-progress">
                <p className="mb-1 text-xs text-muted">{running.stage}…</p>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-line"><div className="h-full bg-primary transition-all" style={{ width: `${Math.round(running.progress * 100)}%` }} /></div>
              </div>
            )}
            {message && (
              <p className={`mt-4 rounded-md px-3 py-2 text-sm ${message.tone === "ok" ? "bg-success/10 text-success" : "bg-danger/10 text-danger"}`} data-testid="setup-message">{message.text}</p>
            )}
          </Card>
        )}
      </div>
    </div>
  );
}

function StatusCard({ status }: { status: SetupStatus | null }) {
  if (!status) return <Card title="Camera setup"><p className="text-sm text-muted" data-testid="setup-status">Checking…</p></Card>;
  if (!status.enabled) {
    return (
      <Card title={<span className="inline-flex items-center gap-2"><Lock size={14} /> Camera setup is switched off</span>}>
        <p className="text-sm text-muted" data-testid="setup-status">
          This server has not enabled camera setup. Set <code>CARPARK_GEOMETRY_ENABLED=1</code> for the api service, recreate it, and start the on-demand setup service
          (<code>docker compose --profile setup up -d setup</code>). Nothing else changes while it is off.
        </p>
      </Card>
    );
  }
  return (
    <Card title="Setup service">
      <p className={`text-sm ${status.service === "up" ? "text-success" : "text-danger"}`} data-testid="setup-status">
        {status.service === "up"
          ? "The setup service is running."
          : "The setup service is not running: start it with docker compose --profile setup up -d setup (models load on first use, which takes a minute)."}
      </p>
    </Card>
  );
}

function VersionsCard({ versions, busy, onActivate, onDiscard, onReanchor }: {
  versions: Versions | null;
  busy: boolean;
  onReanchor: (v: GeometrySummary) => void;
  onActivate: (v: GeometrySummary, acceptUnverified: boolean) => void;
  onDiscard: (v: GeometrySummary) => void;
}) {
  const [confirming, setConfirming] = useState<number | null>(null); // a version without a typed reference: ask once more
  if (!versions) return null;
  return (
    <Card title="Calibrations of this camera" bodyClass="p-0">
      {versions.versions.length === 0 ? (
        <p className="p-4 text-sm text-muted" data-testid="versions-empty">Not calibrated yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm" data-testid="versions-table">
            <thead className="border-b border-line text-xs text-muted uppercase">
              <tr><th className="px-3 py-2">Version</th><th className="px-3 py-2">State</th><th className="px-3 py-2">Confidence</th><th className="px-3 py-2">Height</th>
                  <th className="px-3 py-2">Surface</th><th className="px-3 py-2">Bays</th><th className="px-3 py-2" title="Does the calibration carry a reference for the drift guard?">Drift guard</th>
                  <th className="px-3 py-2">Made</th><th className="px-3 py-2" /></tr>
            </thead>
            <tbody className="divide-y divide-line">
              {versions.versions.map((v) => (
                <tr key={v.ver} data-testid={`version-${v.ver}`}>
                  <td className="px-3 py-2 font-medium">{v.ver}{v.reanchor?.parent ? <span className="ml-1 text-xs font-normal text-muted" data-testid={`version-parent-${v.ver}`}>from v{v.reanchor.parent}</span> : null}</td>
                  <td className="px-3 py-2"><span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${STATE_TONE[v.state]}`}>{v.state}</span></td>
                  <td className="px-3 py-2">{v.confidence ?? "-"}{v.reference_used ? " · reference" : ""}</td>
                  <td className="px-3 py-2">{v.h != null ? `${v.h.toFixed(2)} m` : "-"}</td>
                  <td className="px-3 py-2">{v.area_m2 != null ? `${Math.round(v.area_m2)} m²` : "-"}</td>
                  <td className="px-3 py-2">{v.capacity ?? "-"}</td>
                  <td className="px-3 py-2 text-xs" data-testid={`version-guard-${v.ver}`}>{v.drift_guard ? "✓" : "–"}</td>
                  <td className="px-3 py-2 text-xs text-muted">{when(v.created_at)}</td>
                  <td className="px-3 py-2 text-right">
                    {(v.state === "draft" || v.state === "superseded") && (
                      confirming === v.ver ? (
                        <span className="inline-flex flex-wrap items-center justify-end gap-1.5 text-xs" data-testid={`version-confirm-${v.ver}`}>
                          Not anchored by a typed distance (about ±25 %).
                          <Button disabled={busy} onClick={() => (setConfirming(null), onActivate(v, true))}><span data-testid={`version-confirm-yes-${v.ver}`}>Activate anyway</span></Button>
                          <Button variant="ghost" onClick={() => setConfirming(null)}>Cancel</Button>
                        </span>
                      ) : (
                        <Button variant="outline" disabled={busy} title={v.state === "superseded" ? "Roll back to this version" : "Make this the active calibration"}
                                onClick={() => (needsUnverifiedOk(v.confidence) ? setConfirming(v.ver) : onActivate(v, false))}>
                          <span data-testid={`version-activate-${v.ver}`}>{v.state === "superseded" ? "Roll back" : "Activate"}</span>
                        </Button>
                      )
                    )}
                    {v.state !== "draft" && (
                      <Button variant="ghost" disabled={busy || !v.drift_guard}
                              title={v.drift_guard ? "The camera was moved: carry this calibration over to its new position" : "Made before the drift guard: it has no stored background to match, do the full setup"}
                              onClick={() => onReanchor(v)}>
                        <span data-testid={`version-reanchor-${v.ver}`}>Re-anchor</span>
                      </Button>
                    )}
                    {v.state === "draft" && (
                      <Button variant="ghost" disabled={busy} onClick={() => onDiscard(v)}><span data-testid={`version-discard-${v.ver}`}>Discard</span></Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
