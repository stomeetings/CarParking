import { Check, Crosshair, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import {
  type GeometrySummary, type GeometryVersion, type ReanchorResult, type SetupJob, defaultParent, needsUnverifiedOk, reanchorGate, reanchorLines,
  reanchorParents, waitForJob,
} from "../setupflow";
import type { Still } from "./SlotEditor";
import { Button, Card } from "./ui";

const pts = (poly: [number, number][], w: number, h: number) => poly.map(([x, y]) => `${x * w},${y * h}`).join(" ");

/** Re-anchor a calibration after the camera was bumped or re-aimed, without redoing the setup: a fresh still is matched to the
 * calibration's stored background, the turn is solved and the calibration is carried over (same optics, height and scale).
 * Needs no setup service. The result is a draft linked to its parent; the Admin reviews the bays on the new still and accepts it
 * (api/geometry_routes.py: POST .../setup/reanchor). */
export function ReanchorCard({ base, versions, from, stale, stop, fail, onChanged, onFull, onClose }: {
  base: string;
  versions: GeometrySummary[];
  from: number | null; // the version the Admin picked in the table (null: the server's default)
  stale: boolean;
  stop: { current: { cancelled: boolean } };
  fail: (e: unknown, retry?: () => void) => void;
  onChanged: () => void; // versions / events changed: reload
  onFull: () => void; // "do the full setup instead"
  onClose: () => void;
}) {
  const parents = reanchorParents(versions);
  const [still, setStill] = useState<Still | null>(null);
  const [busy, setBusy] = useState<"still" | "run" | "accept" | null>(null);
  const [job, setJob] = useState<SetupJob | null>(null);
  const [result, setResult] = useState<ReanchorResult | null>(null);
  const [draft, setDraft] = useState<GeometryVersion | null>(null);
  const [okUnverified, setOkUnverified] = useState(false);
  const [note, setNote] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [done, setDone] = useState(false);
  const parent = (from != null ? parents.find((v) => v.ver === from) : null) ?? defaultParent(versions);
  useEffect(() => { setResult(null); setDraft(null); setDone(false); setNote(null); }, [from]);
  const gate = reanchorGate({ still: !!still, parents: parents.length, busy: busy !== null });
  const [w, h] = still ? [still.width, still.height] : [1920, 1080];

  const takeStill = async () => {
    setBusy("still"); setNote(null);
    try {
      setStill(await api.post<Still>(`${base}/still`, {}));
      setResult(null); setDraft(null); setDone(false);
    } catch (e) {
      fail(e, () => void takeStill());
    } finally {
      setBusy(null);
    }
  };

  const run = async () => {
    if (!still || !parent) return;
    setBusy("run"); setNote(null); setJob(null);
    try {
      const started = await api.post<SetupJob<ReanchorResult>>(`${base}/reanchor`, { still: still.id, from_version: parent.ver });
      const fin = await waitForJob<ReanchorResult>(started.id, (j) => setJob(j as SetupJob), stop.current);
      if (fin.state === "error" || !fin.result) {
        setNote({ tone: "error", text: fin.error ?? "the re-anchoring failed" });
      } else {
        setResult(fin.result);
        setOkUnverified(false);
        setDraft(await api.get<GeometryVersion>(`${base}/geometry/${fin.result.version}`));
        onChanged();
      }
    } catch (e) {
      if ((e as Error).message !== "cancelled") fail(e);
    } finally {
      setBusy(null);
    }
  };

  const accept = async () => {
    if (!result) return;
    setBusy("accept"); setNote(null);
    try {
      await api.post(`${base}/activate`, { version: result.version, accept_unverified: okUnverified });
      setDone(true);
      setNote({ tone: "ok", text: `Version ${result.version} is now this camera's active calibration (re-anchored from v${result.parent}). The live system reads it on its next check: nothing was restarted.` });
      onChanged();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(null);
    }
  };

  const discard = async () => {
    if (!result) return;
    setBusy("accept");
    try {
      await api.delete(`${base}/draft?version=${result.version}`);
      setResult(null); setDraft(null);
      setNote({ tone: "ok", text: `Draft ${result.version} discarded.` });
      onChanged();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(null);
    }
  };

  const running = job && (job.state === "queued" || job.state === "running") ? job : null;
  return (
    <Card title={<span className="inline-flex items-center gap-2"><Crosshair size={15} /> Re-anchor calibration</span>}
          actions={<Button variant="ghost" onClick={onClose}>Close</Button>}>
      <div className="space-y-3" data-testid="reanchor">
        <p className="text-sm text-muted">
          {stale ? "The camera moved. " : ""}Carry the calibration over to the camera's new position without redoing the setup: no surface to outline, no distance to type.
          Take a fresh still with the camera where it is now; the calibration's stored background is matched to it and the turn is solved.
          The scale, camera height and lens stay as calibrated. This works with the setup service stopped.
        </p>
        {parents.length === 0 ? (
          <p className="rounded-md bg-warning/15 px-3 py-2 text-sm text-ink" data-testid="reanchor-none">
            No calibration of this camera has a stored background reference (older versions were made before the drift guard). Do the full setup.
          </p>
        ) : (
          <p className="text-sm" data-testid="reanchor-parent">From calibration <strong>v{parent?.ver}</strong> ({parent?.state}, {parent?.confidence ?? "-"}).</p>
        )}
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => void takeStill()} disabled={busy !== null || parents.length === 0}>
            <RefreshCw size={14} className={busy === "still" ? "animate-spin" : ""} /> <span data-testid="reanchor-still">{still ? "Take a new still" : "Take a fresh still"}</span>
          </Button>
          <Button onClick={() => void run()} disabled={!gate.ok || done || !!result}><span data-testid="reanchor-run">Re-anchor</span></Button>
          <Button variant="ghost" onClick={onFull}><span data-testid="reanchor-full">Do the full setup instead</span></Button>
        </div>
        {!gate.ok && gate.why && parents.length > 0 && <p className="text-xs text-muted" data-testid="reanchor-gate">{gate.why}</p>}
        {still && (
          <div className="relative w-full select-none overflow-hidden rounded-md bg-ink" style={{ aspectRatio: `${w} / ${h}` }} data-testid="reanchor-surface">
            <img src={still.image} alt="Fresh camera still" className="absolute inset-0 h-full w-full object-fill" draggable={false} />
            {draft && (
              <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
                <polygon points={pts(draft.record.surface.polygon, w, h)} fill="rgba(46,184,92,0.12)" stroke="#2eb85c" strokeWidth={w / 500} />
                {draft.record.capacity.slots.map((s) => (
                  <polygon key={s.id} points={pts(s.poly, w, h)} fill={s.car ? "rgba(249,177,21,0.18)" : "rgba(46,184,92,0.2)"}
                           stroke={s.car ? "#f9b115" : "#2eb85c"} strokeWidth={w / 600} data-testid={s.car ? "reanchor-bay-car" : "reanchor-bay-free"} />
                ))}
              </svg>
            )}
          </div>
        )}
        {running && (
          <div data-testid="reanchor-progress">
            <p className="mb-1 text-xs text-muted">{running.stage}…</p>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-line"><div className="h-full bg-primary transition-all" style={{ width: `${Math.round(running.progress * 100)}%` }} /></div>
          </div>
        )}
        {result && (
          <div className="space-y-3" data-testid="reanchor-review">
            <ul className="space-y-0.5 text-sm" data-testid="reanchor-report">
              {reanchorLines(result.reanchor).map((l) => <li key={l}>{l}</li>)}
            </ul>
            <p className="text-xs text-muted">Check that the bays sit on the same places as before (orange: on a parked car). Camera height {result.h.toFixed(2)} m, {result.area_m2.toFixed(0)} m², {result.capacity} bays · scale {result.confidence}.</p>
            {result.warnings.length > 0 && (
              <ul className="space-y-1 rounded-md bg-warning/15 px-3 py-2 text-xs text-ink">{result.warnings.map((m) => <li key={m}>{m}</li>)}</ul>
            )}
            {needsUnverifiedOk(result.confidence) && !done && (
              <label className="flex items-start gap-2 text-xs text-ink">
                <input type="checkbox" checked={okUnverified} onChange={(e) => setOkUnverified(e.target.checked)} data-testid="reanchor-unverified" className="mt-0.5" />
                <span>I accept this without a typed reference (the scale was never anchored; about ±{Math.round((result.scale_uncertainty ?? 0.25) * 100)} % on the metres).</span>
              </label>
            )}
            {!done && (
              <div className="flex flex-wrap gap-2">
                <Button onClick={() => void accept()} disabled={busy !== null || (needsUnverifiedOk(result.confidence) && !okUnverified)}>
                  <Check size={14} /> <span data-testid="reanchor-accept">Accept and activate</span>
                </Button>
                <Button variant="outline" onClick={() => void discard()} disabled={busy !== null}><Trash2 size={14} /> <span data-testid="reanchor-discard">Discard</span></Button>
              </div>
            )}
          </div>
        )}
        {note && <p className={`rounded-md px-3 py-2 text-sm ${note.tone === "ok" ? "bg-success/10 text-success" : "bg-danger/10 text-danger"}`} data-testid="reanchor-message">{note.text}</p>}
      </div>
    </Card>
  );
}
