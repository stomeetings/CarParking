import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import type { Link } from "../hooks";

const SYNC_KEY = "carparking.syncEvery";
export const SYNC_CHOICES = [20, 30, 5, 0]; // seconds; 0 = live (about once a second)
export const DEFAULT_SYNC = 20;

export const savedSyncEvery = (): number => {
  try {
    const v = Number(localStorage.getItem(SYNC_KEY) ?? DEFAULT_SYNC);
    return SYNC_CHOICES.includes(v) ? v : DEFAULT_SYNC;
  } catch {
    return DEFAULT_SYNC;
  }
};
export const saveSyncEvery = (v: number) => {
  try {
    localStorage.setItem(SYNC_KEY, String(v));
  } catch {
    /* not remembered in a private window */
  }
};

/** Ticks on its own (every 0.5 s): only this badge re-renders, not the dashboard. */
function useNow(every = 500) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), every);
    return () => clearInterval(t);
  }, [every]);
  return now;
}

export const ago = (s: number) => (s < 10 ? `${s.toFixed(1)} s` : s < 120 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);

/** "Auto-sync 20 s · updated 4 s ago · next ≤ 16 s": routine updates come every `every` seconds (picker), a change
 * (a car parks or leaves, a slot's state, the worker's connection, a setting) at once. Amber when nothing came for
 * the interval + 10 s, grey while (re)connecting. */
export function SyncStatus({ at, link, every, onEvery }: { at: number | null; link: Link; every: number; onEvery: (s: number) => void }) {
  const now = useNow();
  const age = at == null ? null : Math.max(0, (now - at) / 1000);
  const period = every || 1;
  const [tone, text] = link !== "open"
    ? ["bg-muted", link === "unauthorized" ? "Viewer token needed" : link === "connecting" ? "Connecting…" : "Reconnecting…"]
    : age == null ? ["bg-warning", "Waiting for the first update…"]
      : age >= period + 10 ? ["bg-warning", `No update for ${ago(age)}`]
        : ["bg-success", `updated ${ago(age)} ago${every ? ` · next ≤ ${Math.max(0, Math.ceil(every - age))} s` : ""}`];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-white py-0.5 pr-2.5 pl-2 text-xs text-muted tabular-nums"
          title="Counts, slots and stats update in place (never a page reload). Routine updates come at this interval; a car parking or leaving, a slot changing state, the worker going down or a setting changing arrives at once."
          data-testid="sync-status">
      <span className={`h-2 w-2 rounded-full ${tone}`} />
      <label className="inline-flex items-center gap-1">
        <span className="sr-only">Auto-sync interval</span>
        Auto-sync
        <select value={every} onChange={(e) => onEvery(Number(e.target.value))} data-testid="sync-every"
                className="rounded border border-line bg-white px-1 py-0.5 text-xs font-semibold text-ink">
          {SYNC_CHOICES.map((s) => <option key={s} value={s}>{s ? `${s} s` : "live (1 s)"}</option>)}
        </select>
      </label>
      <span data-testid="sync-text">· {text}</span>
    </span>
  );
}

/** Refresh data: reconnects the live streams and refetches the camera list and the ticker in the background (the
 * server answers with the current state at once). Spins until the next update arrives (at most 5 s). */
export function RefreshData({ onRefresh, at }: { onRefresh: () => void; at: number | null }) {
  const [since, setSince] = useState<number | null>(null);
  useEffect(() => {
    if (since != null && at != null && at > since) setSince(null); // fresh data is in
  }, [at, since]);
  useEffect(() => {
    if (since == null) return;
    const t = setTimeout(() => setSince(null), 5000);
    return () => clearTimeout(t);
  }, [since]);
  return (
    <button type="button" onClick={() => (setSince(Date.now()), onRefresh())} disabled={since != null}
            title="Fetch the latest counts, slots and stats now (no page reload)" data-testid="refresh-data"
            className="inline-flex items-center gap-1 rounded-md border border-line bg-white px-2.5 py-1.5 text-xs font-semibold text-muted hover:text-primary disabled:opacity-70">
      <RefreshCw size={14} className={since != null ? "animate-spin" : ""} /> {since != null ? "Refreshing…" : "Refresh data"}
    </button>
  );
}
