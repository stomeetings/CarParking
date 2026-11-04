import { AlertTriangle, Pause, Play, PlugZap } from "lucide-react";
import { useState } from "react";
import { ApiError, api, withCam } from "../api";
import type { Link } from "../hooks";
import type { StateMessage } from "../types";
import { Button } from "./ui";

export type MonitorKind = "ok" | "starting" | "off" | "down" | "offline";

/** Why the dashboard has (or lacks) live slot states. Without the worker tracking, there are no FREE / BUSY states
 * to show: the counters and the overlay stay empty, and this says why instead of showing bare dashes. */
export function monitorState(live: StateMessage | null, link: Link): { kind: MonitorKind; text: string } {
  if (link !== "open") return { kind: "offline", text: "The dashboard can't reach the backend." };
  if (!live?.connected) {
    const age = live?.heartbeat_age;
    return { kind: "down", text: `The worker is down${age != null ? ` (no heartbeat for ${Math.round(age)} s)` : ""}: nothing is tracked or logged.` };
  }
  if (!live.snapshot.enabled) {
    return { kind: "off", text: "Monitoring is off: the worker is idle, so there are no FREE / BUSY states and no entries or exits are logged." };
  }
  if (!live.snapshot.counts) return { kind: "starting", text: `Starting: ${live.snapshot.status ?? "loading"}…` };
  return { kind: "ok", text: "" };
}

/** Shown above the live view whenever slot states are missing, with the switch to start monitoring. */
export function MonitorBanner({ live, link, onNeedAdmin, cam }: {
  cam: string | null;
  live: StateMessage | null;
  link: Link;
  onNeedAdmin: (reason: string, retry?: () => void) => void; // retry: run again once the admin token is accepted
}) {
  const { kind, text } = monitorState(live, link);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (kind === "ok") return null;
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.put(withCam("/api/settings", cam), { enabled: true });
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 401 || err.status === 503) onNeedAdmin("Turning monitoring on needs the admin token.", start);
      else setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  const tone = kind === "starting" ? "border-info/40 bg-info/10" : kind === "off" ? "border-warning/50 bg-warning/10" : "border-danger/40 bg-danger/10";
  const icon = kind === "off" ? <Pause size={18} className="text-[#b07800]" /> : kind === "starting"
    ? <Play size={18} className="text-info" /> : kind === "offline" ? <PlugZap size={18} className="text-danger" /> : <AlertTriangle size={18} className="text-danger" />;
  return (
    <div className={`flex flex-wrap items-center gap-3 rounded-lg border px-4 py-3 ${tone}`} data-testid="monitor-banner">
      {icon}
      <p className="min-w-0 flex-1 text-sm text-ink">{text}</p>
      {kind === "off" && <Button onClick={start} disabled={busy}><Play size={14} /> Turn monitoring on</Button>}
      {error && <p className="w-full text-xs text-danger">{error}</p>}
    </div>
  );
}
