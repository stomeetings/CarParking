import { AnimatePresence, motion } from "framer-motion";
import { ArrowDownLeft, ArrowUpRight, Ban, CircleCheck, CircleHelp, Gauge, PackageX, ParkingSquare, SquareCheck, SquareX, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { autoGridText, clock, eventInfo, eventSlot } from "../lib";
import type { ParkingEvent, SlotFilter, SlotState, Snapshot } from "../types";
import { Card } from "./ui";

function Kpi({ label, value, icon, tone, hint, active = false, dimmed = false, onClick, testid }: {
  label: string;
  value: number | string;
  icon: ReactNode;
  tone: string;
  hint?: string;
  active?: boolean; // its filter is on
  dimmed?: boolean; // another card's filter is on
  onClick?: () => void;
  testid?: string;
}) {
  const body = (
    <>
      <div className="text-3xl leading-none font-bold tabular-nums">{value}</div>
      <div className="mt-1.5 text-sm font-medium text-white/85">{label}</div>
      <div className="absolute -right-2 -bottom-2 text-white/20">{icon}</div>
    </>
  );
  const base = `relative overflow-hidden rounded-lg p-4 text-left text-white shadow-sm ${tone}`;
  if (!onClick) return <div className={base} title={hint}>{body}</div>;
  return (
    <button type="button" onClick={onClick} aria-pressed={active} title={hint} data-testid={testid}
            className={`${base} cursor-pointer transition duration-150 hover:-translate-y-0.5 hover:shadow-md hover:brightness-110
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink/60
              ${active ? "ring-4 ring-white/90 shadow-[0_0_0_6px_rgba(51,153,255,0.35),0_8px_24px_rgba(0,0,0,0.25)]" : ""}
              ${dimmed ? "opacity-50 saturate-50" : ""}`}>
      {body}
    </button>
  );
}

/** FREE / BUSY / TOTAL / UNKNOWN as CoreUI-style widgets. With onFilter they are filter buttons for the video's slot
 * overlay: Free / Busy / Unknown show only those slots; Total slots, or the active card again, shows all. */
export function Counters({ snapshot, note, filter = null, onFilter }: {
  snapshot: Snapshot | undefined;
  note?: string;
  filter?: SlotFilter;
  onFilter?: (f: SlotFilter) => void;
}) {
  const c = snapshot?.counts;
  const dash = (v: number | undefined) => (v == null ? "–" : v);
  const pick = (f: SlotFilter) => onFilter && (() => onFilter(f === filter ? null : f));
  const show = (state: SlotState | "ENCROACHED" | "BLOCKED") => `${filter === state ? "Showing" : "Show"} only ${state} slots on the video${filter === state ? " (click: show all)" : ""}`;
  return (
    <div className="grid grid-cols-2 gap-3" data-testid="counters">
      {!c && note && (
        <p className="col-span-2 rounded-md bg-white px-3 py-2 text-xs font-medium text-muted ring-1 ring-line" data-testid="counters-note">{note}</p>
      )}
      <Kpi label="Free" value={dash(c?.free)} tone="bg-gradient-to-br from-success to-[#1b9e4b]" icon={<SquareCheck size={64} />}
           onClick={pick("FREE")} active={filter === "FREE"} dimmed={filter != null && filter !== "FREE"} testid="filter-free"
           hint={onFilter ? show("FREE") : undefined} />
      <Kpi label="Busy" value={dash(c?.busy)} tone="bg-gradient-to-br from-danger to-[#d63939]" icon={<SquareX size={64} />}
           onClick={pick("BUSY")} active={filter === "BUSY"} dimmed={filter != null && filter !== "BUSY"} testid="filter-busy"
           hint={onFilter ? show("BUSY") : c ? `${c.hidden} hidden behind other vehicles (state kept)` : undefined} />
      <Kpi label="Total slots" value={dash(c?.slots)} tone="bg-gradient-to-br from-primary to-primary-dark" icon={<ParkingSquare size={64} />}
           onClick={onFilter && (() => onFilter(null))} active={false} dimmed={false} testid="filter-all"
           hint={onFilter ? "Show all slots on the video" : undefined} />
      <Kpi label="Unknown" value={dash(c?.unknown)} tone="bg-gradient-to-br from-warning to-[#e29e00]" icon={<CircleHelp size={64} />}
           onClick={pick("UNKNOWN")} active={filter === "UNKNOWN"} dimmed={filter != null && filter !== "UNKNOWN"} testid="filter-unknown"
           hint={onFilter ? show("UNKNOWN") : "not confirmed yet, or a phantom / duplicate slot"} />
      {c?.blocked_objects != null && (
        <div className="col-span-2" data-testid="blocked-card">
          <Kpi label="Blocked by object (free, not usable)" value={c.blocked_objects} tone="bg-gradient-to-br from-[#5b6475] to-[#3f4757]" icon={<PackageX size={64} />}
               onClick={pick("BLOCKED")} active={filter === "BLOCKED"} dimmed={filter != null && filter !== "BLOCKED"} testid="filter-blocked"
               hint={onFilter ? show("BLOCKED") : `a potted plant, vase or box stands in these free bays; the Free card counts only the usable ones (${c.available ?? c.free})`} />
        </div>
      )}
      {c?.encroached != null && (
        <div className="col-span-2" data-testid="encroached-card">
          <Kpi label="Encroached (free, not usable)" value={c.encroached} tone="bg-gradient-to-br from-[#f08c00] to-[#c97100]" icon={<TriangleAlert size={64} />}
               onClick={pick("ENCROACHED")} active={filter === "ENCROACHED"} dimmed={filter != null && filter !== "ENCROACHED"} testid="filter-encroached"
               hint={onFilter ? show("ENCROACHED") : `a neighbour's vehicle stands on these free bays; the Free card counts only the usable ones (${c.available ?? c.free})`} />
        </div>
      )}
    </div>
  );
}

/** What the worker is doing (the Streamlit page's status and metrics lines). */
export function StatusCard({ snapshot, heartbeat }: { snapshot: Snapshot | undefined; heartbeat: number | null | undefined }) {
  const s = snapshot;
  const rows: [string, ReactNode][] = s
    ? [
        ["Camera", s.url ? <span className="font-mono text-xs">{s.url}</span> : s.camera],
        ["Stream", `${s.status ?? ""}${s.shape ? ` · ${s.shape[0]}×${s.shape[1]}` : ""}`],
        ["Speed", s.rate != null ? `${s.rate.toFixed(2)} frames/s · frame age ${Math.max(s.frame_age ?? 0, 0).toFixed(1)} s` : "–"],
        ["Detector", `${s.detector ?? "–"}${s.model?.auto ? " · auto-selected" : ""}`],
        ["Slot map", `${s.auto_slots ? autoGridText(s.auto_slots) : s.mode === "fixed" ? "fixed map" : `learning ${s.learning ?? 0} · phantom boxes ignored ${s.dropped ?? 0}`}`],
        ...(s.obstacle ? [["Obstacle filter", `${s.obstacle.rejected} dropped${Object.keys(s.obstacle.reasons).length
          ? ` (${Object.entries(s.obstacle.reasons).map(([k, v]) => `${k.replace("_", " ")} ${v}`).join(", ")})` : ""} in the last frame`] as [string, ReactNode]] : []),
        ["This frame", `${s.vehicles ?? 0} vehicles · ${s.others ?? 0} other${s.raw_counts && Object.keys(s.raw_counts).length
          ? ` · raw ${Object.entries(s.raw_counts).map(([k, v]) => `${k} ${v}`).join(", ")}` : ""}`],
        ["Heartbeat", heartbeat != null ? `${heartbeat.toFixed(1)} s ago` : "–"],
      ]
    : [];
  return (
    <Card title={<span className="flex items-center gap-2"><Gauge size={16} className="text-primary" /> Worker</span>}>
      {rows.length ? (
        <dl className="grid grid-cols-[6.5rem_1fr] gap-x-3 gap-y-1.5 text-sm">
          {rows.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted">{k}</dt>
              <dd className="min-w-0 break-words text-ink">{v}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="text-sm text-muted">No status from the worker yet.</p>
      )}
      {s?.saved ? <p className="mt-3 text-xs text-muted">📸 {s.saved} frame(s) saved, last {s.last_saved}</p> : null}
    </Card>
  );
}

const TICKER_TONE = { success: "bg-success/15 text-success", info: "bg-info/15 text-info", warning: "bg-warning/15 text-warning", muted: "bg-line text-muted" };
const TICKER_ICON: Record<string, ReactNode> = {
  entry: <ArrowDownLeft size={16} />, exit: <ArrowUpRight size={16} />, encroached: <TriangleAlert size={16} />,
  encroach_cleared: <CircleCheck size={16} />, rejected: <Ban size={16} />, object_blocked: <PackageX size={16} />, object_cleared: <CircleCheck size={16} />,
};

/** The newest events, sliding in as the worker logs them: vehicles in and out, and the notices (a bay a neighbour's vehicle stands on,
 * an object the non-vehicle filter ignored), each told apart by its icon, colour and wording. */
export function EventTicker({ events }: { events: ParkingEvent[] }) {
  return (
    <Card title="Live events" actions={<span className="text-xs text-muted">newest first</span>} bodyClass="p-0">
      <ul className="max-h-[22rem] divide-y divide-line overflow-y-auto" data-testid="ticker">
        <AnimatePresence initial={false}>
          {events.slice(0, 30).map((e) => (
            <motion.li key={e.id} layout initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                       className="flex items-center gap-3 px-4 py-2.5">
              <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${TICKER_TONE[eventInfo(e).tone]}`}
                    data-testid="ticker-icon" data-kind={e.event}>
                {TICKER_ICON[e.event] ?? <Ban size={16} />}
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-ink">
                  {eventSlot(e)} · {eventInfo(e).text}
                </p>
                {e.note && <p className="truncate text-xs text-muted" title={e.note}>{e.note}</p>}
              </div>
              <time className="shrink-0 text-xs tabular-nums text-muted">{clock(e.time)}</time>
            </motion.li>
          ))}
        </AnimatePresence>
        {!events.length && <li className="px-4 py-6 text-center text-sm text-muted">No events logged yet.</li>}
      </ul>
    </Card>
  );
}
