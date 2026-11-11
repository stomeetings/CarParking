import { ChevronLeft, ChevronRight, Download, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { withCam, withToken } from "../api";
import { usePoll } from "../hooks";
import { CAMERA_EVENT_LABELS, clock, duration, eventInfo, eventSlot, pageList, sourceLabel } from "../lib";
import type { CameraEvent, ParkingEvent, SessionPage, SyncStatus } from "../types";
import { Button, Card, Pill } from "./ui";

export const PAGE_SIZE = 20;
/** The History page warns when the log has differed from the live count this long (the worker logs it at the same time). */
export const MISMATCH_WARN_S = 30;

/** A history scope "<camera> | <detector>[ | auto vN]" without the camera's name (the page is that camera's). */
export const scopeLabel = (scope: string | undefined) => (scope ?? "").split(" | ").slice(1).join(" | ") || "-";

/** Previous / page numbers / next, and which rows are shown of how many. */
export function Pagination({ page, pages, total, size, open, onPage }: {
  page: number; pages: number; total: number; size: number; open?: number; onPage: (page: number) => void;
}) {
  const first = total ? (page - 1) * size + 1 : 0;
  const btn = "inline-flex h-8 min-w-8 items-center justify-center rounded-md border px-2 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40";
  return (
    <nav className="flex flex-wrap items-center justify-between gap-2 border-t border-line px-4 py-3" aria-label="Pages" data-testid="pagination">
      <span className="text-xs text-muted" data-testid="page-info">
        {total ? `Showing ${first}–${Math.min(page * size, total)} of ${total} sessions` : "No sessions yet"}
        {open ? <> · <span className="font-semibold text-danger">{open} parked now</span> (listed first)</> : null}
      </span>
      <div className="flex flex-wrap items-center gap-1">
        <button type="button" className={`${btn} border-line bg-surface text-ink hover:border-primary hover:text-primary`}
                disabled={page <= 1} onClick={() => onPage(page - 1)} aria-label="Previous page">
          <ChevronLeft size={14} /> Previous
        </button>
        {pageList(page, pages).map((p, i) => p === null ? (
          <span key={`gap${i}`} className="px-1 text-xs text-muted">…</span>
        ) : (
          <button key={p} type="button" aria-current={p === page ? "page" : undefined} onClick={() => onPage(p)}
                  className={`${btn} ${p === page ? "border-primary bg-primary text-white" : "border-line bg-surface text-ink hover:border-primary hover:text-primary"}`}>
            {p}
          </button>
        ))}
        <button type="button" className={`${btn} border-line bg-surface text-ink hover:border-primary hover:text-primary`}
                disabled={page >= pages} onClick={() => onPage(page + 1)} aria-label="Next page">
          Next <ChevronRight size={14} />
        </button>
      </div>
    </nav>
  );
}

export function HistoryPage({ tokenKey, cam }: { tokenKey: string; cam: string | null }) {
  const [tab, setTab] = useState<"sessions" | "events" | "camera">("sessions");
  const [page, setPage] = useState(1);
  // "All versions": the camera's sessions of every detector / auto grid version, with a Scope column (default: the one in use)
  const [allVersions, setAllVersions] = useState(false);
  useEffect(() => setPage(1), [cam, allVersions]); // another camera / listing: its first page
  // (3 s: the History table follows the live count; the worker syncs to PostgreSQL every 2 s)
  const sessions = usePoll<SessionPage>(tab === "sessions" ? withCam(`/api/sessions/page?page=${page}&size=${PAGE_SIZE}${allVersions ? "&all=1" : ""}`, cam) : null, 3000, tokenKey);
  const pages = sessions.data?.pages ?? 1;
  useEffect(() => { if (page > pages) setPage(pages); }, [page, pages]); // the table shrank (slot map reset)
  // the event log: entries and exits (sessions) and the notices (encroached bays, objects the filter ignored), told apart by a filter
  const [kinds, setKinds] = useState<"all" | "moves" | "notices">("all");
  const events = usePoll<ParkingEvent[]>(tab === "events" ? withCam(`/api/events?limit=200&kinds=${kinds}`, cam) : null, 3000, tokenKey);
  const camEvents = usePoll<CameraEvent[]>(tab === "camera" ? withCam("/api/camera-events?limit=200", cam) : null, 15000, tokenKey);
  const sync = usePoll<SyncStatus>("/api/history/sync", 10000, tokenKey);
  const mine = sync.data?.cameras.find((c) => c.id === cam) ?? sync.data?.cameras[0];
  const current = tab === "sessions" ? sessions : tab === "events" ? events : camEvents;
  const drift = mine?.live && mine.live.mismatch_s >= MISMATCH_WARN_S ? mine.live : null;
  return (
    <Card
      title="Parking history"
      actions={
        <>
          <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${mine?.source === "postgres" ? "bg-primary/10 text-primary" : "bg-warning/15 text-ink"}`}
                data-testid="history-source"
                title={mine ? `PostgreSQL rows: ${mine.rows ? `${mine.rows.sessions} sessions, ${mine.rows.events} events, ${mine.rows.samples} occupancy samples` : "none yet"}${mine.behind.events || mine.behind.sessions ? ` · copy behind by ${mine.behind.events} events` : ""}${mine.error ? ` · ${mine.error}` : ""}` : undefined}>
            {sourceLabel(mine?.source, mine?.age_s)}
          </span>
          <Pill active={tab === "sessions"} onClick={() => setTab("sessions")}>Sessions (in / out)</Pill>
          {tab === "sessions" && (
            <>
              <Pill active={!allVersions} onClick={() => setAllVersions(false)} title="Only the slot map / auto grid version in use now">This grid</Pill>
              <Pill active={allVersions} onClick={() => setAllVersions(true)} title="Every detector and auto grid version of this camera: bay numbers of different versions are not the same bays">All versions</Pill>
            </>
          )}
          <Pill active={tab === "events"} onClick={() => setTab("events")}>Event log</Pill>
          {tab === "events" && (
            <span className="inline-flex items-center gap-1" data-testid="event-kinds">
              <Pill active={kinds === "all"} onClick={() => setKinds("all")}>All</Pill>
              <Pill active={kinds === "moves"} onClick={() => setKinds("moves")} title="Vehicles in and out (they open and close sessions)">Entries &amp; exits</Pill>
              <Pill active={kinds === "notices"} onClick={() => setKinds("notices")} title="Free bays a neighbour's vehicle stands on, and objects the filter ignored">Spillage &amp; obstacles</Pill>
            </span>
          )}
          <Pill active={tab === "camera"} onClick={() => setTab("camera")} title="Worker online / offline, monitoring on / off">Camera status</Pill>
          {tab !== "camera" && (
            <a href={withToken(withCam(tab === "sessions" ? "/api/sessions.csv" : `/api/events.csv?kinds=${kinds}`, cam))}
               className="inline-flex items-center gap-1 rounded-md border border-line px-2.5 py-1 text-xs font-medium text-ink hover:border-primary hover:text-primary">
              <Download size={12} /> CSV
            </a>
          )}
          <Button variant="ghost" onClick={current.reload} title="Refresh"><RefreshCw size={14} /></Button>
        </>
      }
      bodyClass="p-0"
    >
      {current.error && <p className="p-4 text-sm text-danger">{current.error}</p>}
      {drift && (
        <p className="border-b border-line bg-warning/15 px-4 py-2 text-sm text-ink" role="alert" data-testid="history-mismatch">
          The log and the live count differ: {drift.open} open session{drift.open === 1 ? "" : "s"} against {drift.expected} vehicle{drift.expected === 1 ? "" : "s"} counted,
          for {Math.round(drift.mismatch_s)} s. The worker repairs this every frame; if it stays, the history file may be locked or read-only.
          {drift.error ? <> Last error: {drift.error}</> : null}
        </p>
      )}
      <div className="overflow-x-auto">
        {tab === "sessions" ? (
          <table className="w-full text-sm">
            <thead className="bg-body text-left text-xs tracking-wide text-muted uppercase">
              <tr>{["#", ...(allVersions ? ["Scope"] : []), "Slot", "In", "Out", "Duration", "Status", "Note"].map((h) => <th key={h} className="px-4 py-2 font-semibold">{h}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-line">
              {(sessions.data?.items ?? []).map((s) => (
                <tr key={s.id} className="hover:bg-body/60">
                  <td className="px-4 py-2 text-muted">{s.id}</td>
                  {allVersions && <td className="px-4 py-2 text-xs text-muted" data-testid="session-scope">{scopeLabel(s.scope)}</td>}
                  <td className="px-4 py-2 font-medium">{s.slot_id}</td>
                  <td className="px-4 py-2 tabular-nums">{clock(s.in_time)}</td>
                  <td className="px-4 py-2 tabular-nums">{s.out_time ? clock(s.out_time) : <span className="text-success">parked</span>}</td>
                  <td className="px-4 py-2">{duration(s.duration_min)}</td>
                  <td className="px-4 py-2">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${s.out_time ? "bg-body text-muted" : "bg-danger/10 text-danger"}`}>{s.status}</span>
                  </td>
                  <td className="max-w-xs truncate px-4 py-2 text-muted" title={s.note}>{s.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : tab === "camera" ? (
          <table className="w-full text-sm" data-testid="camera-events">
            <thead className="bg-body text-left text-xs tracking-wide text-muted uppercase">
              <tr>{["Time", "Change", "Detail"].map((h) => <th key={h} className="px-4 py-2 font-semibold">{h}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-line">
              {(camEvents.data ?? []).map((e) => (
                <tr key={e.id} className="hover:bg-body/60">
                  <td className="px-4 py-2 tabular-nums">{clock(e.time)}</td>
                  <td className={`px-4 py-2 font-medium ${e.kind === "worker_offline" ? "text-danger" : e.kind === "worker_online" ? "text-success" : ""}`}>
                    {CAMERA_EVENT_LABELS[e.kind] ?? e.kind}
                  </td>
                  <td className="px-4 py-2 text-muted">{e.detail}</td>
                </tr>
              ))}
              {camEvents.data && !camEvents.data.length && (
                <tr><td colSpan={3} className="px-4 py-6 text-center text-sm text-muted">
                  {sync.data?.backend === "postgres" ? "No state changes recorded yet." : "Camera status changes are recorded in PostgreSQL by the sync service."}
                </td></tr>
              )}
            </tbody>
          </table>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-body text-left text-xs tracking-wide text-muted uppercase">
              <tr>{["Time", "Slot", "Event", "Session", "Note"].map((h) => <th key={h} className="px-4 py-2 font-semibold">{h}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-line">
              {events.data?.length === 0 && (
                <tr data-testid="event-empty"><td colSpan={5} className="px-4 py-6 text-center text-muted">No events logged yet.</td></tr>
              )}
              {(events.data ?? []).map((e) => (
                <tr key={e.id} className={`hover:bg-body/60 ${eventInfo(e).group === "notice" ? "bg-body/40" : ""}`} data-testid="event-row" data-kind={e.event}>
                  <td className="px-4 py-2 tabular-nums">{clock(e.time)}</td>
                  <td className="px-4 py-2 font-medium">{e.slot_id ? e.slot_id : <span className="text-muted" title={eventSlot(e)}>–</span>}</td>
                  <td className="px-4 py-2">{eventInfo(e).label}</td>
                  <td className="px-4 py-2 text-muted">{e.session_id ?? ""}</td>
                  <td className="max-w-md truncate px-4 py-2 text-muted" title={e.note}>{e.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {tab === "sessions" && sessions.data && (
        <Pagination page={sessions.data.page} pages={sessions.data.pages} total={sessions.data.total}
                    size={sessions.data.size} open={sessions.data.open} onPage={setPage} />
      )}
    </Card>
  );
}
