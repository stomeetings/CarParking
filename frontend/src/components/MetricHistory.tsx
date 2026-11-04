import { History, TriangleAlert } from "lucide-react";
import { type GeometryEvent, type HistoryResponse, chartLines, eventText } from "../setupflow";
import { Card, Pill } from "./ui";

const W = 600;
const H = 140;
const RANGES: { hours: number; label: string }[] = [{ hours: 6, label: "6 h" }, { hours: 24, label: "24 h" }, { hours: 168, label: "7 days" }];
const time = (t: number) => new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

/** The camera's measured occupancy over time (GET /api/cameras/{id}/geometry/history, bucketed in the database): the average
 * number of occupied bays as an area, its peak as a line, and the bays available as a dashed line. */
export function HistoryCard({ history, hours, onHours }: { history: HistoryResponse | null; hours: number; onHours: (h: number) => void }) {
  const ranges = RANGES.map((r) => (
    <Pill key={r.hours} active={hours === r.hours} onClick={() => onHours(r.hours)}>{r.label}</Pill>
  ));
  if (!history) return null;
  const pts = history.points;
  const c = chartLines(pts, history.since, history.until, W, H);
  return (
    <Card title={<span className="inline-flex items-center gap-2"><History size={15} /> Measured occupancy</span>} actions={ranges}>
      {pts.length === 0 ? (
        <p className="text-sm text-muted" data-testid="history-empty">No samples yet. They are taken once a minute (and when the occupancy changes) while a calibration is active and the worker runs.</p>
      ) : (
        <div data-testid="history-chart">
          <svg viewBox={`0 0 ${W} ${H}`} className="h-36 w-full" preserveAspectRatio="none" role="img" aria-label="Occupied bays over time">
            <line x1={0} y1={H} x2={W} y2={H} stroke="#d8dbe0" />
            <polyline points={c.capacity} fill="none" stroke="#9aa0ac" strokeWidth={1.5} strokeDasharray="5 4" vectorEffect="non-scaling-stroke" data-testid="history-capacity" />
            <polygon points={c.area} fill="rgba(229,83,83,0.22)" data-testid="history-area" />
            <polyline points={c.occupiedMax} fill="none" stroke="#e55353" strokeWidth={1.5} vectorEffect="non-scaling-stroke" data-testid="history-peak" />
            {pts.map((p) => {
              const cx = ((p.t - history.since) / Math.max(1, history.until - history.since)) * W;
              const cy = H - (p.occupied_avg / c.top) * H;
              return (
                <g key={p.t}>
                  {pts.length <= 30 && <circle cx={cx} cy={cy} r={3.5} fill="#e55353" data-testid="history-dot" />}
                  <circle cx={cx} cy={cy} r={8} fill="transparent">
                    <title>{`${time(p.t)}: ${p.occupied_avg.toFixed(1)} of ${p.capacity_max} bays occupied (peak ${p.occupied_max}, ${p.samples} samples)`}</title>
                  </circle>
                </g>
              );
            })}
          </svg>
          <p className="mt-1 flex flex-wrap justify-between gap-2 text-xs text-muted">
            <span>{time(history.since)} → {time(history.until)}</span>
            <span data-testid="history-legend">
              <span className="text-danger">▬</span> occupied (peak) · <span className="text-danger/60">▇</span> average · <span>┄</span> bays ({c.top} max)
              {history.versions.length > 1 && ` · calibrations v${history.versions.join(", v")}`}
            </span>
          </p>
        </div>
      )}
    </Card>
  );
}

const DOT: Record<string, string> = {
  stale: "bg-danger", drift_suspected: "bg-warning", drift_cleared: "bg-success", activated: "bg-success", draft: "bg-primary", reanchored: "bg-primary", discarded: "bg-muted",
};

/** The calibration's story, newest first: drafts, accepts, rollbacks, the drift guard's suspicion and its verdict. */
export function EventsCard({ events }: { events: GeometryEvent[] | null }) {
  if (!events) return null;
  return (
    <Card title="Calibration events" bodyClass="p-0">
      {events.length === 0 ? (
        <p className="p-4 text-sm text-muted" data-testid="events-empty">Nothing yet.</p>
      ) : (
        <ul className="max-h-72 divide-y divide-line overflow-y-auto" data-testid="events-list">
          {events.map((e) => (
            <li key={e.id} className="flex items-start gap-3 px-4 py-2 text-sm" data-testid={`event-${e.kind}`}>
              <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${DOT[e.kind] ?? "bg-muted"}`} />
              <span className="min-w-0 flex-1">{eventText(e)}</span>
              <span className="shrink-0 text-xs text-muted">{time(e.t)}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/** The camera moved and nothing has been accepted since: say so where the Admin works. */
export function StaleBanner({ event, onReanchor }: { event: GeometryEvent | null; onReanchor?: () => void }) {
  if (!event) return null;
  return (
    <p className="flex items-start gap-2 rounded-md bg-warning/15 px-4 py-3 text-sm text-ink" data-testid="stale-banner">
      <TriangleAlert size={16} className="mt-0.5 shrink-0" />
      <span><strong>The camera moved.</strong> {eventText(event)} at {time(event.t)}. Metrics are off until you calibrate again (Still → Known distance → Surface → Calibrate → Accept), or re-anchor the existing calibration to the camera's new position; the dashboard works as before meanwhile.
        {onReanchor && <> <button type="button" onClick={onReanchor} data-testid="stale-reanchor" className="ml-1 rounded-md bg-primary px-2.5 py-1 text-xs font-semibold text-white">Re-anchor calibration</button></>}</span>
    </p>
  );
}
