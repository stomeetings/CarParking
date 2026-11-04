import { useMemo } from "react";
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { hourProfile, metricLabel, metricRows, summarize } from "../analyticsMetrics";
import { usePoll } from "../hooks";
import type { HistoryResponse } from "../setupflow";
import type { GeometryLive } from "../types";
import { Card } from "./ui";

const tile = (label: string, value: string, hint?: string) => (
  <div key={label} className="rounded-lg border border-line bg-white p-3 shadow-sm">
    <div className="text-xs font-semibold tracking-wide text-muted uppercase">{label}</div>
    <div className="mt-1 text-xl font-semibold tabular-nums">{value}</div>
    {hint && <div className="text-xs text-muted">{hint}</div>}
  </div>
);
const pct = (v: number | null) => (v == null ? "–" : `${v.toFixed(0)} %`);

/** Phase 2 (CARPARK_GEOMETRY_ENABLED): the camera's MEASURED occupancy next to the legacy slot analytics: how full the metric bays were
 * (average and peak), the bays available over time, the average fullness by hour of day, and the metric counts of the frame now.
 * Strictly additive: when the feature is off on the server (503), the camera is not calibrated or there are no samples in the range,
 * it renders nothing, so the legacy page is exactly as before. */
export function MetricAnalytics({ tokenKey, cam, days, geometry }: { tokenKey: string; cam: string | null; days: number; geometry: GeometryLive | undefined }) {
  const hours = days * 24;
  const path = cam ? `/api/cameras/${encodeURIComponent(cam)}/geometry/history?hours=${hours}&points=120` : null;
  const h = usePoll<HistoryResponse>(path, 60000, tokenKey); // an error (503: feature off, 404...) just means: no metric section
  const pts = h.data?.points;
  const rows = useMemo(() => metricRows(pts ?? []).map((r) => ({ ...r, label: metricLabel(r.t, hours) })), [pts, hours]);
  const sum = useMemo(() => summarize(pts ?? []), [pts]);
  const profile = useMemo(() => hourProfile(pts ?? []).map((p) => ({ ...p, label: `${p.hour}:00` })), [pts]);
  if (!h.data || h.error || sum.samples === 0) return null;
  const g = geometry && geometry.valid ? geometry : undefined;
  const c = g?.counts;

  return (
    <section className="space-y-3" data-testid="metric-analytics" aria-label="Measured occupancy">
      <h2 className="flex flex-wrap items-baseline gap-2 text-sm font-semibold text-ink">
        Measured occupancy
        <span className="text-xs font-normal text-muted">from the calibrated metric bays · {sum.samples} samples · calibration{h.data.versions.length > 1 ? "s" : ""} v{h.data.versions.join(", v")}</span>
      </h2>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6" data-testid="metric-kpis">
        {tile("Average utilization", pct(sum.avgUtil), sum.avgOccupied != null ? `${sum.avgOccupied} bays occupied on average` : undefined)}
        {tile("Peak utilization", pct(sum.peakUtil), sum.peakOccupied != null ? `${sum.peakOccupied} bays at the busiest` : undefined)}
        {tile("Bays", sum.capacity != null ? String(sum.capacity) : "–", "geometric capacity")}
        {g ? tile("Surface", `${Math.round(g.area_m2)} m²`, g.confidence ? `${g.confidence}${g.uncertainty != null ? ` ±${Math.round(g.uncertainty * 100)} %` : ""}` : undefined) : null}
        {c ? tile("Occupied now", `${c.occupied} / ${c.capacity}`, `${c.free} free`) : null}
        {c ? tile("Vehicles now", String(c.vehicles), `${c.measured} measured${c.beyond_range ? ` · ${c.beyond_range} beyond range` : ""}`) : null}
      </div>
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Bay utilization over time" className="xl:col-span-2" actions={<span className="text-xs text-muted">bucket {Math.round((h.data.bucket_s ?? 60) / 60)} min</span>}>
          <div className="h-64" data-testid="metric-trend">
            <ResponsiveContainer>
              <ComposedChart data={rows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbdfe6" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="occupied" name="Occupied bays (avg)" stroke="#e55353" fill="#e55353" fillOpacity={0.3} />
                <Line type="monotone" dataKey="peak" name="Peak" stroke="#e55353" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="capacity" name="Bays" stroke="#9aa0ac" strokeDasharray="5 4" dot={false} strokeWidth={1.5} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Average fullness by hour of day" actions={<span className="text-xs text-muted">% of bays</span>}>
          <div className="h-64" data-testid="metric-hours">
            <ResponsiveContainer>
              <BarChart data={profile}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbdfe6" />
                <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="util" name="Utilization %" fill="#5856d6" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </section>
  );
}
