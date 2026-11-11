import { useMemo, useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { withCam } from "../api";
import { usePoll } from "../hooks";
import { bucketLabel, duration, sourceLabel } from "../lib";
import type { Analytics, Overview, Snapshot } from "../types";
import { MetricAnalytics } from "./MetricAnalytics";
import { Card, MetricCard, Pill, StatusBadge } from "./ui";

const RANGES: [number, string][] = [[1, "24 h"], [7, "7 days"], [30, "30 days"], [90, "90 days"]];

// Its own chunk (React.lazy in App): recharts is most of the bundle and only this page needs it.
// The numbers are aggregated by the server (PostgreSQL: the central history; SQLite when that copy isn't there).
export default function AnalyticsPage({ tokenKey, snapshot, cam }: { tokenKey: string; snapshot: Snapshot | undefined; cam: string | null }) {
  const [days, setDays] = useState(7);
  const [scope, setScope] = useState<"all" | "current">("all");
  const a = usePoll<Analytics>(withCam(`/api/analytics?days=${days}&scope=${scope}`, cam), 30000, tokenKey);
  const ov = usePoll<Overview>(`/api/analytics/overview?days=${days}`, 30000, tokenKey);
  const d = a.data;
  const occupancy = useMemo(() => (d?.occupancy ?? []).map((o) => ({ ...o, label: bucketLabel(o.t, d!.bucket) })), [d]);
  const perDay = useMemo(() => (d?.per_day ?? []).map((x) => ({ ...x, label: bucketLabel(`${x.day}T00:00`, "1 day") })), [d]);
  const c = snapshot?.counts;
  const now = c ? [
    { name: "Busy", value: c.busy, color: "#e55353" },
    { name: "Free", value: c.free, color: "#2eb85c" },
    { name: "Unknown", value: c.unknown, color: "#f9b115" },
  ] : [];
  const t = d?.totals;
  const kpis: [string, string, string?][] = t ? [
    ["Arrivals", String(t.entries)], ["Departures", String(t.exits)],
    ["Sessions", String(t.sessions), `${t.open} still parked`],
    ["Average stay", t.avg_min != null ? duration(t.avg_min) : "–", t.median_min != null ? `median ${duration(t.median_min)}` : undefined],
    ["Busiest hour", t.busiest_hour != null ? `${t.busiest_hour}:00` : "–", "most arrivals"],
    ["Peak occupancy", t.peak_busy != null ? `${t.peak_busy} busy` : "–", t.peak_busy == null ? "from PostgreSQL samples" : undefined],
  ] : [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2" data-testid="analytics-bar">
        {RANGES.map(([v, label]) => <Pill key={v} active={days === v} onClick={() => setDays(v)}>{label}</Pill>)}
        <span className="mx-1 h-5 w-px bg-line" />
        <Pill active={scope === "all"} onClick={() => setScope("all")} title="Everything this camera logged, any detector or slot map">All history</Pill>
        <Pill active={scope === "current"} onClick={() => setScope("current")} title="Only the detector / slot map running now">Current slot map</Pill>
        <StatusBadge status={d?.source === "postgres" ? "active" : "warn"} className="ml-auto" data-testid="analytics-source">
          {sourceLabel(d?.source)}
        </StatusBadge>
      </div>
      {a.error && <p className="rounded-md bg-danger/10 px-3 py-2 text-sm text-danger">Couldn't load the analytics: {a.error}</p>}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6" data-testid="kpis">
        {kpis.map(([label, value, hint]) => (
          <MetricCard key={label} label={label} value={value} hint={hint} />
        ))}
      </div>

      <MetricAnalytics tokenKey={tokenKey} cam={cam} days={days} geometry={snapshot?.geometry} />

      {(ov.data?.cameras.length ?? 0) > 1 && (
        <Card title="All cameras" bodyClass="p-0" actions={<span className="text-xs text-muted">last {RANGES.find(([v]) => v === days)?.[1]}</span>}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm" data-testid="overview">
              <thead className="bg-body text-left text-xs tracking-wide text-muted uppercase">
                <tr>{["Camera", "State", "Free", "Busy", "Total", "Arrivals", "Departures", "Parked now"].map((h) => <th key={h} className="px-4 py-2 font-semibold">{h}</th>)}</tr>
              </thead>
              <tbody className="divide-y divide-line">
                {ov.data!.cameras.map((x) => (
                  <tr key={x.id} className={x.id === cam ? "bg-primary/5" : ""}>
                    <td className="px-4 py-2 font-medium">{x.display}</td>
                    <td className="px-4 py-2">
                      <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${!x.online ? "bg-danger/10 text-danger" : x.monitoring ? "bg-success/10 text-success" : "bg-warning/15 text-ink"}`}>
                        {!x.online ? "Worker offline" : x.monitoring ? "Tracking" : "Monitoring off"}
                      </span>
                    </td>
                    <td className="px-4 py-2 tabular-nums text-success">{x.monitoring ? x.free ?? "–" : "–"}</td>
                    <td className="px-4 py-2 tabular-nums text-danger">{x.monitoring ? x.busy ?? "–" : "–"}</td>
                    <td className="px-4 py-2 tabular-nums">{x.monitoring ? x.total ?? "–" : "–"}</td>
                    <td className="px-4 py-2 tabular-nums">{x.entries}</td>
                    <td className="px-4 py-2 tabular-nums">{x.exits}</td>
                    <td className="px-4 py-2 tabular-nums">{x.open}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Occupancy over time" className="xl:col-span-2"
              actions={<span className="text-xs text-muted">average per {d?.bucket ?? "…"}</span>}>
          <div className="h-72">
            {occupancy.length ? (
              <ResponsiveContainer>
                <AreaChart data={occupancy}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#dbdfe6" />
                  <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={24} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="busy" name="Busy" stackId="1" stroke="#e55353" fill="#e55353" fillOpacity={0.35} />
                  <Area type="monotone" dataKey="free" name="Free" stackId="1" stroke="#2eb85c" fill="#2eb85c" fillOpacity={0.25} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-muted">
                {d?.source === "sqlite" ? "Occupancy samples are recorded in PostgreSQL by the sync service (it isn't serving right now)."
                  : "No samples in this window yet: the sync service records FREE / BUSY once a minute while monitoring is on."}
              </p>
            )}
          </div>
        </Card>
        <Card title="Occupancy now">
          <div className="h-72">
            {now.length ? (
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={now} dataKey="value" nameKey="name" innerRadius="55%" outerRadius="80%" paddingAngle={2}>
                    {now.map((x) => <Cell key={x.name} fill={x.color} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : <p className="text-sm text-muted">Waiting for the worker’s first frame (it restarts after a settings change)…</p>}
          </div>
        </Card>
        <Card title="Arrivals and departures by hour of day" className="xl:col-span-2">
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={d?.per_hour ?? []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbdfe6" />
                <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="entries" name="Arrivals" fill="#5856d6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="exits" name="Departures" fill="#3399ff" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="How long vehicles stay" actions={<span className="text-xs text-muted">closed sessions</span>}>
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={d?.durations ?? []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbdfe6" />
                <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="sessions" name="Sessions" fill="#2eb85c" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        {days > 1 && (
          <Card title="Arrivals and departures per day" className="xl:col-span-3">
            <div className="h-56">
              <ResponsiveContainer>
                <BarChart data={perDay}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#dbdfe6" />
                  <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={8} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="entries" name="Arrivals" fill="#5856d6" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="exits" name="Departures" fill="#3399ff" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
