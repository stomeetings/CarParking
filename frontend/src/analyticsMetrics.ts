// Phase 2 metric analytics: the pure rules behind the "Measured occupancy" section of the Analytics page (unit-tested in
// analyticsMetrics.test.ts). The data is GET /api/cameras/{id}/geometry/history (bucketed in the database); the page itself is
// components/MetricAnalytics.tsx.

import type { HistoryPoint } from "./setupflow";

const share = (occupied: number, capacity: number) => (capacity > 0 ? Math.min(1, Math.max(0, occupied / capacity)) : null);

export interface MetricRow {
  t: number;
  occupied: number; // average occupied bays in the bucket
  peak: number; // highest occupied count in the bucket
  capacity: number; // bays available
  util: number | null; // average utilization, percent
  peakUtil: number | null;
  vehicles: number;
  samples: number;
  ver: number;
}

export function metricRows(points: HistoryPoint[]): MetricRow[] {
  return points.map((p) => {
    const u = share(p.occupied_avg, p.capacity_avg);
    const pk = share(p.occupied_max, p.capacity_max);
    return {
      t: p.t, occupied: Math.round(p.occupied_avg * 10) / 10, peak: p.occupied_max, capacity: Math.round(p.capacity_avg * 10) / 10,
      util: u == null ? null : Math.round(u * 1000) / 10, peakUtil: pk == null ? null : Math.round(pk * 1000) / 10,
      vehicles: Math.round(p.vehicles_avg * 10) / 10, samples: p.samples, ver: p.ver,
    };
  });
}

export interface MetricSummary {
  samples: number;
  avgUtil: number | null; // percent, weighted by the samples in each bucket
  peakUtil: number | null; // percent, the highest bucket peak
  peakOccupied: number | null;
  capacity: number | null; // the newest bucket's bay count
  avgOccupied: number | null;
}

export function summarize(points: HistoryPoint[]): MetricSummary {
  const n = points.reduce((a, p) => a + p.samples, 0);
  if (!points.length || n === 0) return { samples: 0, avgUtil: null, peakUtil: null, peakOccupied: null, capacity: null, avgOccupied: null };
  const occ = points.reduce((a, p) => a + p.occupied_avg * p.samples, 0) / n;
  const cap = points.reduce((a, p) => a + p.capacity_avg * p.samples, 0) / n;
  const peaks = points.map((p) => share(p.occupied_max, p.capacity_max)).filter((x): x is number => x != null);
  const u = share(occ, cap);
  return {
    samples: n, avgUtil: u == null ? null : Math.round(u * 1000) / 10, peakUtil: peaks.length ? Math.round(Math.max(...peaks) * 1000) / 10 : null,
    peakOccupied: Math.max(...points.map((p) => p.occupied_max)), capacity: Math.round(points[points.length - 1].capacity_avg),
    avgOccupied: Math.round(occ * 10) / 10,
  };
}

/** Average utilization (percent) by hour of the local day, 24 entries; null where there is no data. Weighted by samples. */
export function hourProfile(points: HistoryPoint[]): { hour: number; util: number | null }[] {
  const occ = Array(24).fill(0);
  const cap = Array(24).fill(0);
  for (const p of points) {
    const h = new Date(p.t * 1000).getHours();
    occ[h] += p.occupied_avg * p.samples;
    cap[h] += p.capacity_avg * p.samples;
  }
  return occ.map((o, hour) => {
    const u = share(o, cap[hour]);
    return { hour, util: u == null ? null : Math.round(u * 1000) / 10 };
  });
}

/** The label of a bucket on the x axis: a time for a day or less, a date and hour beyond. */
export function metricLabel(t: number, spanHours: number): string {
  const d = new Date(t * 1000);
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  return spanHours <= 24 ? hm : `${d.getDate()}/${d.getMonth() + 1} ${hm}`;
}
