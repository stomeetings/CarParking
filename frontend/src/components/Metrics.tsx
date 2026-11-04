import { Ruler } from "lucide-react";
import { useState } from "react";
import { CONFIDENCE_TONE, driftNote } from "../setupflow";
import type { GeometryLive } from "../types";
import { Card } from "./ui";

// Phase 2 on the dashboard: what the worker computes from the camera's ACTIVE calibration (carparking/geometry_live.py) and
// publishes as snapshot.geometry. It exists only while CARPARK_GEOMETRY_ENABLED is on and the camera has an accepted
// calibration (the API strips it otherwise); without it none of this renders and the dashboard is exactly as before.

/** The Metrics layer's on / off, remembered on its own (the Slots / Boxes / Area layers keep theirs untouched). */
export const METRICS_KEY = "carparking.metricsLayer";
export const savedMetrics = (): boolean => {
  try {
    return localStorage.getItem(METRICS_KEY) !== "0";
  } catch {
    return true;
  }
};
export function useMetricsLayer(): [boolean, () => void] {
  const [on, setOn] = useState(savedMetrics);
  const toggle = () => setOn((v) => {
    try {
      localStorage.setItem(METRICS_KEY, v ? "0" : "1");
    } catch {
      /* not remembered in a private window: kept for this view only */
    }
    return !v;
  });
  return [on, toggle];
}

/** "1.4 m", with a ≈ while the scale was not anchored by a typed distance (about ±25 %). */
export const metres = (m: number, confidence: string | null | undefined) => `${confidence === "HIGH" ? "" : "≈ "}${m.toFixed(m < 10 ? 1 : 0)} m`;

const BAY = { FREE: "#2eb85c", BUSY: "#e55353" } as const;

/** The metric layer in the camera's own pixels (inside the same SVG as the slots): the bays packed on the ground (outlines,
 * green FREE / red BUSY, dashed where a car stood when the calibration was made), each measured vehicle's footprint on the
 * ground, and the distances between neighbours (and to people / objects) with their lines. */
export function MetricOverlay({ geometry, w, h }: { geometry: GeometryLive; w: number; h: number }) {
  if (!geometry.valid) return null;
  const pts = (poly: [number, number][]) => poly.map(([x, y]) => `${x * w},${y * h}`).join(" ");
  const conf = geometry.confidence;
  return (
    <g data-testid="metric-overlay" pointerEvents="none">
      {geometry.bays.map((b) => (
        <polygon key={b.id} points={pts(b.poly)} fill={b.state === "BUSY" ? "rgba(229,83,83,0.10)" : "rgba(46,184,92,0.10)"} stroke={BAY[b.state]}
                 strokeWidth={1.5} vectorEffect="non-scaling-stroke" strokeDasharray={b.car ? "6 4" : undefined}
                 data-testid="metric-bay" data-state={b.state} />
      ))}
      {geometry.vehicles.map((v) => (
        <polygon key={v.i} points={pts(v.poly)} fill="rgba(57,153,255,0.28)" stroke="#3999ff" strokeWidth={1.5} vectorEffect="non-scaling-stroke"
                 data-testid="metric-footprint" />
      ))}
      {geometry.distances.map((d, i) => {
        const mx = ((d.p[0] + d.q[0]) / 2) * w;
        const my = ((d.p[1] + d.q[1]) / 2) * h;
        const label = metres(d.m, conf);
        return (
          <g key={i} data-testid="metric-distance" data-kind={d.kind}>
            <title>{d.kind === "object" ? `${d.label ?? "object"} to the nearest vehicle: ${label}` : `Between two vehicles: ${label}`}</title>
            <line x1={d.p[0] * w} y1={d.p[1] * h} x2={d.q[0] * w} y2={d.q[1] * h} stroke={d.kind === "object" ? "#d63384" : "#ffffff"}
                  strokeWidth={2} vectorEffect="non-scaling-stroke" strokeDasharray="5 3" />
            <rect x={mx - label.length * w / 280} y={my - w / 130} width={label.length * w / 140} height={w / 65} rx={w / 500} fill="rgba(20,24,40,0.85)" />
            <text x={mx} y={my + w / 300} fontSize={w / 100} fontWeight={700} fill="white" textAnchor="middle">{label}</text>
          </g>
        );
      })}
    </g>
  );
}

/** Capacity from the calibrated geometry: bays that fit on the measured ground, how many a car stands in, how many are free.
 * Beside (not instead of) the slot counters, which keep counting exactly as before. */
export function MetricCard({ geometry }: { geometry: GeometryLive | undefined }) {
  if (!geometry) return null;
  const tone = CONFIDENCE_TONE[geometry.confidence ?? "NONE"] ?? CONFIDENCE_TONE.NONE;
  const c = geometry.counts;
  return (
    <Card title={<span className="inline-flex items-center gap-2"><Ruler size={15} /> Measured capacity</span>}
          actions={<span className={`rounded-full px-2 py-0.5 text-[11px] font-bold ring-1 ${tone}`} data-testid="metric-confidence">{geometry.confidence ?? "—"}</span>}>
      {!geometry.valid || !c ? (
        <p className="text-sm text-muted" data-testid="metric-invalid">{geometry.reason ?? "The calibration does not fit this picture."}</p>
      ) : (
        <div data-testid="metric-card">
          <div className="grid grid-cols-3 gap-2 text-center">
            {[["Bays", c.capacity, "text-ink"], ["Occupied", c.occupied, "text-danger"], ["Free", c.free, "text-success"]].map(([label, n, color]) => (
              <div key={label as string} className="rounded-md bg-body py-2">
                <p className={`text-2xl font-bold tabular-nums ${color}`} data-testid={`metric-${(label as string).toLowerCase()}`}>{n}</p>
                <p className="text-[11px] tracking-wide text-muted uppercase">{label}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-muted" data-testid="metric-notes">
            {geometry.area_m2.toFixed(0)} m² measured · {c.measured} of {c.vehicles} vehicles measured
            {c.beyond_range > 0 && ` (${c.beyond_range} too far to place)`} · calibration v{geometry.version}
            {geometry.confidence !== "HIGH" && ` · metres about ±${Math.round((geometry.uncertainty ?? 0.25) * 100)} %`}
          </p>
          {(() => {
            const d = driftNote(geometry.drift, geometry.drift_guard, Date.now() / 1000);
            return <p className={`mt-1 text-xs ${{ ok: "text-success", warn: "text-warning", muted: "text-muted" }[d.tone]}`} data-testid="metric-drift">{d.text}</p>;
          })()}
        </div>
      )}
    </Card>
  );
}
