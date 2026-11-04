// Suggest slots (the draft layer of the slot editor): the shapes the API returns and the small rules the card draws by.
// Pure functions, unit-tested in suggest.test.ts. Corners are shares of the frame (0..1), like every slot map.

import type { Pt } from "./slotdraw";

export interface Bay {
  id: number;
  name: string;
  poly: Pt[];
  lift?: number;
  /** measured / extrapolated / manual (a pin) / edited (reshaped in the draft) */
  source?: string;
  capacity?: number;
  /** a pin: the id of the nearest bay of the grid */
  binds_to?: number | null;
  /** a Fixed camera's running slot carried into the draft exactly as it runs: it cannot be moved, reshaped or removed */
  protected?: boolean;
}

export interface Diff {
  added: number[];
  removed: number[];
  moved: number[];
  unchanged: number[];
}

export interface SuggestMeta {
  parent: number;
  mode: "quick" | "full";
  kind: "auto" | "fixed";
  lined_up: "vehicles" | "calibration";
  expected: number | null;
  vehicles?: number;
  edited?: boolean;
}

export interface SuggestView {
  version: number;
  state: string;
  created_at: number;
  note: string;
  meta: SuggestMeta;
  grid: Bay[];
  width: number;
  height: number;
  calibration: { confidence: string | null; h: number | null; version: number | null };
  baseline: { kind: "auto" | "fixed"; slots: Bay[] };
  diff: Diff | null;
  pins: Bay[];
  /** the slots that run now changed since the suggestion was made: Accept would be refused */
  stale: boolean;
}

export interface SuggestListItem {
  version: number;
  created_at: number;
  note: string;
  meta: SuggestMeta;
  bays: number;
}

export interface Job {
  id: string;
  kind: string;
  state: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  result: { version: number; bays: number; mode: string; lined_up: string; warnings: string[]; diff: Diff } | null;
  error: string | null;
}

export type BayStatus = "added" | "moved" | "unchanged" | "edited";

/** How a draft bay compares with what runs now. A bay the operator reshaped is "edited" whatever the diff says. */
export function statusOf(bay: Bay, diff: Diff | null): BayStatus {
  if (bay.source === "edited") return "edited";
  if (!diff) return "unchanged";
  if (diff.added.includes(bay.id)) return "added";
  if (diff.moved.includes(bay.id)) return "moved";
  return "unchanged";
}

/** Outline colour per status (the legend of the diff layer). */
export const STATUS_COLOR: Record<BayStatus | "removed", string> = {
  added: "#1f9d55",
  moved: "#d98200",
  unchanged: "#0ea5b7",
  edited: "#4f46e5",
  removed: "#d64545",
};

export function diffText(d: Diff | null): string {
  if (!d) return "no comparison (nothing runs to compare with)";
  const parts = [`${d.added.length} added`, `${d.moved.length} moved`, `${d.removed.length} removed`, `${d.unchanged.length} unchanged`];
  return parts.join(" · ");
}

/** A pin in one line: "P1 · slot 9 · beside B7". */
export function pinText(p: Bay): string {
  return `${p.name} · slot ${p.id}${p.binds_to != null ? ` · next to B${p.binds_to}` : ""}${p.capacity && p.capacity > 1 ? ` · holds ${p.capacity}` : ""}`;
}

/** What Accept will do, for the confirmation: an auto-grid camera switches the live grid (and the history scope), a Fixed camera
 * merges the chosen bays into the drawn map and leaves the rest as drawn. */
export function acceptText(v: SuggestView, taken: number): string {
  if (v.meta.kind === "fixed") {
    return `Merge ${taken} suggested bay(s) into the hand-drawn map. A bay with a drawn slot's id replaces that slot's shape (its name, capacity and history stay); the others are added. Slots you do not take stay as drawn.`
      + (v.grid.some((b) => b.protected) ? " The running slots in the draft are protected: they stay exactly as they run." : "");
  }
  const d = v.diff;
  return `Switch the live grid to this draft: ${d ? `${d.added.length} added, ${d.moved.length} moved, ${d.removed.length} removed` : "all bays"}. `
    + (v.grid.some((b) => b.protected) ? "The running bays in the draft are protected: they keep their ids and shapes exactly. " : "")
    + "Monitoring carries on without a restart; the history scope moves to the new grid version, so parked vehicles show one exit and entry at the switch. "
    + `${v.pins.length ? `Your ${v.pins.length} pin(s) keep their slot ids and capacity.` : ""}`.trim();
}

/** Why Accept is not possible right now, or null. */
export function acceptBlock(v: SuggestView, taken: number): string | null {
  if (v.state !== "draft") return `version ${v.version} is ${v.state}`;
  if (v.stale) return "the slots that run now changed since this suggestion was made: ask for a new one";
  if (v.meta.kind === "fixed" && taken === 0) return "tick at least one suggested bay to take";
  return null;
}

/** The draft as the PUT body wants it (ids, names, shapes; the server keeps lift and decides the source). */
export function gridBody(bays: Bay[]): { grid: Pick<Bay, "id" | "name" | "poly" | "lift">[] } {
  return { grid: bays.map(({ id, name, poly, lift }) => ({ id, name, poly, ...(lift != null ? { lift } : {}) })) };
}

export function sameGrid(a: Bay[], b: Bay[]): boolean {
  return JSON.stringify(gridBody(a)) === JSON.stringify(gridBody(b));
}

/** The draft's line about the running slots it keeps: null when none are protected (an auto-grid camera's draft replaces its grid). */
export function protectedText(bays: Bay[]): string | null {
  const keep = bays.filter((b) => b.protected);
  if (!keep.length) return null;
  const added = bays.length - keep.length;
  return `${keep.length} running slot(s) are protected (${keep.map((b) => b.name).join(", ")}): the draft keeps them exactly as they run, never marks them removed or moved, `
    + `and adds ${added} new bay(s) that do not lie on them.`;
}
