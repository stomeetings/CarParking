// Pure helpers (unit-tested in lib.test.ts).
import type { Area, AutoSlots, BaySource, Detection, ParkingEvent, SettingsValues, Slot, SlotFilter } from "./types";

/** The event log's kinds: movements of a vehicle (they open and close sessions) and notices (spillage / non-vehicle filter). */
export type EventGroup = "move" | "notice";
export interface EventInfo {
  group: EventGroup;
  label: string; // the log table
  text: string; // the ticker
  tone: "success" | "info" | "warning" | "muted";
}
export const EVENT_INFO: Record<string, EventInfo> = {
  entry: { group: "move", label: "🚗 IN", text: "vehicle in", tone: "success" },
  exit: { group: "move", label: "🅿️ OUT", text: "vehicle out", tone: "info" },
  encroached: { group: "notice", label: "⚠️ ENCROACHED", text: "bay encroached by a neighbour", tone: "warning" },
  encroach_cleared: { group: "notice", label: "✅ CLEARED", text: "encroachment cleared", tone: "success" },
  rejected: { group: "notice", label: "🚫 IGNORED", text: "object ignored (not a vehicle)", tone: "muted" },
  object_blocked: { group: "notice", label: "🪴 BLOCKED", text: "bay blocked by an object", tone: "warning" },
  object_cleared: { group: "notice", label: "✅ UNBLOCKED", text: "object gone from the bay", tone: "success" },
};
/** How to show an event; a kind this build does not know is a notice with its own name (never a false "vehicle out"). */
export function eventInfo(e: Pick<ParkingEvent, "event">): EventInfo {
  return EVENT_INFO[e.event] ?? { group: "notice", label: e.event, text: e.event, tone: "muted" };
}
export const isMovement = (e: Pick<ParkingEvent, "event">) => eventInfo(e).group === "move";
/** "Slot 3", or "outside the bays" for a notice that concerns no bay (slot 0). */
export const eventSlot = (e: Pick<ParkingEvent, "slot_id">) => (e.slot_id ? `Slot ${e.slot_id}` : "outside the bays");

/** The non-vehicle filter's reasons, as the page words them. */
export const REJECT_TEXT: Record<string, string> = {
  too_small: "too small for the bay", aspect: "not a vehicle shape", pylon: "cone-coloured object", shadow: "flat shadow",
};

/** SVG "points" for a polygon in frame fractions on a w x h canvas. */
export function points(poly: [number, number][], w: number, h: number): string {
  return poly.map(([x, y]) => `${round(x * w)},${round(y * h)}`).join(" ");
}

/** The auto grid in a sentence: "auto grid · 11 bays (9 measured · 1 extrapolated · 1 pinned) from calibration v3", and what is
 * missing against the expected count. */
export function autoGridText(a: AutoSlots): string {
  const parts = [
    a.measured ? `${a.measured} measured` : "",
    a.extrapolated ? `${a.extrapolated} extrapolated` : "",
    a.manual ? `${a.manual} pinned` : "",
  ].filter(Boolean);
  const make = parts.length > 1 || a.manual || a.extrapolated ? ` (${parts.join(" · ")})` : "";
  const dyn = a.dynamic ? ` + ${a.dynamic} vehicle-driven${a.sub ? ` (${a.sub} sub-bay${a.sub > 1 ? "s" : ""})` : ""}` : "";
  const short = a.short ? ` · ${a.short} of ${a.expected} expected bays could not be placed: pin them` : "";
  return `auto grid · ${a.count} bays${make}${dyn} from calibration v${a.version}${short}`;
}

/** Where a slot of the auto grid comes from (measured when the worker says nothing). */
export function baySource(a: AutoSlots | undefined, slotId: number): BaySource | null {
  return a ? (a.sources?.[String(slotId)] ?? "measured") : null;
}

const round = (v: number) => Math.round(v * 10) / 10;

/** The parking area as one even-odd SVG path: the frame minus the area, for dimming what lies outside the walls. */
export function outsidePath(area: Area | undefined, w: number, h: number): string {
  if (!area) return "";
  const frame = `M0,0 H${w} V${h} H0 Z`;
  const x1 = area.left * w;
  const y1 = area.top * h;
  const x2 = (1 - area.right) * w;
  const y2 = (1 - area.bottom) * h;
  const inner = area.roi?.length
    ? "M" + area.roi.map(([x, y]) => `${round(Math.min(Math.max(x * w, x1), x2))},${round(Math.min(Math.max(y * h, y1), y2))}`).join(" L") + " Z"
    : `M${round(x1)},${round(y1)} H${round(x2)} V${round(y2)} H${round(x1)} Z`;
  return `${frame} ${inner}`;
}

export const SLOT_COLORS: Record<string, { stroke: string; fill: string }> = {
  FREE: { stroke: "#2eb85c", fill: "rgba(46,184,92,0.22)" },
  BUSY: { stroke: "#e55353", fill: "rgba(229,83,83,0.14)" },
  UNKNOWN: { stroke: "#9da5b1", fill: "rgba(157,165,177,0.10)" },
  // a FREE bay a neighbour's vehicle stands on (spillage protection): amber, hatched (the overlay fills it with the ENCROACH_HATCH pattern)
  ENCROACHED: { stroke: "#f08c00", fill: "rgba(240,140,0,0.16)" },
  // a FREE bay with an object standing in it (blocked by object): slate, cross-hatched (BLOCK_HATCH), unusable but not a vehicle
  BLOCKED: { stroke: "#5b6475", fill: "rgba(91,100,117,0.22)" },
};

/** The SVG pattern id of the hatched fill of an ENCROACHED bay (define it once per <svg> with <EncroachHatch />). */
export const ENCROACH_HATCH = "encroach-hatch";
/** ... and of the cross-hatched fill of a BLOCKED bay (<BlockHatch />). */
export const BLOCK_HATCH = "block-hatch";

/** The key into SLOT_COLORS: BLOCKED (an object stands in it) or ENCROACHED (a neighbour's vehicle does) when the protection says so, else the state. */
export function slotColorKey(s: Slot): string {
  return s.avail === "BLOCKED" ? "BLOCKED" : s.avail === "ENCROACHED" ? "ENCROACHED" : s.state;
}

/** Does the slot pass the counter cards' filter? ENCROACHED bays are FREE by state but are counted (and filtered) apart: FREE shows the
 * usable ones only, as the Free card counts them. */
export function slotMatches(s: Slot, filter: SlotFilter): boolean {
  if (filter == null) return true;
  const encroached = s.avail === "ENCROACHED";
  const blocked = s.avail === "BLOCKED";
  if (filter === "BLOCKED") return blocked;
  return filter === "ENCROACHED" ? encroached : !encroached && !blocked && s.state === filter;
}

/** One line of flags for a slot (the Streamlit overlay's HIDDEN / NO ROOM / DUPLICATE labels). */
export function slotFlags(s: Slot): string[] {
  const out: string[] = [];
  if (s.hidden) out.push(s.state === "BUSY" ? "hidden (kept BUSY)" : "hidden behind vehicles (not shown FREE)");
  if (s.blocked) out.push("another vehicle stands in it (not shown FREE)");
  else if (s.note && !s.hidden) out.push(s.note); // e.g. "on the vehicle of slot 6", "overlaps a vehicle"
  if (s.avail === "ENCROACHED") {
    const by = s.encroached_by?.length ? ` by ${s.encroached_by.join(", ")}` : "";
    out.push(`encroached${by}${s.cover != null ? ` (${Math.round(s.cover * 100)} % of the bay${s.free_m != null ? `, ${s.free_m} m of ground left` : ""})` : ""}: free, but not usable`);
  }
  if (s.avail === "BLOCKED") out.push(`blocked by ${s.blocked_by?.length ? s.blocked_by.join(", ") : "an object"}: free, but not usable`);
  if (s.phantom) out.push("no room (phantom)");
  if (s.duplicate_of != null) out.push(`duplicate of ${s.duplicate_of}`);
  if (s.capacity && s.capacity > 1) out.push(`${s.vehicles ?? 0} of ${s.capacity} vehicles${s.count_note ? ` (${s.count_note})` : ""}`);
  return out;
}

export function slotTitle(s: Slot): string {
  return s.name && s.name !== String(s.id) ? `Slot ${s.id} · ${s.name}` : `Slot ${s.id}`;
}

export function detectionTitle(d: Detection): string {
  return `${d.label} ${d.conf.toFixed(2)}${d.src ? ` · ${d.src}` : ""}${d.rejected ? " · ignored" : ""}`;
}

/** Newest first, no duplicates (the WebSocket can resend one the first page load already fetched), at most cap. */
export function mergeEvents(current: ParkingEvent[], incoming: ParkingEvent[], cap = 50): ParkingEvent[] {
  const seen = new Set<number>();
  return [...incoming, ...current]
    .filter((e) => (seen.has(e.id) ? false : (seen.add(e.id), true)))
    .sort((a, b) => b.id - a.id)
    .slice(0, cap);
}

/** The settings that differ from what the server has (what Apply sends). */
export function changedSettings(base: SettingsValues, draft: SettingsValues): Partial<SettingsValues> {
  const out: Partial<SettingsValues> = {};
  for (const key of Object.keys(draft) as (keyof SettingsValues)[]) {
    if (JSON.stringify(draft[key]) !== JSON.stringify(base[key])) {
      (out as Record<string, unknown>)[key] = draft[key];
    }
  }
  return out;
}

/** "2026-10-20T14:03:27+05:00" -> "14:03:27" (today) / "Oct 19 14:03" (other days): the camera's wall clock. */
export function clock(iso: string | null | undefined, today = new Date()): string {
  if (!iso) return "";
  const [date, rest] = iso.split("T");
  const time = (rest ?? "").slice(0, 8);
  const d = new Date(`${date}T00:00:00`);
  const sameDay = d.getFullYear() === today.getFullYear() && d.getMonth() === today.getMonth() && d.getDate() === today.getDate();
  if (sameDay) return time;
  return `${d.toLocaleString("en", { month: "short" })} ${d.getDate()} ${time.slice(0, 5)}`;
}

export function duration(minutes: number | null | undefined): string {
  if (minutes == null) return "";
  if (minutes < 60) return `${Math.round(minutes)} min`;
  const h = Math.floor(minutes / 60);
  return `${h} h ${Math.round(minutes - h * 60)} min`;
}

/** Entries and exits per hour of the camera's day, from the event log (Analytics). */
export function perHour(events: ParkingEvent[]): { hour: string; entries: number; exits: number }[] {
  const rows = Array.from({ length: 24 }, (_, h) => ({ hour: String(h).padStart(2, "0"), entries: 0, exits: 0 }));
  for (const e of events) {
    const h = Number(e.time.slice(11, 13));
    if (Number.isNaN(h)) continue;
    if (e.event === "entry") rows[h].entries++;
    else if (e.event === "exit") rows[h].exits++; // (notices are no movements)
  }
  return rows;
}

/** Closed sessions by length (Analytics). */
export function durationBuckets(minutes: number[]): { bucket: string; sessions: number }[] {
  const edges: [string, number][] = [["< 15 min", 15], ["15–60 min", 60], ["1–2 h", 120], ["2–4 h", 240], ["4–8 h", 480], ["> 8 h", Infinity]];
  return edges.map(([bucket, hi], i) => {
    const lo = i ? edges[i - 1][1] : 0;
    return { bucket, sessions: minutes.filter((m) => m >= lo && m < hi).length };
  });
}

// --- CCTV grid (Step 3) ---------------------------------------------------------------------------------------

export type TileTone = "ok" | "idle" | "starting" | "down" | "stream";

/** One tile's health, most severe first: worker offline > stream trouble > monitoring off > starting > tracking.
 * worker: from the tile's live state (or, until it arrives, the camera list); video: its player's status. */
export function tileHealth(w: { connected: boolean; heartbeatAge: number | null; enabled: boolean; running: boolean; hasCounts: boolean },
                           video: "off" | "connecting" | "playing" | "failed" | "reconnecting"): { tone: TileTone; label: string } {
  if (!w.connected) {
    const age = w.heartbeatAge;
    return { tone: "down", label: age == null ? "Worker offline" : `Worker offline ${age < 120 ? `${Math.round(age)} s` : `${Math.round(age / 60)} min`}` };
  }
  if (video === "reconnecting") return { tone: "stream", label: "Stream reconnecting…" };
  if (video === "failed") return { tone: "stream", label: "Stream unavailable, retrying" };
  if (!w.enabled) return { tone: "idle", label: "Monitoring off" };
  if (!w.running || !w.hasCounts) return { tone: "starting", label: "Starting…" };
  return { tone: "ok", label: "Tracking" };
}

/** Arrow keys move the selection through the visible tiles (cols wide); -> the new id. */
export function moveSelection(ids: string[], current: string | null, key: string, cols: number): string | null {
  if (!ids.length) return null;
  const i = Math.max(0, current ? ids.indexOf(current) : -1);
  const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols }[key];
  if (step === undefined) return current;
  const j = i + step;
  return j >= 0 && j < ids.length ? ids[j] : ids[i];
}

/** The page (0-based) that shows camera id with per tiles per page. */
export function pageOf(ids: string[], id: string | null, per: number): number | null {
  const i = id ? ids.indexOf(id) : -1;
  return i < 0 ? null : Math.floor(i / per);
}

/** A video is stalled when its time hasn't moved for `limit` ms while it should be playing. */
export function stalled(lastTime: number, lastChange: number, nowTime: number, now: number, limit = 8000): boolean {
  return nowTime === lastTime && now - lastChange >= limit;
}

/** "1.2 s" / "45 s" / "3 min": how old the worker's detections are. */
export function ago(seconds: number | null | undefined): string {
  if (seconds == null) return "";
  if (seconds < 10) return `${Math.max(0, seconds).toFixed(1)} s`;
  if (seconds < 120) return `${Math.round(seconds)} s`;
  return `${Math.round(seconds / 60)} min`;
}

// --- central history (Step 4) ---------------------------------------------------------------------------------

/** Occupancy chart axis label for a bucket start ("2026-10-21T14:05"): the time within a day, else the date. */
export function bucketLabel(t: string, bucket: string): string {
  const [d, hm = ""] = t.split("T");
  const date = new Date(`${d}T00:00:00`);
  const day = `${date.toLocaleString("en", { month: "short" })} ${date.getDate()}`;
  if (bucket === "1 day") return day;
  if (bucket === "5 min") return hm.slice(0, 5);
  return `${day} ${hm.slice(0, 2)}h`;
}

/** Where History & Analytics come from, for the badge: "PostgreSQL · synced 2 s ago" / "SQLite (fallback)". */
export function sourceLabel(source: string | undefined, age?: number | null): string {
  if (source === "postgres") return age == null ? "PostgreSQL" : `PostgreSQL · synced ${ago(age)} ago`;
  if (source === "sqlite") return "SQLite (fallback)";
  return "…";
}

export const CAMERA_EVENT_LABELS: Record<string, string> = {
  first_seen: "Tracking started (first seen)", worker_online: "Worker online", worker_offline: "Worker offline",
  monitoring_on: "Monitoring switched on", monitoring_off: "Monitoring switched off", scope_changed: "Detector / slot map changed",
};

/** Page buttons for a pager: first, last and the pages around the current one; null = a gap ("…"). */
export function pageList(page: number, pages: number, around = 1): (number | null)[] {
  const out: (number | null)[] = [];
  for (let p = 1; p <= pages; p++) {
    if (p === 1 || p === pages || Math.abs(p - page) <= around) out.push(p);
    else if (out[out.length - 1] !== null) out.push(null);
  }
  return out;
}
