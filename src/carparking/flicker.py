"""Flicker metrics (post-demo item 4): how often a slot's state changes without the car having gone. Pure functions on a state series, no
I/O, no thresholds of the tracker; the only parameters are the two windows of the *measurement* below, which the caller sets.

  ghost pair    an exit (BUSY -> FREE) followed by an entry (FREE -> BUSY) of the same slot within `window` seconds: the signature of a
                parked car that was lost and found again (what the evaluators saw). The window is a measurement choice, not a tracker
                setting: pairs far apart are real departures and arrivals.
  short state   a decided state that lasted less than `blink` seconds between two changes.
  UNKNOWN       frames in which a slot is not decided (hidden / blocked) are skipped: the last decided state stands, so UNKNOWN is never
                counted as an exit or an entry.

A series is one state per frame (0 FREE / 1 BUSY / None UNKNOWN) with the frame times. analyze() -> FlickerReport; compare() lines several
reports up (live vs replay off vs replay on). With a truth file (the real exits, written by whoever watched the footage) score_exits()
tells missed exits, false exits and the lag of the ones found; report_lag() needs no truth: the delay between an event's time and the
frame that reported it."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


def transitions(times, states) -> list[tuple[float, int]]:
    """[(time, new state)] for every change between decided states. The first decided state is the start, not a change."""
    out, last = [], None
    for t, s in zip(times, states):
        if s is None:
            continue
        if last is not None and s != last:
            out.append((t, s))
        last = s
    return out


@dataclass
class SlotFlicker:
    slot: int
    name: str
    seconds: float  # observed time
    transitions: int
    exits: int
    entries: int
    ghost_pairs: int
    ghost_gaps: list[float]  # seconds between the exit and the entry of each ghost pair
    short_states: int
    longest_stable_s: float  # longest time without a change (a run cut off by the recording's ends counts: a lower bound)
    busy_share: float  # of the decided frames
    unknown_share: float  # of all frames


@dataclass
class FlickerReport:
    label: str
    window: float
    blink: float
    slots: list[SlotFlicker] = field(default_factory=list)

    def _sum(self, attr) -> int:
        return sum(getattr(s, attr) for s in self.slots)

    @property
    def ghost_pairs(self) -> int:
        return self._sum("ghost_pairs")

    @property
    def exits(self) -> int:
        return self._sum("exits")

    @property
    def entries(self) -> int:
        return self._sum("entries")

    @property
    def transitions(self) -> int:
        return self._sum("transitions")

    @property
    def short_states(self) -> int:
        return self._sum("short_states")

    @property
    def slot_hours(self) -> float:
        return sum(s.seconds for s in self.slots) / 3600.0

    @property
    def ghost_pairs_per_slot_hour(self) -> float:
        return self.ghost_pairs / self.slot_hours if self.slot_hours else 0.0

    @property
    def transitions_per_slot_hour(self) -> float:
        return self.transitions / self.slot_hours if self.slot_hours else 0.0

    def summary(self) -> dict:
        return {"label": self.label, "slot_hours": round(self.slot_hours, 4), "transitions": self.transitions, "exits": self.exits,
                "entries": self.entries, "ghost_pairs": self.ghost_pairs, "short_states": self.short_states,
                "ghost_pairs_per_slot_hour": round(self.ghost_pairs_per_slot_hour, 3),
                "transitions_per_slot_hour": round(self.transitions_per_slot_hour, 3)}


def analyze(times, series: dict[int, list], names: dict[int, str] | None = None, label: str = "", window: float = 120.0,
            blink: float = 60.0) -> FlickerReport:
    """series: slot id -> one state per frame, parallel to times."""
    times = list(times)
    rep = FlickerReport(label, window, blink)
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    for k, states in series.items():
        if len(states) != len(times):
            raise ValueError(f"slot {k}: {len(states)} states for {len(times)} frames")
        tr = transitions(times, states)
        exits = [t for t, s in tr if s == 0]
        pairs, gaps = 0, []
        for t, s in tr:  # an entry closes the latest open exit of the slot
            if s == 1:
                before = [x for x in exits if x < t]
                if before and t - before[-1] <= window:
                    pairs += 1
                    gaps.append(t - before[-1])
        marks = [times[0]] + [t for t, _ in tr] + [times[-1]]
        runs = [b - a for a, b in zip(marks, marks[1:])]
        inner = runs[1:-1]  # states with a change on both sides (the first and the last run are cut off by the recording)
        decided = [s for s in states if s is not None]
        rep.slots.append(SlotFlicker(
            k, (names or {}).get(k, str(k)), span, len(tr), len(exits), len(tr) - len(exits), pairs, gaps,
            sum(1 for r in inner if r < blink), max(runs) if runs else 0.0,
            sum(decided) / len(decided) if decided else 0.0, (len(states) - len(decided)) / len(states) if states else 0.0))
    return rep


def report_lag(events: list[dict]) -> dict:
    """Delay between an event's own time (backdated to the departure / arrival) and the frame that reported it, over the events that are
    real changes (not the initial reading): {"n", "median_s", "max_s"} per kind of event."""
    out = {}
    for name, state in (("exit", 0), ("entry", 1)):
        lags = [e["reported"] - e["time"] for e in events if e["state"] == state and not e["initial"]]
        out[name] = {"n": len(lags), "median_s": round(statistics.median(lags), 1) if lags else None,
                     "max_s": round(max(lags), 1) if lags else None}
    return out


def score_exits(events: list[dict], truth: list[dict], tol: float = 60.0) -> dict:
    """events: replay events; truth: [{"slot": id, "t": time of a real departure}]. Each truth exit is matched to the nearest unused exit
    event of its slot within tol seconds of the event's time. -> {"truth", "found", "missed": [...], "false_exits": [events with no
    truth], "lag_s": reported - truth time of the found ones}. False exits are only meaningful when the truth lists every real
    departure in the recording."""
    ex = [e for e in events if e["state"] == 0 and not e["initial"]]
    used, found, missed, lags = set(), 0, [], []
    for tr in sorted(truth, key=lambda x: x["t"]):
        cand = [(abs(e["time"] - tr["t"]), i) for i, e in enumerate(ex) if i not in used and e["slot"] == tr["slot"]
                and abs(e["time"] - tr["t"]) <= tol]
        if cand:
            _, i = min(cand)
            used.add(i)
            found += 1
            lags.append(ex[i]["reported"] - tr["t"])
        else:
            missed.append(tr)
    return {"truth": len(truth), "found": found, "missed": missed, "false_exits": [e for i, e in enumerate(ex) if i not in used],
            "lag_s": [round(x, 1) for x in lags]}


def _fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def compare(reports: list[FlickerReport]) -> str:
    """A side-by-side table of the summaries, and per slot the ghost pairs of each report."""
    if not reports:
        return ""
    rows = [("slot-hours", "slot_hours", 3), ("state changes", "transitions", 0), ("exits", "exits", 0), ("entries", "entries", 0),
            ("ghost pairs", "ghost_pairs", 0), ("ghost pairs / slot-hour", "ghost_pairs_per_slot_hour", 2),
            ("changes / slot-hour", "transitions_per_slot_hour", 2), ("short states", "short_states", 0)]
    width = max(14, *(len(r.label) + 2 for r in reports))
    lines = [f"{'':28}" + "".join(f"{r.label:>{width}}" for r in reports)]
    for text, key, nd in rows:
        lines.append(f"{text:28}" + "".join(f"{_fmt(r.summary()[key], nd):>{width}}" for r in reports))
    first = reports[0]
    lines.append("")
    lines.append(f"ghost pair = exit then entry of one slot within {first.window:g} s; short state < {first.blink:g} s; UNKNOWN frames skipped")
    lines.append("per slot (ghost pairs / state changes / busy share):")
    for i, s in enumerate(first.slots):
        cells = []
        for r in reports:
            o = r.slots[i]
            cells.append(f"{o.ghost_pairs}/{o.transitions}/{o.busy_share:.0%}")
        lines.append(f"  {s.name:26}" + "".join(f"{c:>{width}}" for c in cells))
    return "\n".join(lines)
