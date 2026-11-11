import * as Dialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, ChevronDown, Clock, ExternalLink, FlaskConical, Gauge, RotateCcw, Save, ScanLine, ShieldCheck, SlidersHorizontal, Wrench, X } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { ApiError, api, withCam } from "../api";
import { changedSettings } from "../lib";
import type { ModelInfo, Options, Overflow, SettingsResponse, SettingsValues } from "../types";
import { Button, Field, Slider, Toggle, inputClass } from "./ui";

/** Settings: the everyday ones first, everything else of the Streamlit sidebar under "Advanced / Debug". Changes
 * are sent on Apply only (each applied change restarts the worker's tracking session), with the version read, so a
 * change made meanwhile in Streamlit or another tab is never overwritten silently. */
export function SettingsDrawer({ open, onClose, onNeedAdmin, tokenKey, codeStale, model, cam }: {
  cam: string | null; // the camera these settings belong to
  open: boolean;
  onClose: () => void;
  onNeedAdmin: (reason: string, retry?: () => void) => void; // retry: run again once the admin token is accepted
  tokenKey: string;
  codeStale: boolean;
  model: ModelInfo | undefined;
}) {
  const [server, setServer] = useState<SettingsResponse | null>(null);
  const [draft, setDraft] = useState<SettingsValues | null>(null);
  const [options, setOptions] = useState<Options | null>(null);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [overflow, setOverflow] = useState<Overflow | null>(null);
  // which groups are unfolded: the two everyday ones at first; a group stays mounted when folded (its values are still in the draft)
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({ general: true, ai: true });
  const toggleGroup = (id: string) => setOpenGroups((o) => ({ ...o, [id]: !o[id] }));

  const load = () =>
    api.get<SettingsResponse>(withCam("/api/settings", cam)).then((s) => {
      setServer(s);
      setDraft(s.values);
    }).catch((e) => setMessage({ tone: "error", text: e.message }));

  useEffect(() => {
    if (open) {
      setMessage(null);
      load();
      api.get<Overflow>(withCam("/api/overflow", cam)).then(setOverflow).catch(() => setOverflow(null));
    }
  }, [open, cam]);
  // new tokens (e.g. the admin token asked for on Apply): keep the unsaved changes; only fetch if nothing loaded yet
  useEffect(() => {
    if (open && !draft) load();
  }, [tokenKey]);

  // the detector list depends on the camera name, the class lists on the detector
  useEffect(() => {
    if (!open || !draft) return;
    const q = new URLSearchParams({ camera: draft.camera, detector: draft.detector });
    api.get<Options>(withCam(`/api/settings/options?${q}`, cam)).then(setOptions).catch(() => {});
  }, [open, draft?.camera, draft?.detector]);

  const changes = useMemo(() => (server && draft ? changedSettings(server.values, draft) : {}), [server, draft]);
  const dirty = Object.keys(changes).length;
  const set = <K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) => setDraft((d) => (d ? { ...d, [key]: value } : d));

  const apply = async () => {
    if (!server || !dirty) return;
    setBusy(true);
    setMessage(null);
    try {
      const r = await api.put<SettingsResponse>(withCam("/api/settings", cam), { ...changes, version: server.version });
      setServer(r);
      setDraft(r.values);
      setMessage({ tone: "ok", text: `Applied (version ${r.version}). The worker picks it up within a second.` });
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 401 || err.status === 503) onNeedAdmin(err.status === 503 ? "The server has no admin token set (CARPARK_API_TOKEN): settings are read-only." : "Applying settings needs the admin token.", apply);
      else if (err.status === 409) {
        setMessage({ tone: "error", text: "Someone changed the settings meanwhile (another tab or user). Reloaded: check and apply again." });
        const keep = changes;
        const fresh = await api.get<SettingsResponse>(withCam("/api/settings", cam));
        setServer(fresh);
        setDraft({ ...fresh.values, ...keep });
      } else setMessage({ tone: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  };

  // not a settings field: sent at once (no Apply, no restart), the worker rebuilds its monitor on the next frame
  const switchOverflow = async (enabled: boolean) => {
    setMessage(null);
    try {
      setOverflow(await api.put<Overflow>(withCam("/api/overflow", cam), { enabled }));
      setMessage({ tone: "ok", text: `Dynamic overflow & sub-bays ${enabled ? "on" : "off"}. The worker picks it up on its next frame.` });
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 401 || err.status === 503) onNeedAdmin("Switching dynamic overflow needs the admin token.", () => switchOverflow(enabled));
      else setMessage({ tone: "error", text: err.message });
    }
  };

  const command = async (action: string) => {
    setConfirm(null);
    try {
      const r = await api.post<{ queued?: number; done?: boolean }>(withCam("/api/commands", cam), { action });
      setMessage({ tone: "ok", text: r.done ? `Done: ${action}.` : `Sent to the worker: ${action}.` });
    } catch (e) {
      const err = e as ApiError;
      if (err.status === 401 || err.status === 503) onNeedAdmin("Commands need the admin token.", () => command(action));
      else setMessage({ tone: "error", text: err.message });
    }
  };

  const r = options?.ranges ?? {};
  const range = (k: string, lo: number, hi: number) => (r[k] ?? [lo, hi]) as [number, number];
  const pct = (v: number) => `${Math.round(v * 100)} %`;
  const fixed = draft?.slot_mode === "fixed";
  // unsaved changes per group, shown as a badge on a folded header
  const pending = (id: string) => GROUP_KEYS[id].filter((k) => k in changes).length;
  const group = (id: string, icon: ReactNode, title: string, subtitle: string, children: ReactNode) => (
    <Group id={id} icon={icon} title={title} subtitle={subtitle} open={!!openGroups[id]} onToggle={() => toggleGroup(id)} pending={pending(id)}>
      {children}
    </Group>
  );

  // Radix Dialog as a side sheet: focus trapped inside, Escape / backdrop click close it, no scrolling behind, role="dialog" named "Settings"
  return (
    <Dialog.Root open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <AnimatePresence>
        {open && (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild forceMount>
              <motion.div className="fixed inset-0 z-40 bg-black/30" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
            </Dialog.Overlay>
          <Dialog.Content asChild forceMount aria-describedby={undefined}>
          <motion.aside initial={{ x: 480 }} animate={{ x: 0 }} exit={{ x: 480 }} transition={{ type: "tween", duration: 0.22 }}
                        className="fixed inset-y-0 right-0 z-40 flex w-full max-w-md flex-col border-l border-line bg-surface shadow-2xl"
                        data-testid="settings-drawer">
            <header className="flex h-14 items-center justify-between border-b border-line px-5">
              <div>
                <Dialog.Title className="text-base font-semibold">Settings</Dialog.Title>
                {server && <p className="text-xs text-muted">version {server.version}</p>}
              </div>
              <button type="button" onClick={onClose} className="rounded p-1.5 text-muted hover:bg-body" aria-label="Close"><X size={18} /></button>
            </header>

            {!draft ? (
              <p className="p-5 text-sm text-muted">{message?.text ?? "Loading…"}</p>
            ) : (
              <div className="flex-1 space-y-5 overflow-y-auto p-5">
                {group("general", <SlidersHorizontal size={16} />, "General & Monitoring", "Monitoring, camera and stream quality", (
                  <>
                  <div className="rounded-lg border border-line bg-body/60 p-3"
                       title="When turned On, the system automatically tracks vehicles, logs entries, and logs exits in the background without requiring an open browser window.">
                    <Toggle checked={draft.enabled} onChange={(v) => set("enabled", v)}
                            label={<span className="font-semibold">Monitoring</span>} />
                    <p className="mt-1 text-xs text-muted">Enable or disable automatic background vehicle tracking.</p>
                  </div>


                    <Field label="Camera" hint="Its slot map, parking area and history belong to it. Cameras are added and edited on the Cameras page.">
                      <input className={`${inputClass} bg-body`} value={draft.camera} readOnly />
                    </Field>
                    <Field label="Stream Selection" hint="Select the active camera stream quality for live processing."
                           tip="Choose Main stream for maximum visual resolution or Sub-stream for lower video processing latency.">
                      <select className={inputClass} value={draft.stream} onChange={(e) => set("stream", e.target.value as SettingsValues["stream"])}>
                        <option value="main">Main stream (101, full resolution)</option>
                        <option value="sub">Sub-stream (102, lower latency)</option>
                        <option value="custom">Custom URL</option>
                      </select>
                    </Field>
                    {draft.stream === "custom" && (
                      <Field label="RTSP URL" hint="rtsp://user:pass@host:554/... (an @ in the password is %40). Shown masked; leave it to keep the stored one.">
                        <input className={inputClass} value={draft.custom_url} onChange={(e) => set("custom_url", e.target.value)} />
                      </Field>
                    )}

                  </>
                ))}

                {group("ai", <Brain size={16} />, "AI & Detection Engine", "Detector model and detection confidence", (
                  <>
                    <ModelStatus model={model} />
                    <Field label="AI Detection Model" hint="Choose the AI model used for detecting vehicles."
                           tip="Default / Light (yolov8s) is fast and lightweight. YOLO11m is more accurate for far or partially covered vehicles but uses more computing power. Swaps live without restarting history.">
                      <select className={inputClass} value={draft.model_variant} data-testid="model-variant"
                              onChange={(e) => set("model_variant", e.target.value as SettingsValues["model_variant"])}>
                        <option value="auto">Auto · from the hardware{model ? ` (${modelName(model.auto_choice)})` : ""}</option>
                        <option value="default">Default / Light (yolov8s)</option>
                        <option value="yolo11m">YOLO11m (Medium)</option>
                      </select>
                    </Field>
                    {model && !model.swappable && (
                      <p className="rounded-md bg-warning/15 px-3 py-2 text-xs text-ink">
                        The chosen detector is a camera model, the hybrid or the bay model: it is never swapped. The choice applies to the stock COCO detector.
                      </p>
                    )}

                    <Field label="Detection Confidence" hint="Minimum AI certainty level required to identify a vehicle."
                           tip="Sets the sensitivity for detecting cars. A setting of 0.30 means the AI must be at least 30% confident that an object is a vehicle.">
                      <Slider value={draft.conf} min={range("conf", 0.05, 0.5)[0]} max={range("conf", 0.05, 0.5)[1]} step={0.01}
                              onChange={(v) => set("conf", v)} format={(v) => v.toFixed(2)} />
                    </Field>

                  </>
                ))}

                {group("timing", <Clock size={16} />, "Timing & Hysteresis", "How long before a slot changes state", (
                  <>
                    <Field label="Entry Delay (Seconds)" hint="Time a vehicle must remain stationary to register as Occupied/BUSY."
                           tip="Prevents passing traffic or temporary slow-moving cars from triggering accidental parking entry logs.">
                      <Slider value={draft.hold} min={0} max={120} step={5} onChange={(v) => set("hold", v)} format={(v) => `${v} s`} />
                    </Field>
                    <Field label="Exit Verification Buffer" hint="How long a vehicle must be gone before the slot turns FREE."
                           tip="Ensures a slot is not prematurely marked as empty due to passing shadows or temporary frame glitches.">
                      <Slider value={draft.exit_hold} min={0} max={600} step={15} onChange={(v) => set("exit_hold", v)} format={(v) => `${v} s`} />
                    </Field>
                    <Field label="Exit Verification Buffer (Frames)" hint="How many consecutive frames must agree the vehicle is gone."
                           tip="Ensures a slot is not prematurely marked as empty due to passing shadows or temporary frame glitches.">
                      <Slider value={draft.exit_frames} min={1} max={60} step={1} onChange={(v) => set("exit_frames", v)} />
                    </Field>
                    <Field label="Covered Slot Memory (Coverage)" hint="Covered this much by cars in front = HIDDEN: state kept."
                           tip="Keeps a parking slot marked as BUSY when the vehicle is partially hidden behind pillars, walls, or other vehicles.">
                      <Slider value={draft.occlusion} min={0.1} max={0.9} step={0.05} onChange={(v) => set("occlusion", v)} format={(v) => `${Math.round(v * 100)} %`} />
                    </Field>
                    <Field label="Covered Slot Memory" hint="Time to hold slot status when a car is partially blocked."
                           tip="Keeps a parking slot marked as BUSY when the vehicle is partially hidden behind pillars, walls, or other vehicles.">
                      <Slider value={draft.hidden_max} min={10} max={1800} step={10} onChange={(v) => set("hidden_max", v)} format={(v) => `${v} s`} />
                    </Field>

                  </>
                ))}

                {group("protect", <ShieldCheck size={16} />, "Advanced Smart Protections", "Spillage, the non-vehicle filter and blocked-by-object detection (off by default)", (
                  <>
                    <div data-testid="spillage-switch">
                      <Toggle checked={!!draft.spillage} onChange={(v) => set("spillage", v)} label="Spillage Protection" />
                    </div>
                    <p className="text-xs text-muted">
                      A free bay that a neighbour's vehicle stands on (30 % of it or more, or two vehicles together 45 %) is shown
                      hatched amber as ENCROACHED and is taken out of the Free count (Free = usable bays). Its state stays FREE, so
                      nothing in the history changes.
                    </p>
                    <div data-testid="blocked-switch">
                      <Toggle checked={!!draft.blocked_objects} onChange={(v) => set("blocked_objects", v)} label="Blocked by Object Detection" />
                    </div>
                    <p className="text-xs text-muted">
                      A free bay with a potted plant, vase, box (suitcase), chair or bench standing in it is shown BLOCKED (cross-hatched grey)
                      and is taken out of the Free count (Free = usable bays). It stays FREE for the history and the sessions: no entry or exit is
                      logged. A person or an animal never blocks a bay, and a bay with a vehicle in it is never blocked.
                    </p>
                    <div data-testid="obstacle-switch">
                      <Toggle checked={!!draft.obstacle_filter} onChange={(v) => set("obstacle_filter", v)} label="Non-Vehicle Obstacle Filtering" />
                    </div>
                    <p className="text-xs text-muted">
                      Boxes the detector calls a vehicle that are too small for the bay, the wrong shape, cone-orange or a flat shadow
                      do not make a bay BUSY (a bay that is already BUSY is never affected). The Worker card lists what was dropped.
                    </p>
                    <p className="text-xs text-muted">
                      Both are off by default, apply on the worker's next frame (no restart, no new settings version) and work on
                      fixed maps and the auto grid.
                    </p>

                  </>
                ))}

                {group("counting", <Gauge size={16} />, "Dynamic Counting & Thresholds", "Hidden vehicles, ground taken and multi-slot limits", (
                  <>
                    <p className="text-xs text-muted" data-testid="tuning-note">
                      What is counted FREE / BUSY / UNKNOWN on the video banner and the dashboard. Applied by the worker on its
                      next frame, no restart; the slot map and the history are never changed by these. The first three only
                      decide when a slot shows UNKNOWN; "Bay held" decides when a vehicle makes it BUSY and "Pin grace" how far past a
                      capacity pin's edge a vehicle still counts in it.
                    </p>
                    <Field label="Hidden: vehicles in front cover more than" tip="Higher = a slot must be more blocked before it shows UNKNOWN. Lower = UNKNOWN appears sooner." hint="Of a free slot's ground footprint (where a car's wheels stand): above this it can't be seen, so it shows UNKNOWN, not FREE.">
                      <Slider value={draft.hidden_ground} min={range("hidden_ground", 0.5, 1)[0]} max={range("hidden_ground", 0.5, 1)[1]} step={0.05}
                              onChange={(v) => set("hidden_ground", v)} format={pct} />
                    </Field>
                    <Field label="Ground taken: a vehicle covers more than" tip="Lower = a vehicle standing near a slot blocks it sooner. Higher = it must cover more of the floor." hint="Of a free slot's ground footprint, standing at its depth or behind it: no room for a car, so not FREE. A neighbour's box spilling over the side stays under this.">
                      <Slider value={draft.ground_share} min={range("ground_share", 0.1, 0.9)[0]} max={range("ground_share", 0.1, 0.9)[1]} step={0.05}
                              onChange={(v) => set("ground_share", v)} format={pct} />
                    </Field>
                    <Field label="Single Car Multi-Slot Protection" hint="Prevents a single vehicle from claiming two adjacent slots."
                           tip="If a large vehicle overlaps a neighboring empty slot past this percentage, the second slot shows UNKNOWN instead of a duplicate BUSY count.">
                      <Slider value={draft.vehicle_iou} min={range("vehicle_iou", 0.05, 0.9)[0]} max={range("vehicle_iou", 0.05, 0.9)[1]} step={0.05}
                              onChange={(v) => set("vehicle_iou", v)} format={pct} />
                    </Field>
                    <Field label="Slot Alignment Tolerance" hint="Required portion of a vehicle's footprint inside the bay boundary."
                           tip="A vehicle must have at least this percentage of its bottom footprint inside the bay lines to mark it as BUSY. Prevents overhanging bumpers or tires from triggering neighboring bays.">
                      <Slider value={draft.bay_share} min={range("bay_share", 0.1, 0.95)[0]} max={range("bay_share", 0.1, 0.95)[1]} step={0.05}
                              onChange={(v) => set("bay_share", v)} format={pct} />
                    </Field>
                    <Field label="Pin Boundary Grace Margin" hint="Allowed tolerance margin for Auto-Grid capacity pins."
                           tip="Allows vehicles parked slightly past the drawn pin edge to still be counted inside the capacity pin.">
                      <Slider value={draft.pin_reach} min={range("pin_reach", 0, 0.5)[0]} max={range("pin_reach", 0, 0.5)[1]} step={0.05}
                              onChange={(v) => set("pin_reach", v)} format={pct} />
                    </Field>
                    {options?.tuning_defaults && (
                      <button type="button" data-testid="tuning-reset"
                              disabled={(["hidden_ground", "ground_share", "vehicle_iou", "bay_share", "pin_reach"] as const).every((k) => draft[k] === options.tuning_defaults![k])}
                              onClick={() => setDraft((d) => (d ? { ...d, ...options.tuning_defaults } : d))}
                              className="text-xs font-semibold text-primary hover:underline disabled:text-muted disabled:no-underline">
                        Reset to defaults ({Math.round(options.tuning_defaults.hidden_ground * 100)} % · {Math.round(options.tuning_defaults.ground_share * 100)} % · {Math.round(options.tuning_defaults.vehicle_iou * 100)} % · {Math.round(options.tuning_defaults.bay_share * 100)} % · {Math.round(options.tuning_defaults.pin_reach * 100)} %)
                      </button>
                    )}

                  </>
                ))}

                {group("layout", <ScanLine size={16} />, "Layout & Parking Zone", "Auto grid and the parking-zone crop", (
                  <>
                {overflow && (
                  <div className="space-y-3">
                    <SubHead>Auto grid</SubHead>
                    <div data-testid="overflow-switch">
                      <Toggle checked={overflow.enabled} onChange={switchOverflow} label="Dynamic Overflow & Sub-Bays" />
                    </div>
                    <p className="text-xs text-muted">
                      Dynamic mode on a HIGH calibration only. On: a vehicle parked outside every bay gets a bay of its own (D1, D2 ...)
                      and a second vehicle in an occupied bay gets a sub-bay (B3.2). Off: such vehicles are not counted. Applied at once,
                      no restart.{overflow.auto_dynamic !== overflow.slot_split && " Now only one of the two is on: switching sets both."}
                    </p>
                  </div>
                )}

                    <div className="space-y-3">
                      <SubHead>Parking Zone Crop (Hard Walls)</SubHead>
                      <div className="grid grid-cols-4 gap-2">
                        {(["cut_top", "cut_bottom", "cut_left", "cut_right"] as const).map((k) => (
                          <Field key={k} label={k.replace("cut_", "")}>
                            <input type="number" className={inputClass} min={0} max={60} value={draft[k]}
                                   onChange={(e) => set(k, Math.min(60, Math.max(0, Number(e.target.value) || 0)))} />
                          </Field>
                        ))}
                      </div>
                      <p className="text-xs text-muted"
                         title="Crops out non-parking areas such as adjacent main roads or sidewalks so detection focuses only on valid parking slots.">Percentage cut from each edge of the frame to isolate the parking zone.</p>
                      <Field label="Polygon (optional)" hint="Corners as x,y shares of the frame, separated by ';'.">
                        <input className={inputClass} value={draft.roi} placeholder="0.40,0.10; 0.58,0.10; 0.58,1; 0.17,1"
                               onChange={(e) => set("roi", e.target.value)} />
                      </Field>
                    </div>
                  </>
                ))}

                <div className="rounded-lg border border-line">
                  <button type="button" onClick={() => setAdvanced((a) => !a)} data-testid="advanced-toggle"
                          className="flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-semibold text-ink">
                    <FlaskConical size={16} className="text-primary" />
                    <span className="flex-1">Advanced / Debug Settings</span>
                    <ChevronDown size={16} className={`transition-transform ${advanced ? "rotate-180" : ""}`} />
                  </button>
                  <AnimatePresence initial={false}>
                    {advanced && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                                  className="overflow-hidden">
                        <div className="space-y-4 border-t border-line p-4">
                          <Field label="Detector">
                            <select className={inputClass} value={draft.detector} onChange={(e) => set("detector", e.target.value)}>
                              {(options?.detector ?? [draft.detector]).map((d) => <option key={d} value={d}>{d}</option>)}
                            </select>
                          </Field>
                          <Field label="Bay model (PKLot detector only)" hint="Auto = layout analyser picks per image.">
                            <select className={inputClass} value={draft.bay_model} onChange={(e) => set("bay_model", e.target.value as SettingsValues["bay_model"])}>
                              <option value="auto">Auto (layout analyser)</option>
                              <option value="baseline">YOLOv8 axis-aligned (baseline)</option>
                              <option value="obb">YOLO-OBB (oriented)</option>
                            </select>
                          </Field>
                          <Field label="Vehicle inference" hint="1920 px separates the back-row cars (~1.5 s/frame on this CPU).">
                            <select className={inputClass} value={draft.inference} onChange={(e) => set("inference", e.target.value as SettingsValues["inference"])}>
                              <option value="1280">Whole frame 1280 px</option>
                              <option value="1920">Whole frame 1920 px</option>
                              <option value="sahi">SAHI 640 px tiles</option>
                            </select>
                          </Field>
                          <Chips label="Target classes (count as BUSY)" all={options?.classes ?? []} value={draft.classes} onChange={(v) => set("classes", v)} />
                          {!!options?.extra.length && (
                            <Chips label="Also detect (shown only, never a slot)" all={options.extra} value={draft.extra} onChange={(v) => set("extra", v)} />
                          )}
                          <Field label="Slot mapping mode">
                            <select className={inputClass} value={draft.slot_mode} onChange={(e) => set("slot_mode", e.target.value as SettingsValues["slot_mode"])}>
                              <option value="dynamic">Dynamic (learned from where vehicles stand)</option>
                              <option value="fixed">Fixed (drawn slots)</option>
                            </select>
                          </Field>
                          {fixed && (
                            <>
                              <a href="#/slots" onClick={onClose} className="inline-flex items-center gap-1 text-xs font-medium text-primary">
                                <ExternalLink size={12} /> Draw and edit fixed slots in the Slot editor
                              </a>
                              <Field label="A vehicle fills a slot when">
                                <select className={inputClass} value={draft.match_rule} onChange={(e) => set("match_rule", e.target.value as SettingsValues["match_rule"])}>
                                  <option value="ground">its ground point is in it (low angle)</option>
                                  <option value="center">its centre is in it (top-down)</option>
                                  <option value="iou">its box overlaps it (IoU)</option>
                                </select>
                              </Field>
                              {draft.match_rule === "iou" && (
                                <Field label="Minimum IoU">
                                  <Slider value={draft.match_iou} min={0.05} max={0.8} step={0.05} onChange={(v) => set("match_iou", v)} format={(v) => v.toFixed(2)} />
                                </Field>
                              )}
                            </>
                          )}
                          {!fixed && (
                            <>
                              <Toggle checked={draft.learn} onChange={(v) => set("learn", v)} label="Learn new slots" />
                              <Field label="Seconds before a new slot is learned">
                                <Slider value={draft.learn_hold} min={0} max={300} step={5} onChange={(v) => set("learn_hold", v)} format={(v) => `${v} s`} />
                              </Field>
                            </>
                          )}
                          <Toggle checked={draft.absent_free} onChange={(v) => set("absent_free", v)} label="No detection = FREE (COCO detectors)" />
                          <Field label="Save a raw frame every N seconds (dataset capture)" hint="0 = off. Frames go to dataset_raw/.">
                            <Slider value={draft.save_every} min={0} max={600} step={10} onChange={(v) => set("save_every", v)} format={(v) => (v ? `${v} s` : "off")} />
                          </Field>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                <Section icon={<Wrench size={16} />} title="Actions">
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" onClick={() => command("snapshot")}>📸 Save snapshot</Button>
                    {confirm === "reset" ? (
                      <span className="inline-flex items-center gap-2 rounded-md bg-danger/10 px-2 py-1 text-xs text-danger">
                        {fixed ? "Close all open sessions?" : "Forget the learned slots and close open sessions?"}
                        <Button variant="danger" onClick={() => command("reset")}>Yes</Button>
                        <Button variant="ghost" onClick={() => setConfirm(null)}>No</Button>
                      </span>
                    ) : (
                      <Button variant="outline" onClick={() => setConfirm("reset")}>{fixed ? "Close open sessions" : "Reset slot map"}</Button>
                    )}
                    <Button variant="outline" onClick={() => command("restart")}
                            title={codeStale ? "The worker runs older code than the files" : "Restart the worker process"}>
                      <RotateCcw size={14} /> Restart worker{codeStale ? " (older code)" : ""}
                    </Button>
                  </div>
                </Section>
              </div>
            )}

            <footer className="border-t border-line p-4">
              {message && (
                <p className={`mb-3 rounded-md px-3 py-2 text-sm ${message.tone === "ok" ? "bg-success/10 text-success-ink" : "bg-danger/10 text-danger"}`}>
                  {message.text}
                </p>
              )}
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-muted">{dirty ? `${dirty} unsaved change${dirty > 1 ? "s" : ""}` : "No changes"}</span>
                <div className="flex gap-2">
                  <Button variant="ghost" disabled={!dirty || busy} onClick={() => server && setDraft(server.values)}>Cancel</Button>
                  <Button disabled={!dirty || busy} onClick={apply}><Save size={14} /> Apply</Button>
                </div>
              </div>
            </footer>
          </motion.aside>
          </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
}

export const modelName = (v: string) => (v === "yolo11m" ? "YOLO11m (Medium)" : "Default / Light");

/** What runs now: the active model, whether the hardware picked it, and why. */
export function ModelStatus({ model }: { model: ModelInfo | undefined }) {
  if (!model) return <p className="text-xs text-muted">No status from the worker yet.</p>;
  return (
    <div className="rounded-lg border border-line bg-body/60 px-3 py-2.5 text-sm" data-testid="model-status">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted">Active:</span>
        <span className="font-semibold text-ink">{modelName(model.active)}</span>
        {model.weights && <span className="font-mono text-xs text-muted">{model.weights}</span>}
        {model.auto && (
          <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary">Auto-selected</span>
        )}
      </div>
      <p className="mt-1 text-xs text-muted" title={JSON.stringify(model.hardware)}>
        {model.auto ? "Picked at worker start: " : "Auto would pick " + modelName(model.auto_choice) + ": "}
        {model.reason}
      </p>
    </div>
  );
}

/** The settings fields each group holds, for the "unsaved" badge on a folded header. */
const GROUP_KEYS: Record<string, (keyof SettingsValues)[]> = {
  general: ["enabled", "stream", "custom_url"],
  ai: ["model_variant", "conf"],
  timing: ["hold", "exit_hold", "exit_frames", "occlusion", "hidden_max"],
  protect: ["spillage", "obstacle_filter", "blocked_objects"],
  counting: ["hidden_ground", "ground_share", "vehicle_iou", "bay_share", "pin_reach"],
  layout: ["cut_top", "cut_bottom", "cut_left", "cut_right", "roi"],
};

/** A collapsible card: icon, title and a one-line summary on the header; the body stays mounted when folded (hidden), so nothing
 * in it is lost and the Apply payload is the same whatever is unfolded. */
function Group({ id, icon, title, subtitle, open, onToggle, pending, children }: {
  id: string; icon: ReactNode; title: string; subtitle: string; open: boolean; onToggle: () => void; pending: number; children: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-line" data-testid={`group-${id}`} data-open={open}>
      <button type="button" onClick={onToggle} aria-expanded={open} aria-controls={`group-body-${id}`} data-testid={`group-toggle-${id}`}
              className="flex w-full items-center gap-3 px-4 py-3 text-left">
        <span className="text-primary">{icon}</span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold text-ink">{title}</span>
          <span className="block truncate text-xs text-muted">{subtitle}</span>
        </span>
        {pending > 0 && (
          <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary" data-testid={`group-pending-${id}`}>{pending} changed</span>
        )}
        <ChevronDown size={16} className={`shrink-0 text-muted transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      <div id={`group-body-${id}`} hidden={!open} className="space-y-4 border-t border-line p-4">{children}</div>
    </div>
  );
}

const SubHead = ({ children }: { children: ReactNode }) => (
  <h4 className="text-xs font-semibold tracking-wide text-muted uppercase">{children}</h4>
);

function Section({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h3 className="flex items-center gap-2 text-sm font-semibold text-ink"><span className="text-primary">{icon}</span>{title}</h3>
      {children}
    </section>
  );
}

function Chips({ label, all, value, onChange }: { label: string; all: string[]; value: string[]; onChange: (v: string[]) => void }) {
  return (
    <Field label={label}>
      <div className="flex flex-wrap gap-1.5">
        {all.map((c) => {
          const on = value.includes(c);
          return (
            <button key={c} type="button" onClick={() => onChange(on ? value.filter((v) => v !== c) : [...value, c])}
                    className={`rounded-full border px-2.5 py-0.5 text-xs ${on ? "border-primary bg-primary/10 text-primary" : "border-line text-muted"}`}>
              {c}
            </button>
          );
        })}
      </div>
    </Field>
  );
}
