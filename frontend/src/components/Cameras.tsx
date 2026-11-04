import { Camera, Maximize2, PenTool, Pencil, Plus, Star, Trash2, X } from "lucide-react";
import { useState } from "react";
import { ApiError, api, withCam } from "../api";
import type { CameraInfo } from "../types";
import { Button, Card, Field, Toggle, inputClass } from "./ui";

type Form = { name: string; label: string; rtsp_url: string; rtsp_sub_url: string; location: string; notes: string; sort_order: number };
const EMPTY: Form = { name: "", label: "", rtsp_url: "", rtsp_sub_url: "", location: "", notes: "", sort_order: 0 };

/** Camera management: the registry (PostgreSQL) of cameras, each with its own monitoring switch and Fixed / Dynamic
 * mode, its own settings, slot map and history (isolated). Adding one starts its worker within seconds (idle until
 * monitoring is switched on). */
export function CamerasPage({ cameras, error, onChanged, onNeedAdmin, onOpen, onDraw }: {
  cameras: CameraInfo[];
  error?: string | null; // the camera list couldn't be loaded (shown instead of an empty list)
  onChanged: () => void;
  onNeedAdmin: (reason: string, retry?: () => void) => void; // retry: run again once the admin token is accepted
  onOpen: (id: string) => void;
  onDraw: (id: string) => void;
}) {
  const [form, setForm] = useState<Form | null>(null);
  const [editing, setEditing] = useState<string | null>(null); // camera id, or null = a new one
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const fail = (e: unknown, what: string, retry?: () => void) => {
    const err = e as ApiError;
    if (err.status === 401 || err.status === 503) {
      onNeedAdmin(`${what} needs the admin token (the viewer token allows viewing only).`, retry);
    } else setMessage({ tone: "error", text: err.message });
  };

  const save = async () => {
    if (!form) return;
    setBusy(true);
    setMessage(null);
    const metadata = { location: form.location.trim(), notes: form.notes.trim() };
    try {
      if (editing) {
        await api.patch(`/api/cameras/${editing}`, { label: form.label.trim(), rtsp_url: form.rtsp_url, rtsp_sub_url: form.rtsp_sub_url,
                                                     sort_order: form.sort_order, metadata });
        setMessage({ tone: "ok", text: "Saved. A changed URL restarts that camera's worker." });
      } else {
        const r = await api.post<CameraInfo>("/api/cameras", { name: form.name.trim(), label: form.label.trim() || null, rtsp_url: form.rtsp_url.trim(),
          rtsp_sub_url: form.rtsp_sub_url.trim() || null, sort_order: form.sort_order, metadata });
        setMessage({ tone: "ok", text: `Added "${r.display}" (monitoring off). Its worker starts within seconds; switch monitoring on below.` });
      }
      setForm(null);
      onChanged();
    } catch (e) {
      fail(e, editing ? "Saving the camera" : "Adding a camera", save);
    } finally {
      setBusy(false);
    }
  };

  const settings = async (cam: CameraInfo, patch: Record<string, unknown>) => {
    setMessage(null);
    try {
      if (patch.slot_mode === "fixed") {
        const slots = await api.get<{ slots: unknown[] }>(withCam("/api/fixed-slots", cam.id));
        if (!slots.slots.length) {
          setMessage({ tone: "error", text: `${cam.display}: draw and save its slots first (Slot editor): Fixed mode watches only drawn slots.` });
          return;
        }
      }
      await api.put(withCam("/api/settings", cam.id), patch);
      onChanged();
    } catch (e) {
      fail(e, "Changing a camera's settings", () => settings(cam, patch));
    }
  };

  const remove = async (cam: CameraInfo) => {
    setConfirm(null);
    try {
      await api.delete(`/api/cameras/${cam.id}`);
      setMessage({ tone: "ok", text: `Removed "${cam.display}". Its history stays on disk (data/cameras/${cam.id}).` });
      onChanged();
    } catch (e) {
      fail(e, "Removing a camera", () => remove(cam));
    }
  };

  const edit = (cam?: CameraInfo) => {
    setEditing(cam?.id ?? null);
    setForm(cam ? { name: cam.name, label: cam.display, rtsp_url: cam.rtsp_url, rtsp_sub_url: cam.rtsp_sub_url, sort_order: cam.sort_order,
                    location: String(cam.metadata.location ?? ""), notes: String(cam.metadata.notes ?? "") } : { ...EMPTY });
  };

  return (
    <div className="space-y-4">
      {error && !cameras.length && (
        <p className="rounded-md bg-danger/10 px-3 py-2 text-sm text-danger" data-testid="cameras-error">
          Couldn't load the cameras: {error}. Check the access token (key icon in the header).
        </p>
      )}
      {message && (
        <p className={`rounded-md px-3 py-2 text-sm ${message.tone === "ok" ? "bg-success/10 text-[#1b7a3e]" : "bg-danger/10 text-danger"}`}>{message.text}</p>
      )}
      <Card title={`Cameras (${cameras.length})`} bodyClass="p-0"
            actions={<Button onClick={() => edit()}><Plus size={14} /> Add camera</Button>}>
        <ul className="divide-y divide-line" data-testid="camera-list">
          {cameras.map((cam) => {
            const st = cam.status;
            return (
              <li key={cam.id} className="flex flex-wrap items-center gap-4 px-4 py-3">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><Camera size={20} /></span>
                <div className="min-w-48 flex-1">
                  <p className="flex items-center gap-2 font-semibold text-ink" data-testid={`camera-name-${cam.id}`}>
                    {cam.display}
                    {cam.primary && <span className="inline-flex items-center gap-1 rounded-full bg-warning/15 px-2 py-0.5 text-[10px] font-bold text-[#b07800]" title="Shown in Streamlit; its history continues from before"><Star size={10} /> PRIMARY</span>}
                  </p>
                  <p className="font-mono text-xs text-muted">{cam.id}{cam.display !== cam.name ? ` · key “${cam.name}”` : ""} · {cam.rtsp_url}</p>
                  {(cam.metadata.location as string) && <p className="text-xs text-muted">{String(cam.metadata.location)}</p>}
                </div>
                <div className="w-40 text-xs">
                  <p className={st.connected ? (st.running ? "text-success" : "text-[#b07800]") : "text-danger"}>
                    ● {st.connected ? (st.running ? "tracking" : st.enabled ? "starting" : "idle") : "worker down"}
                  </p>
                  {st.counts && st.running && <p className="text-muted">FREE {st.counts.free} · BUSY {st.counts.busy} · {st.counts.slots} slots</p>}
                </div>
                <div className="w-36">
                  <Toggle checked={st.enabled} onChange={(v) => settings(cam, { enabled: v })} label="Monitoring" />
                </div>
                <div className="flex overflow-hidden rounded-md border border-line text-xs font-semibold" data-testid={`mode-${cam.id}`}>
                  {(["dynamic", "fixed"] as const).map((m) => (
                    <button key={m} type="button" onClick={() => st.slot_mode !== m && settings(cam, { slot_mode: m })}
                            className={`px-3 py-1.5 capitalize ${st.slot_mode === m ? "bg-primary text-white" : "bg-white text-muted hover:text-primary"}`}>
                      {m}
                    </button>
                  ))}
                </div>
                <div className="flex gap-1">
                  <Button variant="ghost" title="Open fullscreen" onClick={() => onOpen(cam.id)}><Maximize2 size={15} /></Button>
                  <Button variant="ghost" title="Draw its fixed slots" onClick={() => onDraw(cam.id)}><PenTool size={15} /></Button>
                  <Button variant="ghost" title="Edit" onClick={() => edit(cam)}><Pencil size={15} /></Button>
                  {confirm === cam.id ? (
                    <span className="inline-flex items-center gap-1 rounded-md bg-danger/10 px-2 text-xs text-danger">
                      Remove?
                      <Button variant="danger" onClick={() => remove(cam)}>Yes</Button>
                      <Button variant="ghost" onClick={() => setConfirm(null)}>No</Button>
                    </span>
                  ) : (
                    <Button variant="ghost" title={cam.primary ? "The primary camera can't be removed" : "Remove"}
                            disabled={cam.primary} onClick={() => setConfirm(cam.id)}><Trash2 size={15} /></Button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </Card>

      {form && (
        <Card title={editing ? `Edit camera · ${form.label || form.name}` : "Add camera"}
              actions={<button type="button" onClick={() => setForm(null)} className="rounded p-1 text-muted hover:bg-body" aria-label="Close form"><X size={16} /></button>}>
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="Display name" hint="Shown everywhere; change it any time (e.g. East Gate, Main Parking Area).">
              <input className={inputClass} value={form.label} data-testid="camera-label" placeholder={form.name || "East Gate"}
                     onChange={(e) => setForm({ ...form, label: e.target.value })} />
            </Field>
            <Field label={editing ? "History key (fixed)" : "History key"}
                   hint={editing ? "Its slot map, settings and history are stored under it, so it never changes (renaming it would lose the history)."
                     : "Stored with its history: can't change later. Short and stable, e.g. gate-01."}>
              <input className={`${inputClass} ${editing ? "bg-body text-muted" : ""}`} value={form.name} readOnly={!!editing} data-testid="camera-key"
                     onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="gate-01" />
            </Field>
            <Field label="Location" hint="Optional.">
              <input className={inputClass} value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} placeholder="East gate, level 1" />
            </Field>
            <Field label="Main stream RTSP URL" hint="Detection and HD video. Shown masked; leave it to keep the stored one. An @ in the password is %40.">
              <input className={inputClass} value={form.rtsp_url} onChange={(e) => setForm({ ...form, rtsp_url: e.target.value })} placeholder="rtsp://user:pass@192.168.20.11:554/Streaming/Channels/101" />
            </Field>
            <Field label="Sub-stream RTSP URL" hint="Optional: the grid tiles play it (cheap). Without it they play HD.">
              <input className={inputClass} value={form.rtsp_sub_url} onChange={(e) => setForm({ ...form, rtsp_sub_url: e.target.value })} placeholder="rtsp://user:pass@192.168.20.11:554/Streaming/Channels/102" />
            </Field>
            <Field label="Order" hint="Position in the grid (after the primary camera).">
              <input type="number" min={0} max={999} className={inputClass} value={form.sort_order} onChange={(e) => setForm({ ...form, sort_order: Number(e.target.value) || 0 })} />
            </Field>
            <Field label="Notes">
              <input className={inputClass} value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
            </Field>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setForm(null)}>Cancel</Button>
            <Button onClick={save} disabled={busy || !form.name.trim() || !form.rtsp_url.trim()}>{editing ? "Save" : "Add camera"}</Button>
          </div>
        </Card>
      )}
    </div>
  );
}
