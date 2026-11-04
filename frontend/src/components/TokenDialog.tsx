import { AnimatePresence, motion } from "framer-motion";
import { KeyRound, X } from "lucide-react";
import { useEffect, useState } from "react";
import { sanitizeTokens, tokens } from "../api";
import type { AuthInfo } from "../types";
import { Button, Field, inputClass } from "./ui";

/** What the server says a token allows (GET /api/auth with it). */
export async function tokenAccess(token: string, fetcher: typeof fetch = fetch): Promise<AuthInfo> {
  const r = await fetcher("/api/auth", { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!r.ok) throw new Error(`the server answered ${r.status}`);
  return r.json();
}

/** Problems with the entered tokens, or {} when the ones asked for are accepted. The classic mix-up: the viewer
 * token pasted where the admin token is needed (it only allows viewing). */
export async function checkTokens(kind: "view" | "admin" | "both", view: string, admin: string,
                                  fetcher: typeof fetch = fetch): Promise<{ view?: string; admin?: string }> {
  const out: { view?: string; admin?: string } = {};
  if (kind !== "admin") {
    const a = await tokenAccess(view, fetcher);
    if (!a.view) out.view = view ? "Not accepted: copy CARPARK_VIEW_TOKEN from the server's .env." : "Required.";
  }
  if (kind === "admin" && !admin) out.admin = "Required: CARPARK_API_TOKEN from the server's .env.";
  else if (kind !== "view" && admin) {
    const a = await tokenAccess(admin, fetcher);
    if (!a.writes_enabled) out.admin = "The server has no admin token set (CARPARK_API_TOKEN): changes are disabled.";
    else if (!a.admin) {
      out.admin = a.view && a.view_required
        ? "This is the viewer token: it only allows viewing. Changes need the admin token (CARPARK_API_TOKEN in .env, not CARPARK_VIEW_TOKEN)."
        : "Not accepted: copy CARPARK_API_TOKEN from the server's .env.";
    }
  }
  return out;
}

/** Enter / change the viewer and admin tokens (kept in this browser), checked with the server before saving.
 * kind: which one is asked for right now. */
export function TokenDialog({ open, kind, reason, onClose, onSaved }: {
  open: boolean;
  kind: "view" | "admin" | "both";
  reason?: string;
  onClose?: () => void;
  onSaved: () => void;
}) {
  const [view, setView] = useState(tokens.view);
  const [admin, setAdmin] = useState(tokens.admin);
  const [errors, setErrors] = useState<{ view?: string; admin?: string; server?: string }>({});
  const [checking, setChecking] = useState(false);
  useEffect(() => {
    if (open) {
      setView(tokens.view);
      setAdmin(tokens.admin);
      setErrors({});
    }
  }, [open]);

  const save = async () => {
    const v = view.trim();
    const a = admin.trim();
    setChecking(true);
    try {
      const found = await checkTokens(kind, v, a);
      setErrors(found);
      if (!found.view) tokens.setView(v); // keep what is right even if the other one is wrong
      if (!found.admin) tokens.setAdmin(a);
      await sanitizeTokens(); // also drops a stored admin token the server rejects (it would shadow the viewer token)
      if (!found.view && !found.admin) onSaved();
    } catch (e) {
      setErrors({ server: `Couldn't check the tokens: ${(e as Error).message}` });
    } finally {
      setChecking(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-50 flex items-center justify-center p-4" initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <div className="absolute inset-0 bg-ink/40" onClick={onClose} />
          <motion.form initial={{ y: 16, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 16, opacity: 0 }}
                       onSubmit={(e) => (e.preventDefault(), save())} data-testid="token-dialog"
                       className="relative w-full max-w-md rounded-lg border border-line bg-white p-5 shadow-xl">
            <div className="mb-4 flex items-center gap-2">
              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-primary"><KeyRound size={18} /></span>
              <h2 className="flex-1 text-base font-semibold">Access</h2>
              {onClose && (
                <button type="button" onClick={onClose} className="rounded p-1 text-muted hover:bg-body" aria-label="Close"><X size={18} /></button>
              )}
            </div>
            {reason && <p className="mb-4 rounded-md bg-warning/15 px-3 py-2 text-sm text-ink">{reason}</p>}
            <div className="space-y-3">
              {kind !== "admin" && (
                <Field label="Viewer token" hint="Lets this browser see the cameras, status and history (CARPARK_VIEW_TOKEN). View only.">
                  <input className={`${inputClass} ${errors.view ? "border-danger" : ""}`} type="password" value={view}
                         onChange={(e) => setView(e.target.value)} autoFocus data-testid="view-token" />
                  {errors.view && <span className="mt-1 block text-xs text-danger">{errors.view}</span>}
                </Field>
              )}
              {kind !== "view" && (
                <Field label="Admin token" hint="Needed to change anything: camera names, settings, slots, monitoring (CARPARK_API_TOKEN). Also allows viewing.">
                  <input className={`${inputClass} ${errors.admin ? "border-danger" : ""}`} type="password" value={admin}
                         onChange={(e) => setAdmin(e.target.value)} autoFocus={kind === "admin"} data-testid="admin-token" />
                  {errors.admin && <span className="mt-1 block text-xs text-danger" data-testid="admin-token-error">{errors.admin}</span>}
                </Field>
              )}
            </div>
            {errors.server && <p className="mt-3 text-xs text-danger">{errors.server}</p>}
            <p className="mt-3 text-xs text-muted">Checked with the server, then stored in this browser only. Both tokens are in the server's .env file.</p>
            <div className="mt-5 flex justify-end gap-2">
              {onClose && <Button variant="ghost" onClick={onClose}>Cancel</Button>}
              <Button type="submit" disabled={checking}>{checking ? "Checking…" : "Save"}</Button>
            </div>
          </motion.form>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
