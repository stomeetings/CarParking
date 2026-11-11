import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, CarFront, Cctv, Eye, FileCode2, FlaskConical, History, KeyRound, LayoutDashboard, Menu, Moon, PenTool, Ruler, Settings2, ShieldCheck, Sun } from "lucide-react";
import type { ReactNode } from "react";
import type { Link } from "../hooks";
import { useTheme } from "../theme";
import type { AuthInfo, CameraInfo, StateMessage } from "../types";
import { Badge, IconButton } from "./ui";

export type Page = "dashboard" | "cameras" | "slots" | "setup" | "sample" | "history" | "analytics";

const NAV: { page: Page; label: string; icon: ReactNode }[] = [
  { page: "dashboard", label: "Live dashboard", icon: <LayoutDashboard size={18} /> },
  { page: "cameras", label: "Cameras", icon: <Cctv size={18} /> },
  { page: "slots", label: "Slot editor", icon: <PenTool size={18} /> },
  { page: "setup", label: "Camera setup", icon: <Ruler size={18} /> },
  { page: "sample", label: "Sample test", icon: <FlaskConical size={18} /> },
  { page: "history", label: "History & logs", icon: <History size={18} /> },
  { page: "analytics", label: "Analytics", icon: <BarChart3 size={18} /> },
];

export function Sidebar({ page, open, onNavigate, onClose }: {
  page: Page;
  open: boolean;
  onNavigate: (p: Page) => void;
  onClose: () => void;
}) {
  const body = (
    <aside className="flex h-full w-64 flex-col border-r border-line bg-surface">
      <div className="flex h-14 items-center gap-2.5 border-b border-line px-5">
        <span className="flex items-center justify-center rounded-xl border border-primary/25 bg-primary/15 p-2 text-primary"><CarFront size={20} strokeWidth={1.75} aria-hidden="true" /></span>
        <span className="text-base font-semibold tracking-tight text-ink">Smart Parking</span>
      </div>
      <nav className="flex-1 space-y-0.5 p-3">
        <p className="px-3 pt-2 pb-1 text-[11px] font-bold tracking-wider text-muted uppercase">Monitor</p>
        {NAV.map((n) => (
          <button key={n.page} type="button" onClick={() => onNavigate(n.page)}
                  aria-current={page === n.page ? "page" : undefined}
                  className={`flex w-full items-center gap-3 rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                    page === n.page ? "bg-primary text-white shadow-md shadow-primary/30" : "text-muted hover:bg-body hover:text-ink"}`}>
            {n.icon}
            {n.label}
          </button>
        ))}
        <p className="px-3 pt-5 pb-1 text-[11px] font-bold tracking-wider text-muted uppercase">Tools</p>
        <a href="/docs" target="_blank" rel="noreferrer"
           className="flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-muted hover:bg-body hover:text-ink">
          <FileCode2 size={18} /> API docs
        </a>
      </nav>
    </aside>
  );
  return (
    <>
      <div className="hidden h-full shrink-0 lg:block">{body}</div>
      <AnimatePresence>
        {open && (
          <motion.div className="fixed inset-0 z-40 lg:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="absolute inset-0 bg-black/40" onClick={onClose} />
            <motion.div className="absolute inset-y-0 left-0" initial={{ x: -260 }} animate={{ x: 0 }} exit={{ x: -260 }}
                        transition={{ type: "tween", duration: 0.2 }}>
              {body}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

/** Light / dark switch: remembered in localStorage, the system preference until the first click. */
function ThemeToggle() {
  const [theme, toggle] = useTheme();
  return <IconButton icon={theme === "dark" ? Sun : Moon} label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"} onClick={toggle} data-testid="theme-toggle" />;
}

export function WorkerBadge({ live, link }: { live: StateMessage | null; link: Link }) {
  let tone: "danger" | "success" | "warning" = "danger";
  let dot = "bg-danger";
  let text = "Worker Down";
  let title = live?.heartbeat_age != null ? `last heartbeat ${live.heartbeat_age.toFixed(0)} s ago` : "no worker has run yet";
  if (link !== "open") {
    text = link === "unauthorized" ? "Viewer token needed" : "Backend offline";
    title = "the dashboard can't reach the API";
  } else if (live?.connected && live.snapshot.enabled) {
    tone = "success";
    dot = "bg-success animate-pulse";
    text = "Worker Connected";
    title = `heartbeat ${live.heartbeat_age?.toFixed(1)} s ago · ${live.snapshot.status ?? ""}`;
  } else if (live?.connected) {
    tone = "warning";
    dot = "bg-warning";
    text = "Monitoring off";
    title = "the worker is running but idle: switch monitoring on in Settings";
  }
  return (
    <Badge tone={tone} dot={dot} title={title} data-testid="worker-badge" className="gap-2 px-3 py-1 text-xs">{text}</Badge>
  );
}

export function Header({ title, live, link, onMenu, onSettings, onToken, cameras, cam, onCam, access, onUnlock }: {
  access: AuthInfo | null;
  onUnlock: () => void;
  cameras: CameraInfo[];
  cam: string | null;
  onCam: (id: string) => void;
  title: string;
  live: StateMessage | null;
  link: Link;
  onMenu: () => void;
  onSettings: () => void;
  onToken: () => void;
}) {
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-line bg-surface px-4">
      <IconButton icon={Menu} label="Menu" onClick={onMenu} className="lg:hidden" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-muted">
          Home <span className="mx-1">/</span> <span className="font-semibold text-ink">{title}</span>
        </p>
      </div>
      {cameras.length > 0 && (
        <select value={cam ?? ""} onChange={(e) => onCam(e.target.value)} data-testid="camera-switcher" title="Camera"
                className="max-w-44 rounded-md border border-line bg-surface px-2 py-1.5 text-sm font-medium text-ink focus:border-primary focus:outline-none">
          {cameras.map((c) => <option key={c.id} value={c.id}>{c.display}</option>)}
        </select>
      )}
      <WorkerBadge live={live} link={link} />
      {access?.admin ? (
        <Badge tone="primary" icon={ShieldCheck} title="Admin token accepted: changes allowed" data-testid="access-badge" className="hidden px-2.5 py-1 text-xs md:inline-flex">Admin</Badge>
      ) : access?.view ? (
        <button type="button" onClick={onUnlock} data-testid="access-badge"
                title="Viewer token: view only. Click to enter the admin token for changes."
                className="hidden items-center gap-1 rounded-full bg-body px-2.5 py-1 text-xs font-semibold text-muted ring-1 ring-line hover:text-primary md:inline-flex">
          <Eye size={13} /> View only
        </button>
      ) : null}
      <ThemeToggle />
      <IconButton icon={KeyRound} label="Access tokens" onClick={onToken} />
      <button type="button" onClick={onSettings} title="Settings"
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primary-dark">
        <Settings2 size={16} /> <span className="hidden sm:inline">Settings</span>
      </button>
    </header>
  );
}
