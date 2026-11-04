import { ArrowLeft, Expand, Grid2x2, Grid3x3, Square } from "lucide-react";
import { type ReactNode, Suspense, lazy, useCallback, useEffect, useState } from "react";
import { sanitizeTokens, saveCam, savedCam, tokens } from "./api";
import { useHeld, useLive, usePoll, useSlotFilter } from "./hooks";
import { CameraSetup } from "./components/CameraSetup";
import { CamerasPage } from "./components/Cameras";
import { CameraGrid, type GridSize } from "./components/Grid";
import { Header, type Page, Sidebar } from "./components/Layout";
import { LiveView } from "./components/LiveView";
import { MetricCard } from "./components/Metrics";
import { MonitorBanner, monitorState } from "./components/Monitor";
import { RefreshData, SyncStatus, saveSyncEvery, savedSyncEvery } from "./components/Sync";
import { HistoryPage } from "./components/Pages";
import { SettingsDrawer } from "./components/SettingsDrawer";
import { SampleSession } from "./components/SampleSession";
import { SlotEditor } from "./components/SlotEditor";
import { Counters, EventTicker, StatusCard } from "./components/SidePanel";
import { TokenDialog } from "./components/TokenDialog";
import type { AuthInfo, CameraInfo } from "./types";

const AnalyticsPage = lazy(() => import("./components/Analytics"));

const TITLES: Record<Page, string> = {
  dashboard: "Live dashboard", cameras: "Cameras", slots: "Slot editor", setup: "Camera setup", sample: "Sample image test", history: "History & logs", analytics: "Analytics",
};
const PAGES = Object.keys(TITLES) as Page[];

const pageFromHash = (): Page => {
  const p = window.location.hash.replace(/^#\/?/, "") as Page;
  return PAGES.includes(p) ? p : "dashboard";
};

const GRID_KEY = "carparking.grid";
const savedGrid = (): GridSize | null => {
  try {
    const v = Number(localStorage.getItem(GRID_KEY));
    return v === 1 || v === 2 || v === 3 ? v : null;
  } catch {
    return null;
  }
};

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash);
  const [menu, setMenu] = useState(false);
  const [settings, setSettings] = useState(false);
  const [tokenKey, setTokenKey] = useState(tokens.view + tokens.admin); // changes when the tokens do: refetch
  const [dialog, setDialog] = useState<{ kind: "view" | "admin" | "both"; reason?: string; required?: boolean; retry?: () => void } | null>(null);
  const [access, setAccess] = useState<AuthInfo | null>(null);
  const [frames, setFrames] = useState(false);
  const cams = usePoll<CameraInfo[]>("/api/cameras", 5000, tokenKey);
  const cameras = cams.data ?? [];
  const [chosen, setChosen] = useState(savedCam);
  const cam = cameras.some((c) => c.id === chosen) ? chosen : (cameras.find((c) => c.primary)?.id ?? null);
  const camera = cameras.find((c) => c.id === cam);
  const [gridPref, setGridPref] = useState<GridSize | null>(savedGrid);
  const grid: GridSize = gridPref ?? (cameras.length > 1 ? 2 : 1);
  const [gridPage, setGridPage] = useState(0);
  const [fullscreen, setFullscreen] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0); // Refresh data: reconnect + refetch in the background, never a page reload
  const [every, setEvery] = useState(savedSyncEvery);
  const [slotFilter, setSlotFilter] = useSlotFilter(cam); // the counter cards' filter for the video overlay: this camera's own, the viewer's
  const live = useLive(frames, tokenKey, cam, true, refresh, every);

  useEffect(() => {
    const on = () => setPage(pageFromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  // a viewer token the server wants and this browser lacks: ask for it (the WebSocket closes with 4401 too)
  useEffect(() => {
    sanitizeTokens().then((a) => {
      setAccess(a);
      const now = tokens.view + tokens.admin;
      if (now !== tokenKey) setTokenKey(now); // a rejected stored token was dropped: refetch with the good one
      if (!a.view) setDialog({ kind: "view", reason: "This dashboard needs the viewer token.", required: true });
      else setDialog((d) => (d?.required ? null : d)); // access is fine now: a "token needed" prompt from a stale request goes
    }).catch(() => {});
  }, [tokenKey]);
  useEffect(() => {
    // the live connection was refused: re-check the stored tokens first (a rejected one is dropped and the connection
    // retried with the good one); only ask when no stored token allows viewing
    if (live.link !== "unauthorized") return;
    sanitizeTokens().then((a) => {
      if (a.view) setTokenKey(tokens.view + tokens.admin + "#" + Date.now());
      else setDialog({ kind: "view", reason: "The viewer token is missing or wrong.", required: true });
    }).catch(() => {});
  }, [live.link]);
  // Esc leaves the fullscreen camera (back to the grid)
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && fullscreen && setFullscreen(null);
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [fullscreen]);

  const navigate = (p: Page) => {
    window.location.hash = `/${p}`;
    setMenu(false);
  };
  const selectCam = (id: string) => {
    setChosen(id);
    saveCam(id);
  };
  const openCam = (id: string) => {
    selectCam(id);
    setFullscreen(id);
    if (page !== "dashboard") navigate("dashboard");
  };
  const setGrid = (g: GridSize) => {
    setGridPref(g);
    setFullscreen(null);
    try {
      localStorage.setItem(GRID_KEY, String(g));
    } catch {
      /* not remembered */
    }
  };
  const onNeedFrames = useCallback((need: boolean) => setFrames(need), []);
  const refreshData = () => {
    setRefresh((n) => n + 1);
    cams.reload();
  };
  const needAdmin = (reason: string, retry?: () => void) => setDialog({ kind: "admin", reason, retry });
  const snap = live.state?.snapshot;
  const single = grid === 1 || fullscreen != null;
  const monitor = monitorState(live.state, live.link).kind;
  // the cards keep their last counts through a model reload / heartbeat gap (useHeld), not when monitoring goes off
  const counted = useHeld(live.state?.connected && snap?.counts ? snap : undefined, `${cam}|${monitor === "off"}`);

  return (
    <div className="flex h-full">
      <Sidebar page={page} open={menu} onNavigate={navigate} onClose={() => setMenu(false)} />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header title={TITLES[page]} live={live.state} link={live.link} onMenu={() => setMenu(true)}
                onSettings={() => setSettings(true)} onToken={() => setDialog({ kind: "both" })}
                cameras={cameras} cam={cam} onCam={selectCam} access={access}
                onUnlock={() => setDialog({ kind: "admin", reason: "Enter the admin token to change cameras, settings and slots." })} />
        <main className="flex-1 overflow-y-auto p-4 lg:p-6">
          {page === "dashboard" && (
            <div className="space-y-4">
              {cams.error && !cameras.length && (
                <p className="rounded-md bg-danger/10 px-3 py-2 text-sm text-danger" data-testid="cameras-error">
                  Couldn't load the cameras: {cams.error}
                </p>
              )}
              <div className="flex flex-wrap items-center gap-2" data-testid="view-bar">
                {fullscreen ? (
                  <button type="button" onClick={() => setFullscreen(null)} data-testid="back-to-grid"
                          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primary-dark">
                    <ArrowLeft size={15} /> Back to grid <span className="text-white/70">(Esc · double-click)</span>
                  </button>
                ) : (
                  <span className="text-xs font-semibold tracking-wide text-muted uppercase">View</span>
                )}
                <div className="flex overflow-hidden rounded-md border border-line bg-white">
                  {([[1, <Square size={14} />, "1×1"], [2, <Grid2x2 size={14} />, "2×2"], [3, <Grid3x3 size={14} />, "3×3"]] as [GridSize, ReactNode, string][]).map(([g, icon, label]) => (
                    <button key={g} type="button" onClick={() => setGrid(g)} data-testid={`grid-${g}`}
                            className={`inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-semibold ${grid === g && !fullscreen ? "bg-primary text-white" : "text-muted hover:text-primary"}`}>
                      {icon} {label}
                    </button>
                  ))}
                </div>
                {single && (
                  <button type="button" title="Browser full screen" onClick={() => document.documentElement.requestFullscreen?.().catch(() => {})}
                          className="inline-flex items-center gap-1 rounded-md border border-line bg-white px-2.5 py-1.5 text-xs font-semibold text-muted hover:text-primary">
                    <Expand size={14} /> Full screen
                  </button>
                )}
                {!single && <span className="text-xs text-muted">Click: select · double-click / Enter: open in HD · arrows: move · PgUp/PgDn: page</span>}
                <div className="ml-auto flex items-center gap-2">
                  <SyncStatus at={live.at} link={live.link} every={every} onEvery={(s) => (setEvery(s), saveSyncEvery(s))} />
                  <RefreshData onRefresh={refreshData} at={live.at} />
                </div>
              </div>
              {single ? (
                <div className="grid gap-4 lg:grid-cols-10">
                  <div className="space-y-4 lg:col-span-7">
                    <MonitorBanner live={live.state} link={live.link} onNeedAdmin={needAdmin} cam={cam} />
                    <LiveView snapshot={snap} frameUrl={live.frameUrl} onNeedFrames={onNeedFrames} camera={camera} tokenKey={tokenKey} refresh={refresh}
                              slotFilter={slotFilter} onClearFilter={() => setSlotFilter(null)}
                              onExit={fullscreen ? () => setFullscreen(null) : undefined} />
                    {snap?.det_text && <p className="text-xs text-muted">{snap.det_text} · slot map: {snap.scope}</p>}
                  </div>
                  <div className="space-y-4 lg:col-span-3">
                    <Counters snapshot={counted} filter={slotFilter} onFilter={setSlotFilter}
                              note={{ ok: undefined, starting: "Starting: no counts yet", off: "Monitoring off: no FREE / BUSY states",
                                      down: "Worker down: no live counts", offline: "Backend offline" }[monitor]} />
                    <MetricCard geometry={snap?.geometry} />
                    <EventTicker events={live.events} />
                    <StatusCard snapshot={snap} heartbeat={live.state?.heartbeat_age} />
                  </div>
                </div>
              ) : (
                <CameraGrid cameras={cameras} size={grid} page={gridPage} onPage={setGridPage} selected={cam} refresh={refresh} every={every}
                            tokenKey={tokenKey} onSelect={selectCam} onOpen={openCam} />
              )}
            </div>
          )}
          {page === "cameras" && (
            <CamerasPage cameras={cameras} error={cams.error} onChanged={cams.reload} onNeedAdmin={needAdmin} onOpen={openCam}
                         onDraw={(id) => (selectCam(id), navigate("slots"))} />
          )}
          {page === "slots" && (
            <SlotEditor snapshot={snap} tokenKey={tokenKey} live={live.state} onNeedAdmin={needAdmin}
                        cam={cam} cameras={cameras} onCam={selectCam} />
          )}
          {page === "setup" && <CameraSetup cam={cam} cameras={cameras} onCam={selectCam} tokenKey={tokenKey} onNeedAdmin={needAdmin} />}
          {page === "sample" && <SampleSession cam={cam} tokenKey={tokenKey} onNeedAdmin={needAdmin} />}
          {page === "history" && <HistoryPage tokenKey={tokenKey} cam={cam} />}
          {page === "analytics" && (
            <Suspense fallback={<p className="text-sm text-muted">Loading charts…</p>}>
              <AnalyticsPage tokenKey={tokenKey} snapshot={snap} cam={cam} />
            </Suspense>
          )}
        </main>
      </div>
      <SettingsDrawer open={settings} onClose={() => setSettings(false)} tokenKey={tokenKey} codeStale={false} model={snap?.model}
                      onNeedAdmin={needAdmin} cam={cam} />
      <TokenDialog open={!!dialog} kind={dialog?.kind ?? "both"} reason={dialog?.reason}
                   onClose={dialog?.required ? undefined : () => setDialog(null)}
                   onSaved={() => {
                     const retry = dialog?.retry;
                     setDialog(null);
                     setTokenKey(tokens.view + tokens.admin);
                     retry?.(); // the change that asked for the admin token, now with it
                   }} />
    </div>
  );
}
