import { Maximize, Minimize } from "lucide-react";
import { type CSSProperties, useEffect, useState } from "react";

/** Native fullscreen of one element (the Fullscreen API): only the camera picture and its overlay fill the monitor,
 * no sidebar, header or panels. Esc (the browser's) or the button leaves it. Where element fullscreen doesn't exist
 * (iPhone Safari) the bare <video> goes fullscreen instead (webkitEnterFullscreen, without the overlay). */
export function useFullscreen(el: React.RefObject<HTMLElement | null>, video?: React.RefObject<HTMLVideoElement | null>) {
  const [active, setActive] = useState(false);
  useEffect(() => {
    const on = () => setActive(!!el.current && document.fullscreenElement === el.current);
    document.addEventListener("fullscreenchange", on);
    return () => document.removeEventListener("fullscreenchange", on);
  }, [el]);
  const toggle = async () => {
    if (document.fullscreenElement) return void (await document.exitFullscreen().catch(() => {}));
    const node = el.current;
    if (node?.requestFullscreen) return void (await node.requestFullscreen({ navigationUI: "hide" }).catch(() => {}));
    const v = video?.current as (HTMLVideoElement & { webkitEnterFullscreen?: () => void }) | null | undefined;
    v?.webkitEnterFullscreen?.();
  };
  return { active, toggle };
}

/** The picture's box in fullscreen: as large as the screen allows at the camera's aspect ratio (no stretching, so
 * the overlay stays aligned), centred on black. Outside fullscreen: the normal full-width box. */
export function stageStyle(active: boolean, w: number, h: number): CSSProperties {
  return active
    ? { aspectRatio: `${w} / ${h}`, width: `min(100vw, calc(100vh * ${w} / ${h}))` }
    : { aspectRatio: `${w} / ${h}` };
}

export function FullscreenButton({ active, onClick, className = "" }: { active: boolean; onClick: () => void; className?: string }) {
  return (
    <button type="button" data-testid="video-fullscreen" aria-label={active ? "Exit video fullscreen" : "Video fullscreen"}
            title={active ? "Exit fullscreen (Esc)" : "Video fullscreen: only this camera fills the screen"}
            onClick={(e) => (e.stopPropagation(), onClick())} onDoubleClick={(e) => e.stopPropagation()}
            className={`rounded-md bg-black/55 p-1.5 text-white backdrop-blur-sm transition hover:bg-black/80 ${className}`}>
      {active ? <Minimize size={16} /> : <Maximize size={16} />}
    </button>
  );
}
