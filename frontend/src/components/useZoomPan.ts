import { type RefObject, useCallback, useEffect, useState } from "react";

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 12;

/** Zoom and pan for a picture shown in an `overflow-hidden` box (`outer`) with the picture and its drawing layer inside one inner element that gets
 * `transform: translate(x px, y px) scale(s)` from `transform(view)`. The wheel zooms about the pointer; `panBy` moves the picture, never off the box.
 * Pointer positions are read from the inner element's getBoundingClientRect, which already includes the zoom and pan, so the shares of the picture stay right.
 * `active`: false while `outer` is not mounted yet (the wheel listener is attached once it is). */
export function useZoomPan(outer: RefObject<HTMLElement | null>, active = true) {
  const [view, setView] = useState({ s: 1, x: 0, y: 0 });

  const zoomAt = useCallback((factor: number, cx: number, cy: number) => {
    setView((v) => {
      const s = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.s * factor));
      if (s === v.s) return v;
      const k = s / v.s;
      const box = outer.current?.getBoundingClientRect();
      const [bw, bh] = box ? [box.width, box.height] : [0, 0];
      const x = Math.min(0, Math.max(bw - bw * s, cx - (cx - v.x) * k)); // the picture never slides off the box
      const y = Math.min(0, Math.max(bh - bh * s, cy - (cy - v.y) * k));
      return { s, x: s === MIN_ZOOM ? 0 : x, y: s === MIN_ZOOM ? 0 : y };
    });
  }, [outer]);

  const zoomCentre = (factor: number) => {
    const r = outer.current?.getBoundingClientRect();
    zoomAt(factor, (r?.width ?? 0) / 2, (r?.height ?? 0) / 2);
  };

  const panBy = (dx: number, dy: number) => setView((v) => {
    const box = outer.current?.getBoundingClientRect();
    const [bw, bh] = box ? [box.width, box.height] : [0, 0];
    return { ...v, x: Math.min(0, Math.max(bw - bw * v.s, v.x + dx)), y: Math.min(0, Math.max(bh - bh * v.s, v.y + dy)) };
  });

  useEffect(() => { // the wheel must not scroll the page while it zooms the picture (a non-passive listener)
    const el = outer.current;
    if (!el || !active) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(e.deltaY < 0 ? 1.25 : 0.8, e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, [outer, zoomAt, active]);

  return {
    view, zoomCentre, panBy,
    fit: () => setView({ s: 1, x: 0, y: 0 }),
    transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})`,
  };
}
