import { ChevronDown, ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

/**
 * A light, nested accordion section for the long setup panels. Its content stays mounted while closed (hidden), so a half-done drawing, a typed number
 * or a placed dot is not lost when the section is folded. `hint` is a short line shown beside the title while closed (a count, a status).
 */
export function Sub({ id, title, hint, open, onToggle, children }: { id: string; title: ReactNode; hint?: ReactNode; open: boolean; onToggle: () => void; children: ReactNode }) {
  return (
    <section className="rounded-md border border-line" data-testid={`sub-${id}`}>
      <button type="button" className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-xs font-semibold tracking-wide text-ink uppercase"
              onClick={onToggle} aria-expanded={open} data-testid={`sub-${id}-toggle`}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>{title}</span>
        {!open && hint ? <span className="ml-auto text-[11px] font-normal tracking-normal text-muted normal-case">{hint}</span> : null}
      </button>
      <div hidden={!open} className="space-y-2 border-t border-line p-2.5" data-testid={`sub-${id}-body`}>{children}</div>
    </section>
  );
}
