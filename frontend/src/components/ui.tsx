import { cva, type VariantProps } from "class-variance-authority";
import type { LucideIcon } from "lucide-react";
import type { MouseEvent, ReactNode } from "react";
import { cn } from "../cn";
import { BLOCK_HATCH, ENCROACH_HATCH } from "../lib";

export function Card({ title, actions, children, className = "", bodyClass = "p-4" }: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section className={cn("overflow-hidden rounded-lg border border-line bg-surface shadow-sm", className)}>
      {title !== undefined && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {actions && <div className="flex flex-wrap items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

const pill = cva("inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors", {
  variants: {
    active: {
      true: "border-primary bg-primary text-white",
      false: "border-line bg-surface text-muted hover:border-primary hover:text-primary",
    },
  },
});

export function Pill({ active, onClick, children, title }: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
  title?: string;
}) {
  return (
    <button type="button" title={title} onClick={onClick} className={pill({ active })}>
      {children}
    </button>
  );
}

const button = cva(
  "inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
  {
    variants: {
      variant: {
        primary: "bg-primary text-white hover:bg-primary-dark",
        ghost: "text-muted hover:bg-body hover:text-ink",
        danger: "bg-danger text-white hover:opacity-90",
        outline: "border border-line bg-surface text-ink hover:border-primary hover:text-primary",
      },
    },
    defaultVariants: { variant: "primary" },
  },
);

export function Button({ children, onClick, variant = "primary", disabled, type = "button", title, "data-testid": testid }: {
  children: ReactNode;
  onClick?: () => void;
  variant?: NonNullable<VariantProps<typeof button>["variant"]>;
  disabled?: boolean;
  type?: "button" | "submit";
  title?: string;
  "data-testid"?: string;
}) {
  return (
    <button type={type} data-testid={testid} title={title} onClick={onClick} disabled={disabled} className={button({ variant })}>
      {children}
    </button>
  );
}

/** One icon size scale for the whole app (Lucide, px): inline in text, in toolbars, in navigation. */
export const ICON_SIZE = { inline: 14, toolbar: 16, nav: 18 } as const;
export const ICON_STROKE = 1.75;

const iconButton = cva(
  "inline-flex shrink-0 items-center justify-center rounded-md transition-colors focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50",
  {
    variants: {
      variant: {
        ghost: "text-muted hover:bg-body hover:text-ink",
        outline: "border border-line bg-surface text-ink hover:border-primary hover:text-primary",
        primary: "bg-primary text-white hover:bg-primary-dark",
        danger: "text-danger hover:bg-danger/10",
      },
      size: { sm: "h-7 w-7", md: "h-8 w-8" },
      active: { true: "bg-primary/10 text-primary", false: "" },
    },
    defaultVariants: { variant: "ghost", size: "md", active: false },
  },
);

/** A button that is only a Lucide icon. `label` is required: it is the accessible name (aria-label) and the tooltip. */
export function IconButton({ icon: Icon, label, onClick, variant = "ghost", size = "md", active = false, disabled, type = "button", className, "data-testid": testid }: {
  icon: LucideIcon;
  label: string;
  onClick?: (e: MouseEvent<HTMLButtonElement>) => void;
  className?: string;
  variant?: NonNullable<VariantProps<typeof iconButton>["variant"]>;
  size?: NonNullable<VariantProps<typeof iconButton>["size"]>;
  active?: boolean;
  disabled?: boolean;
  type?: "button" | "submit";
  "data-testid"?: string;
}) {
  return (
    <button type={type} aria-label={label} title={label} aria-pressed={active || undefined} data-testid={testid} onClick={onClick} disabled={disabled}
            className={cn(iconButton({ variant, size, active }), className)}>
      <Icon size={size === "sm" ? ICON_SIZE.inline : ICON_SIZE.toolbar} strokeWidth={ICON_STROKE} aria-hidden="true" />
    </button>
  );
}

const badge = cva("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ring-inset", {
  variants: {
    tone: {
      neutral: "bg-body text-muted ring-line",
      primary: "bg-primary/10 text-primary ring-primary/30",
      success: "bg-success/15 text-success-ink ring-success/30",
      warning: "bg-warning/15 text-warning-ink ring-warning/40",
      danger: "bg-danger/10 text-danger ring-danger/30",
      info: "bg-info/10 text-info ring-info/30",
    },
  },
  defaultVariants: { tone: "neutral" },
});

/** A small status chip: a tone (colour set) and a label, with an optional leading Lucide icon. */
export function Badge({ tone = "neutral", icon: Icon, dot, title, children, className, "data-testid": testid }: {
  tone?: NonNullable<VariantProps<typeof badge>["tone"]>;
  icon?: LucideIcon;
  dot?: string; // a leading status dot: its colour classes (e.g. "bg-success animate-pulse")
  title?: string;
  children: ReactNode;
  className?: string;
  "data-testid"?: string;
}) {
  return (
    <span className={cn(badge({ tone }), className)} title={title} data-testid={testid}>
      {dot && <span className={cn("h-2 w-2 rounded-full", dot)} />}
      {Icon && <Icon size={ICON_SIZE.inline - 2} strokeWidth={ICON_STROKE} aria-hidden="true" />}
      {children}
    </span>
  );
}

const STATUS_TONE = { ok: "success", warn: "warning", bad: "danger", info: "info", active: "primary", idle: "neutral" } as const;
export type Status = keyof typeof STATUS_TONE;

/** The one status chip: a status word (ok / warn / bad / info / active / idle) instead of a colour set, so the same state always looks the same. */
export function StatusBadge({ status, children, title, className, "data-testid": testid, ...rest }: {
  status: Status;
  children: ReactNode;
  title?: string;
  className?: string;
  "data-testid"?: string;
  [k: `data-${string}`]: string | undefined;
}) {
  return (
    <span className={cn(badge({ tone: STATUS_TONE[status] }), "px-2.5 py-1 text-xs", className)} title={title} data-testid={testid} {...rest}>
      {children}
    </span>
  );
}

/** A KPI tile: small caps label, a big tabular number, an optional hint line. */
export function MetricCard({ label, value, hint, "data-testid": testid }: { label: string; value: ReactNode; hint?: ReactNode; "data-testid"?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface p-3 shadow-sm" data-testid={testid}>
      <div className="text-xs font-semibold tracking-wide text-muted uppercase">{label}</div>
      <div className="mt-1 text-xl font-semibold tabular-nums">{value}</div>
      {hint && <div className="text-xs text-muted">{hint}</div>}
    </div>
  );
}

/** The frame around a picture / video / canvas: a bordered card with a header (title, toolbar `actions`) and an optional `legend` strip below.
 * The children are the drawing surface and are passed through untouched; the frame adds no padding, state or handlers of its own. */
export function CanvasFrame({ title, actions, legend, className, children }: {
  title: ReactNode;
  actions?: ReactNode;
  legend?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <Card title={title} actions={actions} bodyClass="p-0" className={className}>
      {children}
      {legend && <div className="flex flex-wrap items-center gap-3 border-t border-line px-4 py-2 text-xs text-muted" data-testid="canvas-legend">{legend}</div>}
    </Card>
  );
}

export function Field({ label, hint, tip, children }: { label: ReactNode; hint?: ReactNode; tip?: string; children: ReactNode }) {
  return (
    <label className="block" title={tip}>
      <span className="mb-1 block text-xs font-semibold tracking-wide text-muted uppercase">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm text-ink outline-none focus:border-primary focus:ring-2 focus:ring-primary/20";

export function Slider({ value, min, max, step, onChange, format = (v) => String(v) }: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-3">
      <input type="range" className="w-full" min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(Number(e.target.value))} />
      <span className="w-16 shrink-0 text-right text-sm tabular-nums text-ink">{format(value)}</span>
    </div>
  );
}

/** The hatched fill of an ENCROACHED bay, to be put once inside an overlay <svg> (user-space units: `size` is the stripe pitch). */
export function EncroachHatch({ size }: { size: number }) {
  return (
    <defs>
      <pattern id={ENCROACH_HATCH} width={size} height={size} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width={size} height={size} fill="rgba(240,140,0,0.10)" />
        <line x1={0} y1={0} x2={0} y2={size} stroke="#f08c00" strokeWidth={size / 3} opacity={0.6} />
      </pattern>
    </defs>
  );
}

/** The cross-hatched fill of a BLOCKED bay (an object stands in it), once inside an overlay <svg> like <EncroachHatch />. */
export function BlockHatch({ size }: { size: number }) {
  return (
    <defs>
      <pattern id={BLOCK_HATCH} width={size} height={size} patternUnits="userSpaceOnUse">
        <rect width={size} height={size} fill="rgba(91,100,117,0.12)" />
        <line x1={0} y1={0} x2={size} y2={size} stroke="#5b6475" strokeWidth={size / 6} opacity={0.7} />
        <line x1={size} y1={0} x2={0} y2={size} stroke="#5b6475" strokeWidth={size / 6} opacity={0.7} />
      </pattern>
    </defs>
  );
}

const toggleTrack = cva("relative h-5 w-9 shrink-0 rounded-full transition-colors", { variants: { checked: { true: "bg-primary", false: "bg-line" } } });
const toggleKnob = cva("absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all", { variants: { checked: { true: "left-4.5", false: "left-0.5" } } });

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
            className="flex w-full items-center justify-between gap-3 py-1 text-left text-sm text-ink">
      <span>{label}</span>
      <span className={toggleTrack({ checked })}>
        <span className={toggleKnob({ checked })} />
      </span>
    </button>
  );
}
