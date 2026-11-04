import type { ReactNode } from "react";
import { BLOCK_HATCH, ENCROACH_HATCH } from "../lib";

export function Card({ title, actions, children, className = "", bodyClass = "p-4" }: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section className={`overflow-hidden rounded-lg border border-line bg-white shadow-sm ${className}`}>
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

export function Pill({ active, onClick, children, title }: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
  title?: string;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors ${
        active ? "border-primary bg-primary text-white" : "border-line bg-white text-muted hover:border-primary hover:text-primary"
      }`}
    >
      {children}
    </button>
  );
}

export function Button({ children, onClick, variant = "primary", disabled, type = "button", title, "data-testid": testid }: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger" | "outline";
  disabled?: boolean;
  type?: "button" | "submit";
  title?: string;
  "data-testid"?: string;
}) {
  const styles = {
    primary: "bg-primary text-white hover:bg-primary-dark",
    ghost: "text-muted hover:bg-body hover:text-ink",
    danger: "bg-danger text-white hover:opacity-90",
    outline: "border border-line bg-white text-ink hover:border-primary hover:text-primary",
  }[variant];
  return (
    <button
      type={type}
      data-testid={testid}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${styles}`}
    >
      {children}
    </button>
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
  "w-full rounded-md border border-line bg-white px-2.5 py-1.5 text-sm text-ink outline-none focus:border-primary focus:ring-2 focus:ring-primary/20";

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

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
            className="flex w-full items-center justify-between gap-3 py-1 text-left text-sm text-ink">
      <span>{label}</span>
      <span className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${checked ? "bg-primary" : "bg-line"}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${checked ? "left-4.5" : "left-0.5"}`} />
      </span>
    </button>
  );
}
