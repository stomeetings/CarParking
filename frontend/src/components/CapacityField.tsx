import { MAX_CAPACITY } from "../slotdraw";

/** The vehicles-per-slot input of a slot / pin row: 1 (empty) is a normal bay, N makes it N spaces for a pin that holds
 * several vehicles (a truck and a car in depth). The total counts N, BUSY counts the vehicles whose boxes stand on it. */
export function CapacityField({ value, onChange }: { value: number | undefined; onChange: (text: string) => void }) {
  return (
    <label className="flex shrink-0 items-center gap-1 text-xs text-muted" title={`Vehicles this slot holds (1 to ${MAX_CAPACITY}). A big pin with 2 or more vehicles: set it here instead of drawing split lines.`}>
      cap
      <input type="number" min={1} max={MAX_CAPACITY} step={1} placeholder="1" value={value ?? ""} data-testid="capacity"
             className="w-12 rounded border border-line bg-transparent px-1 py-0.5 text-sm text-ink focus:border-primary focus:outline-none"
             onClick={(e) => e.stopPropagation()} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}
