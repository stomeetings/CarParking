import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** Class names joined (clsx: strings, arrays, objects, falsy skipped) and de-conflicted (tailwind-merge: the last of two clashing utilities wins). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
