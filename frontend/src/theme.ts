import { useCallback, useEffect, useState } from "react";

/** "semi": a dark sidebar (command-center look) next to a light workspace. */
export type Theme = "light" | "semi" | "dark";
export const THEMES: readonly Theme[] = ["light", "semi", "dark"];
export const THEME_KEY = "carparking.theme";

/** The saved choice, else the system preference (prefers-color-scheme), else light. */
export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "semi" || saved === "dark") return saved;
  } catch { /* storage blocked: fall through */ }
  return typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** The next theme in the cycle light -> semi -> dark -> light. */
export const nextTheme = (t: Theme): Theme => THEMES[(THEMES.indexOf(t) + 1) % THEMES.length];

/** Puts the theme on <html>: the `dark` class (everything) or the `semi` class (the sidebar only). index.html does the same before first paint. */
export function applyTheme(t: Theme): void {
  document.documentElement.classList.toggle("dark", t === "dark");
  document.documentElement.classList.toggle("semi", t === "semi");
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => applyTheme(theme), [theme]);
  const toggle = useCallback(() => {
    setTheme((t) => {
      const next = nextTheme(t);
      try { localStorage.setItem(THEME_KEY, next); } catch { /* not persisted */ }
      return next;
    });
  }, []);
  return [theme, toggle];
}
