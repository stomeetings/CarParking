import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Sidebar } from "./components/Layout";
import { initialTheme, nextTheme, THEME_KEY, useTheme } from "./theme";

function Probe() {
  const [t, toggle] = useTheme();
  return <button data-testid="probe" onClick={toggle}>{t}</button>;
}

beforeEach(() => { localStorage.clear(); document.documentElement.classList.remove("dark"); });
afterEach(cleanup);

describe("theme", () => {
  it("uses the saved choice, else light when there is no system preference", () => {
    expect(initialTheme()).toBe("light");
    localStorage.setItem(THEME_KEY, "dark");
    expect(initialTheme()).toBe("dark");
  });

  it("toggling flips the class on <html> and is remembered", () => {
    render(<Probe />);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    const root = document.documentElement.classList;
    act(() => screen.getByTestId("probe").click()); // light -> semi
    expect([root.contains("semi"), root.contains("dark"), localStorage.getItem(THEME_KEY)]).toEqual([true, false, "semi"]);
    act(() => screen.getByTestId("probe").click()); // semi -> dark
    expect([root.contains("semi"), root.contains("dark"), localStorage.getItem(THEME_KEY)]).toEqual([false, true, "dark"]);
    act(() => screen.getByTestId("probe").click()); // dark -> light
    expect([root.contains("semi"), root.contains("dark"), localStorage.getItem(THEME_KEY)]).toEqual([false, false, "light"]);
  });

  it("restores a saved semi theme and cycles light -> semi -> dark -> light", () => {
    localStorage.setItem(THEME_KEY, "semi");
    expect(initialTheme()).toBe("semi");
    expect([nextTheme("light"), nextTheme("semi"), nextTheme("dark")]).toEqual(["semi", "dark", "light"]);
  });

  it("marks the active sidebar item as the current page", () => {
    render(<Sidebar page="cameras" open={false} onNavigate={() => {}} onClose={() => {}} />);
    expect(screen.getByRole("button", { name: "Cameras" }).getAttribute("aria-current")).toBe("page");
    expect(screen.getByRole("button", { name: "Analytics" }).getAttribute("aria-current")).toBeNull();
  });
});
