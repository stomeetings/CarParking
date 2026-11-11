import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Sidebar } from "./components/Layout";
import { initialTheme, THEME_KEY, useTheme } from "./theme";

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
    act(() => screen.getByTestId("probe").click());
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(localStorage.getItem(THEME_KEY)).toBe("dark");
    act(() => screen.getByTestId("probe").click());
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(localStorage.getItem(THEME_KEY)).toBe("light");
  });

  it("marks the active sidebar item as the current page", () => {
    render(<Sidebar page="cameras" open={false} onNavigate={() => {}} onClose={() => {}} />);
    expect(screen.getByRole("button", { name: "Cameras" }).getAttribute("aria-current")).toBe("page");
    expect(screen.getByRole("button", { name: "Analytics" }).getAttribute("aria-current")).toBeNull();
  });
});
