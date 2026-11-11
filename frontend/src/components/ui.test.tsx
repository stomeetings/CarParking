import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Eye, Trash2 } from "lucide-react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cn } from "../cn";
import { Badge, Button, CanvasFrame, Card, IconButton, MetricCard, Pill, StatusBadge, Toggle } from "./ui";

afterEach(cleanup);

describe("cn", () => {
  it("joins classes, skips falsy values and lets the last clashing utility win", () => {
    expect(cn("px-2", false && "x", undefined, "py-1")).toBe("px-2 py-1");
    expect(cn("bg-surface p-4", "bg-primary p-2")).toBe("bg-primary p-2");
  });
});

describe("primitives keep their behaviour and hooks", () => {
  it("Button: variant classes, disabled, title and data-testid", () => {
    const click = vi.fn();
    render(<Button onClick={click} variant="outline" title="tip" data-testid="b">Go</Button>);
    const b = screen.getByTestId("b");
    expect(b.className).toContain("bg-surface");
    expect(b.getAttribute("title")).toBe("tip");
    fireEvent.click(b);
    expect(click).toHaveBeenCalledTimes(1);
    cleanup();
    render(<Button disabled data-testid="d">No</Button>);
    expect((screen.getByTestId("d") as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByTestId("d").className).toContain("bg-primary");
  });

  it("Pill: the active one carries bg-primary (the page tests read it)", () => {
    render(<><Pill active onClick={() => {}}>On</Pill><Pill active={false} onClick={() => {}}>Off</Pill></>);
    expect(screen.getByText("On").className).toContain("bg-primary");
    expect(screen.getByText("Off").className).not.toContain("bg-primary");
  });

  it("Card: title, actions, and an extra class that wins over the default", () => {
    render(<Card title="T" actions={<span>act</span>} className="shadow-none">body</Card>);
    expect(screen.getByText("T")).toBeTruthy();
    expect(screen.getByText("act")).toBeTruthy();
    expect(screen.getByText("body").closest("section")!.className).toContain("shadow-none");
    expect(screen.getByText("body").closest("section")!.className).not.toContain("shadow-sm");
  });

  it("Toggle: a switch that reports the opposite value", () => {
    const change = vi.fn();
    render(<Toggle checked={false} onChange={change} label="Dyn" />);
    const sw = screen.getByRole("switch");
    expect(sw.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(sw);
    expect(change).toHaveBeenCalledWith(true);
  });
});

describe("IconButton and Badge", () => {
  it("IconButton is named by its required label, shows a sized hidden icon and reports clicks", () => {
    const click = vi.fn();
    render(<IconButton icon={Trash2} label="Delete bay" onClick={click} data-testid="ib" />);
    const b = screen.getByRole("button", { name: "Delete bay" });
    expect(b.getAttribute("title")).toBe("Delete bay");
    const svg = b.querySelector("svg")!;
    expect(svg.getAttribute("aria-hidden")).toBe("true");
    expect(svg.getAttribute("width")).toBe("16");
    fireEvent.click(b);
    expect(click).toHaveBeenCalledTimes(1);
  });

  it("IconButton: small size uses the inline icon size, active sets aria-pressed, disabled blocks clicks", () => {
    const click = vi.fn();
    render(<IconButton icon={Eye} label="Show" size="sm" active disabled onClick={click} />);
    const b = screen.getByRole("button", { name: "Show" }) as HTMLButtonElement;
    expect(b.querySelector("svg")!.getAttribute("width")).toBe("14");
    expect(b.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(b);
    expect(click).not.toHaveBeenCalled();
  });

  it("Badge: tone classes, optional icon and an extra class", () => {
    render(<><Badge tone="success" icon={Eye} className="ml-2" data-testid="bd">Live</Badge><Badge data-testid="n">Idle</Badge></>);
    const b = screen.getByTestId("bd");
    expect(b.className).toContain("text-success-ink");
    expect(b.className).toContain("ml-2");
    expect(b.querySelector("svg")).toBeTruthy();
    expect(screen.getByTestId("n").className).toContain("text-muted");
    expect(screen.getByTestId("n").querySelector("svg")).toBeNull();
  });

  it("IconButton passes the click event on and takes an extra class; Badge shows a title and a status dot", () => {
    const seen: string[] = [];
    render(<IconButton icon={Eye} label="Peek" className="lg:hidden" onClick={(e) => seen.push(e.type)} />);
    const b = screen.getByRole("button", { name: "Peek" });
    expect(b.className).toContain("lg:hidden");
    fireEvent.click(b);
    expect(seen).toEqual(["click"]);
    cleanup();
    render(<Badge tone="danger" dot="bg-danger" title="why" data-testid="w">Down</Badge>);
    const w = screen.getByTestId("w");
    expect(w.getAttribute("title")).toBe("why");
    expect(w.textContent).toBe("Down");
    expect(w.querySelector("span.bg-danger")).toBeTruthy();
  });
});

describe("MetricCard / StatusBadge / CanvasFrame", () => {
  it("MetricCard shows label, value and an optional hint", () => {
    render(<MetricCard label="Bays" value="12" hint="geometric capacity" data-testid="kpi" />);
    expect(screen.getByTestId("kpi").textContent).toBe("Bays12geometric capacity");
  });

  it("StatusBadge maps a status word to a tone and passes data attributes through", () => {
    render(<StatusBadge status="bad" data-ok="0" data-testid="chip">✗ off</StatusBadge>);
    const chip = screen.getByTestId("chip");
    expect(chip.getAttribute("data-ok")).toBe("0");
    expect(chip.className).toContain("text-danger");
  });

  it("CanvasFrame renders title, actions, the children untouched and the legend", () => {
    render(<CanvasFrame title="Picture" actions={<button>Fit</button>} legend={<span>free</span>}><svg data-testid="surface" /></CanvasFrame>);
    expect(screen.getByRole("heading", { name: "Picture" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Fit" })).not.toBeNull();
    expect(screen.getByTestId("surface")).not.toBeNull();
    expect(screen.getByTestId("canvas-legend").textContent).toBe("free");
  });
});
