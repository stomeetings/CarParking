import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LayoutTools } from "./LayoutTools";

afterEach(cleanup);

const show = () => render(<LayoutTools zoneName="Zone A" next={{ index: "", at: "" }} onNext={vi.fn()} bays={0} expected="" master={null} dots={[]} placing={false} busy={false} answer={null}
                                       onPlace={vi.fn()} onDotIndex={vi.fn()} onDotAt={vi.fn()} onDotRemove={vi.fn()} onDotsClear={vi.fn()} onApply={vi.fn()} />);

describe("the layout panel's sections", () => {
  it("opens the reference dots and keeps the master bay and the snap / margin sections folded until asked", () => {
    show();
    expect(screen.getByTestId("sub-lay-dots-body").hasAttribute("hidden")).toBe(false);
    expect(screen.getByTestId("sub-lay-master-body").hasAttribute("hidden")).toBe(true);
    expect(screen.getByTestId("sub-lay-finish-body").hasAttribute("hidden")).toBe(true);
    fireEvent.click(screen.getByTestId("sub-lay-finish-toggle"));
    expect(screen.getByTestId("sub-lay-finish-body").hasAttribute("hidden")).toBe(false);
    expect(screen.getByTestId("sub-lay-finish-toggle").getAttribute("aria-expanded")).toBe("true");
    fireEvent.click(screen.getByTestId("sub-lay-dots-toggle")); // folding keeps what was typed (the content stays mounted)
    expect(screen.getByTestId("sub-lay-dots-body").hasAttribute("hidden")).toBe(true);
    expect(screen.getByTestId("lay-dot-mode")).toBeTruthy();
  });
});

describe("the bay angle", () => {
  const ask = (onApply: (r: unknown) => void) => render(<LayoutTools zoneName="Zone A" next={{ index: "", at: "" }} onNext={vi.fn()} bays={0} expected="" master={null} dots={[{ p: [0.2, 0.7], index: "", at: "" }]}
                                                          placing={false} busy={false} answer={null} onPlace={vi.fn()} onDotIndex={vi.fn()} onDotAt={vi.fn()} onDotRemove={vi.fn()} onDotsClear={vi.fn()} onApply={onApply} />);

  it("is left out of the request when empty (auto from the dot pair) and sent, held to 20-160, when typed", () => {
    const sent: Record<string, unknown>[] = [];
    ask((r) => sent.push(r as Record<string, unknown>));
    expect(screen.getByTestId("lay-angle-hint").textContent).toContain("front");
    fireEvent.change(screen.getByTestId("lay-width"), { target: { value: "2.5" } });
    fireEvent.click(screen.getByTestId("lay-fit-dots"));
    expect("stall_angle" in sent[0]).toBe(false);
    fireEvent.change(screen.getByTestId("lay-angle"), { target: { value: "60" } });
    fireEvent.click(screen.getByTestId("lay-fit-dots"));
    expect(sent[1].stall_angle).toBe(60);
    fireEvent.change(screen.getByTestId("lay-angle"), { target: { value: "5" } });
    fireEvent.click(screen.getByTestId("lay-fit-dots"));
    expect(sent[2].stall_angle).toBe(20);
  });
});
