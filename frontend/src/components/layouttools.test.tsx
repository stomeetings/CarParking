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
