import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SettingsDrawer } from "./SettingsDrawer";
import { TokenDialog } from "./TokenDialog";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const escape = () => fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape", code: "Escape" });

describe("TokenDialog on Radix Dialog", () => {
  it("is a dialog named Access, with its test ids, and focuses the viewer-token field", async () => {
    render(<TokenDialog open kind="view" onSaved={() => {}} onClose={() => {}} />);
    expect(screen.getByRole("dialog", { name: "Access" })).toBe(screen.getByTestId("token-dialog"));
    expect(screen.getByTestId("view-token")).toBeTruthy();
    await waitFor(() => expect(document.activeElement).toBe(screen.getByTestId("view-token")));
  });

  it("Escape, the Close button and Cancel call onClose", () => {
    const close = vi.fn();
    render(<TokenDialog open kind="both" onSaved={() => {}} onClose={close} />);
    escape();
    expect(close).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByLabelText("Close"));
    fireEvent.click(screen.getByText("Cancel"));
    expect(close).toHaveBeenCalledTimes(3);
  });

  it("without onClose (a token is required) Escape does not close it", () => {
    const close = vi.fn();
    const { rerender } = render(<TokenDialog open kind="view" onSaved={() => {}} />);
    escape();
    expect(screen.getByTestId("token-dialog")).toBeTruthy();
    expect(close).not.toHaveBeenCalled();
    rerender(<TokenDialog open={false} kind="view" onSaved={() => {}} />);
  });

  it("locks the page scroll while open and renders nothing when closed", () => {
    const { rerender } = render(<TokenDialog open kind="view" onSaved={() => {}} onClose={() => {}} />);
    expect(document.body.style.pointerEvents).toBe("none"); // Radix: the page behind is inert while a modal is open
    rerender(<TokenDialog open={false} kind="view" onSaved={() => {}} onClose={() => {}} />);
    return waitFor(() => expect(screen.queryByTestId("token-dialog")).toBeNull()); // (after its exit animation)
  });
});

describe("SettingsDrawer on Radix Dialog", () => {
  const serve = () => vi.stubGlobal("fetch", () => Promise.resolve(new Response("{}", { status: 500 })));

  it("is a dialog named Settings and Escape closes it", () => {
    serve();
    const close = vi.fn();
    render(<SettingsDrawer open onClose={close} onNeedAdmin={() => {}} tokenKey="" codeStale={false} model={undefined} cam={null} />);
    expect(screen.getByRole("dialog", { name: "Settings" })).toBe(screen.getByTestId("settings-drawer"));
    escape();
    expect(close).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByLabelText("Close"));
    expect(close).toHaveBeenCalledTimes(2);
  });
});
