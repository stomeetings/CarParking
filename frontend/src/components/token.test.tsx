import { describe, expect, it } from "vitest";
import { checkTokens } from "./TokenDialog";

// a server whose viewer token is "v" and admin token is "a" (GET /api/auth)
const server = (writes = true) => (async (_: RequestInfo | URL, init?: RequestInit) => {
  const t = (new Headers(init?.headers).get("Authorization") ?? "").replace("Bearer ", "");
  const admin = writes && t === "a";
  return new Response(JSON.stringify({ view: admin || t === "v", admin, view_required: true, writes_enabled: writes }),
                      { headers: { "Content-Type": "application/json" } });
}) as typeof fetch;

describe("token check before saving", () => {
  it("accepts the right tokens", async () => {
    expect(await checkTokens("view", "v", "", server())).toEqual({});
    expect(await checkTokens("admin", "", "a", server())).toEqual({});
    expect(await checkTokens("both", "v", "a", server())).toEqual({});
  });
  it("says so when the viewer token is pasted where the admin token is needed", async () => {
    const r = await checkTokens("admin", "", "v", server());
    expect(r.admin).toMatch(/This is the viewer token: it only allows viewing/);
    expect(r.admin).toContain("CARPARK_API_TOKEN");
  });
  it("refuses wrong or missing tokens with the reason", async () => {
    expect((await checkTokens("view", "nope", "", server())).view).toMatch(/Not accepted.*CARPARK_VIEW_TOKEN/);
    expect((await checkTokens("view", "", "", server())).view).toBe("Required.");
    expect((await checkTokens("admin", "", "", server())).admin).toMatch(/Required/);
    expect((await checkTokens("admin", "", "zzz", server())).admin).toMatch(/Not accepted.*CARPARK_API_TOKEN/);
    expect((await checkTokens("admin", "", "a", server(false))).admin).toMatch(/no admin token set/);
  });
});

describe("stored tokens", () => {
  it("a wrong stored admin token is dropped, so the good viewer token is used again", async () => {
    const { readToken, sanitizeTokens, tokens } = await import("../api");
    tokens.setView("v");
    tokens.setAdmin("typo"); // what made every read fail with 401 (the admin token wins) and the camera list empty
    expect(readToken()).toBe("typo");
    const access = await sanitizeTokens(server());
    expect(tokens.admin).toBe("") ;
    expect(tokens.view).toBe("v");
    expect(readToken()).toBe("v");
    expect(access.view).toBe(true);
    tokens.setAdmin("a"); // a good admin token stays
    expect((await sanitizeTokens(server())).admin).toBe(true);
    expect(tokens.admin).toBe("a");
    tokens.setView("old"); // a viewer token from before the server's .env changed: dropped too
    tokens.setAdmin("");
    expect((await sanitizeTokens(server())).view).toBe(false);
    expect(tokens.view).toBe("");
  });
});
