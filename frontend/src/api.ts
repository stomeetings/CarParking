// The backend client: relative URLs (FastAPI serves this app), tokens kept in localStorage.
// Viewer token: every read when the backend sets CARPARK_VIEW_TOKEN. Admin token: settings and commands.

const VIEW_KEY = "carparking.viewToken";
const ADMIN_KEY = "carparking.adminToken";

const read = (key: string) => {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
};
const write = (key: string, value: string) => {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* private window: the token lasts for this page only */
  }
};

let viewToken = read(VIEW_KEY);
let adminToken = read(ADMIN_KEY);

export const tokens = {
  get view() {
    return viewToken;
  },
  get admin() {
    return adminToken;
  },
  setView(t: string) {
    viewToken = t;
    write(VIEW_KEY, t);
  },
  setAdmin(t: string) {
    adminToken = t;
    write(ADMIN_KEY, t);
  },
};

/** The token for reads: the admin one also allows viewing. Only tokens the server accepted are kept
 * (sanitizeTokens), so a wrong admin token can never shadow a good viewer token. */
export const readToken = () => adminToken || viewToken;

type Access = { view: boolean; admin: boolean; view_required: boolean; writes_enabled: boolean };

async function accessOf(token: string, fetcher: typeof fetch): Promise<Access> {
  const r = await fetcher("/api/auth", { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!r.ok) throw new Error(`auth check: ${r.status}`);
  return r.json();
}

/** Check the stored tokens with the server and forget the ones it rejects (a mistyped or old admin token, a
 * token from before the server's .env changed). -> what this browser may do now. Before this, a wrong stored admin
 * token was sent with every read (it wins over the viewer token): every request failed with 401 and the
 * camera list stayed empty, however often the right viewer token was entered. */
export async function sanitizeTokens(fetcher: typeof fetch = fetch): Promise<Access> {
  if (adminToken && !(await accessOf(adminToken, fetcher)).admin) tokens.setAdmin("");
  if (viewToken && !(await accessOf(viewToken, fetcher)).view) tokens.setView("");
  return accessOf(readToken(), fetcher);
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}, admin = false): Promise<T> {
  const token = admin ? adminToken : readToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const r = await fetch(path, { ...init, headers });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = await r.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, detail);
  }
  return (r.headers.get("content-type") ?? "").includes("json") ? r.json() : (r.text() as Promise<T>);
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  put: <T>(path: string, body: unknown) => request<T>(path, { method: "PUT", body: JSON.stringify(body) }, true),
  post: <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) }, true),
  // a POST that only reads (e.g. Test slots): the viewer token is enough
  query: <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) => request<T>(path, { method: "PATCH", body: JSON.stringify(body) }, true),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }, true),
};

/** A URL a plain <a>/<img>/WebSocket can open (no headers there): the read token as ?token=. */
export function withToken(path: string): string {
  const t = readToken();
  if (!t) return path;
  return `${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(t)}`;
}

export function wsUrl(path: string, loc: Pick<Location, "protocol" | "host"> = window.location): string {
  return `${loc.protocol === "https:" ? "wss" : "ws"}://${loc.host}${withToken(path)}`;
}

/** Adds ?cam=<id> (the camera an endpoint is about; none = the primary camera). */
export function withCam(path: string, cam: string | null | undefined): string {
  if (!cam) return path;
  return `${path}${path.includes("?") ? "&" : "?"}cam=${encodeURIComponent(cam)}`;
}

const CAM_KEY = "carparking.cam";
export const savedCam = () => {
  try {
    return localStorage.getItem(CAM_KEY) ?? "";
  } catch {
    return "";
  }
};
export const saveCam = (id: string) => write(CAM_KEY, id);
