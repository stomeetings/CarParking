import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev: `npm run dev` proxies the API and the WebSocket to a running backend (VITE_API, default the compose api on
// :8080). Production: FastAPI serves dist/ itself, so every URL in the app is relative (/api, /ws).
const api = process.env.VITE_API ?? "http://localhost:8080";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": api,
      "/ws": { target: api.replace(/^http/, "ws"), ws: true },
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    // the app-level tests drive many state messages through the whole dashboard: ~1 s on a workstation, more than the default 5 s
    // in a CPU-limited container build
    testTimeout: 15000,
  },
});
