import { defineConfig } from "vite";

// Dev server proxies /api to the FastAPI backend (docs/plan/07 §4).
export default defineConfig({
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } },
  },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 5000 },
});
