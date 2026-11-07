import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { clarionApi } from './server/api.js'

// The custom plugin exposes /api endpoints directly from the dev/preview server:
//   POST   /api/upload        — import CSV/TSV/JSON as a SQLite table
//   GET    /api/datasets      — list uploads + live schema from the workspace DB
//   DELETE /api/datasets/:id  — remove an uploaded dataset and its table
//   GET    /api/health        — Qwen worker state (loading / ready / error)
//   POST   /api/query         — real pipeline: detect → clarify → generate → run
export default defineConfig({
  plugins: [react(), clarionApi()],
  // The API can import files and execute explicitly confirmed writes, so keep
  // the default development server on this machine unless the user opts in
  // to a LAN-facing deployment with its own access controls.
  server: { port: 5173, host: '127.0.0.1' },
})
