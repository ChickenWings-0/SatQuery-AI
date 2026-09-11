import { fileURLToPath } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    // The API is a local FastAPI on 8000. Proxying rather than hard-coding an
    // origin keeps the app same-origin in dev, so SSE and multipart uploads
    // behave exactly as they will in the packaged build.
    //
    // 127.0.0.1 rather than localhost, deliberately: uvicorn binds IPv4 only,
    // while Node 18+ resolves localhost to ::1 first and every proxied request
    // fails with ECONNREFUSED.
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
