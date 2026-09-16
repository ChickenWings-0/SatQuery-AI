import { fileURLToPath } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  assetsInclude: ['**/*.glsl'],
  // maplibre's worker (`MapStage.tsx`, `?worker&url`) is an ES module that
  // imports a sibling; ESM output keeps that shape instead of an IIFE that
  // would have to swallow `import()` calls it cannot express.
  worker: { format: 'es' },
  server: {
    // `make demo` runs the FastAPI on :8000 and this dev server on :5173.
    // Same-origin `/v1` in development means multipart, SSE and the artifact
    // URLs behave exactly as they do in the packaged build (`api/client.ts`).
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8000', changeOrigin: false },
    },
  },
  build: {
    // Explicit, though it is the default: a `sourcemap: true` copied in from
    // a debugging session must not ship the source to every visitor.
    sourcemap: false,
    manifest: true,
    chunkSizeWarningLimit: 250,
    rollupOptions: {
      output: {
        // Every heavy dependency lives in the chunk of its one lazy owner
        // (`DOCS/frontend_blueprint.md` §6.5): `three` + R3F ride with
        // `Globe.tsx`, `maplibre-gl` with `MapStage.tsx`, `@xyflow/react`
        // with `DagCanvas.tsx`, `pdf-lib` with `export/sitrep/render.ts`, the
        // mask vectoriser with `export/geojson/masks.ts`. Rollup does that from the dynamic-import
        // boundaries alone; the object form of `manualChunks` would name the
        // chunks but also hoists them into the entry's static imports, which
        // is the one thing `scripts/check-bundle.mjs` exists to forbid.
        // The function below only *names* the chunks Rollup already split,
        // so the manifest reads `globe-*.js` / `map-*.js` / `dag-*.js`.
        chunkFileNames: (chunk) => {
          const ids = chunk.moduleIds
          const has = (needle: string) => ids.some((id) => id.includes(needle))
          if (has('node_modules/three/')) return 'assets/globe-[hash].js'
          if (has('node_modules/maplibre-gl/')) return 'assets/map-[hash].js'
          if (has('node_modules/@xyflow/')) return 'assets/dag-[hash].js'
          if (has('node_modules/pdf-lib/') || has('node_modules/@pdf-lib/')) return 'assets/sitrep-[hash].js'
          if (has('/src/export/geojson/masks.ts')) return 'assets/masks-[hash].js'
          return 'assets/[name]-[hash].js'
        },
      },
    },
  },
})
