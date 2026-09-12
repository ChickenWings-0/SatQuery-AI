import { fileURLToPath } from 'node:url'

import { defineConfig } from 'vitest/config'

export default defineConfig({
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    // Node by default; component tests opt into jsdom with a per-file
    // `@vitest-environment jsdom` docblock. Keeps the fast logic suite fast.
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    // Stubs the DOM APIs third-party modules probe at import time; see the file.
    setupFiles: ['./src/test/setup.ts'],
    // In a browser a relative '/v1/…' resolves against the page origin. Node
    // has no page, so the client needs an explicit base or every fetch throws
    // ERR_INVALID_URL. The host is never contacted — MSW intercepts first.
    env: { VITE_API_BASE: 'http://localhost' },
  },
})
