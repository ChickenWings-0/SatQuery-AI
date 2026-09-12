/**
 * Browser tests against the production bundle served by `vite preview`, with
 * the backend replaced by the MSW mock (`?mock=1`). No API, no GPU, no network.
 */
import { defineConfig, devices } from '@playwright/test'

const PORT = 4173

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  fullyParallel: false,
  retries: process.env['CI'] ? 1 : 0,
  reporter: process.env['CI'] ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    // `--host 127.0.0.1`: vite preview binds `localhost`, which resolves to ::1
    // here, and Playwright polls the IPv4 URL. Same gotcha as vite.config.ts.
    command: `npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}/?mock=1`,
    reuseExistingServer: !process.env['CI'],
    timeout: 60_000,
  },
})
