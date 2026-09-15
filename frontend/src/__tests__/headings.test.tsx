/**
 * @vitest-environment happy-dom
 *
 * Every routed page renders exactly one `h1` — the page, not the brand — and
 * sets its own `<title>` and description (React 19 hoists them into <head>).
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, waitFor } from '@testing-library/react'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'

import App from '@/App'
import { handlers } from '@/mocks/handlers'
import { useUiStore, type Section } from '@/state/ui'

const ROUTES: Section[] = ['home', 'explore', 'usecases', 'maps', 'saved', 'projects', 'tools', 'datasets', 'history', 'notFound']

const server = setupServer(...handlers)
beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }))
afterEach(() => {
  server.resetHandlers()
  cleanup()
})
afterAll(() => server.close())

function mount(section: Section) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  useUiStore.setState({ section })
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  )
}

describe.each(ROUTES)('%s', (section) => {
  it('renders exactly one h1 and its own title', async () => {
    mount(section)
    await waitFor(() => expect(document.querySelectorAll('h1').length).toBe(1), { timeout: 4_000 })
    await waitFor(() => expect(document.title).toMatch(/SatQuery AI/), { timeout: 4_000 })
    expect(document.querySelector('meta[name="description"]')?.getAttribute('content')).toBeTruthy()
    // No heading level skips more than one step between consecutive headings.
    const levels = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].map((h) => Number(h.tagName[1]))
    for (let i = 1; i < levels.length; i++) {
      expect(levels[i]! - levels[i - 1]!, `heading jump at index ${i}: ${levels.join(',')}`).toBeLessThanOrEqual(1)
    }
  })
})
