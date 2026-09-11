/**
 * @vitest-environment happy-dom
 *
 * The hard requirement of the whole design: the execution DAG is never on the
 * main screen. It appears only when someone asks for it.
 *
 * This is asserted three ways, because each catches a different regression:
 *
 *   1. Nothing renders a `.react-flow` element on first paint.
 *   2. `@xyflow/react` is imported by exactly one module, so nobody can pull it
 *      into the entry graph by adding a "small" import somewhere convenient.
 *   3. That module is reached through `lazy()`, so it stays a separate chunk.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'

import App from '@/App'
import { handlers } from '@/mocks/handlers'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'

// The shell fetches /v1/health on mount; serving it from the mock keeps the
// render deterministic and exercises the offline path at the same time.
const server = setupServer(...handlers)
beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }))
afterAll(() => server.close())

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  )
}

afterEach(() => {
  cleanup()
  useJobStore.getState().reset()
  useFocusStore.getState().resetForNewRun()
})

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) return sourceFiles(full)
    return /\.tsx?$/.test(entry) ? [full] : []
  })
}

describe('progressive disclosure', () => {
  it('mounts no DAG canvas on the main screen', () => {
    const { container } = renderApp()

    expect(container.querySelector('.react-flow')).toBeNull()
    // The shell itself did render — otherwise this test passes vacuously.
    expect(screen.getByText('SatQuery AI')).toBeTruthy()
  })

  it('shows no pipeline control before a run has produced a plan', () => {
    renderApp()
    expect(screen.queryByText('View Processing Pipeline')).toBeNull()
  })

  it('imports @xyflow/react in exactly one module', () => {
    const importers = sourceFiles('src').filter((file) =>
      /from '@xyflow\/react'|@xyflow\/react\/dist/.test(readFileSync(file, 'utf8')),
    )
    expect(importers.map((file) => file.replace(/\\/g, '/'))).toEqual([
      'src/components/pipeline/DagCanvas.tsx',
    ])
  })

  it('reaches that module only through a lazy boundary', () => {
    const dialog = readFileSync('src/components/pipeline/PipelineDialog.tsx', 'utf8')
    expect(dialog).toMatch(/lazy\(\(\) => import\('@\/components\/pipeline\/DagCanvas'\)\)/)

    const eager = sourceFiles('src').filter(
      (file) =>
        !file.endsWith('PipelineDialog.tsx') &&
        !file.includes('__tests__') &&
        /import .*DagCanvas.* from/.test(readFileSync(file, 'utf8')),
    )
    expect(eager).toEqual([])
  })
})
