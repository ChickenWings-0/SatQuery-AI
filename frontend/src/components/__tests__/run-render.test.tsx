/**
 * @vitest-environment happy-dom
 *
 * F4 and F5 against the recorded run: the evidence, the numbers, the grounded
 * answer, and the one interaction that ties them together.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import App from '@/App'
import { parseJobEvent } from '@/api/events'
import { handlers } from '@/mocks/handlers'
import { eventsFixture } from '@/mocks/fixtures'
import { eventsForScenario, type Scenario } from '@/mocks/scenarios'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'

const server = setupServer(...handlers)
beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }))
afterAll(() => server.close())

/** Drive the store with the recording, exactly as the SSE client would. */
function playRecording(scenario: Scenario = 'canonical') {
  const store = useJobStore.getState()
  for (const recorded of eventsForScenario(eventsFixture, scenario)) {
    store.apply(parseJobEvent(recorded.event, JSON.stringify(recorded.data)))
  }
}

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  useJobStore.getState().reset()
  useFocusStore.getState().resetForNewRun()
})
afterEach(() => {
  cleanup()
  useJobStore.getState().reset()
  useFocusStore.getState().resetForNewRun()
})

describe('F4 — Data Stage', () => {
  it('shows the evidence tray with the views the model was shown', () => {
    playRecording()
    renderApp()

    expect(screen.getByText(/views the model was shown/)).toBeTruthy()
    // Twice over: once as the tray thumbnail, once as the active viewer's title.
    expect(screen.getAllByText('True colour')).toHaveLength(2)
    expect(screen.getByText('NDBI')).toBeTruthy()
  })

  it('marks bi-temporal views as comparable, so the A/B swipe is offered', () => {
    playRecording()
    renderApp()
    // Both the tray badge and the viewer header advertise the pairing.
    expect(screen.getAllByText('pre → post').length).toBeGreaterThan(0)
  })

  it('renders KPI cards from the fact sheet', () => {
    playRecording()
    const { container } = renderApp()

    expect(screen.getByText('Scene changed')).toBeTruthy()
    expect(screen.getByText('5.34')).toBeTruthy()
    // Queried by attribute: "Confidence" is also the thread's section heading,
    // so matching on text alone would find two different things.
    expect(container.querySelector('[data-kpi="__confidence__"]')).toBeTruthy()
  })

  it('fits every KPI on one row rather than orphaning the last card', () => {
    playRecording()
    const { container } = renderApp()
    const grid = container.querySelector('[data-kpi]')?.parentElement
    expect(grid?.getAttribute('style')).toContain('auto-fit')
  })
})

describe('F5 — Interrogation Thread', () => {
  it('renders the answer with citation pills', () => {
    playRecording()
    renderApp()

    const pill = screen.getByRole('button', { name: '5.34%' })
    expect(pill.className).toContain('bg-evidence')
    expect(pill.getAttribute('title')).toContain('step 3')
    expect(pill.getAttribute('title')).toContain('changed_area_pct')
  })

  it('badges a templated answer rather than passing it off as the VLM', () => {
    playRecording()
    renderApp()
    expect(screen.getByText('templated answer')).toBeTruthy()
  })

  it('renders uncited numeric spans with the amber wavy underline', () => {
    playRecording('ungrounded')
    const { container } = renderApp()

    const uncited = [...container.querySelectorAll('.uncited')].map((node) => node.textContent)
    expect(uncited).toEqual(['12 hectares', '8.5%'])
    expect(screen.getByText(/2 ungrounded numbers/)).toBeTruthy()
  })

  it('shows the live pipeline pulse and the disclosure button', () => {
    playRecording()
    renderApp()

    expect(screen.getByText('View Processing Pipeline')).toBeTruthy()
    // The recorded run degraded two steps and skipped one.
    expect(screen.getByText(/3 degraded/)).toBeTruthy()
  })
})

describe('citation → KPI → viewer', () => {
  it('lights the matching KPI card and remembers the step for the pipeline', () => {
    playRecording()
    const { container } = renderApp()

    // changed_area_pct is produced by step 3 and is the headline KPI.
    fireEvent.click(screen.getByRole('button', { name: '5.34%' }))

    expect(useFocusStore.getState().focusedStep).toBe(3)

    const card = container.querySelector('[data-kpi="change_statistics.changed_area_pct"]')
    expect(card).toBeTruthy()
    expect(card?.getAttribute('data-active')).toBe('true')
  })

  it('lights no card when the cited scalar is not one of the headlines', () => {
    // Only a handful of ~50 scalars earn a card, so most citations have no card
    // to light. The viewer switch still carries the grounding; silently lighting
    // the wrong card would be worse than lighting none.
    playRecording()
    const { container } = renderApp()

    fireEvent.click(screen.getByRole('button', { name: '0.07' }))

    expect(useFocusStore.getState().focusedStep).toBe(4)
    expect(container.querySelectorAll('[data-active][data-kpi]')).toHaveLength(0)
  })

  it('switches the viewer to the evidence that step produced', () => {
    playRecording()
    renderApp()

    fireEvent.click(screen.getByRole('button', { name: '0.07' }))
    expect(useFocusStore.getState().activeViewKey).toBe('NDBI')
  })

  it('leaves the viewer alone when the cited step produced no picture', () => {
    playRecording()
    renderApp()

    // change_statistics (step 3) emits SCALARS only — there is nothing to show.
    const before = useFocusStore.getState().activeViewKey
    fireEvent.click(screen.getByRole('button', { name: '5.34%' }))

    expect(useFocusStore.getState().focusedStep).toBe(3)
    expect(useFocusStore.getState().activeViewKey).toBe(before)
  })
})

describe('the pipeline modal', () => {
  it('mounts the DAG only after the button is clicked', async () => {
    playRecording()
    const { container } = renderApp()

    expect(container.querySelector('.react-flow')).toBeNull()
    fireEvent.click(screen.getByText('View Processing Pipeline'))

    // The lazy chunk resolves asynchronously; the dialog itself is immediate.
    expect(await screen.findByText('Processing pipeline')).toBeTruthy()
    expect(screen.getByText(/CHANGE_VQA\|BI_TEMPORAL\|optical/)).toBeTruthy()
  })
})
