/**
 * Mock Service Worker handlers — the offline demo path.
 *
 * Activated with `?mock=1` (see `main.tsx`). Every endpoint the UI touches is
 * served from the recordings in `@/mocks/fixtures`, including a genuinely
 * streaming SSE response: the recorded events are replayed through a
 * `ReadableStream` on a timeline derived from the run's own `est_ms` values, so
 * the DAG fills in and the evidence tray accumulates at roughly the pace the
 * real backend produces them. A mock that returned everything at once would
 * hide exactly the behaviour the streaming UI exists to show.
 */
import { HttpResponse, delay, http } from 'msw'

import {
  MOCK_TRACE_ID,
  eventsFixture,
  healthFixture,
  registryFixture,
  validateFixture,
  validateSingleFixture,
  type RecordedEvent,
} from '@/mocks/fixtures'
import { activeScenario, eventsForScenario } from '@/mocks/scenarios'

/** The recording for whichever scenario the page asked for. */
function events(): RecordedEvent[] {
  return eventsForScenario(eventsFixture, activeScenario())
}

/** Wall-clock gap before each event, in ms. Compressed but proportional. */
function paceOf(event: RecordedEvent): number {
  switch (event.event) {
    case 'queued':
      return 0
    case 'stage':
      return 60
    case 'plan':
      return 180
    case 'step_started': {
      // Long steps should *feel* long — that gap is where a judge watches the
      // node pulse — but a 1.8s change detector need not cost 1.8s every reload.
      const est = (event.data as { est_ms?: number }).est_ms ?? 200
      return Math.min(900, Math.max(120, est / 3))
    }
    case 'artifact':
      return 45
    case 'step_completed':
      return 90
    default:
      return 120
  }
}

function frame(event: RecordedEvent): string {
  return `event: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`
}

export const handlers = [
  http.get('*/v1/health', () => HttpResponse.json(healthFixture)),

  http.get('*/v1/registry', () => HttpResponse.json(registryFixture)),

  http.post('*/v1/validate', async ({ request }) => {
    await delay(350) // the real pre-flight is fast but not instant

    // The mock has to answer the request that was actually made. Serving the
    // bi-temporal recording for a single upload made the panel claim two
    // manifests and a BI_TEMPORAL pair for one file — not a crash, but a mock
    // that lies about the contract is worse than one that fails loudly.
    let count = 2
    try {
      count = (await request.formData()).getAll('images').length
    } catch {
      count = 2
    }
    return HttpResponse.json(count <= 1 ? validateSingleFixture : validateFixture)
  }),

  http.post('*/v1/jobs', async () => {
    await delay(120)
    return HttpResponse.json(
      {
        job_id: MOCK_TRACE_ID,
        status: 'queued',
        poll_url: `/v1/jobs/${MOCK_TRACE_ID}`,
        events_url: `/v1/jobs/${MOCK_TRACE_ID}/events`,
      },
      { status: 202 },
    )
  }),

  http.get('*/v1/jobs/:jobId/events', () => {
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        for (const event of events()) {
          await delay(paceOf(event))
          controller.enqueue(encoder.encode(frame(event)))
        }
        controller.close()
      },
    })
    return new HttpResponse(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
    })
  }),

  http.get('*/v1/jobs/:jobId', () => {
    const stream = events()
    const done = stream.find((event) => event.event === 'done')
    const plan = stream.find((event) => event.event === 'plan')
    const steps = (plan?.data as { steps?: unknown[] } | undefined)?.steps?.length ?? 0
    return HttpResponse.json({
      job_id: MOCK_TRACE_ID,
      status: 'succeeded',
      stage: 'done',
      step: steps,
      total_steps: steps,
      pct: 100,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      result: done?.data ?? null,
      error: null,
    })
  }),

  http.get('*/v1/traces/:traceId', () => {
    const done = events().find((event) => event.event === 'done')
    const trace = (done?.data as { trace?: unknown } | undefined)?.trace
    return trace
      ? HttpResponse.json(trace)
      : HttpResponse.json({ detail: 'No trace.' }, { status: 404 })
  }),

  // Artifacts are served from `public/mock-artifacts/`, which holds the real
  // renders from the recorded run — so the evidence gallery shows genuine
  // multi-spectral imagery offline, not placeholder rectangles.
  http.get('*/v1/artifacts/:traceId/:filename', ({ params }) =>
    HttpResponse.redirect(`/mock-artifacts/${String(params['filename'])}`),
  ),
]
