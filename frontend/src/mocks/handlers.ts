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
import { STAC_ROOT, NOMINATIM_ROOT } from '@/geo/collections'

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

function frame(event: RecordedEvent, seq: number): string {
  return `id: ${seq}\nevent: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`
}

/** The last event id a reconnecting client already holds, if it sent one (§5). */
function lastSeen(request: Request): number {
  const header = request.headers.get('Last-Event-ID')
  const query = new URL(request.url).searchParams.get('after')
  const raw = header ?? query
  return raw !== null && /^\d+$/.test(raw) ? Number(raw) : -1
}

/** The poll snapshot for the recorded run, as the real server would shape it. */
function snapshot(status: 'running' | 'succeeded', stage: string) {
  const stream = events()
  const done = stream.find((event) => event.event === 'done')
  const plan = stream.find((event) => event.event === 'plan')
  const steps = (plan?.data as { steps?: unknown[] } | undefined)?.steps?.length ?? 0
  return {
    job_id: MOCK_TRACE_ID,
    status,
    stage,
    step: status === 'succeeded' ? steps : 0,
    total_steps: steps,
    pct: status === 'succeeded' ? 100 : 45,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    result: status === 'succeeded' ? (done?.data ?? null) : null,
    error: null,
  }
}

export { snapshot }

/**
 * Track 4.3's third-party hosts. `main.tsx` starts the worker with
 * `onUnhandledRequest: 'bypass'`, which means an unhandled call goes to the
 * real network — so every host the Maps page talks to gets a handler here,
 * and the fixtures are static files under `public/samples/stac/` served by
 * redirect, the same way the artifacts are. Nothing in the flow relies on
 * bypass; the Playwright test runs it with the browser offline to prove it.
 */
const STAC_SAMPLES = '/samples/stac'

/** The place fixture whose name the query starts with, else nothing. */
function placeFixture(q: string): string | null {
  const needle = q.trim().toLowerCase()
  for (const name of ['ahmedabad', 'bengaluru', 'chennai']) {
    if (needle && name.startsWith(needle.slice(0, 3)) && name.includes(needle.split(',')[0]!.trim().slice(0, 4))) return name
  }
  return null
}

/** Which recorded search a body corresponds to: sensor by collection, place by bbox centre. */
function searchFixture(body: { collections?: string[]; bbox?: number[] }): string {
  const sar = (body.collections ?? []).some((c) => c.startsWith('sentinel-1'))
  const lon = body.bbox ? (body.bbox[0]! + body.bbox[2]!) / 2 : 72.6
  const lat = body.bbox ? (body.bbox[1]! + body.bbox[3]!) / 2 : 23
  // Chennai is the only one east of 78°; Bengaluru the only one south of 20°.
  const place = lon > 78 ? 'chennai' : lat < 20 ? 'bengaluru' : 'ahmedabad'
  const wanted = `${sar ? 's1' : 's2'}-${place}`
  const recorded = ['s2-ahmedabad', 's1-ahmedabad', 's1-bengaluru', 's2-chennai']
  if (recorded.includes(wanted)) return wanted
  // No recording for that pair: the nearest one of the same sensor.
  return sar ? 's1-ahmedabad' : 's2-ahmedabad'
}

export const trackFourHandlers = [
  http.get(`${NOMINATIM_ROOT}/search`, async ({ request }) => {
    await delay(250)
    const q = new URL(request.url).searchParams.get('q') ?? ''
    const fixture = placeFixture(q)
    if (!fixture) return HttpResponse.json([])
    // Served from within the handler rather than by redirect: a cross-origin
    // request redirected to this origin stays CORS-tainted in the page.
    const response = await fetch(`${STAC_SAMPLES}/places/${fixture}.json`)
    return HttpResponse.json(await response.json())
  }),

  http.post(`${STAC_ROOT}/search`, async ({ request }) => {
    await delay(400)
    let body: { collections?: string[]; bbox?: number[] } = {}
    try {
      body = (await request.json()) as typeof body
    } catch {
      body = {}
    }
    const fixture = searchFixture(body)
    const response = await fetch(`${STAC_SAMPLES}/search/${fixture}.json`)
    const data = (await response.json()) as { features?: { id: string; assets?: Record<string, { href?: string }> }[] }
    // Thumbnails come from the recorded JPEGs, never the data API.
    for (const feature of data.features ?? []) {
      if (feature.assets?.['rendered_preview']) {
        feature.assets['rendered_preview'] = { ...feature.assets['rendered_preview'], href: `${STAC_SAMPLES}/thumbs/${feature.id}.jpg` }
      }
    }
    return HttpResponse.json(data)
  }),

  http.get(`${STAC_ROOT.replace(/\/stac\/v1$/, '/sas/v1/token')}/:collection`, () =>
    HttpResponse.json({ token: 'mock', 'msft:expiry': new Date(Date.now() + 3_600_000).toISOString() }),
  ),

  // The clipped pair is a real Sentinel-1 RTC recording (see the README), so
  // the files that reach the console are genuine rasters, not placeholders.
  http.post('*/v1/imagery/fetch', async () => {
    await delay(900)
    const response = await fetch(`${STAC_SAMPLES}/fetch/pair-ahmedabad.json`)
    const recorded = (await response.json()) as { fetch_id: string; files: { url: string; name: string }[]; warnings: unknown[] }
    return HttpResponse.json({
      fetch_id: recorded.fetch_id,
      files: recorded.files.map((file) => ({ ...file, url: `/v1/imagery/${recorded.fetch_id}/${file.name}` })),
      warnings: recorded.warnings,
    })
  }),
  http.get('*/v1/imagery/:fetchId/:name', async ({ params }) => {
    const response = await fetch(`${STAC_SAMPLES}/fetch/${String(params['name'])}`)
    if (!response.ok) return HttpResponse.json({ error: { code: 'ARTIFACT_NOT_FOUND', http_status: 404, message: 'No such imagery.' } }, { status: 404 })
    return new HttpResponse(await response.arrayBuffer(), { headers: { 'Content-Type': 'image/tiff' } })
  }),
]

export const handlers = [
  ...trackFourHandlers,

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

  http.get('*/v1/jobs/:jobId/events', ({ request }) => {
    const encoder = new TextEncoder()
    const after = lastSeen(request)
    const stream = new ReadableStream({
      async start(controller) {
        for (const [seq, event] of events().entries()) {
          if (seq <= after) continue
          await delay(paceOf(event))
          controller.enqueue(encoder.encode(frame(event, seq)))
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

  http.get('*/v1/jobs/:jobId', () => HttpResponse.json(snapshot('succeeded', 'done'))),

  // Cancel (§4.10). The mock has no run to stop, so it answers the way the
  // server does for a job it is about to stop: 202 with the snapshot.
  http.delete('*/v1/jobs/:jobId', () =>
    HttpResponse.json(snapshot('running', 'executing'), { status: 202 }),
  ),

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
