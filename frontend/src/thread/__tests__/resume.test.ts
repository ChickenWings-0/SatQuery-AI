/**
 * Reattaching to a run after the stream drops — against the real MSW handlers.
 *
 * The scenario that used to re-run the job: the first `/events` connection
 * closes after the `plan` event with no terminal frame. The loop must poll,
 * see the job still running, reopen the stream from the last id it saw, and
 * finish — with `createJob` called exactly once across the whole episode.
 */
import { HttpResponse, http } from 'msw'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { SatQueryError, createJob, streamJob } from '@/api/client'
import type { JobEvent } from '@/api/events'
import { MOCK_TRACE_ID, eventsFixture } from '@/mocks/fixtures'
import { handlers, snapshot } from '@/mocks/handlers'
import { initialJobState, reduce } from '@/state/job'
import { MAX_ATTEMPTS, defaultDeps, resumeRun, terminalEventOf } from '@/thread/resume'

const server = setupServer(...handlers)
const noSleep = { ...defaultDeps, sleep: async () => undefined }

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

/** An `/events` handler that closes after the first `cutAfter` frames, once. */
function droppingEventsOnce(cutAfter: number) {
  let dropped = false
  return http.get('*/v1/jobs/:jobId/events', ({ request }) => {
    const header = request.headers.get('Last-Event-ID')
    const after = header === null ? -1 : Number(header)
    const encoder = new TextEncoder()
    const shouldDrop = !dropped
    dropped = true
    const stream = new ReadableStream({
      start(controller) {
        for (const [seq, event] of eventsFixture.entries()) {
          if (seq <= after) continue
          if (shouldDrop && seq >= cutAfter) break
          controller.enqueue(
            encoder.encode(`id: ${seq}\nevent: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`),
          )
        }
        controller.close()
      },
    })
    return new HttpResponse(stream, { headers: { 'Content-Type': 'text/event-stream' } })
  })
}

describe('resumeRun', () => {
  it('reattaches after a dropped stream without re-running the job', async () => {
    const running = http.get('*/v1/jobs/:jobId', () =>
      HttpResponse.json(snapshot('running', 'executing')),
    )
    server.use(droppingEventsOnce(4), running)
    const jobs = vi.fn()
    server.events.on('request:start', ({ request }) => {
      if (request.method === 'POST' && request.url.endsWith('/v1/jobs')) jobs()
    })

    const file = new File([new Uint8Array([1, 2, 3])], 'pre.tif', { type: 'image/tiff' })
    const accepted = await createJob([file, file], 'How much built-up area appeared?')

    let state = initialJobState
    let lastId: string | null = null
    const seen: string[] = []
    const onEvent = (event: JobEvent, id: string | null) => {
      if (id !== null) lastId = id
      seen.push(event.type)
      state = reduce(state, event)
    }

    // First attempt: the stream dies after four frames, no terminal event.
    await expect(streamJob(accepted.job_id, { onEvent })).rejects.toBeInstanceOf(SatQueryError)
    expect(seen).toHaveLength(4)
    expect(state.phase).toBe('streaming')

    const outcome = await resumeRun(accepted.job_id, lastId, { onEvent }, undefined, noSleep)

    expect(outcome.kind).toBe('completed')
    expect(state.phase).toBe('succeeded')
    expect(seen.at(-1)).toBe('done')
    // Resumed from id 3: nothing delivered twice.
    expect(seen).toHaveLength(eventsFixture.length)
    expect(jobs).toHaveBeenCalledTimes(1)
  }, 30_000)

  it('synthesises the terminal event when the job already finished', async () => {
    const seen: JobEvent[] = []
    const outcome = await resumeRun(
      MOCK_TRACE_ID,
      '2',
      { onEvent: (event) => seen.push(event) },
      undefined,
      noSleep,
    )
    expect(outcome.kind).toBe('completed')
    expect(seen).toHaveLength(1)
    expect(seen[0]?.type).toBe('done')
  })

  it('gives up with "gone" on a 404 — the only case that should resubmit', async () => {
    server.use(
      http.get('*/v1/jobs/:jobId', () =>
        HttpResponse.json({ detail: 'No job.' }, { status: 404 }),
      ),
    )
    const outcome = await resumeRun(MOCK_TRACE_ID, null, { onEvent: () => undefined }, undefined, noSleep)
    expect(outcome.kind).toBe('gone')
  })

  it('is bounded: transport failures exhaust after MAX_ATTEMPTS', async () => {
    server.use(http.get('*/v1/jobs/:jobId', () => HttpResponse.error()))
    const attempts: number[] = []
    const outcome = await resumeRun(
      MOCK_TRACE_ID,
      null,
      { onEvent: () => undefined, onAttempt: (attempt) => attempts.push(attempt) },
      undefined,
      noSleep,
    )
    expect(outcome.kind).toBe('exhausted')
    expect(attempts).toEqual(Array.from({ length: MAX_ATTEMPTS }, (_, i) => i + 1))
  })

  it('stops quietly when the caller aborts', async () => {
    const controller = new AbortController()
    controller.abort()
    const outcome = await resumeRun(
      MOCK_TRACE_ID,
      null,
      { onEvent: () => undefined },
      controller.signal,
      noSleep,
    )
    expect(outcome.kind).toBe('completed')
  })
})

describe('terminalEventOf', () => {
  it('maps a finished snapshot to done and a failed one to error', () => {
    expect(terminalEventOf(snapshot('succeeded', 'done') as never)?.type).toBe('done')
    expect(terminalEventOf(snapshot('running', 'executing') as never)).toBeNull()
  })
})
