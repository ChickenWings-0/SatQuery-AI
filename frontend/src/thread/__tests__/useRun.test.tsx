/**
 * @vitest-environment happy-dom
 *
 * The run hook against the real client and the real MSW handlers: cancel
 * reaches the server, a second question stops the first, and a dropped stream
 * reattaches instead of resubmitting.
 */
import { act, renderHook, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { MOCK_TRACE_ID, eventsFixture } from '@/mocks/fixtures'
import { handlers, snapshot } from '@/mocks/handlers'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'
import { useUiStore } from '@/state/ui'
import * as resume from '@/thread/resume'
import { useRun } from '@/thread/useRun'

const server = setupServer(...handlers)
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

beforeEach(() => {
  useJobStore.getState().reset()
  useFocusStore.getState().resetForNewRun()
  useUiStore.setState({ recentRuns: [] })
  const file = new File([new Uint8Array([1, 2, 3])], 'pre.tif', { type: 'image/tiff' })
  useUiStore.getState().setFiles([file, file])
  // The reattach loop's backoff is real time; the tests do not need to wait it out.
  vi.spyOn(resume.defaultDeps, 'sleep').mockResolvedValue(undefined)
})

function frame(seq: number): string {
  const event = eventsFixture[seq]!
  return `id: ${seq}\nevent: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`
}

/**
 * A server whose job can be cancelled: the stream paces itself and, once a
 * DELETE has landed, ends with the `error{JOB_CANCELLED}` frame the real
 * server sends instead of the rest of the run.
 */
function cancellableServer() {
  const deleted = vi.fn()
  let cancelled = false
  server.use(
    http.delete('*/v1/jobs/:jobId', ({ params }) => {
      deleted(params['jobId'])
      cancelled = true
      return HttpResponse.json(snapshot('running', 'executing'), { status: 202 })
    }),
    http.get('*/v1/jobs/:jobId/events', () => {
      const encoder = new TextEncoder()
      const body = new ReadableStream<Uint8Array>({
        async start(controller) {
          for (let seq = 0; seq < eventsFixture.length; seq += 1) {
            if (cancelled) {
              // One-shot: the mock serves every job under one id, so a cancel
              // must not bleed into the next job's stream.
              cancelled = false
              const error = {
                code: 'JOB_CANCELLED',
                http_status: 499,
                message: 'The run was cancelled by the client.',
                hint: null,
                ref: null,
                trace_id: MOCK_TRACE_ID,
              }
              controller.enqueue(
                encoder.encode(`id: ${seq}\nevent: error\ndata: ${JSON.stringify(error)}\n\n`),
              )
              break
            }
            controller.enqueue(encoder.encode(frame(seq)))
            await new Promise((resolve) => setTimeout(resolve, 15))
          }
          controller.close()
        },
      })
      return new HttpResponse(body, { headers: { 'content-type': 'text/event-stream' } })
    }),
  )
  return deleted
}

describe('cancel', () => {
  it('sends DELETE and lets the server close the run with JOB_CANCELLED', async () => {
    const deleted = cancellableServer()
    const { result } = renderHook(() => useRun())

    act(() => void result.current.submit('what changed?'))
    await waitFor(() => expect(useJobStore.getState().phase).toBe('streaming'))
    await waitFor(() => expect(useJobStore.getState().nodes.length).toBeGreaterThan(0))

    act(() => result.current.cancel())

    await waitFor(() => expect(useJobStore.getState().phase).toBe('failed'))
    expect(deleted).toHaveBeenCalledWith(MOCK_TRACE_ID)
    expect(useJobStore.getState().error?.code).toBe('JOB_CANCELLED')
    expect(useUiStore.getState().recentRuns[0]?.outcome).toBe('failed')
    expect(result.current.failure?.message).toMatch(/cancelled/i)
  })

  it('falls back to closing out locally when the server cannot confirm', async () => {
    // The server restarted between the 202 and the cancel: it has no such job.
    server.use(
      http.delete('*/v1/jobs/:jobId', () => HttpResponse.json({ detail: 'gone' }, { status: 404 })),
    )
    const { result } = renderHook(() => useRun())

    act(() => void result.current.submit('what changed?'))
    await waitFor(() => expect(useJobStore.getState().phase).toBe('streaming'))

    act(() => result.current.cancel())

    await waitFor(() => expect(useJobStore.getState().phase).toBe('failed'))
    expect(useJobStore.getState().error).toBeNull()
    expect(result.current.failure?.message).toMatch(/did not confirm/i)
  })

  it('stops the previous job on the server when a new question is asked', async () => {
    const deleted = cancellableServer()
    const { result } = renderHook(() => useRun())

    act(() => void result.current.submit('first question'))
    await waitFor(() => expect(useJobStore.getState().phase).toBe('streaming'))
    act(() => void result.current.submit('second question'))

    await waitFor(() => expect(deleted).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(useJobStore.getState().phase).toBe('succeeded'), {
      timeout: 10_000,
    })
  }, 15_000)
})

describe('a dropped stream', () => {
  it('reattaches and finishes, submitting the job exactly once', async () => {
    let dropped = false
    const posts = vi.fn()
    server.use(
      http.get('*/v1/jobs/:jobId', () => HttpResponse.json(snapshot('running', 'executing'))),
      http.get('*/v1/jobs/:jobId/events', ({ request }) => {
        const header = request.headers.get('Last-Event-ID')
        const after = header === null ? -1 : Number(header)
        const drop = !dropped
        dropped = true
        const encoder = new TextEncoder()
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            for (let seq = after + 1; seq < eventsFixture.length; seq += 1) {
              if (drop && seq >= 5) break
              controller.enqueue(encoder.encode(frame(seq)))
            }
            controller.close()
          },
        })
        return new HttpResponse(body, { headers: { 'content-type': 'text/event-stream' } })
      }),
    )
    server.events.on('request:start', ({ request }) => {
      if (request.method === 'POST' && request.url.endsWith('/v1/jobs')) posts()
    })
    const { result } = renderHook(() => useRun())

    act(() => void result.current.submit('what changed?'))

    await waitFor(() => expect(useJobStore.getState().phase).toBe('succeeded'), {
      timeout: 10_000,
    })
    expect(posts).toHaveBeenCalledTimes(1)
    expect(result.current.failure).toBeNull()
    expect(useUiStore.getState().recentRuns[0]?.outcome).toBe('succeeded')
  }, 15_000)

  it('offers a resubmit only when the server no longer knows the job', async () => {
    server.use(
      http.get('*/v1/jobs/:jobId', () => HttpResponse.json({ detail: 'gone' }, { status: 404 })),
      http.get('*/v1/jobs/:jobId/events', () => {
        const encoder = new TextEncoder()
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(encoder.encode(frame(0)))
            controller.close()
          },
        })
        return new HttpResponse(body, { headers: { 'content-type': 'text/event-stream' } })
      }),
    )
    const { result } = renderHook(() => useRun())

    act(() => void result.current.submit('what changed?'))

    await waitFor(() => expect(useJobStore.getState().phase).toBe('failed'))
    expect(result.current.failure?.retryable).toBe(true)
    expect(result.current.failure?.message).toMatch(/no longer knows/i)
    expect(useUiStore.getState().recentRuns[0]?.outcome).toBe('failed')
  })
})
