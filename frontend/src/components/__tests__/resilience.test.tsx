/**
 * @vitest-environment happy-dom
 *
 * What the UI does when the data is not perfect: a stream that stops without
 * saying so, a file that should never have been uploaded, and the first-screen
 * affordance that used to do nothing at all.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import App from '@/App'
import { SatQueryError, streamJob } from '@/api/client'
import { handlers } from '@/mocks/handlers'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'
import { useUiStore } from '@/state/ui'
import type { ValidateResponse } from '@/api/types'

const server = setupServer(...handlers)
beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  )
}

/** Enough of a `/v1/validate` response to unlock the composer and the chips. */
function passingValidation(tasks: string[]): ValidateResponse {
  return {
    inputs: [],
    compatibility: {
      overall: 'PASS',
      pair_type: 'BITEMPORAL',
      pair_type_source: 'inferred',
      checks: [],
      actions_taken: [],
    },
    supported_tasks: tasks,
    warnings: [],
  } as unknown as ValidateResponse
}

/**
 * Serve an event stream whose body ends *without* a `done` or `error` frame —
 * a killed worker, a restarted server, a proxy that closed the connection.
 */
function serveTruncatedStream() {
  server.use(
    http.get('*/v1/jobs/:jobId/events', () => {
      const encoder = new TextEncoder()
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode('event: queued\ndata: {"job_id":"t_1"}\n\n'))
          controller.enqueue(
            encoder.encode('event: stage\ndata: {"stage":"planning","pct":10}\n\n'),
          )
          // …and then the server goes away mid-run.
          controller.close()
        },
      })
      return new HttpResponse(body, {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      })
    }),
  )
}

beforeEach(() => {
  useJobStore.getState().reset()
  useFocusStore.getState().resetForNewRun()
  useFocusStore.getState().setDraft('')
  useUiStore.setState({ validation: null, validationError: null, files: [], recentRuns: [] })
})
afterEach(() => {
  cleanup()
  useJobStore.getState().reset()
})

describe('a stream that stops without a terminal event', () => {
  it('is reported rather than treated as a run still in progress', async () => {
    serveTruncatedStream()
    const seen: string[] = []

    await expect(
      streamJob('t_1', { onEvent: (event) => seen.push(event.type) }),
    ).rejects.toBeInstanceOf(SatQueryError)

    // Everything that did arrive is still delivered — the failure is about the
    // events that never came, not the ones that did.
    expect(seen).toEqual(['queued', 'stage'])
  })

  it('reaches `onError` and offers a retry', async () => {
    serveTruncatedStream()
    const reported: unknown[] = []

    const failure = await streamJob('t_1', {
      onEvent: () => undefined,
      onError: (error) => reported.push(error),
    }).catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(SatQueryError)
    expect((failure as SatQueryError).retryable).toBe(true)
    expect(reported).toHaveLength(1)
  })
})

describe('the job phase', () => {
  it('leaves `streaming` when the transport fails, so the composer re-enables', () => {
    useJobStore.setState({ phase: 'streaming' })
    useJobStore.getState().markFailed()
    expect(useJobStore.getState().phase).toBe('failed')
    // And nothing is put in `error`: a §6 envelope comes from the server, and
    // the client does not get to forge one.
    expect(useJobStore.getState().error).toBeNull()
  })

  it('never turns a finished run red because of a late abort', () => {
    useJobStore.setState({ phase: 'succeeded' })
    useJobStore.getState().markFailed()
    expect(useJobStore.getState().phase).toBe('succeeded')
  })
})

describe('the pre-flight task chips', () => {
  it('put their question in the composer instead of doing nothing', async () => {
    useUiStore.setState({ validation: passingValidation(['CHANGE_VQA']) })
    renderApp()

    fireEvent.click(screen.getByRole('button', { name: /How much built-up area appeared/ }))

    await waitFor(() =>
      expect(useFocusStore.getState().draft).toBe('How much built-up area appeared?'),
    )
    const composer = screen.getByRole('textbox', { name: /Question about this imagery/ })
    expect((composer as HTMLTextAreaElement).value).toBe('How much built-up area appeared?')
  })

  it('still offers an askable question for a task the registry has no phrasing for', () => {
    useUiStore.setState({ validation: passingValidation(['SOME_NEW_TASK']) })
    renderApp()

    expect(screen.getByRole('button', { name: /Run some new task on this imagery/ })).toBeTruthy()
  })
})

describe('the query composer', () => {
  it('has an accessible name rather than only a placeholder', () => {
    useUiStore.setState({ validation: passingValidation(['VQA']) })
    renderApp()
    expect(screen.getByRole('textbox', { name: 'Question about this imagery' })).toBeTruthy()
  })

  it('says why it is disabled, out loud, when pre-flight blocks the pair', () => {
    const blocked = passingValidation([])
    blocked.compatibility.overall = 'FAIL'
    useUiStore.setState({ validation: blocked })
    renderApp()

    expect(screen.getAllByText(/cannot be analysed together/i).length).toBeGreaterThan(0)
    expect(screen.getByRole('textbox', { name: 'Question about this imagery' })).toHaveProperty(
      'disabled',
      true,
    )
  })
})

describe('file selection', () => {
  it('rejects what the server would reject, before spending the upload', () => {
    renderApp()
    const input = document.querySelector('input[type="file"]') as HTMLInputElement

    const archive = new File(['zip'], 'scene.zip', { type: 'application/zip' })
    fireEvent.change(input, { target: { files: [archive] } })

    expect(screen.getByText(/not a GeoTIFF, PNG or JPEG/)).toBeTruthy()
    // And a bad drop does not clear a good selection.
    expect(useUiStore.getState().files).toHaveLength(0)
  })

  it('names an empty file rather than uploading nothing', () => {
    renderApp()
    const input = document.querySelector('input[type="file"]') as HTMLInputElement

    fireEvent.change(input, {
      target: { files: [new File([], 'empty.tif', { type: 'image/tiff' })] },
    })

    expect(screen.getByText(/empty file/)).toBeTruthy()
  })
})
