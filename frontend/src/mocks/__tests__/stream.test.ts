/**
 * The offline demo path, end to end.
 *
 * Runs the *real* SSE client against the *real* MSW handlers in Node, and folds
 * the result through the *real* reducer. Nothing here is a stand-in except the
 * network, which is the point: if this passes, `?mock=1` in a browser drives the
 * UI to a complete state with the backend down.
 */
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'

import { createJob, streamJob, validate } from '@/api/client'
import type { JobEvent } from '@/api/events'
import { handlers } from '@/mocks/handlers'
import { MOCK_TRACE_ID } from '@/mocks/fixtures'
import { completedCount, reduceAll } from '@/state/job'

const server = setupServer(...handlers)

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('mocked offline path', () => {
  it('validates an upload without a backend', async () => {
    const file = new File([new Uint8Array([1, 2, 3])], 'pre.tif', { type: 'image/tiff' })
    const result = await validate([file, file])

    expect(result.inputs).toHaveLength(2)
    expect(result.compatibility.pair_type).toBe('BI_TEMPORAL')
    expect(result.supported_tasks).toContain('CHANGE_VQA')
  })

  it('queues a job and streams it to a complete terminal state', async () => {
    const file = new File([new Uint8Array([1, 2, 3])], 'pre.tif', { type: 'image/tiff' })
    const accepted = await createJob([file, file], 'How much built-up area appeared?')
    expect(accepted.job_id).toBe(MOCK_TRACE_ID)

    const received: JobEvent[] = []
    await streamJob(accepted.job_id, { onEvent: (event) => received.push(event) })

    // The stream must terminate on its own; a hung mock would time out here.
    expect(received.at(-1)?.type).toBe('done')

    const state = reduceAll(received)
    expect(state.phase).toBe('succeeded')
    expect(state.jobId).toBe(MOCK_TRACE_ID)
    expect(state.nodes.length).toBeGreaterThan(0)
    expect(completedCount(state)).toBe(state.nodes.length)
    expect(state.artifacts.length).toBeGreaterThan(0)
    expect(state.result?.answer.text).toBeTruthy()
  }, 30_000)
})
