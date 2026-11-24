/**
 * The fast mirror: traffic moves there when told, falls back to the stable
 * origin when it fails, and job events are short-polled there because the
 * mirror (a Cloudflare quick tunnel) buffers streams until they end.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  artifactUrl,
  currentApiBase,
  health,
  streamJob,
  switchToFastBase,
  switchToHomeBase,
} from '@/api/client'
import type { JobEvent } from '@/api/events'

const FAST = 'https://fast.example.trycloudflare.com'

afterEach(() => {
  switchToHomeBase()
  vi.unstubAllGlobals()
})

function sse(...frames: Array<[string, string, object]>): Response {
  const body = frames.map(([id, event, data]) => `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('')
  return new Response(body || ': ping\n\n', { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

describe('switching the API base', () => {
  it('sends requests and artifact URLs to the fast mirror', async () => {
    const home = currentApiBase()
    switchToFastBase(`${FAST}/`)
    expect(currentApiBase()).toBe(FAST)
    expect(artifactUrl({ url: '/v1/artifacts/t/a.png' } as never)).toBe(`${FAST}/v1/artifacts/t/a.png`)
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await health()
    expect(fetchMock.mock.calls[0]![0]).toBe(`${FAST}/v1/health`)
    switchToHomeBase()
    expect(currentApiBase()).toBe(home)
  })

  it('falls back to the stable origin when the mirror is unreachable', async () => {
    const home = currentApiBase()
    switchToFastBase(FAST)
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(health()).rejects.toMatchObject({ transport: true })
    expect(currentApiBase()).toBe(home)
  })
})

describe('streamJob on the fast mirror', () => {
  it('short-polls, resuming after the last id, until the terminal event', async () => {
    switchToFastBase(FAST)
    const replies = [
      sse(['0', 'queued', { job_id: 'j' }], ['1', 'stage', { stage: 'ingesting', pct: 5 }]),
      sse(),
      sse(['2', 'done', {}]),
    ]
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(replies.shift()))
    vi.stubGlobal('fetch', fetchMock)
    const seen: Array<string | null> = []
    const types: string[] = []
    await streamJob('j', {
      onEvent: (event: JobEvent, id) => {
        types.push(event.type)
        seen.push(id)
      },
    }).catch(() => undefined)

    expect(fetchMock).toHaveBeenCalledTimes(3)
    for (const [url] of fetchMock.mock.calls) expect(url).toBe(`${FAST}/v1/jobs/j/events?poll=true`)
    const lastIds = fetchMock.mock.calls.map(([, init]) => (init as RequestInit & { headers: Record<string, string> }).headers['Last-Event-ID'])
    expect(lastIds).toEqual([undefined, '1', '1'])
    expect(types[0]).toBe('queued')
    expect(seen).toContain('2')
  })

  it('opens one stream, without polling, on the stable origin', async () => {
    const fetchMock = vi.fn().mockResolvedValue(sse(['0', 'queued', { job_id: 'j' }], ['1', 'done', {}]))
    vi.stubGlobal('fetch', fetchMock)
    await streamJob('j', { onEvent: () => undefined }).catch(() => undefined)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(String(fetchMock.mock.calls[0]![0])).toMatch(/\/v1\/jobs\/j\/events$/)
  })
})
