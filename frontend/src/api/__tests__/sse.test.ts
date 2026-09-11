/**
 * The SSE reader's edge cases — the ones that only show up over a real socket,
 * where chunk boundaries land wherever TCP decides.
 */
import { describe, expect, it } from 'vitest'

import { parseFrame, readFrames } from '@/api/sse'

function streamOf(...chunks: string[]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(body)
}

async function collect(response: Response) {
  const frames = []
  for await (const frame of readFrames(response)) frames.push(frame)
  return frames
}

describe('parseFrame', () => {
  it('reads an event name and its data', () => {
    expect(parseFrame('event: stage\ndata: {"pct":45}')).toEqual({
      event: 'stage',
      data: '{"pct":45}',
    })
  })

  it('joins multi-line data with newlines', () => {
    expect(parseFrame('event: x\ndata: a\ndata: b')?.data).toBe('a\nb')
  })

  it('returns null for a heartbeat comment', () => {
    expect(parseFrame(': ping')).toBeNull()
  })

  it('strips exactly one leading space, per the spec', () => {
    expect(parseFrame('data:  two-spaces')?.data).toBe(' two-spaces')
  })
})

describe('readFrames', () => {
  it('yields each frame in order', async () => {
    const frames = await collect(
      streamOf('event: queued\ndata: {"job_id":"a"}\n\nevent: done\ndata: {}\n\n'),
    )
    expect(frames.map((f) => f.event)).toEqual(['queued', 'done'])
  })

  it('reassembles a frame split across chunk boundaries', async () => {
    // The failure this guards: a naive reader parses "event: sta" as a frame.
    const frames = await collect(streamOf('event: sta', 'ge\ndata: {"pct"', ':45}\n\n'))
    expect(frames).toEqual([{ event: 'stage', data: '{"pct":45}' }])
  })

  it('skips interleaved heartbeats without losing the frames around them', async () => {
    const frames = await collect(
      streamOf('event: a\ndata: 1\n\n', ': ping\n\n', 'event: b\ndata: 2\n\n'),
    )
    expect(frames.map((f) => f.event)).toEqual(['a', 'b'])
  })

  it('emits a trailing frame the server did not terminate with a blank line', async () => {
    const frames = await collect(streamOf('event: done\ndata: {}'))
    expect(frames.map((f) => f.event)).toEqual(['done'])
  })

  it('handles CRLF line endings', async () => {
    const frames = await collect(streamOf('event: a\r\ndata: 1\r\n\r\n'))
    expect(frames).toEqual([{ event: 'a', data: '1' }])
  })

  it('throws when the response carried no body', async () => {
    await expect(collect(new Response(null))).rejects.toThrow(/no body/)
  })
})
