/**
 * A minimal server-sent-events reader over `fetch`.
 *
 * `EventSource` cannot be used: it is GET-only, so it cannot carry the
 * multipart body that creates a job, and it reconnects on its own schedule,
 * which would replay a finished run. `@microsoft/fetch-event-source` solves the
 * first problem but is browser-only — it reaches for `window` — which makes the
 * whole streaming path untestable outside a DOM. Since what we need is one
 * abortable read of a stream that terminates itself, it is written here.
 *
 * Frame grammar (WHATWG server-sent events), only the parts the contract uses:
 *
 *   - Frames are separated by a blank line.
 *   - `event: <name>` names the frame; `data: <text>` carries it.
 *   - A frame may carry several `data:` lines, joined with "\n".
 *   - A line starting with ":" is a comment — the `: ping` heartbeat of §5.
 *
 * Chunk boundaries fall anywhere, including mid-frame and mid-line, so the
 * buffer is only ever consumed up to the last complete frame.
 */

export interface SseFrame {
  event: string
  data: string
}

const FRAME_BOUNDARY = /\r\n\r\n|\n\n|\r\r/

/** Parse one complete frame's raw text. Returns null for comment-only frames. */
export function parseFrame(raw: string): SseFrame | null {
  let event = 'message'
  const data: string[] = []

  for (const line of raw.split(/\r\n|\n|\r/)) {
    if (line === '' || line.startsWith(':')) continue // heartbeat or padding
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    // "If value starts with a space, remove it" — the spec's one quirk.
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)

    if (field === 'event') event = value
    else if (field === 'data') data.push(value)
    // `id` and `retry` are unused: this stream never reconnects.
  }

  return data.length === 0 ? null : { event, data: data.join('\n') }
}

/**
 * Read `response.body` as SSE frames.
 *
 * `onActivity` fires for every chunk that arrives, including the `: ping`
 * heartbeats that produce no frame. That distinction is the whole point of the
 * callback: a server that is alive but slow sends comments and no data, and
 * without a signal for them a stall watchdog upstream cannot tell "thinking"
 * from "the TCP connection died three minutes ago". Frames alone are not enough.
 *
 * @throws Error if the response carries no body.
 */
export async function* readFrames(
  response: Response,
  onActivity?: () => void,
): AsyncGenerator<SseFrame> {
  if (!response.body) throw new Error('The event stream carried no body.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      onActivity?.()
      buffer += decoder.decode(value, { stream: true })

      for (;;) {
        const match = FRAME_BOUNDARY.exec(buffer)
        if (!match) break
        const raw = buffer.slice(0, match.index)
        buffer = buffer.slice(match.index + match[0].length)
        const frame = parseFrame(raw)
        if (frame) yield frame
      }
    }

    // A server that closes without a trailing blank line still meant to send
    // that last frame, so it is not dropped on the floor.
    const trailing = parseFrame(buffer)
    if (trailing) yield trailing
  } finally {
    // Releasing before cancel avoids "reader has been released" on an early
    // return from the consumer's for-await.
    reader.releaseLock()
    await response.body.cancel().catch(() => undefined)
  }
}
