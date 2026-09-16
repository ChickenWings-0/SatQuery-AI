/**
 * The five-second rule, in one place. Every online call the Maps page makes
 * goes through here: a search that has not answered in five seconds is not
 * slow, it is the venue Wi-Fi, and the HUD needs to say so before the judge
 * wonders whether the button worked.
 */
export const ONLINE_DEADLINE_MS = 5_000

export class OfflineError extends Error {
  /** True when the request never reached a server (timeout, DNS, refused). */
  readonly transport = true
  constructor(message = 'The network did not answer.') {
    super(message)
    this.name = 'OfflineError'
  }
}

export class UpstreamError extends Error {
  readonly status: number
  readonly retryAfterMs: number | null
  constructor(status: number, message: string, retryAfterMs: number | null = null) {
    super(message)
    this.name = 'UpstreamError'
    this.status = status
    this.retryAfterMs = retryAfterMs
  }
}

/**
 * `fetch` with a deadline and an honest error taxonomy. The caller's own
 * signal still aborts; its abort propagates untranslated so `signal.aborted`
 * identifies it upstream.
 */
export async function fetchWithDeadline(
  url: string,
  init: RequestInit = {},
  signal?: AbortSignal,
  deadlineMs = ONLINE_DEADLINE_MS,
): Promise<Response> {
  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, deadlineMs)
  const forward = () => controller.abort()
  signal?.addEventListener('abort', forward, { once: true })
  try {
    return await fetch(url, { ...init, signal: controller.signal })
  } catch (caught) {
    if (signal?.aborted) throw caught
    throw new OfflineError(
      timedOut ? `No answer within ${Math.round(deadlineMs / 1000)} seconds.` : 'The network did not answer.',
    )
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', forward)
  }
}

export function retryAfterOf(response: Response): number | null {
  const raw = response.headers.get('Retry-After')
  if (!raw) return null
  const seconds = Number(raw)
  if (Number.isFinite(seconds)) return seconds * 1000
  const at = Date.parse(raw)
  return Number.isFinite(at) ? Math.max(0, at - Date.now()) : null
}

export function isOffline(error: unknown): error is OfflineError {
  return error instanceof OfflineError || (typeof navigator !== 'undefined' && navigator.onLine === false)
}
