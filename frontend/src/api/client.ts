/**
 * Typed wrappers over the SatQuery API.
 *
 * Base URL comes from `VITE_API_BASE`; it defaults to empty, which makes every
 * request same-origin and lets Vite's dev proxy forward `/v1` to the FastAPI on
 * :8000 (see `vite.config.ts`). Same-origin in dev means SSE and multipart
 * behave in development exactly as they will in the packaged build.
 */
import { ContractViolation, parseJobEvent, type JobEvent } from '@/api/events'
import { readFrames } from '@/api/sse'
import type {
  AnalyzeResponse,
  ApiError,
  ArtifactRef,
  AuditTrace,
  HealthResponse,
  JobAccepted,
  JobStatusResponse,
  RegistryResponse,
  ValidateResponse,
} from '@/api/types'

const BASE = import.meta.env['VITE_API_BASE'] ?? ''

/**
 * Deadlines, in milliseconds.
 *
 * A JSON GET that has not answered in 20 seconds is not slow, it is gone —
 * `/v1/health` is a dictionary lookup. The multipart routes get five minutes
 * because the deadline has to cover the *upload*, and `max_upload_mb` defaults
 * to 512 MB per image over whatever link the operator has. Without any deadline
 * at all a half-open socket leaves the composer disabled and the spinner
 * spinning until the tab is closed, which is what this codebase did.
 */
const TIMEOUT = { json: 20_000, upload: 300_000 } as const

/**
 * How long the event stream may go completely silent before it is declared
 * dead. The server sends `: ping` every 15 s specifically so a long GPU step
 * cannot be mistaken for a dropped connection (§5), so four missed heartbeats
 * is a connection that will never speak again — not a slow tool.
 */
export const STREAM_STALL_MS = 60_000

/** An error carrying the §6 envelope, so `message` and `hint` reach the UI. */
export class SatQueryError extends Error {
  readonly status: number
  readonly error: ApiError | null
  /**
   * True when the request never reached a server: offline, DNS, refused
   * connection, TLS, CORS, or our own deadline. `status` is 0 for all of them.
   * The UI needs the distinction because the recovery differs — "start the
   * API" versus "the API said no".
   */
  readonly transport: boolean

  constructor(status: number, error: ApiError | null, message: string, transport = false) {
    super(message)
    this.name = 'SatQueryError'
    this.status = status
    this.error = error
    this.transport = transport
  }

  /** An actionable next step, when the server offered one. */
  get hint(): string | null {
    return this.error?.hint ?? null
  }

  /** `message` and `hint` as the one string the error panels render. */
  get display(): string {
    return [this.message, this.hint].filter(Boolean).join(' ')
  }

  /** Worth offering a retry button for. A 400 is not; a 503 is. */
  get retryable(): boolean {
    return this.transport || this.status === 429 || this.status >= 500
  }
}

/**
 * What to say when the server returned a status but no §6 envelope — a proxy
 * error page, a 502 from nginx, a crash before the handler. Each line names
 * what happened *and* what to do about it; "Request failed with 503" does
 * neither.
 */
const STATUS_TEXT: Record<number, string> = {
  400: 'The request was rejected as malformed.',
  401: 'This SatQuery instance requires credentials that this session does not have.',
  403: 'This SatQuery instance refused the request.',
  404: 'That resource no longer exists on the server.',
  413: 'The imagery is larger than this server accepts.',
  415: 'The server cannot read that file type. GeoTIFF, PNG and JPEG are accepted.',
  422: 'The server could not process that input.',
  429: 'Too many requests. Wait a moment and try again.',
  500: 'The server failed while handling the request.',
  502: 'The API is not reachable through the proxy in front of it.',
  503: 'The API is up but not ready to serve — it may still be loading models.',
  504: 'The API did not answer in time.',
}

async function toError(response: Response): Promise<SatQueryError> {
  let envelope: ApiError | null = null
  try {
    const body = (await response.json()) as { error?: ApiError }
    envelope = body.error ?? null
  } catch {
    envelope = null
  }
  return new SatQueryError(
    response.status,
    envelope,
    // §6: `message` is user-facing and safe to display verbatim.
    envelope?.message ??
      STATUS_TEXT[response.status] ??
      `The server answered ${response.status}.`,
  )
}

/** The one place a thrown `fetch` becomes something the UI can render. */
function toTransportError(_cause?: unknown): SatQueryError {
  const offline = typeof navigator !== 'undefined' && navigator.onLine === false
  return new SatQueryError(
    0,
    null,
    offline
      ? 'No network connection. Reconnect and try again.'
      : 'Could not reach the SatQuery API. Check that it is running.',
    true,
  )
}

/**
 * Run one `fetch` under both the caller's signal and a deadline.
 *
 * `AbortSignal.any` and `AbortSignal.timeout` would do this in two lines, but
 * they are absent from happy-dom, which would make every streaming and upload
 * path untestable outside a browser — the same reason `@microsoft/fetch-event-source`
 * was rejected in `@/api/sse`. So the composition is written out.
 */
async function send(
  path: string,
  init: RequestInit,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController()
  // A property rather than a `let`: the flag is written inside a callback and
  // read after it, and TypeScript's flow analysis would otherwise narrow a
  // local to its initialiser and call the later comparison unreachable.
  const fired = { timeout: false }

  const timer = setTimeout(() => {
    fired.timeout = true
    controller.abort()
  }, timeoutMs)
  const forward = () => controller.abort()
  signal?.addEventListener('abort', forward, { once: true })

  try {
    return await fetch(`${BASE}${path}`, { ...init, signal: controller.signal })
  } catch (caught) {
    // The caller aborting is not a failure: let it propagate untranslated so
    // `signal.aborted` still identifies it upstream.
    if (signal?.aborted) throw caught
    if (fired.timeout) {
      throw new SatQueryError(
        0,
        null,
        `The API did not answer within ${Math.round(timeoutMs / 1000)} seconds.`,
        true,
      )
    }
    throw toTransportError(caught)
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', forward)
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await send(
    path,
    { headers: { accept: 'application/json' } },
    signal,
    TIMEOUT.json,
  )
  if (!response.ok) throw await toError(response)
  try {
    return (await response.json()) as T
  } catch {
    throw new SatQueryError(0, null, 'The API returned a response this client could not read.')
  }
}

async function postForm<T>(path: string, body: FormData, signal?: AbortSignal): Promise<T> {
  const response = await send(path, { method: 'POST', body }, signal, TIMEOUT.upload)
  if (!response.ok) throw await toError(response)
  try {
    return (await response.json()) as T
  } catch {
    throw new SatQueryError(0, null, 'The API returned a response this client could not read.')
  }
}

/** Options accepted by `/v1/analyze` and `/v1/jobs` (API_CONTRACT §4.1). */
export interface AnalyzeOptions {
  pair_type_hint?: string
  roles?: Record<string, string>
  include_trace?: boolean
  include_rendered_views?: boolean
  enable_tools?: string[] | null
  disable_tools?: string[]
  seed?: number
}

function submission(files: File[], query: string | null, options?: AnalyzeOptions): FormData {
  const form = new FormData()
  for (const file of files) form.append('images', file, file.name)
  if (query !== null) form.append('query', query)
  if (options) form.append('options', JSON.stringify(options))
  return form
}

// ------------------------------------------------------------------ endpoints

export const health = (signal?: AbortSignal) => getJson<HealthResponse>('/v1/health', signal)

export const registry = (signal?: AbortSignal) => getJson<RegistryResponse>('/v1/registry', signal)

export const trace = (traceId: string, signal?: AbortSignal) =>
  getJson<AuditTrace>(`/v1/traces/${traceId}`, signal)

export const jobStatus = (jobId: string, signal?: AbortSignal) =>
  getJson<JobStatusResponse>(`/v1/jobs/${jobId}`, signal)

/**
 * GPU-free pre-flight. Call this on file *select*, before the user types:
 * `supported_tasks` says which questions are worth offering (§8.2).
 */
export const validate = (
  files: File[],
  options?: Pick<AnalyzeOptions, 'pair_type_hint' | 'roles'>,
  signal?: AbortSignal,
) => postForm<ValidateResponse>('/v1/validate', submission(files, null, options), signal)

/** Synchronous analysis. Prefer {@link createJob} for anything bi-temporal. */
export const analyze = (
  files: File[],
  query: string,
  options?: AnalyzeOptions,
  signal?: AbortSignal,
) => postForm<AnalyzeResponse>('/v1/analyze', submission(files, query, options), signal)

/** Queue an analysis. The returned `job_id` *is* the eventual `trace_id`. */
export const createJob = (
  files: File[],
  query: string,
  options?: AnalyzeOptions,
  signal?: AbortSignal,
) => postForm<JobAccepted>('/v1/jobs', submission(files, query, options), signal)

/**
 * Ask the server to stop a running job (`DELETE /v1/jobs/{id}`, §4.10).
 *
 * Best-effort on both sides: the server lets the tool step in flight finish
 * and skips the rest, then ends the job with an `error{JOB_CANCELLED}` event
 * that closes any open stream. A 409 means the job had already finished — not
 * a failure worth showing, so it resolves rather than throws.
 */
export async function cancelJob(jobId: string, signal?: AbortSignal): Promise<JobStatusResponse | null> {
  const response = await send(
    `/v1/jobs/${jobId}`,
    { method: 'DELETE', headers: { accept: 'application/json' } },
    signal,
    TIMEOUT.json,
  )
  if (response.status === 409) return null
  if (!response.ok) throw await toError(response)
  try {
    return (await response.json()) as JobStatusResponse
  } catch {
    return null
  }
}

/** The URL of one artifact. Immutable and cacheable for ever (§4.6). */
export function artifactUrl(ref: ArtifactRef): string | null {
  return ref.url ? `${BASE}${ref.url}` : null
}

// ----------------------------------------------------------------------- SSE

export interface StreamHandlers {
  /** `id` is the frame's sequence number (§5), or null on a stream without ids. */
  onEvent: (event: JobEvent, id: string | null) => void
  /** Transport failures and contract violations both land here. */
  onError?: (error: unknown) => void
}

export interface StreamOptions {
  /**
   * The last event id this client has already applied. Sent as
   * `Last-Event-ID`, so the server replays only what follows it (§5). Omit
   * for a fresh subscription, which replays the whole history.
   */
  lastEventId?: string | null
}

/**
 * Subscribe to a job's event stream until it terminates.
 *
 * The server buffers every event and replays it to a late subscriber, so it is
 * always safe to connect after the 202 has come back — which is the normal
 * case, since the client needs the `job_id` first. Returns when the stream ends,
 * which the server guarantees happens after `done` or `error`.
 *
 * Two things can go wrong that no status code reports, and both used to leave
 * the UI stuck mid-run for ever:
 *
 *   - **The stream ends without a terminal event.** A killed worker, a proxy
 *     that closed the connection, a server restart. The loop simply finished,
 *     the phase stayed `streaming`, and the composer stayed disabled. That is
 *     now an explicit error.
 *   - **The connection stops delivering without ending.** A half-open socket
 *     reads as an infinitely slow GPU step. The heartbeat is what distinguishes
 *     them, so silence past {@link STREAM_STALL_MS} aborts the read.
 *
 * The stream does not reconnect *itself*: the decision belongs to the caller,
 * who knows whether the job is worth reattaching to (see `@/thread/resume`).
 * What it does provide is the means — every frame's `id` reaches the handler,
 * and `options.lastEventId` asks the server for only what follows it.
 */
export async function streamJob(
  jobId: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
  options: StreamOptions = {},
): Promise<void> {
  const controller = new AbortController()
  const abort = () => controller.abort()
  signal?.addEventListener('abort', abort, { once: true })

  const fired = { stall: false }
  let watchdog: ReturnType<typeof setTimeout> | undefined
  const touch = () => {
    clearTimeout(watchdog)
    watchdog = setTimeout(() => {
      fired.stall = true
      controller.abort()
    }, STREAM_STALL_MS)
  }

  let terminated = false
  try {
    touch()
    const headers: Record<string, string> = { accept: 'text/event-stream' }
    if (options.lastEventId) headers['Last-Event-ID'] = options.lastEventId
    const response = await fetch(`${BASE}/v1/jobs/${jobId}/events`, {
      headers,
      signal: controller.signal,
    })
    if (!response.ok) throw await toError(response)

    for await (const frame of readFrames(response, touch)) {
      // A contract violation is surfaced, not swallowed: §2 is explicit that an
      // unknown value must be reported rather than silently coerced.
      const event = parseJobEvent(frame.event, frame.data)
      handlers.onEvent(event, frame.id)
      if (event.type === 'done' || event.type === 'error') {
        terminated = true
        break
      }
    }

    if (!terminated) {
      throw new SatQueryError(
        0,
        null,
        'The run stopped reporting before it finished. The server may have restarted.',
        true,
      )
    }
  } catch (error) {
    if (fired.stall) {
      const stall = new SatQueryError(
        0,
        null,
        `No word from the run for ${Math.round(STREAM_STALL_MS / 1000)} seconds. The connection was lost.`,
        true,
      )
      handlers.onError?.(stall)
      throw stall
    }
    // An abort is the caller's own doing, not a failure worth reporting.
    if (controller.signal.aborted) return
    const failure =
      error instanceof SatQueryError || error instanceof ContractViolation
        ? error
        : toTransportError(error)
    handlers.onError?.(failure)
    throw failure
  } finally {
    clearTimeout(watchdog)
    controller.abort()
    signal?.removeEventListener('abort', abort)
  }
}
