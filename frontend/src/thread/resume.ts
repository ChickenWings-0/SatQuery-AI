/**
 * Reattaching to a run after the stream to it is lost.
 *
 * A dropped SSE connection used to be treated as a failed run: the phase went
 * to `failed` and the only recovery offered was "try again", which re-uploaded
 * the files and started a fresh 20-60 s GPU job. But the job was never in
 * trouble — only our socket was — and the server keeps every event it has
 * published and replays them to a late subscriber. Reattaching is the correct
 * recovery, and this module is the bounded loop that does it.
 *
 * The loop is deliberately not "reconnect for ever":
 *
 *   1. Poll `GET /v1/jobs/{id}` first. It is cheap, and it settles the two
 *      cases that need no stream at all: the job already finished (synthesise
 *      the terminal event from the snapshot) and the job is gone (404 — the
 *      server restarted or evicted it, and re-running is the only option).
 *   2. Otherwise reopen the stream with the last event id we applied, so the
 *      server sends only what followed it.
 *   3. Back off between attempts and give up after `MAX_ATTEMPTS`, at which
 *      point the caller is told and offers the resubmit.
 *
 * Dependencies are injected so the loop is testable without timers or MSW.
 */
import { SatQueryError, jobStatus as fetchJobStatus, streamJob as openStream } from '@/api/client'
import type { JobEvent } from '@/api/events'
import type { JobStatusResponse } from '@/api/types'

export const MAX_ATTEMPTS = 5
/** Backoff before each attempt, in ms; the last value repeats. */
export const BACKOFF_MS = [1_000, 2_000, 4_000, 8_000, 8_000] as const

export type ResumeOutcome =
  /** The stream delivered a terminal event, or the snapshot carried one. */
  | { kind: 'completed' }
  /** The server no longer knows the job; only a fresh submission can recover. */
  | { kind: 'gone'; error: SatQueryError }
  /** Every attempt failed on transport; the job may still be running. */
  | { kind: 'exhausted'; error: unknown }

export interface ResumeDeps {
  jobStatus: (jobId: string, signal?: AbortSignal) => Promise<JobStatusResponse>
  streamJob: typeof openStream
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>
}

export interface ResumeCallbacks {
  onEvent: (event: JobEvent, id: string | null) => void
  /** Fires before each attempt; `attempt` is 1-based. For the banner. */
  onAttempt?: (attempt: number, max: number) => void
}

function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(done, ms)
    function done() {
      signal?.removeEventListener('abort', done)
      clearTimeout(timer)
      resolve()
    }
    signal?.addEventListener('abort', done, { once: true })
  })
}

export const defaultDeps: ResumeDeps = {
  jobStatus: fetchJobStatus,
  streamJob: openStream,
  sleep: defaultSleep,
}

/** The terminal event a finished job's snapshot stands for, if it is finished. */
export function terminalEventOf(status: JobStatusResponse): JobEvent | null {
  if (status.status === 'succeeded' && status.result) {
    return { type: 'done', data: status.result }
  }
  if (status.status === 'failed' && status.error) {
    return { type: 'error', data: status.error }
  }
  return null
}

/**
 * Reattach to `jobId`, applying events through `callbacks.onEvent`.
 *
 * `lastEventId` is the last frame id already applied; it is updated from the
 * events this loop delivers, so a second drop resumes from the right place.
 * Resolves with the outcome; never throws except on the caller's own abort,
 * which resolves as `completed` so the caller's abort handling stays the
 * single source of truth.
 */
export async function resumeRun(
  jobId: string,
  lastEventId: string | null,
  callbacks: ResumeCallbacks,
  signal?: AbortSignal,
  deps: ResumeDeps = defaultDeps,
): Promise<ResumeOutcome> {
  let cursor = lastEventId
  let lastError: unknown = null

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    await deps.sleep(BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length) - 1] ?? 8_000, signal)
    if (signal?.aborted) return { kind: 'completed' }
    callbacks.onAttempt?.(attempt, MAX_ATTEMPTS)

    // Step 1: where does the server think the job is?
    let status: JobStatusResponse
    try {
      status = await deps.jobStatus(jobId, signal)
    } catch (caught) {
      if (signal?.aborted) return { kind: 'completed' }
      if (caught instanceof SatQueryError && caught.status === 404) {
        return { kind: 'gone', error: caught }
      }
      lastError = caught
      continue
    }
    const terminal = terminalEventOf(status)
    if (terminal) {
      callbacks.onEvent(terminal, null)
      return { kind: 'completed' }
    }

    // Step 2: still running — reopen the stream from where we left off.
    let terminated = false
    try {
      await deps.streamJob(
        jobId,
        {
          onEvent: (event, id) => {
            if (id !== null) cursor = id
            if (event.type === 'done' || event.type === 'error') terminated = true
            callbacks.onEvent(event, id)
          },
        },
        signal,
        { lastEventId: cursor },
      )
      if (signal?.aborted || terminated) return { kind: 'completed' }
      // The stream closed cleanly without a terminal event: try again.
      lastError = new SatQueryError(0, null, 'The run stopped reporting before it finished.', true)
    } catch (caught) {
      if (signal?.aborted) return { kind: 'completed' }
      if (caught instanceof SatQueryError && caught.status === 404) {
        return { kind: 'gone', error: caught }
      }
      lastError = caught
    }
  }
  return { kind: 'exhausted', error: lastError }
}
