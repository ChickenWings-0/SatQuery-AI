/**
 * Submitting a run and streaming it into the job store.
 *
 * `/v1/jobs` + SSE rather than `/v1/analyze`: change detection plus VLM
 * synthesis pushes past a comfortable synchronous wait, and the whole UI is
 * built to show the run happening (API_CONTRACT §8.3).
 *
 * The hard part is not the happy path, it is every way a long-lived stream can
 * end without saying so. Three rules hold the state machine together:
 *
 *   1. **The phase always terminates.** Any failure after the 202 calls
 *      `markFailed()`, so `streaming` is never the resting state. Without this
 *      a dropped connection disabled the composer permanently and the only
 *      recovery was reloading the page.
 *   2. **The last attempt is remembered.** Every failure is offered with a
 *      retry that re-submits the same question against the same files, because
 *      "try again" that makes the user retype the question is not a recovery.
 *   3. **Nothing outlives the component.** The in-flight controller is aborted
 *      on unmount, and a settled attempt only writes state if it is still the
 *      current one — a fast second question must not be overwritten by the
 *      first one's failure arriving late.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { SatQueryError, createJob, streamJob } from '@/api/client'
import { ContractViolation } from '@/api/events'
import { useJobStore } from '@/state/job'
import { useFocusStore } from '@/state/focus'
import { useUiStore } from '@/state/ui'

export interface RunFailure {
  message: string
  /** Worth showing a retry button for — transport, 429, 5xx. */
  retryable: boolean
}

function describe(caught: unknown): RunFailure {
  if (caught instanceof SatQueryError) {
    return { message: caught.display, retryable: caught.retryable }
  }
  if (caught instanceof ContractViolation) {
    return {
      // A contract violation is not the operator's fault and retrying will
      // reproduce it, so it names the mismatch instead of offering a button.
      message: `${caught.message} This client and the API disagree about the event contract.`,
      retryable: false,
    }
  }
  return { message: 'The run could not be started. Is the API running?', retryable: true }
}

export function useRun() {
  const [failure, setFailure] = useState<RunFailure | null>(null)
  const inFlight = useRef<AbortController | null>(null)
  const lastQuery = useRef<string | null>(null)

  // An unmount mid-run would otherwise leave the fetch, the reader and the
  // watchdog timer alive, all writing into a store nothing is reading.
  useEffect(() => () => inFlight.current?.abort(), [])

  const submit = useCallback(async (raw: string) => {
    const query = raw.trim()
    const files = useUiStore.getState().files.map((entry) => entry.file)
    if (files.length === 0 || query === '') return

    inFlight.current?.abort()
    const controller = new AbortController()
    inFlight.current = controller
    lastQuery.current = query
    const current = () => inFlight.current === controller

    setFailure(null)
    useJobStore.getState().reset()
    useFocusStore.getState().resetForNewRun()

    let accepted = false
    try {
      const job = await createJob(files, query, undefined, controller.signal)
      accepted = true
      useUiStore.getState().rememberRun({ traceId: job.job_id, query, at: Date.now() })
      await streamJob(
        job.job_id,
        { onEvent: (event) => useJobStore.getState().apply(event) },
        controller.signal,
      )
    } catch (caught) {
      // Aborts are this hook's own doing — a new question or an unmount.
      if (controller.signal.aborted || !current()) return
      // Past the 202 the run is live in the store, so the phase has to be
      // closed out or the UI waits for a stream that has already gone.
      if (accepted) useJobStore.getState().markFailed()
      setFailure(describe(caught))
    }
  }, [])

  const retry = useCallback(() => {
    const query = lastQuery.current
    if (query) void submit(query)
  }, [submit])

  /** Abandon the current run. The server keeps working; we stop watching. */
  const cancel = useCallback(() => {
    inFlight.current?.abort()
    inFlight.current = null
    useJobStore.getState().markFailed()
    setFailure({ message: 'Run cancelled. The question is still in the box.', retryable: true })
  }, [])

  const dismiss = useCallback(() => setFailure(null), [])

  return { submit, retry, cancel, dismiss, failure }
}
