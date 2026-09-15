/**
 * Submitting a run and streaming it into the job store.
 *
 * `/v1/jobs` + SSE rather than `/v1/analyze`: change detection plus VLM
 * synthesis pushes past a comfortable synchronous wait, and the whole UI is
 * built to show the run happening (API_CONTRACT §8.3).
 *
 * The hard part is not the happy path, it is every way a long-lived stream can
 * end without saying so. Four rules hold the state machine together:
 *
 *   1. **The phase always terminates.** Any failure after the 202 ends in
 *      `failed` or `succeeded`, never a resting `streaming`/`reconnecting`.
 *      Without this a dropped connection disabled the composer permanently.
 *   2. **A lost stream is not a lost run.** Past the 202 the job lives on the
 *      server, which replays its history to a late subscriber. A transport
 *      failure or stall therefore *reattaches* (`@/thread/resume`) and only
 *      offers "try again" — a fresh upload and a fresh GPU run — when the
 *      server no longer knows the job, or every reattach attempt failed.
 *   3. **Cancel means cancel.** The cancel button, a new question while one is
 *      running, and unmount all send `DELETE /v1/jobs/{id}`; the server stops
 *      at the next step boundary and closes the stream with `JOB_CANCELLED`.
 *      Only when the DELETE itself fails does the client close out locally.
 *   4. **Nothing outlives the component.** The in-flight controller is aborted
 *      on unmount, and a settled attempt only writes state if it is still the
 *      current one — a fast second question must not be overwritten by the
 *      first one's failure arriving late.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { SatQueryError, cancelJob, createJob, streamJob } from '@/api/client'
import { ContractViolation, type JobEvent } from '@/api/events'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'
import { notify } from '@/state/notifications'
import { composeQuery, useSettingsStore } from '@/state/settings'
import { useUiStore } from '@/state/ui'
import { resumeRun } from '@/thread/resume'

export interface RunFailure {
  message: string
  /** Worth showing a retry button for — transport, 429, 5xx. */
  retryable: boolean
}

export interface Reattach {
  attempt: number
  max: number
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

/**
 * Tell the notification centre a run settled. Success is news only when the
 * user is not looking at the stage; failure is always news. Both are real
 * events, and the badge is the only place a judge who tabbed to Maps learns
 * the run they started came back.
 */
function announce(jobId: string, outcome: 'succeeded' | 'failed', detail?: string): void {
  const ui = useUiStore.getState()
  const query = ui.recentRuns.find((run) => run.traceId === jobId)?.query ?? 'the run'
  const short = query.length > 48 ? `${query.slice(0, 47)}…` : query
  if (outcome === 'succeeded') {
    if (ui.section === 'explore' && document.visibilityState === 'visible') return
    notify({
      kind: 'run',
      tone: 'ok',
      title: 'Run finished',
      body: `“${short}”`,
      action: { label: 'Open run', run: () => useUiStore.getState().openRun(jobId) },
    })
    return
  }
  notify({ kind: 'run', tone: 'fail', title: 'Run failed', body: detail ?? `“${short}”` })
}

/** Fold one event into the stores, and settle the history entry on a terminal one. */
function applyEvent(event: JobEvent, jobId: string): void {
  useJobStore.getState().apply(event)
  if (event.type === 'done') {
    useUiStore.getState().settleRun(jobId, 'succeeded')
    announce(jobId, 'succeeded')
  } else if (event.type === 'error') {
    useUiStore.getState().settleRun(jobId, 'failed')
    announce(jobId, 'failed', event.data.message)
  }
}

export function useRun() {
  const [failure, setFailure] = useState<RunFailure | null>(null)
  const [reattach, setReattach] = useState<Reattach | null>(null)
  const inFlight = useRef<AbortController | null>(null)
  const lastQuery = useRef<string | null>(null)
  /** The job the current attempt is watching, once the 202 has landed. */
  const liveJob = useRef<string | null>(null)

  /**
   * Tell the server to stop the job this hook is watching, if any.
   *
   * Fire-and-forget with its own controller: the UI must not wait on the
   * server to feel responsive, and the hook's own abort must not cancel the
   * DELETE it is trying to send.
   */
  const stopServerJob = useCallback((): Promise<boolean> => {
    const jobId = liveJob.current
    liveJob.current = null
    if (!jobId) return Promise.resolve(false)
    return cancelJob(jobId).then(
      () => true,
      () => false,
    )
  }, [])

  // An unmount mid-run would otherwise leave the fetch, the reader and the
  // watchdog timer alive, all writing into a store nothing is reading — and
  // the server working on an answer nobody will read.
  useEffect(
    () => () => {
      inFlight.current?.abort()
      void stopServerJob()
    },
    [stopServerJob],
  )

  const submit = useCallback(
    async (raw: string) => {
      const query = raw.trim()
      const files = useUiStore.getState().files.map((entry) => entry.file)
      if (files.length === 0 || query === '') return

      // A previous run still streaming is abandoned on both ends.
      inFlight.current?.abort()
      void stopServerJob()
      const controller = new AbortController()
      inFlight.current = controller
      lastQuery.current = query
      const current = () => inFlight.current === controller

      setFailure(null)
      setReattach(null)
      useJobStore.getState().reset()
      useFocusStore.getState().resetForNewRun()

      let jobId: string | null = null
      let lastEventId: string | null = null
      const onEvent = (event: JobEvent, id: string | null) => {
        // A chunk already in the reader's buffer can surface after this
        // attempt was abandoned; applying it would drag a `failed` phase back
        // to `streaming`.
        if (controller.signal.aborted || !current()) return
        if (id !== null) lastEventId = id
        if (jobId) applyEvent(event, jobId)
      }

      try {
        // Custom instructions ride ahead of the question inside `query`, and
        // a non-zero seed goes in `options` — both are contract fields, and
        // the history entry keeps the question the user actually typed.
        const settings = useSettingsStore.getState()
        const job = await createJob(
          files,
          composeQuery(query, settings.customInstructions),
          settings.seed > 0 ? { seed: settings.seed } : undefined,
          controller.signal,
        )
        jobId = job.job_id
        liveJob.current = jobId
        useUiStore
          .getState()
          .rememberRun({ traceId: jobId, query, at: Date.now(), outcome: 'running' })
        await streamJob(jobId, { onEvent }, controller.signal)
        if (current()) liveJob.current = null
        return
      } catch (caught) {
        // Aborts are this hook's own doing — a new question or an unmount.
        if (controller.signal.aborted || !current()) return
        if (!jobId) {
          // The request never became a job: nothing to reattach to.
          setFailure(describe(caught))
          return
        }
        if (caught instanceof ContractViolation) {
          // Reattaching would reproduce it.
          useJobStore.getState().markFailed()
          setFailure(describe(caught))
          return
        }
      }

      // Past the 202 and the stream is gone: the job is presumed alive.
      useJobStore.getState().markReconnecting()
      const outcome = await resumeRun(
        jobId,
        lastEventId,
        {
          onEvent,
          onAttempt: (attempt, max) => {
            if (current()) setReattach({ attempt, max })
          },
        },
        controller.signal,
      )
      if (controller.signal.aborted || !current()) return
      setReattach(null)
      if (outcome.kind === 'completed') {
        liveJob.current = null
        return
      }
      useJobStore.getState().markFailed()
      useUiStore.getState().settleRun(jobId, 'failed')
      if (outcome.kind === 'gone') {
        liveJob.current = null
        setFailure({
          message: 'The server no longer knows this run — it may have restarted. Ask again to start a new one.',
          retryable: true,
        })
      } else {
        setFailure({
          message:
            'Lost the connection to the run and could not reattach. The server may still be working; ask again to start a new one.',
          retryable: true,
        })
      }
    },
    [stopServerJob],
  )

  const retry = useCallback(() => {
    const query = lastQuery.current
    if (query) void submit(query)
  }, [submit])

  /**
   * Stop the current run on the server and stop watching it.
   *
   * The stream is left open on purpose: the server answers the DELETE by
   * closing the job with `error{JOB_CANCELLED}`, and letting that event arrive
   * closes the phase through the reducer like any other terminal event. Only
   * if the DELETE fails — the job is unknown, the server is gone — does the
   * client close out locally and say so.
   */
  const cancel = useCallback(() => {
    const controller = inFlight.current
    const jobId = liveJob.current
    if (!jobId) {
      controller?.abort()
      inFlight.current = null
      useJobStore.getState().markFailed()
      setFailure({ message: 'Run cancelled. The question is still in the box.', retryable: true })
      return
    }
    void stopServerJob().then((stopped) => {
      if (inFlight.current !== controller) return
      if (stopped) {
        setFailure({ message: 'Run cancelled. The question is still in the box.', retryable: true })
        return
      }
      controller?.abort()
      inFlight.current = null
      useJobStore.getState().markFailed()
      useUiStore.getState().settleRun(jobId, 'failed')
      setFailure({
        message:
          'Stopped watching the run. The server did not confirm the cancel and may still be working.',
        retryable: true,
      })
    })
  }, [stopServerJob])

  const dismiss = useCallback(() => setFailure(null), [])

  return { submit, retry, cancel, dismiss, failure, reattach }
}
