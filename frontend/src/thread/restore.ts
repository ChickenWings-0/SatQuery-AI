/**
 * Reopen a finished run in the workspace, from the server's stored trace.
 *
 * The page keeps a saved run's summary on this device (IndexedDB) and its
 * session history in memory; the run itself — answer, evidence views,
 * pipeline — lives on the server as an audit trace, and survives a reload or a
 * server restart. `GET /v1/jobs/{id}` rebuilds a finished job from that trace,
 * so reopening is a replay: the same `queued → plan → step_* → done` sequence
 * a live run streams, synthesised from the trace and fed through the one job
 * reducer, so the restored workspace is exactly what the run showed.
 */
import { jobStatus } from '@/api/client'
import type { JobEvent } from '@/api/events'
import type { AnalyzeResponse } from '@/api/types'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'
import { useUiStore } from '@/state/ui'

/** The event sequence a live run would have streamed, rebuilt from its result. */
export function replayEvents(result: AnalyzeResponse): JobEvent[] {
  const events: JobEvent[] = [{ type: 'queued', data: { job_id: result.trace_id } }]
  const trace = result.trace
  if (trace) {
    events.push({ type: 'plan', data: { steps: trace.plan.steps ?? [] } })
    for (const execution of trace.executions) {
      events.push({
        type: 'step_started',
        data: { step: execution.step, tool: execution.tool, est_ms: 0 },
      })
      events.push({
        type: 'step_completed',
        data: {
          step: execution.step,
          status: execution.status,
          duration_ms: execution.duration_ms,
          confidence: execution.confidence,
          output_refs: execution.output_refs ?? [],
        },
      })
    }
  }
  const artifacts = result.artifacts?.length ? result.artifacts : (trace?.artifacts ?? [])
  for (const artifact of artifacts) events.push({ type: 'artifact', data: artifact })
  events.push({ type: 'done', data: { ...result, artifacts } })
  return events
}

/**
 * Load run *traceId* into the workspace. Throws when the server cannot
 * produce it (pruned, or the API unreachable); the caller decides the fallback.
 */
export async function restoreRun(traceId: string, signal?: AbortSignal): Promise<void> {
  const status = await jobStatus(traceId, signal)
  const result = status.result
  if (status.status !== 'succeeded' || !result) {
    throw new Error(`run ${traceId} did not finish, so there is nothing to reopen`)
  }

  const job = useJobStore.getState()
  job.reset()
  useFocusStore.getState().resetForNewRun()
  for (const event of replayEvents(result)) job.apply(event)

  const ui = useUiStore.getState()
  const trace = result.trace
  if (trace) {
    // The manifests are what the viewer reads sensor, GSD and dates from.
    ui.setValidation({
      inputs: trace.inputs,
      compatibility: trace.compatibility,
      supported_tasks: [],
      warnings: [],
      previews: {},
    })
    const query = trace.query.raw
    useFocusStore.getState().setDraft(query)
    if (!ui.recentRuns.some((run) => run.traceId === traceId)) {
      ui.rememberRun({
        traceId,
        query,
        at: Date.parse(trace.created_at) || Date.now(),
        outcome: 'succeeded',
      })
    }
  }
}
