/**
 * Runs from this session, rehydrated through `GET /v1/traces/{id}`.
 *
 * The backend has no "list traces" route, so the ids come from what this client
 * has seen. That is honest — it does not imply a server-side history that does
 * not exist — and each entry is still the real, persisted audit trace.
 */
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { SatQueryError, trace as fetchTrace } from '@/api/client'
import { decimal, duration, timestamp } from '@/format'
import { useUiStore } from '@/state/ui'

function TraceDetail({ traceId }: { traceId: string }) {
  const { data, isPending, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['trace', traceId],
    queryFn: ({ signal }) => fetchTrace(traceId, signal),
    retry: false,
  })

  if (isPending) return <p className="text-text-lo">Loading trace…</p>
  if (isError || !data) {
    // "Trace unavailable" named neither the problem nor the recovery. A 404 is
    // a trace the server has pruned; a transport failure is a server that is
    // no longer there, and only one of those is worth a retry button.
    const failure = error instanceof SatQueryError ? error : null
    return (
      <div role="alert" className="space-y-2">
        <p className="text-[13px] [overflow-wrap:anywhere]">
          {failure?.status === 404
            ? 'The server no longer holds this trace. Session history keeps the id, not the trace itself.'
            : (failure?.display ?? 'The trace could not be loaded.')}
        </p>
        {(failure?.retryable ?? true) && (
          <button
            type="button"
            onClick={() => void refetch()}
            disabled={isFetching}
            className="text-[12px] font-medium underline disabled:opacity-50"
          >
            {isFetching ? 'Retrying…' : 'Try again'}
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="tabular space-y-1.5 text-sm">
      <p>
        <span className="text-text-lo">Task </span>
        <span className="font-mono">{data.resolved_task.primary}</span>
      </p>
      <p>
        <span className="text-text-lo">Policy key </span>
        <span className="font-mono">{data.plan.policy_key}</span>
      </p>
      <p>
        <span className="text-text-lo">Steps </span>
        {data.executions.length} · <span className="text-text-lo">Duration </span>
        {duration(data.duration_ms)} · <span className="text-text-lo">Confidence </span>
        {decimal(data.confidence.overall, 2)}
      </p>
      <p className="pt-1 text-text-lo [overflow-wrap:anywhere]">{data.query.raw}</p>
    </div>
  )
}

export function HistoryPanel() {
  const recentRuns = useUiStore((state) => state.recentRuns)
  const selectedTraceId = useUiStore((state) => state.selectedTraceId)
  const [selected, setSelected] = useState<string | null>(selectedTraceId)

  // Arriving from a sidebar entry opens that run rather than a bare list.
  // Adjusted during render rather than in an effect, which is React's own
  // pattern for "state that follows a prop": no extra commit, no flash of the
  // previous selection.
  const [seenTraceId, setSeenTraceId] = useState(selectedTraceId)
  if (selectedTraceId !== seenTraceId) {
    setSeenTraceId(selectedTraceId)
    if (selectedTraceId) setSelected(selectedTraceId)
  }

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="t-page">History</h1>
      <p className="mt-1.5 text-[13px] text-text-lo">
        Runs from this session. Each one is fetched back from the server's stored audit trace.
      </p>

      {recentRuns.length === 0 ? (
        <p className="mt-8 rounded-lg border border-dashed border-line p-8 text-center text-text-lo">
          No runs yet. Start a new query and ask something of your imagery.
        </p>
      ) : (
        <ul className="mt-6 space-y-2">
          {recentRuns.map((run) => (
            <li key={run.traceId} className="rounded-xl border border-line bg-surface-card p-3.5">
              <button
                type="button"
                onClick={() => setSelected(selected === run.traceId ? null : run.traceId)}
                aria-expanded={selected === run.traceId}
                className="w-full text-left"
              >
                <p className="text-[13px] font-medium [overflow-wrap:anywhere]">{run.query}</p>
                <p className="tabular mt-1 font-mono text-[11px] text-text-lo [overflow-wrap:anywhere]">
                  {run.traceId} · {timestamp(run.at)}
                  {run.outcome !== 'succeeded' && (
                    <span className={run.outcome === 'failed' ? 'text-fail' : 'text-accent-warm'}>
                      {' '}
                      · {run.outcome}
                    </span>
                  )}
                </p>
              </button>
              {selected === run.traceId && (
                <div className="mt-3 border-t border-line pt-3">
                  {run.outcome === 'succeeded' ? (
                    <TraceDetail traceId={run.traceId} />
                  ) : (
                    <p className="text-[13px] text-text-lo">
                      {run.outcome === 'running'
                        ? 'This run has not finished; its trace is written when it does.'
                        : 'This run did not produce an answer, so there is no trace to show.'}
                    </p>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
