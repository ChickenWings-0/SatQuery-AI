/**
 * The session's questions, newest first — the History tab.
 *
 * Moved here from the navigation rail. It is a *shortcut* to the full History
 * page, not a replacement for it: each row opens that run expanded, and
 * "View all" goes to the page itself. The list is session-only and says so —
 * it does not imply a server-side history that does not exist.
 *
 * The dot on each row is what this session knows about that run: the one
 * currently streaming, the one that failed, or done. Older runs are "done"
 * because a run that is remembered is a run that was accepted; the trace
 * itself, with its real status, is one click away.
 */
import { DotIcon } from '@/components/ui/icons'
import { relativeTime } from '@/format'
import { isLive, useJobStore } from '@/state/job'
import { useUiStore } from '@/state/ui'

export function PreviousQueries() {
  const recentRuns = useUiStore((state) => state.recentRuns)
  const openRun = useUiStore((state) => state.openRun)
  const setSection = useUiStore((state) => state.setSection)
  const jobId = useJobStore((state) => state.jobId)
  const phase = useJobStore((state) => state.phase)

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="t-eyebrow">Previous queries</h2>
        {recentRuns.length > 0 && (
          <button
            type="button"
            onClick={() => setSection('history')}
            className="text-[12px] font-medium text-accent-warm-text hover:underline"
          >
            View all →
          </button>
        )}
      </div>

      {recentRuns.length === 0 ? (
        <p className="mt-3 text-[13px] leading-relaxed text-text-lo">
          Questions you ask this session will collect here.
        </p>
      ) : (
        <ul className="mt-3 divide-y divide-line-soft rounded-xl border border-line bg-bg-main/40">
          {recentRuns.map((run) => {
            // The live job's phase is the freshest word on its own entry; every
            // other entry shows the outcome recorded when it settled.
            const state =
              run.traceId === jobId && isLive(phase)
                ? 'running'
                : run.traceId === jobId && phase === 'failed'
                  ? 'failed'
                  : run.outcome === 'succeeded'
                    ? 'done'
                    : run.outcome
            const tone =
              state === 'failed' ? 'text-fail' : state === 'running' ? 'text-accent-warm' : 'text-ok'
            return (
              <li key={run.traceId}>
                <button
                  type="button"
                  // Navigating to History without saying *which* run made the
                  // user re-find by hand the entry they had just clicked.
                  onClick={() => openRun(run.traceId)}
                  title={`${run.query}\n${run.traceId}`}
                  className="flex min-h-11 w-full items-start gap-2.5 px-3 py-2.5 text-left transition-colors hover:bg-accent-glow"
                >
                  <DotIcon
                    className={`mt-1.5 ${tone}`}
                    aria-hidden={false}
                    role="img"
                    aria-label={`Run ${state}`}
                  />
                  <span className="min-w-0 flex-1">
                    {/* Two lines, clamped: the question is what you scan for.
                        `overflow-wrap` because a pasted URL or an unbroken
                        CJK question has nowhere to break. */}
                    <span className="line-clamp-2 text-[12.5px] leading-snug [overflow-wrap:anywhere]">
                      {run.query}
                    </span>
                    <span className="tabular mt-0.5 block font-mono text-[10px] text-text-lo">
                      {relativeTime(run.at)} · {run.traceId.slice(0, 8)}
                    </span>
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
