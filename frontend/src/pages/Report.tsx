/**
 * The printable record: `/report/<trace>` for one run, `/report/project/<id>`
 * for every run in a project. Re-fetches each trace from the API (the saved
 * summary is the fallback when the server no longer has it), renders the
 * answer with its citations, the measurements, the plan, and the executions,
 * and hands the browser's own "Save as PDF" the job — no PDF library.
 */
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo } from 'react'
import { useShallow } from 'zustand/shallow'

import { trace as fetchTrace } from '@/api/client'
import type { AuditTrace } from '@/api/types'
import { GroundedAnswer } from '@/components/thread/GroundedAnswer'
import { KpiCards } from '@/components/stage/KpiCards'
import { StatusDot } from '@/components/ui/StatusDot'
import { duration, timestamp } from '@/format'
import { confidenceCard, selectKpis } from '@/kpi/registry'
import { useLibrary } from '@/pages/saved/useLibrary'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { runsInProject, useLibraryStore, type SavedRun } from '@/state/library'

function ids(): { traceIds: string[]; projectId: string | null } {
  const path = window.location.pathname
  const project = /^\/report\/project\/([^/]+)/.exec(path)
  if (project?.[1]) return { traceIds: [], projectId: decodeURIComponent(project[1]) }
  const single = /^\/report\/([^/]+)/.exec(path)
  return { traceIds: single?.[1] ? [decodeURIComponent(single[1])] : [], projectId: null }
}

function TraceSection({ traceId, saved }: { traceId: string; saved: SavedRun | null }) {
  const { data, isPending, isError } = useQuery({
    queryKey: ['trace', traceId],
    queryFn: ({ signal }) => fetchTrace(traceId, signal),
    retry: false,
  })
  const cards = useMemo(() => (data ? selectKpis(data.fact_sheet) : []), [data])

  return (
    <section className="break-after-page border-t border-line pt-8 first:border-t-0 first:pt-0" aria-labelledby={`r-${traceId}`}>
      <h2 id={`r-${traceId}`} className="t-panel text-[18px]">
        “{data?.query.raw ?? saved?.query ?? traceId}”
      </h2>
      <p className="t-coord mt-2 text-text-lo">
        trace {traceId}
        {data ? ` · ${timestamp(Date.parse(data.created_at))} · ${duration(data.duration_ms)} · schema ${data.schema_version}` : ''}
      </p>

      {isPending ? (
        <p className="t-meta mt-4">Fetching the trace…</p>
      ) : isError || !data ? (
        <div className="mt-4 rounded-lg bg-warn/8 p-3 text-[13px]">
          <p className="font-medium text-warn-text">Trace no longer on the server.</p>
          {saved ? (
            <p className="t-meta mt-1">
              Showing the summary saved on this device: {saved.taskType}, {saved.headline?.value}{' '}
              {saved.headline?.label}, confidence {saved.confidence?.toFixed(2) ?? '—'}.
            </p>
          ) : null}
        </div>
      ) : (
        <Body trace={data} cards={cards} />
      )}
    </section>
  )
}

function Body({ trace, cards }: { trace: AuditTrace; cards: ReturnType<typeof selectKpis> }) {
  const answer = trace.answer
  const all = [...cards, confidenceCard({ ...trace, artifacts: trace.artifacts } as never)]
  return (
    <>
      <div className="mt-5">
        <h3 className="t-eyebrow">Answer</h3>
        <div className="mt-2 text-[14px] leading-relaxed">
          <GroundedAnswer answer={answer} onCitation={() => undefined} />
        </div>
      </div>
      <div className="mt-6">
        <KpiCards cards={all} activeId={null} degradedIds={new Set()} />
      </div>
      <div className="mt-6">
        <h3 className="t-eyebrow">Plan · {trace.plan.planner}</h3>
        <ol className="mt-2 space-y-1 text-[13px]">
          {(trace.plan.steps ?? []).map((step) => (
            <li key={step.step} className="flex gap-3">
              <span className="t-mono w-6 text-text-lo">{step.step}</span>
              <span className="t-mono">{step.tool}</span>
              <span className="text-text-lo">{step.reason}</span>
            </li>
          ))}
        </ol>
      </div>
      <div className="mt-6">
        <h3 className="t-eyebrow">Executions</h3>
        <table className="mt-2 w-full text-[12.5px]">
          <thead>
            <tr className="t-eyebrow border-b border-line text-left">
              <th className="py-1">Step</th>
              <th>Tool</th>
              <th>Version</th>
              <th>Status</th>
              <th>Duration</th>
              <th>Conf.</th>
            </tr>
          </thead>
          <tbody>
            {trace.executions.map((ex) => (
              <tr key={ex.step} className="border-b border-line-soft">
                <td className="t-mono py-1">{ex.step}</td>
                <td className="t-mono">{ex.tool}</td>
                <td className="t-mono text-text-lo">{ex.version}</td>
                <td>
                  <span className="flex items-center gap-1.5">
                    <StatusDot status={ex.status} />
                    {ex.status}
                  </span>
                </td>
                <td className="t-mono">{duration(ex.duration_ms)}</td>
                <td className="t-mono">{ex.confidence.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {trace.warnings && trace.warnings.length > 0 ? (
        <div className="mt-6">
          <h3 className="t-eyebrow">Warnings</h3>
          <ul className="mt-2 space-y-1 text-[12.5px] text-warn-text">
            {trace.warnings.map((w, i) => (
              <li key={i}>
                <span className="t-mono">{w.code}</span> — {w.message}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </>
  )
}

export default function Report() {
  const hydrated = useLibrary()
  const { traceIds, projectId } = useMemo(() => ids(), [])
  const project = useLibraryStore((state) => (projectId ? state.projects[projectId] : null))
  const members = useLibraryStore(useShallow((state) => (projectId ? runsInProject(state, projectId) : [])))
  const runs = useLibraryStore((state) => state.runs)
  const list = projectId ? members.map((r) => r.traceId) : traceIds

  useEffect(() => {
    document.documentElement.classList.add('print-report')
    return () => document.documentElement.classList.remove('print-report')
  }, [])

  return (
    <div className="mx-auto max-w-3xl px-6 py-10 print:max-w-none print:px-0">
      <DocumentMeta page="report" subject={project?.name ?? traceIds[0]?.slice(0, 8)} />
      <header className="mb-8 flex flex-wrap items-end justify-between gap-3 border-b border-line pb-5">
        <div>
          <p className="t-eyebrow">SatQuery AI · report</p>
          <h1 className="t-page mt-1">{project ? project.name : 'Run report'}</h1>
          {project?.description ? <p className="t-lede mt-1">{project.description}</p> : null}
        </div>
        <button type="button" onClick={() => window.print()} className="btn-primary print:hidden">
          Save as PDF
        </button>
      </header>
      {!hydrated ? (
        <p className="t-meta">Loading…</p>
      ) : list.length === 0 ? (
        <p className="t-meta">Nothing to report: the address names no run on this device.</p>
      ) : (
        <div className="space-y-10">
          {list.map((id) => (
            <TraceSection key={id} traceId={id} saved={runs[id] ?? null} />
          ))}
        </div>
      )}
    </div>
  )
}
