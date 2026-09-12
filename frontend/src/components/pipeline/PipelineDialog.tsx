/**
 * The progressive-disclosure boundary.
 *
 * `DagCanvas` — and with it `@xyflow/react` and `dagre` — sits behind
 * `React.lazy()`, so the chunk is fetched the first time someone asks for the
 * pipeline and never before. Nothing renders here at all until `open` is true,
 * which is what guarantees no `.react-flow` element exists on the main screen.
 *
 * The modal is also where the rest of "show me the machine" lives — the
 * FactSheet and the raw AuditTrace — so the product has exactly one such
 * surface rather than several competing ones.
 */
import { Suspense, lazy, useMemo, useState } from 'react'

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog'
import { StatusDot } from '@/components/ui/StatusDot'
import { bytes, decimal, duration, scalar } from '@/format'
import type { AnalyzeResponse } from '@/api/types'
import type { StepNode } from '@/state/job'

/**
 * Past this, the raw trace is offered as a download rather than painted.
 *
 * A bi-temporal run with rendered views serialises to a few hundred kilobytes,
 * and `<pre>{JSON.stringify(trace, null, 2)}</pre>` puts every byte of that in
 * one text node with `white-space: pre-wrap`. Layout on a megabyte of it locks
 * the tab for seconds on the machine this runs on. 256 kB is roughly where a
 * human stops scrolling and starts downloading anyway.
 */
const INLINE_TRACE_LIMIT = 256 * 1024

const DagCanvas = lazy(() => import('@/components/pipeline/DagCanvas'))

type Tab = 'graph' | 'facts' | 'trace'

/**
 * Save the raw trace to a file.
 *
 * The object URL is revoked on the next macrotask rather than on the line after
 * `click()`. Chrome copies the blob synchronously and forgives the immediate
 * revoke; Firefox does not, and cancels the download. Nothing in the UI reports
 * that — the user simply gets no file, which is a bad way to fail on the
 * "download the evidence" button of an audit tool.
 */
function downloadTrace(serialised: string, traceId: string | undefined) {
  if (!serialised) return
  const url = URL.createObjectURL(new Blob([serialised], { type: 'application/json' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `trace-${traceId ?? 'unknown'}.json`
  link.rel = 'noopener'
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

function FactSheetTable({ sheet }: { sheet: Record<string, unknown> }) {
  const byTool = new Map<string, Array<[string, unknown]>>()
  for (const [key, value] of Object.entries(sheet)) {
    // Keys are namespaced `<tool>.<scalar>` (§9 row 4) precisely so two tools
    // can both emit `mean` without silently overwriting each other.
    const dot = key.indexOf('.')
    const tool = dot === -1 ? 'other' : key.slice(0, dot)
    const scalar = dot === -1 ? key : key.slice(dot + 1)
    byTool.set(tool, [...(byTool.get(tool) ?? []), [scalar, value]])
  }

  return (
    <div className="space-y-5">
      {[...byTool.entries()].map(([tool, rows]) => (
        <section key={tool}>
          <h4 className="font-mono text-[12px] font-semibold">{tool}</h4>
          <table className="mt-1.5 w-full text-left text-[13px]">
            <tbody>
              {rows.map(([name, value]) => (
                <tr key={name} className="border-t border-line-soft">
                  <td className="w-1/2 py-1 font-mono text-text-lo [overflow-wrap:anywhere]">
                    {name}
                  </td>
                  {/* `String(value)` here rendered "[object Object]" for any
                      tool that emitted a list or a nested map — in the one tab
                      built to prove the run was real. */}
                  <td className="tabular py-1 font-mono [overflow-wrap:anywhere]">
                    {scalar(value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}
    </div>
  )
}

export function PipelineDialog({
  open,
  onOpenChange,
  nodes,
  result,
  focusedStep,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  nodes: StepNode[]
  result: AnalyzeResponse | null
  focusedStep: number | null
}) {
  const [tab, setTab] = useState<Tab>('graph')
  const [selected, setSelected] = useState<number | null>(focusedStep)

  // The dialog never unmounts — only its content is gated on `open` — so this
  // `useState` seeded once, on first render, and every later citation opened
  // the inspector on whichever node had been clicked first. Clearing on each
  // open hands the node choice back to `focusedStep`.
  const [wasOpen, setWasOpen] = useState(open)
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) setSelected(null)
  }

  const trace = result?.trace ?? null
  const executions = trace?.executions ?? []
  const step = selected ?? focusedStep
  const selectedExecution = executions.find((execution) => execution.step === step)
  // DEGRADED and SKIPPED are different outcomes and the header should not
  // blur them: one ran with a fallback, the other never ran at all.
  const counts = {
    degraded: nodes.filter((node) => node.state === 'DEGRADED').length,
    failed: nodes.filter((node) => node.state === 'FAILED').length,
    skipped: nodes.filter((node) => node.state === 'SKIPPED').length,
  }
  const serialised = useMemo(
    () => (trace ? JSON.stringify(trace, null, 2) : ''),
    [trace],
  )

  const outcome = [
    counts.degraded && `${counts.degraded} degraded`,
    counts.failed && `${counts.failed} failed`,
    counts.skipped && `${counts.skipped} skipped`,
  ]
    .filter(Boolean)
    .join(', ')

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {open && (
        <DialogContent aria-describedby="pipeline-description">
          {/* `gap-x-3 gap-y-2` rather than `gap-3`: at 390px this header wraps
              to three rows, and a uniform gap made the wrapped rows as far
              apart as the columns, which read as three separate bars. */}
          <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line px-4 py-3 md:px-5">
            <DialogTitle className="text-base font-semibold">Processing pipeline</DialogTitle>
            <DialogDescription id="pipeline-description" className="text-sm text-text-lo">
              {trace ? (
                <>
                  <span className="font-mono">{trace.plan.policy_key}</span> · planner{' '}
                  <span className="font-mono">{trace.plan.planner}</span> · {nodes.length} steps
                  {outcome && ` · ${outcome}`}
                </>
              ) : (
                'Running…'
              )}
            </DialogDescription>

            <nav className="order-last ml-auto flex gap-1 md:order-none">
              {(['graph', 'facts', 'trace'] as const).map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => setTab(name)}
                  className={`rounded px-2.5 py-1 text-sm capitalize transition-colors ${
                    tab === name ? 'bg-accent-cool text-on-accent-cool' : 'hover:bg-bg-main'
                  }`}
                >
                  {name === 'facts' ? 'FactSheet' : name}
                </button>
              ))}
            </nav>
            <DialogClose className="rounded px-2 py-1 text-sm text-text-lo hover:bg-bg-main">
              Close ✕
            </DialogClose>
          </header>

          {/* Column on a phone, row on a laptop: the inspector was a fixed
              `w-80` beside the graph at every width, which left about 35px of
              canvas at 375px. */}
          <div className="flex min-h-0 flex-1 flex-col md:flex-row">
            <div className="min-h-0 min-w-0 flex-1">
              {tab === 'graph' && (
                <Suspense
                  fallback={<p className="p-8 text-sm text-text-lo">Loading the graph…</p>}
                >
                  <DagCanvas
                    nodes={nodes}
                    executions={executions}
                    focusedStep={step}
                    onSelectStep={setSelected}
                  />
                </Suspense>
              )}
              {tab === 'facts' && (
                <div className="h-full overflow-y-auto p-5">
                  {trace ? (
                    <FactSheetTable sheet={trace.fact_sheet ?? {}} />
                  ) : (
                    <p className="text-text-lo">No FactSheet yet.</p>
                  )}
                </div>
              )}
              {tab === 'trace' && (
                <div className="h-full overflow-auto p-5">
                  <div className="mb-3 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      disabled={!trace}
                      onClick={() => downloadTrace(serialised, trace?.trace_id)}
                      className="rounded border border-accent-warm-text px-3 py-1.5 text-sm text-accent-warm-text transition-colors hover:bg-accent-warm-strong hover:text-white disabled:opacity-40"
                    >
                      Download AuditTrace
                    </button>
                    {trace && <span className="t-meta">{bytes(serialised.length)} of JSON</span>}
                  </div>

                  {!trace ? (
                    <p className="text-sm text-text-lo">
                      The trace is written when the run finishes. Nothing to show yet.
                    </p>
                  ) : serialised.length > INLINE_TRACE_LIMIT ? (
                    <>
                      <p className="mb-3 rounded-lg border border-line bg-surface-card px-4 py-3 text-sm text-text-lo">
                        This trace is {bytes(serialised.length)} — too large to render without
                        stalling the tab. The first {bytes(INLINE_TRACE_LIMIT)} are shown; download
                        it for the whole thing.
                      </p>
                      <pre className="rounded-lg border border-line bg-surface-card p-4 font-mono text-[11px] whitespace-pre-wrap [overflow-wrap:anywhere]">
                        {serialised.slice(0, INLINE_TRACE_LIMIT)}
                        {'\n…'}
                      </pre>
                    </>
                  ) : (
                    <pre className="rounded-lg border border-line bg-surface-card p-4 font-mono text-[11px] whitespace-pre-wrap [overflow-wrap:anywhere]">
                      {serialised}
                    </pre>
                  )}
                </div>
              )}
            </div>

            {tab === 'graph' && (
              <aside className="max-h-[45%] min-h-0 shrink-0 overflow-y-auto border-t border-line bg-surface-card p-4 md:max-h-none md:w-80 md:border-t-0 md:border-l">
                {selectedExecution ? (
                  <>
                    <p className="flex items-center gap-2 font-mono text-sm font-medium">
                      <StatusDot status={selectedExecution.status} />
                      <span className="min-w-0 [overflow-wrap:anywhere]">
                        {selectedExecution.tool}
                      </span>
                    </p>
                    <p className="tabular mt-1 font-mono text-[11px] text-text-lo [overflow-wrap:anywhere]">
                      v{selectedExecution.version} · {selectedExecution.device_used} ·{' '}
                      {duration(selectedExecution.duration_ms)} · conf{' '}
                      {decimal(selectedExecution.confidence, 2)}
                    </p>
                    {selectedExecution.error && (
                      <p className="mt-2 rounded border border-fail/40 bg-fail/8 p-2 text-[12px] [overflow-wrap:anywhere]">
                        {selectedExecution.error}
                      </p>
                    )}

                    <h5 className="mt-4 text-[10px] font-semibold tracking-[0.1em] text-text-lo uppercase">
                      Effective parameters
                    </h5>
                    <pre className="mt-1 overflow-x-auto rounded border border-line-soft p-2 font-mono text-[11px]">
                      {JSON.stringify(selectedExecution.params ?? {}, null, 2)}
                    </pre>

                    <h5 className="mt-4 text-[10px] font-semibold tracking-[0.1em] text-text-lo uppercase">
                      Scalars
                    </h5>
                    <pre className="mt-1 overflow-x-auto rounded border border-line-soft p-2 font-mono text-[11px]">
                      {JSON.stringify(selectedExecution.scalars ?? {}, null, 2)}
                    </pre>

                    <p className="tabular mt-4 font-mono text-[11px] text-text-lo [overflow-wrap:anywhere]">
                      in: {selectedExecution.input_refs?.join(', ') || '—'}
                      <br />
                      out: {selectedExecution.output_refs?.join(', ') || '—'}
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-text-lo">
                    Select a node to see the exact parameters it ran with.
                  </p>
                )}
              </aside>
            )}
          </div>
        </DialogContent>
      )}
    </Dialog>
  )
}
