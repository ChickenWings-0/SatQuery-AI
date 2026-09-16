/**
 * F5 — the right column, in four tabs.
 *
 *   Chat       the composer, the quick chips, the audited response card, the
 *              pipeline pulse and the confidence — the whole run, in order
 *   Results    the confidence breakdown and the headline numbers, without the
 *              prose around them
 *   Citations  every claim the answer made and the measurement it points at
 *   History    the questions asked this session
 *
 * The tab is local state. Nothing outside this panel needs it, and a store
 * field would only invite something to depend on it. Chat is the default and
 * the tab a new run returns to: the run *is* the chat.
 *
 * The DAG modal is rendered here but mounts nothing until it is opened.
 *
 * Sections within a tab are separated by full-bleed rules rather than by gaps,
 * so the column reads as one panel with regions instead of a stack of floating
 * cards — which is what made the empty space below the fold feel like a hole.
 */
import { useEffect, useMemo, useState } from 'react'

import { GroundedAnswer, type CitationTarget } from '@/components/thread/GroundedAnswer'
import { PipelinePulse } from '@/components/thread/PipelinePulse'
import { PreviousQueries } from '@/components/thread/PreviousQueries'
import { QueryComposer } from '@/components/thread/QueryComposer'
import { SidebarTabs } from '@/components/thread/SidebarTabs'
import { panelId, tabId, type Tab } from '@/components/thread/tabs'
import { SuggestionChips } from '@/components/thread/SuggestionChips'
import { PipelineDialog } from '@/components/pipeline/PipelineDialog'
import { SitrepAction } from '@/components/export/SitrepAction'
import { BookmarkIcon, CopyIcon, SatelliteIcon, ShareIcon } from '@/components/ui/icons'
import { countOf } from '@/format'
import { groupForScalar, groupViews } from '@/evidence/views'
import { cardMatchesScalar, confidenceCard, formatKpi, selectKpis } from '@/kpi/registry'
import { completedCount, degradedCount, isLive, useJobStore } from '@/state/job'
import { useFocusStore } from '@/state/focus'
import { toast } from '@/state/notifications'
import { useUiStore } from '@/state/ui'
import { parseSource } from '@/thread/annotate'
import { stripBboxTokens } from '@/thread/bbox'
import { useRun } from '@/thread/useRun'

function Section({
  title,
  children,
  right,
  className = '',
}: {
  title: string
  children: React.ReactNode
  right?: React.ReactNode
  className?: string
}) {
  return (
    <section className={`border-t border-line px-4 py-5 md:px-5 ${className}`}>
      <div className="flex items-baseline gap-2">
        <h2 className="t-eyebrow">{title}</h2>
        {right && <span className="ml-auto">{right}</span>}
      </div>
      <div className="mt-3 max-w-[720px]">{children}</div>
    </section>
  )
}

function ConfidenceBlock() {
  const result = useJobStore((state) => state.result)
  if (!result) return null
  const { overall, components, caps_applied: caps } = result.confidence

  return (
    <>
      <div className="flex items-baseline gap-2">
        <span className="tabular text-[28px] leading-none font-semibold tracking-[-0.02em]">
          {overall.toFixed(2)}
        </span>
        <span className="t-meta">overall</span>
      </div>

      <ul className="mt-3.5 space-y-1.5">
        {Object.entries(components ?? {}).map(([name, value]) => (
          <li key={name} className="flex items-center gap-2.5 text-[11px]">
            <span className="w-32 shrink-0 text-text-lo">{name.replace(/_/g, ' ')}</span>
            <span className="h-1 flex-1 overflow-hidden rounded-full bg-line">
              <span
                // The filled part of a meter is non-text content carrying a
                // value, so it has to clear 3:1 against its own track.
                className="block h-full rounded-full bg-accent-warm-strong"
                style={{ width: `${Math.round(Number(value) * 100)}%` }}
              />
            </span>
            <span className="tabular w-7 shrink-0 text-right font-mono">
              {Number(value).toFixed(2)}
            </span>
          </li>
        ))}
      </ul>

      {(caps?.length ?? 0) > 0 && (
        // A clamped confidence should say which rule clamped it, or the number
        // looks arbitrary.
        <p className="mt-3.5 flex flex-wrap items-center gap-1.5 text-[11px] text-text-lo">
          <span>capped by</span>
          {caps?.map((cap) => (
            <span key={cap} className="chip border border-line">
              {cap}
            </span>
          ))}
        </p>
      )}
    </>
  )
}

/** A small icon control in the response card's action row. */
function CardAction({
  label,
  onClick,
  children,
  done,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
  done?: string | undefined
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={done ?? label}
      onClick={onClick}
      className="grid size-7 place-items-center rounded-md text-text-lo transition-colors hover:bg-accent-glow hover:text-text-hi"
    >
      {children}
      {done && (
        <span role="status" className="sr-only">
          {done}
        </span>
      )}
    </button>
  )
}

export function ThreadPanel() {
  const { submit, retry, cancel, dismiss, failure, reattach } = useRun()
  const phase = useJobStore((state) => state.phase)
  const nodes = useJobStore((state) => state.nodes)
  const artifacts = useJobStore((state) => state.artifacts)
  const result = useJobStore((state) => state.result)
  const jobError = useJobStore((state) => state.error)
  const validation = useUiStore((state) => state.validation)
  const recentRuns = useUiStore((state) => state.recentRuns)

  // Per-field selectors, not `useFocusStore()`. Subscribing to the whole store
  // re-rendered this entire column — answer, pulse and confidence bars — on
  // every `setSwipe` during an A/B drag, i.e. once per pointermove frame.
  const pipelineOpen = useFocusStore((state) => state.pipelineOpen)
  const focusedStep = useFocusStore((state) => state.focusedStep)
  const openPipeline = useFocusStore((state) => state.openPipeline)
  const closePipeline = useFocusStore((state) => state.closePipeline)
  const focusCitation = useFocusStore((state) => state.focusCitation)

  const [tab, setTab] = useState<Tab>('chat')
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1800)
    return () => clearTimeout(timer)
  }, [copied])

  // A new run is a new conversation turn: bring the reader back to it.
  // Adjusted during render (React's "state that follows a prop" pattern), so
  // the tab switches in the same commit the phase changes in.
  const [seenPhase, setSeenPhase] = useState(phase)
  if (phase !== seenPhase) {
    setSeenPhase(phase)
    if (phase === 'streaming') setTab('chat')
  }

  const groups = useMemo(() => groupViews(artifacts), [artifacts])
  const cards = useMemo(() => selectKpis(result?.trace?.fact_sheet), [result])
  const allCards = useMemo(
    () => (result ? [...cards, confidenceCard(result)] : cards),
    [cards, result],
  )
  const degraded = degradedCount({ nodes })
  const citations = result?.answer.citations ?? []

  function handleCitation(target: CitationTarget) {
    const execution = result?.trace?.executions.find((candidate) => candidate.step === target.step)
    const outputs = execution?.output_refs ?? []
    const tool = execution?.tool ?? ''

    // Three things at once: reveal the evidence, light the matching KPI, and
    // remember the step so the pipeline modal opens on it.
    const group = groupForScalar(groups, outputs, target.paths.join(' '))
    const card = allCards.find((candidate) => cardMatchesScalar(candidate, tool, target.paths))

    focusCitation({ viewKey: group?.key, kpiId: card?.id, step: target.step })
  }

  // Remembered per trace, so a new run's card is unsaved without an effect.
  const [savedTrace, setSavedTrace] = useState<string | null>(null)
  const saved = savedTrace !== null && savedTrace === result?.trace_id
  async function saveRun() {
    if (!result) return
    // The library and its IndexedDB adapter load on first save, not at boot.
    const [{ captureCurrentRun }, { useLibraryStore }] = await Promise.all([
      import('@/pages/saved/capture'),
      import('@/state/library'),
    ])
    const asked = useUiStore.getState().recentRuns.find((r) => r.traceId === result.trace_id)
    const run = await captureCurrentRun(asked?.query ?? '')
    if (!run) return
    await useLibraryStore.getState().save(run)
    setSavedTrace(run.traceId)
    toast('Run saved on this device.', 'ok', {
      label: 'Open Saved',
      run: () => useUiStore.getState().setSection('saved'),
    })
  }

  async function copyAnswer() {
    if (!result) return
    try {
      await navigator.clipboard.writeText(stripBboxTokens(result.answer.text))
      setCopied(true)
    } catch {
      // No clipboard permission: the text is on screen to select by hand.
    }
  }

  const composer = (
    <div className="px-4 pt-4 pb-4 md:px-5">
      <QueryComposer
        onSubmit={(query) => void submit(query)}
        onCancel={cancel}
        busy={isLive(phase)}
      />
      <div className="mt-3">
        <SuggestionChips />
      </div>

      {/* The stream dropped and the run is being reattached to. Distinct from a
          failure: nothing has gone wrong with the run, only with our socket,
          and the DAG on screen is still live. */}
      {phase === 'reconnecting' && (
        <div
          role="status"
          className="mt-3 rounded-lg border border-warn/40 bg-warn/8 px-3 py-2 text-[13px]"
        >
          <p>
            Connection lost — reattaching to the run
            {reattach ? ` (attempt ${reattach.attempt} of ${reattach.max})` : ''}…
          </p>
          <div className="mt-2 flex gap-3 text-[12px]">
            <button type="button" onClick={cancel} className="text-text-lo underline">
              Stop
            </button>
          </div>
        </div>
      )}

      {/* Client-side failures: the request never landed, or the stream died.
          `role="alert"` because a screen-reader user gets no other notice
          that the run they started has stopped. */}
      {failure && (
        <div
          role="alert"
          className="mt-3 rounded-lg border border-fail/40 bg-fail/8 px-3 py-2 text-[13px]"
        >
          <p className="[overflow-wrap:anywhere]">{failure.message}</p>
          <div className="mt-2 flex gap-3 text-[12px]">
            {failure.retryable && (
              <button type="button" onClick={retry} className="font-medium underline">
                Try again
              </button>
            )}
            <button type="button" onClick={dismiss} className="text-text-lo underline">
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* The server's own §6 envelope, shown verbatim — it is the one place
          the reason for a failed run is authoritative. */}
      {jobError && (
        <p
          role="alert"
          className="mt-3 rounded-lg border border-fail/40 bg-fail/8 px-3 py-2 text-[13px] [overflow-wrap:anywhere]"
        >
          <span className="font-mono text-[11px]">{jobError.code}</span> {jobError.message}
          {jobError.hint && <span className="mt-1 block text-text-lo">{jobError.hint}</span>}
        </p>
      )}
    </div>
  )

  return (
    <div className="flex min-h-full flex-col">
      <div className="flex items-center gap-3 border-b border-line px-4 md:px-5">
        <SidebarTabs
          value={tab}
          onChange={setTab}
          badges={{ citations: citations.length, history: recentRuns.length }}
        />
      </div>

      {/* One panel mounted at a time. The others hold nothing that needs to
          stay warm — the draft lives in the store, the run in its own — and
          a hidden panel's text would still be in the accessibility tree. */}

      {/* ── Chat ─────────────────────────────────────────────────────────── */}
      {tab === 'chat' && (
        <div
          role="tabpanel"
          id={panelId('chat')}
          aria-labelledby={tabId('chat')}
          className="flex min-h-0 flex-1 flex-col"
        >
          {composer}

          {/*
           * The column would otherwise be a tall blank until the first run,
           * which reads as a broken panel rather than an empty one.
           *
           * Hidden below `md`, where the problem it solves does not exist: a
           * phone has no tall empty column.
           */}
          {!result && !isLive(phase) && (
            <Section title="What happens next" className="hidden md:block">
              <ol className="space-y-2.5 text-[13px] text-text-lo">
                {[
                  ['Pre-flight', 'Compatibility is checked before you type — no GPU touched.'],
                  ['Evidence', 'Named views are rendered and shown to the model.'],
                  ['Grounded answer', 'Every number is traced back to the tool that measured it.'],
                ].map(([title, body], index) => (
                  <li key={title} className="flex gap-2.5">
                    <span className="chip mt-px h-5 shrink-0 border border-line">{index + 1}</span>
                    <span>
                      <span className="font-medium text-text-hi">{title}. </span>
                      {body}
                    </span>
                  </li>
                ))}
              </ol>
              {validation && (
                <p className="t-meta mt-4">
                  {countOf(validation.supported_tasks.length, {
                    one: 'question type',
                    other: 'question types',
                  })}{' '}
                  available for these images.
                </p>
              )}
            </Section>
          )}

          {isLive(phase) && !result && (
            <Section title="Running">
              <p role="status" className="text-[13px] text-text-lo">
                {nodes.length > 0
                  ? `${completedCount({ nodes })} of ${countOf(nodes.length, { one: 'step', other: 'steps' })} finished. Evidence appears in the Data Stage as each tool completes.`
                  : 'Planning the run. Evidence appears in the Data Stage as each tool finishes.'}
              </p>
            </Section>
          )}

          {result && (
            <Section title="Answer">
              {/* The audited response card: who said it, when, what you can do
                with it, and — inside — the answer with its grounding. */}
              <article className="card p-4">
                <header className="flex items-center gap-2.5">
                  <span
                    aria-hidden
                    className="grid size-7 shrink-0 place-items-center rounded-lg bg-accent-warm text-white"
                  >
                    <SatelliteIcon size={15} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-[12.5px] leading-tight font-medium">SatQuery AI</p>
                    <p className="text-[11px] text-text-lo">
                      {result.answer.template_fallback ? 'Templated · ' : ''}
                      Just now
                    </p>
                  </div>
                  <div className="flex items-center gap-0.5">
                    <CardAction
                      label="Copy answer"
                      onClick={() => void copyAnswer()}
                      done={copied ? 'Answer copied' : undefined}
                    >
                      <CopyIcon size={14} />
                    </CardAction>
                    <CardAction
                      label={saved ? 'Saved on this device' : 'Save this run on this device'}
                      onClick={() => void saveRun()}
                      done={saved ? 'Saved' : undefined}
                    >
                      <BookmarkIcon size={14} />
                    </CardAction>
                    <CardAction label="Share: copy link" onClick={() => void copyAnswer()}>
                      <ShareIcon size={14} />
                    </CardAction>
                    <SitrepAction />
                  </div>
                </header>

                <div className="mt-3.5">
                  {result.answer.text.trim() === '' ? (
                    // A run can succeed and still produce no prose — every tool
                    // degraded, or the VLM returned nothing. Saying so beats an
                    // empty paragraph that reads as a rendering fault.
                    <p className="text-[13px] text-text-lo">
                      The run finished without producing an answer. The measurements below and in
                      the pipeline are still real.
                    </p>
                  ) : (
                    <GroundedAnswer answer={result.answer} onCitation={handleCitation} />
                  )}
                </div>

                {citations.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2 border-t border-line-soft pt-3">
                    <button
                      type="button"
                      onClick={() => setTab('citations')}
                      className="btn-ghost !px-3 !py-1.5 !text-[12px]"
                    >
                      View Sources
                    </button>
                  </div>
                )}
              </article>
            </Section>
          )}

          {nodes.length > 0 && (
            <Section
              title="Pipeline"
              right={
                degraded > 0 ? (
                  <span className="chip bg-warn/20">{degraded} degraded</span>
                ) : (
                  <span className="t-meta">
                    {countOf(nodes.length, { one: 'step', other: 'steps' })}
                  </span>
                )
              }
            >
              <PipelinePulse nodes={nodes} onOpen={(step) => openPipeline(step)} />
              <button
                type="button"
                onClick={() => openPipeline()}
                className="btn-ghost mt-3 flex w-full items-center justify-between"
              >
                <span>View Processing Pipeline</span>
                <span aria-hidden>→</span>
              </button>
            </Section>
          )}

          {result && (
            <Section title="Confidence">
              <ConfidenceBlock />
            </Section>
          )}

          {/* Absorbs the remaining height so the last rule sits above dead
            space rather than floating mid-column. */}
          <div className="flex-1 border-t border-line" />
        </div>
      )}

      {/* ── Results ──────────────────────────────────────────────────────── */}
      {tab === 'results' && (
        <div role="tabpanel" id={panelId('results')} aria-labelledby={tabId('results')}>
          {result ? (
            <>
              <Section title="Confidence" className="border-t-0">
                <ConfidenceBlock />
              </Section>
              {allCards.length > 0 && (
                <Section title="Key insights">
                  <ul className="divide-y divide-line-soft">
                    {allCards.map((card) => (
                      <li key={card.id} className="flex items-baseline gap-3 py-2 text-[13px]">
                        <span className="min-w-0 flex-1 truncate text-text-lo" title={card.label}>
                          {card.label}
                        </span>
                        <span className="tabular shrink-0 font-mono font-medium">
                          {formatKpi(card)}
                          {card.unit && (
                            <span className="ml-1 text-[11px] text-accent-warm-text">
                              {card.unit}
                            </span>
                          )}
                        </span>
                      </li>
                    ))}
                  </ul>
                </Section>
              )}
            </>
          ) : (
            <p className="px-4 py-6 text-[13px] text-text-lo md:px-5">
              Results appear here once a run has finished.
            </p>
          )}
        </div>
      )}

      {/* ── Citations ────────────────────────────────────────────────────── */}
      {tab === 'citations' && (
        <div role="tabpanel" id={panelId('citations')} aria-labelledby={tabId('citations')}>
          {citations.length > 0 ? (
            <Section
              title="Sources"
              className="border-t-0"
              right={
                <span className="t-meta">
                  {countOf(citations.length, {
                    one: 'citation',
                    other: 'citations',
                  })}
                </span>
              }
            >
              <ul className="space-y-2">
                {citations.map((citation, index) => {
                  const parsed = parseSource(citation.source)
                  const execution = parsed
                    ? result?.trace?.executions.find((e) => e.step === parsed.step)
                    : undefined
                  return (
                    <li key={`${citation.source}-${index}`}>
                      <button
                        type="button"
                        disabled={!parsed}
                        onClick={() =>
                          parsed &&
                          handleCitation({
                            citation,
                            step: parsed.step,
                            paths: parsed.paths,
                          })
                        }
                        className="card w-full px-3.5 py-3 text-left transition-colors hover:border-accent-warm/40 disabled:cursor-default"
                      >
                        <p className="text-[13px] [overflow-wrap:anywhere]">
                          <span className="rounded bg-evidence px-1 py-0.5 font-medium text-on-evidence">
                            {citation.claim}
                          </span>
                        </p>
                        <p className="tabular mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[11px] text-text-lo">
                          {parsed && <span>step {parsed.step}</span>}
                          {execution && <span className="text-text-hi">{execution.tool}</span>}
                          <span className="[overflow-wrap:anywhere]">{citation.source}</span>
                          {citation.value !== undefined && citation.value !== null && (
                            <span className="ml-auto text-text-hi">= {String(citation.value)}</span>
                          )}
                        </p>
                      </button>
                    </li>
                  )
                })}
              </ul>
            </Section>
          ) : (
            <p className="px-4 py-6 text-[13px] text-text-lo md:px-5">
              {result
                ? 'This answer cited no measurements.'
                : 'Every number in an answer is traced to the tool that measured it. The sources collect here.'}
            </p>
          )}
        </div>
      )}

      {/* ── History ──────────────────────────────────────────────────────── */}
      {tab === 'history' && (
        <div
          role="tabpanel"
          id={panelId('history')}
          aria-labelledby={tabId('history')}
          className="px-4 py-5 md:px-5"
        >
          <PreviousQueries />
        </div>
      )}

      <PipelineDialog
        open={pipelineOpen}
        onOpenChange={(next) => (next ? openPipeline() : closePipeline())}
        nodes={nodes}
        result={result}
        focusedStep={focusedStep}
      />
    </div>
  )
}
