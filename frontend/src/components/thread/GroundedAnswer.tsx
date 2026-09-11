/**
 * F5a — the answer, with its grounding made visible.
 *
 * Two renderings carry the whole honesty story:
 *
 *   - **Citation pills.** `--color-evidence` as a *fill* behind dark text.
 *     Sand text on the dark ground would pass, but a fill is what makes a
 *     pill read as a control rather than a highlight (9.3:1). Hovering
 *     names the tool and the exact `step:N/scalars.path`; clicking reveals the
 *     evidence, lights the KPI card and arms the pipeline modal on that node.
 *   - **Amber wavy underlines** on `uncited_numeric_spans` — a number the
 *     CitationValidator could not resolve to any measurement. API_CONTRACT §8.5
 *     asks for this explicitly: it is the proof the guard is live rather than
 *     claimed, so a non-empty list is a feature being displayed, not an error.
 *
 * And one thing removed before either happens: bounding-box tokens. A
 * grounding answer carries `<|box_start|>(112,340),(288,512)<|box_end|>` for
 * every object it found, and those are drawn on the imagery by `BboxOverlay`,
 * not read as prose. Stripping is safe for the citations because a claim is
 * located by substring, not by offset — see `annotate.ts`.
 */
import { Fragment, useId, useMemo } from 'react'

import { countOf } from '@/format'
import { annotate, parseSource } from '@/thread/annotate'
import { parseBboxTokens, stripBboxTokens } from '@/thread/bbox'
import type { Answer, Citation } from '@/api/types'

export interface CitationTarget {
  citation: Citation
  step: number
  paths: string[]
}

function CitationPill({
  text,
  citation,
  onActivate,
}: {
  text: string
  citation: Citation
  onActivate: (target: CitationTarget) => void
}) {
  const parsed = parseSource(citation.source)
  const title = parsed
    ? `step ${parsed.step} · ${parsed.paths.join(' | ')} = ${citation.value ?? '—'}`
    : citation.source
  const describedBy = useId()

  return (
    <>
      <button
        type="button"
        // The 2.5.8 "inline target" exemption, claimed explicitly: padding a
        // citation to 44px would double-space the answer paragraph and wreck
        // the reading rhythm the grounding depends on. See theme.css.
        data-inline-target
        title={title}
        // Described, not labelled: the pill's accessible name stays the number
        // you can see, which is what WCAG 2.5.3 wants and what voice control
        // needs to click it. The provenance is a description — the same
        // sentence `title` shows on hover, which reaches neither touch nor
        // most screen readers on its own.
        aria-describedby={describedBy}
        // A `source` this client cannot parse cannot focus anything, so the
        // pill stops pretending to be actionable.
        disabled={!parsed}
        onClick={() =>
          parsed && onActivate({ citation, step: parsed.step, paths: parsed.paths })
        }
        className="mx-px rounded bg-evidence px-1 py-0.5 font-medium text-on-evidence transition-opacity hover:opacity-80 disabled:cursor-default"
      >
        {text}
      </button>
      <span id={describedBy} className="sr-only">
        grounded in {title}
      </span>
    </>
  )
}

export function GroundedAnswer({
  answer,
  onCitation,
}: {
  answer: Answer
  onCitation: (target: CitationTarget) => void
}) {
  const uncited = answer.uncited_numeric_spans ?? []
  const { text, boxCount } = useMemo(
    () => ({
      text: stripBboxTokens(answer.text),
      boxCount: parseBboxTokens(answer.text).length,
    }),
    [answer.text],
  )
  const segments = annotate(text, answer.citations ?? [], uncited)

  return (
    <div>
      <p className="max-w-[68ch] text-[13.5px] leading-[1.65] [overflow-wrap:anywhere]">
        {segments.map((segment, index) => {
          if (segment.kind === 'citation' && segment.citation) {
            return (
              <CitationPill
                key={`c-${index}`}
                text={segment.text}
                citation={segment.citation}
                onActivate={onCitation}
              />
            )
          }
          if (segment.kind === 'uncited') {
            return (
              // The wavy underline is the whole §8.5 honesty signal and it is
              // purely visual: without a spoken equivalent a screen-reader user
              // hears the ungrounded number as if it had been measured. Visibly
              // hidden text rather than `aria-label`, which a role-less `<span>`
              // does not reliably expose — and kept *outside* the `.uncited`
              // span so the marked text is still exactly what was written.
              <Fragment key={`u-${index}`}>
                <span
                  className="uncited"
                  title="Not grounded in any tool output — the citation validator could not resolve this number."
                >
                  {segment.text}
                </span>
                <span className="sr-only"> (not grounded in any tool output)</span>
              </Fragment>
            )
          }
          return <span key={`p-${index}`}>{segment.text}</span>
        })}
      </p>

      {uncited.length > 0 && (
        // The disclaimer, in prose, for the reader who does not know what a
        // wavy underline means. Amber like the underline, and only when there
        // is something to disclaim.
        <div
          role="note"
          className="mt-3.5 flex gap-2.5 rounded-lg border-l-2 border-warn bg-warn/10 px-3 py-2.5 text-[12px] leading-relaxed text-warn-text"
        >
          <span aria-hidden className="shrink-0 text-[11px]">
            ▲
          </span>
          <p>
            {uncited.length === 1 ? 'One value' : `${uncited.length} values`} in this answer{' '}
            {uncited.length === 1 ? 'is' : 'are'} model-derived estimates the citation validator
            could not resolve to a measurement. Verify against higher-resolution data before
            relying on {uncited.length === 1 ? 'it' : 'them'}.
          </p>
        </div>
      )}

      <div className="mt-3.5 flex min-w-0 flex-wrap items-center gap-1.5">
        {boxCount > 0 && (
          <span
            className="chip border border-accent-warm/60 bg-accent-glow font-medium text-accent-warm-text"
            title="Drawn on the imagery in the Data Stage."
          >
            {countOf(boxCount, { one: 'region located', other: 'regions located' })}
          </span>
        )}
        {uncited.length > 0 && (
          <span className="chip border border-warn/60 bg-warn/15 font-medium">
            <span aria-hidden>⚠ </span>
            {countOf(uncited.length, {
              one: 'ungrounded number',
              other: 'ungrounded numbers',
            })}
          </span>
        )}
        {answer.template_fallback && (
          <span
            className="chip border border-line text-text-lo"
            title="Written from the measured evidence by the template, not by the VLM."
          >
            templated answer
          </span>
        )}
        <span
          className="chip-truncate border border-line text-text-lo"
          title={`Answer written by ${answer.generator}`}
        >
          {answer.generator}
        </span>
      </div>
    </div>
  )
}
