/**
 * F4c — Key Insights.
 *
 * Every number here is a `fact_sheet` scalar, chosen by `@/kpi/registry`. A card
 * lights its terracotta edge when a citation naming its scalar is clicked, which
 * is the visible link between the sentence and the measurement.
 *
 * Each card opens with a sand glyph that says what *kind* of number it holds —
 * an area, a count, a vegetation index, a confidence — so the row scans by
 * shape before it is read. The glyph is chosen by the card's id and unit, and
 * falls back to a plain grid mark; it never changes the number.
 *
 * The grid is `auto-fit` rather than a fixed column count: with five cards and
 * four columns the fifth was orphaned on a row of its own, which looked like a
 * bug. `auto-fit` lets the row absorb whatever the run happened to produce.
 */
import type { ComponentType } from 'react'

import { DatasetsIcon, type IconProps } from '@/components/ui/icons'
import { decimal } from '@/format'
import { formatKpi, type KpiCard } from '@/kpi/registry'

function AreaIcon(props: IconProps) {
  return (
    <svg width={props.size ?? 20} height={props.size ?? 20} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3.5 6.5 10 3l6.5 3.5-6.5 3.5-6.5-3.5ZM3.5 10 10 13.5 16.5 10M3.5 13.5 10 17l6.5-3.5" />
    </svg>
  )
}

function LeafIcon(props: IconProps) {
  return (
    <svg width={props.size ?? 20} height={props.size ?? 20} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M16.5 3.5c-7 0-11.5 3.5-11.5 9 0 1.25.25 2.25.75 3.25C7.5 10 11 7.5 16.5 3.5ZM5.75 15.75C7 12.5 9.5 9.5 13 7.5" />
      <path d="M16.5 3.5c.5 7-3 12-8.5 12.5" />
    </svg>
  )
}

function CountIcon(props: IconProps) {
  return (
    <svg width={props.size ?? 20} height={props.size ?? 20} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 4v12M13 4v12M3.5 8h13M3.5 12h13" />
    </svg>
  )
}

function GaugeIcon(props: IconProps) {
  return (
    <svg width={props.size ?? 20} height={props.size ?? 20} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3.5 13.5a6.5 6.5 0 0 1 13 0" />
      <path d="m10 13.5 3-4" />
      <circle cx="10" cy="13.5" r="1" fill="currentColor" stroke="none" />
    </svg>
  )
}

function WaterIcon(props: IconProps) {
  return (
    <svg width={props.size ?? 20} height={props.size ?? 20} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10 3.5s5 5.25 5 8.5a5 5 0 0 1-10 0c0-3.25 5-8.5 5-8.5Z" />
    </svg>
  )
}

/** The glyph for a card, from what its id and unit say it measures. */
function glyphFor(card: KpiCard): ComponentType<IconProps> {
  const id = card.id
  if (id === '__confidence__' || /agreement/.test(id)) return GaugeIcon
  if (/ndvi|vegetation/.test(id)) return LeafIcon
  if (/ndwi|water/.test(id)) return WaterIcon
  if (/count|regions/.test(id)) return CountIcon
  if (card.unit === 'km²' || card.unit === 'm²' || card.unit === '%') return AreaIcon
  return DatasetsIcon
}

/** The parenthetical under the number: where it came from, in four words. */
function noteFor(card: KpiCard, degraded: boolean): string {
  if (degraded) return '(model estimate)'
  if (card.id === '__confidence__') return '(overall)'
  if (card.delta !== undefined) return card.delta >= 0 ? '(increase)' : '(decrease)'
  return '(in analysed region)'
}

export function KpiCards({
  cards,
  activeId,
  degradedIds,
}: {
  cards: KpiCard[]
  activeId: string | null
  degradedIds: ReadonlySet<string>
}) {
  if (cards.length === 0) return null

  return (
    <section className="shrink-0">
      <h2 className="t-eyebrow">Key insights</h2>
      <ul
        className="mt-3 grid gap-3"
        style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))' }}
      >
        {cards.map((card) => {
          const active = card.id === activeId
          const degraded = degradedIds.has(card.id)
          const Glyph = glyphFor(card)
          return (
            <li
              key={card.id}
              data-kpi={card.id}
              // The selection, as state rather than as a class name, so a
              // test or a stylesheet can find it without knowing the colour.
              data-active={active || undefined}
              className={`rounded-xl border bg-surface-card px-4 py-3.5 transition-colors ${
                active ? 'border-accent-warm ring-2 ring-accent-glow' : 'border-line'
              }`}
            >
              <p className="flex items-center gap-2 text-[11px] font-medium text-text-lo">
                <Glyph size={16} className="shrink-0 text-accent-warm-text" />
                <span className="truncate" title={card.label}>
                  {card.label}
                </span>
                {degraded && (
                  <span
                    role="img"
                    aria-label="Produced by a step that did not run cleanly"
                    className="ml-auto shrink-0 text-[10px] text-warn"
                    title="Produced by a step that did not run cleanly"
                  >
                    ▲
                  </span>
                )}
              </p>

              <p className="tabular mt-2 flex flex-wrap items-baseline gap-1">
                {/* `min-w-0` plus wrapping: `largest_component_m2` on a big
                    scene is nine digits with separators, and at 160px of card
                    it used to run straight out of the border. */}
                <span className="min-w-0 text-[30px] leading-none font-semibold tracking-[-0.02em] [overflow-wrap:anywhere]">
                  {formatKpi(card)}
                </span>
                {card.unit && (
                  <span className="text-[13px] font-medium text-accent-warm-text">
                    {card.unit}
                  </span>
                )}
              </p>

              {/* Reserved whether or not there is a delta, so every card in the
                  row shares one baseline. */}
              <p className="tabular mt-1.5 flex flex-wrap items-center gap-x-1.5 text-[11px] text-text-lo">
                {card.delta !== undefined && (
                  <span>
                    <span className={card.delta >= 0 ? 'text-ok-text' : 'text-fail'}>
                      {card.delta >= 0 ? '▲' : '▼'}
                    </span>{' '}
                    {decimal(Math.abs(card.delta), card.precision)}
                    {card.unit ?? ''} vs pre
                  </span>
                )}
                <span className={degraded ? 'text-warn-text' : ''}>{noteFor(card, degraded)}</span>
              </p>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
