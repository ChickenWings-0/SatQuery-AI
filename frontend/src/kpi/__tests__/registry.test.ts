import { describe, expect, it } from 'vitest'

import { MIN_CARDS, formatKpi, selectKpis } from '@/kpi/registry'
import { eventsFixture } from '@/mocks/fixtures'

const trace = (
  eventsFixture.find((event) => event.event === 'done')!.data as {
    trace: { fact_sheet: Record<string, unknown> }
  }
).trace

describe('selectKpis on the recorded bi-temporal run', () => {
  const cards = selectKpis(trace.fact_sheet)

  it('picks the headline change metric first', () => {
    expect(cards[0]?.label).toBe('Scene changed')
    expect(cards[0]?.unit).toBe('%')
    expect(cards[0]?.value).toBeCloseTo(5.3422, 3)
  })

  it('produces a usable number of cards', () => {
    expect(cards.length).toBeGreaterThanOrEqual(MIN_CARDS)
    expect(cards.length).toBeLessThanOrEqual(5)
  })

  it('never surfaces a diagnostic scalar as a headline', () => {
    for (const card of cards) {
      expect(card.id).not.toMatch(/_min|_max|_std|_valid_pct|pixel_count|views_/)
    }
  })

  it('folds a pre/post pair into one card carrying the delta', () => {
    const builtUp = cards.find((card) => card.label === 'Built-up cover')
    expect(builtUp).toBeDefined()
    // pre 99.4998, post 99.5894 — the card shows post and the movement.
    expect(builtUp?.value).toBeCloseTo(99.5894, 3)
    expect(builtUp?.delta).toBeCloseTo(0.0896, 3)
    expect(builtUp?.sourceKeys).toHaveLength(2)
  })

  it('shows no duplicate labels', () => {
    const labels = cards.map((card) => card.label)
    expect(new Set(labels).size).toBe(labels.length)
  })
})

describe('selectKpis fallback', () => {
  it('humanises unknown scalars when nothing curated matches', () => {
    const cards = selectKpis({
      'future_tool.canopy_height_mean': 12.5,
      'future_tool.canopy_cover_pct': 61.25,
      'future_tool.tile_count': 9,
    })
    expect(cards.map((card) => card.label)).toContain('Canopy height mean')
    expect(cards.find((card) => card.label === 'Canopy cover pct')?.unit).toBe('%')
  })

  it('ignores non-numeric and non-finite scalars', () => {
    const cards = selectKpis({
      'spectral_index_analyzer.indices_computed': 'ndvi,ndwi,ndbi',
      'x.broken': Number.NaN,
    })
    expect(cards).toEqual([])
  })

  it('returns nothing for an absent fact sheet', () => {
    expect(selectKpis(undefined)).toEqual([])
  })
})

describe('formatKpi', () => {
  it('uses thousands separators and the declared precision', () => {
    expect(
      formatKpi({
        id: 'x',
        label: 'x',
        value: 350106.12,
        precision: 0,
        sourceKeys: [],
        curated: true,
      }),
    ).toBe('350,106')
  })
})
