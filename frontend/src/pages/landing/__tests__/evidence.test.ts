/**
 * The landing page quotes the recording; this pins the quote to the source.
 */
import { describe, expect, it } from 'vitest'

import events from '@/mocks/captured/events.bitemporal.json'
import { CHANGE, CROSSMODAL, GROUNDING } from '@/pages/landing/evidence'

interface Done {
  answer: { citations: { source: string; value: number }[] }
  confidence: { overall: number }
}

const done = (events as { event: string; data: unknown }[]).find((e) => e.event === 'done')!.data as Done
const cited = (source: string) => done.answer.citations.find((c) => c.source === source)?.value

describe('landing evidence', () => {
  it('quotes the change figures from the recording', () => {
    expect(CHANGE.kind).toBe('recorded')
    expect(cited('step:3/scalars.changed_area_pct')).toBeCloseTo(CHANGE.pct, 2)
    expect(cited('step:3/scalars.changed_area_km2')).toBeCloseTo(CHANGE.km2, 2)
    expect(cited('step:3/scalars.component_count')).toBe(CHANGE.regions)
    expect(done.confidence.overall).toBe(CHANGE.confidence)
  })

  it('declares what nothing measured', () => {
    expect(CROSSMODAL.kind).toBe('synthetic')
    expect(GROUNDING.kind).toBe('synthetic')
    expect(GROUNDING.generator).toMatch(/SYNTHETIC FIXTURE/)
  })
})
