/**
 * The SITREP model, held to the recorded CDVQA run. The property under test
 * is honesty: every measurement on the page came through `selectKpis`, every
 * citation count matches the answer, and an ungrounded answer is flagged
 * rather than hidden.
 */
import { describe, expect, it } from 'vitest'

import type { AnalyzeResponse, ArtifactRef } from '@/api/types'
import { composeSitrep, sitrepFileName, type SitrepInput } from '@/export/sitrep/compose'
import { formatKpi, selectKpis } from '@/kpi/registry'
import { eventsFixture, validateFixture } from '@/mocks/fixtures'
import { eventsForScenario, type Scenario } from '@/mocks/scenarios'

function inputFor(scenario: Scenario): SitrepInput {
  const events = eventsForScenario(eventsFixture, scenario)
  const done = events.find((e) => e.event === 'done')!.data as AnalyzeResponse
  const artifacts = events.filter((e) => e.event === 'artifact').map((e) => e.data as ArtifactRef)
  return {
    traceId: done.trace_id,
    query: 'What changed between these two images?',
    result: done,
    trace: done.trace ?? null,
    manifests: validateFixture.inputs,
    artifacts,
    generatedAt: new Date(2026, 8, 15, 14, 32),
  }
}

describe('composeSitrep', () => {
  it('carries every citation as a numbered superscript and no uncited badge', () => {
    const model = composeSitrep(inputFor('canonical'))
    const input = inputFor('canonical')
    const citations = input.result!.answer.citations ?? []
    expect(model.answer.citations).toHaveLength(citations.length)
    expect(model.answer.citations.map((c) => c.index)).toEqual(citations.map((_, i) => i + 1))
    expect(model.citations).toEqual({ bound: citations.length, uncited: 0 })
    expect(model.answer.segments.filter((s) => s.kind === 'citation')).toHaveLength(citations.length)
  })

  it('flags an ungrounded answer instead of hiding it', () => {
    const model = composeSitrep(inputFor('ungrounded'))
    expect(model.citations.uncited).toBe(2)
    expect(model.answer.segments.filter((s) => s.kind === 'uncited')).toHaveLength(2)
  })

  it('shows only the numbers the KPI cards show', () => {
    const input = inputFor('canonical')
    const model = composeSitrep(input)
    const cards = selectKpis(input.result!.trace!.fact_sheet)
    expect(model.measurements.length).toBeGreaterThan(0)
    expect(model.measurements.length).toBeLessThanOrEqual(5)
    for (const m of model.measurements) {
      const card = cards.find((c) => c.label === m.label)
      expect(card, `${m.label} is not a KPI card`).toBeDefined()
      expect(m.value).toBe(formatKpi(card!))
      expect(card!.sourceKeys).toContain(m.source)
    }
  })

  it('reads the tool chain and the compatibility count from the trace', () => {
    const model = composeSitrep(inputFor('canonical'))
    expect(model.toolChain.map((s) => s.tool)).toEqual(['spectral_renderer', 'image_diff_change', 'change_statistics', 'spectral_index_analyzer', 'vlm_change_vqa'])
    expect(model.checks?.total).toBeGreaterThan(0)
    expect(model.header.status).toBe('DEGRADED') // the recording has degraded steps
    expect(model.scene.mode).toBe('pair')
    expect(model.scene.panes).toHaveLength(2)
    expect(model.scene.bounds).not.toBeNull()
  })

  it('burns in the grounded boxes', () => {
    const model = composeSitrep(inputFor('grounded'))
    expect(model.scene.boxes).toHaveLength(1)
    expect(model.scene.boxes[0]?.label).toBe('compound')
  })

  it('degrades honestly without a trace', () => {
    const base = inputFor('canonical')
    const model = composeSitrep({
      ...base,
      result: null,
      trace: null,
      manifests: [],
      artifacts: [],
      saved: {
        traceId: base.traceId,
        query: base.query,
        taskType: 'CHANGE_VQA',
        pairType: 'BI_TEMPORAL',
        sensors: ['Sentinel-2'],
        savedAt: 0,
        ranAt: 0,
        outcome: 'succeeded',
        confidence: 0.61,
        headline: { label: 'Scene changed', value: '5.34' },
        boxes: [],
        bounds: [77.17, 28.54, 77.2, 28.57],
        thumb: null,
        projectId: null,
        tags: [],
      },
    })
    expect(model.toolChain).toEqual([])
    expect(model.notes.join(' ')).toMatch(/no longer on the server/)
    expect(model.measurements[0]).toMatchObject({ label: 'Scene changed', value: '5.34' })
    expect(model.header.status).toBe('OK')
  })

  it('names the file after the scene and the minute', () => {
    expect(sitrepFileName('s2_pre', new Date(2026, 8, 15, 14, 32))).toBe('SITREP-s2_pre-20260915-1432.pdf')
    expect(sitrepFileName('a/b c', new Date(2026, 0, 1, 0, 0))).toBe('SITREP-a-b-c-20260101-0000.pdf')
  })
})
