/**
 * The page itself, rendered in node with the standard fonts (no `fetch`
 * for the woff2 here, so the Helvetica fallback is what is exercised) and
 * parsed back: exactly one page, A4, with the title set.
 */
import { PDFDocument } from 'pdf-lib'
import { describe, expect, it } from 'vitest'

import type { AnalyzeResponse, ArtifactRef } from '@/api/types'
import { composeSitrep } from '@/export/sitrep/compose'
import { renderSitrep } from '@/export/sitrep/render'
import { eventsFixture, validateFixture } from '@/mocks/fixtures'
import { eventsForScenario } from '@/mocks/scenarios'

describe('renderSitrep', () => {
  it('produces exactly one A4 page', async () => {
    const events = eventsForScenario(eventsFixture, 'ungrounded')
    const done = events.find((e) => e.event === 'done')!.data as AnalyzeResponse
    const artifacts = events.filter((e) => e.event === 'artifact').map((e) => e.data as ArtifactRef)
    const model = composeSitrep({
      traceId: done.trace_id,
      query: 'What changed between these two images? '.repeat(6),
      result: done,
      trace: done.trace ?? null,
      manifests: validateFixture.inputs,
      artifacts,
      generatedAt: new Date(2026, 8, 15, 14, 32),
    })
    const bytes = await renderSitrep(model, null)
    expect(bytes.byteLength).toBeGreaterThan(2_000)
    const parsed = await PDFDocument.load(bytes)
    expect(parsed.getPageCount()).toBe(1)
    const [w, h] = parsed.getPage(0).getSize() ? [parsed.getPage(0).getWidth(), parsed.getPage(0).getHeight()] : [0, 0]
    expect(Math.round(w)).toBe(595)
    expect(Math.round(h)).toBe(842)
    expect(parsed.getTitle()).toMatch(/^SITREP /)
  })
})
