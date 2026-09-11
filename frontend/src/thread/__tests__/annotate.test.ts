import { describe, expect, it } from 'vitest'

import { annotate, parseSource } from '@/thread/annotate'
import type { Citation } from '@/api/types'

const cite = (claim: string, source = 'step:1/scalars.x'): Citation =>
  ({ claim, source, value: 0 }) as Citation

describe('annotate', () => {
  it('marks a simple claim', () => {
    const segments = annotate('About 7.4% changed.', [cite('7.4%')])
    expect(segments.map((s) => [s.kind, s.text])).toEqual([
      ['plain', 'About '],
      ['citation', '7.4%'],
      ['plain', ' changed.'],
    ])
  })

  it('does not let a one-digit claim land inside a bigger number', () => {
    // The real failure from the recorded run: "1" inside "350,106.12".
    const text = 'across 1 distinct region, the largest covering 350,106.12 m2.'
    const segments = annotate(text, [cite('350,106.12 m2'), cite('1')])
    const cited = segments.filter((s) => s.kind === 'citation').map((s) => s.text)
    expect(cited).toEqual(['1', '350,106.12 m2'])
    // Reassembling the segments must reproduce the answer exactly.
    expect(segments.map((s) => s.text).join('')).toBe(text)
  })

  it('does not match a claim glued to an adjacent digit', () => {
    expect(annotate('value 0.169 here', [cite('0.16')]).some((s) => s.kind === 'citation')).toBe(
      false,
    )
  })

  it('marks the first of two identical claims and leaves the second plain', () => {
    const text = 'NDBI is unchanged (0.07 then 0.07).'
    const segments = annotate(text, [cite('0.07')])
    expect(segments.filter((s) => s.kind === 'citation')).toHaveLength(1)
    expect(segments.map((s) => s.text).join('')).toBe(text)
  })

  it('marks uncited spans with their own kind', () => {
    const segments = annotate('Roughly 3 km2 of it.', [], ['3 km2'])
    expect(segments.find((s) => s.kind === 'uncited')?.text).toBe('3 km2')
  })

  it('never overlaps a citation with an uncited span', () => {
    const text = 'It grew 7.4% or 3 km2.'
    const segments = annotate(text, [cite('7.4%')], ['3 km2'])
    expect(segments.filter((s) => s.kind === 'citation')).toHaveLength(1)
    expect(segments.filter((s) => s.kind === 'uncited')).toHaveLength(1)
    expect(segments.map((s) => s.text).join('')).toBe(text)
  })

  it('skips a claim the text does not contain rather than throwing', () => {
    const text = 'Nothing numeric here.'
    expect(annotate(text, [cite('9.9%')]).map((s) => s.text).join('')).toBe(text)
  })

  it('is lossless for the empty and no-claim cases', () => {
    expect(annotate('')).toEqual([])
    expect(annotate('plain text').map((s) => s.text).join('')).toBe('plain text')
  })
})

describe('parseSource', () => {
  it('parses a single-scalar source', () => {
    expect(parseSource('step:3/scalars.changed_area_pct')).toEqual({
      step: 3,
      paths: ['changed_area_pct'],
    })
  })

  it('parses the pipe-joined multi-scalar form', () => {
    expect(parseSource('step:4/scalars.ndbi_mean_pre|ndbi_mean_post')).toEqual({
      step: 4,
      paths: ['ndbi_mean_pre', 'ndbi_mean_post'],
    })
  })

  it('returns null for anything else', () => {
    expect(parseSource('nonsense')).toBeNull()
  })
})
