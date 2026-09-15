import { describe, expect, it } from 'vitest'

import { QUOTES } from '@/shell/quotes'
import { quoteFor, quoteIndex } from '@/shell/useDailyQuote'

describe('the daily quote', () => {
  it('has thirty-one entries, each with a speaker and a source', () => {
    expect(QUOTES).toHaveLength(31)
    for (const quote of QUOTES) {
      expect(quote.text.trim().length).toBeGreaterThan(10)
      expect(quote.by.trim().length).toBeGreaterThan(2)
      expect(quote.source.trim().length).toBeGreaterThan(3)
      expect(quote.source).not.toMatch(/attributed/i)
    }
  })

  it('indexes by day of month', () => {
    expect(quoteIndex(new Date(2026, 0, 1))).toBe(0)
    expect(quoteIndex(new Date(2026, 0, 31))).toBe(30)
    expect(quoteFor(new Date(2026, 8, 12))).toBe(QUOTES[11])
  })

  it('never repeats a line', () => {
    expect(new Set(QUOTES.map((q) => q.text)).size).toBe(QUOTES.length)
  })
})
