/**
 * The formatting floor: what happens when the number is not a number.
 *
 * Every case here is reachable from a conforming server response —
 * API_CONTRACT §1 makes an unknown field `null`, not `0` — and each one used to
 * be a `.toFixed()` throwing inside a render.
 */
import { describe, expect, it } from 'vitest'

import {
  ABSENT,
  bytes,
  countOf,
  decimal,
  duration,
  integer,
  percent,
  plural,
  ratioAsPercent,
  scalar,
} from '@/format'

describe('decimal', () => {
  it('renders a finite number at the requested precision', () => {
    expect(decimal(5.336, 2)).toBe('5.34')
    // The thousands separator is the reader's, so only the digits are asserted.
    expect(integer(1500).replace(/\D/g, '')).toBe('1500')
  })

  it('refuses to invent a number the server did not send', () => {
    for (const absent of [null, undefined, NaN, Infinity, -Infinity, 'seven', {}]) {
      expect(decimal(absent)).toBe(ABSENT)
    }
  })
})

describe('percent', () => {
  it('formats a 0-100 value and a 0-1 ratio differently', () => {
    expect(percent(12.34)).toBe('12.3%')
    expect(ratioAsPercent(0.92)).toBe('92%')
  })

  it('degrades to a dash rather than "NaN%"', () => {
    expect(percent(null)).toBe(ABSENT)
    expect(ratioAsPercent(undefined)).toBe(ABSENT)
  })
})

describe('plural', () => {
  it('picks the form the locale asks for', () => {
    expect(plural(1, { one: 'file', other: 'files' })).toBe('file')
    expect(plural(0, { one: 'file', other: 'files' })).toBe('files')
    expect(plural(2, { one: 'file', other: 'files' })).toBe('files')
  })

  it('falls back to `other` for a category the caller did not supply', () => {
    expect(plural(1, { other: 'files' })).toBe('files')
  })

  it('never throws on a value that is not a count', () => {
    expect(countOf(NaN, { one: 'step', other: 'steps' })).toBe(`${ABSENT} steps`)
  })
})

describe('bytes', () => {
  it('promotes to the largest unit that keeps the number above 1', () => {
    expect(bytes(512)).toBe('512 B')
    expect(bytes(1536)).toBe('1.5 kB')
    // A round limit reads as a round number: "512.0 MB" looks like a measurement.
    expect(bytes(512 * 1024 * 1024)).toBe('512 MB')
  })

  it('rejects a negative size rather than rendering one', () => {
    expect(bytes(-1)).toBe(ABSENT)
  })
})

describe('duration', () => {
  it('changes unit as the run gets longer', () => {
    expect(duration(240)).toBe('240 ms')
    expect(duration(2400)).toBe('2.4 s')
    expect(duration(125_000)).toBe('2 min 5 s')
  })
})

describe('scalar', () => {
  it('renders a fact_sheet value of any shape', () => {
    expect(scalar(0.5)).toBe('0.5')
    expect(scalar('OK')).toBe('OK')
    expect(scalar(true)).toBe('true')
    expect(scalar(null)).toBe(ABSENT)
  })

  it('does not print "[object Object]" for a nested value', () => {
    // A tool is free to put a list or a map in the sheet, and the FactSheet tab
    // is the one surface built to prove the run was real.
    expect(scalar({ a: 1 })).toBe('{"a":1}')
    expect(scalar([1, 2])).toBe('[1,2]')
  })
})
