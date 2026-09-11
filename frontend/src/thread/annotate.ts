/**
 * Turning an answer's text plus its citation list into renderable segments.
 *
 * The contract gives `citations[].claim` as a *substring* of the answer, not an
 * offset, so the client has to locate it — and naive `indexOf` gets this wrong
 * in ways that are easy to miss and embarrassing on a projector. From a real
 * recorded run:
 *
 *     "…across 1 distinct region, the largest covering 350,106.12 m2."
 *
 * with claims `"350,106.12 m2"` and `"1"`. Plain `indexOf("1")` lands inside
 * `350,106.12`, striking a pill through the middle of another number. Two rules
 * fix it:
 *
 *   1. **Longest claim first.** A long claim reserves its range before a short
 *      one can steal a character out of it.
 *   2. **Numeric boundaries.** A claim that begins or ends with a digit will not
 *      match where the neighbouring character is a digit, comma or period, so
 *      `"1"` cannot land inside `350,106.12` or `0.16`.
 *
 * `uncited_numeric_spans` go through the same machinery, because they have the
 * same problem and must never overlap a citation.
 */
import type { Citation } from '@/api/types'

export type SegmentKind = 'plain' | 'citation' | 'uncited'

export interface Segment {
  kind: SegmentKind
  text: string
  /** Present when `kind` is `'citation'`. */
  citation?: Citation
  /** Index into the original arrays, for stable React keys and focus. */
  index?: number
}

interface Claim {
  kind: 'citation' | 'uncited'
  text: string
  index: number
  citation?: Citation
}

interface Range {
  start: number
  end: number
  claim: Claim
}

const DIGIT = /\d/

/**
 * True when `text.slice(start, end)` is not glued to a surrounding number.
 *
 * A separator only continues a number when a digit follows it, which is what
 * distinguishes the "." inside `350,106.12` from the full stop that ends
 * `…350,106.12 m2.` — treating both as numeric rejects the whole claim and the
 * pill silently never renders.
 */
function hasCleanBoundaries(text: string, start: number, end: number): boolean {
  const startsNumeric = DIGIT.test(text[start] ?? '')
  const endsNumeric = DIGIT.test(text[end - 1] ?? '')

  if (startsNumeric && start > 0) {
    const before = text[start - 1]!
    const beforeThat = start > 1 ? text[start - 2]! : ''
    if (DIGIT.test(before)) return false
    if ((before === '.' || before === ',') && DIGIT.test(beforeThat)) return false
  }

  if (endsNumeric && end < text.length) {
    const after = text[end]!
    const afterThat = end + 1 < text.length ? text[end + 1]! : ''
    if (DIGIT.test(after)) return false
    if ((after === '.' || after === ',') && DIGIT.test(afterThat)) return false
  }

  return true
}

function overlaps(ranges: Range[], start: number, end: number): boolean {
  return ranges.some((range) => start < range.end && end > range.start)
}

/** First acceptable occurrence of `claim`, or -1. */
function locate(text: string, claim: string, taken: Range[]): number {
  if (!claim) return -1
  let from = 0
  for (;;) {
    const at = text.indexOf(claim, from)
    if (at === -1) return -1
    const end = at + claim.length
    if (!overlaps(taken, at, end) && hasCleanBoundaries(text, at, end)) return at
    from = at + 1
  }
}

/**
 * Split `text` into ordered segments, marking cited claims and uncited spans.
 *
 * A claim the text does not contain is skipped rather than throwing: the server
 * resolved it against its own string, and a client that blanked the answer over
 * a whitespace difference would be worse than one that renders it unmarked.
 */
export function annotate(
  text: string,
  citations: Citation[] = [],
  uncitedSpans: string[] = [],
): Segment[] {
  if (!text) return []

  const claims: Claim[] = [
    ...citations.map((citation, index) => ({
      kind: 'citation' as const,
      text: citation.claim,
      index,
      citation,
    })),
    ...uncitedSpans.map((span, index) => ({ kind: 'uncited' as const, text: span, index })),
  ]

  // Longest first, so a short claim cannot consume part of a longer one.
  // Ties broken by kind then index, purely so the result is deterministic.
  claims.sort(
    (a, b) => b.text.length - a.text.length || a.kind.localeCompare(b.kind) || a.index - b.index,
  )

  const ranges: Range[] = []
  for (const claim of claims) {
    const at = locate(text, claim.text, ranges)
    if (at === -1) continue
    ranges.push({ start: at, end: at + claim.text.length, claim })
  }

  ranges.sort((a, b) => a.start - b.start)

  const segments: Segment[] = []
  let cursor = 0
  for (const range of ranges) {
    if (range.start > cursor) {
      segments.push({ kind: 'plain', text: text.slice(cursor, range.start) })
    }
    segments.push({
      kind: range.claim.kind,
      text: text.slice(range.start, range.end),
      index: range.claim.index,
      ...(range.claim.citation ? { citation: range.claim.citation } : {}),
    })
    cursor = range.end
  }
  if (cursor < text.length) segments.push({ kind: 'plain', text: text.slice(cursor) })

  return segments
}

/** The parsed form of a `step:{n}/scalars.{a}|{b}` citation source (§3.9). */
export interface ParsedSource {
  step: number
  /** One or more dotted scalar paths; a multi-scalar claim joins them with `|`. */
  paths: string[]
}

export function parseSource(source: string): ParsedSource | null {
  const match = /^step:(\d+)\/scalars\.(.+)$/.exec(source)
  if (!match?.[1] || !match[2]) return null
  return { step: Number(match[1]), paths: match[2].split('|') }
}
