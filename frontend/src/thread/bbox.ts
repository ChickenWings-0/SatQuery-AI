/**
 * Bounding boxes out of an answer's text.
 *
 * A port of `src/satquery/models/prompts/box_format.py` — the backend's one
 * box serialiser — so that what the frontend draws is byte-for-byte what the
 * grounding tool parsed. The canonical form is Qwen's normalised box syntax:
 * coordinates scaled to 0–1000 of the image's width and height, wrapped in the
 * box sentinels, optionally preceded by an object reference:
 *
 *     <|object_ref_start|>aircraft<|object_ref_end|><|box_start|>(112,340),(288,512)<|box_end|>
 *
 * Parsing is deliberately more permissive than that, exactly as the backend
 * is. Three forms are tried in order, and the first that yields anything wins:
 *
 *   1. tagged   — the canonical sentinel form above
 *   2. JSON     — `[{"bbox_2d": [x1, y1, x2, y2], "label": "…"}]`, which
 *                 Qwen2.5-VL-derived checkpoints emit
 *   3. bare     — `buildings(170,527),(238,577)`: coordinates with the label in
 *                 front and no sentinels at all. This is what the fine-tuned
 *                 checkpoint actually produces, and the reason the backend grew
 *                 the fallback: a box list that is correct in every digit and
 *                 missing only its wrapper must still reach the map.
 *
 * Every coordinate is clamped to `[0, BOX_SCALE]`, an inverted box is put the
 * right way round, and a box with no extent is dropped — one bad box is a
 * mistake in one box, not in the whole answer. Duplicates collapse.
 */

/** Qwen normalises box coordinates to 0–1000 of each axis, not to 0–1. */
export const BOX_SCALE = 1000

export const REF_START = '<|object_ref_start|>'
export const REF_END = '<|object_ref_end|>'
export const BOX_START = '<|box_start|>'
export const BOX_END = '<|box_end|>'

/** One box in Qwen's normalised 0–1000 frame, top-left origin. */
export interface NormalisedBox {
  xMin: number
  yMin: number
  xMax: number
  yMax: number
  label: string | null
  score: number | null
}

function escapeRe(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

const BOX_BODY = String.raw`\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)`

/** The canonical sentinel form. `s` so a label may span a line, as in Python's DOTALL. */
const TAGGED_RE = new RegExp(
  `(?:${escapeRe(REF_START)}(?<label>.*?)${escapeRe(REF_END)}\\s*)?` +
    `${escapeRe(BOX_START)}\\s*(?<body>${BOX_BODY})\\s*${escapeRe(BOX_END)}`,
  'gs',
)
const BODY_RE = new RegExp(BOX_BODY, 'g')
const JSON_BLOCK_RE = /\[\s*\{.*?\}\s*\]/gs

/** A label does not span a sentence: only the text after the last break counts. */
const SENTENCE_BREAK_RE = /[\n\r.;!?]/
/** Separators that sit between a label and its box, or bracket the label. */
const LABEL_STRIP = new Set([...' \t\r\n,:;-–—"\'`([{'])
/**
 * A referring expression longer than this is prose that happens to precede a
 * box, not a label. The tail is kept rather than the head: the words nearest
 * the box are the ones describing it.
 */
const MAX_LABEL_CHARS = 64

function clamp(value: number): number {
  return Math.max(0, Math.min(BOX_SCALE, Math.trunc(value)))
}

/**
 * Construct a box, or `null` for a degenerate one.
 *
 * Mirrors `_build` + `NormalisedBox.__post_init__`: min/max are sorted so an
 * inverted box is corrected, then clamped, then rejected if it has no extent.
 */
function build(
  values: readonly [number, number, number, number],
  label: string | null,
  score: number | null,
): NormalisedBox | null {
  const [x1, y1, x2, y2] = values
  if (![x1, y1, x2, y2].every(Number.isFinite)) return null
  const box: NormalisedBox = {
    xMin: clamp(Math.min(x1, x2)),
    yMin: clamp(Math.min(y1, y2)),
    xMax: clamp(Math.max(x1, x2)),
    yMax: clamp(Math.max(y1, y2)),
    label,
    score,
  }
  if (box.xMax <= box.xMin || box.yMax <= box.yMin) return null
  return box
}

function bodyValues(match: RegExpMatchArray): [number, number, number, number] {
  return [Number(match[1]), Number(match[2]), Number(match[3]), Number(match[4])]
}

/** Boxes in the canonical sentinel form. */
function parseTagged(text: string): NormalisedBox[] {
  const found: NormalisedBox[] = []
  for (const match of text.matchAll(TAGGED_RE)) {
    const body = new RegExp(BOX_BODY).exec(match.groups?.body ?? '')
    if (!body) continue
    const label = (match.groups?.label ?? '').trim() || null
    const box = build(bodyValues(body), label, null)
    if (box) found.push(box)
  }
  return found
}

function stripLabel(text: string): string {
  let start = 0
  let end = text.length
  while (start < end && LABEL_STRIP.has(text[start]!)) start++
  while (end > start && LABEL_STRIP.has(text[end - 1]!)) end--
  return text.slice(start, end)
}

/**
 * Read the label out of the text between the previous box and this one.
 *
 * `null` when the gap holds no label — an empty gap, or nothing but
 * separators — which is what makes a label carry forward across a run of
 * boxes rather than resetting to unlabelled.
 */
function labelBefore(gap: string): string | null {
  const parts = gap.split(SENTENCE_BREAK_RE)
  let tail = stripLabel(parts[parts.length - 1] ?? '')
  if (tail.length > MAX_LABEL_CHARS) {
    // Keep the words nearest the box, and drop the leading fragment the cut
    // is likely to have sliced through.
    const cut = tail.slice(-MAX_LABEL_CHARS)
    const space = cut.indexOf(' ')
    tail = stripLabel(space === -1 ? '' : cut.slice(space + 1))
  }
  return tail || null
}

/**
 * Boxes written as bare `(x1,y1),(x2,y2)`, label in front, no sentinels.
 *
 * Tried last, because it is the most permissive reading available: any pair
 * of coordinate pairs in the text becomes a box. Labels carry forward, because
 * the observed form puts one label in front of a run of boxes —
 * `buildings(100,738),(330,998)(370,768),(610,1000)` is two buildings, not one
 * building and one anonymous box.
 */
function parseBare(text: string): NormalisedBox[] {
  const found: NormalisedBox[] = []
  let cursor = 0
  let label: string | null = null
  for (const match of text.matchAll(BODY_RE)) {
    const gap = text.slice(cursor, match.index)
    const next = labelBefore(gap)
    if (next !== null) label = next
    cursor = match.index + match[0].length
    const box = build(bodyValues(match), label, null)
    if (box) found.push(box)
  }
  return found
}

function labelOf(entry: Record<string, unknown>): string | null {
  for (const key of ['label', 'text', 'ref', 'object']) {
    const value = entry[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return null
}

function scoreOf(entry: Record<string, unknown>): number | null {
  for (const key of ['score', 'confidence']) {
    const value = entry[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
  }
  return null
}

/**
 * Boxes in the `{"bbox_2d": [...]}` form some Qwen checkpoints emit.
 *
 * Tolerated, never emitted. Absolute-pixel coordinates are only detectable by
 * exceeding the normalised frame, and rescaling them needs the frame they
 * were measured on — so without `width`/`height` such an entry is skipped
 * rather than drawn in the wrong place.
 */
function parseJson(text: string, width?: number, height?: number): NormalisedBox[] {
  const found: NormalisedBox[] = []
  for (const block of text.matchAll(JSON_BLOCK_RE)) {
    let entries: unknown
    try {
      entries = JSON.parse(block[0])
    } catch {
      continue
    }
    if (!Array.isArray(entries)) continue
    for (const entry of entries) {
      if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) continue
      const record = entry as Record<string, unknown>
      const raw = record['bbox_2d'] ?? record['bbox'] ?? record['box']
      if (!Array.isArray(raw) || raw.length !== 4) continue
      const values = raw.map(Number)
      if (!values.every(Number.isFinite)) continue

      const label = labelOf(record)
      const score = scoreOf(record)
      if (Math.max(...values) > BOX_SCALE) {
        if (!width || !height) continue
        const [x1, y1, x2, y2] = values as [number, number, number, number]
        const box = build(
          [
            Math.round((x1 * BOX_SCALE) / width),
            Math.round((y1 * BOX_SCALE) / height),
            Math.round((x2 * BOX_SCALE) / width),
            Math.round((y2 * BOX_SCALE) / height),
          ],
          label,
          score,
        )
        if (box) found.push(box)
        continue
      }
      const box = build(
        values.map((v) => Math.round(v)) as [number, number, number, number],
        label,
        score,
      )
      if (box) found.push(box)
    }
  }
  return found
}

/**
 * Extract every box from an answer, in the order it emitted them.
 *
 * A generation carrying no boxes yields an empty list rather than throwing:
 * "I could not find one" is a legitimate grounding answer.
 */
export function parseBboxTokens(text: string, width?: number, height?: number): NormalisedBox[] {
  let found = parseTagged(text)
  if (found.length === 0) found = parseJson(text, width, height)
  if (found.length === 0) found = parseBare(text)

  const seen = new Set<string>()
  const unique: NormalisedBox[] = []
  for (const box of found) {
    const key = `${box.xMin},${box.yMin},${box.xMax},${box.yMax},${box.label ?? ''}`
    if (seen.has(key)) continue
    seen.add(key)
    unique.push(box)
  }
  return unique
}

/**
 * Remove every box form this module understands, leaving the prose around it.
 *
 * The same three passes as the backend's `strip_boxes`, in the same order, and
 * each replaced by a single space so words on either side do not fuse. Runs of
 * whitespace that result are collapsed so "found <box><box>." reads as
 * "found ." rather than "found   .".
 */
export function stripBboxTokens(text: string): string {
  let out = text.replace(TAGGED_RE, ' ')
  out = out.replace(JSON_BLOCK_RE, ' ')
  out = out.replace(BODY_RE, ' ')
  // Stray sentinels or bare labels glued to a removed box — "buildings ." —
  // are left alone: they are the model's words, and the honesty story here
  // is to show what it said, not to tidy it.
  return out
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/ +([.,;:!?])/g, '$1')
    .trim()
}

/** True when the answer carried at least one drawable box. */
export function hasBboxTokens(text: string): boolean {
  return parseBboxTokens(text).length > 0
}
