/**
 * Where the boxes the map draws come from.
 *
 * The backend's grounding tool emits a `BBOX_SET` artifact whose `inline`
 * payload carries every box it parsed, in pixels *and* in Qwen's 0–1000
 * normalised frame, plus a WGS84 envelope where the raster was georeferenced
 * (`text_grounding.py`). That is the structured, already-validated record of
 * what was found. The frontend nevertheless re-parsed the raw answer text with
 * `@/thread/bbox` — a hand-port of the backend parser, tested for parity — so
 * any drift between the two showed up as "the map disagrees with the trace".
 *
 * This module makes the artifact the source of truth and keeps the text parser
 * as the fallback for answers that carry boxes without a grounding step having
 * run (a synthesiser that volunteered coordinates). The result says which was
 * used, so the overlay can tell the judge when it is drawing from prose.
 */
import type { AnalyzeResponse, ArtifactRef } from '@/api/types'
import { BOX_SCALE, parseBboxTokens, type NormalisedBox } from '@/thread/bbox'

export type BoxSource = 'artifact' | 'text' | 'none'

export interface ResolvedBoxes {
  boxes: NormalisedBox[]
  source: BoxSource
  /** The input the boxes were located on (`img_0`, `img_1`), when the artifact says. */
  sourceImage: string | null
}

export const NO_BOXES: ResolvedBoxes = { boxes: [], source: 'none', sourceImage: null }

/** One record of a `BBOX_SET` artifact's `inline.boxes`, as the backend writes it. */
export interface BboxRecord {
  id: string
  label: string | null
  score: number | null
  bbox_px: [number, number, number, number]
  bbox_normalised: [number, number, number, number]
  bbox_wgs84: [number, number, number, number] | null
}

export interface BboxSetInline {
  boxes: BboxRecord[]
  frame: { width: number; height: number }
  source_image: string | null
}

function isQuad(value: unknown): value is [number, number, number, number] {
  return (
    Array.isArray(value) &&
    value.length === 4 &&
    value.every((n) => typeof n === 'number' && Number.isFinite(n))
  )
}

function isRecord(value: unknown): value is BboxRecord {
  if (typeof value !== 'object' || value === null) return false
  const record = value as Record<string, unknown>
  return (
    typeof record['id'] === 'string' &&
    (record['label'] === null || typeof record['label'] === 'string') &&
    (record['score'] === null || typeof record['score'] === 'number') &&
    isQuad(record['bbox_px']) &&
    isQuad(record['bbox_normalised'])
  )
}

/**
 * Runtime guard over `inline`, which is `Record<string, unknown>` on the wire.
 * A malformed payload must degrade to the text fallback, never throw.
 */
export function isBboxSetInline(value: unknown): value is BboxSetInline {
  if (typeof value !== 'object' || value === null) return false
  const inline = value as Record<string, unknown>
  const frame = inline['frame'] as Record<string, unknown> | undefined
  return (
    Array.isArray(inline['boxes']) &&
    inline['boxes'].every(isRecord) &&
    typeof frame === 'object' &&
    frame !== null &&
    typeof frame['width'] === 'number' &&
    typeof frame['height'] === 'number'
  )
}

const clamp = (n: number) => Math.min(BOX_SCALE, Math.max(0, Math.round(n)))

/** An artifact record as the overlay draws it. Inverted boxes are righted; empty ones dropped. */
function toNormalised(record: BboxRecord): NormalisedBox | null {
  const [a, b, c, d] = record.bbox_normalised
  const xMin = clamp(Math.min(a, c))
  const xMax = clamp(Math.max(a, c))
  const yMin = clamp(Math.min(b, d))
  const yMax = clamp(Math.max(b, d))
  if (xMax <= xMin || yMax <= yMin) return null
  return { xMin, yMin, xMax, yMax, label: record.label, score: record.score }
}

/** The last `BBOX_SET` in the run — the same "last synthesiser wins" rule the aggregator uses. */
export function lastBboxSet(artifacts: readonly ArtifactRef[]): ArtifactRef | null {
  for (let index = artifacts.length - 1; index >= 0; index -= 1) {
    const artifact = artifacts[index]
    if (artifact?.type === 'BBOX_SET' && isBboxSetInline(artifact.inline)) return artifact
  }
  return null
}

/**
 * The boxes to draw for an answer, and where they came from.
 *
 * Artifact first; the text parser only when no usable `BBOX_SET` exists.
 */
export function boxesForAnswer(
  answerText: string | null,
  artifacts: readonly ArtifactRef[],
): ResolvedBoxes {
  const set = lastBboxSet(artifacts)
  if (set) {
    const inline = set.inline as unknown as BboxSetInline
    const boxes = inline.boxes.map(toNormalised).filter((box): box is NormalisedBox => box !== null)
    return {
      boxes,
      source: 'artifact',
      sourceImage: typeof inline.source_image === 'string' ? inline.source_image : null,
    }
  }
  if (!answerText) return NO_BOXES
  const boxes = parseBboxTokens(answerText)
  return boxes.length > 0 ? { boxes, source: 'text', sourceImage: null } : NO_BOXES
}

/**
 * {@link boxesForAnswer} for a whole result. `artifacts` is the run's streamed
 * list, which is what the viewer already holds — the result's own `artifacts`
 * may be empty when `include_rendered_views` was off, so it is the fallback.
 */
export function boxesForResult(
  result: AnalyzeResponse | null,
  artifacts: readonly ArtifactRef[],
): ResolvedBoxes {
  return boxesForAnswer(
    result?.answer.text ?? null,
    artifacts.length > 0 ? artifacts : (result?.artifacts ?? []),
  )
}
