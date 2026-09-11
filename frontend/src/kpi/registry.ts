/**
 * Choosing the handful of numbers that go on the Key Insights cards.
 *
 * `fact_sheet` is a free-form, tool-namespaced map (`<tool>.<scalar>`,
 * API_CONTRACT §9 row 4). There is no "headline metric" field to read, and a
 * real bi-temporal run emits ~50 scalars — `ndwi_min_pre` is in there next to
 * `changed_area_pct`. Something has to decide which three or four matter.
 *
 * So: a curated table keyed by namespaced scalar, giving a human label, a unit
 * and a priority. Two properties make it safe to ship:
 *
 *   - **Curated entries make the demo scenarios read well.** Every key here was
 *     taken from a real recorded run or from a `scalars_schema` in
 *     `configs/registry.yaml`, not invented.
 *   - **The fallback means an unknown tool is never an empty row.** If fewer
 *     than {@link MIN_CARDS} curated keys are present, unmatched numeric
 *     scalars are humanised and used to fill.
 *
 * Pre/post pairs are folded into one card carrying a delta, because "built-up
 * 99.6% (+0.1)" is the bi-temporal story and two adjacent cards reading 99.5
 * and 99.6 are not.
 */
import { decimal } from '@/format'
import type { AnalyzeResponse } from '@/api/types'

export interface KpiSpec {
  label: string
  unit?: string
  /** Decimal places. Defaults to 2, or 0 when the value is integral. */
  precision?: number
  /** Higher wins when there are more candidates than slots. */
  priority: number
}

export const MIN_CARDS = 3
export const MAX_CARDS = 5

/**
 * Curated scalars, keyed by the namespaced `fact_sheet` key with any `_pre` /
 * `_post` suffix stripped — the pair is folded into one card.
 */
export const KPI_REGISTRY: Record<string, KpiSpec> = {
  // change_statistics — the bi-temporal headline
  'change_statistics.changed_area_pct': { label: 'Scene changed', unit: '%', priority: 100 },
  'change_statistics.changed_area_km2': { label: 'Area changed', unit: 'km²', priority: 95 },
  'change_statistics.component_count': {
    label: 'Change regions',
    precision: 0,
    priority: 72,
  },
  'change_statistics.largest_component_m2': {
    label: 'Largest region',
    unit: 'm²',
    precision: 0,
    priority: 68,
  },
  'change_statistics.changed_area_m2': { label: 'Area changed', unit: 'm²', priority: 40 },

  // image_diff_change — the declared fallback for the detector
  'image_diff_change.changed_area_pct': { label: 'Scene changed', unit: '%', priority: 90 },
  'image_diff_change.magnitude_mean': { label: 'Mean change magnitude', priority: 45 },

  // spectral_index_analyzer — land cover
  'spectral_index_analyzer.built_up_fraction_pct': {
    label: 'Built-up cover',
    unit: '%',
    priority: 86,
  },
  'spectral_index_analyzer.vegetation_fraction_pct': {
    label: 'Vegetation cover',
    unit: '%',
    priority: 84,
  },
  'spectral_index_analyzer.water_fraction_pct': { label: 'Water cover', unit: '%', priority: 82 },
  'spectral_index_analyzer.ndvi_mean': { label: 'Mean NDVI', priority: 60 },
  'spectral_index_analyzer.ndbi_mean': { label: 'Mean NDBI', priority: 58 },
  'spectral_index_analyzer.ndwi_mean': { label: 'Mean NDWI', priority: 56 },

  // sar_backscatter_analyzer
  'sar_backscatter_analyzer.sigma0_vv_mean_db': {
    label: 'Mean VV backscatter',
    unit: 'dB',
    priority: 80,
  },
  'sar_backscatter_analyzer.sigma0_vh_mean_db': {
    label: 'Mean VH backscatter',
    unit: 'dB',
    priority: 74,
  },
  'sar_backscatter_analyzer.vv_vh_ratio_mean': { label: 'VV/VH ratio', priority: 70 },

  // object_counter / raster_statistics
  'object_counter.object_count': { label: 'Objects detected', precision: 0, priority: 88 },
  'raster_statistics.valid_pixel_pct': { label: 'Valid pixels', unit: '%', priority: 30 },
  'raster_statistics.brightness_mean': { label: 'Mean brightness', priority: 25 },

  // crossmodal / physics agreement
  'crossmodal_consistency.agreement_pct': { label: 'Cross-modal agreement', unit: '%', priority: 87 },
  'physics_agreement.agreement_score': { label: 'Physics agreement', priority: 76 },
}

/** Scalars that are diagnostics, never headlines, even via the fallback. */
const NEVER_A_CARD = /(_min|_max|_std|_valid_pct|threshold|pixel_count|views_|min_component)/

export interface KpiCard {
  /** Stable id: the base (suffix-stripped) fact_sheet key. */
  id: string
  label: string
  value: number
  unit?: string
  precision: number
  /** `post - pre` when the scalar exists for both halves of a pair. */
  delta?: number
  /** Every fact_sheet key this card was built from — used for citation focus. */
  sourceKeys: string[]
  /** True when it came from the curated table rather than the fallback. */
  curated: boolean
}

const SUFFIX = /_(pre|post)$/

function humanise(key: string): string {
  const scalar = key.includes('.') ? key.slice(key.indexOf('.') + 1) : key
  const words = scalar.replace(SUFFIX, '').replace(/_/g, ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function unitOf(key: string): string | undefined {
  if (/_pct$/.test(key)) return '%'
  if (/_km2$/.test(key)) return 'km²'
  if (/_m2$/.test(key)) return 'm²'
  if (/_db$/.test(key)) return 'dB'
  return undefined
}

interface Bucket {
  base: string
  pre?: number
  post?: number
  plain?: number
  keys: string[]
}

/** Group `<tool>.<scalar>_pre` / `_post` under one base key. */
function bucketise(sheet: Record<string, unknown>): Map<string, Bucket> {
  const buckets = new Map<string, Bucket>()
  for (const [key, raw] of Object.entries(sheet)) {
    if (typeof raw !== 'number' || !Number.isFinite(raw)) continue
    const suffix = SUFFIX.exec(key)?.[1]
    const base = key.replace(SUFFIX, '')
    const bucket = buckets.get(base) ?? { base, keys: [] }
    bucket.keys.push(key)
    if (suffix === 'pre') bucket.pre = raw
    else if (suffix === 'post') bucket.post = raw
    else bucket.plain = raw
    buckets.set(base, bucket)
  }
  return buckets
}

function cardFrom(bucket: Bucket, spec: KpiSpec | undefined, curated: boolean): KpiCard | null {
  // For a pre/post pair the post value is the answer and the delta is the story.
  const value = bucket.plain ?? bucket.post ?? bucket.pre
  if (value === undefined) return null

  const delta =
    bucket.pre !== undefined && bucket.post !== undefined ? bucket.post - bucket.pre : undefined

  const precision =
    spec?.precision ?? (Number.isInteger(value) && Math.abs(value) < 1e6 ? 0 : 2)

  const unit = spec?.unit ?? unitOf(bucket.base)

  return {
    id: bucket.base,
    label: spec?.label ?? humanise(bucket.base),
    value,
    precision,
    sourceKeys: bucket.keys,
    curated,
    ...(unit !== undefined ? { unit } : {}),
    ...(delta !== undefined ? { delta } : {}),
  }
}

/**
 * Select the cards for one finished run.
 *
 * Curated keys first by priority; if that yields fewer than {@link MIN_CARDS},
 * top up from the remaining numeric scalars so a tool nobody anticipated still
 * produces something rather than an empty row.
 */
export function selectKpis(sheet: Record<string, unknown> | undefined): KpiCard[] {
  if (!sheet) return []
  const buckets = bucketise(sheet)

  const curated: Array<{ card: KpiCard; priority: number }> = []
  const rest: KpiCard[] = []

  for (const bucket of buckets.values()) {
    const spec = KPI_REGISTRY[bucket.base]
    if (spec) {
      const card = cardFrom(bucket, spec, true)
      if (card) curated.push({ card, priority: spec.priority })
    } else if (!NEVER_A_CARD.test(bucket.base)) {
      const card = cardFrom(bucket, undefined, false)
      if (card) rest.push(card)
    }
  }

  curated.sort((a, b) => b.priority - a.priority)
  const chosen = curated.slice(0, MAX_CARDS).map((entry) => entry.card)

  // Deduplicate by label: `change_statistics.changed_area_km2` and `_m2` are the
  // same fact in two units, and two "Area changed" cards is noise.
  const seen = new Set(chosen.map((card) => card.label))
  if (chosen.length < MIN_CARDS) {
    for (const card of rest) {
      if (chosen.length >= MIN_CARDS) break
      if (seen.has(card.label)) continue
      seen.add(card.label)
      chosen.push(card)
    }
  }
  return chosen.filter(
    (card, index) => chosen.findIndex((other) => other.label === card.label) === index,
  )
}

/**
 * Format a card's value for display, with thousands separators.
 *
 * `bucketise` already refuses non-finite scalars, so this should never see one
 * — but {@link confidenceCard} reads `confidence.overall` straight off the
 * response, and a headline card reading "NaN" is an assertion the server never
 * made. `decimal` renders it as `—` instead.
 */
export function formatKpi(card: KpiCard): string {
  return decimal(card.value, card.precision)
}

/** Confidence is not a fact_sheet scalar, but it belongs on the row. */
export function confidenceCard(result: AnalyzeResponse): KpiCard {
  return {
    id: '__confidence__',
    label: 'Confidence',
    value: result.confidence.overall,
    precision: 2,
    sourceKeys: [],
    curated: true,
  }
}

/** True when this card was built from any of the scalars a citation names. */
export function cardMatchesScalar(card: KpiCard, toolName: string, paths: string[]): boolean {
  return paths.some((path) => card.sourceKeys.includes(`${toolName}.${path}`))
}
