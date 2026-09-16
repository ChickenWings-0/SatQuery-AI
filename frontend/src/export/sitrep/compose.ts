/**
 * `SitrepInput` → `SitrepModel`: everything the one-page brief will say,
 * decided here and nowhere else. Pure and synchronous, so vitest can hold
 * it to the property that matters: no number reaches the PDF that did not
 * come through `selectKpis` (the same rows the KPI cards show) or through a
 * citation's own `value`.
 */
import type { AnalyzeResponse, ArtifactRef, AuditTrace, InputManifest, NodeState } from '@/api/types'
import type { Bounds } from '@/export/geojson/build'
import { sceneBounds, sceneIdOf } from '@/export/geojson/provenance'
import { georefOf } from '@/evidence/georef'
import { groupViews, primaryOf, type ViewGroup } from '@/evidence/views'
import { formatKpi, selectKpis, MAX_CARDS } from '@/kpi/registry'
import type { SavedRun } from '@/state/library'
import { annotate, type Segment } from '@/thread/annotate'
import { boxesForResult } from '@/thread/boxes'
import { stripBboxTokens, type NormalisedBox } from '@/thread/bbox'

export interface SitrepInput {
  traceId: string
  query: string
  result: AnalyzeResponse | null
  trace: AuditTrace | null
  manifests: InputManifest[]
  artifacts: ArtifactRef[]
  /** The saved summary, when the run is being exported from the library. */
  saved?: SavedRun | null
  generatedAt: Date
}

export type SitrepStatus = 'OK' | 'DEGRADED' | 'FAILED'

export interface SitrepScenePane {
  /** Artifact id; the renderer resolves the URL. */
  artifactId: string | null
  label: string
}

export interface SitrepModel {
  header: { sceneId: string; generatedAt: string; traceId: string; version: string; status: SitrepStatus }
  query: string
  answer: {
    text: string
    /** Superscript numbering: citation index + 1. */
    segments: Segment[]
    citations: { index: number; claim: string; source: string; value: string }[]
    empty: boolean
  }
  scene: {
    mode: 'single' | 'pair' | 'cross-modal'
    panes: SitrepScenePane[]
    boxes: NormalisedBox[]
    bounds: Bounds | null
    crs: string | null
    gsdM: number | null
    /** The raster's width in pixels, for the scale bar. */
    widthPx: number | null
    sensor: string | null
  }
  measurements: { label: string; value: string; unit?: string; source: string }[]
  toolChain: { step: number; tool: string; version: string; state: NodeState }[]
  checks: { passed: number; total: number } | null
  citations: { bound: number; uncited: number }
  confidence: number | null
  /** What could not be included and why — printed as a footnote, never hidden. */
  notes: string[]
}

const pad = (n: number) => String(n).padStart(2, '0')

/** `2026-09-15 14:32` in local time — a brief is read in the room it was made. */
export function stampOf(at: Date): string {
  return `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())} ${pad(at.getHours())}:${pad(at.getMinutes())}`
}

export function sitrepFileName(sceneId: string, at: Date): string {
  const safe = sceneId.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'scene'
  return `SITREP-${safe}-${at.getFullYear()}${pad(at.getMonth() + 1)}${pad(at.getDate())}-${pad(at.getHours())}${pad(at.getMinutes())}.pdf`
}

function statusOf(result: AnalyzeResponse | null, saved: SavedRun | null | undefined): SitrepStatus {
  if (!result) return saved?.outcome === 'failed' ? 'FAILED' : saved ? 'OK' : 'FAILED'
  const executions = result.trace?.executions ?? []
  if (executions.some((e) => e.status === 'FAILED')) return 'DEGRADED'
  if (executions.some((e) => e.status === 'DEGRADED')) return 'DEGRADED'
  if (result.compatibility.checks.some((c) => c.status === 'FAIL')) return 'DEGRADED'
  return 'OK'
}

function panesOf(result: AnalyzeResponse | null, groups: ViewGroup[]): { mode: SitrepModel['scene']['mode']; panes: SitrepScenePane[] } {
  const pairType = result?.compatibility.pair_type
  const primary = groups[0] ?? null
  if (pairType === 'BI_TEMPORAL' && primary?.pre && primary.post) {
    return {
      mode: 'pair',
      panes: [
        { artifactId: primary.pre.id, label: `T1 · ${primary.name}` },
        { artifactId: primary.post.id, label: `T2 · ${primary.name}` },
      ],
    }
  }
  if (pairType === 'CROSS_MODAL') {
    const optical = groups.find((g) => !/^SAR/.test(g.code)) ?? primary
    const sar = groups.find((g) => /^SAR/.test(g.code)) ?? groups[1] ?? null
    const panes: SitrepScenePane[] = []
    if (optical) panes.push({ artifactId: primaryOf(optical)?.id ?? null, label: `Optical · ${optical.name}` })
    if (sar && sar !== optical) panes.push({ artifactId: primaryOf(sar)?.id ?? null, label: `SAR · ${sar.name}` })
    if (panes.length > 0) return { mode: 'cross-modal', panes }
  }
  return {
    mode: 'single',
    panes: primary ? [{ artifactId: primaryOf(primary)?.id ?? null, label: primary.name }] : [],
  }
}

export function composeSitrep(input: SitrepInput): SitrepModel {
  const { result, trace, saved } = input
  const notes: string[] = []
  const manifests = input.manifests.length > 0 ? input.manifests : (trace?.inputs ?? [])
  const artifacts = input.artifacts.length > 0 ? input.artifacts : (result?.artifacts ?? trace?.artifacts ?? [])
  const groups = groupViews(artifacts)
  const first = manifests[0]
  const georef = first ? georefOf(first) : null

  const answerText = result ? stripBboxTokens(result.answer.text) : ''
  const citations = result?.answer.citations ?? []
  const uncited = result?.answer.uncited_numeric_spans ?? []
  const segments = result ? annotate(answerText, citations, uncited) : []

  const sheet = trace?.fact_sheet ?? result?.trace?.fact_sheet
  const cards = selectKpis(sheet).slice(0, MAX_CARDS)
  const measurements = cards.map((card) => ({
    label: card.label,
    value: formatKpi(card),
    ...(card.unit ? { unit: card.unit } : {}),
    source: card.sourceKeys[0] ?? card.id,
  }))
  if (measurements.length === 0 && saved?.headline) {
    measurements.push({ label: saved.headline.label, value: saved.headline.value, source: 'saved summary' })
  }

  const executions = trace?.executions ?? result?.trace?.executions ?? []
  const toolChain = executions.map((e) => ({ step: e.step, tool: e.tool, version: e.version, state: e.status }))
  if (toolChain.length === 0) notes.push('Tool chain not available — the trace is no longer on the server.')

  const checks = result
    ? {
        passed: result.compatibility.checks.filter((c) => c.status === 'PASS').length,
        total: result.compatibility.checks.filter((c) => c.status !== 'SKIP').length,
      }
    : null

  const boxes = result ? boxesForResult(result, artifacts).boxes : (saved?.boxes ?? [])
  const { mode, panes } = panesOf(result, groups)
  if (panes.length === 0) notes.push('No rendered view was available for the scene panel.')

  const sceneId = sceneIdOf(manifests) ?? input.traceId.slice(0, 8)

  return {
    header: {
      sceneId,
      generatedAt: stampOf(input.generatedAt),
      traceId: input.traceId,
      version: trace?.schema_version ?? result?.trace?.schema_version ?? '1.0',
      status: statusOf(result, saved),
    },
    query: input.query,
    answer: {
      text: answerText,
      segments,
      citations: citations.map((c, index) => ({ index: index + 1, claim: c.claim, source: c.source, value: String(c.value) })),
      empty: answerText.trim() === '',
    },
    scene: {
      mode,
      panes,
      boxes,
      bounds: georef?.bounds ?? saved?.bounds ?? sceneBounds(manifests),
      crs: georef?.crs ?? first?.crs ?? null,
      gsdM: georef?.gsdM ?? first?.gsd_m ?? null,
      widthPx: first?.width ?? null,
      sensor: first?.sensor_guess ?? first?.modality ?? saved?.sensors[0] ?? null,
    },
    measurements,
    toolChain,
    checks,
    citations: { bound: citations.length, uncited: uncited.length },
    confidence: result?.confidence.overall ?? saved?.confidence ?? null,
    notes,
  }
}
