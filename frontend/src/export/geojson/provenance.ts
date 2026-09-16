/**
 * What the audit trace says about where the boxes and masks came from —
 * pure, shared by the GeoJSON exporter and the SITREP so both attribute the
 * same step to the same box.
 */
import type { AnalyzeResponse, ArtifactRef, AuditTrace, InputManifest } from '@/api/types'
import { NO_PROVENANCE, type BoxLayer, type Bounds, type Provenance } from '@/export/geojson/build'
import type { MaskSource } from '@/export/geojson/masks'
import { georefOf } from '@/evidence/georef'
import { groupForArtifact, groupViews } from '@/evidence/views'
import { artifactUrl } from '@/api/client'
import { boxesForResult } from '@/thread/boxes'
import { lastBboxSet } from '@/thread/boxes'
import { parseSource } from '@/thread/annotate'

export function sceneBounds(manifests: readonly InputManifest[] | undefined): Bounds | null {
  for (const manifest of manifests ?? []) {
    const g = georefOf(manifest)
    if (g) return g.bounds
  }
  return null
}

export function sceneIdOf(manifests: readonly InputManifest[] | undefined): string | null {
  const first = manifests?.[0]
  if (!first) return null
  return first.filename.replace(/\.[a-z0-9]+$/i, '') || first.id
}

/** Provenance for the run as a whole; per-box fields are filled by {@link boxLayers}. */
export function runProvenance(trace: AuditTrace | null | undefined, traceId: string | null): Provenance {
  const manifests = trace?.inputs ?? []
  const first = manifests[0]
  return {
    ...NO_PROVENANCE,
    trace_id: trace?.trace_id ?? traceId,
    scene_id: sceneIdOf(manifests),
    acquired_at: first?.acquisition_time ?? null,
    sensor: first?.sensor_guess ?? first?.modality ?? null,
  }
}

/**
 * The boxes as layers, one per view they were drawn on. A single-image run
 * is one layer; a T1/T2 pair yields the same boxes on both halves so QGIS
 * gets two toggleable layers, which is what the roadmap promised.
 */
export function boxLayers(result: AnalyzeResponse, artifacts: readonly ArtifactRef[]): BoxLayer[] {
  const resolved = boxesForResult(result, artifacts)
  if (resolved.boxes.length === 0) return []
  const all = artifacts.length > 0 ? artifacts : (result.artifacts ?? [])
  const set = lastBboxSet(all)
  const step = set?.produced_by_step ?? null
  const execution = result.trace?.executions.find((e) => e.step === step) ?? null
  const citation =
    result.answer.citations?.find((c) => parseSource(c.source)?.step === step)?.source ?? null
  const provenance: Partial<Provenance> = {
    source_step: step,
    tool: execution?.tool ?? null,
    tool_version: execution?.version ?? null,
    citation,
  }
  const groups = groupViews([...all])
  const home = (resolved.sourceImage && groupForArtifact(groups, resolved.sourceImage)) || groups[0] || null
  const halves = home ? [home.pre, home.post, home.single].filter((a): a is ArtifactRef => a !== null) : []
  if (halves.length <= 1) {
    return [{ view_id: halves[0]?.id ?? home?.key ?? null, boxes: resolved.boxes, provenance }]
  }
  return halves.map((half) => ({ view_id: half.id, boxes: resolved.boxes, provenance }))
}

/** The mask rasters this run rendered, as inputs to the lazy vectoriser. */
export function maskSources(result: AnalyzeResponse | null, artifacts: readonly ArtifactRef[]): MaskSource[] {
  const all = artifacts.length > 0 ? artifacts : (result?.artifacts ?? [])
  const sources: MaskSource[] = []
  for (const artifact of all) {
    if (artifact.type !== 'CHANGE_MASK' && artifact.type !== 'SEGMENTATION') continue
    const url = artifactUrl(artifact)
    if (!url) continue
    sources.push({
      url,
      id: artifact.id,
      kind: artifact.type,
      label: artifact.label,
      produced_by_step: artifact.produced_by_step,
    })
  }
  return sources
}
