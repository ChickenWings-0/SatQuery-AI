/**
 * Planning and building a GeoJSON export — the part that touches the stores
 * and the builder. Loaded on the click, never at boot.
 */
import { trace as fetchTrace } from '@/api/client'
import type { AuditTrace } from '@/api/types'
import { downloadJson } from '@/export/download'
import { buildCollection, type Feature } from '@/export/geojson/build'
import { boxLayers, maskSources, runProvenance, sceneBounds } from '@/export/geojson/provenance'
import type { GeoJsonPlan, GeoJsonSource } from '@/export/geojson/useGeoJsonExport'
import { useJobStore } from '@/state/job'
import type { SavedRun } from '@/state/library'
import { toast } from '@/state/notifications'
import { useUiStore } from '@/state/ui'

async function planLive(): Promise<GeoJsonPlan | null> {
  const job = useJobStore.getState()
  const result = job.result
  if (!result) return null
  const manifests = useUiStore.getState().validation?.inputs ?? result.trace?.inputs
  const artifacts = job.artifacts.length > 0 ? job.artifacts : (result.artifacts ?? [])
  return {
    fileName: `satquery-${result.trace_id.slice(0, 8)}.geojson`,
    bounds: sceneBounds(manifests),
    layers: boxLayers(result, artifacts),
    provenance: runProvenance(result.trace, result.trace_id),
    masks: maskSources(result, artifacts),
    degraded: null,
  }
}

async function planSaved(run: SavedRun): Promise<GeoJsonPlan> {
  let trace: AuditTrace | null = null
  let degraded: string | null = null
  try {
    trace = await fetchTrace(run.traceId)
  } catch {
    degraded = 'Exported without tool provenance — the server no longer has this trace.'
  }
  const base: GeoJsonPlan = {
    fileName: `satquery-${run.traceId.slice(0, 8)}.geojson`,
    bounds: run.bounds,
    layers: [{ view_id: null, boxes: run.boxes }],
    provenance: runProvenance(trace, run.traceId),
    masks: [],
    degraded,
  }
  if (!trace) return base
  const result = {
    trace_id: trace.trace_id,
    answer: trace.answer,
    artifacts: trace.artifacts,
    confidence: trace.confidence,
    compatibility: trace.compatibility,
    resolved_task: trace.resolved_task,
    trace,
  }
  const layers = boxLayers(result, trace.artifacts)
  return {
    ...base,
    bounds: run.bounds ?? sceneBounds(trace.inputs),
    layers: layers.length > 0 ? layers : base.layers,
    masks: maskSources(result, trace.artifacts),
  }
}

export async function planGeoJson(source: GeoJsonSource): Promise<GeoJsonPlan | null> {
  return source.kind === 'live' ? planLive() : planSaved(source.run)
}

export async function exportGeoJson(plan: GeoJsonPlan, includeMasks: boolean): Promise<void> {
  let masks: Feature[] = []
  if (includeMasks && plan.masks.length > 0) {
    // The vectoriser is its own chunk; nothing pays for it until this click.
    const { vectoriseMask } = await import('@/export/geojson/masks')
    const built = await Promise.all(
      plan.masks.map((source) =>
        vectoriseMask(source, {
          bounds: plan.bounds,
          extraProperties: { trace_id: plan.provenance.trace_id, scene_id: plan.provenance.scene_id },
        }),
      ),
    )
    masks = built.flat()
  }
  const collection = buildCollection({
    layers: plan.layers,
    bounds: plan.bounds,
    provenance: plan.provenance,
    masks,
  })
  downloadJson(plan.fileName, collection)
  if (plan.degraded) toast(plan.degraded, 'warn')
  else toast(plan.bounds ? 'GeoJSON exported in WGS84.' : 'GeoJSON exported in pixel space (no georeference).', 'ok')
}

