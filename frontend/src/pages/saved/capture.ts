/**
 * Turn the current run into a `SavedRun`.
 *
 * Called from the answer card's bookmark. Reads the job store's result and
 * artifacts, the pre-flight manifests (for bounds and sensors), the first KPI
 * (as the headline), the boxes (from the artifact, or the prose fallback),
 * and draws a 320px thumbnail of the primary view onto a canvas. The image
 * fetch is same-origin (`/v1/artifacts/...`), so the canvas is not tainted.
 */
import { artifactUrl } from '@/api/client'
import { groupViews, primaryOf } from '@/evidence/views'
import { confidenceCard, formatKpi, selectKpis } from '@/kpi/registry'
import { useJobStore } from '@/state/job'
import type { SavedRun } from '@/state/library'
import { useUiStore } from '@/state/ui'
import { boxesForResult } from '@/thread/boxes'

const THUMB_W = 320

async function thumbnail(url: string | null): Promise<Blob | null> {
  if (!url || typeof document === 'undefined') return null
  try {
    const img = new Image()
    img.decoding = 'async'
    img.src = url
    await img.decode()
    const scale = Math.min(1, THUMB_W / img.naturalWidth)
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(img.naturalWidth * scale))
    canvas.height = Math.max(1, Math.round(img.naturalHeight * scale))
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    return await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/webp', 0.8))
  } catch {
    return null
  }
}

export async function captureCurrentRun(query: string): Promise<SavedRun | null> {
  const job = useJobStore.getState()
  const result = job.result
  if (!result) return null
  const ui = useUiStore.getState()
  const manifests = ui.validation?.inputs ?? []
  const first = manifests[0]
  const bounds = first?.bounds_wgs84 && first.bounds_wgs84.length === 4
    ? ([...first.bounds_wgs84] as [number, number, number, number])
    : null
  const sensors = [...new Set(manifests.map((m) => m.sensor_guess ?? m.modality).filter(Boolean))]
  const artifacts = job.artifacts.length > 0 ? job.artifacts : (result.artifacts ?? [])
  const groups = groupViews(artifacts)
  const primary = groups[0] ? primaryOf(groups[0]) : null
  const kpis = selectKpis(result.trace?.fact_sheet)
  const head = kpis[0] ?? confidenceCard(result)
  const remembered = ui.recentRuns.find((run) => run.traceId === result.trace_id)
  return {
    traceId: result.trace_id,
    query,
    taskType: result.resolved_task.primary,
    pairType: result.compatibility.pair_type,
    sensors,
    savedAt: Date.now(),
    ranAt: remembered?.at ?? Date.now(),
    outcome: 'succeeded',
    confidence: result.confidence.overall,
    headline: { label: head.label, value: formatKpi(head) },
    boxes: boxesForResult(result, artifacts).boxes,
    bounds,
    thumb: await thumbnail(primary ? artifactUrl(primary) : null),
    projectId: null,
    tags: [],
  }
}
