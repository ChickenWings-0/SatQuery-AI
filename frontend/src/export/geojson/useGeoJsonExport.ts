/**
 * The seam between the stores and the GeoJSON builder.
 *
 * Two sources: the live run (job store + pre-flight manifests) and a saved
 * run (its summary, plus the trace re-fetched for the provenance columns).
 * When the run rendered a mask the export goes through the dialog so the
 * user can choose to include it; otherwise it is one click. The planning
 * and building code lives in `plan.ts`, reached through `import()`, so the
 * console entry carries only this hook and its types.
 */
import { useCallback, useState } from 'react'

import type { BoxLayer, Bounds, Provenance } from '@/export/geojson/build'
import type { MaskSource } from '@/export/geojson/masks'
import type { SavedRun } from '@/state/library'
import { toast } from '@/state/notifications'

export type GeoJsonSource = { kind: 'live' } | { kind: 'saved'; run: SavedRun }

export interface GeoJsonPlan {
  fileName: string
  bounds: Bounds | null
  layers: BoxLayer[]
  provenance: Provenance
  masks: MaskSource[]
  /** Set when the saved run's trace could not be fetched: exported without provenance. */
  degraded: string | null
}

let planner: Promise<typeof import('@/export/geojson/plan')> | null = null
function loadPlanner() {
  planner ??= import('@/export/geojson/plan')
  return planner
}

/** The button label: a judge must not load a pixel-space file thinking it is georeferenced. */
export function geoJsonLabel(bounds: Bounds | null | undefined): string {
  return bounds ? 'Export GeoJSON' : 'Export GeoJSON (pixel space)'
}

export function useGeoJsonExport() {
  const [pending, setPending] = useState<GeoJsonPlan | null>(null)
  const [busy, setBusy] = useState(false)

  const start = useCallback(async (source: GeoJsonSource) => {
    if (busy) return
    setBusy(true)
    try {
      const { planGeoJson } = await loadPlanner()
      const plan = await planGeoJson(source)
      if (!plan) {
        toast('Nothing to export yet — run a query first.', 'info')
        return
      }
      if (plan.layers.every((layer) => layer.boxes.length === 0) && plan.masks.length === 0) {
        toast('This run produced no boxes or masks to export.', 'info')
        return
      }
      if (plan.masks.length > 0) setPending(plan)
      else await (await loadPlanner()).exportGeoJson(plan, false)
    } catch (error) {
      toast(`Could not build the GeoJSON — ${error instanceof Error ? error.message : 'unknown error'}.`, 'fail')
    } finally {
      setBusy(false)
    }
  }, [busy])

  const confirm = useCallback(
    async (includeMasks: boolean) => {
      const plan = pending
      if (!plan) return
      setPending(null)
      setBusy(true)
      try {
        const { exportGeoJson } = await loadPlanner()
        await exportGeoJson(plan, includeMasks)
      } catch (error) {
        toast(`Could not build the GeoJSON — ${error instanceof Error ? error.message : 'unknown error'}.`, 'fail')
      } finally {
        setBusy(false)
      }
    },
    [pending],
  )

  const cancel = useCallback(() => setPending(null), [])

  return { start, confirm, cancel, pending, busy }
}
