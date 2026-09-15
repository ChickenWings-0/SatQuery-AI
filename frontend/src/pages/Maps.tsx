/**
 * The loaded scene in the world. A HUD around the console's own evidence,
 * not a GIS: the map itself (`maplibre-gl`) is lazy and lives in `MapStage`;
 * with no run loaded the page is an invitation, and with no CRS it is the
 * pixel-space viewer with an honest line.
 */
import { lazy, Suspense, useMemo } from 'react'

import { sceneGeoref } from '@/evidence/georef'
import { groupViews } from '@/evidence/views'
import { Graticule } from '@/components/ui/Graticule'
import { Skeleton } from '@/components/ui/Skeleton'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { useJobStore } from '@/state/job'
import { useUiStore } from '@/state/ui'

const MapStage = lazy(() => import('@/pages/maps/MapStage'))
const PixelStage = lazy(() => import('@/pages/maps/PixelStage'))

function MapSkeleton() {
  return (
    <div className="absolute inset-0 bg-bg-main" aria-busy="true" aria-label="Loading the map">
      <Graticule fade={false} />
      <div className="absolute top-4 left-4 hidden wide:block">
        <Skeleton className="h-64 w-60" />
      </div>
      <div className="absolute top-4 right-4">
        <Skeleton className="h-28 w-9" />
      </div>
      <div className="absolute bottom-4 left-4">
        <Skeleton className="h-9 w-64" />
      </div>
    </div>
  )
}

export default function Maps() {
  const artifacts = useJobStore((state) => state.artifacts)
  const result = useJobStore((state) => state.result)
  const validation = useUiStore((state) => state.validation)
  const setSection = useUiStore((state) => state.setSection)

  const groups = useMemo(
    () => groupViews(artifacts.length > 0 ? artifacts : (result?.artifacts ?? [])),
    [artifacts, result],
  )
  const georef = useMemo(() => sceneGeoref(validation), [validation])
  const sensor = validation?.inputs[0]?.sensor_guess ?? validation?.inputs[0]?.modality ?? null

  return (
    <div className="absolute inset-0">
      <DocumentMeta page="maps" />
      {groups.length === 0 ? (
        <div className="absolute inset-0 grid place-items-center bg-bg-main">
          <Graticule fade={false} />
          <div className="glass relative max-w-sm p-6 text-center">
            <h1 className="t-page">Nothing on the map yet.</h1>
            <p className="t-meta mt-2">
              Run a use case or start a new query; once a run has rendered views, they are placed
              here by the manifest’s georeference.
            </p>
            <div className="mt-5 flex flex-wrap justify-center gap-2.5">
              <button type="button" onClick={() => setSection('usecases')} className="btn-primary">
                Run a use case
              </button>
              <button type="button" onClick={() => setSection('explore')} className="btn-ghost">
                New query
              </button>
            </div>
          </div>
        </div>
      ) : (
        <Suspense fallback={<MapSkeleton />}>
          <h1 className="sr-only">Maps</h1>
          {georef ? (
            <MapStage groups={groups} georef={georef} sensor={sensor} />
          ) : (
            <PixelStage groups={groups} sensor={sensor} />
          )}
        </Suspense>
      )}
    </div>
  )
}
