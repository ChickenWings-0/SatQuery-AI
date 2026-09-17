/**
 * The loaded scene in the world. A HUD around the console's own evidence,
 * not a GIS: the map itself (`maplibre-gl`) is lazy and lives in `MapStage`;
 * with no run loaded the page is an invitation — and, since Track 4.3, the
 * place to *find* a scene: the discovery column opens over a world view and
 * sends one or two Sentinel scenes to the console as a new query.
 */
import { lazy, Suspense, useMemo, useState } from 'react'

import { sceneGeoref } from '@/evidence/georef'
import { groupViews } from '@/evidence/views'
import { Graticule } from '@/components/ui/Graticule'
import { SearchIcon } from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/Skeleton'
import { Discover } from '@/pages/maps/hud/Discover'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { useJobStore } from '@/state/job'
import { useStacStore } from '@/state/stac'
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
  const [discovering, setDiscovering] = useState(false)

  const groups = useMemo(
    () => groupViews(artifacts.length > 0 ? artifacts : (result?.artifacts ?? [])),
    [artifacts, result],
  )
  const georef = useMemo(() => sceneGeoref(validation), [validation])
  const sensor = validation?.inputs[0]?.sensor_guess ?? validation?.inputs[0]?.modality ?? null
  // Once a place has been searched the world view stays: closing the column
  // must not drop a judge from footprints back to the empty state.
  const explored = useStacStore((state) => state.flyTarget !== null || state.results.length > 0)
  const hasScene = groups.length > 0
  const showMap = (hasScene && georef !== null) || discovering || explored

  return (
    <div className="absolute inset-0">
      <DocumentMeta page="maps" />
      {!showMap && !hasScene ? (
        <div className="absolute inset-0 grid place-items-center bg-bg-main">
          <Graticule fade={false} />
          <div className="glass relative max-w-sm p-6 text-center">
            <h1 className="t-page">Nothing on the map yet.</h1>
            <p className="t-meta mt-2">
              Run a use case or start a new query; once a run has rendered views, they are placed
              here by the manifest’s georeference. Or search the Sentinel catalogue for a scene.
            </p>
            <div className="mt-5 flex flex-wrap justify-center gap-2.5">
              <button type="button" onClick={() => setDiscovering(true)} className="btn-primary inline-flex items-center gap-1.5">
                <SearchIcon size={14} />
                Find imagery
              </button>
              <button type="button" onClick={() => setSection('usecases')} className="btn-ghost">
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
          {showMap ? (
            <MapStage
              groups={georef ? groups : []}
              georef={georef}
              sensor={sensor}
              // The column is 380px at `wide:right-16`: 444px of the stage's right edge.
              insetRight={discovering ? 444 : 0}
              hudCovered={discovering}
            />
          ) : (
            <PixelStage groups={groups} sensor={sensor} />
          )}
        </Suspense>
      )}

      {/*
       * The discovery column. From `wide` it floats on the right, clear of the
       * zoom column (right-4, 36px wide) and of the scale bar (bottom-10 plus
       * its own height) — the bottom inset is what keeps the two from
       * overlapping on a short window. On a phone the map area is ~300px
       * tall under the thread sheet, so the column is a sheet over the whole
       * stage and scrolls as one piece; the map's own HUD sits beneath it and
       * comes back when it closes. The "Find imagery" button lives top-left on
       * a phone because the zoom column already owns the top-right corner.
       */}
      {showMap ? (
        <div className="pointer-events-none absolute inset-0 flex wide:inset-auto wide:top-4 wide:right-16 wide:bottom-[88px] wide:justify-end">
          {discovering ? (
            <div className="pointer-events-auto flex h-full w-full p-2 wide:h-auto wide:max-h-full wide:w-auto wide:p-0">
              <Discover onClose={() => setDiscovering(false)} />
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setDiscovering(true)}
              className="glass pointer-events-auto mt-4 ml-4 flex h-9 items-center gap-1.5 self-start px-3 text-[12.5px] text-text-hi hover:text-accent-warm-text wide:mt-0 wide:ml-0"
            >
              <SearchIcon size={14} className="text-text-lo" />
              Find imagery
            </button>
          )}
        </div>
      ) : null}
    </div>
  )
}
