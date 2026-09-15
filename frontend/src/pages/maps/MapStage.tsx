/**
 * The georeferenced map: `maplibre-gl`, lazy-loaded on this page only.
 *
 * Every rendered view of the current run is an `image` source placed by the
 * manifest's WGS84 corners; the layer switcher toggles which is visible. A
 * compare view is a second image layer clipped by a CSS `clip-path` on the
 * canvas container — cheaper than two synchronised maps and exact enough for
 * a swipe. The basemap is an online raster source, off by default, and if
 * its tiles fail the store switches it off and says so.
 */
import * as maplibregl from 'maplibre-gl'
import type { ImageSource, Map as MapLibre, MapMouseEvent } from 'maplibre-gl'
import maplibreCss from 'maplibre-gl/dist/maplibre-gl.css?inline'
import { useEffect, useMemo, useRef, useState } from 'react'

import { artifactUrl } from '@/api/client'
import { cornersOf, type Georef } from '@/evidence/georef'
import { primaryOf, type ViewGroup } from '@/evidence/views'
import { Reticle } from '@/components/ui/Reticle'
import { CoordinateLocator } from '@/pages/maps/hud/CoordinateLocator'
import { LayerSwitcher } from '@/pages/maps/hud/LayerSwitcher'
import { ScaleBar } from '@/pages/maps/hud/ScaleBar'
import { SwipeHandle } from '@/pages/maps/hud/SwipeHandle'
import { ZoomHud } from '@/pages/maps/hud/ZoomHud'
import { useReducedMotion } from '@/shell/useReducedMotion'
import { useMapStore } from '@/state/map'
import { toast } from '@/state/notifications'

// Esri World Imagery: the one public satellite tile service with usable terms
// for a demo. Only requested when the user switches the basemap on.
const BASEMAP_TILES =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
const BASEMAP_ATTRIBUTION = 'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics'

let cssInjected = false
function injectCss() {
  if (cssInjected || typeof document === 'undefined') return
  const style = document.createElement('style')
  style.dataset['maplibre'] = ''
  style.textContent = maplibreCss
  document.head.append(style)
  cssInjected = true
}

const EMPTY_STYLE = {
  version: 8 as const,
  sources: {},
  layers: [],
}

export default function MapStage({
  groups,
  georef,
  sensor,
}: {
  groups: ViewGroup[]
  georef: Georef
  sensor: string | null
}) {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibre | null>(null)
  const [ready, setReady] = useState(false)
  const reduced = useReducedMotion()

  const layerKey = useMapStore((state) => state.layerKey)
  const compareKey = useMapStore((state) => state.compareKey)
  const split = useMapStore((state) => state.split)
  const basemap = useMapStore((state) => state.basemap)
  const opacity = useMapStore((state) => state.opacity)
  const zoom = useMapStore((state) => state.zoom)
  const cursor = useMapStore((state) => state.cursor)
  const setLayer = useMapStore((state) => state.setLayer)
  const setCompare = useMapStore((state) => state.setCompare)
  const setSplit = useMapStore((state) => state.setSplit)
  const setBasemap = useMapStore((state) => state.setBasemap)
  const setOpacity = useMapStore((state) => state.setOpacity)
  const setZoom = useMapStore((state) => state.setZoom)
  const setCursor = useMapStore((state) => state.setCursor)

  const active = useMemo(() => groups.find((g) => g.key === layerKey) ?? groups[0] ?? null, [groups, layerKey])
  const compare = useMemo(() => groups.find((g) => g.key === compareKey) ?? null, [groups, compareKey])

  // Mount once.
  useEffect(() => {
    if (!container.current) return
    injectCss()
    const map = new maplibregl.Map({
      container: container.current,
      style: EMPTY_STYLE,
      bounds: georef.bounds,
      fitBoundsOptions: { padding: 48 },
      attributionControl: false,
      pitchWithRotate: false,
      dragRotate: false,
    })
    map.touchZoomRotate.disableRotation()
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')
    mapRef.current = map

    map.on('load', () => {
      // The scene footprint, drawn immediately so the user sees *where* before
      // the raster arrives.
      const [w, s, e, n] = georef.bounds
      map.addSource('footprint', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
        },
      })
      map.addLayer({
        id: 'footprint-line',
        type: 'line',
        source: 'footprint',
        paint: { 'line-color': '#dfa878', 'line-width': 1.5, 'line-dasharray': [3, 2], 'line-opacity': 0.8 },
      })
      setReady(true)
    })
    map.on('move', () => setZoom(map.getZoom()))
    map.on('mousemove', (event: MapMouseEvent) => setCursor({ lon: event.lngLat.lng, lat: event.lngLat.lat }))
    map.on('mouseout', () => setCursor(null))
    map.on('error', (event) => {
      const source = (event as unknown as { sourceId?: string }).sourceId
      if (source === 'basemap' && useMapStore.getState().basemap === 'satellite') {
        useMapStore.getState().setBasemap('none', { remember: false })
        toast('Basemap unreachable — showing scene only.', 'warn', {
          label: 'Retry',
          run: () => useMapStore.getState().setBasemap('satellite'),
        })
      }
    })
    return () => {
      map.remove()
      mapRef.current = null
      setReady(false)
    }
    // The map is rebuilt when the scene changes; the georef identity is the key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [georef.bounds.join(',')])

  // Scene layers: one image source per group, visibility follows the store.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const corners = cornersOf(georef.bounds)
    for (const group of groups) {
      const primary = primaryOf(group)
      const url = primary ? artifactUrl(primary) : null
      if (!url) continue
      const id = `scene-${group.key}`
      if (!map.getSource(id)) {
        map.addSource(id, { type: 'image', url, coordinates: corners })
        map.addLayer(
          { id, type: 'raster', source: id, paint: { 'raster-opacity': 0, 'raster-fade-duration': reduced ? 0 : 220 } },
          'footprint-line',
        )
      }
      const isA = active?.key === group.key
      const isB = compare?.key === group.key
      map.setLayoutProperty(id, 'visibility', isA || isB ? 'visible' : 'none')
      map.setPaintProperty(id, 'raster-opacity', isA || isB ? opacity : 0)
    }
  }, [groups, active, compare, opacity, ready, georef.bounds, reduced])

  // The swipe: clip the B layer's canvas copy. maplibre draws all layers to one
  // canvas, so B is rendered on a second map is overkill — instead A and B are
  // both drawn, and a CSS mask over a duplicate canvas is not possible either.
  // The honest, cheap approach: B is drawn *above* A and clipped by a
  // `raster` `clip` via a polygon source that follows the split.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const aId = active ? `scene-${active.key}` : null
    const bId = compare ? `scene-${compare.key}` : null
    if (!aId || !bId || !map.getLayer(aId) || !map.getLayer(bId)) return
    // Ensure B is above A.
    map.moveLayer(bId, 'footprint-line')
    map.moveLayer(aId, bId)
    // Clip B to the right of the split using the viewport: convert the split
    // x to a longitude at the current view and rebuild B's coordinates.
    const canvas = map.getCanvas()
    const width = canvas.clientWidth
    const x = (split / 100) * width
    const top = map.unproject([x, 0])
    const bottom = map.unproject([x, canvas.clientHeight])
    const [w, s, e, n] = georef.bounds
    const lonSplit = Math.min(e, Math.max(w, (top.lng + bottom.lng) / 2))
    const source = map.getSource(bId) as ImageSource | undefined
    source?.setCoordinates([
      [lonSplit, n],
      [e, n],
      [e, s],
      [lonSplit, s],
    ])
  }, [split, active, compare, ready, georef.bounds, zoom])

  // Basemap on/off.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const has = Boolean(map.getSource('basemap'))
    if (basemap === 'satellite' && !has) {
      map.addSource('basemap', { type: 'raster', tiles: [BASEMAP_TILES], tileSize: 256, attribution: BASEMAP_ATTRIBUTION })
      const first = map.getStyle().layers?.[0]?.id
      map.addLayer({ id: 'basemap', type: 'raster', source: 'basemap' }, first)
    } else if (basemap === 'none' && has) {
      if (map.getLayer('basemap')) map.removeLayer('basemap')
      map.removeSource('basemap')
    }
  }, [basemap, ready])

  function fly(lon: number, lat: number) {
    const map = mapRef.current
    if (!map) return
    if (reduced) map.jumpTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 12) })
    else map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 12), duration: 600, essential: true })
  }

  const centerLat = georef.center[1]

  return (
    <div className="absolute inset-0">
      <div
        ref={container}
        className="absolute inset-0 cursor-crosshair bg-bg-main [&_.maplibregl-ctrl-attrib]:!bg-[var(--color-glass)] [&_.maplibregl-ctrl-attrib]:!text-[10px] [&_.maplibregl-ctrl-attrib]:!text-[var(--color-text-lo)]"
        role="application"
        aria-label={`Map of the loaded scene, ${sensor ?? 'unknown sensor'}, ${georef.crs}`}
        tabIndex={0}
        onKeyDown={(event) => {
          const map = mapRef.current
          if (!map) return
          if (event.key === '+' || event.key === '=') map.zoomIn()
          else if (event.key === '-') map.zoomOut()
          else if (event.key === '0') map.fitBounds(georef.bounds, { padding: 48, duration: reduced ? 0 : 400 })
        }}
      />
      {cursor ? (
        <Reticle size={32} className="absolute top-1/2 left-1/2 hidden -translate-x-1/2 -translate-y-1/2 [@media(hover:hover)]:block" style={{ opacity: 0.35 }} />
      ) : null}

      {compare ? <SwipeHandle split={split} onChange={setSplit} labels={[active?.name ?? 'A', compare.name]} /> : null}

      {/* HUD — desk */}
      <div className="absolute top-4 left-4 hidden wide:block">
        <LayerSwitcher
          groups={groups}
          layerKey={active?.key ?? null}
          compareKey={compare?.key ?? null}
          basemap={basemap}
          opacity={opacity}
          sensor={sensor}
          onLayer={setLayer}
          onCompare={setCompare}
          onBasemap={setBasemap}
          onOpacity={setOpacity}
        />
      </div>
      <div className="absolute top-4 right-4">
        <ZoomHud
          onIn={() => mapRef.current?.zoomIn()}
          onOut={() => mapRef.current?.zoomOut()}
          onFit={() => mapRef.current?.fitBounds(georef.bounds, { padding: 48, duration: reduced ? 0 : 400 })}
        />
      </div>
      <div className="absolute bottom-4 left-4 flex items-end gap-2">
        <div className="wide:hidden">
          <LayerSwitcher
            sheet
            groups={groups}
            layerKey={active?.key ?? null}
            compareKey={compare?.key ?? null}
            basemap={basemap}
            opacity={opacity}
            sensor={sensor}
            onLayer={setLayer}
            onCompare={setCompare}
            onBasemap={setBasemap}
            onOpacity={setOpacity}
          />
        </div>
        <CoordinateLocator cursor={cursor} zoom={zoom} onGo={fly} />
      </div>
      <div className="absolute right-4 bottom-10">
        <ScaleBar zoom={zoom} lat={centerLat} gsdM={georef.gsdM} />
      </div>
    </div>
  )
}
