/**
 * The georeferenced map: `maplibre-gl`, lazy-loaded on this page only.
 *
 * Every rendered view of the current run is an `image` source placed by the
 * manifest's WGS84 corners; the layer switcher toggles which is visible. A
 * compare view is a second image layer whose corners follow the split. The
 * basemap is an online raster source, off by default, and if its tiles fail
 * the store switches it off and says so.
 *
 * Two things arrived with Track 4.3. The stage can mount with **no run**
 * (`georef === null`): a world view over a tinted land mask, so a judge can
 * search for imagery before anything is loaded. And it draws the discovery
 * store's scene footprints — the real STAC geometry, a rotated SAR swath
 * rather than its bbox — with hover and selection shared with the shelf.
 * The rule that keeps this file sane: **MapStage owns every maplibre object;
 * `state/stac.ts` owns every fact.** Nothing here calls the network.
 *
 * Globe projection is client-side geometry (`map.setProjection`), so it
 * works with the venue Wi-Fi down; the swipe is disabled on the globe
 * because `unproject` near the horizon is not a longitude line.
 */
import * as maplibregl from 'maplibre-gl'
import type { GeoJSONSource, ImageSource, Map as MapLibre, MapMouseEvent } from 'maplibre-gl'
import maplibreCss from 'maplibre-gl/dist/maplibre-gl.css?inline'
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { useEffect, useMemo, useRef, useState } from 'react'

import { artifactUrl } from '@/api/client'
import { cornersOf, type Georef } from '@/evidence/georef'
import { primaryOf, type ViewGroup } from '@/evidence/views'
import { Reticle } from '@/components/ui/Reticle'
import type { StacItem } from '@/geo/stac'
import { CoordinateLocator } from '@/pages/maps/hud/CoordinateLocator'
import { LayerSwitcher } from '@/pages/maps/hud/LayerSwitcher'
import { ScaleBar } from '@/pages/maps/hud/ScaleBar'
import { SwipeHandle } from '@/pages/maps/hud/SwipeHandle'
import { ZoomHud } from '@/pages/maps/hud/ZoomHud'
import { useReducedMotion } from '@/shell/useReducedMotion'
import { mockRequested } from '@/mocks/mode'
import { useMapStore } from '@/state/map'
import { toast } from '@/state/notifications'
import { useStacStore } from '@/state/stac'

// Esri World Imagery: the one public satellite tile service with usable terms
// for a demo. Only requested when the user switches the basemap on.
const BASEMAP_TILES =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
const BASEMAP_ATTRIBUTION = 'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics'

// Natural Earth 110m land, same origin, 130 KB: the world view is never a
// black rectangle and the globe never a black sphere with the Wi-Fi off. A
// vector fill rather than the landing globe's raster mask, because an
// `image` source spanning the world is culled once the map is zoomed in.
const LAND = '/samples/globe/land-110m.json'
const WORLD_CENTER: [number, number] = [78, 22]
const WORLD_ZOOM = 3.2

// MapLibre 6 finds its worker with `new URL('./maplibre-gl-worker.mjs',
// import.meta.url)`. In the dev server that lands in node_modules and works;
// in the built bundle `import.meta.url` is `assets/map-<hash>.js`, the file
// does not exist, and the worker fails silently — no geojson or vector
// source ever loads and the stage is a black rectangle. `?worker&url` has
// Vite bundle the worker (with the 500 KB shared module it imports) as its
// own hashed asset and hand back the address; told once, before any Map.
maplibregl.setWorkerUrl(maplibreWorkerUrl)

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

const FOOTPRINTS = 'stac-footprints'

function footprintCollection(items: readonly StacItem[], selection: { t1: string | null; t2: string | null }, hover: string | null) {
  return {
    type: 'FeatureCollection' as const,
    features: items.map((item) => ({
      type: 'Feature' as const,
      id: item.id,
      properties: {
        id: item.id,
        slot: selection.t1 === item.id ? 'T1' : selection.t2 === item.id ? 'T2' : '',
        hover: hover === item.id,
        sensor: item.sensor,
      },
      geometry: item.geometry ?? {
        type: 'Polygon' as const,
        coordinates: [
          [
            [item.bbox[0], item.bbox[1]],
            [item.bbox[2], item.bbox[1]],
            [item.bbox[2], item.bbox[3]],
            [item.bbox[0], item.bbox[3]],
            [item.bbox[0], item.bbox[1]],
          ],
        ],
      },
    })),
  }
}

export default function MapStage({
  groups,
  georef,
  sensor,
  insetRight = 0,
  hudCovered = false,
}: {
  groups: ViewGroup[]
  /** Null when no run is loaded: the world view, for discovery. */
  georef: Georef | null
  sensor: string | null
  /**
   * Pixels of the stage's right edge covered by a floating column, so a
   * fly-to centres the place in the map the user can see rather than under
   * the panel. Ignored on a phone, where the column is a sheet over all of it.
   */
  insetRight?: number
  /** True while a sheet covers the stage on a phone: the HUD would only ghost through the glass. */
  hudCovered?: boolean
}) {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibre | null>(null)
  // T1/T2 tags on the chosen footprints. DOM markers rather than a symbol
  // layer: an empty style has no glyph source, and two letters do not earn one.
  const markers = useRef<Map<string, maplibregl.Marker>>(new Map())
  const [ready, setReady] = useState(false)
  // Null until the constructor has run; false when this browser gave maplibre
  // no WebGL context (a locked-down lab machine, a headless runner without a
  // GPU path). The stage then says so instead of taking the page down with
  // an error thrown out of an effect — the discovery column still works.
  const [webgl, setWebgl] = useState<boolean | null>(null)
  const reduced = useReducedMotion()

  const layerKey = useMapStore((state) => state.layerKey)
  const compareKey = useMapStore((state) => state.compareKey)
  const split = useMapStore((state) => state.split)
  const basemap = useMapStore((state) => state.basemap)
  const projection = useMapStore((state) => state.projection)
  const opacity = useMapStore((state) => state.opacity)
  const zoom = useMapStore((state) => state.zoom)
  const cursor = useMapStore((state) => state.cursor)
  const setLayer = useMapStore((state) => state.setLayer)
  const setCompare = useMapStore((state) => state.setCompare)
  const setSplit = useMapStore((state) => state.setSplit)
  const setBasemap = useMapStore((state) => state.setBasemap)
  const setProjection = useMapStore((state) => state.setProjection)
  const setOpacity = useMapStore((state) => state.setOpacity)
  const setZoom = useMapStore((state) => state.setZoom)
  const setCursor = useMapStore((state) => state.setCursor)

  // Discovery facts, one field at a time: a keystroke never re-renders the map.
  const stacResults = useStacStore((state) => state.results)
  const stacSelection = useStacStore((state) => state.selection)
  const stacHover = useStacStore((state) => state.hover)
  const flyTarget = useStacStore((state) => state.flyTarget)

  const active = useMemo(() => groups.find((g) => g.key === layerKey) ?? groups[0] ?? null, [groups, layerKey])
  const compare = useMemo(() => groups.find((g) => g.key === compareKey) ?? null, [groups, compareKey])
  const sceneKey = georef?.bounds.join(',') ?? 'world'

  // Mount once per scene (or once for the world view).
  useEffect(() => {
    if (!container.current) return
    injectCss()
    const tags = markers.current
    let map: MapLibre
    try {
      map = new maplibregl.Map({
        container: container.current,
        style: EMPTY_STYLE,
        ...(georef
          ? { bounds: georef.bounds, fitBoundsOptions: { padding: 48 } }
          : { center: WORLD_CENTER, zoom: WORLD_ZOOM }),
        attributionControl: false,
        pitchWithRotate: false,
        dragRotate: false,
      })
    } catch (error) {
      console.warn('[maps] no WebGL context:', error instanceof Error ? error.message : error)
      setWebgl(false)
      return
    }
    setWebgl(true)
    map.touchZoomRotate.disableRotation()
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')
    mapRef.current = map
    // The rehearsal harness (`e2e/track4.spec.ts`) reads layer state off this.
    if (mockRequested()) (window as unknown as { __sqMap?: MapLibre }).__sqMap = map

    map.on('load', () => {
      map.addSource('land', { type: 'geojson', data: LAND })
      map.addLayer({ id: 'land', type: 'fill', source: 'land', paint: { 'fill-color': '#dfa878', 'fill-opacity': 0.16 } })
      map.addLayer({
        id: 'land-line',
        type: 'line',
        source: 'land',
        paint: { 'line-color': '#dfa878', 'line-opacity': 0.35, 'line-width': 0.8 },
      })

      // The discovery footprints sit above the land and below the scene.
      // A search returns a dozen scenes of the *same* tile, so their
      // footprints coincide; any resting fill stacks a dozen times into an
      // opaque plate over the place the user just flew to. At rest a
      // footprint is only its outline — the fill exists for hover and the
      // chosen pair (and as the hit target, which ignores opacity).
      map.addSource(FOOTPRINTS, { type: 'geojson', data: footprintCollection([], { t1: null, t2: null }, null), promoteId: 'id' })
      map.addLayer({
        id: `${FOOTPRINTS}-fill`,
        type: 'fill',
        source: FOOTPRINTS,
        paint: {
          'fill-color': ['case', ['==', ['get', 'sensor'], 'sar'], '#9aa6b2', '#dfa878'],
          'fill-opacity': ['case', ['!=', ['get', 'slot'], ''], 0.22, ['get', 'hover'], 0.14, 0],
        },
      })
      map.addLayer({
        id: `${FOOTPRINTS}-line`,
        type: 'line',
        source: FOOTPRINTS,
        paint: {
          'line-color': ['case', ['!=', ['get', 'slot'], ''], '#ab6242', ['get', 'hover'], '#dfa878', '#8a7462'],
          'line-width': ['case', ['!=', ['get', 'slot'], ''], 2.5, ['get', 'hover'], 1.8, 1],
          'line-opacity': ['case', ['!=', ['get', 'slot'], ''], 1, ['get', 'hover'], 0.95, 0.55],
        },
      })
      map.on('mousemove', `${FOOTPRINTS}-fill`, (event) => {
        const id = event.features?.[0]?.properties?.['id']
        useStacStore.getState().setHover(typeof id === 'string' ? id : null)
        map.getCanvas().style.cursor = 'pointer'
      })
      map.on('mouseleave', `${FOOTPRINTS}-fill`, () => {
        useStacStore.getState().setHover(null)
        map.getCanvas().style.cursor = ''
      })
      map.on('click', `${FOOTPRINTS}-fill`, (event) => {
        // The topmost (smallest) footprint under the pointer wins.
        const hit = event.features?.[0]?.properties?.['id']
        if (typeof hit === 'string') useStacStore.getState().pick(hit)
      })

      if (georef) {
        // The scene footprint, drawn immediately so the user sees *where*
        // before the raster arrives.
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
      }
      // The HUD reads zoom from the store, which only `move` writes; seed it
      // so the scale bar and the locator are right before the first drag.
      setZoom(map.getZoom())
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
        return
      }
      // A listener on 'error' silences maplibre's own console output; anything
      // else is still worth a line in the console.
      console.warn('[maps]', (event as unknown as { error?: Error }).error?.message ?? event)
    })
    return () => {
      for (const marker of tags.values()) marker.remove()
      tags.clear()
      map.remove()
      mapRef.current = null
      setReady(false)
    }
    // The map is rebuilt when the scene changes; the georef identity is the key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneKey])

  // Scene layers: one image source per group, visibility follows the store.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !georef) return
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
  }, [groups, active, compare, opacity, ready, georef, reduced])

  // The swipe: B is drawn above A and clipped to the right of the split by
  // rebuilding its image coordinates from the split's longitude. Off on the
  // globe, where a screen x is not a meridian.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !georef || projection === 'globe') return
    const aId = active ? `scene-${active.key}` : null
    const bId = compare ? `scene-${compare.key}` : null
    if (!aId || !bId || !map.getLayer(aId) || !map.getLayer(bId)) return
    map.moveLayer(bId, 'footprint-line')
    map.moveLayer(aId, bId)
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
  }, [split, active, compare, ready, georef, zoom, projection])

  // Basemap on/off — above the land mask, below everything else.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const has = Boolean(map.getSource('basemap'))
    if (basemap === 'satellite' && !has) {
      map.addSource('basemap', { type: 'raster', tiles: [BASEMAP_TILES], tileSize: 256, attribution: BASEMAP_ATTRIBUTION })
      map.addLayer({ id: 'basemap', type: 'raster', source: 'basemap' }, `${FOOTPRINTS}-fill`)
    } else if (basemap === 'none' && has) {
      if (map.getLayer('basemap')) map.removeLayer('basemap')
      map.removeSource('basemap')
    }
  }, [basemap, ready])

  // Projection: pure geometry, no tiles involved.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    map.setProjection({ type: projection === 'globe' ? 'globe' : 'mercator' })
  }, [projection, ready])

  // Discovery footprints follow the store.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource(FOOTPRINTS) as GeoJSONSource | undefined
    source?.setData(footprintCollection(stacResults, stacSelection, stacHover))

    const wanted = new Map<string, string>()
    if (stacSelection.t1) wanted.set('T1', stacSelection.t1)
    if (stacSelection.t2) wanted.set('T2', stacSelection.t2)
    for (const [slot, marker] of markers.current) {
      if (!wanted.has(slot)) {
        marker.remove()
        markers.current.delete(slot)
      }
    }
    for (const [slot, id] of wanted) {
      const item = stacResults.find((r) => r.id === id)
      if (!item) continue
      const center: [number, number] = [(item.bbox[0] + item.bbox[2]) / 2, (item.bbox[1] + item.bbox[3]) / 2]
      let marker = markers.current.get(slot)
      if (!marker) {
        const el = document.createElement('span')
        el.className = 'rounded-md bg-accent-cool px-1.5 py-0.5 font-mono text-[11px] font-semibold text-white shadow-md'
        el.textContent = slot
        el.setAttribute('aria-hidden', 'true')
        marker = new maplibregl.Marker({ element: el }).setLngLat(center).addTo(map)
        markers.current.set(slot, marker)
      } else {
        marker.setLngLat(center)
      }
    }
  }, [stacResults, stacSelection, stacHover, ready])

  // The padding a fit uses: even all round, plus whatever the discovery
  // column covers on a window wide enough to show it beside the map.
  const fitPadding = (base: number) => {
    const el = container.current
    const wide = el ? el.clientWidth >= 768 && window.innerHeight >= 500 : false
    return { top: base, bottom: base, left: base, right: base + (wide ? insetRight : 0) }
  }

  // A chosen place: fly to it.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !flyTarget) return
    map.fitBounds(flyTarget.bbox, { padding: fitPadding(64), duration: reduced ? 0 : 700, maxZoom: 11 })
    // `insetRight` is read at flight time on purpose: opening or closing the
    // column must not re-fly the map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flyTarget, ready, reduced])

  function fly(lon: number, lat: number) {
    const map = mapRef.current
    if (!map) return
    if (reduced) map.jumpTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 12) })
    else map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 12), duration: 600, essential: true })
  }

  function fit() {
    const map = mapRef.current
    if (!map) return
    if (georef) map.fitBounds(georef.bounds, { padding: fitPadding(48), duration: reduced ? 0 : 400 })
    else if (flyTarget) map.fitBounds(flyTarget.bbox, { padding: fitPadding(64), duration: reduced ? 0 : 400, maxZoom: 11 })
    else map.easeTo({ center: WORLD_CENTER, zoom: WORLD_ZOOM, duration: reduced ? 0 : 400 })
  }

  const centerLat = georef?.center[1] ?? (cursor?.lat ?? WORLD_CENTER[1])
  const showSwipe = compare !== null && projection !== 'globe'
  // Below `wide` the discovery sheet covers the whole stage; the HUD comes
  // back with the map when it closes.
  const hud = hudCovered ? 'hidden wide:block' : ''

  return (
    <div className="absolute inset-0">
      <div
        ref={container}
        // Inline, not `absolute inset-0`: maplibre's own stylesheet sets
        // `.maplibregl-map { position: relative }` and lands after Tailwind's,
        // which collapsed the container to zero height.
        style={{ position: 'absolute', inset: 0 }}
        className="cursor-crosshair bg-bg-main [&_.maplibregl-ctrl-attrib]:!bg-[var(--color-glass)] [&_.maplibregl-ctrl-attrib]:!text-[10px] [&_.maplibregl-ctrl-attrib]:!text-[var(--color-text-lo)]"
        role="application"
        aria-label={
          georef
            ? `Map of the loaded scene, ${sensor ?? 'unknown sensor'}, ${georef.crs}`
            : 'World map for finding imagery'
        }
        tabIndex={0}
        onKeyDown={(event) => {
          const map = mapRef.current
          if (!map) return
          if (event.key === '+' || event.key === '=') map.zoomIn()
          else if (event.key === '-') map.zoomOut()
          else if (event.key === '0') fit()
        }}
      />
      {webgl === false ? (
        <div role="status" className="absolute inset-0 grid place-items-center p-4">
          <div className="glass max-w-xs p-4 text-center">
            <p className="text-[13px] text-text-hi">This browser cannot draw the map.</p>
            <p className="t-meta mt-1">
              WebGL is unavailable here, so footprints and scenes stay in the shelf. Imagery search still works.
            </p>
          </div>
        </div>
      ) : null}
      {cursor ? (
        <Reticle size={32} className="absolute top-1/2 left-1/2 hidden -translate-x-1/2 -translate-y-1/2 [@media(hover:hover)]:block" style={{ opacity: 0.35 }} />
      ) : null}

      {showSwipe && compare ? <SwipeHandle split={split} onChange={setSplit} labels={[active?.name ?? 'A', compare.name]} /> : null}
      {compare && projection === 'globe' ? (
        <p className="glass absolute top-4 left-1/2 -translate-x-1/2 px-3 py-1.5 text-[12px] text-text-lo">Swipe is available in flat view.</p>
      ) : null}

      {/* HUD — desk */}
      {groups.length > 0 ? (
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
      ) : null}
      <div className={`absolute top-4 right-4 ${hud}`}>
        <ZoomHud
          onIn={() => mapRef.current?.zoomIn()}
          onOut={() => mapRef.current?.zoomOut()}
          onFit={fit}
          projection={projection}
          onProjection={setProjection}
        />
      </div>
      <div className={`absolute bottom-4 left-4 flex items-end gap-2 ${hudCovered ? 'hidden wide:flex' : ''}`}>
        {groups.length > 0 ? (
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
        ) : null}
        <CoordinateLocator cursor={cursor} zoom={zoom} onGo={fly} />
      </div>
      <div className={`absolute right-4 bottom-10 ${hud}`}>
        <ScaleBar zoom={zoom} lat={centerLat} gsdM={georef?.gsdM ?? null} />
      </div>
    </div>
  )
}
