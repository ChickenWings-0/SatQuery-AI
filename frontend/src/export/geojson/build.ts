/**
 * Boxes (and, when asked, masks) → an RFC 7946 FeatureCollection.
 *
 * When the run's manifest carried WGS84 bounds, the normalised box corners
 * are mapped through them; otherwise the features are emitted in the 0–1000
 * image frame and the collection says so at the top level — never lon/lat
 * that the run did not have. There is deliberately no `crs` member: 7946 §4
 * forbids it, and QGIS assumes CRS84 for a file without one, which is what
 * a georeferenced export is. A pixel-space file carries `crs: 'pixel'` on
 * every feature so it cannot be mistaken for one.
 */
import { BOX_SCALE, type NormalisedBox } from '@/thread/bbox'
import { bboxOfRings, rewind, roundRing, type Position, type Ring } from '@/export/geojson/winding'

export type Bounds = [west: number, south: number, east: number, north: number]

export interface Polygon {
  type: 'Polygon'
  coordinates: Ring[]
}
export interface MultiPolygon {
  type: 'MultiPolygon'
  coordinates: Ring[][]
}

export interface Feature {
  type: 'Feature'
  geometry: Polygon | MultiPolygon
  properties: Record<string, unknown>
  bbox?: [number, number, number, number]
}

export interface Collection {
  type: 'FeatureCollection'
  features: Feature[]
  bbox?: [number, number, number, number]
  /** Present only in pixel space: the honest line at the top of the file. */
  note?: string
}

/** What the audit trace knows about where a box came from. */
export interface Provenance {
  source_step: number | null
  tool: string | null
  tool_version: string | null
  /** `step:n/scalars.path`, when the box is the subject of a cited scalar. */
  citation: string | null
  trace_id: string | null
  scene_id: string | null
  acquired_at: string | null
  sensor: string | null
}

export const NO_PROVENANCE: Provenance = {
  source_step: null,
  tool: null,
  tool_version: null,
  citation: null,
  trace_id: null,
  scene_id: null,
  acquired_at: null,
  sensor: null,
}

/** One layer a GIS user can toggle: the boxes as seen on one view. */
export interface BoxLayer {
  /** `ViewGroup.key` or the artifact id — a T1/T2 pair yields two layers. */
  view_id: string | null
  boxes: NormalisedBox[]
  provenance?: Partial<Provenance>
}

export const PIXEL_NOTE = `No georeference in the manifest: coordinates are in the 0–${BOX_SCALE} normalised image frame.`

/** Normalised (0–1000, y-down) → lon/lat through the bounds, or unchanged. */
export function corner(x: number, y: number, bounds: Bounds | null): Position {
  if (!bounds) return [x, y]
  const [w, s, e, n] = bounds
  return [w + (x / BOX_SCALE) * (e - w), n - (y / BOX_SCALE) * (n - s)]
}

function boxRing(box: NormalisedBox, bounds: Bounds | null): Ring {
  const raw: Ring = [
    corner(box.xMin, box.yMin, bounds),
    corner(box.xMax, box.yMin, bounds),
    corner(box.xMax, box.yMax, bounds),
    corner(box.xMin, box.yMax, bounds),
  ]
  return roundRing(rewind(raw, true, bounds === null))
}

function pixelProps(bounds: Bounds | null): Record<string, unknown> {
  return bounds ? {} : { crs: 'pixel', frame: BOX_SCALE }
}

function withBbox(feature: Feature): Feature {
  const rings = feature.geometry.type === 'Polygon' ? feature.geometry.coordinates : feature.geometry.coordinates.flat()
  const bbox = bboxOfRings(rings)
  return bbox ? { ...feature, bbox } : feature
}

/** Finish a collection: bbox over everything, and the pixel-space note. */
export function finishCollection(features: Feature[], bounds: Bounds | null): Collection {
  const collection: Collection = { type: 'FeatureCollection', features }
  const bbox = bboxOfRings(
    features.flatMap((f) => (f.geometry.type === 'Polygon' ? f.geometry.coordinates : f.geometry.coordinates.flat())),
  )
  if (bbox) collection.bbox = bbox
  if (!bounds) collection.note = PIXEL_NOTE
  return collection
}

/**
 * The plain form: one layer of boxes, optional extra properties. `SavedList`
 * merges several of these into one file, so it returns the features ready
 * to concatenate.
 */
export function boxesToGeoJson(
  boxes: NormalisedBox[],
  bounds: Bounds | null,
  properties: Record<string, unknown> = {},
): Collection {
  return buildCollection({ layers: [{ view_id: null, boxes }], bounds, properties })
}

export function buildCollection({
  layers,
  bounds,
  provenance = {},
  properties = {},
  masks = [],
}: {
  layers: BoxLayer[]
  bounds: Bounds | null
  provenance?: Partial<Provenance>
  properties?: Record<string, unknown>
  /** Already-built mask features from `masks.ts`, appended as-is. */
  masks?: Feature[]
}): Collection {
  const features: Feature[] = []
  for (const layer of layers) {
    const prov: Provenance = { ...NO_PROVENANCE, ...provenance, ...layer.provenance }
    layer.boxes.forEach((box, index) => {
      features.push(
        withBbox({
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [boxRing(box, bounds)] },
          properties: {
            kind: 'box',
            index,
            label: box.label,
            confidence: box.score,
            view_id: layer.view_id,
            ...prov,
            ...properties,
            ...pixelProps(bounds),
          },
        }),
      )
    })
  }
  return finishCollection([...features, ...masks], bounds)
}

export function footprintToGeoJson(bounds: Bounds, properties: Record<string, unknown> = {}): Collection {
  const [w, s, e, n] = bounds
  const ring = roundRing(rewind([[w, s], [e, s], [e, n], [w, n]], true))
  return finishCollection(
    [withBbox({ type: 'Feature', geometry: { type: 'Polygon', coordinates: [ring] }, properties: { kind: 'footprint', ...properties } })],
    bounds,
  )
}
