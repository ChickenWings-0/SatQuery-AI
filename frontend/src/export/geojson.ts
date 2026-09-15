/**
 * Boxes → GeoJSON. When the run's manifest carried WGS84 bounds, the
 * normalised box corners are mapped through them; otherwise the features are
 * emitted in pixel space and the collection says so at the top level — never
 * lon/lat that the run did not have.
 */
import { BOX_SCALE, type NormalisedBox } from '@/thread/bbox'

export type Bounds = [west: number, south: number, east: number, north: number]

interface Feature {
  type: 'Feature'
  geometry: { type: 'Polygon'; coordinates: number[][][] }
  properties: Record<string, unknown>
}

export interface Collection {
  type: 'FeatureCollection'
  features: Feature[]
  note?: string
  crs?: { type: 'name'; properties: { name: string } }
}

function corner(x: number, y: number, bounds: Bounds | null): [number, number] {
  if (!bounds) return [x, y]
  const [w, s, e, n] = bounds
  return [w + (x / BOX_SCALE) * (e - w), n - (y / BOX_SCALE) * (n - s)]
}

export function boxesToGeoJson(
  boxes: NormalisedBox[],
  bounds: Bounds | null,
  properties: Record<string, unknown> = {},
): Collection {
  const features: Feature[] = boxes.map((box, index) => ({
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          corner(box.xMin, box.yMin, bounds),
          corner(box.xMax, box.yMin, bounds),
          corner(box.xMax, box.yMax, bounds),
          corner(box.xMin, box.yMax, bounds),
          corner(box.xMin, box.yMin, bounds),
        ],
      ],
    },
    properties: {
      index,
      label: box.label,
      score: box.score,
      ...properties,
      ...(bounds ? {} : { crs: 'pixel', frame: BOX_SCALE }),
    },
  }))
  const collection: Collection = { type: 'FeatureCollection', features }
  if (bounds) {
    collection.crs = { type: 'name', properties: { name: 'urn:ogc:def:crs:OGC:1.3:CRS84' } }
  } else {
    collection.note = `No georeference in the manifest: coordinates are in the 0–${BOX_SCALE} normalised image frame.`
  }
  return collection
}

export function footprintToGeoJson(bounds: Bounds, properties: Record<string, unknown> = {}): Collection {
  const [w, s, e, n] = bounds
  return {
    type: 'FeatureCollection',
    crs: { type: 'name', properties: { name: 'urn:ogc:def:crs:OGC:1.3:CRS84' } },
    features: [
      {
        type: 'Feature',
        geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
        properties,
      },
    ],
  }
}

export function downloadJson(name: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/geo+json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.append(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1_000)
}
