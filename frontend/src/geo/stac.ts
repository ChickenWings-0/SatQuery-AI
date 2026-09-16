/**
 * STAC item search against Planetary Computer. Plain `fetch`, so MSW sees
 * it under `?mock=1` and vitest replays the recorded response; no React, no
 * stores, no map. The normalised `StacItem` is what the rest of the page
 * reads — the raw item's band-level metadata never leaves this module.
 */
import { type CollectionId, STAC_ROOT, specOf } from '@/geo/collections'
import { fetchWithDeadline, retryAfterOf, UpstreamError } from '@/geo/deadline'
import type { Bounds } from '@/geo/nominatim'

export interface StacQueryInput {
  bbox: Bounds
  /** ISO dates, inclusive. */
  datetime: [from: string, to: string]
  collections: CollectionId[]
  /** Sentinel-2 only; ignored for SAR. */
  cloudMax: number
  limit?: number
}

export interface Footprint {
  type: 'Polygon' | 'MultiPolygon'
  coordinates: number[][][] | number[][][][]
}

export interface StacItem {
  id: string
  collection: CollectionId
  sensor: 'optical' | 'sar'
  datetime: string
  bbox: Bounds
  /** The real footprint — a rotated SAR swath, not its bbox. */
  geometry: Footprint | null
  cloud: number | null
  orbitState: 'ascending' | 'descending' | null
  relativeOrbit: number | null
  platform: string | null
  tile: string | null
  polarisations: string[] | null
  thumbHref: string | null
  epsg: number | null
}

interface RawItem {
  id: string
  collection: string
  bbox?: number[]
  geometry?: StacItem['geometry']
  properties?: Record<string, unknown>
  assets?: Record<string, { href?: string }>
}

export interface RawSearchResponse {
  features?: RawItem[]
  numberMatched?: number
}

export function normaliseItem(raw: RawItem): StacItem | null {
  const spec = specOf(raw.collection)
  if (!spec || !raw.bbox || raw.bbox.length < 4) return null
  const p = raw.properties ?? {}
  const orbit = p['sat:orbit_state']
  const cloud = p['eo:cloud_cover']
  const rel = p['sat:relative_orbit']
  const pol = p['sar:polarizations']
  return {
    id: raw.id,
    collection: spec.id,
    sensor: spec.sensor,
    datetime: String(p['datetime'] ?? ''),
    bbox: [raw.bbox[0]!, raw.bbox[1]!, raw.bbox[2]!, raw.bbox[3]!],
    geometry: raw.geometry ?? null,
    cloud: typeof cloud === 'number' ? cloud : null,
    orbitState: orbit === 'ascending' || orbit === 'descending' ? orbit : null,
    relativeOrbit: typeof rel === 'number' ? rel : null,
    platform: typeof p['platform'] === 'string' ? (p['platform'] as string) : null,
    tile: typeof p['s2:mgrs_tile'] === 'string' ? (p['s2:mgrs_tile'] as string) : null,
    polarisations: Array.isArray(pol) ? pol.map(String) : null,
    thumbHref: raw.assets?.['rendered_preview']?.href ?? null,
    epsg: typeof p['proj:epsg'] === 'number' ? (p['proj:epsg'] as number) : null,
  }
}

/** The request body, exposed so the mock handler and the tests agree on it. */
export function searchBody(input: StacQueryInput): Record<string, unknown> {
  const body: Record<string, unknown> = {
    collections: input.collections,
    bbox: input.bbox,
    datetime: `${input.datetime[0]}T00:00:00Z/${input.datetime[1]}T23:59:59Z`,
    limit: input.limit ?? 24,
    sortby: [{ field: 'properties.datetime', direction: 'desc' }],
  }
  // The cloud filter only makes sense on optical, and a filter naming a
  // property SAR items do not have would drop them all.
  if (input.collections.every((c) => c === 'sentinel-2-l2a')) {
    body['filter-lang'] = 'cql2-json'
    body['filter'] = { op: '<', args: [{ property: 'eo:cloud_cover' }, input.cloudMax] }
  }
  return body
}

export async function searchItems(input: StacQueryInput, signal?: AbortSignal): Promise<StacItem[]> {
  // Optical and SAR carry different filters, so "both" is two requests.
  const optical = input.collections.filter((c) => c === 'sentinel-2-l2a')
  const sar = input.collections.filter((c) => c !== 'sentinel-2-l2a')
  const parts = [optical, sar].filter((list) => list.length > 0)
  const results = await Promise.all(parts.map((collections) => searchOne({ ...input, collections }, signal)))
  return results
    .flat()
    .filter((item) => !input.collections.every((c) => c === 'sentinel-2-l2a') || item.cloud === null || item.cloud < input.cloudMax)
    .sort((a, b) => (a.datetime < b.datetime ? 1 : -1))
}

async function searchOne(input: StacQueryInput, signal?: AbortSignal): Promise<StacItem[]> {
  const response = await fetchWithDeadline(
    `${STAC_ROOT}/search`,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/geo+json' },
      body: JSON.stringify(searchBody(input)),
    },
    signal,
  )
  if (!response.ok) {
    throw new UpstreamError(response.status, `Scene search failed (${response.status}).`, retryAfterOf(response))
  }
  const data = (await response.json()) as RawSearchResponse
  return (data.features ?? []).map(normaliseItem).filter((item): item is StacItem => item !== null)
}

/** `[bbox]` of every item, or null when there are none — for a fit-to-results. */
export function unionBbox(items: readonly StacItem[]): Bounds | null {
  if (items.length === 0) return null
  let [w, s, e, n] = items[0]!.bbox
  for (const item of items) {
    w = Math.min(w, item.bbox[0])
    s = Math.min(s, item.bbox[1])
    e = Math.max(e, item.bbox[2])
    n = Math.max(n, item.bbox[3])
  }
  return [w, s, e, n]
}

/** A T1/T2 SAR pair must share an orbit or the co-registration check fails, correctly but confusingly. */
export function pairIssue(t1: StacItem, t2: StacItem): string | null {
  if (t1.sensor !== t2.sensor) return 'T1 and T2 are different sensors — pick two optical or two SAR scenes for a change pair.'
  if (t1.sensor === 'sar' && t1.orbitState && t2.orbitState && t1.orbitState !== t2.orbitState) {
    return 'These two scenes are on different orbits; the co-registration check will fail. Pick a same-orbit pair.'
  }
  if (t1.sensor === 'sar' && t1.relativeOrbit !== null && t2.relativeOrbit !== null && t1.relativeOrbit !== t2.relativeOrbit) {
    return 'These two scenes are on different relative orbits; pick a same-orbit pair for a clean co-registration.'
  }
  if (t1.id === t2.id) return 'T1 and T2 are the same scene.'
  return null
}
