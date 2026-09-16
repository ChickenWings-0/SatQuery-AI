/**
 * Place search over OSM Nominatim. Usage policy: one request a second, an
 * identifying `Referer` (browsers will not let a page set `User-Agent`), and
 * results cached for the session so the same query never asks twice.
 */
import { NOMINATIM_ROOT } from '@/geo/collections'
import { fetchWithDeadline, retryAfterOf, UpstreamError } from '@/geo/deadline'

export type Bounds = [west: number, south: number, east: number, north: number]

export interface Place {
  id: string
  name: string
  /** `city, state, country` — the part after the first comma of the display name. */
  detail: string
  center: [lon: number, lat: number]
  bbox: Bounds
  kind: string
}

interface NominatimRow {
  place_id: number | string
  display_name: string
  lat: string
  lon: string
  boundingbox: [string, string, string, string]
  type?: string
  category?: string
  class?: string
  name?: string
}

export const MIN_QUERY = 3
const MIN_INTERVAL_MS = 1_000

let lastRequestAt = 0
const cache = new Map<string, Place[]>()

export function normalise(row: NominatimRow): Place {
  const [south, north, west, east] = row.boundingbox.map(Number) as [number, number, number, number]
  const display = row.display_name
  const comma = display.indexOf(',')
  const name = row.name?.trim() || (comma > 0 ? display.slice(0, comma) : display)
  return {
    id: String(row.place_id),
    name,
    detail: comma > 0 ? display.slice(comma + 1).trim() : '',
    center: [Number(row.lon), Number(row.lat)],
    bbox: [west, south, east, north],
    kind: row.type ?? row.category ?? row.class ?? 'place',
  }
}

/** Search; throws `OfflineError` on transport failure, `UpstreamError` on 4xx/5xx. */
export async function search(query: string, signal?: AbortSignal): Promise<Place[]> {
  const q = query.trim()
  if (q.length < MIN_QUERY) return []
  const hit = cache.get(q.toLowerCase())
  if (hit) return hit

  const wait = lastRequestAt + MIN_INTERVAL_MS - Date.now()
  if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait))
  if (signal?.aborted) return []
  lastRequestAt = Date.now()

  const url = `${NOMINATIM_ROOT}/search?${new URLSearchParams({ q, format: 'jsonv2', limit: '6', addressdetails: '0' })}`
  const response = await fetchWithDeadline(url, { headers: { accept: 'application/json' } }, signal)
  if (!response.ok) {
    throw new UpstreamError(
      response.status,
      response.status === 429 ? 'Place search is rate-limited — try again in a moment.' : `Place search failed (${response.status}).`,
      retryAfterOf(response),
    )
  }
  const rows = (await response.json()) as NominatimRow[]
  const places = rows.map(normalise)
  cache.set(q.toLowerCase(), places)
  return places
}
