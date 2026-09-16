/**
 * The geo clients against the recorded Planetary Computer and Nominatim
 * responses in `public/samples/stac/`, replayed through a stubbed `fetch`.
 */
import { readFileSync } from 'node:fs'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { fetchWithDeadline, OfflineError } from '@/geo/deadline'
import { normalise, search } from '@/geo/nominatim'
import { normaliseItem, pairIssue, searchBody, searchItems, unionBbox, type RawSearchResponse } from '@/geo/stac'

const SAMPLES = new URL('../../../public/samples/stac/', import.meta.url).pathname
const load = (rel: string) => JSON.parse(readFileSync(`${SAMPLES}${rel}`, 'utf8')) as unknown

afterEach(() => vi.unstubAllGlobals())

describe('nominatim', () => {
  it('normalises a recorded response', () => {
    const rows = load('places/ahmedabad.json') as Parameters<typeof normalise>[0][]
    const place = normalise(rows[0]!)
    expect(place.name).toBe('Ahmedabad')
    expect(place.center[0]).toBeCloseTo(72.58, 1)
    expect(place.bbox[0]).toBeLessThan(place.bbox[2])
    expect(place.bbox[1]).toBeLessThan(place.bbox[3])
  })

  it('does not ask below three characters and replays the recording', async () => {
    const fetchMock = vi.fn(async (_url: string) => new Response(JSON.stringify(load('places/chennai.json')), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    expect(await search('Ch')).toEqual([])
    expect(fetchMock).not.toHaveBeenCalled()
    const places = await search('Chennai')
    expect(places[0]?.name).toMatch(/Chennai/)
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain('nominatim.openstreetmap.org/search')
  })
})

describe('stac', () => {
  it('normalises Sentinel-2 and Sentinel-1 items from the recordings', () => {
    const s2 = (load('search/s2-ahmedabad.json') as RawSearchResponse).features!
    const s1 = (load('search/s1-ahmedabad.json') as RawSearchResponse).features!
    const optical = normaliseItem(s2[0]!)!
    const sar = normaliseItem(s1[0]!)!
    expect(optical.sensor).toBe('optical')
    expect(optical.cloud).not.toBeNull()
    expect(optical.tile).toMatch(/^\d{2}[A-Z]{3}$/)
    expect(sar.sensor).toBe('sar')
    expect(sar.orbitState).toBe('descending')
    expect(sar.polarisations).toEqual(['VV', 'VH'])
    expect(sar.geometry?.type).toBe('Polygon')
  })

  it('applies the cloud filter to optical only', () => {
    const base = { bbox: [72, 22, 73, 23] as [number, number, number, number], datetime: ['2026-01-01', '2026-09-01'] as [string, string], cloudMax: 20 }
    expect(searchBody({ ...base, collections: ['sentinel-2-l2a'] })['filter']).toBeDefined()
    expect(searchBody({ ...base, collections: ['sentinel-1-rtc'] })['filter']).toBeUndefined()
  })

  it('replays a recorded search, newest first', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(load('search/s1-ahmedabad.json')), { status: 200 })))
    const items = await searchItems({ bbox: [72, 22, 73, 23], datetime: ['2025-09-01', '2026-09-15'], collections: ['sentinel-1-rtc'], cloudMax: 20 })
    expect(items.length).toBeGreaterThan(10)
    for (let i = 1; i < items.length; i += 1) expect(items[i - 1]!.datetime >= items[i]!.datetime).toBe(true)
    expect(unionBbox(items)).not.toBeNull()
  })

  it('refuses a cross-orbit SAR pair and a mixed-sensor pair', () => {
    const s1 = (load('search/s1-ahmedabad.json') as RawSearchResponse).features!.map((f) => normaliseItem(f)!)
    const s2 = (load('search/s2-ahmedabad.json') as RawSearchResponse).features!.map((f) => normaliseItem(f)!)
    expect(pairIssue(s1[0]!, s1[3]!)).toBeNull()
    expect(pairIssue({ ...s1[0]!, orbitState: 'ascending' }, s1[3]!)).toMatch(/different orbits/)
    expect(pairIssue(s1[0]!, s2[0]!)).toMatch(/different sensors/)
    expect(pairIssue(s1[0]!, s1[0]!)).toMatch(/same scene/)
  })
})

describe('deadline', () => {
  it('turns a hanging request into an OfflineError within the deadline', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init?: RequestInit) => new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
      })),
    )
    await expect(fetchWithDeadline('https://example.invalid/', {}, undefined, 30)).rejects.toBeInstanceOf(OfflineError)
  })

  it('turns a refused connection into an OfflineError', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    }))
    await expect(fetchWithDeadline('https://example.invalid/')).rejects.toBeInstanceOf(OfflineError)
  })
})
