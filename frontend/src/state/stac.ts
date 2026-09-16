/**
 * The Maps page's discovery flow: a place, a date window, a sensor choice,
 * the scenes that match, and which one or two the user picked. Session-only
 * and never persisted — a shelf of yesterday's rehearsal scenes appearing
 * on stage is worse than an empty one.
 *
 * `MapStage` owns every maplibre object; this store owns every fact. The map
 * subscribes to `results`, `selection`, `hover` and `flyTarget` one field at
 * a time so a keystroke in the search box never re-renders it.
 */
import { create } from 'zustand'

import { collectionsFor, type CollectionId, type SensorChoice } from '@/geo/collections'
import { isOffline, UpstreamError } from '@/geo/deadline'
import type { Bounds, Place } from '@/geo/nominatim'
import { pairIssue, searchItems, type StacItem } from '@/geo/stac'

export type StacStatus = 'idle' | 'searching' | 'ok' | 'empty' | 'error' | 'offline'
export type Slot = 't1' | 't2'

export interface StacQuery {
  placeText: string
  place: Place | null
  bbox: Bounds | null
  datetime: [from: string, to: string]
  sensor: SensorChoice
  /** Sentinel-2 only. */
  cloudMax: number
  /** One scene, or a T1/T2 pair. */
  pair: boolean
}

export interface FlyTarget {
  bbox: Bounds
  /** Bumped on every request so the same place twice still flies. */
  seq: number
}

interface StacState {
  query: StacQuery
  status: StacStatus
  error: string | null
  results: StacItem[]
  selection: { t1: string | null; t2: string | null }
  hover: string | null
  flyTarget: FlyTarget | null
  /** Set once when the network comes back so the last query re-runs, once. */
  autoRetried: boolean

  setPlaceText: (text: string) => void
  setPlace: (place: Place | null) => void
  setBbox: (bbox: Bounds | null) => void
  setDatetime: (range: [string, string]) => void
  setSensor: (sensor: SensorChoice) => void
  setCloudMax: (cloudMax: number) => void
  setPair: (pair: boolean) => void
  search: () => Promise<void>
  select: (slot: Slot, id: string | null) => void
  /** Put the item in the first free slot (T1, then T2 when pairing). */
  pick: (id: string) => void
  setHover: (id: string | null) => void
  markOnline: () => void
  reset: () => void
}

const isoDate = (d: Date) => d.toISOString().slice(0, 10)

export function defaultWindow(now = new Date()): [string, string] {
  const from = new Date(now)
  from.setMonth(from.getMonth() - 6)
  return [isoDate(from), isoDate(now)]
}

export const DEFAULT_QUERY: StacQuery = {
  placeText: '',
  place: null,
  bbox: null,
  datetime: defaultWindow(),
  sensor: 'optical',
  cloudMax: 20,
  pair: false,
}

let inflight: AbortController | null = null
let flySeq = 0

export const useStacStore = create<StacState>((set, get) => ({
  query: DEFAULT_QUERY,
  status: 'idle',
  error: null,
  results: [],
  selection: { t1: null, t2: null },
  hover: null,
  flyTarget: null,
  autoRetried: false,

  setPlaceText: (placeText) => set((state) => ({ query: { ...state.query, placeText } })),
  setPlace: (place) => {
    flySeq += 1
    set((state) => ({
      query: { ...state.query, place, placeText: place?.name ?? state.query.placeText, bbox: place?.bbox ?? state.query.bbox },
      flyTarget: place ? { bbox: place.bbox, seq: flySeq } : state.flyTarget,
    }))
    if (place) void get().search()
  },
  setBbox: (bbox) => set((state) => ({ query: { ...state.query, bbox } })),
  setDatetime: (datetime) => {
    set((state) => ({ query: { ...state.query, datetime } }))
    if (get().query.bbox) void get().search()
  },
  setSensor: (sensor) => {
    set((state) => ({ query: { ...state.query, sensor }, selection: { t1: null, t2: null } }))
    if (get().query.bbox) void get().search()
  },
  // No search here: a range input fires per pixel; the picker debounces.
  setCloudMax: (cloudMax) => set((state) => ({ query: { ...state.query, cloudMax } })),
  setPair: (pair) => set((state) => ({ query: { ...state.query, pair }, selection: pair ? state.selection : { ...state.selection, t2: null } })),

  search: async () => {
    const { query } = get()
    if (!query.bbox) return
    inflight?.abort()
    const controller = new AbortController()
    inflight = controller
    set({ status: 'searching', error: null })
    const collections: CollectionId[] = collectionsFor(query.sensor)
    try {
      const results = await searchItems(
        { bbox: query.bbox, datetime: query.datetime, collections, cloudMax: query.cloudMax },
        controller.signal,
      )
      if (controller.signal.aborted) return
      const ids = new Set(results.map((r) => r.id))
      set((state) => ({
        results,
        status: results.length === 0 ? 'empty' : 'ok',
        autoRetried: false,
        // Keep a selection only if it is still on the shelf.
        selection: {
          t1: state.selection.t1 && ids.has(state.selection.t1) ? state.selection.t1 : null,
          t2: state.selection.t2 && ids.has(state.selection.t2) ? state.selection.t2 : null,
        },
      }))
    } catch (error) {
      if (controller.signal.aborted) return
      if (isOffline(error)) {
        // Existing results stay on the shelf and on the map: nothing already
        // shown is taken away by the network going.
        set({ status: 'offline', error: 'Search needs a network — the loaded scene is still here.' })
      } else if (error instanceof UpstreamError) {
        set({ status: 'error', error: error.message })
      } else {
        set({ status: 'error', error: error instanceof Error ? error.message : 'Scene search failed.' })
      }
    } finally {
      if (inflight === controller) inflight = null
    }
  },

  select: (slot, id) => set((state) => ({ selection: { ...state.selection, [slot]: id } })),
  pick: (id) => {
    const { selection, query } = get()
    if (selection.t1 === id) return set({ selection: { ...selection, t1: null } })
    if (selection.t2 === id) return set({ selection: { ...selection, t2: null } })
    if (!selection.t1) return set({ selection: { ...selection, t1: id } })
    if (query.pair) return set({ selection: { ...selection, t2: id } })
    set({ selection: { t1: id, t2: null } })
  },
  setHover: (hover) => {
    if (get().hover !== hover) set({ hover })
  },
  markOnline: () => {
    const state = get()
    if (state.status !== 'offline' || state.autoRetried) return
    set({ status: 'idle', error: null, autoRetried: true })
    void state.search()
  },
  reset: () => {
    inflight?.abort()
    set({ query: { ...DEFAULT_QUERY, datetime: defaultWindow() }, status: 'idle', error: null, results: [], selection: { t1: null, t2: null }, hover: null })
  },
}))

/** The chosen items, in slot order, with the reason they cannot be a pair (if any). */
export function selectedItems(state: Pick<StacState, 'results' | 'selection' | 'query'>): {
  t1: StacItem | null
  t2: StacItem | null
  issue: string | null
} {
  const t1 = state.results.find((r) => r.id === state.selection.t1) ?? null
  const t2 = state.query.pair ? (state.results.find((r) => r.id === state.selection.t2) ?? null) : null
  return { t1, t2, issue: t1 && t2 ? pairIssue(t1, t2) : null }
}
