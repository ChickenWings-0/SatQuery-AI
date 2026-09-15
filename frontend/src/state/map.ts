/**
 * The Maps page's HUD state.
 *
 * Two modes, declared in the HUD. *Scene* is always available: the current
 * run's rendered views placed by the manifest's WGS84 bounds. *Basemap* is an
 * online raster tile source behind the scene — off by default, remembered
 * once switched on, and auto-switched off when the tiles cannot be reached,
 * because this console is judged in halls without a route to the internet.
 */
import { create } from 'zustand'

import { safeStorage } from '@/shell/storage'

export type Basemap = 'none' | 'satellite'

const BASEMAP_KEY = 'satquery.maps.basemap'

export interface MapCursor {
  lon: number
  lat: number
}

interface MapState {
  layerKey: string | null
  compareKey: string | null
  split: number
  basemap: Basemap
  opacity: number
  zoom: number
  cursor: MapCursor | null
  locatorOpen: boolean

  setLayer: (key: string | null) => void
  setCompare: (key: string | null) => void
  setSplit: (split: number) => void
  nudgeSplit: (delta: number) => void
  setBasemap: (basemap: Basemap, options?: { remember?: boolean }) => void
  setOpacity: (opacity: number) => void
  setZoom: (zoom: number) => void
  setCursor: (cursor: MapCursor | null) => void
  setLocatorOpen: (open: boolean) => void
}

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

export const useMapStore = create<MapState>((set) => ({
  layerKey: null,
  compareKey: null,
  split: 50,
  basemap: safeStorage.getItem(BASEMAP_KEY) === 'satellite' ? 'satellite' : 'none',
  opacity: 1,
  zoom: 0,
  cursor: null,
  locatorOpen: false,

  setLayer: (layerKey) => set({ layerKey }),
  setCompare: (compareKey) => set({ compareKey, split: 50 }),
  setSplit: (split) => set({ split: clamp(split, 0, 100) }),
  nudgeSplit: (delta) => set((state) => ({ split: clamp(state.split + delta, 0, 100) })),
  setBasemap: (basemap, options) => {
    set({ basemap })
    if (options?.remember !== false) safeStorage.setItem(BASEMAP_KEY, basemap)
  },
  setOpacity: (opacity) => set({ opacity: clamp(opacity, 0, 1) }),
  setZoom: (zoom) => set({ zoom }),
  setCursor: (cursor) => set({ cursor }),
  setLocatorOpen: (locatorOpen) => set({ locatorOpen }),
}))
