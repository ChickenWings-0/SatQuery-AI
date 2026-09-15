/**
 * What the manifest says about where the scene is.
 *
 * `bounds_wgs84` is `[west, south, east, north]` when the raster carried a
 * CRS; null otherwise, and the Maps page then falls back to pixel space and
 * says so. Nothing here invents a location.
 */
import type { InputManifest, ValidateResponse } from '@/api/types'

export type Bounds = [west: number, south: number, east: number, north: number]

export interface Georef {
  bounds: Bounds
  crs: string
  gsdM: number | null
  center: [lon: number, lat: number]
}

export function georefOf(manifest: InputManifest | null | undefined): Georef | null {
  if (!manifest?.crs) return null
  const b = manifest.bounds_wgs84
  if (!b || b.length !== 4 || !b.every((n) => Number.isFinite(n))) return null
  const bounds: Bounds = [b[0]!, b[1]!, b[2]!, b[3]!]
  return {
    bounds,
    crs: manifest.crs,
    gsdM: manifest.gsd_m ?? null,
    center: [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2],
  }
}

export function sceneGeoref(validation: ValidateResponse | null): Georef | null {
  if (!validation) return null
  for (const manifest of validation.inputs) {
    const g = georefOf(manifest)
    if (g) return g
  }
  return null
}

/** The four corners maplibre wants for an image source: TL, TR, BR, BL. */
export function cornersOf(bounds: Bounds): [[number, number], [number, number], [number, number], [number, number]] {
  const [w, s, e, n] = bounds
  return [
    [w, n],
    [e, n],
    [e, s],
    [w, s],
  ]
}

const pad = (n: number) => String(n).padStart(2, '0')

/** `12°58′41″N 77°35′22″E` */
export function formatDms(lon: number, lat: number): string {
  const part = (value: number, pos: string, neg: string) => {
    const abs = Math.abs(value)
    const d = Math.floor(abs)
    const mF = (abs - d) * 60
    const m = Math.floor(mF)
    const s = Math.round((mF - m) * 60)
    return `${d}°${pad(m)}′${pad(s)}″${value >= 0 ? pos : neg}`
  }
  return `${part(lat, 'N', 'S')} ${part(lon, 'E', 'W')}`
}

/** Parse `12.978, 77.589` or `12°58′N 77°35′E`; returns `[lon, lat]`. */
export function parseCoordinate(text: string): [number, number] | null {
  const dec = /^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/.exec(text)
  if (dec) {
    const lat = Number(dec[1])
    const lon = Number(dec[2])
    if (Math.abs(lat) <= 90 && Math.abs(lon) <= 180) return [lon, lat]
    return null
  }
  const dms = /(\d+)°\s*(\d+)?′?\s*(\d+)?″?\s*([NS])[ ,]+(\d+)°\s*(\d+)?′?\s*(\d+)?″?\s*([EW])/i.exec(text)
  if (!dms) return null
  const toDec = (d: string, m?: string, s?: string) => Number(d) + Number(m ?? 0) / 60 + Number(s ?? 0) / 3600
  const lat = toDec(dms[1]!, dms[2], dms[3]) * (dms[4]!.toUpperCase() === 'S' ? -1 : 1)
  const lon = toDec(dms[5]!, dms[6], dms[7]) * (dms[8]!.toUpperCase() === 'W' ? -1 : 1)
  return [lon, lat]
}

/** Metres per pixel of a Web Mercator map at a zoom and latitude. */
export function metersPerPixel(zoom: number, lat: number): number {
  return (156543.03392 * Math.cos((lat * Math.PI) / 180)) / 2 ** zoom
}
