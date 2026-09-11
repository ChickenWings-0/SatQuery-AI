/**
 * What the viewer can say about the scene it is showing, and where it says it
 * from.
 *
 * Two sources, neither invented:
 *
 *   - the artifact label, which the renderer writes as
 *     "Image 1 (optical true colour, pre-change, Sentinel-2, 2019-04-12)" —
 *     the date and sensor are read out of it by pattern, tolerantly
 *   - the pre-flight's `InputManifest`s, which carry the sensor guess, ground
 *     sample distance, CRS and native bounds of each upload
 *
 * Everything here returns `null` for what it cannot find. The header prints a
 * placeholder for a null; it never prints a number the server did not send.
 */
import type { ArtifactRef, InputManifest, ValidateResponse } from '@/api/types'
import type { ViewGroup } from '@/evidence/views'

const DATE_RE = /\b(\d{4}-\d{2}-\d{2})\b/
const SENSOR_RE = /\b(Sentinel-[12][ABC]?|Landsat[- ]?\d|WorldView-?\d|PlanetScope|SkySat|MODIS|NAIP|Pléiades|SPOT[- ]?\d)\b/i

/** The ISO date in a renderer label, or null. */
export function dateOf(artifact: ArtifactRef | null | undefined): string | null {
  return artifact ? (DATE_RE.exec(artifact.label)?.[1] ?? null) : null
}

/** The sensor named in a renderer label, or null. */
export function sensorOf(artifact: ArtifactRef | null | undefined): string | null {
  return artifact ? (SENSOR_RE.exec(artifact.label)?.[1] ?? null) : null
}

/** A manifest's sensor guess, shortened to its family name. */
function sensorFamily(manifest: InputManifest | undefined): string | null {
  const guess = manifest?.sensor_guess
  if (!guess) return null
  return SENSOR_RE.exec(guess)?.[1] ?? guess.split('(')[0]!.trim()
}

export interface SceneMeta {
  sensor: string | null
  /** Ground sample distance in metres. */
  gsdM: number | null
  /** Scene width on the ground, in kilometres — width × GSD. */
  widthKm: number | null
  crs: string | null
  /** Centre of the native bounds, as "x, y" in the CRS's units. */
  centre: string | null
  t1: string | null
  t2: string | null
}

function manifestFor(inputs: InputManifest[], role: 'pre' | 'post'): InputManifest | undefined {
  return inputs.find((input) => input.role === role) ?? (role === 'pre' ? inputs[0] : inputs[1])
}

/** A date from a manifest's acquisition time, as YYYY-MM-DD. */
function acquired(manifest: InputManifest | undefined): string | null {
  const time = manifest?.acquisition_time
  return time ? (DATE_RE.exec(time)?.[1] ?? null) : null
}

/**
 * Everything the scene header and the viewer chrome can say about the scene.
 *
 * The artifact labels win for dates and sensor, because they are the strings
 * the VLM was shown; the manifests fill in the rest.
 */
export function sceneMeta(
  validation: ValidateResponse | null,
  group: ViewGroup | null | undefined,
): SceneMeta {
  const inputs = validation?.inputs ?? []
  const pre = manifestFor(inputs, 'pre')
  const post = manifestFor(inputs, 'post')
  const first = inputs[0]

  const gsdM = typeof first?.gsd_m === 'number' && first.gsd_m > 0 ? first.gsd_m : null
  const widthKm = gsdM && first && first.width > 0 ? (first.width * gsdM) / 1000 : null

  const bounds = first?.bounds_native
  const centre =
    bounds && bounds.length === 4
      ? `${Math.round((bounds[0]! + bounds[2]!) / 2).toLocaleString('en-US')}, ${Math.round((bounds[1]! + bounds[3]!) / 2).toLocaleString('en-US')}`
      : null

  return {
    sensor: sensorOf(group?.pre ?? group?.single ?? group?.post) ?? sensorFamily(first),
    gsdM,
    widthKm,
    crs: first?.crs ?? null,
    centre,
    t1: dateOf(group?.pre) ?? dateOf(group?.single) ?? acquired(pre),
    t2: dateOf(group?.post) ?? acquired(post),
  }
}

/** "Sentinel-2 · 10 m" for the evidence tray's sublabel, or as much as is known. */
export function resolutionLabel(meta: SceneMeta): string | null {
  const parts: string[] = []
  if (meta.sensor) parts.push(meta.sensor)
  if (meta.gsdM) parts.push(`${meta.gsdM < 1 ? meta.gsdM.toFixed(2) : Math.round(meta.gsdM)} m`)
  return parts.length ? parts.join(' · ') : null
}

/**
 * A round distance for the scale bar that fits inside `maxPx` screen pixels,
 * given how many pixels one kilometre currently spans.
 *
 * Returns `null` when the scene's scale is unknown: a scale bar with no basis
 * is a measurement the product made up, and this product does not do that.
 */
export function scaleBar(
  pxPerKm: number | null,
  maxPx: number,
): { km: number; px: number; label: string } | null {
  if (!pxPerKm || !Number.isFinite(pxPerKm) || pxPerKm <= 0) return null
  const steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
  let pick: number | null = null
  for (const km of steps) {
    if (km * pxPerKm <= maxPx) pick = km
  }
  if (pick === null) return null
  const label = pick < 1 ? `${Math.round(pick * 1000)} m` : `${pick} km`
  return { km: pick, px: pick * pxPerKm, label }
}
