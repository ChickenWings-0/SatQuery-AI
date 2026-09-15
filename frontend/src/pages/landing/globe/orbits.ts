/**
 * The three sensors the product listens to, as orbits around a unit sphere.
 * Inclinations are the real ones (sun-synchronous, ~98°); altitudes are
 * scaled to the sphere; periods are scaled so a satellite crosses the visible
 * face in a few seconds rather than fifty minutes.
 */
import { Vector3 } from 'three'

export interface Orbit {
  name: string
  r: number
  inclDeg: number
  raanDeg: number
  periodS: number
  phase: number
}

export const ORBITS: readonly Orbit[] = [
  { name: 'Sentinel-2', r: 1.123, inclDeg: 98.6, raanDeg: 20, periodS: 44, phase: 0 },
  { name: 'Sentinel-1', r: 1.109, inclDeg: 98.2, raanDeg: 110, periodS: 40, phase: 2.1 },
  { name: 'VHR', r: 1.078, inclDeg: 97.4, raanDeg: 200, periodS: 36, phase: 4.3 },
]

const X_AXIS = new Vector3(1, 0, 0)
const Y_AXIS = new Vector3(0, 1, 0)

export function positionAt(orbit: Orbit, t: number, out: Vector3): Vector3 {
  const a = orbit.phase + (2 * Math.PI * t) / orbit.periodS
  out.set(Math.cos(a) * orbit.r, 0, Math.sin(a) * orbit.r)
  out.applyAxisAngle(X_AXIS, (orbit.inclDeg * Math.PI) / 180)
  out.applyAxisAngle(Y_AXIS, (orbit.raanDeg * Math.PI) / 180)
  return out
}
