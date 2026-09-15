import { Line } from '@react-three/drei/core/Line'
import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import { Color, Vector3, type Group } from 'three'
import type { Line2 } from 'three-stdlib'

import { positionAt, type Orbit } from '@/pages/landing/globe/orbits'
import { tokens } from '@/pages/landing/globe/tokens'
import { useLandingStore } from '@/state/landing'

/** The trail at rest; the buffer holds 1.6× so it can stretch with the scroll. */
const REST = 96
const TRAIL = 154
const ORIGIN = new Vector3(0, 0, 0)

/**
 * A low-poly body with two panels, orbiting per `orbits.ts`, leaving a trail
 * of the last 96 positions that fades to nothing. No allocations per frame:
 * the vector, the position array and the colour array are reused.
 */
export function Satellite({
  orbit,
  dark,
  bright,
  length,
}: {
  orbit: Orbit
  dark: boolean
  /** 0–1: the drop-over acknowledgment brightens the head. */
  bright: number
  /** 0–1: the entry sequence grows the trail. */
  length: number
}) {
  const group = useRef<Group>(null)
  const line = useRef<Line2>(null)
  // Deterministic start: the phase decides where each satellite begins.
  const clock = useRef((orbit.phase / (2 * Math.PI)) * orbit.periodS)
  const tick = useRef(0)
  const scratch = useMemo(() => new Vector3(), [])
  // The initial trail shape, pure from the orbit; the per-frame buffer copies
  // it on the first frame and is never read during render.
  const seed = useMemo(() => {
    const v = new Vector3()
    const t0 = (orbit.phase / (2 * Math.PI)) * orbit.periodS
    return Array.from({ length: TRAIL }, (_, i) => {
      positionAt(orbit, t0 - (i * orbit.periodS) / 400, v)
      return [v.x, v.y, v.z] as [number, number, number]
    })
  }, [orbit])
  const positions = useRef<[number, number, number][] | null>(null)
  const colors = useMemo(() => {
    const base = dark ? tokens.sand : tokens.paperAccent
    const out: [number, number, number][] = []
    for (let i = 0; i < TRAIL; i++) {
      const a = 1 - i / TRAIL
      out.push([base[0] * a, base[1] * a, base[2] * a])
    }
    return out
  }, [dark])

  const bodyColor = useMemo(() => new Color(...(dark ? tokens.sand : tokens.paperAccent)), [dark])
  const panelColor = useMemo(() => new Color(...(dark ? tokens.terracotta : tokens.paperText)), [dark])

  useFrame((_, dt) => {
    // A hidden tab must not teleport the satellite.
    clock.current += Math.min(dt, 1 / 30)
    positionAt(orbit, clock.current, scratch)
    if (group.current) {
      group.current.position.copy(scratch)
      group.current.lookAt(ORIGIN)
    }
    // Write the trail every other frame.
    tick.current = (tick.current + 1) % 2
    if (tick.current === 0 && line.current) {
      if (!positions.current) positions.current = seed.map((p) => [...p] as [number, number, number])
      const buf = positions.current
      for (let i = TRAIL - 1; i > 0; i--) {
        const prev = buf[i - 1]!
        buf[i]![0] = prev[0]
        buf[i]![1] = prev[1]
        buf[i]![2] = prev[2]
      }
      buf[0]![0] = scratch.x
      buf[0]![1] = scratch.y
      buf[0]![2] = scratch.z
      // Orbits stretch as the camera pulls away (the hero's exit).
      const stretch = 1 + 0.6 * useLandingStore.getState().scroll
      const visible = Math.min(TRAIL, Math.max(2, Math.round(REST * length * stretch)))
      const geometry = line.current.geometry
      geometry.setPositions(buf.slice(0, visible).flat())
      const head = 0.55 + 0.35 * bright
      geometry.setColors(
        colors
          .slice(0, visible)
          .map(([r, g, b], i) => {
            const a = head * (1 - i / visible)
            return [r * a, g * a, b * a]
          })
          .flat(),
      )
    }
  })

  return (
    <>
      <group ref={group}>
        <mesh>
          <boxGeometry args={[0.018, 0.012, 0.026]} />
          <meshBasicMaterial color={bodyColor} />
        </mesh>
        <mesh position={[0.042, 0, 0]}>
          <planeGeometry args={[0.06, 0.016]} />
          <meshBasicMaterial color={panelColor} transparent opacity={0.55} side={2} />
        </mesh>
        <mesh position={[-0.042, 0, 0]}>
          <planeGeometry args={[0.06, 0.016]} />
          <meshBasicMaterial color={panelColor} transparent opacity={0.55} side={2} />
        </mesh>
      </group>
      <Line
        ref={line}
        points={seed}
        vertexColors={colors}
        lineWidth={1.2}
        transparent
        opacity={0.9}
        depthTest
      />
    </>
  )
}
