import { useFrame, useLoader } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import { Color, DoubleSide, ShaderMaterial, TextureLoader, Vector3, type Mesh } from 'three'

import frag from '@/pages/landing/globe/earth.frag.glsl?raw'
import { tokens } from '@/pages/landing/globe/tokens'
import vert from '@/pages/landing/globe/earth.vert.glsl?raw'
import { useLandingStore } from '@/state/landing'

const LAND_MASK = '/samples/globe/land.png'

function rgb([r, g, b]: readonly [number, number, number]): Color {
  return new Color(r, g, b)
}

/**
 * The sphere: one shader, no lights. `dark` flips the uniforms between the
 * two themes; `rim` is ramped by the parent during the entry sequence.
 *
 * Two things answer the reader. The scroll (`store.scroll`, the hero's exit)
 * adds to the rotation — a full exit turns the globe ~52° on top of its
 * idle drift — and brightens the limb, so the last thing the eye sees of it
 * is the rim. The pointer (`store.pointer`) swings the terminator ~20°
 * toward the cursor, damped, so the globe reads as lit by the room the
 * reader is in. Both are `getState()` reads inside `useFrame`; a
 * subscription here would re-render per frame.
 */
export function Earth({ dark, rim }: { dark: boolean; rim: number }) {
  const mesh = useRef<Mesh>(null)
  const texture = useLoader(TextureLoader, LAND_MASK)
  const sun = useMemo(() => new Vector3(1, 0.3, 0.6), [])
  const clockSun = useMemo(() => new Vector3(), [])
  const cursorSun = useMemo(() => new Vector3(), [])
  const lastScroll = useRef(0)

  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: vert,
        fragmentShader: frag,
        side: DoubleSide,
        transparent: false,
        uniforms: {
          uGround: { value: rgb(tokens.bgMain) },
          uGrid: { value: rgb(tokens.sand) },
          uLand: { value: rgb(tokens.terracotta) },
          uRim: { value: rgb(tokens.sand) },
          uGridAlpha: { value: 0.16 },
          uRimStrength: { value: 0 },
          uTheme: { value: 0 },
          uSun: { value: sun.clone() },
          uLandMask: { value: texture },
          uHasMask: { value: 1 },
          uTime: { value: 0 },
        },
      }),
    // Built once; theme and rim are written into the uniforms below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  useFrame((_, dt) => {
    const u = material.uniforms
    const { scroll, pointer } = useLandingStore.getState()
    u['uTime']!.value += Math.min(dt, 1 / 30)
    // The terminator: one revolution in 180 s, leaning toward the pointer.
    const t = (u['uTime']!.value / 180) * Math.PI * 2
    clockSun.set(Math.cos(t), 0.3, Math.sin(t))
    cursorSun.set(pointer[0] - 0.5, -(pointer[1] - 0.5), 0.6).normalize()
    const target = clockSun.lerp(cursorSun, 0.35).normalize()
    ;(u['uSun']!.value as Vector3).lerp(target, 0.05).normalize()
    // The limb brightens as the globe is left behind.
    const rimTarget = rim * (1 + 0.55 * scroll)
    u['uRimStrength']!.value += (rimTarget - (u['uRimStrength']!.value as number)) * 0.08
    const theme = dark ? 0 : 1
    if (u['uTheme']!.value !== theme) {
      u['uTheme']!.value = theme
      ;(u['uGround']!.value as Color).set(
        dark ? rgb(tokens.bgMain).multiplyScalar(0.85) : rgb(tokens.paper).multiplyScalar(0.94),
      )
      ;(u['uGrid']!.value as Color).set(dark ? rgb(tokens.sand) : rgb(tokens.paperAccent))
      ;(u['uLand']!.value as Color).set(dark ? rgb(tokens.terracotta) : rgb(tokens.paperAccent))
      ;(u['uRim']!.value as Color).set(dark ? rgb(tokens.sand) : rgb(tokens.paperAccent))
      u['uGridAlpha']!.value = dark ? 0.16 : 0.11
    }
    if (mesh.current) {
      const scrollDelta = scroll - lastScroll.current
      lastScroll.current = scroll
      mesh.current.rotation.y += dt * 0.02 + scrollDelta * 0.9
    }
  })

  return (
    <mesh ref={mesh} material={material}>
      <icosahedronGeometry args={[1, 5]} />
    </mesh>
  )
}
