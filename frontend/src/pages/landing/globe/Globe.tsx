/**
 * The R3F canvas: the chunk boundary for `three`. Mounted only when WebGL is
 * available and motion is not reduced (`Hero` decides), rendering only while
 * on screen (`frameloop` follows `globeVisible`), at ≤ 1.75 dpr.
 */
import { Canvas } from '@react-three/fiber'
import { Suspense, useEffect, useRef, useState } from 'react'

import { Controls } from '@/pages/landing/globe/Controls'
import { Earth } from '@/pages/landing/globe/Earth'
import { ORBITS } from '@/pages/landing/globe/orbits'
import { Satellite } from '@/pages/landing/globe/Satellite'
import { MinusIcon, PlusIcon } from '@/components/ui/icons'
import { useLandingStore } from '@/state/landing'
import { useThemeStore } from '@/state/theme'

export default function Globe({ onReady }: { onReady?: () => void }) {
  const dark = useThemeStore((state) => state.resolved === 'dark')
  const visible = useLandingStore((state) => state.globeVisible)
  const dragOver = useLandingStore((state) => state.dragOver)
  const zoomRef = useRef<((delta: number) => void) | null>(null)
  const [rim, setRim] = useState(0)
  const [length, setLength] = useState(0)

  // "Coming online": rim and trails ramp over 900 ms once the canvas exists.
  useEffect(() => {
    onReady?.()
    const start = performance.now()
    let raf = 0
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / 900)
      const eased = 1 - (1 - t) ** 3
      setRim(0.55 * eased)
      setLength(eased)
      if (t < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [onReady])

  return (
    <div className="relative size-full">
      <Canvas
        dpr={[1, 1.75]}
        frameloop={visible ? 'always' : 'never'}
        gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
        camera={{ position: [0, 0.3, 3.9], fov: 34 }}
        onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
        style={{ touchAction: 'pan-y' }}
        aria-hidden
      >
        <Suspense fallback={null}>
          <Earth dark={dark} rim={rim} />
          {ORBITS.map((orbit) => (
            <Satellite key={orbit.name} orbit={orbit} dark={dark} bright={dragOver ? 1 : 0} length={dragOver ? 1 : length} />
          ))}
          <Controls zoomRef={zoomRef} />
        </Suspense>
      </Canvas>
      <div className="glass absolute right-[14%] bottom-[10%] flex divide-x divide-line-soft p-0" role="group" aria-label="Globe zoom">
        <button type="button" aria-label="Zoom in" onClick={() => zoomRef.current?.(-0.2)} className="grid size-7 place-items-center text-text-lo hover:text-text-hi">
          <PlusIcon size={13} />
        </button>
        <button type="button" aria-label="Zoom out" onClick={() => zoomRef.current?.(0.2)} className="grid size-7 place-items-center text-text-lo hover:text-text-hi">
          <MinusIcon size={13} />
        </button>
      </div>
    </div>
  )
}
