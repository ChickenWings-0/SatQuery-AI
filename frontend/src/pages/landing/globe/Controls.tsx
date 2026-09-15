import { OrbitControls } from '@react-three/drei/core/OrbitControls'
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useRef } from 'react'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

import { useLandingStore } from '@/state/landing'

/**
 * Gentle rotation, no pan, no wheel zoom (the wheel must scroll the page),
 * pinch and `+`/`−` within a narrow distance band, auto-rotate that pauses
 * while dragging. Rotation on touch engages only on horizontal intent, so a
 * one-finger vertical swipe over the globe still scrolls.
 *
 * As the hero scrolls out the camera eases up (y 0.3 → 0.55), so the globe
 * rolls slightly under the page as it leaves. Written through the same path
 * as the zoom — set the position, `update()` — so damping never fights it.
 */
export function Controls({ zoomRef }: { zoomRef: React.MutableRefObject<((delta: number) => void) | null> }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera, gl } = useThree()

  useEffect(() => {
    zoomRef.current = (delta) => {
      const c = controls.current
      if (!c) return
      const dist = camera.position.length()
      const next = Math.min(4.6, Math.max(3.4, dist + delta))
      camera.position.setLength(next)
      c.update()
    }
    return () => {
      zoomRef.current = null
    }
  }, [camera, zoomRef])

  // Horizontal-intent gate for one-finger touch.
  useEffect(() => {
    const el = gl.domElement
    let startX = 0
    let startY = 0
    let decided = false
    const down = (event: PointerEvent) => {
      if (event.pointerType !== 'touch') return
      startX = event.clientX
      startY = event.clientY
      decided = false
      if (controls.current) controls.current.enableRotate = false
    }
    const move = (event: PointerEvent) => {
      if (event.pointerType !== 'touch' || decided) return
      const dx = Math.abs(event.clientX - startX)
      const dy = Math.abs(event.clientY - startY)
      if (dx < 8 && dy < 8) return
      decided = true
      if (controls.current) controls.current.enableRotate = dx > dy
    }
    const up = () => {
      if (controls.current) controls.current.enableRotate = true
    }
    el.addEventListener('pointerdown', down)
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    el.addEventListener('pointercancel', up)
    return () => {
      el.removeEventListener('pointerdown', down)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
      el.removeEventListener('pointercancel', up)
    }
  }, [gl])

  useFrame((frame) => {
    const c = controls.current
    if (!c) return
    const { scroll } = useLandingStore.getState()
    const cam = frame.camera
    const targetY = 0.3 + 0.25 * scroll
    const dy = targetY - cam.position.y
    if (Math.abs(dy) < 0.0005) return
    const dist = cam.position.length()
    cam.position.y += dy * 0.06
    cam.position.setLength(dist)
    c.update()
  })

  return (
    <OrbitControls
      ref={controls}
      enablePan={false}
      enableZoom={false}
      rotateSpeed={0.35}
      minPolarAngle={Math.PI / 2 - 0.55}
      maxPolarAngle={Math.PI / 2 + 0.35}
      autoRotate
      autoRotateSpeed={0.28}
      enableDamping
      dampingFactor={0.06}
      makeDefault
    />
  )
}
