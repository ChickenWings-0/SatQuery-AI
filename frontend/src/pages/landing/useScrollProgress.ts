/**
 * The hero's exit, as a number.
 *
 * Writes `--p` (0 at the top of the page, 1 once 70 % of the hero has
 * scrolled away) onto the element and into the landing store. CSS reads the
 * property on browsers without `animation-timeline` (Firefox, as of this
 * writing); the globe reads the store on every browser, because a shader
 * cannot read a CSS scroll timeline. rAF-throttled: one write per frame at
 * most, nothing on frames where the page did not move.
 */
import { useEffect, type RefObject } from 'react'

import { useLandingStore } from '@/state/landing'

export function useScrollProgress(ref: RefObject<HTMLElement | null>, exitAt = 0.7): void {
  useEffect(() => {
    const el = ref.current
    if (!el || typeof window === 'undefined') return
    let raf = 0
    let last = -1
    const measure = () => {
      raf = 0
      const rect = el.getBoundingClientRect()
      const p = Math.min(1, Math.max(0, -rect.top / (rect.height * exitAt)))
      if (Math.abs(p - last) < 0.002) return
      last = p
      el.style.setProperty('--p', p.toFixed(3))
      useLandingStore.getState().setScroll(p)
    }
    const schedule = () => {
      if (!raf) raf = window.requestAnimationFrame(measure)
    }
    measure()
    window.addEventListener('scroll', schedule, { passive: true })
    window.addEventListener('resize', schedule)
    return () => {
      window.removeEventListener('scroll', schedule)
      window.removeEventListener('resize', schedule)
      if (raf) window.cancelAnimationFrame(raf)
      el.style.removeProperty('--p')
      useLandingStore.getState().setScroll(0)
    }
  }, [ref, exitAt])
}

/**
 * The pointer as two fractions of the element, written as `--mx` / `--my`
 * and (for the hero) into the store for the globe's rim light. Only on a
 * fine pointer: touch gets the scroll version of the page.
 */
export function useCursorLight(ref: RefObject<HTMLElement | null>, toStore = false): void {
  useEffect(() => {
    const el = ref.current
    if (!el || typeof window === 'undefined') return
    if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return
    let raf = 0
    let mx = 0.5
    let my = 0.5
    const write = () => {
      raf = 0
      el.style.setProperty('--mx', mx.toFixed(3))
      el.style.setProperty('--my', my.toFixed(3))
      if (toStore) useLandingStore.getState().setPointer(mx, my)
    }
    const move = (event: PointerEvent) => {
      const rect = el.getBoundingClientRect()
      mx = (event.clientX - rect.left) / rect.width
      my = (event.clientY - rect.top) / rect.height
      if (!raf) raf = window.requestAnimationFrame(write)
    }
    el.addEventListener('pointermove', move, { passive: true })
    return () => {
      el.removeEventListener('pointermove', move)
      if (raf) window.cancelAnimationFrame(raf)
    }
  }, [ref, toStore])
}
