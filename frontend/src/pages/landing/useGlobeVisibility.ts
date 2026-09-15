import { useEffect, type RefObject } from 'react'

import { useLandingStore } from '@/state/landing'

/** Flip the globe's frameloop off the moment the hero leaves the viewport. */
export function useGlobeVisibility(ref: RefObject<HTMLElement | null>): void {
  const setVisible = useLandingStore((state) => state.setGlobeVisible)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') {
      setVisible(true)
      return
    }
    const io = new IntersectionObserver(
      ([entry]) => setVisible(Boolean(entry?.isIntersecting) && document.visibilityState === 'visible'),
      { threshold: 0.1 },
    )
    io.observe(el)
    const onVis = () => setVisible(document.visibilityState === 'visible')
    document.addEventListener('visibilitychange', onVis)
    return () => {
      io.disconnect()
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [ref, setVisible])
}
