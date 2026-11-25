/**
 * A pinned section whose scroll distance is divided into steps.
 *
 * While the section's sticky stage is pinned, the page's scroll position is the
 * control: progress through the section's extra height maps to a step index,
 * and `--cap-progress` (0 → 1) is written on the section so each step's track
 * can fill from CSS. rAF-throttled, one write per frame at most.
 *
 * Active only where the stage actually pins (the `desk` breakpoint, mirrored
 * here as a media query); elsewhere the section is an ordinary tab list and
 * this hook does nothing. `scrollToStep` is how a click or a key keeps the
 * scroll position and the selection in agreement.
 */
import { useCallback, useEffect, useState, type RefObject } from 'react'

const PINNED_QUERY = '(min-width: 1280px) and (min-height: 500px)'

function progressOf(section: HTMLElement): number {
  const rect = section.getBoundingClientRect()
  const travel = rect.height - window.innerHeight
  if (travel <= 0) return 0
  return Math.min(1, Math.max(0, -rect.top / travel))
}

export function usePinnedSteps(
  ref: RefObject<HTMLElement | null>,
  steps: number,
  onStep: (index: number) => void,
): { pinned: boolean; scrollToStep: (index: number) => void } {
  const [pinned, setPinned] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(PINNED_QUERY).matches,
  )

  useEffect(() => {
    if (typeof window === 'undefined') return
    const query = window.matchMedia(PINNED_QUERY)
    const update = () => setPinned(query.matches)
    query.addEventListener('change', update)
    return () => query.removeEventListener('change', update)
  }, [])

  useEffect(() => {
    const section = ref.current
    if (!pinned || !section) return
    let raf = 0
    let lastStep = -1
    let lastProgress = -1
    const measure = () => {
      raf = 0
      const progress = progressOf(section)
      if (Math.abs(progress - lastProgress) >= 0.001) {
        lastProgress = progress
        section.style.setProperty('--cap-progress', progress.toFixed(4))
      }
      const step = Math.min(steps - 1, Math.floor(progress * steps))
      if (step !== lastStep) {
        lastStep = step
        onStep(step)
      }
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
      section.style.removeProperty('--cap-progress')
    }
  }, [ref, pinned, steps, onStep])

  const scrollToStep = useCallback(
    (index: number) => {
      const section = ref.current
      if (!pinned || !section) return
      const travel = section.offsetHeight - window.innerHeight
      const top = section.getBoundingClientRect().top + window.scrollY
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      // The middle of the step, so a small scroll either way stays on it.
      window.scrollTo({
        top: top + travel * ((index + 0.5) / steps),
        behavior: reduced ? 'auto' : 'smooth',
      })
    },
    [ref, pinned, steps],
  )

  return { pinned, scrollToStep }
}
