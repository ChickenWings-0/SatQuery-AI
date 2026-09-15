/**
 * Once-only scroll reveals. An element marked `data-reveal` is hidden only
 * after JS has marked it `pending` (so a failed script never hides content),
 * and animates in the first time 25 % of it is visible. Children marked
 * `data-reveal-child` stagger at 40 ms, capped at 240 ms.
 */
import { useEffect, type RefObject } from 'react'

const STAGGER_MS = 40
const CAP_MS = 240

export function useReveal(root: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const el = root.current
    if (!el) return
    const targets = [...el.querySelectorAll<HTMLElement>('[data-reveal]')]
    if (typeof IntersectionObserver === 'undefined' || targets.length === 0) return
    for (const t of targets) t.dataset['reveal'] = 'pending'
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue
          const target = entry.target as HTMLElement
          target.dataset['reveal'] = 'in'
          target.querySelectorAll<HTMLElement>('[data-reveal-child]').forEach((child, i) => {
            child.style.setProperty('--reveal-delay', `${Math.min(CAP_MS, i * STAGGER_MS)}ms`)
            child.dataset['reveal'] = 'in'
          })
          io.unobserve(target)
        }
      },
      { threshold: 0.05, rootMargin: '0px 0px -8% 0px' },
    )
    for (const t of targets) io.observe(t)
    // Belt and braces: whatever the observer has not reached in four seconds
    // (a print, a full-page capture, an odd embed) is revealed anyway.
    const fallback = window.setTimeout(() => {
      for (const t of targets) if (t.dataset['reveal'] === 'pending') t.dataset['reveal'] = 'in'
    }, 4_000)
    return () => {
      io.disconnect()
      window.clearTimeout(fallback)
    }
  }, [root])
}
