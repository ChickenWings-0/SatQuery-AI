/**
 * Whether the visitor has asked for reduced motion.
 *
 * Almost everything in this app honours the preference in CSS, which is where
 * it belongs. This hook exists for the one case CSS cannot reach: React Flow
 * takes edge animation as a *prop* (`animated`), so the DAG has to know the
 * preference in JavaScript to offer a still alternative instead.
 *
 * It subscribes rather than reading once. The preference is a system setting
 * and can change while the tab is open — macOS and Windows both expose it in
 * accessibility panes people visit mid-session, and a run that keeps animating
 * after the user turned motion off is exactly the failure the setting exists to
 * prevent.
 */
import { useSyncExternalStore } from 'react'

const QUERY = '(prefers-reduced-motion: reduce)'

function subscribe(onChange: () => void): () => void {
  // `matchMedia` is absent in the test environment and in any non-DOM runtime,
  // where the honest answer is "no preference expressed" rather than a crash.
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return () => undefined
  }
  const list = window.matchMedia(QUERY)
  list.addEventListener('change', onChange)
  return () => list.removeEventListener('change', onChange)
}

function getSnapshot(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false
  return window.matchMedia(QUERY).matches
}

export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false)
}
