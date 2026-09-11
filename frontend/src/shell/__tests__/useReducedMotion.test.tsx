/**
 * @vitest-environment happy-dom
 */
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useReducedMotion } from '@/shell/useReducedMotion'

function Probe() {
  return <span data-testid="value">{String(useReducedMotion())}</span>
}

/** A `matchMedia` whose value can be flipped, like the OS setting can. */
function stubMatchMedia(initial: boolean) {
  const listeners = new Set<() => void>()
  const list = {
    matches: initial,
    media: '(prefers-reduced-motion: reduce)',
    addEventListener: (_: string, fn: () => void) => void listeners.add(fn),
    removeEventListener: (_: string, fn: () => void) => void listeners.delete(fn),
  }
  vi.stubGlobal('matchMedia', () => list)
  return {
    set(value: boolean) {
      list.matches = value
      for (const fn of listeners) fn()
    },
    get listenerCount() {
      return listeners.size
    },
  }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('useReducedMotion', () => {
  it('reports the preference', () => {
    stubMatchMedia(true)
    const { getByTestId } = render(<Probe />)
    expect(getByTestId('value').textContent).toBe('true')
  })

  it('reacts when the setting changes mid-session', () => {
    // The preference is an OS setting, not a page-load constant: someone can
    // turn it on in an accessibility pane while a run is streaming, and a DAG
    // that keeps animating afterwards is the exact failure it exists to stop.
    const media = stubMatchMedia(false)
    const { getByTestId } = render(<Probe />)
    expect(getByTestId('value').textContent).toBe('false')

    // `act` because the change arrives from outside React — a media-query
    // listener is exactly the external store `useSyncExternalStore` is for.
    act(() => media.set(true))
    expect(getByTestId('value').textContent).toBe('true')
  })

  it('unsubscribes on unmount', () => {
    const media = stubMatchMedia(false)
    const { unmount } = render(<Probe />)
    expect(media.listenerCount).toBe(1)
    unmount()
    expect(media.listenerCount).toBe(0)
  })

  it('answers "no preference" where matchMedia does not exist', () => {
    // Not a crash: a non-DOM runtime has no opinion, and the honest default is
    // the one that does not silently disable a signal.
    vi.stubGlobal('matchMedia', undefined)
    const { getByTestId } = render(<Probe />)
    expect(getByTestId('value').textContent).toBe('false')
  })
})
