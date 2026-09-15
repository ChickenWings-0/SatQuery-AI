/**
 * @vitest-environment happy-dom
 *
 * The scroll-driven layer's fallback: on a browser without
 * `animation-timeline` (Firefox), `--p` and the store carry the hero's exit.
 */
import { act, cleanup, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useScrollProgress } from '@/pages/landing/useScrollProgress'
import { useLandingStore } from '@/state/landing'

function Probe({ top }: { top: number }) {
  const ref = useRef<HTMLDivElement>(null)
  useScrollProgress(ref)
  return <div ref={ref} data-testid="hero" style={{ height: 1000 }} data-top={top} />
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useScrollProgress', () => {
  it('writes --p and the store from the hero rect, once per frame', () => {
    let top = 0
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
      () => ({ top, height: 1000, bottom: top + 1000, left: 0, right: 0, width: 0, x: 0, y: top, toJSON: () => '' }) as DOMRect,
    )
    let frame: FrameRequestCallback | null = null
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      frame = cb
      return 1
    })
    const { getByTestId, unmount } = render(<Probe top={0} />)
    const hero = getByTestId('hero')
    expect(hero.style.getPropertyValue('--p')).toBe('0.000')

    top = -350 // half of the 70 % exit range
    act(() => {
      window.dispatchEvent(new Event('scroll'))
      window.dispatchEvent(new Event('scroll'))
      frame!(0)
    })
    expect(hero.style.getPropertyValue('--p')).toBe('0.500')
    expect(useLandingStore.getState().scroll).toBe(0.5)

    top = -2000
    act(() => {
      window.dispatchEvent(new Event('scroll'))
      frame!(0)
    })
    expect(hero.style.getPropertyValue('--p')).toBe('1.000')

    unmount()
    expect(useLandingStore.getState().scroll).toBe(0)
  })
})
