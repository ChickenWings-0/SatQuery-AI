/**
 * @vitest-environment happy-dom
 *
 * Selection is a decision, not a hover. The panel follows the selected tab
 * and nothing else; the keyboard reaches every tab; "Try this" seeds the
 * question of the tab that is selected, whatever the pointer is over.
 */
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { Capabilities } from '@/pages/landing/Capabilities'
import { CAPABILITIES } from '@/pages/landing/capabilities'
import { useFocusStore } from '@/state/focus'
import { useLandingStore } from '@/state/landing'
import { useUiStore } from '@/state/ui'

beforeEach(() => {
  useLandingStore.getState().reset()
  vi.useFakeTimers()
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const tab = (id: string) => screen.getByRole('tab', { name: new RegExp(CAPABILITIES.find((c) => c.id === id)!.label) })
const panelFor = () => screen.getByRole('tabpanel').getAttribute('aria-labelledby')

describe('Capabilities', () => {
  it('does not change the panel on hover', () => {
    render(<Capabilities />)
    expect(panelFor()).toBe('cap-tab-change')
    fireEvent.pointerEnter(tab('grounding'))
    fireEvent.pointerOver(tab('grounding'))
    expect(panelFor()).toBe('cap-tab-change')
    expect(tab('change').getAttribute('aria-selected')).toBe('true')
  })

  it('switches on click and keeps aria-selected and the panel in step', () => {
    render(<Capabilities />)
    fireEvent.click(tab('crossmodal'))
    expect(tab('crossmodal').getAttribute('aria-selected')).toBe('true')
    expect(tab('change').getAttribute('aria-selected')).toBe('false')
    expect(panelFor()).toBe('cap-tab-crossmodal')
    expect(useLandingStore.getState().capabilityDirection).toBe(1)
  })

  it('walks the list with the arrow keys, Home and End', () => {
    render(<Capabilities />)
    fireEvent.keyDown(tab('change'), { key: 'ArrowDown' })
    expect(panelFor()).toBe('cap-tab-crossmodal')
    fireEvent.keyDown(tab('crossmodal'), { key: 'End' })
    expect(panelFor()).toBe('cap-tab-grounding')
    fireEvent.keyDown(tab('grounding'), { key: 'ArrowDown' })
    expect(panelFor()).toBe('cap-tab-change')
    fireEvent.keyDown(tab('change'), { key: 'ArrowUp' })
    expect(panelFor()).toBe('cap-tab-grounding')
    fireEvent.keyDown(tab('grounding'), { key: 'ArrowUp' })
    expect(panelFor()).toBe('cap-tab-crossmodal')
    // The slide follows the list order: up the list is -1.
    expect(useLandingStore.getState().capabilityDirection).toBe(-1)
    fireEvent.keyDown(tab('crossmodal'), { key: 'Home' })
    expect(panelFor()).toBe('cap-tab-change')
  })

  it('seeds the selected question, whatever the pointer is over', () => {
    render(<Capabilities />)
    fireEvent.click(tab('grounding'))
    fireEvent.pointerEnter(tab('change'))
    fireEvent.click(screen.getByRole('button', { name: /Try this in the console/ }))
    act(() => {
      vi.advanceTimersByTime(400)
    })
    expect(useUiStore.getState().section).toBe('explore')
    expect(useFocusStore.getState().draft).toBe(CAPABILITIES[2]!.question)
  })

  it('keeps the outgoing panel mounted for its exit, then drops it', () => {
    render(<Capabilities />)
    fireEvent.click(tab('grounding'))
    expect(document.querySelectorAll('[data-cap-panel]')).toHaveLength(2)
    expect(document.querySelector('[data-cap-panel="out"] [data-hud]')?.getAttribute('data-hud')).toBe('change')
    act(() => {
      vi.advanceTimersByTime(300)
    })
    expect(document.querySelectorAll('[data-cap-panel]')).toHaveLength(1)
  })

  it('renders every HUD with and without an image', () => {
    render(<Capabilities />)
    for (const cap of CAPABILITIES) {
      fireEvent.click(tab(cap.id))
      act(() => {
        vi.advanceTimersByTime(300)
      })
      const img = screen.getByAltText(cap.alt)
      expect(document.querySelector(`[data-hud="${cap.id}"]`)?.hasAttribute('data-has-image')).toBe(true)
      fireEvent.error(img)
      expect(screen.queryByAltText(cap.alt)).toBeNull()
      expect(document.querySelector(`[data-hud="${cap.id}"] .hud-scene`)).not.toBeNull()
    }
  })
})
