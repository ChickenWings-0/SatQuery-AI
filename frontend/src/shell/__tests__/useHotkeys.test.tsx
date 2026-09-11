/**
 * @vitest-environment happy-dom
 *
 * The keyboard layer, driven through the real store.
 */
import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { useHotkeys } from '@/shell/useHotkeys'
import { useFocusStore } from '@/state/focus'
import { useUiStore } from '@/state/ui'

function Harness() {
  useHotkeys()
  return (
    <>
      <textarea data-testid="composer" />
      {/* The three widgets that own the arrow keys themselves. */}
      <div data-testid="slider" role="slider" tabIndex={0} aria-valuenow={50} />
      <div className="react-flow">
        <button data-testid="node" type="button" />
      </div>
      <div data-testid="dialog" role="dialog" tabIndex={0} />
    </>
  )
}

function press(key: string, target: EventTarget = window, init: KeyboardEventInit = {}) {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init })
  target.dispatchEvent(event)
  return event
}

beforeEach(() => {
  useFocusStore.setState({
    activeViewKey: null,
    viewKeys: ['TC', 'FCIR', 'NDBI'],
    swipe: 50,
    composer: null,
    activeKpiId: null,
    focusedStep: null,
    pipelineOpen: false,
  })
})
afterEach(() => {
  // Auto-cleanup only registers under `globals: true`; without this every
  // render leaves its window listener attached and each key fires N times.
  cleanup()
  useFocusStore.getState().resetForNewRun()
  useUiStore.setState({ shortcutsEnabled: true })
})

describe('useHotkeys', () => {
  it('nudges the swipe with [ and ]', () => {
    render(<Harness />)
    press(']')
    expect(useFocusStore.getState().swipe).toBe(55)
    press('[')
    press('[')
    expect(useFocusStore.getState().swipe).toBe(45)
  })

  it('clamps the swipe at both ends', () => {
    render(<Harness />)
    useFocusStore.setState({ swipe: 98 })
    press(']')
    expect(useFocusStore.getState().swipe).toBe(100)
    useFocusStore.setState({ swipe: 2 })
    press('[')
    expect(useFocusStore.getState().swipe).toBe(0)
  })

  it('steps the evidence tray with the arrow keys', () => {
    render(<Harness />)
    press('ArrowRight')
    expect(useFocusStore.getState().activeViewKey).toBe('FCIR')
    press('ArrowRight')
    expect(useFocusStore.getState().activeViewKey).toBe('NDBI')
    // Clamped, not wrapped: wrapping past the end reads as a glitch.
    press('ArrowRight')
    expect(useFocusStore.getState().activeViewKey).toBe('NDBI')
    press('ArrowLeft')
    expect(useFocusStore.getState().activeViewKey).toBe('FCIR')
  })

  it('toggles the pipeline with P, in either case', () => {
    render(<Harness />)
    press('p')
    expect(useFocusStore.getState().pipelineOpen).toBe(true)
    press('P')
    expect(useFocusStore.getState().pipelineOpen).toBe(false)
  })

  it('focuses the composer with /', () => {
    const { getByTestId } = render(<Harness />)
    const composer = getByTestId('composer') as HTMLTextAreaElement
    useFocusStore.getState().setComposer(composer)

    press('/')
    expect(document.activeElement).toBe(composer)
  })

  it('ignores every shortcut while the user is typing', () => {
    const { getByTestId } = render(<Harness />)
    const composer = getByTestId('composer') as HTMLTextAreaElement

    // "/" in a question is a slash, not a command.
    press('/', composer)
    press(']', composer)
    press('ArrowRight', composer)
    press('p', composer)

    expect(useFocusStore.getState().swipe).toBe(50)
    expect(useFocusStore.getState().activeViewKey).toBeNull()
    expect(useFocusStore.getState().pipelineOpen).toBe(false)
  })

  it('leaves modified chords to the browser', () => {
    render(<Harness />)
    // ⌘← is "go back", not "previous view".
    press('ArrowLeft', window, { metaKey: true })
    press(']', window, { ctrlKey: true })

    expect(useFocusStore.getState().activeViewKey).toBeNull()
    expect(useFocusStore.getState().swipe).toBe(50)
  })

  it('does nothing when there are no views to step through', () => {
    useFocusStore.setState({ viewKeys: [] })
    render(<Harness />)
    press('ArrowRight')
    expect(useFocusStore.getState().activeViewKey).toBeNull()
  })

  it('leaves the arrow keys to the A/B slider handle', () => {
    // The viewer's own footer advertises that the handle moves with the arrow
    // keys; this listener used to `preventDefault()` them from `window` first.
    const { getByTestId } = render(<Harness />)
    const event = press('ArrowRight', getByTestId('slider'))

    expect(useFocusStore.getState().activeViewKey).toBeNull()
    expect(event.defaultPrevented).toBe(false)
  })

  it('leaves the arrow keys to the DAG canvas', () => {
    const { getByTestId } = render(<Harness />)
    // Fired on a node *inside* `.react-flow`, which is where focus actually is.
    const event = press('ArrowLeft', getByTestId('node'))

    expect(useFocusStore.getState().activeViewKey).toBeNull()
    expect(event.defaultPrevented).toBe(false)
  })

  it('does not steal the swipe or pipeline keys from inside the dialog', () => {
    const { getByTestId } = render(<Harness />)
    const dialog = getByTestId('dialog')
    press(']', dialog)
    press('p', dialog)

    expect(useFocusStore.getState().swipe).toBe(50)
    expect(useFocusStore.getState().pipelineOpen).toBe(false)
  })

  it('still focuses the composer from inside a dialog', () => {
    // `/` is not an arrow key and nothing else claims it: the escape hatch back
    // to the question box has to keep working wherever focus happens to be.
    const { getByTestId } = render(<Harness />)
    const composer = getByTestId('composer') as HTMLTextAreaElement
    useFocusStore.getState().setComposer(composer)

    press('/', getByTestId('dialog'))
    expect(document.activeElement).toBe(composer)
  })

  it('goes silent when the user turns single-key shortcuts off', () => {
    // WCAG 2.1.4: the switch is in the sidebar footer and is persisted.
    render(<Harness />)
    useUiStore.setState({ shortcutsEnabled: false })

    press(']')
    press('ArrowRight')
    press('p')

    expect(useFocusStore.getState().swipe).toBe(50)
    expect(useFocusStore.getState().activeViewKey).toBeNull()
    expect(useFocusStore.getState().pipelineOpen).toBe(false)
  })

  it('stops listening once unmounted', () => {
    const { unmount } = render(<Harness />)
    unmount()
    press(']')
    expect(useFocusStore.getState().swipe).toBe(50)
  })
})
