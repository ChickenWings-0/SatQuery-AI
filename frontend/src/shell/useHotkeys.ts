/**
 * The keyboard layer (roadmap F6).
 *
 *   /        focus the query composer
 *   [ / ]    nudge the A/B swipe by 5%
 *   ← / →    step through the evidence tray
 *   P        toggle the processing pipeline
 *   Esc      close the pipeline
 *
 * One listener on `window` rather than a handler per component: the shortcuts
 * act across all three columns, and scattering them would mean three components
 * each guarding against the same edge cases.
 *
 * Three of those edge cases matter.
 *
 * Typing must never trigger a shortcut — `/` in the question box is a slash, not
 * a focus command — so anything originating in a text field is ignored. And a
 * modifier means the chord belongs to the browser or the OS: `⌘←` is "go back",
 * not "previous view".
 *
 * The third is subtler and was live for a while: a text field is not the only
 * thing that owns the arrow keys. The A/B slider handle, the React Flow canvas
 * and anything inside the pipeline dialog all move with `←`/`→`, and this
 * listener was calling `preventDefault()` on every one of them from `window`.
 * The viewer's own footer advertised "the handle is focusable, so arrow keys
 * move it too" while this file made that impossible. {@link OWNS_ARROWS} is the
 * list of widgets that get their keys back.
 *
 * WCAG 2.1.4 requires single-character shortcuts to be switchable off, which is
 * what `shortcutsEnabled` in the UI store is for; `Esc` is handled by Radix
 * inside the dialog, so it is deliberately absent here rather than racing it.
 */
import { useEffect } from 'react'

import { useFocusStore } from '@/state/focus'
import { useUiStore } from '@/state/ui'

const SWIPE_STEP = 5

/**
 * Widgets whose own key handling outranks the global layer.
 *
 * `[role="slider"]` is `ReactCompareSlider`'s handle, `.react-flow` the DAG
 * canvas, `[role="dialog"]` everything the pipeline modal contains, and
 * `[role="listbox"]`/`[role="menu"]` are here so any future popup inherits the
 * rule rather than rediscovering the bug.
 */
const OWNS_ARROWS =
  '[role="slider"], [role="dialog"], [role="listbox"], [role="menu"], .react-flow'

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable
}

/** True when the focused element belongs to a widget that steers itself. */
function ownsArrows(target: EventTarget | null): boolean {
  return target instanceof Element && target.closest(OWNS_ARROWS) !== null
}

export function useHotkeys(): void {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      // ⌘K / Ctrl+K — "New Query". A chord, not a single key, so it is not
      // covered by the WCAG 2.1.4 switch and works from inside a text field:
      // its whole job is to get the caret to the composer from anywhere.
      if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        useUiStore.getState().setSection('explore')
        useFocusStore.getState().setDraft('')
        useFocusStore.getState().focusComposer()
        return
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (isTyping(event.target)) return
      if (!useUiStore.getState().shortcutsEnabled) return

      const store = useFocusStore.getState()
      const arrows = event.key === 'ArrowLeft' || event.key === 'ArrowRight'
      // The swipe and pipeline keys are checked against the same list: `[`/`]`
      // inside the dialog belong to whatever is focused there, and `P` would
      // otherwise close the dialog from inside its own trap.
      if (ownsArrows(event.target)) {
        if (arrows || event.key === '[' || event.key === ']') return
        if (event.key === 'p' || event.key === 'P') return
      }

      switch (event.key) {
        case '/':
          event.preventDefault()
          store.focusComposer()
          return
        case '[':
          event.preventDefault()
          store.nudgeSwipe(-SWIPE_STEP)
          return
        case ']':
          event.preventDefault()
          store.nudgeSwipe(SWIPE_STEP)
          return
        case 'ArrowLeft':
          event.preventDefault()
          store.stepView(-1)
          return
        case 'ArrowRight':
          event.preventDefault()
          store.stepView(1)
          return
        default:
          break
      }

      if (event.key === 'p' || event.key === 'P') {
        event.preventDefault()
        store.togglePipeline()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])
}

/** The shortcuts, for the help strip. */
export const SHORTCUTS: ReadonlyArray<[string, string]> = [
  ['/', 'ask'],
  ['←→', 'evidence'],
  ['[ ]', 'swipe'],
  ['P', 'pipeline'],
]
