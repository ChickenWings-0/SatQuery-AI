/**
 * The keyboard layer, dispatching over `@/shell/shortcuts`'s binding table.
 *
 *   ⌘K       new query — a chord, works from inside a text field
 *   ⌘/       the shortcuts guide — the always-on route in
 *   ⌘J       toggle theme
 *   ⌘,       settings
 *   ?        the shortcuts guide
 *   /        focus the query composer
 *   [ / ]    nudge the A/B swipe by 5% (or the Maps split, on Maps)
 *   ← / →    step through the evidence tray
 *   P        toggle the processing pipeline
 *   G then … go to a section (600 ms window)
 *   Esc      close the pipeline / a popover — handled by Radix, absent here
 *
 * One listener on `window` rather than a handler per component: the shortcuts
 * act across all three columns, and scattering them would mean three
 * components each guarding against the same edge cases.
 *
 * Three of those edge cases matter. Typing must never trigger a shortcut —
 * `/` in the question box is a slash — so anything from a text field is
 * ignored. A modifier means the chord belongs to the browser or the OS
 * unless it is one of ours. And a text field is not the only thing that owns
 * the arrow keys: the A/B slider handle, the React Flow canvas, the map and
 * anything inside a dialog all move with `←`/`→`; {@link OWNS_ARROWS} is the
 * list of widgets that get their keys back.
 *
 * WCAG 2.1.4 requires single-character shortcuts to be switchable off, which
 * is what `useShortcutStore.enabled` is for. Chords are never gated.
 */
import { useEffect } from 'react'

import { SEQUENCE_WINDOW_MS } from '@/shell/shortcuts'
import { useFocusStore } from '@/state/focus'
import { useMapStore } from '@/state/map'
import { useSettingsStore } from '@/state/settings'
import { useShortcutStore } from '@/state/shortcuts'
import { useThemeStore } from '@/state/theme'
import { useUiStore, type Section } from '@/state/ui'

const SWIPE_STEP = 5

const OWNS_ARROWS =
  '[role="slider"], [role="dialog"], [role="listbox"], [role="menu"], .react-flow, .maplibregl-map'

const GO: Record<string, Section> = {
  h: 'home',
  e: 'explore',
  u: 'usecases',
  m: 'maps',
  s: 'saved',
  p: 'projects',
}

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable
}

function ownsArrows(target: EventTarget | null): boolean {
  return target instanceof Element && target.closest(OWNS_ARROWS) !== null
}

export function useHotkeys(): void {
  useEffect(() => {
    let pendingGo: number | null = null

    function clearGo() {
      if (pendingGo !== null) window.clearTimeout(pendingGo)
      pendingGo = null
    }

    function onKeyDown(event: KeyboardEvent) {
      const mod = event.metaKey || event.ctrlKey
      const key = event.key

      // Chords: never gated, work from inside a text field.
      if (mod && !event.altKey) {
        const lower = key.toLowerCase()
        if (lower === 'k') {
          event.preventDefault()
          useUiStore.getState().setSection('explore')
          useFocusStore.getState().setDraft('')
          useFocusStore.getState().focusComposer()
          return
        }
        if (key === '/') {
          event.preventDefault()
          useShortcutStore.getState().toggleGuide()
          return
        }
        if (lower === 'j') {
          event.preventDefault()
          useThemeStore.getState().toggle()
          return
        }
        if (key === ',') {
          event.preventDefault()
          useSettingsStore.getState().openSettings()
          return
        }
        return
      }
      if (event.altKey) return
      if (isTyping(event.target)) return
      if (!useShortcutStore.getState().enabled) return

      // `G` then a letter.
      if (pendingGo !== null) {
        clearGo()
        const target = GO[key.toLowerCase()]
        if (target) {
          event.preventDefault()
          useUiStore.getState().setSection(target)
          return
        }
      }
      if (key === 'g' || key === 'G') {
        pendingGo = window.setTimeout(clearGo, SEQUENCE_WINDOW_MS)
        return
      }

      const store = useFocusStore.getState()
      const onMaps = useUiStore.getState().section === 'maps'
      const arrows = key === 'ArrowLeft' || key === 'ArrowRight'
      if (ownsArrows(event.target)) {
        if (arrows || key === '[' || key === ']') return
        if (key === 'p' || key === 'P') return
      }

      switch (key) {
        case '?':
          event.preventDefault()
          useShortcutStore.getState().toggleGuide()
          return
        case '/':
          event.preventDefault()
          store.focusComposer()
          return
        case '[':
          event.preventDefault()
          if (onMaps) useMapStore.getState().nudgeSplit(-SWIPE_STEP)
          else store.nudgeSwipe(-SWIPE_STEP)
          return
        case ']':
          event.preventDefault()
          if (onMaps) useMapStore.getState().nudgeSplit(SWIPE_STEP)
          else store.nudgeSwipe(SWIPE_STEP)
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

      if (key === 'p' || key === 'P') {
        event.preventDefault()
        store.togglePipeline()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      clearGo()
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [])
}

export { SHORTCUTS } from '@/shell/shortcuts'
