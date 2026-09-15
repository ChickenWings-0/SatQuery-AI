/**
 * The single-key shortcut switch and the shortcuts guide.
 *
 * WCAG 2.1.4 requires single-character shortcuts to be turnable off — they
 * fire under speech input and under any assistive technology that synthesises
 * keystrokes, where an unintended `P` opening a modal is genuinely
 * disorienting. The switch is global (the popover, the guide's footer and the
 * hotkey layer all read it) and persisted, because a preference that resets
 * every reload is not a preference. Chords (`⌘K`, `⌘/`) are never gated: they
 * cannot be typed by accident, and `⌘/` is how the guide is reached with the
 * single keys off.
 */
import { create } from 'zustand'

import { safeStorage } from '@/shell/storage'

export const SHORTCUTS_KEY = 'satquery.shortcuts'

interface ShortcutState {
  enabled: boolean
  guideOpen: boolean
  setEnabled: (enabled: boolean) => void
  openGuide: () => void
  closeGuide: () => void
  toggleGuide: () => void
}

export const useShortcutStore = create<ShortcutState>((set) => ({
  enabled: safeStorage.getItem(SHORTCUTS_KEY) !== 'off',
  guideOpen: false,

  setEnabled: (enabled) => {
    set({ enabled })
    safeStorage.setItem(SHORTCUTS_KEY, enabled ? 'on' : 'off')
  },
  openGuide: () => set({ guideOpen: true }),
  closeGuide: () => set({ guideOpen: false }),
  toggleGuide: () => set((state) => ({ guideOpen: !state.guideOpen })),
}))
