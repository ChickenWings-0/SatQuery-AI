/**
 * Dark / light / system.
 *
 * The class goes on `<html>` (`.dark`), which is what `theme.css` keys the
 * light overrides off (`:root:not(.dark)`), and `color-scheme` follows it so
 * native controls, scrollbars and the file picker agree. An inline script in
 * `index.html` applies the stored preference before first paint; this store
 * hydrates from the same key, so the DOM and the store never disagree.
 */
import { create } from 'zustand'

import { safeStorage } from '@/shell/storage'

export type ThemePref = 'dark' | 'light' | 'system'
export type ResolvedTheme = 'dark' | 'light'

export const THEME_KEY = 'satquery.theme'
const QUERY = '(prefers-color-scheme: dark)'

function systemTheme(): ResolvedTheme {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return 'dark'
  return window.matchMedia(QUERY).matches ? 'dark' : 'light'
}

function readPref(): ThemePref {
  const raw = safeStorage.getItem(THEME_KEY)
  return raw === 'light' || raw === 'system' ? raw : 'dark'
}

function resolve(pref: ThemePref): ResolvedTheme {
  return pref === 'system' ? systemTheme() : pref
}

export function applyTheme(resolved: ResolvedTheme): void {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  root.classList.toggle('dark', resolved === 'dark')
  root.style.colorScheme = resolved
  // The browser chrome takes the ground colour; keep it honest on toggle.
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]:not([media])')
  if (meta) meta.content = resolved === 'dark' ? '#100c0a' : '#f8f5f1'
}

interface ThemeState {
  pref: ThemePref
  resolved: ResolvedTheme
  setPref: (pref: ThemePref) => void
  /** dark ⇄ light; from `system`, the opposite of what is currently shown. */
  toggle: () => void
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  pref: readPref(),
  resolved: resolve(readPref()),

  setPref: (pref) => {
    const resolved = resolve(pref)
    safeStorage.setItem(THEME_KEY, pref)
    const swap = () => {
      applyTheme(resolved)
      set({ pref, resolved })
    }
    // A cross-fade between the two grounds, where the platform can do it
    // cheaply. Never `transition: background-color` on `*`: that paints every
    // node in the tree for 200 ms beside a saturated GPU.
    const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown }
    const reduced =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (typeof doc.startViewTransition === 'function' && !reduced) doc.startViewTransition(swap)
    else swap()
  },

  toggle: () => get().setPref(get().resolved === 'dark' ? 'light' : 'dark'),
}))

/** Keep `system` live while the tab is open. Call once from `main.tsx`. */
export function bindTheme(): () => void {
  applyTheme(useThemeStore.getState().resolved)
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return () => undefined
  const list = window.matchMedia(QUERY)
  const onChange = () => {
    const { pref } = useThemeStore.getState()
    if (pref !== 'system') return
    const resolved = resolve(pref)
    applyTheme(resolved)
    useThemeStore.setState({ resolved })
  }
  list.addEventListener('change', onChange)
  return () => list.removeEventListener('change', onChange)
}
