/**
 * The binding table: the single source of truth for what the keyboard does.
 *
 * `useHotkeys` dispatches over it, the shortcuts guide renders it, and the
 * viewer's help strip derives its four entries from it, so none of the three
 * can drift. `chord: true` bindings carry a modifier and are never gated by
 * the WCAG 2.1.4 switch; the rest are single keys and are.
 *
 * `G` followed by a letter is a *sequence*: the second key has 600 ms to
 * arrive, and the pair is treated as one shortcut for the switch's purposes.
 */
export type ShortcutScope = 'global' | 'thread' | 'stage' | 'navigation' | 'maps'

export interface Binding {
  id: string
  /** Display keys, in press order. `⌘` means ⌘ on macOS and Ctrl elsewhere. */
  keys: readonly string[]
  label: string
  scope: ShortcutScope
  chord: boolean
}

export const BINDINGS: readonly Binding[] = [
  { id: 'new-query', keys: ['⌘', 'K'], label: 'New query', scope: 'global', chord: true },
  { id: 'guide-chord', keys: ['⌘', '/'], label: 'Keyboard shortcuts', scope: 'global', chord: true },
  { id: 'theme', keys: ['⌘', 'J'], label: 'Toggle dark / light', scope: 'global', chord: true },
  { id: 'settings', keys: ['⌘', ','], label: 'Settings', scope: 'global', chord: true },
  { id: 'guide', keys: ['?'], label: 'Keyboard shortcuts', scope: 'global', chord: false },
  { id: 'focus-composer', keys: ['/'], label: 'Ask', scope: 'thread', chord: false },
  { id: 'view-prev', keys: ['←'], label: 'Previous evidence view', scope: 'stage', chord: false },
  { id: 'view-next', keys: ['→'], label: 'Next evidence view', scope: 'stage', chord: false },
  { id: 'swipe-left', keys: ['['], label: 'Swipe left 5 %', scope: 'stage', chord: false },
  { id: 'swipe-right', keys: [']'], label: 'Swipe right 5 %', scope: 'stage', chord: false },
  { id: 'pipeline', keys: ['P'], label: 'Toggle processing pipeline', scope: 'stage', chord: false },
  { id: 'go-home', keys: ['G', 'H'], label: 'Go to landing', scope: 'navigation', chord: false },
  { id: 'go-explore', keys: ['G', 'E'], label: 'Go to New Query', scope: 'navigation', chord: false },
  { id: 'go-usecases', keys: ['G', 'U'], label: 'Go to Use cases', scope: 'navigation', chord: false },
  { id: 'go-maps', keys: ['G', 'M'], label: 'Go to Maps', scope: 'navigation', chord: false },
  { id: 'go-saved', keys: ['G', 'S'], label: 'Go to Saved', scope: 'navigation', chord: false },
  { id: 'go-projects', keys: ['G', 'P'], label: 'Go to Projects', scope: 'navigation', chord: false },
]

export const SCOPE_LABELS: Record<ShortcutScope, string> = {
  global: 'Anywhere',
  thread: 'The question',
  stage: 'The imagery',
  navigation: 'Go to',
  maps: 'Maps',
}

/** The viewer's help strip: the stage bindings, as `[keys, short label]`. */
export const SHORTCUTS: ReadonlyArray<[string, string]> = [
  ['/', 'ask'],
  ['←→', 'evidence'],
  ['[ ]', 'swipe'],
  ['P', 'pipeline'],
]

export const SEQUENCE_WINDOW_MS = 600
