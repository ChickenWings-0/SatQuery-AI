/**
 * Preferences the settings dialog owns, and the dialog's own open state.
 *
 * Two of these change what is sent to the server, and both are wired to
 * fields the wire contract actually has (API_CONTRACT §4.1):
 *
 *   customInstructions   Prepended to every question, as part of `query`.
 *                        There is no separate "system prompt" field on the
 *                        wire and the frontend does not invent one; the
 *                        planner reads the query, so the instructions ride
 *                        ahead of it. `query` is capped at 1000 characters
 *                        server-side, so the composer's own ceiling is what
 *                        is left after the instructions ({@link queryRoom}),
 *                        and the instructions are capped so that what is
 *                        left is always a usable question.
 *   seed                 `options.seed`. 0 means "server default", exactly as
 *                        the contract says; anything else makes a rerun
 *                        byte-identical, trace included.
 *
 * Everything else here is UI. Theme, the single-key switch and the library
 * layout keep their own stores — this one does not duplicate them, the
 * dialog just renders their controls in one place.
 */
import { create } from 'zustand'

import { safeStorage } from '@/shell/storage'

export const SETTINGS_KEY = 'satquery.settings'
export type SettingsTab = 'personalization' | 'data' | 'appearance'

/** The server's ceiling on a `query` string. */
export const QUERY_MAX = 1000
/** Long enough for a paragraph of house style, short enough to leave room for a question. */
export const INSTRUCTIONS_MAX = 400

export interface Preferences {
  customInstructions: string
  seed: number
  /**
   * Whether the Maps page may call the internet — place search, scene
   * search, the imagery fetch. Off by default because this console is judged
   * in halls without a route out; remembered once switched on. `?mock=1`
   * treats it as on, because the mock serves those calls from fixtures.
   */
  onlineFeatures: boolean
}

const DEFAULT: Preferences = { customInstructions: '', seed: 0, onlineFeatures: false }

function read(): Preferences {
  const raw = safeStorage.getItem(SETTINGS_KEY)
  if (!raw) return DEFAULT
  try {
    const parsed = JSON.parse(raw) as Partial<Preferences>
    return {
      customInstructions:
        typeof parsed.customInstructions === 'string'
          ? parsed.customInstructions.slice(0, INSTRUCTIONS_MAX)
          : DEFAULT.customInstructions,
      seed: Number.isInteger(parsed.seed) && (parsed.seed as number) >= 0 ? (parsed.seed as number) : DEFAULT.seed,
      onlineFeatures: parsed.onlineFeatures === true,
    }
  } catch {
    return DEFAULT
  }
}

function write(prefs: Preferences): void {
  safeStorage.setItem(SETTINGS_KEY, JSON.stringify(prefs))
}

interface SettingsState extends Preferences {
  open: boolean
  tab: SettingsTab
  setCustomInstructions: (text: string) => void
  setSeed: (seed: number) => void
  setOnlineFeatures: (on: boolean) => void
  /** Back to defaults; the caller decides whether to also clear storage. */
  reset: () => void
  openSettings: (tab?: SettingsTab) => void
  closeSettings: () => void
  setTab: (tab: SettingsTab) => void
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  ...read(),
  open: false,
  tab: 'personalization',

  setCustomInstructions: (text) => {
    const customInstructions = text.slice(0, INSTRUCTIONS_MAX)
    set({ customInstructions })
    write({ customInstructions, seed: get().seed, onlineFeatures: get().onlineFeatures })
  },
  setSeed: (seed) => {
    const next = Number.isInteger(seed) && seed >= 0 ? seed : 0
    set({ seed: next })
    write({ customInstructions: get().customInstructions, seed: next, onlineFeatures: get().onlineFeatures })
  },
  setOnlineFeatures: (onlineFeatures) => {
    set({ onlineFeatures })
    write({ customInstructions: get().customInstructions, seed: get().seed, onlineFeatures })
  },
  reset: () => {
    safeStorage.removeItem(SETTINGS_KEY)
    set(DEFAULT)
  },

  openSettings: (tab) => set(tab ? { open: true, tab } : { open: true }),
  closeSettings: () => set({ open: false }),
  setTab: (tab) => set({ tab }),
}))

/** Characters the instructions take out of the query budget: the text plus the blank line between. */
export function instructionsCost(instructions: string): number {
  const trimmed = instructions.trim()
  return trimmed ? trimmed.length + 2 : 0
}

/** What the composer may still accept once the instructions are counted. */
export function queryRoom(instructions: string): number {
  return QUERY_MAX - instructionsCost(instructions)
}

/** The `query` actually sent: instructions, a blank line, the question. */
export function composeQuery(question: string, instructions: string): string {
  const trimmed = instructions.trim()
  return trimmed ? `${trimmed}\n\n${question}` : question
}
