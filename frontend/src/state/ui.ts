/**
 * Client-only UI state: which nav section is showing, and what the pre-flight
 * found. Kept apart from `@/state/job`, which is strictly a projection of the
 * server's event stream — mixing the two would make the reducer untestable.
 */
import { create } from 'zustand'

import { SatQueryError, validate } from '@/api/client'
import type { InputManifest, ValidateResponse } from '@/api/types'

/** One question asked this session, and the trace it produced. */
export type RunOutcome = 'running' | 'succeeded' | 'failed'

export interface RecentRun {
  traceId: string
  query: string
  at: number
  /**
   * What this session knows became of the run. Entries used to be written on
   * the 202 and never updated, so a run that failed, was cancelled or died
   * with the server sat in History looking like every other and 404ed when
   * opened. `running` is the entry's state until a terminal event lands.
   */
  outcome: RunOutcome
}

/** The rail's items, in the order the rail shows them. */
export const NAV_SECTIONS = [
  'home',
  'explore',
  'datasets',
  'tools',
  'usecases',
  'maps',
  'saved',
  'projects',
] as const
export type NavSection = (typeof NAV_SECTIONS)[number]

/**
 * Everything the centre column can show. History left the rail — its shortcut
 * list lives in the thread's History tab now — but the full-page session
 * history is still a real destination that `openRun` and "View all" reach.
 */
export type Section = NavSection | 'history'

export interface UploadedFile {
  file: File
  /** Object URL for the local thumbnail; revoked when the selection changes. */
  previewUrl: string
}

const SHORTCUTS_KEY = 'satquery.shortcuts'

/**
 * Read the shortcut preference without assuming storage works.
 *
 * `localStorage` throws outright in a Safari private window and in any context
 * where the user has blocked site data, and a throw here happens during module
 * evaluation — i.e. it takes down the whole app before React mounts. Defaults
 * to on, because the shortcuts are the product's keyboard story.
 */
function readShortcutPreference(): boolean {
  try {
    return localStorage.getItem(SHORTCUTS_KEY) !== 'off'
  } catch {
    return true
  }
}

/**
 * The in-flight pre-flight, at module scope.
 *
 * Deliberately outside the store: it is not state anything renders, and putting
 * a live `AbortController` in the store would make every subscriber re-render
 * when a request starts.
 */
let preflight: AbortController | null = null

interface UiState {
  section: Section
  files: UploadedFile[]
  validation: ValidateResponse | null
  validating: boolean
  validationError: string | null
  /** Questions asked this session, newest first — the History tab and page. */
  recentRuns: RecentRun[]
  /** The run History should open expanded, set by clicking a sidebar entry. */
  selectedTraceId: string | null
  /**
   * Whether the single-character shortcuts are live.
   *
   * WCAG 2.1.4 requires single-key shortcuts to be turnable off — they fire
   * under speech input and under any assistive technology that synthesises
   * keystrokes, where an unintended `P` opening a modal is genuinely
   * disorienting. Persisted, because a preference that resets every reload is
   * not a preference.
   */
  shortcutsEnabled: boolean

  setSection: (section: Section) => void
  setFiles: (files: File[]) => void
  clearFiles: () => void
  /**
   * Take a new selection and run pre-flight on it.
   *
   * This lives in the store rather than in `Dropzone` because the request
   * outlives the component that starts it: `setFiles` is what replaces the drop
   * target with the manifest list, so by the time `/v1/validate` answers, the
   * `Dropzone` has unmounted. An `AbortController` scoped to that component
   * therefore cancelled its own request on the way out, and the panel sat on
   * "Running pre-flight…" for ever.
   */
  selectFiles: (files: File[]) => Promise<void>
  startValidating: () => void
  setValidation: (result: ValidateResponse) => void
  setValidationError: (message: string) => void
  rememberRun: (run: RecentRun) => void
  /** Record how a remembered run ended. Unknown ids are ignored. */
  settleRun: (traceId: string, outcome: RunOutcome) => void
  /** Jump to History with one run already open. */
  openRun: (traceId: string) => void
  setShortcutsEnabled: (enabled: boolean) => void
}

export const useUiStore = create<UiState>((set, get) => ({
  section: 'explore',
  files: [],
  validation: null,
  validating: false,
  validationError: null,
  recentRuns: [],
  selectedTraceId: null,
  shortcutsEnabled: readShortcutPreference(),

  setSection: (section) => set({ section }),

  setFiles: (files) => {
    for (const existing of get().files) URL.revokeObjectURL(existing.previewUrl)
    set({
      files: files.map((file) => ({ file, previewUrl: URL.createObjectURL(file) })),
      validation: null,
      validationError: null,
    })
  },

  clearFiles: () => {
    preflight?.abort()
    preflight = null
    for (const existing of get().files) URL.revokeObjectURL(existing.previewUrl)
    set({ files: [], validation: null, validationError: null, validating: false })
  },

  selectFiles: async (files) => {
    // One pre-flight at a time. Dropping a second pair while the first is
    // still validating used to race: whichever response landed last won, so a
    // slow answer about the *old* files could describe the new selection.
    preflight?.abort()
    const controller = new AbortController()
    preflight = controller

    get().setFiles(files)
    get().startValidating()
    try {
      const result = await validate(files, undefined, controller.signal)
      if (preflight === controller) get().setValidation(result)
    } catch (error) {
      if (controller.signal.aborted || preflight !== controller) return
      get().setValidationError(
        error instanceof SatQueryError ? error.display : 'Pre-flight failed. Is the API running?',
      )
    }
  },

  startValidating: () => set({ validating: true, validationError: null }),
  setValidation: (validation) => set({ validation, validating: false }),
  setValidationError: (validationError) => set({ validationError, validating: false }),

  rememberRun: (run) =>
    set((state) => ({
      // Newest first, one entry per trace: re-running the same job id replaces
      // its entry rather than stacking duplicates.
      recentRuns: [run, ...state.recentRuns.filter((r) => r.traceId !== run.traceId)].slice(0, 20),
    })),

  settleRun: (traceId, outcome) =>
    set((state) => ({
      recentRuns: state.recentRuns.map((run) =>
        run.traceId === traceId ? { ...run, outcome } : run,
      ),
    })),

  // Navigating to History without saying *which* run left the user to re-find
  // the one they had just clicked.
  openRun: (selectedTraceId) => set({ section: 'history', selectedTraceId }),

  setShortcutsEnabled: (shortcutsEnabled) => {
    set({ shortcutsEnabled })
    try {
      localStorage.setItem(SHORTCUTS_KEY, shortcutsEnabled ? 'on' : 'off')
    } catch {
      // A preference that cannot be persisted still applies to this session.
    }
  },
}))

/** Every manifest field that is `null` carries a reason in `warnings` (§1). */
export function warningsFor(manifest: InputManifest): string[] {
  return manifest.warnings ?? []
}
