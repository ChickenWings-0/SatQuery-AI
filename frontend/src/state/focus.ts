/**
 * Cross-column focus: what the viewer is showing, which KPI is lit, and where
 * the pipeline modal should open.
 *
 * This is the state that makes a citation click do three things at once — reveal
 * the evidence that backs it, highlight the KPI card carrying the same number,
 * and remember the producing step so that *if* the user then opens the pipeline
 * it opens centred on that node. Keeping it in one store is what lets those
 * three live in three different columns without prop-drilling through the shell.
 */
import { create } from 'zustand'

interface FocusState {
  /** `ViewGroup.key` currently in the viewer; null means "first available". */
  activeViewKey: string | null
  /** Ordered tray keys, published by the Data Stage so `←`/`→` can walk them. */
  viewKeys: string[]
  /** A/B swipe position, 0-100. Lifted here so `[`/`]` can drive it. */
  swipe: number
  /**
   * Bumped only by a *programmatic* move, never by a drag.
   *
   * `ReactCompareSlider` v4 is uncontrolled and its context `setPosition` does
   * not take effect from the `handle` slot — the slider simply echoes its old
   * position back through `onPositionChange`, reverting the store. So a
   * keyboard nudge re-seeds the slider by remounting it with a new
   * `defaultPosition`, and this counter is the remount key. Dragging updates
   * `swipe` without touching the epoch, so the common case never remounts.
   */
  swipeEpoch: number
  /** The query textarea, registered by the composer so `/` can focus it. */
  composer: HTMLTextAreaElement | null
  /**
   * The text in the question box.
   *
   * Lifted out of the composer's own `useState` so that something in another
   * column can put a question into it — which is what the pre-flight's
   * "questions these images can answer" chips do. They were `<button>`s with
   * hover styling and no handler: the most inviting control on the first
   * screen, and it did nothing.
   */
  draft: string
  /** `KpiCard.id` to highlight, cleared on the next selection. */
  activeKpiId: string | null
  /** Plan step a citation pointed at — the pipeline modal opens on it. */
  focusedStep: number | null
  pipelineOpen: boolean

  selectView: (key: string) => void
  setViewKeys: (keys: string[]) => void
  /** Walk the evidence tray by *delta* entries, clamped at both ends. */
  stepView: (delta: number) => void
  setSwipe: (position: number) => void
  nudgeSwipe: (delta: number) => void
  setComposer: (element: HTMLTextAreaElement | null) => void
  setDraft: (text: string) => void
  focusComposer: () => void
  /** Put a suggested question in the box and hand the user the caret. */
  proposeQuestion: (text: string) => void
  togglePipeline: () => void
  /** The citation click: evidence, KPI and pending pipeline focus together. */
  focusCitation: (options: {
    viewKey?: string | undefined
    kpiId?: string | undefined
    step: number
  }) => void
  clearFocus: () => void
  openPipeline: (step?: number) => void
  closePipeline: () => void
  resetForNewRun: () => void
}

export const useFocusStore = create<FocusState>((set) => ({
  activeViewKey: null,
  viewKeys: [],
  swipe: 50,
  swipeEpoch: 0,
  composer: null,
  draft: '',
  activeKpiId: null,
  focusedStep: null,
  pipelineOpen: false,

  selectView: (activeViewKey) => set({ activeViewKey }),

  setViewKeys: (viewKeys) =>
    set((state) =>
      // Referential equality matters: this is called from a render-time memo,
      // and a fresh array every time would loop the effect that publishes it.
      state.viewKeys.length === viewKeys.length &&
      state.viewKeys.every((key, index) => key === viewKeys[index])
        ? state
        : { viewKeys },
    ),

  stepView: (delta) =>
    set((state) => {
      if (state.viewKeys.length === 0) return state
      const current = state.activeViewKey
        ? state.viewKeys.indexOf(state.activeViewKey)
        : 0
      const next = Math.min(
        state.viewKeys.length - 1,
        Math.max(0, (current === -1 ? 0 : current) + delta),
      )
      return { activeViewKey: state.viewKeys[next] ?? state.activeViewKey }
    }),

  /** Report a drag. Does not remount the slider. */
  setSwipe: (swipe) => set({ swipe: Math.min(100, Math.max(0, swipe)) }),

  /** Move it from the keyboard. Re-seeds the slider. */
  nudgeSwipe: (delta) =>
    set((state) => {
      const swipe = Math.min(100, Math.max(0, state.swipe + delta))
      return swipe === state.swipe ? state : { swipe, swipeEpoch: state.swipeEpoch + 1 }
    }),

  setComposer: (composer) => set({ composer }),

  setDraft: (draft) => set({ draft }),

  focusComposer: () => {
    const element = useFocusStore.getState().composer
    element?.focus()
    element?.select()
  },

  proposeQuestion: (draft) => {
    set({ draft })
    // The caret goes to the end rather than selecting the text: the suggestion
    // is a starting point to edit, and a full selection means the next
    // keystroke destroys it.
    const element = useFocusStore.getState().composer
    if (!element) return
    element.focus()
    // After the controlled value has actually landed in the DOM node.
    requestAnimationFrame(() => {
      const end = element.value.length
      element.setSelectionRange(end, end)
    })
  },

  togglePipeline: () => set((state) => ({ pipelineOpen: !state.pipelineOpen })),

  focusCitation: ({ viewKey, kpiId, step }) =>
    set((state) => ({
      // A citation whose step produced only SCALARS has no picture to reveal;
      // leaving the viewer alone beats blanking it.
      activeViewKey: viewKey ?? state.activeViewKey,
      activeKpiId: kpiId ?? null,
      focusedStep: step,
    })),

  clearFocus: () => set({ activeKpiId: null }),

  openPipeline: (step) =>
    set((state) => ({ pipelineOpen: true, focusedStep: step ?? state.focusedStep })),

  closePipeline: () => set({ pipelineOpen: false }),

  resetForNewRun: () =>
    set({
      activeViewKey: null,
      activeKpiId: null,
      focusedStep: null,
      pipelineOpen: false,
      swipe: 50,
      swipeEpoch: 0,
    }),
}))
