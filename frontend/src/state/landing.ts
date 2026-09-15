/**
 * Landing-page state. Small, unpersisted; it exists because the globe, the
 * hero copy and the drop overlay live in different subtrees and share a few
 * facts. Nothing in the R3F tree subscribes to it — a Zustand subscription
 * inside `useFrame` is a re-render per frame — actions read via `getState()`.
 *
 * Hover is not here. The capability list previews on click, Enter and the
 * arrow keys only; a row's hover treatment is CSS, and nothing about it
 * belongs in a store.
 */
import { create } from 'zustand'

export type HeroPhase = 'poster' | 'acquiring' | 'live'
export type GlobeSupport = 'unknown' | 'webgl' | 'none'
export type CapabilityId = 'change' | 'crossmodal' | 'grounding'
export const CAPABILITY_ORDER: readonly CapabilityId[] = ['change', 'crossmodal', 'grounding']
export type StageId = 1 | 2 | 3 | 4

interface LandingState {
  phase: HeroPhase
  globe: GlobeSupport
  globeVisible: boolean
  dragOver: boolean
  activeCapability: CapabilityId
  /** Which way the preview slides on the next switch: +1 down the list, -1 up. */
  capabilityDirection: 1 | -1
  /** Hero exit progress, 0 at the top of the page to 1 when the hero has left. */
  scroll: number
  /** The pointer over the hero, as fractions; the globe's rim light leans toward it. */
  pointer: [x: number, y: number]
  storyPlayed: StageId[]
  storyReplaying: StageId | null

  setPhase: (phase: HeroPhase) => void
  setGlobe: (globe: GlobeSupport) => void
  setGlobeVisible: (visible: boolean) => void
  setDragOver: (over: boolean) => void
  setCapability: (id: CapabilityId) => void
  setScroll: (scroll: number) => void
  setPointer: (x: number, y: number) => void
  markStagePlayed: (stage: StageId) => void
  replayStory: () => Promise<void>
  reset: () => void
}

const initial = {
  phase: 'poster' as HeroPhase,
  globe: 'unknown' as GlobeSupport,
  globeVisible: false,
  dragOver: false,
  activeCapability: 'change' as CapabilityId,
  capabilityDirection: 1 as 1 | -1,
  scroll: 0,
  pointer: [0.5, 0.5] as [number, number],
  storyPlayed: [] as StageId[],
  storyReplaying: null,
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export const useLandingStore = create<LandingState>((set, get) => ({
  ...initial,
  setPhase: (phase) => set({ phase }),
  setGlobe: (globe) => set({ globe }),
  setGlobeVisible: (globeVisible) => set({ globeVisible }),
  setDragOver: (dragOver) => set({ dragOver }),
  setCapability: (activeCapability) =>
    set((state) => {
      if (state.activeCapability === activeCapability) return state
      const from = CAPABILITY_ORDER.indexOf(state.activeCapability)
      const to = CAPABILITY_ORDER.indexOf(activeCapability)
      return { activeCapability, capabilityDirection: to > from ? 1 : -1 }
    }),
  // Written per frame while the hero scrolls out; nothing subscribes to it
  // from React, so the write is a property set and a `getState()` read.
  setScroll: (scroll) => set({ scroll }),
  setPointer: (x, y) => set({ pointer: [x, y] }),
  markStagePlayed: (stage) =>
    set((state) =>
      state.storyPlayed.includes(stage) ? state : { storyPlayed: [...state.storyPlayed, stage] },
    ),
  replayStory: async () => {
    if (get().storyReplaying !== null) return
    set({ storyPlayed: [] })
    for (const stage of [1, 2, 3, 4] as const) {
      set({ storyReplaying: stage })
      get().markStagePlayed(stage)
      await wait(1400)
    }
    set({ storyReplaying: null })
  },
  reset: () => set(initial),
}))
