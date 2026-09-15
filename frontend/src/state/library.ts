/**
 * Saved runs and projects, on this device.
 *
 * There is no server-side save. Everything here lives in IndexedDB (via
 * `idb-keyval`, one store, two keys), so a run survives a reload and a project
 * survives a week — and the UI says "on this device" wherever it lists them,
 * never "cloud". A saved run keeps a summary and a small thumbnail; the full
 * trace is re-fetched from `GET /v1/traces/{id}` on open, and if the server no
 * longer has it the card says so and offers the summary only.
 *
 * Writes go to memory first and IndexedDB second, so the UI never waits on
 * the disk, and a failed write is reported through a toast rather than by a
 * card that silently never appeared.
 */
import { createStore, del, get as idbGet, set as idbSet } from 'idb-keyval'
import { create } from 'zustand'

import type { TaskType } from '@/api/types'
import { toast } from '@/state/notifications'
import type { NormalisedBox } from '@/thread/bbox'

export type PairType = 'SINGLE' | 'BI_TEMPORAL' | 'CROSS_MODAL' | 'INCOMPATIBLE'
export type RunOutcome = 'running' | 'succeeded' | 'failed'
export type ProjectColour = 'terracotta' | 'sand' | 'ok' | 'warn'

export interface SavedRun {
  traceId: string
  query: string
  taskType: TaskType
  pairType: PairType
  sensors: string[]
  savedAt: number
  ranAt: number
  outcome: RunOutcome
  confidence: number | null
  headline: { label: string; value: string } | null
  boxes: NormalisedBox[]
  /** WGS84 `[west, south, east, north]`, when the manifest carried one. */
  bounds: [number, number, number, number] | null
  thumb: Blob | null
  projectId: string | null
  tags: string[]
}

export interface Project {
  id: string
  name: string
  description: string
  createdAt: number
  updatedAt: number
  colour: ProjectColour
  aoi: [number, number, number, number] | null
  pinned: boolean
}

export type LibraryView = 'cards' | 'rows'
export type LibrarySort = 'savedAt' | 'ranAt' | 'confidence'

const STORE = createStore('satquery', 'library')
const RUNS_KEY = 'runs'
const PROJECTS_KEY = 'projects'
const VIEW_KEY = 'satquery.saved.view'

function readView(): LibraryView {
  try {
    return localStorage.getItem(VIEW_KEY) === 'rows' ? 'rows' : 'cards'
  } catch {
    return 'cards'
  }
}

function uid(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

function unionBounds(runs: SavedRun[]): Project['aoi'] {
  let out: Project['aoi'] = null
  for (const run of runs) {
    if (!run.bounds) continue
    out = out
      ? [
          Math.min(out[0], run.bounds[0]),
          Math.min(out[1], run.bounds[1]),
          Math.max(out[2], run.bounds[2]),
          Math.max(out[3], run.bounds[3]),
        ]
      : [...run.bounds]
  }
  return out
}

interface LibraryState {
  runs: Record<string, SavedRun>
  projects: Record<string, Project>
  hydrated: boolean
  view: LibraryView
  sort: LibrarySort
  filter: { tasks: TaskType[]; projectId: string | null; text: string }
  selected: string[]

  hydrate: () => Promise<void>
  save: (run: SavedRun) => Promise<void>
  remove: (traceId: string) => Promise<SavedRun | null>
  restore: (run: SavedRun) => Promise<void>
  assign: (traceIds: string[], projectId: string | null) => Promise<void>
  createProject: (input: Pick<Project, 'name' | 'description' | 'colour'>) => Promise<Project>
  updateProject: (id: string, patch: Partial<Pick<Project, 'name' | 'description' | 'colour' | 'pinned'>>) => Promise<void>
  deleteProject: (id: string, options?: { keepRuns?: boolean }) => Promise<{ project: Project; runs: SavedRun[] } | null>
  restoreProject: (project: Project, runs: SavedRun[]) => Promise<void>
  clearAll: () => Promise<void>

  setView: (view: LibraryView) => void
  setSort: (sort: LibrarySort) => void
  setFilter: (patch: Partial<LibraryState['filter']>) => void
  toggleSelect: (traceId: string) => void
  clearSelection: () => void
}

async function persist(state: Pick<LibraryState, 'runs' | 'projects'>): Promise<void> {
  try {
    await idbSet(RUNS_KEY, state.runs, STORE)
    await idbSet(PROJECTS_KEY, state.projects, STORE)
  } catch {
    toast('Could not write to this device’s storage — changes apply to this session only.', 'warn')
  }
}

export const useLibraryStore = create<LibraryState>((set, get) => ({
  runs: {},
  projects: {},
  hydrated: false,
  view: readView(),
  sort: 'savedAt',
  filter: { tasks: [], projectId: null, text: '' },
  selected: [],

  hydrate: async () => {
    if (get().hydrated) return
    try {
      const [runs, projects] = await Promise.all([
        idbGet<Record<string, SavedRun>>(RUNS_KEY, STORE),
        idbGet<Record<string, Project>>(PROJECTS_KEY, STORE),
      ])
      set({ runs: runs ?? {}, projects: projects ?? {}, hydrated: true })
    } catch {
      set({ hydrated: true })
    }
  },

  save: async (run) => {
    set((state) => ({ runs: { ...state.runs, [run.traceId]: run } }))
    await persist(get())
  },

  remove: async (traceId) => {
    const run = get().runs[traceId] ?? null
    if (!run) return null
    set((state) => {
      const runs = { ...state.runs }
      delete runs[traceId]
      return { runs, selected: state.selected.filter((id) => id !== traceId) }
    })
    await persist(get())
    return run
  },

  restore: async (run) => {
    await get().save(run)
  },

  assign: async (traceIds, projectId) => {
    set((state) => {
      const runs = { ...state.runs }
      for (const id of traceIds) {
        const run = runs[id]
        if (run) runs[id] = { ...run, projectId }
      }
      const projects = { ...state.projects }
      for (const project of Object.values(projects)) {
        const members = Object.values(runs).filter((r) => r.projectId === project.id)
        projects[project.id] = { ...project, aoi: unionBounds(members), updatedAt: Date.now() }
      }
      return { runs, projects, selected: [] }
    })
    await persist(get())
  },

  createProject: async (input) => {
    const now = Date.now()
    const project: Project = {
      id: uid('p'),
      name: input.name.trim() || 'Untitled investigation',
      description: input.description.trim(),
      colour: input.colour,
      createdAt: now,
      updatedAt: now,
      aoi: null,
      pinned: false,
    }
    set((state) => ({ projects: { ...state.projects, [project.id]: project } }))
    await persist(get())
    return project
  },

  updateProject: async (id, patch) => {
    set((state) => {
      const project = state.projects[id]
      if (!project) return state
      return { projects: { ...state.projects, [id]: { ...project, ...patch, updatedAt: Date.now() } } }
    })
    await persist(get())
  },

  deleteProject: async (id, options) => {
    const project = get().projects[id]
    if (!project) return null
    const keepRuns = options?.keepRuns ?? true
    const members = Object.values(get().runs).filter((r) => r.projectId === id)
    set((state) => {
      const projects = { ...state.projects }
      delete projects[id]
      const runs = { ...state.runs }
      for (const run of members) {
        if (keepRuns) runs[run.traceId] = { ...run, projectId: null }
        else delete runs[run.traceId]
      }
      return { projects, runs }
    })
    await persist(get())
    return { project, runs: members }
  },

  restoreProject: async (project, runs) => {
    set((state) => {
      const next = { ...state.runs }
      for (const run of runs) next[run.traceId] = run
      return { projects: { ...state.projects, [project.id]: project }, runs: next }
    })
    await persist(get())
  },

  clearAll: async () => {
    set({ runs: {}, projects: {}, selected: [] })
    try {
      await del(RUNS_KEY, STORE)
      await del(PROJECTS_KEY, STORE)
    } catch {
      // Cleared in memory regardless.
    }
  },

  setView: (view) => {
    set({ view })
    try {
      localStorage.setItem(VIEW_KEY, view)
    } catch {
      // Session-only, then.
    }
  },
  setSort: (sort) => set({ sort }),
  setFilter: (patch) => set((state) => ({ filter: { ...state.filter, ...patch } })),
  toggleSelect: (traceId) =>
    set((state) => ({
      selected: state.selected.includes(traceId)
        ? state.selected.filter((id) => id !== traceId)
        : [...state.selected, traceId],
    })),
  clearSelection: () => set({ selected: [] }),
}))

/** The runs the Saved page lists, filtered and sorted per the store. */
export function visibleRuns(state: LibraryState): SavedRun[] {
  const { tasks, projectId, text } = state.filter
  const needle = text.trim().toLowerCase()
  const list = Object.values(state.runs).filter((run) => {
    if (tasks.length > 0 && !tasks.includes(run.taskType)) return false
    if (projectId && run.projectId !== projectId) return false
    if (needle && !run.query.toLowerCase().includes(needle)) return false
    return true
  })
  const key = state.sort
  return list.sort((a, b) => {
    if (key === 'confidence') return (b.confidence ?? -1) - (a.confidence ?? -1)
    return b[key] - a[key]
  })
}

export function runsInProject(state: LibraryState, projectId: string): SavedRun[] {
  return Object.values(state.runs)
    .filter((run) => run.projectId === projectId)
    .sort((a, b) => b.savedAt - a.savedAt)
}

export function projectsSorted(state: LibraryState): Project[] {
  return Object.values(state.projects).sort((a, b) => {
    if (a.pinned !== b.pinned) return a.pinned ? -1 : 1
    return b.updatedAt - a.updatedAt
  })
}
