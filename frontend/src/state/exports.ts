/**
 * Keyboard-triggered exports.
 *
 * `⌘⇧G` and `⌘⇧S` are dispatched by `useHotkeys`, which knows nothing about
 * which panel currently owns the export button. The store is a counter per
 * export kind; the mounted button subscribes and runs its own handler when
 * the counter moves. Nothing mounted means nothing happens, and the hotkey
 * layer says so.
 */
import { useEffect, useRef } from 'react'
import { create } from 'zustand'

export type ExportKind = 'geojson' | 'sitrep'

interface ExportState {
  requests: Record<ExportKind, number>
  handlers: Record<ExportKind, number>
  request: (kind: ExportKind) => boolean
  attach: (kind: ExportKind) => () => void
}

export const useExportStore = create<ExportState>((set, get) => ({
  requests: { geojson: 0, sitrep: 0 },
  handlers: { geojson: 0, sitrep: 0 },
  request: (kind) => {
    if (get().handlers[kind] === 0) return false
    set((state) => ({ requests: { ...state.requests, [kind]: state.requests[kind] + 1 } }))
    return true
  },
  attach: (kind) => {
    set((state) => ({ handlers: { ...state.handlers, [kind]: state.handlers[kind] + 1 } }))
    return () =>
      set((state) => ({ handlers: { ...state.handlers, [kind]: Math.max(0, state.handlers[kind] - 1) } }))
  },
}))

/** Run `handler` whenever the shortcut for `kind` fires while this component is mounted. */
export function useExportRequest(kind: ExportKind, handler: () => void): void {
  const latest = useRef(handler)
  useEffect(() => {
    latest.current = handler
  })
  useEffect(() => useExportStore.getState().attach(kind), [kind])
  useEffect(() => {
    // Subscribe outside render: only a *change* after mount runs the handler,
    // so a component mounting after a stale request does not fire it.
    let seen = useExportStore.getState().requests[kind]
    return useExportStore.subscribe((state) => {
      const count = state.requests[kind]
      if (count === seen) return
      seen = count
      latest.current()
    })
  }, [kind])
}
