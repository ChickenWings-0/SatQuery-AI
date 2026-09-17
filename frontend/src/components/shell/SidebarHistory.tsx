/**
 * The rail's list of past runs, under "New Query".
 *
 * It reads the same library the Saved page does — the runs kept in IndexedDB
 * on this device — so it is the one list, not a second memory that could
 * disagree with it. Newest first, the question and how long ago; a click
 * opens the run's results in History and a trash icon removes it (with Undo,
 * the way Saved does). Nothing here is fetched from a server: a run that is
 * listed is a run that is on this disk.
 *
 * Lazy on purpose (`Sidebar.tsx` imports it with `React.lazy` when the
 * disclosure opens): it is the only thing in the shell that needs the
 * library store and its IndexedDB adapter, and the entry chunk has a budget.
 */
import { useEffect, useState } from 'react'

import { TrashIcon } from '@/components/ui/icons'
import { relativeTime } from '@/format'
import { useLibraryStore, type SavedRun } from '@/state/library'
import { toast } from '@/state/notifications'
import { useUiStore } from '@/state/ui'

const LIMIT = 12

function recent(runs: Record<string, SavedRun>): SavedRun[] {
  return Object.values(runs)
    .sort((a, b) => b.ranAt - a.ranAt)
    .slice(0, LIMIT)
}

export default function SidebarHistory() {
  const runs = useLibraryStore((state) => state.runs)
  const hydrated = useLibraryStore((state) => state.hydrated)
  const hydrate = useLibraryStore((state) => state.hydrate)
  const remove = useLibraryStore((state) => state.remove)
  const restore = useLibraryStore((state) => state.restore)
  const openRun = useUiStore((state) => state.openRun)
  const setSection = useUiStore((state) => state.setSection)
  const selectedTraceId = useUiStore((state) => state.selectedTraceId)
  const section = useUiStore((state) => state.section)

  // "3 min ago" goes stale while the rail sits open; tick once a minute.
  const [, setTick] = useState(0)
  useEffect(() => {
    void hydrate()
    const timer = setInterval(() => setTick((n) => n + 1), 60_000)
    return () => clearInterval(timer)
  }, [hydrate])

  async function del(run: SavedRun) {
    const removed = await remove(run.traceId)
    if (!removed) return
    toast('Run removed from this device.', 'info', { label: 'Undo', run: () => void restore(removed) })
  }

  if (!hydrated) {
    return (
      <ul className="space-y-1 px-1" aria-busy="true" aria-label="Loading past queries">
        {Array.from({ length: 3 }, (_, i) => (
          <li key={i} className="h-8 animate-pulse rounded-md bg-sidebar-hi" />
        ))}
      </ul>
    )
  }

  const list = recent(runs)
  if (list.length === 0) {
    return (
      <p className="px-2.5 py-1.5 text-[12px] leading-snug text-sidebar-text-lo">
        No past queries on this device yet. Save a run from the answer card and it lands here.
      </p>
    )
  }

  const total = Object.keys(runs).length
  return (
    <>
      <ul className="space-y-0.5" aria-label="Past queries">
        {list.map((run) => {
          const open = section === 'history' && selectedTraceId === run.traceId
          return (
            <li key={run.traceId} className="group relative">
              <button
                type="button"
                onClick={() => openRun(run.traceId)}
                aria-current={open ? 'true' : undefined}
                aria-label={`Open “${run.query || 'Untitled query'}”, ${relativeTime(run.ranAt)}`}
                title={run.query}
                className={`flex w-full flex-col items-start gap-0.5 rounded-md py-1.5 pr-8 pl-2.5 text-left transition-colors duration-[120ms] ${
                  open ? 'bg-sidebar-hi text-sidebar-text' : 'text-sidebar-text-lo hover:bg-sidebar-hi hover:text-sidebar-text'
                }`}
              >
                <span className="w-full truncate text-[12.5px] leading-snug">{run.query || 'Untitled query'}</span>
                <span className="flex items-center gap-1.5 text-[11px] leading-none text-sidebar-text-lo">
                  <span
                    aria-hidden
                    className={`size-1.5 rounded-full ${
                      run.outcome === 'succeeded' ? 'bg-ok' : run.outcome === 'failed' ? 'bg-fail' : 'bg-warn'
                    }`}
                  />
                  <span className="tabular">{relativeTime(run.ranAt)}</span>
                </span>
              </button>
              <button
                type="button"
                onClick={() => void del(run)}
                aria-label={`Remove “${run.query || 'Untitled query'}” from this device`}
                className="absolute top-1/2 right-1 grid size-7 -translate-y-1/2 place-items-center rounded-md text-sidebar-text-lo opacity-0 transition-opacity duration-[120ms] group-hover:opacity-100 hover:bg-fail/12 hover:text-fail focus-visible:opacity-100 [@media(hover:none)]:opacity-100"
              >
                <TrashIcon size={14} />
              </button>
            </li>
          )
        })}
      </ul>
      {total > LIMIT ? (
        <button
          type="button"
          onClick={() => setSection('saved')}
          className="mt-1 px-2.5 py-1 text-[11.5px] text-sidebar-text-lo underline-offset-2 hover:text-sidebar-text hover:underline"
        >
          All {total} saved runs
        </button>
      ) : null}
    </>
  )
}
