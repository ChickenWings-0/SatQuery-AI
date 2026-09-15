/**
 * The shared list: cards or rows, search, sort, bulk bar. Used by the Saved
 * page and, filtered to one project, by the project detail page.
 */
import { useEffect, useMemo, useState } from 'react'

import { boxesToGeoJson, downloadJson } from '@/export/geojson'
import { Graticule } from '@/components/ui/Graticule'
import { GridIcon, RowsIcon, SearchIcon, UndoIcon } from '@/components/ui/icons'
import { Reticle } from '@/components/ui/Reticle'
import { Segmented } from '@/components/ui/Segmented'
import { CardSkeleton } from '@/components/ui/Skeleton'
import { RunCard } from '@/pages/saved/RunCard'
import { RunRow } from '@/pages/saved/RunRow'
import { useLibrary } from '@/pages/saved/useLibrary'
import { toast } from '@/state/notifications'
import {
  useLibraryStore,
  visibleRuns,
  type LibrarySort,
  type LibraryView,
  type SavedRun,
} from '@/state/library'
import { useUiStore } from '@/state/ui'

export function SavedList({
  projectId = null,
  emptyTitle = 'Nothing saved yet.',
  emptyBody = 'Runs you keep will appear here — with their boxes, numbers and trace.',
}: {
  projectId?: string | null
  emptyTitle?: string
  emptyBody?: string
}) {
  const hydrated = useLibrary()
  const runs = useLibraryStore((state) => state.runs)
  const view = useLibraryStore((state) => state.view)
  const setView = useLibraryStore((state) => state.setView)
  const sort = useLibraryStore((state) => state.sort)
  const setSort = useLibraryStore((state) => state.setSort)
  const filter = useLibraryStore((state) => state.filter)
  const setFilter = useLibraryStore((state) => state.setFilter)
  const selected = useLibraryStore((state) => state.selected)
  const toggleSelect = useLibraryStore((state) => state.toggleSelect)
  const clearSelection = useLibraryStore((state) => state.clearSelection)
  const remove = useLibraryStore((state) => state.remove)
  const restore = useLibraryStore((state) => state.restore)
  const assign = useLibraryStore((state) => state.assign)
  const projects = useLibraryStore((state) => state.projects)
  const openRun = useUiStore((state) => state.openRun)
  const setSection = useUiStore((state) => state.setSection)
  const [text, setText] = useState(filter.text)

  useEffect(() => {
    setFilter({ projectId })
    return () => setFilter({ projectId: null })
  }, [projectId, setFilter])

  useEffect(() => {
    const timer = window.setTimeout(() => setFilter({ text }), 150)
    return () => window.clearTimeout(timer)
  }, [text, setFilter])

  const list = useMemo(
    () => visibleRuns({ ...useLibraryStore.getState(), runs, sort, filter }),
    [runs, sort, filter],
  )
  const total = Object.values(runs).filter((r) => !projectId || r.projectId === projectId).length

  async function del(run: SavedRun) {
    const removed = await remove(run.traceId)
    if (!removed) return
    toast('Run removed from this device.', 'info', { label: 'Undo', run: () => void restore(removed) })
  }

  function exportSelected() {
    const chosen = selected.map((id) => runs[id]).filter((r): r is SavedRun => Boolean(r))
    const features = chosen.flatMap((run) =>
      boxesToGeoJson(run.boxes, run.bounds, { traceId: run.traceId, query: run.query }).features,
    )
    downloadJson(`satquery-${chosen.length}-runs.geojson`, { type: 'FeatureCollection', features })
    toast(`${chosen.length} runs exported.`, 'ok')
  }

  if (!hydrated) {
    return (
      <div className="grid grid-cols-1 gap-4 wide:grid-cols-2 desk:grid-cols-3" aria-busy="true">
        {Array.from({ length: 6 }, (_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    )
  }

  if (total === 0) {
    return (
      <div className="relative overflow-hidden rounded-xl border border-line px-6 py-20 text-center">
        <Graticule major />
        <Reticle size={72} breathe className="absolute top-8 left-[22%]" />
        <Reticle size={72} breathe className="absolute right-[18%] bottom-10" style={{ animationDelay: '1.3s' }} />
        <div className="relative mx-auto max-w-sm">
          <p className="t-page">{emptyTitle}</p>
          <p className="t-meta mt-2">{emptyBody}</p>
          <div className="mt-6 flex flex-wrap justify-center gap-2.5">
            <button type="button" onClick={() => setSection('usecases')} className="btn-primary">
              Run a use case
            </button>
            <button type="button" onClick={() => setSection('explore')} className="btn-ghost">
              Open the console
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="relative">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <label className="flex h-8 min-w-0 flex-1 items-center gap-2 rounded-lg border border-line bg-surface-card px-2.5 focus-within:border-accent-warm">
          <SearchIcon size={14} className="shrink-0 text-text-lo" />
          <input
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Search saved questions"
            aria-label="Search saved questions"
            className="min-w-0 flex-1 bg-transparent text-[12.5px] text-text-hi placeholder:text-text-lo focus:outline-none"
          />
        </label>
        <select
          aria-label="Sort"
          value={sort}
          onChange={(event) => setSort(event.target.value as LibrarySort)}
          className="h-8 rounded-lg border border-line bg-surface-card px-2 text-[12px] text-text-lo"
        >
          <option value="savedAt">Newest saved</option>
          <option value="ranAt">Newest run</option>
          <option value="confidence">Highest confidence</option>
        </select>
        <Segmented<LibraryView>
          size="sm"
          label="Layout"
          value={view}
          onChange={setView}
          options={[
            { value: 'cards', label: <GridIcon size={13} />, title: 'Cards' },
            { value: 'rows', label: <RowsIcon size={13} />, title: 'Rows' },
          ]}
        />
      </div>

      {list.length === 0 ? (
        <p className="t-meta py-10 text-center">No saved run matches “{filter.text}”.</p>
      ) : view === 'cards' ? (
        <div className="grid grid-cols-1 gap-4 wide:grid-cols-2 desk:grid-cols-3">
          {list.map((run) => (
            <RunCard
              key={run.traceId}
              run={run}
              onOpen={() => openRun(run.traceId)}
              onDelete={() => void del(run)}
              selected={selected.includes(run.traceId)}
              onToggle={() => toggleSelect(run.traceId)}
            />
          ))}
        </div>
      ) : (
        <div className="card-flush overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="t-eyebrow h-9 border-b border-line text-left">
                <th className="pl-2" scope="col">
                  <span className="sr-only">Select</span>
                </th>
                <th scope="col">
                  <span className="sr-only">Preview</span>
                </th>
                <th scope="col">Question</th>
                <th scope="col">Task</th>
                <th scope="col" className="hidden wide:table-cell">
                  Sensors
                </th>
                <th scope="col">Conf.</th>
                <th scope="col" className="hidden desk:table-cell">
                  Saved
                </th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {list.map((run) => (
                <RunRow
                  key={run.traceId}
                  run={run}
                  onOpen={() => openRun(run.traceId)}
                  onDelete={() => void del(run)}
                  selected={selected.includes(run.traceId)}
                  onToggle={() => toggleSelect(run.traceId)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected.length > 0 ? (
        <div className="glass sq-arrive sticky bottom-3 mt-4 flex flex-wrap items-center gap-2 px-3 py-2 text-[12.5px]">
          <span className="font-medium text-text-hi">{selected.length} selected</span>
          <select
            aria-label="Add selected to project"
            defaultValue=""
            onChange={(event) => {
              if (event.target.value) void assign(selected, event.target.value)
            }}
            className="h-7 rounded-md border border-line bg-bg-main px-1.5 text-[11.5px] text-text-lo"
          >
            <option value="">Add to project…</option>
            {Object.values(projects).map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <button type="button" onClick={exportSelected} className="btn-ghost-sm">
            Export GeoJSON
          </button>
          <button
            type="button"
            onClick={() => {
              const chosen = selected.map((id) => runs[id]).filter((r): r is SavedRun => Boolean(r))
              void Promise.all(chosen.map((run) => remove(run.traceId))).then(() =>
                toast(`${chosen.length} runs removed.`, 'info', {
                  label: 'Undo',
                  run: () => void Promise.all(chosen.map((run) => restore(run))),
                }),
              )
            }}
            className="btn-ghost-sm !border-fail !text-fail hover:!bg-fail hover:!text-white"
          >
            Delete
          </button>
          <button type="button" onClick={clearSelection} className="t-meta ml-auto flex items-center gap-1 hover:text-text-hi">
            <UndoIcon size={12} />
            Clear
          </button>
        </div>
      ) : null}
    </div>
  )
}
