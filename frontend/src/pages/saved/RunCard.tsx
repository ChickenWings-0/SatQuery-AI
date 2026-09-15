import { BboxThumb } from '@/pages/saved/BboxThumb'
import { ExportMenu } from '@/pages/saved/ExportMenu'
import { confidenceTone } from '@/pages/saved/tone'
import { TrashIcon } from '@/components/ui/icons'
import { relativeTime } from '@/format'
import { useLibraryStore, type SavedRun } from '@/state/library'

export function RunCard({
  run,
  onOpen,
  onDelete,
  selected,
  onToggle,
}: {
  run: SavedRun
  onOpen: () => void
  onDelete: () => void
  selected: boolean
  onToggle: () => void
}) {
  const projects = useLibraryStore((state) => state.projects)
  const assign = useLibraryStore((state) => state.assign)
  const project = run.projectId ? projects[run.projectId] : null

  return (
    <article className={`card-flush lift group flex flex-col ${selected ? '!border-accent-warm' : ''}`}>
      <div className="relative">
        <BboxThumb run={run} className="aspect-[16/10]" />
        <label className="absolute top-2.5 left-2.5 flex cursor-pointer items-center">
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggle}
            aria-label={`Select ${run.query}`}
            className="size-4 accent-[var(--color-accent-warm)]"
          />
        </label>
        <span className="glass absolute right-2.5 bottom-2.5 !rounded-md px-2 py-1 text-[10.5px] font-semibold tracking-wide text-accent-warm-text">
          {run.taskType}
        </span>
        {run.confidence !== null ? (
          <span className={`glass t-coord absolute bottom-2.5 left-2.5 !rounded-md px-2 py-1 ${confidenceTone(run.confidence)}`}>
            conf {run.confidence.toFixed(2)}
          </span>
        ) : null}
      </div>
      <div className="flex flex-1 flex-col gap-2 p-4">
        <button type="button" onClick={onOpen} className="t-panel line-clamp-2 text-left hover:text-accent-warm-text">
          “{run.query}”
        </button>
        {run.headline ? (
          <p className="text-[13px] text-text-hi">
            <span className="t-mono text-[13px] font-semibold">{run.headline.value}</span>{' '}
            <span className="text-text-lo">{run.headline.label}</span>
          </p>
        ) : null}
        <p className="t-meta">
          {run.sensors.join(' · ') || 'unknown sensor'} · {run.pairType.toLowerCase().replace('_', '-')} ·{' '}
          {relativeTime(run.savedAt)}
        </p>
        <div className="mt-auto flex items-center gap-1.5 pt-1">
          <select
            aria-label="Project"
            value={run.projectId ?? ''}
            onChange={(event) => void assign([run.traceId], event.target.value || null)}
            className="h-7 min-w-0 flex-1 rounded-md border border-line bg-bg-main px-1.5 text-[11.5px] text-text-lo"
          >
            <option value="">{project ? 'No project' : 'Add to project…'}</option>
            {Object.values(projects).map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <ExportMenu run={run} compact />
          <button
            type="button"
            aria-label="Delete saved run"
            onClick={onDelete}
            className="grid size-7 place-items-center rounded-md text-text-lo hover:bg-sidebar-hi hover:text-fail"
          >
            <TrashIcon size={14} />
          </button>
        </div>
      </div>
    </article>
  )
}
