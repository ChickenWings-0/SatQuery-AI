import { BboxThumb } from '@/pages/saved/BboxThumb'
import { ExportMenu } from '@/pages/saved/ExportMenu'
import { confidenceTone } from '@/pages/saved/tone'
import { TrashIcon } from '@/components/ui/icons'
import { relativeTime } from '@/format'
import type { SavedRun } from '@/state/library'

export function RunRow({
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
  return (
    <tr
      className={`h-10 border-b border-line-soft text-[12.5px] transition-colors duration-[120ms] hover:bg-sidebar-hi ${selected ? 'bg-accent-glow' : ''}`}
      onKeyDown={(event) => {
        if (event.key === 'Enter' && event.target === event.currentTarget) onOpen()
      }}
      tabIndex={0}
    >
      <td className="w-8 pl-2">
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          aria-label={`Select ${run.query}`}
          className="size-3.5 accent-[var(--color-accent-warm)]"
        />
      </td>
      <td className="w-14 py-1">
        <BboxThumb run={run} className="h-8 w-12 rounded" />
      </td>
      <td className="max-w-0 pr-3">
        <button type="button" onClick={onOpen} className="block w-full truncate text-left text-text-hi hover:text-accent-warm-text">
          {run.query}
        </button>
      </td>
      <td className="pr-3">
        <span className="chip bg-accent-glow text-accent-warm-text">{run.taskType}</span>
      </td>
      <td className="hidden pr-3 text-text-lo wide:table-cell">{run.sensors.join(' · ') || '—'}</td>
      <td className={`t-mono pr-3 ${confidenceTone(run.confidence)}`}>
        {run.confidence === null ? '—' : run.confidence.toFixed(2)}
      </td>
      <td className="t-mono hidden pr-3 text-text-lo desk:table-cell">{relativeTime(run.savedAt)}</td>
      <td className="w-16 pr-2">
        <div className="flex items-center justify-end gap-0.5">
          <ExportMenu run={run} compact />
          <button
            type="button"
            aria-label="Delete saved run"
            onClick={onDelete}
            className="grid size-7 place-items-center rounded-md text-text-lo hover:text-fail"
          >
            <TrashIcon size={14} />
          </button>
        </div>
      </td>
    </tr>
  )
}
