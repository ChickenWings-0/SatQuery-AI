/**
 * A project's card: its AOI drawn as a dashed footprint on the graticule, a
 * stack of member thumbnails, and the run tally with the status vocabulary.
 */
import { useShallow } from 'zustand/shallow'

import { BboxThumb } from '@/pages/saved/BboxThumb'
import { PinIcon, TrashIcon } from '@/components/ui/icons'
import { Graticule } from '@/components/ui/Graticule'
import { relativeTime } from '@/format'
import { COLOUR } from '@/pages/projects/colours'
import { useLibraryStore, runsInProject, type Project } from '@/state/library'

function Footprint({ aoi }: { aoi: Project['aoi'] }) {
  if (!aoi) return null
  const [w, s, e, n] = aoi
  // Fit the box into the plate with a margin; aspect from the bounds.
  const width = Math.max(e - w, 1e-6)
  const height = Math.max(n - s, 1e-6)
  const scale = Math.min(60 / width, 40 / height)
  const bw = width * scale
  const bh = height * scale
  return (
    <svg viewBox="0 0 100 62.5" aria-hidden className="absolute inset-0 size-full">
      <rect
        x={50 - bw / 2}
        y={31.25 - bh / 2}
        width={bw}
        height={bh}
        fill="var(--color-bbox-fill)"
        stroke="var(--color-surface-sand)"
        strokeWidth="0.8"
        strokeDasharray="2 1.5"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}

export function ProjectCard({
  project,
  onOpen,
  onDelete,
}: {
  project: Project
  onOpen: () => void
  onDelete: () => void
}) {
  const runs = useLibraryStore(useShallow((state) => runsInProject(state, project.id)))
  const updateProject = useLibraryStore((state) => state.updateProject)
  const ok = runs.filter((r) => r.outcome === 'succeeded').length
  const failed = runs.filter((r) => r.outcome === 'failed').length

  return (
    <article className="card-flush lift group flex flex-col" aria-labelledby={`pr-${project.id}`}>
      <button type="button" onClick={onOpen} className="relative block aspect-[16/10] w-full overflow-hidden bg-bg-main text-left">
        <Graticule fade={false} module={32} />
        <Footprint aoi={project.aoi} />
        {runs.length > 0 ? (
          <div className="absolute top-2.5 right-2.5 flex -space-x-3">
            {runs.slice(0, 4).map((run) => (
              <BboxThumb key={run.traceId} run={run} className="size-10 rounded-md border border-line-soft shadow-[var(--shadow-lift)]" />
            ))}
            {runs.length > 4 ? (
              <span className="glass t-coord grid size-10 place-items-center !rounded-md text-text-hi">
                +{runs.length - 4}
              </span>
            ) : null}
          </div>
        ) : (
          <span className="t-coord absolute bottom-2.5 left-2.5 text-text-lo">no runs filed yet</span>
        )}
        <span aria-hidden className={`absolute bottom-0 left-0 h-0.5 w-10 ${COLOUR[project.colour]}`} />
      </button>
      <div className="flex flex-1 flex-col gap-1.5 p-4">
        <div className="flex items-start gap-2">
          <h2 id={`pr-${project.id}`} className="t-panel min-w-0 flex-1 truncate">
            {project.name}
          </h2>
          <button
            type="button"
            aria-label={project.pinned ? 'Unpin project' : 'Pin project'}
            aria-pressed={project.pinned}
            onClick={() => void updateProject(project.id, { pinned: !project.pinned })}
            className={`grid size-6 shrink-0 place-items-center rounded-md ${project.pinned ? 'text-accent-warm-text' : 'text-text-lo opacity-0 group-focus-within:opacity-100 group-hover:opacity-100'}`}
          >
            <PinIcon size={13} />
          </button>
        </div>
        <p className="t-meta">
          {runs.length} {runs.length === 1 ? 'run' : 'runs'}
          {ok > 0 ? ` · ${ok} succeeded` : ''}
          {failed > 0 ? ` · ${failed} failed` : ''}
        </p>
        <p className="t-meta">updated {relativeTime(project.updatedAt)}</p>
        <div className="mt-auto flex items-center gap-1.5 pt-2">
          <button type="button" onClick={onOpen} className="btn-ghost-sm">
            Open
          </button>
          <button
            type="button"
            aria-label="Delete project"
            onClick={onDelete}
            className="ml-auto grid size-7 place-items-center rounded-md text-text-lo hover:bg-sidebar-hi hover:text-fail"
          >
            <TrashIcon size={14} />
          </button>
        </div>
      </div>
    </article>
  )
}
