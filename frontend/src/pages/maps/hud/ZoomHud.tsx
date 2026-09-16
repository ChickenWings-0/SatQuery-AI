import { FitIcon, GlobeIcon, MinusIcon, PlusIcon } from '@/components/ui/icons'
import type { Projection } from '@/state/map'

export function ZoomHud({
  onIn,
  onOut,
  onFit,
  projection,
  onProjection,
}: {
  onIn: () => void
  onOut: () => void
  onFit: () => void
  projection?: Projection
  onProjection?: (next: Projection) => void
}) {
  const cls = 'grid size-9 place-items-center text-text-lo transition-colors duration-[120ms] hover:text-text-hi'
  return (
    <div className="glass flex flex-col divide-y divide-line-soft p-0" role="group" aria-label="Zoom">
      {projection && onProjection ? (
        <button
          type="button"
          role="switch"
          aria-checked={projection === 'globe'}
          aria-label="Globe projection"
          aria-keyshortcuts="B"
          title={projection === 'globe' ? 'Flat map (B)' : 'Globe (B)'}
          onClick={() => onProjection(projection === 'globe' ? 'mercator' : 'globe')}
          className={`${cls} ${projection === 'globe' ? '!text-accent-warm-text' : ''}`}
        >
          <GlobeIcon size={15} />
        </button>
      ) : null}
      <button type="button" aria-label="Zoom in" aria-keyshortcuts="+" onClick={onIn} className={cls}>
        <PlusIcon size={15} />
      </button>
      <button type="button" aria-label="Zoom out" aria-keyshortcuts="-" onClick={onOut} className={cls}>
        <MinusIcon size={15} />
      </button>
      <button type="button" aria-label="Fit to scene" aria-keyshortcuts="0" onClick={onFit} className={cls}>
        <FitIcon size={15} />
      </button>
    </div>
  )
}
