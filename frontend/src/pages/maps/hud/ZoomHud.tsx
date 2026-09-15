import { FitIcon, MinusIcon, PlusIcon } from '@/components/ui/icons'

export function ZoomHud({ onIn, onOut, onFit }: { onIn: () => void; onOut: () => void; onFit: () => void }) {
  const cls = 'grid size-9 place-items-center text-text-lo transition-colors duration-[120ms] hover:text-text-hi'
  return (
    <div className="glass flex flex-col divide-y divide-line-soft p-0" role="group" aria-label="Zoom">
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
