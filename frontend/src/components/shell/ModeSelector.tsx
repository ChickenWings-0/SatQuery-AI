/**
 * The analysis-mode chip in the top bar. "Earth View" is the only mode today,
 * so the chip is scaffolding: it carries the ARIA of a menu button and no
 * menu, which is the honest shape for a control whose options arrive later.
 * When mode switching lands, this is the one file that grows.
 */
import { ChevronDownIcon, MapsIcon } from '@/components/ui/icons'

export function ModeSelector() {
  return (
    <button
      type="button"
      aria-haspopup="menu"
      aria-expanded={false}
      aria-label="Analysis mode: Earth View"
      onClick={() => {
        /* Mode switching is not implemented yet. */
      }}
      className="flex h-9 shrink-0 items-center gap-2 rounded-full border border-line bg-surface-card pr-2.5 pl-3 text-[13px] font-medium text-text-hi transition-colors hover:border-accent-warm/40"
    >
      <MapsIcon size={16} className="text-accent-warm-text" />
      <span>Earth View</span>
      <ChevronDownIcon size={14} className="text-text-lo" />
    </button>
  )
}
