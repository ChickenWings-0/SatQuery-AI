/**
 * The SITREP control in the answer card's action row. It belongs to the
 * answer, not the panel: a brief is the answer, folded for the room.
 */
import { SitrepIcon } from '@/components/ui/icons'
import { useSitrep } from '@/export/sitrep/useSitrep'
import { useExportRequest } from '@/state/exports'

export function SitrepAction() {
  const { run, busy } = useSitrep()
  useExportRequest('sitrep', () => void run({ kind: 'live' }))
  const label = busy ? 'Building the SITREP…' : 'SITREP: one-page brief as PDF'
  return (
    <button
      type="button"
      aria-label={label}
      title={busy ? label : 'SITREP — one-page brief as PDF (⌘⇧S)'}
      aria-keyshortcuts="Meta+Shift+S Control+Shift+S"
      aria-busy={busy}
      disabled={busy}
      onClick={() => void run({ kind: 'live' })}
      className="grid size-7 place-items-center rounded-md text-text-lo transition-colors hover:bg-accent-glow hover:text-text-hi disabled:opacity-60"
    >
      <SitrepIcon size={14} className={busy ? 'animate-pulse' : undefined} />
      {busy && (
        <span role="status" className="sr-only">
          Building the SITREP
        </span>
      )}
    </button>
  )
}
