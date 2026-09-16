/**
 * The bottom of the discovery column: what is picked, why it cannot be a
 * pair (when it cannot), and the one button that sends it to the console.
 */
import { ArrowRightIcon } from '@/components/ui/icons'
import { bytes } from '@/format'
import { useImageryFetch } from '@/pages/maps/hud/useImageryFetch'
import { useStacStore, selectedItems } from '@/state/stac'

function shortDate(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

export function FetchBar() {
  const results = useStacStore((state) => state.results)
  const selection = useStacStore((state) => state.selection)
  const query = useStacStore((state) => state.query)
  const { t1, t2, issue } = selectedItems({ results, selection, query })
  const { run, reset, progress, busy } = useImageryFetch()
  const needsTwo = query.pair && !t2
  const ready = Boolean(t1) && !needsTwo && !issue && !busy

  const label = busy
    ? progress.phase === 'clipping'
      ? `Clipping ${progress.total === 2 ? 'two scenes' : 'the scene'} to the pipeline's window…`
      : `Fetching ${progress.done + 1} of ${progress.total} — ${bytes(progress.loadedBytes)} of ${bytes(progress.totalBytes)}`
    : query.pair
      ? 'Analyse change'
      : 'Analyse'

  return (
    <div className="border-t border-line-soft pt-3">
      <p className="t-coord min-h-4 text-text-lo">
        {t1 ? (
          <>
            <span className="text-text-hi">T1</span> {shortDate(t1.datetime)}
            {query.pair ? (
              <>
                {'  ·  '}
                <span className="text-text-hi">T2</span> {t2 ? shortDate(t2.datetime) : 'pick a second scene'}
              </>
            ) : null}
          </>
        ) : query.pair ? (
          'Pick T1 and T2 from the shelf, or click their footprints on the map.'
        ) : (
          'Pick a scene from the shelf, or click its footprint on the map.'
        )}
      </p>
      {issue ? (
        <p role="alert" className="mt-2 rounded-lg border border-warn/40 bg-warn/8 px-3 py-2 text-[12.5px]">
          {issue}
        </p>
      ) : null}
      {progress.phase === 'failed' ? (
        <p role="alert" className="mt-2 rounded-lg border border-fail/40 bg-fail/8 px-3 py-2 text-[12.5px]">
          {progress.error}{' '}
          <button type="button" onClick={reset} className="underline">
            Dismiss
          </button>
        </p>
      ) : null}
      <button
        type="button"
        disabled={!ready}
        aria-busy={busy}
        onClick={() => void run()}
        className="btn-primary mt-2.5 flex w-full items-center justify-center gap-2"
      >
        {label}
        {busy ? null : <ArrowRightIcon size={14} />}
      </button>
      {busy ? (
        <div className="mt-2 h-1 overflow-hidden rounded-full bg-line" aria-hidden>
          <div
            className="h-full rounded-full bg-accent-warm-strong transition-[width] duration-200"
            style={{ width: progress.phase === 'clipping' ? '15%' : `${15 + (progress.totalBytes ? (progress.loadedBytes / progress.totalBytes) * 85 : 0)}%` }}
          />
        </div>
      ) : null}
    </div>
  )
}
