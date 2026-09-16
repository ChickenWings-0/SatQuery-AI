/**
 * The scenes that matched, newest first. Each card carries what a picker
 * needs to choose well: the date, cloud cover for optical, orbit direction
 * for SAR (a T1/T2 SAR pair must share one), and the thumbnail. Hover and
 * selection are shared with the map through the store, so the footprint
 * lights up when the card does.
 */
import { CheckIcon } from '@/components/ui/icons'
import { specOf } from '@/geo/collections'
import type { StacItem } from '@/geo/stac'
import { useStacStore, type Slot } from '@/state/stac'

function dateOf(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

function SlotBadge({ slot }: { slot: Slot }) {
  return (
    <span className="inline-flex h-5 items-center rounded-md bg-accent-cool px-1.5 font-mono text-[10.5px] font-semibold text-white">
      {slot.toUpperCase()}
    </span>
  )
}

function Card({ item }: { item: StacItem }) {
  const t1 = useStacStore((state) => state.selection.t1)
  const t2 = useStacStore((state) => state.selection.t2)
  const pair = useStacStore((state) => state.query.pair)
  const hovered = useStacStore((state) => state.hover === item.id)
  const pick = useStacStore((state) => state.pick)
  const select = useStacStore((state) => state.select)
  const setHover = useStacStore((state) => state.setHover)
  const slot: Slot | null = t1 === item.id ? 't1' : t2 === item.id ? 't2' : null
  const spec = specOf(item.collection)

  return (
    <li
      onMouseEnter={() => setHover(item.id)}
      onMouseLeave={() => setHover(null)}
      className={`relative flex gap-3 rounded-lg border p-2 transition-colors duration-[120ms] ${
        slot ? 'border-accent-cool bg-accent-glow' : hovered ? 'border-line bg-sidebar-hi' : 'border-line-soft'
      }`}
    >
      <button
        type="button"
        onClick={() => pick(item.id)}
        aria-pressed={slot !== null}
        aria-label={`${slot ? 'Deselect' : 'Use'} ${spec?.label ?? item.collection} scene from ${dateOf(item.datetime)}`}
        className="relative size-16 shrink-0 overflow-hidden rounded-md bg-bg-main"
      >
        {item.thumbHref ? (
          <img src={item.thumbHref} alt="" aria-hidden="true" loading="lazy" decoding="async" className="size-full object-cover" />
        ) : (
          <span className="grid size-full place-items-center font-mono text-[10px] text-text-lo">{spec?.short ?? '?'}</span>
        )}
        {slot ? (
          <span className="absolute inset-0 grid place-items-center bg-accent-cool/40 text-white">
            <CheckIcon size={18} />
          </span>
        ) : null}
      </button>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-[12.5px] font-medium text-text-hi">{dateOf(item.datetime)}</span>
          <span className="t-coord text-text-lo">{spec?.short ?? item.collection}</span>
          {slot ? <SlotBadge slot={slot} /> : null}
        </div>
        <p className="t-coord mt-1 text-text-lo">
          {item.sensor === 'optical'
            ? `${item.cloud === null ? '—' : `${item.cloud.toFixed(0)} % cloud`}${item.tile ? ` · ${item.tile}` : ''}`
            : `${item.orbitState ?? 'orbit —'}${item.relativeOrbit !== null ? ` · rel ${item.relativeOrbit}` : ''}${item.polarisations ? ` · ${item.polarisations.join('+')}` : ''}`}
        </p>
        {pair ? (
          <div className="mt-1.5 flex gap-1">
            {(['t1', 't2'] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => select(s, slot === s ? null : item.id)}
                aria-pressed={slot === s}
                className={`chip border ${slot === s ? 'border-accent-cool bg-accent-cool text-white' : 'border-line hover:border-accent-warm hover:text-text-hi'}`}
              >
                Use as {s.toUpperCase()}
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </li>
  )
}

export function SceneShelf() {
  const results = useStacStore((state) => state.results)
  const status = useStacStore((state) => state.status)
  const error = useStacStore((state) => state.error)
  const search = useStacStore((state) => state.search)
  const bbox = useStacStore((state) => state.query.bbox)

  return (
    <section aria-label="Matching scenes" className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2">
        <p className="t-meta">
          {status === 'searching' ? 'Searching the catalogue…' : results.length > 0 ? `${results.length} scenes, newest first` : bbox ? 'Scenes' : 'Pick a place to search'}
        </p>
        {status === 'offline' ? (
          <span className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-warn/40 bg-warn/8 px-1.5 py-0.5 text-[11px] text-warn-text">
            offline
            <button type="button" onClick={() => void search()} className="underline">
              Retry
            </button>
          </span>
        ) : null}
      </div>

      {status === 'offline' && results.length === 0 ? (
        <p role="status" className="mt-2 rounded-lg border border-warn/40 bg-warn/8 px-3 py-2 text-[12.5px]">
          {error}
        </p>
      ) : null}
      {status === 'error' ? (
        <p role="alert" className="mt-2 rounded-lg border border-fail/40 bg-fail/8 px-3 py-2 text-[12.5px]">
          {error}{' '}
          <button type="button" onClick={() => void search()} className="underline">
            Try again
          </button>
        </p>
      ) : null}
      {status === 'empty' ? (
        <p className="mt-2 text-[12.5px] text-text-lo">No scenes in this window. Widen the dates or raise the cloud limit.</p>
      ) : null}

      {results.length > 0 ? (
        <ul className="mt-2 min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-1" aria-busy={status === 'searching'}>
          {results.map((item) => (
            <Card key={item.id} item={item} />
          ))}
        </ul>
      ) : null}
    </section>
  )
}
