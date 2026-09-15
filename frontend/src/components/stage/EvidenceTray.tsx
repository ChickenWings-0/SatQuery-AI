/**
 * F4b — the multi-spectral strip: one card per view the model was shown.
 *
 * Entries append as `artifact` events arrive, so evidence visibly accumulates
 * during a run. The active entry carries a terracotta edge and a glow ring:
 * `--color-accent-cool` means "selected" and nothing else in this product,
 * which is why it needs no other marker.
 *
 * Every card is the same height whether or not it carries a "pre → post" line,
 * because a ragged bottom edge across a scrolling strip reads as a rendering
 * fault rather than as varying content.
 */
import { artifactUrl } from '@/api/client'
import { countOf } from '@/format'
import { primaryOf, type ViewGroup } from '@/evidence/views'

export function EvidenceTray({
  groups,
  activeKey,
  onSelect,
  sublabel,
}: {
  groups: ViewGroup[]
  activeKey: string | null
  onSelect: (key: string) => void
  /** "Sentinel-2 · 10 m" — the scene's sensor and resolution, if known. */
  sublabel?: string | null
}) {
  if (groups.length === 0) return null

  return (
    <section className="shrink-0" aria-label="Evidence">
      <p className="t-meta">
        {countOf(groups.length, { one: 'view', other: 'views' })} the model was shown
      </p>

      <ul className="mt-2.5 flex gap-3 overflow-x-auto pb-1">
        {groups.map((group) => {
          const primary = primaryOf(group)
          const url = primary ? artifactUrl(primary) : null
          const active = group.key === activeKey
          return (
            // `sq-arrive` runs once, when React mounts a new `<li>`. During a
            // live run that is the moment the `artifact` event lands, which is
            // the product's whole claim made visible; on a re-render nothing
            // remounts, so nothing replays.
            <li key={group.key} className="sq-arrive shrink-0">
              <button
                type="button"
                onClick={() => onSelect(group.key)}
                aria-pressed={active}
                // The visible label is a short view code; the full label is the
                // exact string the VLM was shown, and `title` alone never
                // reaches touch or most screen readers.
                aria-label={primary?.label ? `${group.name} — ${primary.label}` : group.name}
                title={primary?.label}
                // `transition-colors`, not `transition-all`: the ring and
                // border are what change, and `all` invites the browser to
                // animate layout properties nobody asked it to.
                className={`flex h-full w-[160px] flex-col overflow-hidden rounded-xl border bg-surface-card text-left transition-colors ${
                  active
                    ? 'border-accent-warm ring-2 ring-accent-glow'
                    : 'border-line hover:border-accent-warm/40'
                }`}
              >
                {url && (
                  <img
                    src={url}
                    alt={primary?.label ?? group.name}
                    loading="lazy"
                    className="aspect-[4/3] w-full border-b border-line-soft object-cover [image-rendering:pixelated]"
                  />
                )}
                <span className="flex flex-1 flex-col justify-center px-3 py-2.5">
                  <span className="truncate text-[12.5px] font-medium">{group.name}</span>
                  <span className="mt-1 flex items-center gap-1.5">
                    <span className="truncate text-[10.5px] text-text-lo">
                      {sublabel ?? group.code}
                    </span>
                    {group.comparable && (
                      <span className="chip ml-auto shrink-0 bg-accent-cool !text-[9.5px] text-on-accent-cool">
                        pre → post
                      </span>
                    )}
                  </span>
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
