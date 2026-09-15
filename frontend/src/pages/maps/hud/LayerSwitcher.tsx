/**
 * Which rendered view is A, which (if any) is B, and whether the basemap is
 * on. The swatches are the legend colours `thread/legend.ts` already assigns
 * to view codes (see `thread/legend.ts` for the change legend); the sensor tag is what the manifest said, never a guess.
 */
import { type ViewCode, type ViewGroup } from '@/evidence/views'
import { LayersIcon } from '@/components/ui/icons'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import type { Basemap } from '@/state/map'

const SWATCH: Partial<Record<ViewCode, string>> = {
  TC: 'bg-accent-warm',
  FCIR: 'bg-fail',
  NDVI: 'bg-ok',
  NDBI: 'bg-warn',
  NDWI: 'bg-sidebar-text',
  SARFC: 'bg-text-lo',
  SARDB: 'bg-text-lo',
  SWIR: 'bg-accent-warm-strong',
  PAN: 'bg-sidebar-text-lo',
  CHANGE: 'bg-warn',
}

export function LayerSwitcher({
  groups,
  layerKey,
  compareKey,
  basemap,
  opacity,
  sensor,
  onLayer,
  onCompare,
  onBasemap,
  onOpacity,
  sheet = false,
}: {
  groups: ViewGroup[]
  layerKey: string | null
  compareKey: string | null
  basemap: Basemap
  opacity: number
  sensor: string | null
  onLayer: (key: string) => void
  onCompare: (key: string | null) => void
  onBasemap: (b: Basemap) => void
  onOpacity: (v: number) => void
  /** Phone: render inside a popover behind a button. */
  sheet?: boolean
}) {
  const body = (
    <div className="w-60">
      <p className="t-eyebrow px-1 pb-2">Layer</p>
      <div role="radiogroup" aria-label="Layer" className="space-y-0.5">
        {groups.map((group) => {
          const active = group.key === layerKey
          return (
            <button
              key={group.key}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onLayer(group.key)}
              className={`menu-row !py-1.5 ${active ? 'bg-accent-glow' : ''}`}
            >
              <span aria-hidden className={`size-2.5 rounded-full ${SWATCH[group.code] ?? 'bg-skip'}`} />
              <span className="min-w-0 flex-1 truncate">{group.name}</span>
              {sensor ? <span className="t-coord text-text-lo">{sensor}</span> : null}
            </button>
          )
        })}
      </div>
      <label className="mt-3 block px-1">
        <span className="t-eyebrow">Compare with</span>
        <select
          value={compareKey ?? ''}
          onChange={(event) => onCompare(event.target.value || null)}
          className="mt-1.5 h-8 w-full rounded-md border border-line bg-bg-main px-2 text-[12px] text-text-hi"
        >
          <option value="">Nothing — single view</option>
          {groups
            .filter((g) => g.key !== layerKey)
            .map((g) => (
              <option key={g.key} value={g.key}>
                {g.name}
              </option>
            ))}
        </select>
      </label>
      <div className="mt-3 border-t border-line-soft pt-3">
        <div className="menu-row !py-1.5">
          <span className="flex-1">Basemap</span>
          <button
            type="button"
            role="switch"
            aria-checked={basemap === 'satellite'}
            aria-label="Online satellite basemap"
            onClick={() => onBasemap(basemap === 'satellite' ? 'none' : 'satellite')}
            className="switch"
          />
        </div>
        <p className="t-meta px-2.5">needs internet · off by default for offline judging</p>
        {basemap === 'satellite' ? (
          <label className="mt-2 block px-2.5">
            <span className="t-meta">Scene opacity</span>
            <input
              type="range"
              min={0.2}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(event) => onOpacity(Number(event.target.value))}
              aria-label="Scene opacity over basemap"
              className="mt-1 w-full accent-[var(--color-accent-warm)]"
            />
          </label>
        ) : null}
      </div>
    </div>
  )

  if (sheet) {
    return (
      <Popover>
        <PopoverTrigger asChild>
          <button type="button" aria-label="Layers" className="glass grid size-9 place-items-center text-text-hi">
            <LayersIcon size={16} />
          </button>
        </PopoverTrigger>
        <PopoverContent side="top" align="start" className="p-2.5">
          {body}
        </PopoverContent>
      </Popover>
    )
  }

  return <div className="glass p-2.5">{body}</div>
}
