/**
 * One investigation. The plate on top is drawn, not stocked: the family's
 * evidence plate (`FamilyPlate` — a change seam, a swipe between optical and
 * SAR, ticked bounding boxes, a land-cover mosaic), the graticule, the
 * reticle at the centroid, and the coordinate — so a row without a supplied
 * thumbnail still has a face that is *about* the investigation, and the eight
 * rows do not read as one wireframe stamped eight times. A `thumb.webp` in
 * the sample folder paints over the plate and keeps the reticle.
 */
import { useState } from 'react'

import { Graticule } from '@/components/ui/Graticule'
import { PlayIcon } from '@/components/ui/icons'
import { Reticle } from '@/components/ui/Reticle'
import { decimal } from '@/format'
import { dms, sampleUrl, type UseCase } from '@/pages/usecases/catalogue'
import { FamilyPlate } from '@/pages/usecases/FamilyPlate'
import type { SampleProgress } from '@/pages/usecases/useLoadSample'

const SENSOR_SHORT: Record<UseCase['sensor'], string> = {
  'Sentinel-2': 'S-2',
  'Sentinel-1': 'S-1',
  'VHR optical': 'VHR',
  'Sentinel-2 + Sentinel-1': 'S-2 + S-1',
}

export function UseCaseCard({
  useCase,
  onLoad,
  busy,
  progress,
  error,
  disabled,
}: {
  useCase: UseCase
  onLoad: () => void
  busy: boolean
  progress: SampleProgress | null
  error: string | null
  disabled: boolean
}) {
  const [thumbOk, setThumbOk] = useState(true)
  const [expanded, setExpanded] = useState(false)
  const pct = progress ? Math.min(100, Math.round((progress.loadedBytes / progress.totalBytes) * 100)) : 0
  const tools = expanded ? useCase.tools : useCase.tools.slice(0, 2)
  const hidden = useCase.tools.length - 2

  return (
    <article
      className={`card-flush lift group flex flex-col ${disabled ? 'pointer-events-none opacity-60' : ''}`}
      aria-labelledby={`uc-${useCase.slug}`}
    >
      <div className="relative aspect-[4/3] overflow-hidden bg-bg-main">
        {/* The plate is always underneath: it shows while the thumbnail is
            still arriving and stays when there is none, so the card never
            has a blank second. */}
        <FamilyPlate
          family={useCase.family}
          slug={useCase.slug}
          className="transition-transform duration-[var(--dur-state)] ease-[var(--ease-out-quint)] group-hover:scale-[1.03]"
        />
        {thumbOk ? (
          <img
            src={sampleUrl(useCase, 'thumb.webp')}
            alt={`${useCase.title} — ${useCase.sensor}, ${useCase.years.join(' to ')}`}
            loading="lazy"
            decoding="async"
            onError={() => setThumbOk(false)}
            className="absolute inset-0 size-full object-cover saturate-[0.85] transition-[filter] duration-[220ms] group-hover:saturate-100"
          />
        ) : null}
        <Graticule fade={false} module={32} />
        <Reticle
          size={56}
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 transition-transform duration-[220ms] ease-[var(--ease-out-quint)] group-hover:scale-90"
        />
        <span className="glass t-coord absolute top-2.5 left-2.5 !rounded-md px-2 py-1 text-text-hi">
          {dms(useCase.centroid)}
        </span>
        <span className="glass absolute top-2.5 right-2.5 !rounded-md px-2 py-1 text-[10.5px] font-semibold tracking-wide text-accent-warm-text">
          {SENSOR_SHORT[useCase.sensor]}
        </span>
      </div>

      <div className="flex flex-1 flex-col gap-2.5 p-4">
        <div>
          <h2 id={`uc-${useCase.slug}`} className="t-panel">
            {useCase.title}
          </h2>
          <p className="t-meta mt-0.5">
            {useCase.region} · {useCase.years.join(' → ')}
          </p>
        </div>

        <ul className="flex flex-wrap gap-1" aria-label="Task and tools">
          <li className="chip bg-accent-glow text-accent-warm-text">{useCase.task}</li>
          {tools.map((tool) => (
            <li key={tool} className="chip-truncate bg-line text-text-lo" title={tool}>
              {tool}
            </li>
          ))}
          {!expanded && hidden > 0 ? (
            <li>
              <button
                type="button"
                onClick={() => setExpanded(true)}
                className="chip bg-line text-text-lo hover:text-text-hi"
                aria-label={`Show ${hidden} more tools`}
              >
                +{hidden}
              </button>
            </li>
          ) : null}
        </ul>

        <p className="text-[12.5px] leading-snug text-text-hi">“{useCase.question}”</p>

        <div className="mt-auto pt-1">
          {busy && progress ? (
            <div>
              <div
                role="progressbar"
                aria-valuenow={pct}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label="Fetching sample"
                className="h-1 overflow-hidden rounded-full bg-line"
              >
                <div className="h-full rounded-full bg-accent-warm-strong transition-[width] duration-[120ms]" style={{ width: `${pct}%` }} />
              </div>
              <p className="t-coord mt-1.5 text-text-lo">
                Fetching {decimal(progress.loadedBytes / 1024 / 1024, 0)} / {useCase.sizeMb} MB
              </p>
            </div>
          ) : (
            <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
              <button type="button" onClick={onLoad} className="btn-ghost-sm flex items-center gap-1.5 whitespace-nowrap">
                <PlayIcon size={12} />
                Load into canvas
              </button>
              <span className="t-coord whitespace-nowrap text-text-lo">{useCase.sizeMb} MB</span>
            </div>
          )}
          {error ? <p className="mt-1.5 text-[11.5px] text-fail">{error}</p> : null}
        </div>
      </div>
    </article>
  )
}
