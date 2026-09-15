/**
 * A saved run's face: its thumbnail with the boxes it found stroked over it,
 * or — when the run found none — the reticle at the centre, so grounding runs
 * and change runs read differently at a glance.
 */
import { useEffect, useMemo } from 'react'

import { Graticule } from '@/components/ui/Graticule'
import { Reticle } from '@/components/ui/Reticle'
import type { SavedRun } from '@/state/library'
import { BOX_SCALE } from '@/thread/bbox'

export function BboxThumb({ run, className = '' }: { run: SavedRun; className?: string }) {
  const url = useMemo(() => (run.thumb ? URL.createObjectURL(run.thumb) : null), [run.thumb])
  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url)
  }, [url])

  return (
    <div className={`relative overflow-hidden bg-bg-main ${className}`}>
      {url ? (
        <img
          src={url}
          alt={`${run.taskType} run — “${run.query}”`}
          className="absolute inset-0 size-full object-cover [image-rendering:pixelated]"
        />
      ) : (
        <Graticule fade={false} module={24} />
      )}
      {run.boxes.length > 0 ? (
        <svg
          viewBox={`0 0 ${BOX_SCALE} ${BOX_SCALE}`}
          preserveAspectRatio="none"
          aria-hidden
          className="absolute inset-0 size-full"
        >
          {run.boxes.map((box, i) => (
            <rect
              key={i}
              x={box.xMin}
              y={box.yMin}
              width={box.xMax - box.xMin}
              height={box.yMax - box.yMin}
              fill="var(--color-bbox-fill)"
              stroke="var(--color-bbox)"
              strokeWidth="1.5"
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      ) : (
        <Reticle size={48} className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
      )}
    </div>
  )
}
