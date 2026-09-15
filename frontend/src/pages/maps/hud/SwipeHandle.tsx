/**
 * The split between A and B: a 2px sand line and a glass grip, draggable, a
 * `role="slider"` so `[`/`]` (via useHotkeys) and the arrow keys move it.
 */
import { useRef } from 'react'

import { SplitIcon } from '@/components/ui/icons'

export function SwipeHandle({
  split,
  onChange,
  labels,
}: {
  split: number
  onChange: (value: number) => void
  labels: [string, string]
}) {
  const dragging = useRef(false)

  function fromPointer(event: React.PointerEvent) {
    const parent = (event.currentTarget as HTMLElement).parentElement
    if (!parent) return
    const rect = parent.getBoundingClientRect()
    onChange(((event.clientX - rect.left) / rect.width) * 100)
  }

  return (
    <>
      <span
        className="t-coord glass pointer-events-none absolute top-4 -translate-x-full px-2 py-1 text-text-hi"
        style={{ left: `calc(${split}% - 8px)` }}
      >
        A · {labels[0]}
      </span>
      <span className="t-coord glass pointer-events-none absolute top-4 px-2 py-1 text-text-hi" style={{ left: `calc(${split}% + 8px)` }}>
        B · {labels[1]}
      </span>
      <div
        role="slider"
        aria-label="Compare split"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(split)}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'ArrowLeft') onChange(split - 2)
          else if (event.key === 'ArrowRight') onChange(split + 2)
          else if (event.key === 'Home') onChange(0)
          else if (event.key === 'End') onChange(100)
          else return
          event.preventDefault()
        }}
        onPointerDown={(event) => {
          dragging.current = true
          ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
          fromPointer(event)
        }}
        onPointerMove={(event) => dragging.current && fromPointer(event)}
        onPointerUp={(event) => {
          dragging.current = false
          ;(event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId)
        }}
        className="absolute inset-y-0 z-10 w-6 -translate-x-1/2 cursor-col-resize touch-none"
        style={{ left: `${split}%` }}
      >
        <span aria-hidden className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-surface-sand" />
        <span aria-hidden className="glass absolute top-1/2 left-1/2 grid size-8 -translate-x-1/2 -translate-y-1/2 place-items-center !rounded-full text-text-hi">
          <SplitIcon size={14} />
        </span>
      </div>
    </>
  )
}
