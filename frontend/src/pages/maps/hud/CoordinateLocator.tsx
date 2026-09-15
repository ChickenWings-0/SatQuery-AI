/**
 * Live cursor position in DMS and the zoom; click to type a coordinate and
 * fly to it. In pixel mode (no georeference) it shows pixels and says why.
 */
import { useState } from 'react'

import { formatDms, parseCoordinate } from '@/evidence/georef'
import { CopyIcon, LocateIcon } from '@/components/ui/icons'
import { toast } from '@/state/notifications'

export function CoordinateLocator({
  cursor,
  zoom,
  onGo,
  pixelMode = false,
  pixel,
}: {
  cursor: { lon: number; lat: number } | null
  zoom: number
  onGo: (lon: number, lat: number) => void
  pixelMode?: boolean
  pixel?: { x: number; y: number } | null
}) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [bad, setBad] = useState(false)

  if (pixelMode) {
    return (
      <div className="glass t-coord flex items-center gap-2 px-3 py-2 text-text-hi">
        <LocateIcon size={13} className="text-text-lo" />
        {pixel ? `px ${Math.round(pixel.x)}, ${Math.round(pixel.y)}` : 'px —'}
        <span className="border-l border-line-soft pl-2 text-warn-text">no georeference in manifest</span>
      </div>
    )
  }

  return (
    <div className="glass flex items-center gap-2 px-2 py-1.5">
      {open ? (
        <form
          className="flex items-center gap-1.5"
          onSubmit={(event) => {
            event.preventDefault()
            const parsed = parseCoordinate(text)
            if (!parsed) {
              setBad(true)
              return
            }
            setBad(false)
            onGo(parsed[0], parsed[1])
            setOpen(false)
          }}
        >
          <input
            autoFocus
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="12.978, 77.589 or 12°58′N 77°35′E"
            aria-label="Go to coordinate"
            aria-invalid={bad}
            className={`t-coord h-7 w-56 rounded-md border bg-bg-main px-2 text-text-hi placeholder:text-text-lo focus:outline-none ${bad ? 'border-fail' : 'border-line focus:border-accent-warm'}`}
          />
          <button type="submit" className="btn-ghost-sm">
            Go
          </button>
          <button type="button" onClick={() => setOpen(false)} className="t-meta px-1 hover:text-text-hi">
            Esc
          </button>
        </form>
      ) : (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="t-coord flex items-center gap-2 px-1 py-0.5 text-text-hi hover:text-accent-warm-text"
          title="Go to a coordinate"
        >
          <LocateIcon size={13} className="text-text-lo" />
          {cursor ? formatDms(cursor.lon, cursor.lat) : '—'}
          <span className="border-l border-line-soft pl-2 text-text-lo">z {zoom.toFixed(1)}</span>
        </button>
      )}
      {cursor && !open ? (
        <button
          type="button"
          aria-label="Copy decimal degrees"
          onClick={() => {
            void navigator.clipboard
              .writeText(`${cursor.lat.toFixed(5)}, ${cursor.lon.toFixed(5)}`)
              .then(() => toast('Coordinate copied.', 'ok'))
              .catch(() => toast('Could not reach the clipboard.', 'warn'))
          }}
          className="grid size-6 place-items-center rounded-md text-text-lo hover:text-text-hi"
        >
          <CopyIcon size={12} />
        </button>
      ) : null}
    </div>
  )
}
