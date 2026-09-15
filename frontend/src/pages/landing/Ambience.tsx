/**
 * The hero's atmosphere: the graticule (masked to fade at the edges), two
 * glows, two reticles, and the scan-line that sweeps once on entry. All
 * `aria-hidden`, all CSS; the sequence timings live in theme.css under
 * `[data-hero]`. The large glow leans toward the pointer and drifts with the
 * scroll (`useScrollProgress`, `useCursorLight`) — it used to run an 18 s
 * idle loop as well, and two idle motions on one element was one too many.
 */
import { useEffect, useState } from 'react'

import { Graticule } from '@/components/ui/Graticule'
import { Reticle } from '@/components/ui/Reticle'

export function Ambience() {
  const [scanned, setScanned] = useState(false)
  useEffect(() => {
    const timer = window.setTimeout(() => setScanned(true), 1_000)
    return () => window.clearTimeout(timer)
  }, [])

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      <div data-hero="grid" className="absolute inset-0">
        <Graticule />
      </div>
      <div
        data-hero="glow"
        className="absolute size-[900px] rounded-full will-change-transform"
        style={{
          left: 'calc(78% - 450px)',
          top: 'calc(40% - 450px)',
          background: 'radial-gradient(circle, var(--color-glow) 0%, transparent 60%)',
        }}
      />
      <div
        className="absolute size-[420px] rounded-full opacity-50"
        style={{
          left: 'calc(12% - 210px)',
          top: 'calc(82% - 210px)',
          background: 'radial-gradient(circle, var(--color-glow) 0%, transparent 60%)',
        }}
      />
      {!scanned ? (
        <div data-hero="scan" className="absolute inset-x-0 top-0 h-px bg-surface-sand opacity-25" />
      ) : null}
      <Reticle size={56} breathe className="absolute top-[26%] right-[36%] hidden opacity-70 desk:block" />
      <Reticle size={40} breathe className="absolute top-[70%] right-[8%] hidden opacity-50 desk:block" style={{ animationDelay: '2s' }} />
    </div>
  )
}
