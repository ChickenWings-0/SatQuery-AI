/**
 * The still: the same sphere as an inline SVG — graticule ellipses, rim, one
 * satellite with a static trail. It is the Suspense fallback, the no-WebGL
 * fallback and the reduced-motion frame, so a machine that never runs the
 * shader still gets the composition rather than a spinner or an apology.
 */
export function GlobePoster({ className = '' }: { className?: string }) {
  const rings = [0.2, 0.45, 0.7, 0.9]
  return (
    <svg
      viewBox="-120 -120 240 240"
      aria-hidden
      className={`size-full ${className}`}
      fill="none"
      stroke="var(--color-reticle)"
    >
      <defs>
        <radialGradient id="sq-poster-rim" cx="50%" cy="50%" r="50%">
          <stop offset="78%" stopColor="var(--color-bg-main)" stopOpacity="0" />
          <stop offset="100%" stopColor="var(--color-surface-sand)" stopOpacity="0.28" />
        </radialGradient>
        <linearGradient id="sq-poster-trail" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="var(--color-surface-sand)" stopOpacity="0" />
          <stop offset="1" stopColor="var(--color-surface-sand)" stopOpacity="0.7" />
        </linearGradient>
      </defs>
      <circle r="100" fill="var(--color-surface-card)" stroke="none" />
      <circle r="100" fill="url(#sq-poster-rim)" stroke="none" />
      {/* meridians */}
      {rings.map((k) => (
        <ellipse key={`m${k}`} rx={100 * k} ry="100" strokeWidth="0.6" />
      ))}
      <line x1="0" y1="-100" x2="0" y2="100" strokeWidth="0.6" />
      {/* parallels */}
      {[-0.7, -0.4, 0, 0.4, 0.7].map((k) => {
        const y = 100 * k
        const rx = Math.sqrt(1 - k * k) * 100
        return <line key={`p${k}`} x1={-rx} y1={y} x2={rx} y2={y} strokeWidth={k === 0 ? 1 : 0.6} />
      })}
      <circle r="100" strokeWidth="1" stroke="var(--color-surface-sand)" strokeOpacity="0.45" />
      {/* one satellite and its trail */}
      <path d="M-118 -20 A 118 118 0 0 1 -20 -118" stroke="url(#sq-poster-trail)" strokeWidth="1.2" />
      <g transform="translate(-20 -118) rotate(-45)">
        <rect x="-2.5" y="-1.5" width="5" height="3" fill="var(--color-surface-sand)" stroke="none" />
        <rect x="-10" y="-0.8" width="6" height="1.6" fill="var(--color-accent-warm)" stroke="none" />
        <rect x="4" y="-0.8" width="6" height="1.6" fill="var(--color-accent-warm)" stroke="none" />
      </g>
    </svg>
  )
}
