/**
 * The crosshair: two hairlines, four ticks, an open square. Drawn once as an
 * SVG so the hero's focal lock, a bounding-box preview and the notification
 * radar all share one mark. `breathe` runs the 4 s opacity loop (paused under
 * reduced motion by theme.css); `lock` plays the entry scale-in.
 */
import type { SVGProps } from 'react'

export function Reticle({
  size = 40,
  breathe = false,
  className = '',
  ...rest
}: SVGProps<SVGSVGElement> & { size?: number; breathe?: boolean }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      stroke="var(--color-reticle)"
      strokeWidth="1"
      aria-hidden="true"
      data-breathe={breathe ? '' : undefined}
      className={`pointer-events-none ${className}`}
      {...rest}
    >
      <path d="M20 0v10M20 30v10M0 20h10M30 20h10" />
      <path d="M20 13v2M20 25v2M13 20h2M25 20h2" strokeWidth="1.5" />
      <rect x="15" y="15" width="10" height="10" />
    </svg>
  )
}
