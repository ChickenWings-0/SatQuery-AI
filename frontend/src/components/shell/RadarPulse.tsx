/**
 * The notification centre's empty state: the reticle, two rings, and a
 * sector sweeping once every four seconds — only while the popover is open,
 * and static at 45° under reduced motion (theme.css). Not a control.
 */
import { Reticle } from '@/components/ui/Reticle'

export function RadarPulse({ size = 96 }: { size?: number }) {
  return (
    <div aria-hidden className="relative mx-auto" style={{ width: size, height: size }}>
      <svg viewBox="0 0 96 96" width={size} height={size} className="absolute inset-0" fill="none">
        <circle cx="48" cy="48" r="46" stroke="var(--color-reticle)" strokeOpacity="0.5" />
        <circle cx="48" cy="48" r="28" stroke="var(--color-reticle)" strokeOpacity="0.35" />
        <g data-sweep style={{ transformOrigin: '48px 48px' }}>
          <path
            d="M48 48 L94 48 A46 46 0 0 0 71 8.2 Z"
            fill="url(#sq-sweep)"
          />
        </g>
        <defs>
          <linearGradient id="sq-sweep" x1="48" y1="48" x2="94" y2="8" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="var(--color-reticle)" />
            <stop offset="1" stopColor="var(--color-reticle)" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>
      <Reticle size={40} className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
    </div>
  )
}
