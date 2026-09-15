/**
 * The shared grammar of the three capability HUDs. One `viewBox`, one layer
 * order — ground, scene, graticule, evidence, readouts, frame — one glow
 * filter that the evidence layer alone may use, and one scan-line that sweeps
 * once when the HUD becomes active.
 *
 * The scene layer is the drawn stand-in for imagery; when a real
 * `public/samples/capabilities/<id>.webp` loads it is hidden (`data-has-image`)
 * and every other layer stays. So the HUD is never a fallback that vanishes:
 * it is the annotation the real imagery gets.
 *
 * Colours are theme tokens read through CSS variables — the contrast suite
 * knows every one of them. Gradient stops name their colour explicitly:
 * `currentColor` does not resolve from a `<stop>` (see FamilyPlate).
 */
import type { ReactNode } from 'react'

export const W = 640
export const H = 400

export const SAND = 'var(--color-evidence)'
export const TERRA = 'var(--color-accent-warm)'
export const LABEL = 'var(--color-accent-warm-text)'
export const OK = 'var(--color-ok)'
export const WARN = 'var(--color-warn)'
export const FAIL = 'var(--color-fail)'
export const SKIP = 'var(--color-skip)'
export const BBOX = 'var(--color-bbox)'
export const BBOX_FILL = 'var(--color-bbox-fill)'
export const BBOX_LABEL = 'var(--color-bbox-label)'
export const GRID = 'var(--color-grid)'
export const GRID_MAJOR = 'var(--color-grid-major)'
export const HI = 'var(--color-text-hi)'

/** Corner ticks: the reticle's own mark, at the four corners of the frame. */
function Frame() {
  const t = 10
  const m = 5
  const d = `M${m} ${m + t}v-${t}h${t} M${W - m - t} ${m}h${t}v${t} M${m} ${H - m - t}v${t}h${t} M${W - m} ${H - m - t}v${t}h-${t}`
  return (
    <g fill="none" stroke="var(--color-reticle)" strokeWidth="1">
      <path d={d} />
      <rect x={0.5} y={0.5} width={W - 1} height={H - 1} stroke="var(--color-glass-edge)" />
    </g>
  )
}

/** A mono readout inside the SVG, for the labels that belong to the drawing. */
export function Tag({
  x,
  y,
  anchor = 'start',
  fill = LABEL,
  children,
}: {
  x: number
  y: number
  anchor?: 'start' | 'middle' | 'end'
  fill?: string
  children: string
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      fill={fill}
      style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '0.1em', fontWeight: 500 }}
    >
      {children}
    </text>
  )
}

export function Hud({
  id,
  hasImage,
  scene,
  evidence,
  defs,
  children,
}: {
  id: string
  hasImage: boolean
  /** The drawn imagery stand-in; hidden under a real image. */
  scene: ReactNode
  /** Masks, boxes, seams — the layer allowed to glow. */
  evidence: ReactNode
  defs?: ReactNode
  /** HTML readouts, positioned over the drawing. */
  children?: ReactNode
}) {
  const glow = `hud-glow-${id}`
  return (
    <div data-hud={id} data-has-image={hasImage ? '' : undefined} className="absolute inset-0">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice" className="absolute inset-0 size-full" aria-hidden>
        <defs>
          <filter id={glow} x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <pattern id={`hud-grid-${id}`} width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M40 0H0V40" fill="none" stroke={GRID} strokeWidth="1" />
          </pattern>
          {defs}
        </defs>
        <rect width={W} height={H} fill="var(--color-bg-main)" />
        <g className="hud-scene">{scene}</g>
        <rect width={W} height={H} fill={`url(#hud-grid-${id})`} />
        <g filter={`url(#${glow})`}>{evidence}</g>
        {/* One sweep, top to bottom, on activation. */}
        <rect className="hud-scan" x={0} y={-2} width={W} height={1} fill={SAND} opacity={0.25} />
        <Frame />
      </svg>
      {children}
    </div>
  )
}

/** A glass readout chip in a corner of the HUD. The bottom-right one is the
 * legend or the coordinate — secondary, and hidden on a phone where it would
 * sit on top of the bottom-left figure. */
export function Readout({
  at,
  className = '',
  children,
}: {
  at: 'tl' | 'tr' | 'bl' | 'br'
  className?: string
  children: ReactNode
}) {
  const pos = { tl: 'top-3 left-3', tr: 'top-3 right-3', bl: 'bottom-3 left-3', br: 'bottom-3 right-3' }[at]
  return <span className={`hud-fade glass absolute ${pos} px-2.5 py-1.5 text-[12px] text-text-hi ${className}`}>{children}</span>
}
