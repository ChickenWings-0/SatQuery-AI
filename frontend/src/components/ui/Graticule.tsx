/**
 * The ground-station grid, as a layer. Pure CSS (`.graticule` in theme.css),
 * `aria-hidden`, positioned by its parent. `fade` masks it to nothing at the
 * edges so it reads as atmosphere rather than wallpaper; `major` raises the
 * hairline alpha for the one place a grid may be loud — an empty state.
 */
export function Graticule({
  fade = true,
  major = false,
  module,
  className = '',
}: {
  fade?: boolean
  major?: boolean
  /** Override the cell size, e.g. 24px for a popover header. */
  module?: number
  className?: string
}) {
  return (
    <div
      aria-hidden
      className={`graticule pointer-events-none absolute inset-0 ${fade ? 'graticule-fade' : ''} ${className}`}
      style={{
        ...(module ? { ['--grid-module' as string]: `${module}px` } : {}),
        ...(major ? { ['--color-grid' as string]: 'var(--color-grid-major)' } : {}),
      }}
    />
  )
}
