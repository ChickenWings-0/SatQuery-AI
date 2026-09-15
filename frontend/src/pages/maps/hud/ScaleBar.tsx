import { metersPerPixel } from '@/evidence/georef'

const NICE = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

export function ScaleBar({ zoom, lat, gsdM }: { zoom: number; lat: number; gsdM: number | null }) {
  const mpp = metersPerPixel(zoom, lat)
  const target = mpp * 120
  const metres = NICE.find((n) => n >= target) ?? NICE[NICE.length - 1]!
  const px = Math.round(metres / mpp)
  const label = metres >= 1000 ? `${metres / 1000} km` : `${metres} m`
  return (
    <div className="glass flex items-center gap-3 px-3 py-2" aria-label={`Scale: ${label}`}>
      <span className="flex flex-col gap-1">
        <span className="block h-0.5 rounded-full bg-text-hi" style={{ width: px }} />
        <span className="t-coord text-text-hi">{label}</span>
      </span>
      {gsdM !== null ? (
        <span className="t-coord border-l border-line-soft pl-3 text-text-lo">{gsdM} m/px sensor</span>
      ) : null}
    </div>
  )
}
