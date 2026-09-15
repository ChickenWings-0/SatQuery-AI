/**
 * HUD A — multimodal change detection. Two dates of the same place on a
 * vertical seam; the later date has grown six built-up parcels, and the
 * change mask — the CHANGE view's own mark — is drawn around them. The
 * numbers are the recording's (`landing/evidence.ts`).
 */
import { between, rng } from '@/lib/rng'
import { CHANGE } from '@/pages/landing/evidence'
import { H, Hud, LABEL, Readout, SAND, SKIP, Tag, TERRA, W, WARN } from '@/pages/landing/hud/Hud'

const SEAM = W / 2

interface Parcel {
  x: number
  y: number
  w: number
  h: number
}

function parcels(r: () => number, count: number, xLo: number, xHi: number, yLo: number, yHi: number): Parcel[] {
  const out: Parcel[] = []
  for (let i = 0; i < count; i++) {
    out.push({ x: between(r, xLo, xHi), y: between(r, yLo, yHi), w: between(r, 26, 70), h: between(r, 16, 46) })
  }
  return out
}

/** A road: three gentle bends across a half-frame. */
function road(r: () => number, dx: number, y: number): string {
  const pts = [0, 0.3, 0.65, 1].map((t) => [dx + t * SEAM, y + between(r, -28, 28)] as const)
  return pts.map(([x, py], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${py.toFixed(1)}`).join(' ')
}

/** The mask: the union of the grown parcels' outlines, as one rounded path. */
function maskPath(grown: Parcel[]): string {
  const x0 = Math.min(...grown.map((p) => p.x)) - 10
  const y0 = Math.min(...grown.map((p) => p.y)) - 10
  const x1 = Math.max(...grown.map((p) => p.x + p.w)) + 10
  const y1 = Math.max(...grown.map((p) => p.y + p.h)) + 10
  const my = (y0 + y1) / 2
  const mx = (x0 + x1) / 2
  // Six anchors, alternately pushed in, so the outline reads as measured
  // rather than as a rectangle around the parcels.
  return [
    `M${x0 + 12} ${y0}`,
    `Q${mx} ${y0 - 6} ${x1 - 12} ${y0 + 4}`,
    `Q${x1 + 4} ${my} ${x1 - 6} ${y1 - 10}`,
    `Q${mx + 20} ${y1 + 6} ${mx - 30} ${y1 - 2}`,
    `Q${x0 - 4} ${y1 - 4} ${x0 + 2} ${my + 8}`,
    `Q${x0 - 6} ${y0 + 20} ${x0 + 12} ${y0}Z`,
  ].join(' ')
}

export function ChangeHud({ seed, hasImage }: { seed: string; hasImage: boolean }) {
  const r = rng(seed)
  const fields = parcels(r, 14, 12, SEAM - 84, 30, H - 80)
  const roads = [road(r, 0, 130), road(r, 0, 290)]
  // The plan's overlay contract: the mask sits at 62–92 % × 38–66 %, where
  // the supplied .webp, if already cropped to it, still lines up.
  const grown = parcels(r, 6, SEAM + 84, W - 80, 160, 240)
  const hatch = `hud-hatch-${seed}`
  const mask = maskPath(grown)

  return (
    <Hud
      id="change"
      hasImage={hasImage}
      defs={
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke={TERRA} strokeWidth="1.4" />
        </pattern>
      }
      scene={
        <>
          {[0, SEAM].map((dx) => (
            <g key={dx}>
              {fields.map((p, i) => (
                <rect key={i} x={p.x + dx} y={p.y} width={p.w} height={p.h} rx={1.5} fill={SAND} fillOpacity={0.14} stroke={SAND} strokeOpacity={0.35} />
              ))}
              {roads.map((d, i) => (
                <path key={i} d={d} transform={`translate(${dx} 0)`} fill="none" stroke={SKIP} strokeWidth="1.5" strokeOpacity={0.7} />
              ))}
            </g>
          ))}
          {grown.map((p, i) => (
            <g key={i} className="hud-fade" style={{ ['--hud-at' as string]: '900ms' }}>
              <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={1.5} fill={TERRA} fillOpacity={0.22} />
              <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={1.5} fill={`url(#${hatch})`} stroke={TERRA} strokeOpacity={0.9} />
            </g>
          ))}
          <Tag x={SEAM - 12} y={H - 22} anchor="end">
            T₀
          </Tag>
          <Tag x={SEAM + 12} y={H - 22}>
            T₁ · Δ
          </Tag>
        </>
      }
      evidence={
        <>
          {/* The mask: the outline draws, then the fill develops. */}
          <path className="hud-fade" style={{ ['--hud-at' as string]: '960ms' }} d={mask} fill={WARN} fillOpacity={0.16} />
          <path className="hud-draw" style={{ ['--hud-at' as string]: '260ms', ['--hud-dur' as string]: '700ms' }} d={mask} fill="none" stroke={WARN} strokeWidth="1.25" pathLength={1} />
          {/* The seam and its swipe handle, arriving from the right. */}
          <g className="hud-seam">
            <line x1={SEAM} y1={0} x2={SEAM} y2={H} stroke={LABEL} strokeOpacity={0.8} strokeWidth="1.5" strokeDasharray="6 6" />
            <circle cx={SEAM} cy={H / 2} r={14} fill="var(--color-glass)" stroke="var(--color-glass-edge)" />
            <path d={`M${SEAM - 5} ${H / 2 - 4}h10M${SEAM - 5} ${H / 2}h10M${SEAM - 5} ${H / 2 + 4}h10`} stroke={LABEL} strokeWidth="1" />
          </g>
        </>
      }
    >
      <span className="t-coord glass absolute top-3 left-3 px-2 py-1 text-text-hi">A · {CHANGE.dates[0]} · TC</span>
      <span className="t-coord glass absolute top-3 right-3 px-2 py-1 text-text-hi">B · {CHANGE.dates[1]} · CHANGE</span>
      <Readout at="bl" className="flex items-center gap-2">
        <span className="t-mono font-semibold">{CHANGE.pct} %</span>
        <span className="text-text-lo">changed</span>
        <span aria-hidden className="text-text-lo">·</span>
        <span className="t-mono font-semibold">{CHANGE.km2} km²</span>
      </Readout>
      <Readout at="br" className="t-coord hidden items-center gap-3 !py-2 text-text-lo wide:flex">
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block size-2.5 rounded-[2px] border border-warn bg-warn/20" />
          change mask
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block size-2.5 rounded-[2px] border border-accent-warm bg-[repeating-linear-gradient(45deg,var(--color-accent-warm)_0_1px,transparent_1px_3px)]" />
          built-up (B)
        </span>
      </Readout>
    </Hud>
  )
}
