/**
 * HUD B — SAR / optical consistency. A diagonal seam: reflectance on the
 * left (soft spectral blobs, two clouds), backscatter on the right (speckle,
 * bright streaks). Two regions cross the seam; the physics rules agree on
 * one and not the other — and the disagreement *is* the cloud, which is the
 * whole point of asking SAR.
 *
 * Nothing here was measured: the verdicts are declared synthetic in
 * `landing/evidence.ts` and the readout counts the drawn regions.
 */
import { between, rng } from '@/lib/rng'
import { CROSSMODAL } from '@/pages/landing/evidence'
import { FAIL, H, HI, Hud, LABEL, OK, Readout, SAND, SKIP, W } from '@/pages/landing/hud/Hud'

// The seam runs from (520, 0) to (360, 400): steep enough that the
// disagreeing region straddles it, with the cloud on its optical side.
const seamX = (y: number) => 520 - (y / H) * 160

// Overlay contract: regions at 18 % and 66 % of the width, 30–52 % tall.
const REGIONS = [
  { x: 0.18 * W, y: 0.3 * H, w: 0.2 * W, h: 0.22 * H },
  { x: 0.66 * W, y: 0.3 * H, w: 0.2 * W, h: 0.22 * H },
] as const

export function CrossModalHud({ seed, hasImage }: { seed: string; hasImage: boolean }) {
  const r = rng(seed)
  const blobs = Array.from({ length: 9 }, () => ({
    cx: between(r, 30, 440),
    cy: between(r, 40, H - 40),
    rx: between(r, 34, 78),
    ry: between(r, 22, 48),
    veg: r() > 0.45,
  }))
  const clouds = [
    // Inside region 2, on the optical side of the seam: the disagreement.
    { cx: 0.7 * W, cy: 0.41 * H, rx: 44, ry: 28 },
    { cx: between(r, 80, 200), cy: between(r, 260, 340), rx: between(r, 40, 60), ry: between(r, 22, 32) },
  ]
  const streaks = Array.from({ length: 14 }, () => ({
    x: between(r, 380, W - 60),
    y: between(r, 20, H - 20),
    len: between(r, 18, 56),
  }))
  const dots = [0.18, 0.42, 0.63, 0.84].map((t, i) => ({ y: t * H, ok: i !== 2 }))
  const id = (s: string) => `hud-xm-${s}-${seed}`
  const disagree = CROSSMODAL.regions.filter((region) => region.verdict === 'disagree').length

  return (
    <Hud
      id="crossmodal"
      hasImage={hasImage}
      defs={
        <>
          <radialGradient id={id('veg')}>
            <stop offset="0%" stopColor={OK} stopOpacity={0.32} />
            <stop offset="100%" stopColor={OK} stopOpacity={0} />
          </radialGradient>
          <radialGradient id={id('soil')}>
            <stop offset="0%" stopColor={SAND} stopOpacity={0.3} />
            <stop offset="100%" stopColor={SAND} stopOpacity={0} />
          </radialGradient>
          <pattern id={id('speckle')} width="24" height="24" patternUnits="userSpaceOnUse">
            {[
              [3, 5],
              [14, 2],
              [20, 11],
              [8, 15],
              [17, 20],
              [2, 21],
            ].map(([x, y], i) => (
              <rect key={i} x={x} y={y} width="1.5" height="1.5" fill={SKIP} opacity={0.55} />
            ))}
          </pattern>
          <filter id={id('cloud')}>
            <feGaussianBlur stdDeviation="8" />
          </filter>
          <clipPath id={id('sar')}>
            <path d={`M${seamX(0)} 0H${W}V${H}H${seamX(H)}Z`} />
          </clipPath>
        </>
      }
      scene={
        <>
          {blobs.map((b, i) => (
            <ellipse key={i} cx={b.cx} cy={b.cy} rx={b.rx} ry={b.ry} fill={`url(#${id(b.veg ? 'veg' : 'soil')})`} />
          ))}
          {clouds.map((c, i) => (
            <ellipse key={i} cx={c.cx} cy={c.cy} rx={c.rx} ry={c.ry} fill={HI} opacity={0.12} filter={`url(#${id('cloud')})`} />
          ))}
          <g clipPath={`url(#${id('sar')})`}>
            <rect className="hud-acquire" width={W} height={H} fill={`url(#${id('speckle')})`} />
            {streaks.map((s, i) => (
              <line
                key={i}
                className="hud-streak"
                style={{ ['--hud-at' as string]: `${200 + i * 30}ms`, transformOrigin: `${s.x}px ${s.y}px` }}
                x1={s.x}
                y1={s.y}
                x2={s.x + s.len}
                y2={s.y}
                stroke={SAND}
                strokeOpacity={0.7}
                strokeWidth="1.5"
              />
            ))}
          </g>
        </>
      }
      evidence={
        <>
          {/* Agreement samples along the seam. */}
          <line x1={seamX(0)} y1={0} x2={seamX(H)} y2={H} stroke={LABEL} strokeOpacity={0.9} strokeWidth="1.5" />
          {dots.map((d, i) => (
            <circle key={i} className="hud-fade" style={{ ['--hud-at' as string]: `${700 + i * 90}ms` }} cx={seamX(d.y)} cy={d.y} r={3} fill={d.ok ? OK : FAIL} />
          ))}
          {/* Region 1 agrees; region 2 does not, and a cloud sits inside it. */}
          <rect className="hud-fade" style={{ ['--hud-at' as string]: '1000ms' }} x={REGIONS[0].x} y={REGIONS[0].y} width={REGIONS[0].w} height={REGIONS[0].h} rx={3} fill={OK} fillOpacity={0.12} />
          <rect className="hud-draw" style={{ ['--hud-at' as string]: '600ms', ['--hud-dur' as string]: '500ms' }} x={REGIONS[0].x} y={REGIONS[0].y} width={REGIONS[0].w} height={REGIONS[0].h} rx={3} fill="none" stroke={OK} pathLength={1} />
          <rect className="hud-fade" style={{ ['--hud-at' as string]: '1000ms' }} x={REGIONS[1].x} y={REGIONS[1].y} width={REGIONS[1].w} height={REGIONS[1].h} rx={3} fill={FAIL} fillOpacity={0.12} />
          <g className="hud-pulse" style={{ transformOrigin: `${REGIONS[1].x + REGIONS[1].w / 2}px ${REGIONS[1].y + REGIONS[1].h / 2}px` }}>
            <rect x={REGIONS[1].x} y={REGIONS[1].y} width={REGIONS[1].w} height={REGIONS[1].h} rx={3} fill="none" stroke={FAIL} strokeDasharray="4 3" />
          </g>
        </>
      }
    >
      <span className="t-coord glass absolute top-3 left-3 px-2 py-1 text-text-hi">optical · TC</span>
      <span className="t-coord glass absolute top-3 right-3 px-2 py-1 text-text-hi">SAR · VV/VH</span>
      <Readout at="bl" className="flex items-center gap-2">
        physics_agreement
        <span className="chip bg-fail/12 text-fail">
          DISAGREE · {disagree} {disagree === 1 ? 'region' : 'regions'}
        </span>
        <span className="t-coord text-skip-text">synthetic</span>
      </Readout>
      <Readout at="br" className="t-coord hidden items-center gap-3 !py-2 text-text-lo wide:flex">
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block size-2.5 rounded-[2px] border border-ok bg-ok/20" />
          agree
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block size-2.5 rounded-[2px] border border-dashed border-fail bg-fail/20" />
          disagree
        </span>
      </Readout>
    </Hud>
  )
}
