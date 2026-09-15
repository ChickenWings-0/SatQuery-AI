/**
 * The drawn face of a use case that has no `thumb.webp` yet.
 *
 * Four plates, one per task family, each a stylised version of what that
 * family's *evidence* looks like in the console — not a stock photo and not a
 * blank wireframe:
 *
 *   change      two dates side by side on a seam, the delta hatched in
 *               terracotta on the later date (the CHANGE view's own mark)
 *   crossmodal  optical and SAR meeting on a swipe seam: soft spectral blobs
 *               on one side, speckle and bright backscatter on the other,
 *               with agree / disagree markers along the join
 *   grounding   objects on the ground with the console's ticked-corner
 *               bounding boxes around them
 *   classify    a land-cover mosaic, four classes in the status and accent
 *               tints, spatially coherent the way a real segmentation is
 *
 * Everything is seeded from the slug, so two cards in the same family draw
 * different parcels, blobs and boxes, and the same card draws the same plate
 * on every render. Colours are the theme's own tokens, read through CSS
 * variables, so the plates follow the light theme without a second palette.
 * All of it is decoration under `aria-hidden`; the card's text carries the
 * meaning.
 */
import { between, rng } from '@/lib/rng'
import type { TaskFamily } from '@/pages/usecases/catalogue'

const W = 400
const H = 300

const SAND = 'var(--color-evidence)'
const TERRA = 'var(--color-accent-warm)'
const OK = 'var(--color-ok)'
const WARN = 'var(--color-warn)'
const SKIP = 'var(--color-skip)'
const LABEL = 'var(--color-accent-warm-text)'

/** A HUD readout on the plate: mono, tracked, quiet. */
function Label({ x, y, anchor = 'start', children }: { x: number; y: number; anchor?: 'start' | 'end'; children: string }) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      fill={LABEL}
      opacity={0.95}
      style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '0.1em', fontWeight: 500 }}
    >
      {children}
    </text>
  )
}

interface Parcel {
  x: number
  y: number
  w: number
  h: number
}

function parcels(r: () => number, count: number, xMax: number): Parcel[] {
  const out: Parcel[] = []
  for (let i = 0; i < count; i++) {
    out.push({
      x: between(r, 8, xMax - 60),
      y: between(r, 24, H - 70),
      w: between(r, 22, 62),
      h: between(r, 14, 44),
    })
  }
  return out
}

function ChangePlate({ slug }: { slug: string }) {
  const r = rng(slug)
  const before = parcels(r, 9, W / 2)
  const grown = parcels(r, 5, W / 2)
  const id = `hatch-${slug}`
  return (
    <>
      <defs>
        <pattern id={id} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke={TERRA} strokeWidth="1.4" />
        </pattern>
      </defs>
      {/* the same parcels on both dates — what did not change */}
      {[0, W / 2].map((dx) =>
        before.map((p, i) => (
          <rect
            key={`${dx}-${i}`}
            x={p.x + dx}
            y={p.y}
            width={p.w}
            height={p.h}
            rx={1.5}
            fill={SAND}
            fillOpacity={0.14}
            stroke={SAND}
            strokeOpacity={0.35}
          />
        )),
      )}
      {/* the delta, on the later date only */}
      {grown.map((p, i) => (
        <g key={i}>
          <rect x={p.x + W / 2} y={p.y} width={p.w} height={p.h} rx={1.5} fill={TERRA} fillOpacity={0.22} />
          <rect
            x={p.x + W / 2}
            y={p.y}
            width={p.w}
            height={p.h}
            rx={1.5}
            fill={`url(#${id})`}
            stroke={TERRA}
            strokeOpacity={0.9}
            strokeWidth={1}
          />
        </g>
      ))}
      <line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke={LABEL} strokeOpacity={0.7} strokeDasharray="5 5" />
      <Label x={12} y={H - 16}>
        T₀
      </Label>
      <Label x={W / 2 + 12} y={H - 16}>
        T₁ · Δ
      </Label>
    </>
  )
}

function CrossModalPlate({ slug }: { slug: string }) {
  const r = rng(slug)
  const seamTop = between(r, 220, 280)
  const seamBottom = seamTop - 120
  const optical = `opt-${slug}`
  const sar = `sar-${slug}`
  const glow = `glow-${slug}`
  const blobs = Array.from({ length: 7 }, () => ({
    cx: between(r, 20, 260),
    cy: between(r, 30, H - 30),
    rx: between(r, 28, 70),
    ry: between(r, 20, 48),
    veg: r() > 0.55,
  }))
  const streaks = Array.from({ length: 14 }, () => ({
    x: between(r, 150, W - 20),
    y: between(r, 20, H - 20),
    w: between(r, 6, 26),
    h: between(r, 2, 4),
    rot: between(r, -30, 30),
  }))
  const speckle = Array.from({ length: 64 }, () => ({ x: between(r, 0, 24), y: between(r, 0, 24), o: between(r, 0.15, 0.6) }))
  const markers = Array.from({ length: 4 }, (_, i) => {
    const t = (i + 0.5) / 4
    return { x: seamTop + (seamBottom - seamTop) * t, y: H * t, agree: i !== 2 }
  })
  return (
    <>
      <defs>
        <clipPath id={optical}>
          <polygon points={`0,0 ${seamTop},0 ${seamBottom},${H} 0,${H}`} />
        </clipPath>
        <clipPath id={sar}>
          <polygon points={`${seamTop},0 ${W},0 ${W},${H} ${seamBottom},${H}`} />
        </clipPath>
        {/* One gradient per class: `currentColor` inside a <stop> resolves
            from the gradient, not from the shape that references it. */}
        <radialGradient id={`${glow}-sand`}>
          <stop offset="0" stopColor={SAND} stopOpacity="0.6" />
          <stop offset="1" stopColor={SAND} stopOpacity="0" />
        </radialGradient>
        <radialGradient id={`${glow}-veg`}>
          <stop offset="0" stopColor={OK} stopOpacity="0.5" />
          <stop offset="1" stopColor={OK} stopOpacity="0" />
        </radialGradient>
        <pattern id={`${sar}-p`} width="24" height="24" patternUnits="userSpaceOnUse">
          {speckle.map((s, i) => (
            <rect key={i} x={s.x} y={s.y} width="1.6" height="1.6" fill={SKIP} opacity={s.o} />
          ))}
        </pattern>
      </defs>
      <g clipPath={`url(#${optical})`}>
        {blobs.map((b, i) => (
          <ellipse
            key={i}
            cx={b.cx}
            cy={b.cy}
            rx={b.rx}
            ry={b.ry}
            fill={`url(#${glow}-${b.veg ? 'veg' : 'sand'})`}
          />
        ))}
      </g>
      <g clipPath={`url(#${sar})`}>
        <rect x={0} y={0} width={W} height={H} fill={`url(#${sar}-p)`} />
        {streaks.map((s, i) => (
          <rect
            key={i}
            x={s.x}
            y={s.y}
            width={s.w}
            height={s.h}
            rx={1}
            fill={SAND}
            fillOpacity={0.7}
            transform={`rotate(${s.rot} ${s.x} ${s.y})`}
          />
        ))}
      </g>
      <line x1={seamTop} y1={0} x2={seamBottom} y2={H} stroke={LABEL} strokeOpacity={0.8} strokeWidth={1.5} />
      <circle
        cx={(seamTop + seamBottom) / 2}
        cy={H / 2}
        r={7}
        fill="var(--color-glass)"
        stroke={LABEL}
        strokeOpacity={0.9}
        strokeWidth={1.25}
      />
      {markers.map((m, i) => (
        <circle key={i} cx={m.x} cy={m.y} r={3.5} fill={m.agree ? OK : WARN} />
      ))}
      <Label x={12} y={H - 16}>
        OPTICAL
      </Label>
      <Label x={W - 12} y={H - 16} anchor="end">
        SAR VV
      </Label>
    </>
  )
}

function GroundingPlate({ slug }: { slug: string }) {
  const r = rng(slug)
  const objects = Array.from({ length: 6 }, () => {
    const w = between(r, 14, 26)
    const h = between(r, 22, 40)
    return {
      x: between(r, 24, W - 60),
      y: between(r, 40, H - 80),
      w,
      h,
      rot: between(r, -40, 40),
      pad: between(r, 6, 10),
    }
  })
  const tick = 7
  return (
    <>
      {objects.map((o, i) => {
        const bx = o.x - o.pad
        const by = o.y - o.pad
        const bw = o.w + o.pad * 2
        const bh = o.h + o.pad * 2
        return (
          <g key={i}>
            <rect
              x={o.x}
              y={o.y}
              width={o.w}
              height={o.h}
              rx={o.w / 2}
              fill={SAND}
              fillOpacity={0.4}
              transform={`rotate(${o.rot} ${o.x + o.w / 2} ${o.y + o.h / 2})`}
            />
            <rect x={bx} y={by} width={bw} height={bh} fill="var(--color-bbox-fill)" stroke="var(--color-bbox)" strokeWidth={1} />
            <path
              d={`M${bx} ${by + tick}v${-tick}h${tick} M${bx + bw - tick} ${by}h${tick}v${tick} M${bx + bw} ${by + bh - tick}v${tick}h${-tick} M${bx + tick} ${by + bh}h${-tick}v${-tick}`}
              stroke="var(--color-bbox-label)"
              strokeWidth={2}
              fill="none"
            />
          </g>
        )
      })}
      <Label x={12} y={H - 16}>
        GROUNDED · BBOX
      </Label>
    </>
  )
}

function ClassifyPlate({ slug }: { slug: string }) {
  const r = rng(slug)
  const cols = 12
  const rows = 9
  const cw = W / cols
  const ch = H / rows
  const fx = between(r, 0.5, 1.1)
  const fy = between(r, 0.5, 1.1)
  const px = between(r, 0, 6)
  const py = between(r, 0, 6)
  const classes = [OK, SAND, TERRA, SKIP]
  const cells: { x: number; y: number; c: number }[] = []
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const v = Math.sin(x * fx + px) + Math.cos(y * fy + py) + Math.sin((x + y) * 0.7 + px) * 0.6 + (r() - 0.5) * 0.7
      const c = Math.min(3, Math.max(0, Math.floor(((v + 2.6) / 5.2) * 4)))
      cells.push({ x, y, c })
    }
  }
  return (
    <>
      {cells.map((cell) => (
        <rect
          key={`${cell.x}-${cell.y}`}
          x={cell.x * cw + 1}
          y={cell.y * ch + 1}
          width={cw - 2}
          height={ch - 2}
          rx={2}
          fill={classes[cell.c]}
          fillOpacity={0.3}
        />
      ))}
      {['VEG', 'BARE', 'BUILT', 'WATER'].map((name, i) => (
        <g key={name} transform={`translate(${12 + i * 60} ${H - 25})`}>
          <rect width={8} height={8} rx={1.5} fill={classes[i]} fillOpacity={0.9} />
          <Label x={13} y={8}>
            {name}
          </Label>
        </g>
      ))}
    </>
  )
}

const PLATES: Record<TaskFamily, (props: { slug: string }) => React.JSX.Element> = {
  change: ChangePlate,
  crossmodal: CrossModalPlate,
  grounding: GroundingPlate,
  classify: ClassifyPlate,
}

export function FamilyPlate({ family, slug, className = '' }: { family: TaskFamily; slug: string; className?: string }) {
  const Plate = PLATES[family]
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 size-full ${className}`}
    >
      <Plate slug={slug} />
    </svg>
  )
}
