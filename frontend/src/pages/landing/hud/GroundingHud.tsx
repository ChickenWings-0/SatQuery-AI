/**
 * HUD C — sub-pixel grounding. An apron with three aircraft; three ticked
 * boxes, each carrying its score, each with the raster grid subdivided
 * inside it — which is what "validated against the grid" looks like. The
 * reticle locks on the middle aircraft and the readout gives that box's
 * origin as a pixel pair and, through the fixture's transform, as a
 * coordinate.
 *
 * The boxes are `mocks/grounding.ts` — a declared synthetic fixture, said so
 * on the HUD. Their placement here is their `bbox_normalised` scaled to the
 * viewBox, so the drawing and the readout cannot disagree.
 */
import { Reticle } from '@/components/ui/Reticle'
import { between, rng } from '@/lib/rng'
import { GROUNDING } from '@/pages/landing/evidence'
import { BBOX, BBOX_FILL, BBOX_LABEL, GRID_MAJOR, H, Hud, Readout, SAND, SKIP, Tag, W } from '@/pages/landing/hud/Hud'
import { dms } from '@/pages/usecases/catalogue'

/** A narrow-body silhouette, nose up, ~48 × 40 in local units. */
const AIRCRAFT =
  'M0 -20 L3 -12 L3 -2 L22 8 L22 12 L3 6 L3 14 L8 18 L8 20 L0 18 L-8 20 L-8 18 L-3 14 L-3 6 L-22 12 L-22 8 L-3 -2 L-3 -12 Z'

const toView = ([x0, y0, x1, y1]: readonly [number, number, number, number]) => ({
  x: (x0 / 1000) * W,
  y: (y0 / 1000) * H,
  w: ((x1 - x0) / 1000) * W,
  h: ((y1 - y0) / 1000) * H,
})

export function GroundingHud({ seed, hasImage }: { seed: string; hasImage: boolean }) {
  const r = rng(seed)
  const boxes = GROUNDING.boxes.map((box) => ({ ...box, view: toView(box.bbox_normalised), tilt: between(r, -20, 20) }))
  const locked = boxes[1]!
  const { col, row, wgs84 } = GROUNDING.locked
  const cx = locked.view.x + locked.view.w / 2
  const cy = locked.view.y + locked.view.h / 2

  return (
    <Hud
      id="grounding"
      hasImage={hasImage}
      scene={
        <>
          {/* The apron and its taxiways. */}
          <path d={`M70 60H${W - 30}L${W - 60} ${H - 40}H40Z`} fill={SKIP} fillOpacity={0.18} />
          <path d={`M70 60L40 ${H - 40}M${W - 30} 60L${W - 60} ${H - 40}`} stroke={SKIP} strokeOpacity={0.4} />
          {[120, 220, 320].map((y) => (
            <line key={y} x1={40} y1={y} x2={W - 60} y2={y} stroke={SKIP} strokeOpacity={0.4} />
          ))}
          <line x1={40} y1={H / 2} x2={W - 60} y2={H / 2} stroke={SAND} strokeOpacity={0.45} strokeDasharray="10 8" />
          {boxes.map((box) => (
            <path
              key={box.id}
              d={AIRCRAFT}
              transform={`translate(${box.view.x + box.view.w / 2} ${box.view.y + box.view.h / 2}) rotate(${box.tilt.toFixed(1)}) scale(1.1)`}
              fill={SAND}
              fillOpacity={0.45}
            />
          ))}
        </>
      }
      evidence={
        <>
          {boxes.map((box, i) => {
            const { x, y, w, h } = box.view
            const at = `${480 + i * 120}ms`
            const tick = 8
            const corners = `M${x} ${y + tick}V${y}H${x + tick} M${x + w - tick} ${y}H${x + w}V${y + tick} M${x} ${y + h - tick}V${y + h}H${x + tick} M${x + w - tick} ${y + h}H${x + w}V${y + h - tick}`
            const cols = Math.round(w / 10)
            const rows = Math.round(h / 10)
            return (
              <g key={box.id}>
                {/* The sub-pixel grid inside the box only. */}
                <g className="hud-fade" style={{ ['--hud-at' as string]: at }} stroke={GRID_MAJOR}>
                  {Array.from({ length: cols - 1 }, (_, k) => (
                    <line key={`c${k}`} x1={x + ((k + 1) * w) / cols} y1={y} x2={x + ((k + 1) * w) / cols} y2={y + h} />
                  ))}
                  {Array.from({ length: rows - 1 }, (_, k) => (
                    <line key={`r${k}`} x1={x} y1={y + ((k + 1) * h) / rows} x2={x + w} y2={y + ((k + 1) * h) / rows} />
                  ))}
                </g>
                <rect className="hud-fade" style={{ ['--hud-at' as string]: at }} x={x} y={y} width={w} height={h} fill={BBOX_FILL} />
                <rect className="hud-draw" style={{ ['--hud-at' as string]: at, ['--hud-dur' as string]: '300ms' }} x={x} y={y} width={w} height={h} fill="none" stroke={BBOX} pathLength={1} />
                <path className="hud-fade" style={{ ['--hud-at' as string]: `${780 + i * 120}ms` }} d={corners} fill="none" stroke={BBOX_LABEL} strokeWidth="1.5" />
                <g className="hud-rise" style={{ ['--hud-at' as string]: `${820 + i * 120}ms` }}>
                  <Tag x={x} y={y - 7} fill={BBOX_LABEL}>
                    {`${box.label} · ${box.score.toFixed(2)}`}
                  </Tag>
                </g>
              </g>
            )
          })}
        </>
      }
    >
      {/* Centred by the wrapper; the lock animates the mark's own transform. */}
      <span aria-hidden className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${(cx / W) * 100}%`, top: `${(cy / H) * 100}%` }}>
        <Reticle size={64} className="hud-lock block" />
      </span>
      <span className="t-coord glass absolute top-3 left-3 px-2 py-1 text-text-hi">VHR · {GROUNDING.frame.gsd_m} m GSD</span>
      <span className="t-coord glass absolute top-3 right-3 px-2 py-1 text-text-hi">GROUNDING · COUNT</span>
      <Readout at="bl" className="flex items-center gap-2">
        <span className="t-mono font-semibold">{GROUNDING.boxes.length}</span>
        <span className="text-text-lo">validated positions</span>
        <span className="t-coord text-skip-text">synthetic</span>
      </Readout>
      <Readout at="br" className="t-coord hidden text-text-lo wide:inline">
        px {col},{row} → {dms(wgs84)}
      </Readout>
    </Hud>
  )
}
