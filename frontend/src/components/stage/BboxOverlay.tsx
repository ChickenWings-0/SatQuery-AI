/**
 * The spatial grounding layer: the answer's boxes, drawn over the image.
 *
 * The SVG's `viewBox` is `0 0 1000 1000` — the normalised frame the boxes are
 * already in — and `preserveAspectRatio="none"` stretches it to the image's
 * own aspect ratio, so a box maps onto the raster with no pixel arithmetic at
 * all. The element sits inside the `TransformComponent`, over the image, so
 * zoom and pan carry it along for free; `pointer-events: none` lets the drag
 * and the swipe handle pass straight through.
 *
 * Because the layer is stretched, so is anything drawn in it: a 3-unit stroke
 * would be thinner across the long axis of a non-square image. `vector-effect:
 * non-scaling-stroke` holds the stroke at 1.5 screen pixels at every zoom and
 * every aspect ratio, and the labels are HTML rather than `<text>` so the type
 * never distorts.
 *
 * The label's position is in percent of the frame, which is the same 0–1000
 * scale divided by ten. It sits just above its box, and flips below when the
 * box touches the top edge — a label pushed off the top of the image is a
 * label nobody reads.
 */
import type { NormalisedBox } from '@/thread/bbox'

export function BboxOverlay({ boxes }: { boxes: NormalisedBox[] }) {
  if (boxes.length === 0) return null

  return (
    <div
      data-testid="bbox-overlay"
      aria-hidden="true"
      className="sq-arrive pointer-events-none absolute inset-0 z-10"
    >
      <svg
        viewBox="0 0 1000 1000"
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
      >
        {boxes.map((box, index) => (
          <rect
            key={index}
            x={box.xMin}
            y={box.yMin}
            width={box.xMax - box.xMin}
            height={box.yMax - box.yMin}
            fill="var(--color-bbox-fill)"
            stroke="var(--color-bbox)"
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>

      {boxes.map((box, index) => {
        if (!box.label) return null
        const flip = box.yMin < 40
        return (
          <span
            key={index}
            className="absolute max-w-[40%] truncate rounded bg-bg-main/80 px-1 py-px font-mono text-[10px] leading-tight font-semibold text-bbox-label"
            style={{
              left: `${box.xMin / 10}%`,
              ...(flip
                ? { top: `${box.yMax / 10}%`, marginTop: 2 }
                : { bottom: `${100 - box.yMin / 10}%`, marginBottom: 2 }),
            }}
          >
            {box.label}
          </span>
        )
      })}
    </div>
  )
}
