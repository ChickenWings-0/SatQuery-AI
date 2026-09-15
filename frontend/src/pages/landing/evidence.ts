/**
 * Every number the landing page shows, in one place, each with its
 * provenance. Two kinds:
 *
 *   recorded   read from the canonical recording
 *              (`mocks/captured/events.bitemporal.json`) — `evidence.test.ts`
 *              fails if the two ever disagree, so the page cannot drift from
 *              the fixture it quotes.
 *   synthetic  drawn geometry with no tool behind it, declared as such here
 *              and on the HUD that shows it. Counts on a synthetic HUD are
 *              derived from what is drawn, never typed.
 *
 * The JSON itself is not imported here: the recording is 2 500 lines and
 * the landing chunk should not carry it for four scalars.
 */
import { GROUNDING_BOXES, GROUNDING_FIXTURE, GROUNDING_FRAME, pixelToWgs84 } from '@/mocks/grounding'

export const CHANGE = {
  kind: 'recorded',
  pct: 5.34,
  km2: 0.35,
  regions: 1,
  confidence: 0.65,
  source: 'step:3/scalars.changed_area_pct',
  dates: ['2019', '2024'],
} as const

/** Two regions straddle the seam in the drawing; the readout counts them. */
export const CROSSMODAL = {
  kind: 'synthetic',
  regions: [
    { id: 'r0', verdict: 'agree' },
    { id: 'r1', verdict: 'disagree' },
  ],
} as const

export const GROUNDING = {
  kind: 'synthetic',
  boxes: GROUNDING_BOXES,
  frame: GROUNDING_FRAME,
  generator: GROUNDING_FIXTURE.generator,
  /** The locked box — the reticle's target — and its origin as a coordinate. */
  locked: (() => {
    const box = GROUNDING_BOXES[1]!
    const [col, row] = box.bbox_px
    return { box, col, row, wgs84: pixelToWgs84(col, row) }
  })(),
} as const
