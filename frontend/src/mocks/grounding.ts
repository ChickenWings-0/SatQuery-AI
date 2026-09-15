/**
 * A grounding fixture for the airbase-apron scene: three aircraft boxes on a
 * VHR frame, in the shape of a `BBOX_SET` artifact's `inline` payload.
 *
 * **Synthetic, and labelled as such** — `generator` says so and every surface
 * that shows it says so. The canonical recording is a bi-temporal change run
 * and carries no aircraft boxes; the alternative was three confidences typed
 * into a landing page as if a tool had measured them, which is the one thing
 * this project does not do.
 *
 * What *is* honest by construction: `bbox_normalised` is derived from
 * `bbox_px` over the declared frame (never typed separately), and a pixel maps
 * to a coordinate through the declared affine transform, whose origin is
 * placed so that the frame's centre is the catalogue's centroid for the scene.
 * `grounding.test.ts` holds both.
 */
import { USE_CASES } from '@/pages/usecases/catalogue'

export interface GroundingBox {
  id: string
  label: string
  score: number
  bbox_px: [number, number, number, number]
  bbox_normalised: [number, number, number, number]
}

/** Metres per degree of latitude; longitude scales by cos(lat). */
const M_PER_DEG = 111_320

export const GROUNDING_FRAME = { width: 1280, height: 800, gsd_m: 0.5 } as const

const scene = USE_CASES.find((useCase) => useCase.slug === 'airbase-apron')
if (!scene) throw new Error('grounding fixture: the airbase-apron use case is gone')

const [latC, lonC] = scene.centroid
const degLat = GROUNDING_FRAME.gsd_m / M_PER_DEG
const degLon = GROUNDING_FRAME.gsd_m / (M_PER_DEG * Math.cos((latC * Math.PI) / 180))

/** GDAL-style affine: [px size x, 0, origin lon, 0, -px size y, origin lat]. */
export const GROUNDING_TRANSFORM = [
  degLon,
  0,
  lonC - (GROUNDING_FRAME.width / 2) * degLon,
  0,
  -degLat,
  latC + (GROUNDING_FRAME.height / 2) * degLat,
] as const

/** Pixel (column, row) → [lat, lon] through the frame's transform. */
export function pixelToWgs84(col: number, row: number): [lat: number, lon: number] {
  const [a, , c, , e, f] = GROUNDING_TRANSFORM
  return [f + row * e, c + col * a]
}

function normalise([x0, y0, x1, y1]: [number, number, number, number]): [number, number, number, number] {
  const { width, height } = GROUNDING_FRAME
  return [
    Math.round((x0 / width) * 1000),
    Math.round((y0 / height) * 1000),
    Math.round((x1 / width) * 1000),
    Math.round((y1 / height) * 1000),
  ]
}

function box(id: string, score: number, bbox_px: [number, number, number, number]): GroundingBox {
  return { id, label: 'aircraft', score, bbox_px, bbox_normalised: normalise(bbox_px) }
}

/** Three boxes, each 179 × 80 px (≈ 90 × 40 m at 0.5 m GSD — a narrow-body airliner). */
export const GROUNDING_BOXES: readonly GroundingBox[] = [
  box('box_0', 0.94, [282, 224, 461, 304]),
  box('box_1', 0.89, [614, 352, 793, 432]),
  box('box_2', 0.97, [845, 192, 1024, 272]),
]

export const GROUNDING_FIXTURE = {
  id: 'art_grounding_fixture',
  type: 'BBOX_SET',
  label: 'Aircraft on the apron',
  generator: 'text_grounding@1.0.0+adapter:pending (SYNTHETIC FIXTURE)',
  source_image: 'img_0',
  frame: GROUNDING_FRAME,
  boxes: GROUNDING_BOXES,
  centroid: scene.centroid,
} as const
