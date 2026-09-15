import { describe, expect, it } from 'vitest'

import { GROUNDING_BOXES, GROUNDING_FRAME, pixelToWgs84 } from '@/mocks/grounding'
import { USE_CASES } from '@/pages/usecases/catalogue'

describe('grounding fixture', () => {
  it('derives every normalised box from its pixel box over the frame', () => {
    for (const box of GROUNDING_BOXES) {
      const [x0, y0, x1, y1] = box.bbox_px
      expect(box.bbox_normalised).toEqual([
        Math.round((x0 / GROUNDING_FRAME.width) * 1000),
        Math.round((y0 / GROUNDING_FRAME.height) * 1000),
        Math.round((x1 / GROUNDING_FRAME.width) * 1000),
        Math.round((y1 / GROUNDING_FRAME.height) * 1000),
      ])
      expect(x1).toBeGreaterThan(x0)
      expect(y1).toBeGreaterThan(y0)
      expect(x1).toBeLessThanOrEqual(GROUNDING_FRAME.width)
      expect(y1).toBeLessThanOrEqual(GROUNDING_FRAME.height)
    }
  })

  it('places the frame centre on the catalogue centroid', () => {
    const scene = USE_CASES.find((useCase) => useCase.slug === 'airbase-apron')!
    const [lat, lon] = pixelToWgs84(GROUNDING_FRAME.width / 2, GROUNDING_FRAME.height / 2)
    expect(lat).toBeCloseTo(scene.centroid[0], 6)
    expect(lon).toBeCloseTo(scene.centroid[1], 6)
  })

  it('moves one pixel by one GSD', () => {
    const [lat0] = pixelToWgs84(0, 0)
    const [lat1] = pixelToWgs84(0, 1)
    expect((lat0 - lat1) * 111_320).toBeCloseTo(GROUNDING_FRAME.gsd_m, 3)
  })
})
