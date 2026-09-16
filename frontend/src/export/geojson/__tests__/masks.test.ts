import { describe, expect, it } from 'vitest'

import { assemble, traceRings } from '@/export/geojson/masks'
import { signedArea } from '@/export/geojson/winding'

describe('mask vectoriser', () => {
  it('traces a solid block into one closed ring of four corners', () => {
    // 4×4 grid with a 2×2 block of class 1 in the middle.
    const cells = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    const rings = traceRings(cells, 4, 4, 1)
    expect(rings).toHaveLength(1)
    expect(rings[0]).toHaveLength(4)
    expect(Math.abs(signedArea(rings[0]!))).toBe(4)
  })

  it('separates a hole from its exterior', () => {
    // 5×5 ring of class 1 around an empty centre.
    const cells = [
      1, 1, 1, 1, 1,
      1, 0, 0, 0, 1,
      1, 0, 0, 0, 1,
      1, 0, 0, 0, 1,
      1, 1, 1, 1, 1,
    ]
    const rings = traceRings(cells, 5, 5, 1)
    expect(rings).toHaveLength(2)
    const polygons = assemble(rings)
    expect(polygons).toHaveLength(1)
    expect(polygons[0]).toHaveLength(2)
    expect(Math.abs(signedArea(polygons[0]![0]!))).toBe(25)
    expect(Math.abs(signedArea(polygons[0]![1]!))).toBe(9)
  })

  it('emits nothing for a class that is absent', () => {
    expect(traceRings([0, 0, 0, 0], 2, 2, 1)).toEqual([])
  })
})
