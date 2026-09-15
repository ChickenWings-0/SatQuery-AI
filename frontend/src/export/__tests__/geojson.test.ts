import { describe, expect, it } from 'vitest'

import { boxesToGeoJson } from '@/export/geojson'

const box = { xMin: 0, yMin: 0, xMax: 500, yMax: 1000, label: 'a', score: 0.9 }

describe('GeoJSON export', () => {
  it('maps normalised boxes through WGS84 bounds', () => {
    const out = boxesToGeoJson([box], [70, 10, 80, 20])
    const ring = out.features[0]!.geometry.coordinates[0]!
    expect(ring[0]).toEqual([70, 20])
    expect(ring[2]).toEqual([75, 10])
    expect(out.crs?.properties.name).toContain('CRS84')
    expect(out.note).toBeUndefined()
  })

  it('never invents lon/lat without bounds', () => {
    const out = boxesToGeoJson([box], null)
    expect(out.note).toMatch(/No georeference/)
    expect(out.features[0]!.properties['crs']).toBe('pixel')
    expect(out.features[0]!.geometry.coordinates[0]![2]).toEqual([500, 1000])
  })
})
