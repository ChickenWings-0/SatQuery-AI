import { describe, expect, it } from 'vitest'

import { boxesToGeoJson, buildCollection, footprintToGeoJson } from '@/export/geojson/build'
import { signedArea } from '@/export/geojson/winding'

const box = { xMin: 0, yMin: 0, xMax: 500, yMax: 1000, label: 'a', score: 0.9 }

describe('GeoJSON export', () => {
  it('maps normalised boxes through WGS84 bounds', () => {
    const out = boxesToGeoJson([box], [70, 10, 80, 20])
    const ring = out.features[0]!.geometry.coordinates[0] as number[][]
    expect(ring).toContainEqual([70, 20])
    expect(ring).toContainEqual([75, 10])
    expect(out.note).toBeUndefined()
  })

  it('never invents lon/lat without bounds', () => {
    const out = boxesToGeoJson([box], null)
    expect(out.note).toMatch(/No georeference/)
    expect(out.features[0]!.properties['crs']).toBe('pixel')
    const ring = out.features[0]!.geometry.coordinates[0] as number[][]
    expect(ring).toContainEqual([500, 1000])
  })

  it('has no crs member — RFC 7946 §4 forbids it', () => {
    expect('crs' in boxesToGeoJson([box], [70, 10, 80, 20])).toBe(false)
    expect('crs' in footprintToGeoJson([70, 10, 80, 20])).toBe(false)
  })

  it('winds exterior rings counter-clockwise and closes them', () => {
    for (const bounds of [[70, 10, 80, 20] as const, null]) {
      const out = boxesToGeoJson([box], bounds ? [...bounds] : null)
      const ring = out.features[0]!.geometry.coordinates[0] as [number, number][]
      expect(ring[0]).toEqual(ring[ring.length - 1])
      expect(ring).toHaveLength(5)
      // Pixel space is y-down, so the sign flips while the on-screen sense stays CCW.
      const area = signedArea(ring) * (bounds ? 1 : -1)
      expect(area).toBeGreaterThan(0)
    }
  })

  it('carries a bbox on every feature and on the collection', () => {
    const out = boxesToGeoJson([box, { ...box, xMin: 600, xMax: 1000 }], [70, 10, 80, 20])
    expect(out.bbox).toEqual([70, 10, 80, 20])
    expect(out.features[0]!.bbox).toEqual([70, 10, 75, 20])
  })

  it('rounds coordinates to seven decimals', () => {
    const out = boxesToGeoJson([{ ...box, xMax: 333 }], [70.123456789, 10, 80.987654321, 20])
    const ring = out.features[0]!.geometry.coordinates[0] as number[][]
    for (const [x, y] of ring) {
      expect(String(x).split('.')[1]?.length ?? 0).toBeLessThanOrEqual(7)
      expect(String(y).split('.')[1]?.length ?? 0).toBeLessThanOrEqual(7)
    }
  })

  it('emits one feature per box per view, with provenance columns', () => {
    const out = buildCollection({
      layers: [
        { view_id: 'art_1', boxes: [box] },
        { view_id: 'art_2', boxes: [box] },
      ],
      bounds: [70, 10, 80, 20],
      provenance: { tool: 'text_grounding', tool_version: '1.0.0', source_step: 3, trace_id: 'abc', citation: null },
    })
    expect(out.features).toHaveLength(2)
    expect(out.features.map((f) => f.properties['view_id'])).toEqual(['art_1', 'art_2'])
    expect(out.features[0]!.properties).toMatchObject({
      label: 'a',
      confidence: 0.9,
      tool: 'text_grounding',
      tool_version: '1.0.0',
      source_step: 3,
      trace_id: 'abc',
      citation: null,
      scene_id: null,
    })
  })

  it('validates as GeoJSON', () => {
    const out = boxesToGeoJson([box], [70, 10, 80, 20])
    expect(out.type).toBe('FeatureCollection')
    for (const feature of out.features) {
      expect(feature.type).toBe('Feature')
      expect(feature.geometry.type).toBe('Polygon')
      for (const ring of feature.geometry.coordinates as number[][][]) {
        expect(ring.length).toBeGreaterThanOrEqual(4)
        for (const pos of ring) {
          expect(pos).toHaveLength(2)
          expect(pos.every((n) => Number.isFinite(n))).toBe(true)
        }
      }
    }
  })
})
