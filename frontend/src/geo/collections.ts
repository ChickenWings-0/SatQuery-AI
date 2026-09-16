/**
 * The catalogue the Maps page can search: Microsoft Planetary Computer, no
 * key for search, a SAS token for asset reads (which the backend does).
 * Sentinel-1 RTC is preferred over GRD: terrain-corrected, and what the
 * BigEarthNet-v2 SAR the adapter trained on looks like.
 */
export type CollectionId = 'sentinel-2-l2a' | 'sentinel-1-rtc' | 'sentinel-1-grd'
export type Sensor = 'optical' | 'sar'
export type SensorChoice = 'optical' | 'sar' | 'both'

export interface CollectionSpec {
  id: CollectionId
  label: string
  short: string
  sensor: Sensor
  /** The assets the pipeline reads — for the copy, the backend decides for real. */
  assets: readonly string[]
  gsdM: number
}

export const COLLECTIONS: readonly CollectionSpec[] = [
  {
    id: 'sentinel-2-l2a',
    label: 'Sentinel-2 L2A',
    short: 'S2',
    sensor: 'optical',
    assets: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
    gsdM: 10,
  },
  {
    id: 'sentinel-1-rtc',
    label: 'Sentinel-1 RTC',
    short: 'S1',
    sensor: 'sar',
    assets: ['vv', 'vh'],
    gsdM: 10,
  },
  {
    id: 'sentinel-1-grd',
    label: 'Sentinel-1 GRD',
    short: 'S1 GRD',
    sensor: 'sar',
    assets: ['vv', 'vh'],
    gsdM: 10,
  },
]

export const STAC_ROOT = 'https://planetarycomputer.microsoft.com/api/stac/v1'
export const NOMINATIM_ROOT = 'https://nominatim.openstreetmap.org'

export function collectionsFor(choice: SensorChoice): CollectionId[] {
  if (choice === 'optical') return ['sentinel-2-l2a']
  if (choice === 'sar') return ['sentinel-1-rtc']
  return ['sentinel-2-l2a', 'sentinel-1-rtc']
}

export function specOf(id: string): CollectionSpec | undefined {
  return COLLECTIONS.find((c) => c.id === id)
}
