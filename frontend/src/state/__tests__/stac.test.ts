/**
 * The discovery store's offline contract: results already on the shelf are
 * never taken away by the network going, and one automatic retry runs when
 * it comes back.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { OfflineError } from '@/geo/deadline'
import * as stac from '@/geo/stac'
import { selectedItems, useStacStore } from '@/state/stac'

const item = (id: string, extra: Partial<stac.StacItem> = {}): stac.StacItem => ({
  id,
  collection: 'sentinel-1-rtc',
  sensor: 'sar',
  datetime: '2026-09-01T00:00:00Z',
  bbox: [72, 22, 73, 23],
  geometry: null,
  cloud: null,
  orbitState: 'descending',
  relativeOrbit: 34,
  platform: null,
  tile: null,
  polarisations: ['VV', 'VH'],
  thumbHref: null,
  epsg: null,
  ...extra,
})

beforeEach(() => {
  useStacStore.getState().reset()
  vi.restoreAllMocks()
})

describe('stac store', () => {
  it('keeps the shelf when a later search goes offline, and retries once on reconnect', async () => {
    const spy = vi.spyOn(stac, 'searchItems').mockResolvedValueOnce([item('a'), item('b')])
    useStacStore.getState().setBbox([72, 22, 73, 23])
    await useStacStore.getState().search()
    expect(useStacStore.getState().status).toBe('ok')
    expect(useStacStore.getState().results).toHaveLength(2)

    spy.mockRejectedValueOnce(new OfflineError())
    await useStacStore.getState().search()
    expect(useStacStore.getState().status).toBe('offline')
    expect(useStacStore.getState().results).toHaveLength(2)
    expect(useStacStore.getState().error).toMatch(/still here/)

    spy.mockResolvedValueOnce([item('c')])
    useStacStore.getState().markOnline()
    await vi.waitFor(() => expect(useStacStore.getState().status).toBe('ok'))
    expect(useStacStore.getState().results.map((r) => r.id)).toEqual(['c'])
  })

  it('fills T1 then T2 when pairing, and names an orbit mismatch', async () => {
    vi.spyOn(stac, 'searchItems').mockResolvedValueOnce([item('a'), item('b', { orbitState: 'ascending' }), item('c')])
    useStacStore.getState().setBbox([72, 22, 73, 23])
    await useStacStore.getState().search()
    useStacStore.getState().setPair(true)
    useStacStore.getState().pick('a')
    useStacStore.getState().pick('b')
    let picked = selectedItems(useStacStore.getState())
    expect(picked.t1?.id).toBe('a')
    expect(picked.t2?.id).toBe('b')
    expect(picked.issue).toMatch(/different orbits/)
    useStacStore.getState().pick('b') // deselect
    useStacStore.getState().pick('c')
    picked = selectedItems(useStacStore.getState())
    expect(picked.t2?.id).toBe('c')
    expect(picked.issue).toBeNull()
  })

  it('drops a selection that a new search no longer lists', async () => {
    const spy = vi.spyOn(stac, 'searchItems').mockResolvedValueOnce([item('a')])
    useStacStore.getState().setBbox([72, 22, 73, 23])
    await useStacStore.getState().search()
    useStacStore.getState().pick('a')
    spy.mockResolvedValueOnce([item('z')])
    await useStacStore.getState().search()
    expect(useStacStore.getState().selection.t1).toBeNull()
  })
})
