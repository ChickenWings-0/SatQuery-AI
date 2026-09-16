/**
 * Optical, SAR, or both — and the cloud ceiling when optical is in play.
 */
import { useEffect, useRef } from 'react'

import { Segmented } from '@/components/ui/Segmented'
import { CloudIcon } from '@/components/ui/icons'
import type { SensorChoice } from '@/geo/collections'
import { useStacStore } from '@/state/stac'

const OPTIONS = [
  { value: 'optical', label: 'Sentinel-2', title: 'Sentinel-2 L2A, optical' },
  { value: 'sar', label: 'Sentinel-1', title: 'Sentinel-1 RTC, SAR' },
  { value: 'both', label: 'Both', title: 'Optical and SAR' },
] as const satisfies readonly { value: SensorChoice; label: string; title: string }[]

export function CollectionPicker() {
  const sensor = useStacStore((state) => state.query.sensor)
  const cloudMax = useStacStore((state) => state.query.cloudMax)
  const setSensor = useStacStore((state) => state.setSensor)
  const setCloudMax = useStacStore((state) => state.setCloudMax)
  const search = useStacStore((state) => state.search)
  const bbox = useStacStore((state) => state.query.bbox)

  // The slider settles before the search runs, and the first render never searches.
  const first = useRef(true)
  useEffect(() => {
    if (first.current) {
      first.current = false
      return
    }
    if (!bbox || sensor === 'sar') return
    const timer = setTimeout(() => void search(), 350)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloudMax])

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Segmented value={sensor} options={OPTIONS} onChange={setSensor} label="Sensor" size="sm" />
      {sensor !== 'sar' ? (
        <label className="flex items-center gap-2 text-[12px] text-text-lo">
          <CloudIcon size={14} />
          <input
            type="range"
            min={0}
            max={60}
            step={5}
            value={cloudMax}
            aria-label="Maximum cloud cover"
            onChange={(event) => setCloudMax(Number(event.target.value))}
            className="w-20 accent-[var(--color-accent-warm)]"
          />
          <span className="t-coord w-9 text-text-hi">≤ {cloudMax}%</span>
        </label>
      ) : null}
    </div>
  )
}
