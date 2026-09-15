/**
 * The fallback when the manifest carries no CRS: the primary view, full
 * bleed, pan/zoom, with the HUD in pixel mode and an honest line about why.
 */
import { useState } from 'react'
import { TransformComponent, TransformWrapper } from 'react-zoom-pan-pinch'

import { artifactUrl } from '@/api/client'
import { primaryOf, type ViewGroup } from '@/evidence/views'
import { CoordinateLocator } from '@/pages/maps/hud/CoordinateLocator'
import { LayerSwitcher } from '@/pages/maps/hud/LayerSwitcher'
import { ZoomHud } from '@/pages/maps/hud/ZoomHud'
import { useMapStore } from '@/state/map'

export default function PixelStage({ groups, sensor }: { groups: ViewGroup[]; sensor: string | null }) {
  const layerKey = useMapStore((state) => state.layerKey)
  const setLayer = useMapStore((state) => state.setLayer)
  const active = groups.find((g) => g.key === layerKey) ?? groups[0] ?? null
  const primary = active ? primaryOf(active) : null
  const url = primary ? artifactUrl(primary) : null
  const [pixel, setPixel] = useState<{ x: number; y: number } | null>(null)

  return (
    <div className="absolute inset-0 bg-bg-main">
      <TransformWrapper minScale={0.5} maxScale={12} centerOnInit>
        {({ zoomIn, zoomOut, resetTransform }) => (
          <>
            <TransformComponent wrapperClass="!size-full" contentClass="!size-full grid place-items-center">
              {url ? (
                <img
                  src={url}
                  alt={active ? `${active.name} view, pixel space` : 'Scene'}
                  className="max-h-full max-w-full [image-rendering:pixelated]"
                  onMouseMove={(event) => {
                    const img = event.currentTarget
                    const rect = img.getBoundingClientRect()
                    setPixel({
                      x: ((event.clientX - rect.left) / rect.width) * img.naturalWidth,
                      y: ((event.clientY - rect.top) / rect.height) * img.naturalHeight,
                    })
                  }}
                  onMouseLeave={() => setPixel(null)}
                />
              ) : null}
            </TransformComponent>
            <div className="absolute top-4 left-4 hidden wide:block">
              <LayerSwitcher
                groups={groups}
                layerKey={active?.key ?? null}
                compareKey={null}
                basemap="none"
                opacity={1}
                sensor={sensor}
                onLayer={setLayer}
                onCompare={() => undefined}
                onBasemap={() => undefined}
                onOpacity={() => undefined}
              />
              <p className="t-meta mt-2 max-w-60 text-warn-text">Manifest carries no CRS — showing pixel space.</p>
            </div>
            <div className="absolute top-4 right-4">
              <ZoomHud onIn={() => zoomIn()} onOut={() => zoomOut()} onFit={() => resetTransform()} />
            </div>
            <div className="absolute bottom-4 left-4">
              <CoordinateLocator cursor={null} zoom={0} onGo={() => undefined} pixelMode pixel={pixel} />
            </div>
          </>
        )}
      </TransformWrapper>
    </div>
  )
}
