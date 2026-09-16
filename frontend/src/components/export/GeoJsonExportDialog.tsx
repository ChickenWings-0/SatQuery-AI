/**
 * The one question the GeoJSON export has to ask: masks or not. Shown only
 * when the run rendered a change mask or a class raster; a run with boxes
 * alone exports on the click. The CRS line is the honesty guard — the file
 * says pixel space in the dialog before it says it in the file.
 */
import { useState } from 'react'

import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { countOf } from '@/format'
import type { GeoJsonPlan } from '@/export/geojson/useGeoJsonExport'

export function GeoJsonExportDialog({
  plan,
  busy,
  onConfirm,
  onCancel,
}: {
  plan: GeoJsonPlan | null
  busy: boolean
  onConfirm: (includeMasks: boolean) => void
  onCancel: () => void
}) {
  const [masks, setMasks] = useState(true)
  const boxes = plan?.layers.reduce((n, layer) => n + layer.boxes.length, 0) ?? 0
  const layers = plan?.layers.length ?? 0

  return (
    <Dialog open={plan !== null} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="!inset-auto !top-[16vh] !left-1/2 w-[min(420px,calc(100vw-1.5rem))] !-translate-x-1/2 !border-0 !bg-transparent !shadow-none">
        <div className="glass glass-glow p-5">
          <DialogTitle className="t-panel">Export GeoJSON</DialogTitle>
          <DialogDescription className="t-meta mt-1">
            {plan?.bounds
              ? 'WGS84 (CRS84). Opens directly in QGIS and ArcGIS Pro.'
              : 'Pixel space — not georeferenced. Coordinates are the 0–1000 image frame, and the file says so.'}
          </DialogDescription>

          <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[12.5px]">
            <dt className="text-text-lo">Boxes</dt>
            <dd className="tabular">
              {countOf(boxes, { one: 'box', other: 'boxes' })}
              {layers > 1 ? ` on ${countOf(layers, { one: 'view', other: 'views' })}` : ''}
            </dd>
            <dt className="text-text-lo">Masks</dt>
            <dd>{plan?.masks.map((m) => m.label).join(', ')}</dd>
          </dl>

          <label className="mt-4 flex items-start gap-3">
            <button
              type="button"
              role="switch"
              aria-checked={masks}
              aria-label="Include segmentation masks"
              onClick={() => setMasks((v) => !v)}
              className="switch mt-0.5"
            />
            <span>
              <span className="block text-[13px] font-medium text-text-hi">Include masks</span>
              <span className="t-meta block">
                Vectorised on this device, one MultiPolygon per class, simplified at half a pixel.
              </span>
            </span>
          </label>

          <div className="mt-5 flex justify-end gap-2">
            <button type="button" onClick={onCancel} className="btn-ghost">
              Cancel
            </button>
            <button type="button" disabled={busy} onClick={() => onConfirm(masks)} className="btn-primary">
              {busy ? 'Building…' : 'Export'}
            </button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
