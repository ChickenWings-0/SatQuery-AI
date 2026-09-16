/**
 * The evidence tray's export control. The label carries the CRS so nobody
 * loads a pixel-space file into QGIS believing it is georeferenced.
 */
import { GeoJsonExportDialog } from '@/components/export/GeoJsonExportDialog'
import { FileIcon } from '@/components/ui/icons'
import { sceneGeoref } from '@/evidence/georef'
import { geoJsonLabel, useGeoJsonExport } from '@/export/geojson/useGeoJsonExport'
import { useExportRequest } from '@/state/exports'
import { useUiStore } from '@/state/ui'

export function ExportGeoJsonButton() {
  const validation = useUiStore((state) => state.validation)
  const bounds = sceneGeoref(validation)?.bounds ?? null
  const { start, confirm, cancel, pending, busy } = useGeoJsonExport()
  useExportRequest('geojson', () => void start({ kind: 'live' }))

  return (
    <>
      <button
        type="button"
        disabled={busy}
        onClick={() => void start({ kind: 'live' })}
        aria-keyshortcuts="Meta+Shift+G Control+Shift+G"
        className="btn-ghost-sm inline-flex items-center gap-1.5"
      >
        <FileIcon size={13} />
        {busy ? 'Building…' : geoJsonLabel(bounds)}
      </button>
      <GeoJsonExportDialog plan={pending} busy={busy} onConfirm={(masks) => void confirm(masks)} onCancel={cancel} />
    </>
  )
}
