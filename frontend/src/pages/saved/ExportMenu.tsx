/**
 * Export a saved run. GeoJSON and the SITREP are built client-side from the
 * summary plus the re-fetched trace; the PDF report is still the print route
 * (`/report/<trace>`) for the multi-page audit record.
 */
import { GeoJsonExportDialog } from '@/components/export/GeoJsonExportDialog'
import { CopyIcon, DownloadIcon, FileIcon, MapsIcon, SitrepIcon } from '@/components/ui/icons'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useGeoJsonExport } from '@/export/geojson/useGeoJsonExport'
import { useSitrep } from '@/export/sitrep/useSitrep'
import type { SavedRun } from '@/state/library'
import { toast } from '@/state/notifications'
import { useUiStore } from '@/state/ui'

export function ExportMenu({ run, compact = false }: { run: SavedRun; compact?: boolean }) {
  const setSection = useUiStore((state) => state.setSection)
  const geojson = useGeoJsonExport()
  const sitrep = useSitrep()

  async function copyTrace() {
    try {
      await navigator.clipboard.writeText(run.traceId)
      toast('Trace id copied.', 'ok')
    } catch {
      toast('Could not reach the clipboard.', 'warn')
    }
  }

  return (
    <>
      <Popover>
        <PopoverTrigger asChild>
          <button
            type="button"
            aria-label={`Export ${run.query}`}
            className={compact ? 'grid size-7 place-items-center rounded-md text-text-lo hover:bg-sidebar-hi hover:text-text-hi' : 'btn-ghost-sm flex items-center gap-1.5'}
          >
            <DownloadIcon size={14} />
            {compact ? null : 'Export'}
          </button>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-60 p-1.5" aria-label="Export">
          <button type="button" className="menu-row" disabled={sitrep.busy} onClick={() => void sitrep.run({ kind: 'saved', run })}>
            <SitrepIcon size={15} className="text-text-lo" />
            {sitrep.busy ? 'Building SITREP…' : 'SITREP'}
            <span className="t-coord ml-auto text-text-lo">PDF · 1 page</span>
          </button>
          <button type="button" className="menu-row" disabled={geojson.busy} onClick={() => void geojson.start({ kind: 'saved', run })}>
            <FileIcon size={15} className="text-text-lo" />
            {geojson.busy ? 'Building…' : 'GeoJSON'}
            <span className="t-coord ml-auto text-text-lo">{run.bounds ? 'WGS84' : 'px'}</span>
          </button>
          <a
            className="menu-row"
            href={`/report/${run.traceId}`}
            target="_blank"
            rel="noopener"
            aria-label="Full report, opens in a new tab"
          >
            <FileIcon size={15} className="text-text-lo" />
            Full report
            <span aria-hidden className="ml-auto text-text-lo">
              ↗
            </span>
          </a>
          <button type="button" className="menu-row" onClick={() => void copyTrace()}>
            <CopyIcon size={15} className="text-text-lo" />
            Copy trace id
          </button>
          <button
            type="button"
            className="menu-row disabled:opacity-40"
            disabled={!run.bounds}
            title={run.bounds ? undefined : 'This run carries no georeference'}
            onClick={() => setSection('maps')}
          >
            <MapsIcon size={15} className="text-text-lo" />
            Open in Maps
          </button>
        </PopoverContent>
      </Popover>
      <GeoJsonExportDialog plan={geojson.pending} busy={geojson.busy} onConfirm={(masks) => void geojson.confirm(masks)} onCancel={geojson.cancel} />
    </>
  )
}
