/**
 * Export a saved run. GeoJSON is generated client-side from the boxes; the
 * PDF report is the print route (`/report/<trace>`) with a print stylesheet —
 * no PDF library, the browser's own "Save as PDF" is the deliverable.
 */
import { boxesToGeoJson, downloadJson } from '@/export/geojson'
import { CopyIcon, DownloadIcon, FileIcon, MapsIcon } from '@/components/ui/icons'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import type { SavedRun } from '@/state/library'
import { toast } from '@/state/notifications'
import { useUiStore } from '@/state/ui'

export function ExportMenu({ run, compact = false }: { run: SavedRun; compact?: boolean }) {
  const setSection = useUiStore((state) => state.setSection)

  function geojson() {
    downloadJson(
      `satquery-${run.traceId.slice(0, 8)}.geojson`,
      boxesToGeoJson(run.boxes, run.bounds, { traceId: run.traceId, query: run.query }),
    )
    toast(run.bounds ? 'GeoJSON exported in WGS84.' : 'GeoJSON exported in pixel space (no georeference).', 'ok')
  }

  async function copyTrace() {
    try {
      await navigator.clipboard.writeText(run.traceId)
      toast('Trace id copied.', 'ok')
    } catch {
      toast('Could not reach the clipboard.', 'warn')
    }
  }

  return (
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
      <PopoverContent align="end" className="w-56 p-1.5" aria-label="Export">
        <button type="button" className="menu-row" onClick={geojson}>
          <FileIcon size={15} className="text-text-lo" />
          GeoJSON
          <span className="t-coord ml-auto text-text-lo">{run.bounds ? 'WGS84' : 'px'}</span>
        </button>
        <a
          className="menu-row"
          href={`/report/${run.traceId}`}
          target="_blank"
          rel="noopener"
          aria-label="PDF report, opens in a new tab"
        >
          <FileIcon size={15} className="text-text-lo" />
          PDF report
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
  )
}
