/**
 * An address that points at nothing. Rendered inside the shell — the sidebar
 * still works; the user is lost, not locked out — with `noindex` so a crawler
 * that followed a bad link does not index the page (an SPA cannot send 404).
 */
import { Graticule } from '@/components/ui/Graticule'
import { ArrowRightIcon } from '@/components/ui/icons'
import { Reticle } from '@/components/ui/Reticle'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { useUiStore } from '@/state/ui'

export default function NotFound() {
  const path = useUiStore((state) => state.unknownPath)
  const setSection = useUiStore((state) => state.setSection)
  const canGoBack = typeof window !== 'undefined' && window.history.length > 1

  return (
    <div className="relative flex min-h-full items-center justify-center overflow-hidden rounded-xl">
      <DocumentMeta page="notFound" />
      <Graticule major />
      <Reticle
        size={120}
        breathe
        className="absolute top-[18%] left-[62%] -translate-x-1/2 -translate-y-1/2"
      />
      <div className="relative max-w-md px-6 py-16 text-center">
        <h1 className="t-page">We can’t find that page.</h1>
        <p className="t-meta mt-2">
          The address may be wrong, or the run it pointed at was never saved on this device.
        </p>
        {path ? <p className="t-coord mt-3 text-text-lo">{path}</p> : null}
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2.5">
          <button type="button" onClick={() => setSection('explore')} className="btn-primary flex items-center gap-2">
            Open the console
            <ArrowRightIcon size={14} />
          </button>
          {canGoBack ? (
            <button type="button" onClick={() => window.history.back()} className="btn-ghost">
              Back
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}
