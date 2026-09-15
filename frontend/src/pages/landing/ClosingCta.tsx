import { ArrowRightIcon } from '@/components/ui/icons'
import { Graticule } from '@/components/ui/Graticule'
import { useUiStore } from '@/state/ui'

export function ClosingCta() {
  const setSection = useUiStore((state) => state.setSection)
  return (
    <section aria-labelledby="close-title" className="relative overflow-hidden py-28">
      {/* The grid drifts under the copy at a slower rate than the page
          (`[data-closing-grid]` in theme.css); the fade stays put. */}
      <div aria-hidden data-closing-grid className="absolute inset-x-0 -inset-y-10" style={{ maskImage: 'linear-gradient(to bottom, black 20%, transparent 100%)' }}>
        <Graticule fade={false} />
      </div>
      <div className="relative mx-auto max-w-[1280px] px-4 text-center wide:px-6" data-reveal>
        <h2 id="close-title" className="t-section text-text-hi">
          Bring your own scene.
        </h2>
        <p className="t-lede mx-auto mt-4">
          GeoTIFF, PNG or JPEG. One image or a pair. The pre-flight tells you what it can answer
          before anything runs.
        </p>
        <button type="button" onClick={() => setSection('explore')} className="btn-primary group mt-8 inline-flex h-11 items-center gap-2 !px-5 text-[15px]">
          Open the console
          <ArrowRightIcon size={15} className="transition-transform duration-[120ms] group-hover:translate-x-0.5" />
        </button>
      </div>
    </section>
  )
}

export function LandingFooter() {
  const setSection = useUiStore((state) => state.setSection)
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-[1280px] flex-wrap items-center gap-x-6 gap-y-3 px-4 py-6 text-[12.5px] text-text-lo wide:px-6">
        <p className="font-semibold text-text-hi">SatQuery AI</p>
        <p>Smart India Hackathon 2026 · PS 26167 · ISRO / SAC</p>
        <nav aria-label="Footer" className="ml-auto flex items-center gap-4">
          <a href="/use-cases" onClick={(event) => { event.preventDefault(); setSection('usecases') }} className="hover:text-text-hi">Use cases</a>
          <a href="/maps" onClick={(event) => { event.preventDefault(); setSection('maps') }} className="hover:text-text-hi">Maps</a>
          <a href="/tools" onClick={(event) => { event.preventDefault(); setSection('tools') }} className="hover:text-text-hi">Tools</a>
        </nav>
      </div>
    </footer>
  )
}
