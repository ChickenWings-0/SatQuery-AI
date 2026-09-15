import { ThemeButton } from '@/components/shell/ThemeToggle'
import { ArrowRightIcon, SatelliteIcon } from '@/components/ui/icons'
import { useUiStore } from '@/state/ui'

export function LandingNav() {
  const setSection = useUiStore((state) => state.setSection)
  return (
    <div className="sticky top-3 z-30 px-4 wide:px-6">
      <nav aria-label="Landing" className="glass mx-auto flex h-14 max-w-[1280px] items-center gap-3 px-3 wide:px-4">
        <a href="/" onClick={(event) => { event.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }) }} className="flex items-center gap-2.5 rounded-lg">
          <span aria-hidden className="grid size-8 place-items-center rounded-lg bg-accent-warm text-white">
            <SatelliteIcon size={18} />
          </span>
          <span className="text-[15px] font-semibold text-text-hi">SatQuery AI</span>
        </a>
        <div className="ml-6 hidden items-center gap-5 text-[13px] text-text-lo wide:flex">
          <a href="/use-cases" onClick={(event) => { event.preventDefault(); setSection('usecases') }} className="nav-underline hover:text-text-hi">Use cases</a>
          <a href="/maps" onClick={(event) => { event.preventDefault(); setSection('maps') }} className="nav-underline hover:text-text-hi">Maps</a>
          <a href="#story" className="nav-underline hover:text-text-hi">How it works</a>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <ThemeButton />
          <button type="button" onClick={() => setSection('explore')} className="btn-ghost flex h-9 items-center gap-1.5 !py-0">
            Open console
            <ArrowRightIcon size={14} />
          </button>
        </div>
      </nav>
    </div>
  )
}
