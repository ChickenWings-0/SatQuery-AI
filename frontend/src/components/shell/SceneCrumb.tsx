/**
 * The header's left slot, where the dead "Earth View" chip used to sit: the
 * section name and, once a scene is loaded, the scene itself. The bar keeps
 * its grid row (it also spans the thread column and holds the health strip),
 * so the slot gives it a reason to exist rather than a hole.
 */
import { useUiStore, type Section } from '@/state/ui'

const NAMES: Record<Section, string> = {
  home: 'Home',
  explore: 'New Query',
  datasets: 'Datasets',
  tools: 'Tools',
  usecases: 'Use cases',
  maps: 'Maps',
  saved: 'Saved',
  projects: 'Projects',
  history: 'History',
  notFound: 'Not found',
}

export function SceneCrumb() {
  const section = useUiStore((state) => state.section)
  const files = useUiStore((state) => state.files)
  const validation = useUiStore((state) => state.validation)
  const scene = files[0]?.file.name ?? null
  const pair = validation?.compatibility.pair_type ?? null

  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="t-eyebrow shrink-0">{NAMES[section]}</span>
      {scene && (section === 'explore' || section === 'maps') ? (
        <>
          <span aria-hidden className="text-text-lo">
            ›
          </span>
          <span className="truncate text-[13px] font-medium text-text-hi" title={scene}>
            {scene}
          </span>
          {pair ? <span className="chip shrink-0 bg-line text-text-lo">{pair}</span> : null}
        </>
      ) : null}
    </div>
  )
}
