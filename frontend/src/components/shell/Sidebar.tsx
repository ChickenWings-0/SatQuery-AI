/**
 * Navigation. A horizontal bar on a phone, the rail column from `wide` up.
 *
 * One component rather than two rendered behind `hidden`/`wide:block`: a
 * duplicated nav means two buttons per section, two `aria-current="page"`
 * markers and two tab stops, and a screen reader does not care which of them
 * CSS is hiding. So the same DOM reflows — `flex-row` to `flex-col` — and only
 * genuinely secondary content is dropped at the narrow end.
 *
 * What the rail holds, top to bottom: the wordmark, the one call to action,
 * the eight sections, the mission line, and the user card. What it no longer
 * holds: the recent-queries list (now the thread's History tab) and the device
 * health strip (now the top bar). Both moved rather than vanished — the line
 * that separates adapting from amputating.
 *
 * The active row carries `.nav-active`: a 3px terracotta bar on its leading
 * edge and a wash of the accent behind it. Drawn with `box-shadow` so the row
 * does not shift by 3px when it becomes active.
 */
import type { ComponentType } from 'react'

import {
  DatasetsIcon,
  ExploreIcon,
  HomeIcon,
  MapsIcon,
  PlusIcon,
  ProjectsIcon,
  SatelliteIcon,
  SavedIcon,
  SettingsIcon,
  ToolsIcon,
  UseCasesIcon,
  type IconProps,
} from '@/components/ui/icons'
import { useFocusStore } from '@/state/focus'
import { NAV_SECTIONS, useUiStore, type NavSection } from '@/state/ui'

const LABELS: Record<NavSection, string> = {
  home: 'Home',
  explore: 'Explore',
  datasets: 'Datasets',
  tools: 'Tools',
  usecases: 'Use Cases',
  maps: 'Maps',
  saved: 'Saved',
  projects: 'Projects',
}

const ICONS: Record<NavSection, ComponentType<IconProps>> = {
  home: HomeIcon,
  explore: ExploreIcon,
  datasets: DatasetsIcon,
  tools: ToolsIcon,
  usecases: UseCasesIcon,
  maps: MapsIcon,
  saved: SavedIcon,
  projects: ProjectsIcon,
}

/**
 * Sections the centre column can actually draw. The rest are scaffolding for
 * the roadmap and route to Explore, so a click never lands on a blank page.
 */
const LIVE: ReadonlySet<NavSection> = new Set(['explore', 'datasets', 'tools'])

export function Sidebar() {
  const section = useUiStore((state) => state.section)
  const setSection = useUiStore((state) => state.setSection)
  const shortcutsEnabled = useUiStore((state) => state.shortcutsEnabled)
  const setShortcutsEnabled = useUiStore((state) => state.setShortcutsEnabled)
  const focusComposer = useFocusStore((state) => state.focusComposer)
  const setDraft = useFocusStore((state) => state.setDraft)

  function newQuery() {
    // The composer is the product's front door; "New Query" is a shortcut to
    // it. It clears the draft — that is what "new" means — and hands over the
    // caret, on the Explore stage where the imagery is.
    setSection('explore')
    setDraft('')
    focusComposer()
  }

  return (
    <aside
      data-region="sidebar"
      className="
        flex items-center gap-3 border-b border-line bg-sidebar px-3 text-sidebar-text
        pt-[env(safe-area-inset-top)]
        wide:h-full wide:min-h-0 wide:flex-col wide:items-stretch wide:gap-0 wide:border-r wide:border-b-0 wide:px-0 wide:pt-0
      "
    >
      {/* Scrolls on its own from `wide` so a landscape phone or a short window
          does not clip the nav off the bottom of the column. */}
      <div className="flex min-w-0 flex-1 items-center gap-3 wide:min-h-0 wide:flex-col wide:items-stretch wide:gap-0 wide:overflow-y-auto wide:px-3 wide:pt-5 wide:pb-4">
        <div className="flex shrink-0 items-center gap-2.5 wide:px-2">
          <span
            aria-hidden
            className="grid size-8 shrink-0 place-items-center rounded-lg bg-accent-warm text-white"
          >
            <SatelliteIcon size={18} />
          </span>
          <div className="min-w-0">
            <h1 className="text-[15px] leading-tight font-semibold tracking-[-0.01em]">
              SatQuery AI
            </h1>
            {/* The tagline is the first thing to go: it is voice, not
                wayfinding, and a phone bar has no room for voice. */}
            <p className="mt-0.5 hidden text-[11px] leading-snug text-sidebar-text-lo wide:block">
              From Space to Answers
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={newQuery}
          className="btn-primary mt-0 hidden shrink-0 items-center gap-2 !px-3 !py-2 wide:mt-5 wide:flex"
        >
          <PlusIcon size={16} />
          <span className="flex-1 text-left">New Query</span>
          <kbd className="kbd !border-white/25 !bg-transparent !text-white/80">⌘K</kbd>
        </button>

        {/*
         * Horizontally scrollable on a phone. Eight items do not fit at 390px
         * and a nav that silently clips its last item is worse than one that
         * scrolls, so the bar scrolls and every item stays reachable.
         */}
        <nav
          className="-mx-1 flex min-w-0 flex-1 gap-1 overflow-x-auto px-1 wide:mx-0 wide:mt-5 wide:flex-none wide:flex-col wide:space-y-0.5 wide:overflow-visible wide:px-0"
          aria-label="Primary"
        >
          {NAV_SECTIONS.map((item) => {
            const Glyph = ICONS[item]
            const active = item === section || (item === 'explore' && section === 'history')
            return (
              <button
                key={item}
                type="button"
                onClick={() => setSection(LIVE.has(item) ? item : 'explore')}
                aria-current={active ? 'page' : undefined}
                className={`flex min-h-10 shrink-0 items-center gap-2 rounded-lg px-2.5 text-left text-[13px] transition-colors wide:w-full wide:gap-3 ${
                  active
                    ? 'nav-active font-medium text-sidebar-text'
                    : 'text-sidebar-text-lo hover:bg-sidebar-hi hover:text-sidebar-text'
                }`}
              >
                <Glyph
                  size={18}
                  className={`shrink-0 ${active ? 'text-accent-warm-text' : ''}`}
                />
                {LABELS[item]}
              </button>
            )
          })}
        </nav>

        {/* The mission line, with the accent bar under it. Voice, so it is
            the second thing dropped on a phone. */}
        <div className="mt-8 hidden px-2.5 wide:block">
          <p className="text-[12px] leading-relaxed text-sidebar-text-lo">
            A clearer planet for a brighter tomorrow
          </p>
          <span aria-hidden className="mt-2.5 block h-0.5 w-8 rounded-full bg-accent-warm" />
        </div>
      </div>

      <div className="shrink-0 wide:space-y-3 wide:border-t wide:border-line wide:px-3 wide:py-3">
        {/* The user card. Static: there is no account system yet, and a card
            that looks like one is the honest placeholder for where it goes. */}
        <div className="flex items-center gap-2 wide:px-1">
          <span
            aria-hidden
            className="grid size-7 shrink-0 place-items-center rounded-full bg-surface-sand text-[11px] font-semibold text-on-evidence"
          >
            A
          </span>
          <div className="hidden min-w-0 flex-1 wide:block">
            <p className="truncate text-[13px] leading-tight font-medium">Aksh</p>
            <p className="text-[10.5px] leading-tight text-sidebar-text-lo">Student · Researcher</p>
          </div>
          <button
            type="button"
            aria-label="Settings"
            className="grid size-7 shrink-0 place-items-center rounded-lg text-sidebar-text-lo transition-colors hover:bg-sidebar-hi hover:text-sidebar-text"
          >
            <SettingsIcon size={16} />
          </button>
        </div>

        {/* WCAG 2.1.4: single-character shortcuts must be switchable off. They
            are, here, under the user card rather than buried in a settings
            screen this product does not have. Hidden on a phone, where there
            is no hardware keyboard to fire them. */}
        <label className="hidden cursor-pointer items-center gap-2 px-2 text-[11px] text-sidebar-text-lo wide:flex">
          <input
            type="checkbox"
            checked={shortcutsEnabled}
            onChange={(event) => setShortcutsEnabled(event.target.checked)}
            className="size-3.5 shrink-0 accent-[var(--color-accent-warm)]"
          />
          <span>Single-key shortcuts</span>
        </label>
      </div>
    </aside>
  )
}
