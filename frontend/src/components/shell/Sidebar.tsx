/**
 * Navigation. A horizontal bar on a phone, the rail column from `wide` up.
 *
 * One component rather than two rendered behind `hidden`/`wide:block`: a
 * duplicated nav means two buttons per section, two `aria-current="page"`
 * markers and two tab stops, and a screen reader does not care which of them
 * CSS is hiding. So the same DOM reflows — `flex-row` to `flex-col` — and only
 * genuinely secondary content is dropped at the narrow end.
 *
 * What the rail holds, top to bottom: the wordmark, the nav, the mission line,
 * and the user card. The nav opens with the product's one call to action —
 * "New Query", the way into the workspace — followed by the six sections.
 * There is no "Home" row: the wordmark is the way to the landing page and it
 * is on every screen, so a row saying the same thing was one link twice. And
 * there is no separate "Explore" row under the call to action: the workspace
 * is where a new query goes, so the button *is* the row, and carries
 * `aria-current` when the workspace is showing.
 *
 * The active row carries `.nav-active`: a 3px terracotta bar on its leading
 * edge and a wash of the accent behind it. Drawn with `box-shadow` so the row
 * does not shift by 3px when it becomes active.
 */
import { lazy, Suspense, useState, type ComponentType } from 'react'

import { AccountPopover } from '@/components/shell/AccountPopover'
import { DailyQuote } from '@/components/shell/DailyQuote'
import {
  ChevronDownIcon,
  DatasetsIcon,
  MapsIcon,
  PlusIcon,
  ProjectsIcon,
  SatelliteIcon,
  SavedIcon,
  ToolsIcon,
  UseCasesIcon,
  type IconProps,
} from '@/components/ui/icons'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'
import { NAV_SECTIONS, useUiStore, type NavSection } from '@/state/ui'
import { NEW_QUERY_EVENT } from '@/thread/newQuery'

// The past-runs list needs the library store and its IndexedDB adapter, which
// the entry chunk does not carry; it loads the first time the disclosure opens.
const SidebarHistory = lazy(() => import('@/components/shell/SidebarHistory'))

const HISTORY_OPEN_KEY = 'satquery.rail.history'

function readHistoryOpen(): boolean {
  try {
    return localStorage.getItem(HISTORY_OPEN_KEY) !== 'closed'
  } catch {
    return true
  }
}

const LABELS: Record<NavSection, string> = {
  datasets: 'Datasets',
  tools: 'Tools',
  usecases: 'Use Cases',
  maps: 'Maps',
  saved: 'Saved',
  projects: 'Projects',
}

const ICONS: Record<NavSection, ComponentType<IconProps>> = {
  datasets: DatasetsIcon,
  tools: ToolsIcon,
  usecases: UseCasesIcon,
  maps: MapsIcon,
  saved: SavedIcon,
  projects: ProjectsIcon,
}

/**
 * Routes that prefetch their chunk on intent: hovering or focusing the nav
 * button starts the `import()` 150 ms before the click, which in practice
 * removes the Suspense fallback from the transition.
 */
const PREFETCH: Partial<Record<NavSection, () => Promise<unknown>>> = {
  usecases: () => import('@/pages/UseCases'),
  maps: () => import('@/pages/Maps'),
  saved: () => import('@/pages/Saved'),
  projects: () => import('@/pages/Projects'),
}

const prefetchLanding = () => import('@/pages/Landing')

export function Sidebar() {
  const section = useUiStore((state) => state.section)
  const setSection = useUiStore((state) => state.setSection)
  const focusComposer = useFocusStore((state) => state.focusComposer)
  const setDraft = useFocusStore((state) => state.setDraft)

  // The workspace and its full-page history are one place as far as the nav
  // is concerned: both are "the query you are working on".
  const inWorkspace = section === 'explore' || section === 'history'

  const [historyOpen, setHistoryOpen] = useState(readHistoryOpen)
  function toggleHistory() {
    const next = !historyOpen
    setHistoryOpen(next)
    try {
      localStorage.setItem(HISTORY_OPEN_KEY, next ? 'open' : 'closed')
    } catch {
      // Session-only, then.
    }
  }

  function newQuery() {
    // The composer is the product's front door; "New Query" is a shortcut to
    // it. "New" means a clean slate: a run still streaming is abandoned on
    // both ends (`useRun` listens for the event), the answer, DAG and
    // evidence are cleared, the uploaded scenes and their pre-flight go, the
    // draft is emptied — and the caret lands on the workspace stage where the
    // imagery is. Saved runs are untouched; they are in the list below.
    window.dispatchEvent(new Event(NEW_QUERY_EVENT))
    useJobStore.getState().reset()
    useFocusStore.getState().resetForNewRun()
    useUiStore.getState().clearFiles()
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
        {/* The wordmark is the way home. A `<p>`, not an `<h1>`: the page
            title is the heading, and this is on every page. */}
        <button
          type="button"
          onClick={() => setSection('home')}
          onPointerEnter={() => void prefetchLanding()}
          onFocus={() => void prefetchLanding()}
          aria-label="SatQuery AI — go to the landing page"
          className="flex shrink-0 items-center gap-2.5 rounded-lg text-left transition-colors duration-[120ms] hover:bg-sidebar-hi wide:-mx-2 wide:px-2 wide:py-1"
        >
          <span
            aria-hidden
            className="grid size-8 shrink-0 place-items-center rounded-lg bg-accent-warm text-white"
          >
            <SatelliteIcon size={18} />
          </span>
          <span className="min-w-0">
            <span className="block text-[15px] leading-tight font-semibold tracking-[-0.01em]">
              SatQuery AI
            </span>
            <span className="mt-0.5 hidden text-[11px] leading-snug text-sidebar-text-lo wide:block">
              From Space to Answers
            </span>
          </span>
        </button>

        {/*
         * Horizontally scrollable on a phone. Seven items do not fit at 390px
         * and a nav that silently clips its last item is worse than one that
         * scrolls, so the bar scrolls and every item stays reachable.
         */}
        <nav
          className="-mx-1 flex min-w-0 flex-1 items-center gap-1 overflow-x-auto px-1 wide:mx-0 wide:mt-5 wide:flex-none wide:flex-col wide:items-stretch wide:space-y-0.5 wide:overflow-visible wide:px-0"
          aria-label="Primary"
        >
          {/* The call to action is the first nav item, not a control above
              the nav: it is how the workspace is reached, so it belongs in
              the list of places, and it is the only filled button in the
              column. Once the workspace is showing the fill stays — it is
              still the way to start over — and `aria-current` says so; a
              visible ring on top of the fill read as a stuck focus ring. */}
          <button
            type="button"
            onClick={newQuery}
            aria-current={inWorkspace ? 'page' : undefined}
            className="btn-primary flex min-h-10 shrink-0 items-center gap-2 !px-3 !py-2 wide:mb-4 wide:w-full"
          >
            <PlusIcon size={16} />
            <span className="flex-1 text-left">New Query</span>
            <kbd className="kbd hidden !border-white/25 !bg-transparent !text-white/80 wide:inline-flex">
              ⌘K
            </kbd>
          </button>

          {/* Past runs, directly under the call to action. A disclosure, not a
              popover: on the rail there is room, and a list that is simply
              there is one fewer thing to open. Rail-only — the phone bar is
              a single row and the thread's History tab covers it there. */}
          <div className="hidden wide:-mt-2 wide:mb-3 wide:block">
            <button
              type="button"
              onClick={toggleHistory}
              aria-expanded={historyOpen}
              aria-controls="rail-history"
              className="flex min-h-8 w-full items-center gap-2 rounded-md px-2.5 text-left text-[11px] font-medium tracking-[0.06em] text-sidebar-text-lo uppercase transition-colors duration-[120ms] hover:bg-sidebar-hi hover:text-sidebar-text"
            >
              <span className="flex-1">Past queries</span>
              <ChevronDownIcon
                size={14}
                className={`shrink-0 transition-transform duration-[220ms] ease-[var(--ease-out-quint)] ${historyOpen ? '' : '-rotate-90'}`}
              />
            </button>
            {historyOpen ? (
              <div id="rail-history" className="mt-0.5">
                <Suspense
                  fallback={
                    <ul className="space-y-1 px-1" aria-busy="true" aria-label="Loading past queries">
                      {Array.from({ length: 3 }, (_, i) => (
                        <li key={i} className="h-8 animate-pulse rounded-md bg-sidebar-hi" />
                      ))}
                    </ul>
                  }
                >
                  <SidebarHistory />
                </Suspense>
              </div>
            ) : null}
          </div>

          {NAV_SECTIONS.map((item) => {
            const Glyph = ICONS[item]
            const active = item === section
            return (
              <button
                key={item}
                type="button"
                onClick={() => setSection(item)}
                onPointerEnter={() => void PREFETCH[item]?.()}
                onFocus={() => void PREFETCH[item]?.()}
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

        {/* One sourced line a day. Voice, so it is the second thing dropped
            on a phone. */}
        <DailyQuote />
      </div>

      <div className="shrink-0 wide:border-t wide:border-line wide:px-2 wide:py-2">
        {/* The user card opens the account popover: name, role, storage,
            settings, the shortcuts switch, theme, and sign-out. */}
        <AccountPopover />
      </div>
    </aside>
  )
}
