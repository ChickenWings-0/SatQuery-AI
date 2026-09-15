/**
 * URL ↔ section, without a router library.
 *
 * `App.tsx` has always switched on a store field rather than a route, and
 * still does. What changed is that the landing page needs an address a judge
 * can type, and the sections deserve the back button. Forty lines of
 * `history` sync buy both; React Router would be scaffolding for its own
 * sake.
 *
 * The section store is the source of truth. `setSection` pushes a history
 * entry (the store is subscribed here, so no caller has to remember), and
 * `popstate` sets the section back. Unknown paths resolve to `notFound` —
 * the shell still renders, so nobody lands on a blank page.
 */
import { NAV_SECTIONS, useUiStore, type Section } from '@/state/ui'

export const PATHS: Record<Section, string> = {
  home: '/',
  explore: '/explore',
  datasets: '/datasets',
  tools: '/tools',
  usecases: '/use-cases',
  maps: '/maps',
  saved: '/saved',
  projects: '/projects',
  history: '/history',
  notFound: '/404',
}

const BY_PATH = new Map<string, Section>(
  (Object.entries(PATHS) as [Section, string][]).map(([section, path]) => [path, section]),
)

/** Strip a trailing slash (but not the root) and the query/hash. */
export function normalisePath(path: string): string {
  const bare = path.split(/[?#]/)[0] ?? '/'
  if (bare.length > 1 && bare.endsWith('/')) return bare.slice(0, -1)
  return bare || '/'
}

export function sectionFromPath(path: string): Section {
  const clean = normalisePath(path)
  if (clean === '/404') return 'notFound'
  const direct = BY_PATH.get(clean)
  if (direct) return direct
  // `/projects/<id>` and `/report/<id>` belong to their section pages, which
  // read the id from the location themselves.
  if (clean.startsWith('/projects/')) return 'projects'
  if (clean.startsWith('/report/')) return 'saved'
  return 'notFound'
}

export function pathFor(section: Section): string {
  return PATHS[section]
}

export function isNavSection(section: Section): boolean {
  return (NAV_SECTIONS as readonly string[]).includes(section)
}

/**
 * Start the sync. Returns the unbind function. Safe to call in a test: with
 * no `window`, it does nothing.
 */
export function bindRouter(): () => void {
  if (typeof window === 'undefined') return () => undefined

  const initial = sectionFromPath(window.location.pathname)
  useUiStore.setState({
    section: initial,
    unknownPath: initial === 'notFound' ? window.location.pathname : null,
  })
  // Normalise a trailing slash without adding a history entry.
  const clean = normalisePath(window.location.pathname)
  if (clean !== window.location.pathname && initial !== 'notFound') {
    window.history.replaceState(null, '', clean + window.location.search)
  }

  let applying = false

  const unsubscribe = useUiStore.subscribe((state, previous) => {
    if (applying || state.section === previous.section) return
    const next = pathFor(state.section)
    // A section page that owns a deeper path (`/projects/<id>`) pushes it
    // itself; only push when the current location is not already inside it.
    const here = normalisePath(window.location.pathname)
    if (here === next || (next !== '/' && here.startsWith(next + '/'))) return
    window.history.pushState({ section: state.section }, '', next)
  })

  const onPop = () => {
    applying = true
    const section = sectionFromPath(window.location.pathname)
    useUiStore.setState({
      section,
      unknownPath: section === 'notFound' ? window.location.pathname : null,
    })
    applying = false
  }
  window.addEventListener('popstate', onPop)

  return () => {
    unsubscribe()
    window.removeEventListener('popstate', onPop)
  }
}
