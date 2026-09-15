/**
 * Section routing. Deliberately not a router: `@/shell/router` syncs the
 * store's `section` with the URL in forty lines, and this file switches on
 * it. Every page that carries a heavy dependency (`three`, `maplibre-gl`,
 * IndexedDB) is lazy, so the console entry stays small; each boundary has a
 * final-size fallback and an error boundary, so a chunk that 404s after a
 * deploy is a `Reload` button rather than a white page.
 */
import { lazy, Suspense } from 'react'

import { KeyboardShortcutsModal } from '@/components/shell/KeyboardShortcutsModal'
import { AppShell } from '@/components/shell/AppShell'
import { DatasetsPanel } from '@/components/panels/DatasetsPanel'
import { HistoryPanel } from '@/components/panels/HistoryPanel'
import { ToolsPanel } from '@/components/panels/ToolsPanel'
import { DataStage } from '@/components/stage/DataStage'
import { ThreadPanel } from '@/components/thread/ThreadPanel'
import { ErrorBoundary } from '@/components/ui/ErrorBoundary'
import { Graticule } from '@/components/ui/Graticule'
import { SectionSkeleton } from '@/components/ui/Skeleton'
import { Toasts } from '@/components/ui/Toast'
import { DocumentMeta } from '@/shell/DocumentMeta'
import { useHotkeys } from '@/shell/useHotkeys'
import { useSettingsStore } from '@/state/settings'
import { useUiStore } from '@/state/ui'

const Landing = lazy(() => import('@/pages/Landing'))
const UseCases = lazy(() => import('@/pages/UseCases'))
const Maps = lazy(() => import('@/pages/Maps'))
const Saved = lazy(() => import('@/pages/Saved'))
const Projects = lazy(() => import('@/pages/Projects'))
const NotFound = lazy(() => import('@/pages/NotFound'))
const Report = lazy(() => import('@/pages/Report'))
const SettingsDialog = lazy(() => import('@/components/shell/SettingsDialog'))

/**
 * The overlays every shell variant carries: the shortcuts guide, settings and
 * toasts. Settings is fetched on first open and mounted only while open, so
 * the console entry pays nothing for it and Radix's entrance plays each time.
 */
function ShellOverlays() {
  const settingsOpen = useSettingsStore((state) => state.open)
  return (
    <>
      <KeyboardShortcutsModal />
      {settingsOpen ? (
        <Suspense fallback={null}>
          <SettingsDialog />
        </Suspense>
      ) : null}
      <Toasts />
    </>
  )
}

function LandingSkeleton() {
  return (
    <div className="relative min-h-full bg-bg-main" aria-busy="true" aria-label="Loading">
      <Graticule />
      <div className="glass mx-4 mt-3 h-14 wide:mx-6" />
    </div>
  )
}

export default function App() {
  const section = useUiStore((state) => state.section)
  useHotkeys()

  const isReport = typeof window !== 'undefined' && window.location.pathname.startsWith('/report/')

  if (isReport) {
    return (
      <ErrorBoundary title="Report">
        <Suspense fallback={<SectionSkeleton title="report" />}>
          <Report />
        </Suspense>
      </ErrorBoundary>
    )
  }

  if (section === 'home') {
    return (
      <>
        <ErrorBoundary title="Landing page">
          <Suspense fallback={<LandingSkeleton />}>
            <Landing />
          </Suspense>
        </ErrorBoundary>
        <ShellOverlays />
      </>
    )
  }

  const stage =
    section === 'tools' ? (
      <>
        <DocumentMeta page="tools" />
        <ToolsPanel />
      </>
    ) : section === 'datasets' ? (
      <>
        <DocumentMeta page="datasets" />
        <DatasetsPanel />
      </>
    ) : section === 'history' ? (
      <>
        <DocumentMeta page="history" />
        <HistoryPanel />
      </>
    ) : section === 'usecases' ? (
      <UseCases />
    ) : section === 'maps' ? (
      <Maps />
    ) : section === 'saved' ? (
      <Saved />
    ) : section === 'projects' ? (
      <Projects />
    ) : section === 'notFound' ? (
      <NotFound />
    ) : (
      <>
        <DocumentMeta page="explore" />
        <DataStage />
      </>
    )

  const title =
    section === 'usecases' ? 'Use cases' : section === 'maps' ? 'Maps' : section === 'saved' ? 'Saved' : section === 'projects' ? 'Projects' : 'This view'

  return (
    <>
      <a href="#main" className="skip-link">
        Skip to main content
      </a>
      <AppShell
        bleed={section === 'maps'}
        stage={
          <ErrorBoundary title={title}>
            <Suspense fallback={section === 'maps' ? <Graticule /> : <SectionSkeleton title={title} />}>
              {stage}
            </Suspense>
          </ErrorBoundary>
        }
        thread={<ThreadPanel />}
      />
      <ShellOverlays />
    </>
  )
}
