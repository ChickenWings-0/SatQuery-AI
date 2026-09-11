/**
 * Section routing. Deliberately not a router: there are four sections, no URLs
 * worth deep-linking yet, and adding React Router now would be scaffolding for
 * its own sake.
 */
import { AppShell } from '@/components/shell/AppShell'
import { DatasetsPanel } from '@/components/panels/DatasetsPanel'
import { HistoryPanel } from '@/components/panels/HistoryPanel'
import { ToolsPanel } from '@/components/panels/ToolsPanel'
import { DataStage } from '@/components/stage/DataStage'
import { ThreadPanel } from '@/components/thread/ThreadPanel'
import { useHotkeys } from '@/shell/useHotkeys'
import { useUiStore } from '@/state/ui'

export default function App() {
  const section = useUiStore((state) => state.section)
  useHotkeys()

  const stage =
    section === 'tools' ? (
      <ToolsPanel />
    ) : section === 'datasets' ? (
      <DatasetsPanel />
    ) : section === 'history' ? (
      <HistoryPanel />
    ) : (
      <DataStage />
    )

  return <AppShell stage={stage} thread={<ThreadPanel />} />
}
