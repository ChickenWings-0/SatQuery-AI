/**
 * The centre column. Pre-flight until a run starts, then viewer → tray → KPIs.
 *
 * The viewer appears as soon as the *first* artifact arrives rather than waiting
 * for the run to finish, so the wait is spent watching evidence accumulate.
 *
 * The column is a flex chain and every link carries `min-h-0`. Without it the
 * viewer refuses to shrink below its content, overflows the grid cell, and gets
 * painted over the Evidence heading below — which is exactly what it did.
 */
import { useEffect, useMemo, useRef } from 'react'

import { EvidenceTray } from '@/components/stage/EvidenceTray'
import { ImageViewer } from '@/components/stage/ImageViewer'
import { KpiCards } from '@/components/stage/KpiCards'
import { PreflightPanel } from '@/components/stage/PreflightPanel'
import { SceneHeader } from '@/components/stage/SceneHeader'
import { groupViews } from '@/evidence/views'
import { confidenceCard, selectKpis } from '@/kpi/registry'
import { resolutionLabel, sceneMeta } from '@/evidence/scene'
import { useJobStore } from '@/state/job'
import { useFocusStore } from '@/state/focus'
import { useUiStore } from '@/state/ui'

export function DataStage() {
  const phase = useJobStore((state) => state.phase)
  const artifacts = useJobStore((state) => state.artifacts)
  const result = useJobStore((state) => state.result)
  const nodes = useJobStore((state) => state.nodes)

  const activeViewKey = useFocusStore((state) => state.activeViewKey)
  const activeKpiId = useFocusStore((state) => state.activeKpiId)
  const selectView = useFocusStore((state) => state.selectView)
  const setViewKeys = useFocusStore((state) => state.setViewKeys)

  const groups = useMemo(() => groupViews(artifacts), [artifacts])
  /** The viewer's cell, for the header's fullscreen control. */
  const viewerRef = useRef<HTMLDivElement>(null)

  // The hotkey layer walks the tray by key, so it needs the order the tray is
  // actually rendering in — not the artifact arrival order.
  useEffect(() => {
    setViewKeys(groups.map((group) => group.key))
  }, [groups, setViewKeys])
  // "Sentinel-2 · 10 m" under every thumbnail — one scene, one resolution.
  const validation = useUiStore((state) => state.validation)
  const sublabel = useMemo(
    () => resolutionLabel(sceneMeta(validation, groups[0])),
    [validation, groups],
  )

  const cards = useMemo(() => {
    const selected = selectKpis(result?.trace?.fact_sheet)
    return result ? [...selected, confidenceCard(result)] : selected
  }, [result])

  /** KPI ids whose producing step did not run cleanly — §8.6, never hide it. */
  const degradedIds = useMemo(() => {
    const bad = new Set<string>()
    const executions = result?.trace?.executions ?? []
    for (const card of cards) {
      for (const key of card.sourceKeys) {
        // A fact_sheet key is namespaced `<tool>.<scalar>` (§9), but a tool
        // that forgot the namespace would produce `slice(0, -1)` — the whole
        // key minus its last character — and silently match nothing.
        const dot = key.indexOf('.')
        if (dot === -1) continue
        const tool = key.slice(0, dot)
        const execution = executions.find((candidate) => candidate.tool === tool)
        if (execution && execution.status !== 'OK') bad.add(card.id)
      }
    }
    return bad
  }, [cards, result])

  if (phase === 'idle' && groups.length === 0) return <PreflightPanel />

  const active = groups.find((group) => group.key === activeViewKey) ?? groups[0]

  return (
    <div className="flex min-h-full flex-col gap-5">
      <SceneHeader group={active ?? null} viewerRef={viewerRef} />

      {active ? (
        // A floor as well as a ceiling: the viewer must not collapse to nothing
        // on a short window, nor eat the tray and the cards on a tall one.
        // `bg-bg-main` so a fullscreened viewer has a ground, not transparency.
        <div
          ref={viewerRef}
          className="min-h-[clamp(220px,40vh,620px)] flex-1 basis-0 bg-bg-main md:min-h-[clamp(340px,48vh,620px)]"
        >
          <ImageViewer group={active} />
        </div>
      ) : (
        <div className="grid min-h-[clamp(220px,40vh,620px)] flex-1 basis-0 place-items-center rounded-xl border border-dashed border-line px-6 text-center md:min-h-[clamp(340px,48vh,620px)]">
          {/* A failed run leaves this panel empty for good, so it must stop
              claiming that something is still on its way. */}
          <p className="t-meta" role="status">
            {phase === 'failed'
              ? 'The run stopped before any evidence was rendered. The reason is in the Ask column.'
              : nodes.length > 0
                ? 'Rendering evidence…'
                : 'Starting the pipeline…'}
          </p>
        </div>
      )}

      <EvidenceTray
        groups={groups}
        activeKey={active?.key ?? null}
        onSelect={selectView}
        sublabel={sublabel}
      />

      <KpiCards cards={cards} activeId={activeKpiId} degradedIds={degradedIds} />
    </div>
  )
}
