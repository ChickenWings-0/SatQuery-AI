/**
 * The one seam for the SITREP: reads the stores, drives the pure composer,
 * the canvas, and — behind `import()` — the pdf-lib renderer. The button
 * and the menu row both call `run()`; `⌘⇧S` reaches it through the export
 * store. A second click while the chunk is loading is ignored, and the
 * loaded module is kept so the second brief costs no fetch.
 */
import { useCallback, useState } from 'react'

import { trace as fetchTrace } from '@/api/client'
import type { AnalyzeResponse, AuditTrace } from '@/api/types'
import { downloadBlob } from '@/export/download'
import type { SitrepInput } from '@/export/sitrep/compose'
import { useJobStore } from '@/state/job'
import type { SavedRun } from '@/state/library'
import { toast } from '@/state/notifications'
import { useUiStore } from '@/state/ui'

export type SitrepSource = { kind: 'live' } | { kind: 'saved'; run: SavedRun }

/** A brief that has not rendered in this long has hung, and the button must come back. */
const BUILD_DEADLINE_MS = 30_000

function withDeadline<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`the renderer did not finish within ${ms / 1000} s`)), ms)
    promise.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (error: unknown) => {
        clearTimeout(timer)
        reject(error instanceof Error ? error : new Error(String(error)))
      },
    )
  })
}
export type SitrepPhase = 'idle' | 'composing' | 'rendering' | 'done' | 'failed'

let builderPromise: Promise<typeof import('@/export/sitrep/build')> | null = null
function loadBuilder() {
  builderPromise ??= import('@/export/sitrep/build')
  return builderPromise
}

function resultFromTrace(trace: AuditTrace): AnalyzeResponse {
  return {
    trace_id: trace.trace_id,
    answer: trace.answer,
    artifacts: trace.artifacts,
    confidence: trace.confidence,
    compatibility: trace.compatibility,
    resolved_task: trace.resolved_task,
    trace,
  }
}

async function inputFor(source: SitrepSource): Promise<SitrepInput | null> {
  const generatedAt = new Date()
  if (source.kind === 'live') {
    const job = useJobStore.getState()
    const result = job.result
    if (!result) return null
    const ui = useUiStore.getState()
    const query = ui.recentRuns.find((r) => r.traceId === result.trace_id)?.query ?? result.trace?.query.raw ?? ''
    return {
      traceId: result.trace_id,
      query,
      result,
      trace: result.trace ?? null,
      manifests: ui.validation?.inputs ?? [],
      artifacts: job.artifacts,
      generatedAt,
    }
  }
  const run = source.run
  let trace: AuditTrace | null = null
  try {
    trace = await fetchTrace(run.traceId)
  } catch {
    trace = null
  }
  return {
    traceId: run.traceId,
    query: run.query,
    result: trace ? resultFromTrace(trace) : null,
    trace,
    manifests: trace?.inputs ?? [],
    artifacts: trace?.artifacts ?? [],
    saved: run,
    generatedAt,
  }
}

/** Build the PDF bytes for a source. Exported for the e2e harness and tests. */
export async function buildSitrep(source: SitrepSource): Promise<{ name: string; bytes: Uint8Array } | null> {
  const input = await inputFor(source)
  if (!input) return null
  const { buildSitrepFromInput } = await loadBuilder()
  return buildSitrepFromInput(input)
}

export function useSitrep() {
  const [phase, setPhase] = useState<SitrepPhase>('idle')
  const busy = phase === 'composing' || phase === 'rendering'

  const run = useCallback(
    async (source: SitrepSource) => {
      if (busy) return
      setPhase('composing')
      try {
        const input = await inputFor(source)
        if (!input) {
          toast('Nothing to brief yet — run a query first.', 'info')
          setPhase('idle')
          return
        }
        setPhase('rendering')
        const built = await withDeadline(buildSitrep(source), BUILD_DEADLINE_MS)
        if (!built) throw new Error('nothing to render')
        downloadBlob(built.name, new Blob([built.bytes.buffer as ArrayBuffer], { type: 'application/pdf' }))
        setPhase('done')
        toast('SITREP saved.', 'ok')
      } catch (error) {
        setPhase('failed')
        toast(`Could not build the SITREP — ${error instanceof Error ? error.message : 'unknown error'}.`, 'fail')
      }
    },
    [busy],
  )

  return { run, phase, busy }
}
