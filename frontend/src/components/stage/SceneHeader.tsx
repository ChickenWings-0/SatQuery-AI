/**
 * The strip above the viewer: what this scene is, and the three things you
 * can do with it.
 *
 *   Urban expansion analysis                    [Share] [Download] [⛶]
 *   Sentinel-2 · EPSG:32643 · 713,620, 3,160,500 · T1 2019-04-12 vs T2 2024-04-08
 *
 * The title is the resolved task, written as a phrase. The metadata line is
 * read from the artifact labels and the pre-flight manifests — every field
 * that is unknown prints as an em dash, so a missing sensor guess cannot turn
 * into a confident-looking wrong one.
 *
 * Share copies a link to this trace; Download saves the primary evidence
 * image; Fullscreen puts the viewer on the whole screen through the native
 * API. None of them needs a backend it does not already have.
 */
import { useEffect, useState, type RefObject } from 'react'

import { artifactUrl } from '@/api/client'
import type { TaskType } from '@/api/types'
import { DownloadIcon, FullscreenIcon, ShareIcon } from '@/components/ui/icons'
import { sceneMeta } from '@/evidence/scene'
import { primaryOf, type ViewGroup } from '@/evidence/views'
import { countOf } from '@/format'
import { isLive, useJobStore } from '@/state/job'
import { useUiStore } from '@/state/ui'

const TITLES: Partial<Record<TaskType, string>> = {
  VQA: 'Scene question',
  CAPTION: 'Scene description',
  GROUNDING: 'Object grounding',
  SEGMENTATION: 'Segmentation',
  COUNT: 'Object count',
  SCENE_CLASSIFY: 'Scene classification',
  CHANGE_VQA: 'Change analysis',
  CHANGE_CAPTION: 'Change description',
  CHANGE_MAP: 'Change mapping',
  CROSS_MODAL_VQA: 'Cross-modal question',
  CROSS_MODAL_COMPARE: 'Cross-modal comparison',
}

function titleFor(task: TaskType | undefined): string {
  if (!task) return 'Analysis'
  return TITLES[task] ?? task.toLowerCase().replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())
}

function IconButton({
  label,
  onClick,
  children,
  done,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
  done?: string | undefined
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={done ?? label}
      onClick={onClick}
      className="grid size-9 place-items-center rounded-lg border border-line bg-surface-card text-text-lo transition-colors hover:border-accent-warm/40 hover:text-text-hi"
    >
      {children}
      {done && (
        <span role="status" className="sr-only">
          {done}
        </span>
      )}
    </button>
  )
}

export function SceneHeader({
  group,
  viewerRef,
}: {
  group: ViewGroup | null
  viewerRef: RefObject<HTMLDivElement | null>
}) {
  const result = useJobStore((state) => state.result)
  const phase = useJobStore((state) => state.phase)
  const stage = useJobStore((state) => state.stage)
  const nodes = useJobStore((state) => state.nodes)
  const artifacts = useJobStore((state) => state.artifacts)
  const validation = useUiStore((state) => state.validation)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1800)
    return () => clearTimeout(timer)
  }, [copied])

  const meta = sceneMeta(validation, group)
  const task = result?.resolved_task.primary
  const primary = group ? primaryOf(group) : null

  async function share() {
    // The trace id is the one durable handle on a run — History and the
    // pipeline modal both key on it — so that is what gets shared.
    const url = new URL(window.location.href)
    if (result?.trace_id) url.searchParams.set('trace', result.trace_id)
    try {
      await navigator.clipboard.writeText(url.toString())
      setCopied(true)
    } catch {
      // Clipboard access is a permission; without it the button is inert
      // rather than broken, and the URL bar already holds the link.
    }
  }

  function download() {
    const href = primary ? artifactUrl(primary) : null
    if (!primary || !href) return
    const anchor = document.createElement('a')
    anchor.href = href
    anchor.download = `${primary.id}.png`
    anchor.rel = 'noopener'
    anchor.click()
  }

  function fullscreen() {
    const element = viewerRef.current
    if (!element) return
    if (document.fullscreenElement) void document.exitFullscreen()
    else void element.requestFullscreen?.()
  }

  const fields: string[] = [
    meta.sensor ?? '—',
    meta.crs ? `${meta.crs}${meta.centre ? ` · ${meta.centre}` : ''}` : (meta.centre ?? '—'),
    meta.t2 ? `T1 ${meta.t1 ?? '—'} vs T2 ${meta.t2}` : `T1 ${meta.t1 ?? '—'}`,
  ]

  return (
    <header className="flex shrink-0 flex-wrap items-start gap-x-4 gap-y-3">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2 className="t-page">{titleFor(task)}</h2>
          {task && <span className="chip border border-line text-text-lo">{task}</span>}
          {/* The only place a screen-reader user learns that a run has
              started, changed stage or finished — the rest of the change is
              pixels. */}
          <span role="status" className="t-meta">
            {phase === 'reconnecting'
              ? 'reconnecting…'
              : isLive(phase)
                ? `${stage ?? 'running'}…`
              : phase === 'failed'
                ? 'Run stopped'
                : `${countOf(nodes.length, { one: 'step', other: 'steps' })} · ${countOf(artifacts.length, { one: 'artifact', other: 'artifacts' })}`}
          </span>
        </div>
        <p className="tabular mt-1 flex flex-wrap items-center gap-x-2 font-mono text-[11px] text-text-lo">
          {fields.map((field, index) => (
            <span key={index} className="flex items-center gap-x-2">
              {index > 0 && (
                <span aria-hidden className="text-line-soft">
                  ·
                </span>
              )}
              <span className="[overflow-wrap:anywhere]">{field}</span>
            </span>
          ))}
        </p>
      </div>

      <div className="flex shrink-0 items-center gap-1.5">
        <IconButton label="Copy link to this run" onClick={() => void share()} done={copied ? 'Link copied' : undefined}>
          <ShareIcon size={16} />
        </IconButton>
        <IconButton label="Download the current view" onClick={download}>
          <DownloadIcon size={16} />
        </IconButton>
        <IconButton label="Toggle fullscreen" onClick={fullscreen}>
          <FullscreenIcon size={16} />
        </IconButton>
      </div>
    </header>
  )
}
