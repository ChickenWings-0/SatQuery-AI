/**
 * Selection → console. The backend clips the scenes and answers with file
 * URLs; the files are downloaded with a byte count (the wait is real and
 * said so), wrapped as `File`s, and handed to the *same* `selectFiles` the
 * dropzone uses. A fetched scene is exactly the demo flow, not a special
 * path: pre-flight is already running when the console appears.
 */
import { useCallback, useState } from 'react'

import { fetchImagery, imageryUrl, SatQueryError } from '@/api/client'
import type { ImageryFile } from '@/api/types'
import type { StacItem } from '@/geo/stac'
import { useFocusStore } from '@/state/focus'
import { toast } from '@/state/notifications'
import { useStacStore, selectedItems } from '@/state/stac'
import { useUiStore } from '@/state/ui'

export type FetchPhase = 'idle' | 'clipping' | 'downloading' | 'done' | 'failed'

export interface FetchProgress {
  phase: FetchPhase
  done: number
  total: number
  loadedBytes: number
  totalBytes: number
  error: string | null
}

const IDLE: FetchProgress = { phase: 'idle', done: 0, total: 0, loadedBytes: 0, totalBytes: 0, error: null }

async function download(file: ImageryFile, onBytes: (delta: number) => void, signal: AbortSignal): Promise<File> {
  const response = await fetch(imageryUrl(file), { signal })
  if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${file.name}`)
  if (!response.body) {
    const blob = await response.blob()
    onBytes(blob.size)
    return new File([blob], file.name, { type: 'image/tiff' })
  }
  const reader = response.body.getReader()
  const chunks: BlobPart[] = []
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    onBytes(value.byteLength)
  }
  return new File(chunks, file.name, { type: 'image/tiff' })
}

/** The question the console opens with, from what was picked. */
export function suggestedQuery(t1: StacItem, t2: StacItem | null): string {
  if (t2) return t1.sensor === 'sar' ? 'What changed between these two SAR acquisitions?' : 'What changed between these two images?'
  if (t1.sensor === 'sar') return 'What does the backscatter say about the surface here?'
  return 'Is this area built-up?'
}

export function useImageryFetch() {
  const [progress, setProgress] = useState<FetchProgress>(IDLE)
  const busy = progress.phase === 'clipping' || progress.phase === 'downloading'

  const run = useCallback(async () => {
    if (busy) return
    const state = useStacStore.getState()
    const { t1, t2, issue } = selectedItems(state)
    if (!t1 || issue || !state.query.bbox) return
    const items = [t1, ...(t2 ? [t2] : [])]
    const controller = new AbortController()
    setProgress({ ...IDLE, phase: 'clipping', total: items.length })
    try {
      const result = await fetchImagery(
        { items: items.map((item) => ({ collection: item.collection, id: item.id })), bbox: state.query.bbox },
        controller.signal,
      )
      const totalBytes = result.files.reduce((n, f) => n + f.size_bytes, 0)
      let loadedBytes = 0
      let done = 0
      setProgress({ phase: 'downloading', done, total: result.files.length, loadedBytes, totalBytes, error: null })
      const files: File[] = []
      for (const file of result.files) {
        files.push(
          await download(
            file,
            (delta) => {
              loadedBytes += delta
              setProgress((p) => ({ ...p, loadedBytes }))
            },
            controller.signal,
          ),
        )
        done += 1
        setProgress((p) => ({ ...p, done }))
      }
      for (const warning of result.warnings ?? []) toast(warning.message, 'warn')
      // T1 before T2, whatever order the server answered in.
      const ordered = items.map((item) => files.find((f) => f.name.includes(item.id.slice(0, 40))) ?? files[0]!)
      useFocusStore.getState().setDraft(suggestedQuery(t1, t2))
      useUiStore.getState().setSection('explore')
      void useUiStore.getState().selectFiles(ordered)
      setProgress({ ...IDLE, phase: 'done' })
      toast(t2 ? 'Two scenes loaded — pre-flight is running.' : 'Scene loaded — pre-flight is running.', 'ok')
    } catch (error) {
      const message =
        error instanceof SatQueryError
          ? error.display
          : error instanceof Error
            ? error.message
            : 'The fetch failed.'
      setProgress({ ...IDLE, phase: 'failed', error: message })
    }
  }, [busy])

  const reset = useCallback(() => setProgress(IDLE), [])
  return { run, reset, progress, busy }
}
