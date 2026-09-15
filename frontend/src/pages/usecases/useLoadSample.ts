/**
 * Fetch a use case's rasters into the console.
 *
 * The files are downloaded with a byte count against the row's declared size
 * (honest about the wait), wrapped as `File`s, and handed to the *same*
 * `selectFiles` the dropzone uses — so playing a use case is exactly the demo
 * flow, not a special path. The question lands in the composer, pre-flight
 * is already running when the stage appears.
 */
import { useCallback, useState } from 'react'

import { sampleUrl, type UseCase } from '@/pages/usecases/catalogue'
import { useFocusStore } from '@/state/focus'
import { toast } from '@/state/notifications'
import { useUiStore } from '@/state/ui'

export interface SampleProgress {
  slug: string
  loadedBytes: number
  totalBytes: number
}

async function download(
  url: string,
  onBytes: (delta: number) => void,
  signal: AbortSignal,
): Promise<Blob> {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  const type = response.headers.get('content-type') ?? 'image/tiff'
  if (!response.body) {
    const blob = await response.blob()
    onBytes(blob.size)
    return blob
  }
  const reader = response.body.getReader()
  const chunks: BlobPart[] = []
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    onBytes(value.byteLength)
  }
  return new Blob(chunks, { type })
}

export function useLoadSample() {
  const [progress, setProgress] = useState<SampleProgress | null>(null)
  const [error, setError] = useState<{ slug: string; message: string } | null>(null)
  const loading = useUiStore((state) => state.loadingSample)
  const setLoadingSample = useUiStore((state) => state.setLoadingSample)

  const load = useCallback(
    async (useCase: UseCase) => {
      if (useUiStore.getState().loadingSample) return
      setError(null)
      setLoadingSample(useCase.slug)
      const totalBytes = Math.round(useCase.sizeMb * 1024 * 1024)
      let loadedBytes = 0
      setProgress({ slug: useCase.slug, loadedBytes, totalBytes })
      const controller = new AbortController()
      try {
        const files: File[] = []
        for (const name of useCase.files) {
          const blob = await download(
            sampleUrl(useCase, name),
            (delta) => {
              loadedBytes += delta
              setProgress({ slug: useCase.slug, loadedBytes, totalBytes })
            },
            controller.signal,
          )
          files.push(new File([blob], `${useCase.slug}_${name}`, { type: blob.type }))
        }
        void useUiStore.getState().selectFiles(files)
        useFocusStore.getState().resetForNewRun()
        useUiStore.getState().setSection('explore')
        useFocusStore.getState().proposeQuestion(useCase.question)
        try {
          if (window.matchMedia('(pointer: coarse)').matches) navigator.vibrate?.(8)
        } catch {
          // No haptics, then.
        }
        toast(`${useCase.title} loaded — pre-flight is running.`, 'ok')
      } catch (caught) {
        const message = caught instanceof Error ? caught.message : 'unknown error'
        setError({
          slug: useCase.slug,
          message: /HTTP 404/.test(message)
            ? 'Sample rasters are not installed on this build.'
            : `Sample unavailable (${message}).`,
        })
      } finally {
        setLoadingSample(null)
        setProgress(null)
      }
    },
    [setLoadingSample],
  )

  return { load, loading, progress, error }
}
