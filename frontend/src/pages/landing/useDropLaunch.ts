/**
 * The whole hero is a drop target. While a file drag is over the window the
 * store says so (the overlay and the satellites react); a drop runs the same
 * `selectFiles` the console's dropzone uses and lands on the workspace (New Query) with
 * pre-flight already running.
 */
import { useEffect } from 'react'

import { useLandingStore } from '@/state/landing'
import { useUiStore } from '@/state/ui'

const ACCEPT = /\.(tiff?|png|jpe?g)$/i

export function launchWithFiles(list: FileList | File[] | null): void {
  if (!list) return
  const files = [...list].filter((file) => ACCEPT.test(file.name) || /^image\//.test(file.type))
  if (files.length === 0) return
  void useUiStore.getState().selectFiles(files)
  useUiStore.getState().setSection('explore')
}

export function useDropLaunch(): void {
  const setDragOver = useLandingStore((state) => state.setDragOver)
  useEffect(() => {
    let depth = 0
    const hasFiles = (event: DragEvent) => Boolean(event.dataTransfer?.types.includes('Files'))
    const enter = (event: DragEvent) => {
      if (!hasFiles(event)) return
      depth += 1
      setDragOver(true)
    }
    const leave = (event: DragEvent) => {
      if (!hasFiles(event)) return
      depth = Math.max(0, depth - 1)
      if (depth === 0) setDragOver(false)
    }
    const over = (event: DragEvent) => {
      if (hasFiles(event)) event.preventDefault()
    }
    const drop = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      depth = 0
      setDragOver(false)
      launchWithFiles(event.dataTransfer?.files ?? null)
    }
    window.addEventListener('dragenter', enter)
    window.addEventListener('dragleave', leave)
    window.addEventListener('dragover', over)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('dragover', over)
      window.removeEventListener('drop', drop)
      setDragOver(false)
    }
  }, [setDragOver])
}
