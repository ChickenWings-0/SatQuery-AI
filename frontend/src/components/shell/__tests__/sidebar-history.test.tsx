/**
 * @vitest-environment happy-dom
 *
 * The rail's "New Query" is a clean slate, and the list beneath it is the
 * library on this device — opened by a click, removed by the trash icon.
 * The IndexedDB adapter is stubbed: the store's in-memory half is what the
 * rail reads, and `persist` is a detail the Saved page's tests own.
 */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { NEW_QUERY_EVENT } from '@/thread/newQuery'
import { useFocusStore } from '@/state/focus'
import { useJobStore } from '@/state/job'
import { useLibraryStore, type SavedRun } from '@/state/library'
import { useUiStore } from '@/state/ui'

vi.mock('idb-keyval', () => ({
  createStore: () => ({}),
  get: async () => undefined,
  set: async () => undefined,
  del: async () => undefined,
}))

import { Sidebar } from '@/components/shell/Sidebar'

function run(traceId: string, query: string, ranAt: number): SavedRun {
  return {
    traceId,
    query,
    taskType: 'VQA',
    pairType: 'SINGLE',
    sensors: ['optical'],
    savedAt: ranAt,
    ranAt,
    outcome: 'succeeded',
    confidence: 0.9,
    headline: null,
    boxes: [],
    bounds: null,
    thumb: null,
    projectId: null,
    tags: [],
  }
}

beforeEach(() => {
  localStorage.clear()
  useLibraryStore.setState({ runs: {}, projects: {}, hydrated: true })
  useUiStore.setState({ section: 'explore', selectedTraceId: null })
})

afterEach(() => {
  cleanup()
  useJobStore.getState().reset()
  useFocusStore.getState().resetForNewRun()
})

describe('the rail', () => {
  it('"New Query" abandons the run and clears the job, focus and upload state', () => {
    const abandoned = vi.fn()
    window.addEventListener(NEW_QUERY_EVENT, abandoned)
    useJobStore.setState({ phase: 'streaming' })
    useFocusStore.setState({ pipelineOpen: true, draft: 'half a question' })
    useUiStore.setState({
      section: 'maps',
      files: [{ file: new File(['x'], 'pre.tif'), previewUrl: 'blob:pre' }],
      validating: true,
    })

    render(<Sidebar />)
    fireEvent.click(screen.getByRole('button', { name: /New Query/ }))

    expect(abandoned).toHaveBeenCalledTimes(1)
    expect(useJobStore.getState().phase).toBe('idle')
    expect(useFocusStore.getState().pipelineOpen).toBe(false)
    expect(useFocusStore.getState().draft).toBe('')
    expect(useUiStore.getState().files).toEqual([])
    expect(useUiStore.getState().validating).toBe(false)
    expect(useUiStore.getState().section).toBe('explore')
    window.removeEventListener(NEW_QUERY_EVENT, abandoned)
  })

  it('lists past runs newest first, opens one on click, and removes one from the trash icon', async () => {
    const now = Date.now()
    useLibraryStore.setState({
      runs: {
        a: run('a', 'How much built-up area changed?', now - 3 * 60_000),
        b: run('b', 'Count the ships in the harbour', now - 5_000),
      },
    })

    render(<Sidebar />)
    const list = await screen.findByRole('list', { name: 'Past queries' })
    const items = list.querySelectorAll('li')
    expect(items).toHaveLength(2)
    expect(items[0]?.textContent).toContain('Count the ships in the harbour')
    expect(items[0]?.textContent).toContain('Just now')
    expect(items[1]?.textContent).toContain('3 min ago')

    fireEvent.click(screen.getByRole('button', { name: /^Open “How much built-up area changed/ }))
    expect(useUiStore.getState().section).toBe('history')
    expect(useUiStore.getState().selectedTraceId).toBe('a')

    fireEvent.click(screen.getByRole('button', { name: /Remove “Count the ships in the harbour”/ }))
    await waitFor(() => expect(useLibraryStore.getState().runs['b']).toBeUndefined())
    await waitFor(() => expect(list.querySelectorAll('li')).toHaveLength(1))
  })

  it('collapses and remembers it', async () => {
    render(<Sidebar />)
    const toggle = screen.getByRole('button', { name: 'Past queries' })
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(localStorage.getItem('satquery.rail.history')).toBe('closed')
    expect(screen.queryByRole('list', { name: 'Past queries' })).toBeNull()
  })
})
