import { beforeEach, describe, expect, it, vi } from 'vitest'

import { toast, unreadCount, useNotificationStore } from '@/state/notifications'

beforeEach(() => {
  useNotificationStore.setState({ items: [], toasts: [], open: false, arrivals: 0 })
})

describe('the notification store', () => {
  it('lists real items newest first and counts unread', () => {
    const { push } = useNotificationStore.getState()
    push({ kind: 'run', tone: 'ok', title: 'one' })
    push({ kind: 'run', tone: 'fail', title: 'two' })
    const state = useNotificationStore.getState()
    expect(state.items.map((i) => i.title)).toEqual(['two', 'one'])
    expect(unreadCount(state)).toBe(2)
    expect(state.arrivals).toBe(2)
  })

  it('caps at fifty', () => {
    for (let i = 0; i < 60; i++) useNotificationStore.getState().push({ kind: 'run', tone: 'ok', title: `${i}` })
    expect(useNotificationStore.getState().items).toHaveLength(50)
  })

  it('never lists a toast, and expires it', () => {
    vi.useFakeTimers()
    toast('saved')
    expect(useNotificationStore.getState().items).toHaveLength(0)
    expect(useNotificationStore.getState().toasts).toHaveLength(1)
    vi.advanceTimersByTime(6_500)
    expect(useNotificationStore.getState().toasts).toHaveLength(0)
    vi.useRealTimers()
  })

  it('marks everything read on close', () => {
    useNotificationStore.getState().push({ kind: 'health', tone: 'warn', title: 'x' })
    useNotificationStore.getState().setOpen(true)
    useNotificationStore.getState().setOpen(false)
    expect(unreadCount(useNotificationStore.getState())).toBe(0)
  })
})
