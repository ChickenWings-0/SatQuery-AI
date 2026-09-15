/**
 * The notification centre's store.
 *
 * Everything in it is a real event the app observed — a run settling while
 * the user was elsewhere, the health poll changing state, the library
 * changing, a basemap that could not be reached. Nothing is synthesised to
 * make the bell look busy. Session-only: a notification about a run that
 * finished yesterday is noise, not news.
 *
 * Two lanes share the store. Listed items sit behind the bell until read;
 * `transient` items are toasts — shown for six seconds at the bottom of the
 * stage and never listed — for confirmations that need no archive.
 */
import { create } from 'zustand'

export type NotificationKind = 'run' | 'health' | 'device' | 'library' | 'maps'
export type NotificationTone = 'ok' | 'warn' | 'fail' | 'info'

export interface Notification {
  id: string
  kind: NotificationKind
  tone: NotificationTone
  title: string
  body?: string
  at: number
  read: boolean
  action?: { label: string; run: () => void }
  transient?: boolean
}

export type NotificationInput = Omit<Notification, 'id' | 'at' | 'read'>

const CAP = 50
const TOAST_MS = 6_000

let counter = 0

interface NotificationState {
  items: Notification[]
  toasts: Notification[]
  open: boolean
  /** Bumped on each unread arrival so the bell can pulse once, not loop. */
  arrivals: number
  push: (input: NotificationInput) => string
  markAllRead: () => void
  dismiss: (id: string) => void
  dismissToast: (id: string) => void
  clear: () => void
  setOpen: (open: boolean) => void
}

export const useNotificationStore = create<NotificationState>((set, get) => ({
  items: [],
  toasts: [],
  open: false,
  arrivals: 0,

  push: (input) => {
    const id = `n-${Date.now().toString(36)}-${(counter += 1)}`
    const item: Notification = { ...input, id, at: Date.now(), read: false }
    if (input.transient) {
      set((state) => ({ toasts: [...state.toasts, item] }))
      setTimeout(() => get().dismissToast(id), TOAST_MS)
      return id
    }
    set((state) => ({
      items: [item, ...state.items].slice(0, CAP),
      arrivals: state.arrivals + 1,
    }))
    return id
  },

  markAllRead: () =>
    set((state) => ({ items: state.items.map((item) => ({ ...item, read: true })) })),

  dismiss: (id) => set((state) => ({ items: state.items.filter((item) => item.id !== id) })),

  dismissToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((item) => item.id !== id) })),

  clear: () => set({ items: [] }),

  setOpen: (open) => {
    set({ open })
    if (!open) get().markAllRead()
  },
}))

export function unreadCount(state: NotificationState): number {
  return state.items.reduce((n, item) => n + (item.read ? 0 : 1), 0)
}

/** Convenience for producers outside React. */
export function notify(input: NotificationInput): string {
  return useNotificationStore.getState().push(input)
}

export function toast(title: string, tone: NotificationTone = 'info', action?: Notification['action']): string {
  return notify({ kind: 'library', tone, title, transient: true, ...(action ? { action } : {}) })
}
