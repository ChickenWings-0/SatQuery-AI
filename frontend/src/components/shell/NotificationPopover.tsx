/**
 * The bell and what is behind it.
 *
 * Everything listed is an event the app observed — a run settling while the
 * user was elsewhere, the health poll changing state, a basemap that could
 * not be reached. Unread rows carry a 2px sand bar on the leading edge (the
 * rail's own active-row idiom), the badge pulses once per arrival, and
 * closing the popover marks everything read.
 */
import { RadarPulse } from '@/components/shell/RadarPulse'
import { BellIcon, CloseIcon } from '@/components/ui/icons'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { StatusDot } from '@/components/ui/StatusDot'
import { relativeTime } from '@/format'
import { unreadCount, useNotificationStore, type NotificationTone } from '@/state/notifications'

const DOT: Record<NotificationTone, 'OK' | 'WARN' | 'FAIL' | 'PENDING'> = {
  ok: 'OK',
  warn: 'WARN',
  fail: 'FAIL',
  info: 'PENDING',
}

export function NotificationPopover() {
  const items = useNotificationStore((state) => state.items)
  const unread = useNotificationStore(unreadCount)
  const arrivals = useNotificationStore((state) => state.arrivals)
  const open = useNotificationStore((state) => state.open)
  const setOpen = useNotificationStore((state) => state.setOpen)
  const markAllRead = useNotificationStore((state) => state.markAllRead)
  const dismiss = useNotificationStore((state) => state.dismiss)
  const clear = useNotificationStore((state) => state.clear)

  // The badge pulses once per arrival: keyed on the arrival count, the span
  // remounts and its single-run keyframe plays again.

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={unread > 0 ? `Notifications, ${unread} unread` : 'Notifications'}
          className="relative grid size-9 place-items-center rounded-full border border-line bg-surface-card text-text-lo transition-colors duration-[120ms] hover:border-accent-warm/40 hover:text-text-hi"
        >
          <BellIcon size={16} />
          {unread > 0 ? (
            <span
              key={arrivals}
              aria-hidden
              data-badge-new=""
              className="absolute top-1.5 right-1.5 size-2 rounded-full bg-accent-warm ring-2 ring-surface-card"
            />
          ) : null}
        </button>
      </PopoverTrigger>

      <PopoverContent align="end" className="w-[320px] p-0" aria-label="Notifications">
        <header className="flex items-center justify-between px-3.5 py-2.5">
          <h2 className="t-panel">Notifications</h2>
          {items.length > 0 ? (
            <button type="button" onClick={markAllRead} className="t-meta hover:text-text-hi">
              Mark all read
            </button>
          ) : null}
        </header>

        {items.length === 0 ? (
          <div className="border-t border-line-soft px-6 py-7 text-center">
            <RadarPulse />
            <p className="t-panel mt-4">Zero unread alerts.</p>
            <p className="t-meta mt-1">
              Run outcomes, device health and library changes will land here.
            </p>
          </div>
        ) : (
          <ul className="max-h-[min(60vh,420px)] overflow-y-auto border-t border-line-soft">
            {items.map((item) => (
              <li
                key={item.id}
                className={`group relative flex gap-2.5 px-3.5 py-2.5 transition-colors duration-[120ms] hover:bg-accent-glow ${
                  item.read ? '' : 'shadow-[inset_2px_0_0_var(--color-surface-sand)]'
                }`}
              >
                <StatusDot status={DOT[item.tone]} className="mt-1.5" />
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] leading-snug font-medium text-text-hi">{item.title}</p>
                  {item.body ? <p className="t-meta line-clamp-1">{item.body}</p> : null}
                  {item.action ? (
                    <button
                      type="button"
                      onClick={() => {
                        item.action?.run()
                        setOpen(false)
                      }}
                      className="mt-1.5 text-[12px] font-medium text-accent-warm-text underline-offset-2 hover:underline"
                    >
                      {item.action.label}
                    </button>
                  ) : null}
                </div>
                <span className="t-coord shrink-0 pt-1 text-text-lo">{relativeTime(item.at)}</span>
                <button
                  type="button"
                  aria-label="Dismiss"
                  onClick={() => dismiss(item.id)}
                  className="absolute top-2 right-2 grid size-6 place-items-center rounded-md text-text-lo opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100 hover:text-text-hi"
                >
                  <CloseIcon size={12} />
                </button>
              </li>
            ))}
          </ul>
        )}

        {items.length > 0 ? (
          <footer className="border-t border-line-soft px-3.5 py-2">
            <button type="button" onClick={clear} className="t-meta hover:text-text-hi">
              Clear
            </button>
          </footer>
        ) : null}
      </PopoverContent>
    </Popover>
  )
}
