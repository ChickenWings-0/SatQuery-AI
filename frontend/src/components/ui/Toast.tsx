/**
 * Transient notifications, bottom-centre of the stage. They come from the
 * notification store's `transient` lane, live six seconds, and carry at most
 * one action (`Undo`, `Retry`). `role="status"`: announced, not interrupting.
 */
import { CloseIcon } from '@/components/ui/icons'
import { StatusDot } from '@/components/ui/StatusDot'
import { useNotificationStore, type NotificationTone } from '@/state/notifications'

const DOT: Record<NotificationTone, 'OK' | 'WARN' | 'FAIL' | 'PENDING'> = {
  ok: 'OK',
  warn: 'WARN',
  fail: 'FAIL',
  info: 'PENDING',
}

export function Toasts() {
  const toasts = useNotificationStore((state) => state.toasts)
  const dismiss = useNotificationStore((state) => state.dismissToast)
  if (toasts.length === 0) return null
  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 bottom-4 z-40 flex flex-col items-center gap-2 px-4"
    >
      {toasts.map((item) => (
        <div
          key={item.id}
          className="glass sq-arrive pointer-events-auto flex max-w-md items-center gap-2.5 py-2 pr-1.5 pl-3 text-[12.5px] text-text-hi"
        >
          <StatusDot status={DOT[item.tone]} />
          <span className="min-w-0 flex-1">{item.title}</span>
          {item.action ? (
            <button
              type="button"
              onClick={() => {
                item.action?.run()
                dismiss(item.id)
              }}
              className="btn-ghost-sm"
            >
              {item.action.label}
            </button>
          ) : null}
          <button
            type="button"
            aria-label="Dismiss"
            onClick={() => dismiss(item.id)}
            className="grid size-7 place-items-center rounded-md text-text-lo hover:text-text-hi"
          >
            <CloseIcon size={14} />
          </button>
        </div>
      ))}
    </div>
  )
}
