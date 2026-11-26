/**
 * The access code, asked for when it is needed — never at the door.
 *
 * Anyone can look through the whole site: the API keeps health and the tools
 * catalogue public. Running anything (reading an upload, fetching imagery,
 * asking a question, opening a run's results) needs the code. The site renders
 * at once; this component probes `/v1/health` to learn whether the API wants a
 * code (`access_required`) and whether a faster mirror is up, and owns the
 * dialog that `requireAccess()` (an action about to run) or a 401 (a held code
 * that went stale) opens. A code is checked against the gated `GET /v1/access`
 * before it is kept; the waiting action then continues on its own.
 */
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'

import {
  clearAccessCode,
  onAccessPrompt,
  onAccessRequired,
  setAccessCode,
  setAccessRequired,
  settleAccess,
  withAccessHeader,
} from '@/api/access'
import { checkAccess, health, SatQueryError, switchToFastBase } from '@/api/client'
import type { HealthResponse } from '@/api/types'
import { DialogOverlay } from '@/components/ui/dialog'

const FAST_PROBE_MS = 4_000

/**
 * Move traffic to the faster mirror `/v1/health` reported, but only once it has
 * answered a health check itself, in time. A mirror that is down or still
 * starting leaves everything on the stable origin.
 */
async function preferFastBase(report: HealthResponse): Promise<void> {
  const fast = report.fast_base_url?.replace(/\/+$/, '')
  if (!fast) return
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), FAST_PROBE_MS)
  try {
    const response = await fetch(`${fast}/v1/health`, {
      ...withAccessHeader({ headers: { accept: 'application/json' } }),
      signal: controller.signal,
    })
    if (response.ok) switchToFastBase(fast)
  } catch {
    // Stay on the stable origin.
  } finally {
    clearTimeout(timer)
  }
}

export function AccessGate({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    health(controller.signal)
      .then(async (report) => {
        setAccessRequired(report.access_required ?? false)
        await preferFastBase(report)
      })
      .catch(() => {
        // The app has its own offline states; there is nothing to gate here.
      })
    const offPrompt = onAccessPrompt(() => setOpen(true))
    const offRequired = onAccessRequired(() => {
      setAccessRequired(true)
      setOpen(true)
    })
    return () => {
      controller.abort()
      offPrompt()
      offRequired()
    }
  }, [])

  function close(granted: boolean) {
    setOpen(false)
    settleAccess(granted)
  }

  return (
    <>
      {children}
      <DialogPrimitive.Root open={open} onOpenChange={(next) => (next ? setOpen(true) : close(false))}>
        <DialogPrimitive.Portal>
          <DialogOverlay />
          <DialogPrimitive.Content
            data-sq-dialog
            className="fixed top-1/2 left-1/2 z-50 w-[min(26rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-line bg-surface-elevated p-6 shadow-[0_24px_64px_-16px_rgb(0_0_0/0.6)] outline-none"
          >
            {open ? <AccessForm onDone={close} /> : null}
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>
    </>
  )
}

function AccessForm({ onDone }: { onDone: (granted: boolean) => void }) {
  const [code, setCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function submit(event: FormEvent) {
    event.preventDefault()
    const trimmed = code.trim()
    if (!trimmed || busy) return
    setBusy(true)
    setMessage(null)
    setAccessCode(trimmed)
    try {
      await checkAccess()
      onDone(true)
    } catch (error) {
      clearAccessCode()
      setMessage(
        error instanceof SatQueryError ? error.display : 'Could not check the code. Try again.',
      )
      setBusy(false)
      inputRef.current?.select()
    }
  }

  return (
    <form onSubmit={(event) => void submit(event)} aria-describedby="access-why">
      <DialogPrimitive.Title className="t-panel text-[17px] text-text-hi">
        Enter the access code to run this
      </DialogPrimitive.Title>
      <DialogPrimitive.Description id="access-why" className="t-meta mt-2 leading-relaxed">
        Anyone can look around. Running an analysis uses a private GPU, so it needs the code you
        were given.
      </DialogPrimitive.Description>
      <label htmlFor="access-code" className="t-meta mt-5 block text-text-hi">
        Access code
      </label>
      <input
        ref={inputRef}
        id="access-code"
        type="password"
        autoComplete="off"
        autoCapitalize="off"
        spellCheck={false}
        // Radix focuses the first focusable element; this is it.
        value={code}
        onChange={(event) => setCode(event.target.value)}
        aria-invalid={message ? true : undefined}
        aria-describedby={message ? 'access-error' : undefined}
        className="mt-1.5 w-full rounded-md border border-line bg-bg-main px-3 py-2 text-[14px] text-text-hi"
      />
      {message ? (
        <p id="access-error" role="alert" className="t-meta mt-2 text-accent-warm-text">
          {message}
        </p>
      ) : null}
      <div className="mt-5 flex items-center justify-end gap-2">
        <DialogPrimitive.Close asChild>
          <button type="button" className="btn-ghost">
            Not now
          </button>
        </DialogPrimitive.Close>
        <button type="submit" disabled={busy || !code.trim()} className="btn-primary">
          {busy ? 'Checking…' : 'Continue'}
        </button>
      </div>
    </form>
  )
}
