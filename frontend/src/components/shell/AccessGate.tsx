/**
 * Asks for the access code before the app renders, when the API requires one.
 *
 * Self-configuring: it probes `/v1/health` once. A 200 means the API is open
 * (local dev, the MSW mock, a server without `SATQUERY_ACCESS_CODE`) or the
 * stored code still works, and the app renders straight away. A 401 shows the
 * form. Any other failure — API down, network gone — also renders the app,
 * which has its own offline states; the gate only answers "who are you".
 *
 * A 401 later in the session (code rotated on the server) brings the form
 * back through `onAccessRequired`.
 */
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'

import { onAccessRequired, setAccessCode, clearAccessCode, withAccessHeader } from '@/api/access'
import { health, SatQueryError, switchToFastBase } from '@/api/client'
import type { HealthResponse } from '@/api/types'
import { Graticule } from '@/components/ui/Graticule'

type GateState = 'checking' | 'open' | 'locked'

function isAccessDenied(error: unknown): boolean {
  return error instanceof SatQueryError && error.status === 401
}

const FAST_PROBE_MS = 4_000

/**
 * Move traffic to the faster mirror `/v1/health` reported, but only once it
 * has answered a health check itself, with the code, in time. A mirror that
 * is down or still starting leaves everything on the stable origin.
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
  const [state, setState] = useState<GateState>('checking')

  useEffect(() => {
    const controller = new AbortController()
    health(controller.signal)
      .then(async (report) => {
        await preferFastBase(report)
        if (!controller.signal.aborted) setState('open')
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return
        setState(isAccessDenied(error) ? 'locked' : 'open')
      })
    const unsubscribe = onAccessRequired(() => setState('locked'))
    return () => {
      controller.abort()
      unsubscribe()
    }
  }, [])

  if (state === 'open') return <>{children}</>
  if (state === 'checking') {
    return <div className="min-h-full bg-bg-main" aria-busy="true" aria-label="Loading" />
  }
  return <AccessForm onUnlocked={() => setState('open')} />
}

function AccessForm({ onUnlocked }: { onUnlocked: () => void }) {
  const [code, setCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  async function submit(event: FormEvent) {
    event.preventDefault()
    const trimmed = code.trim()
    if (!trimmed || busy) return
    setBusy(true)
    setMessage(null)
    setAccessCode(trimmed)
    try {
      const report = await health()
      await preferFastBase(report)
      onUnlocked()
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
    <main className="relative flex min-h-full items-center justify-center overflow-hidden bg-bg-main">
      <Graticule major />
      <form
        onSubmit={(event) => void submit(event)}
        className="glass relative mx-4 w-full max-w-sm rounded-xl px-6 py-7"
        aria-labelledby="access-title"
      >
        <h1 id="access-title" className="t-page">
          SatQuery AI
        </h1>
        <p className="t-meta mt-2">
          This demo runs on a private GPU. Enter the access code you were given to continue.
        </p>
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
        <button
          type="submit"
          disabled={busy || !code.trim()}
          className="btn-primary mt-5 w-full justify-center"
        >
          {busy ? 'Checking…' : 'Continue'}
        </button>
      </form>
    </main>
  )
}
