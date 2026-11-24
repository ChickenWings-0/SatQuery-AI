/**
 * The shared access code a guarded SatQuery API requires (`satquery.api.access`).
 *
 * The server is the gate; this module only remembers what the visitor typed
 * and attaches it. `fetch` calls carry it in a header. URLs the browser loads
 * on its own — `<img src>`, a download link — cannot set headers, so they
 * carry it as `?code=` instead.
 *
 * Any 401 from the API means the stored code is missing, wrong or rotated:
 * {@link reportAccessRequired} forgets it and tells the gate to ask again.
 */
import { safeStorage } from '@/shell/storage'

export const ACCESS_HEADER = 'X-SatQuery-Code'
const STORAGE_KEY = 'satquery.access-code'
const QUERY_PARAM = 'code'

const listeners = new Set<() => void>()

/**
 * The code for this page load. Storage is only the cross-visit memory: where
 * it throws (a Safari private window, blocked site data) the code must still
 * stick for the session, or the gate would accept it and immediately ask again.
 */
let current: string | null = null

export function getAccessCode(): string | null {
  const code = current ?? safeStorage.getItem(STORAGE_KEY)
  return code && code.trim() ? code.trim() : null
}

export function setAccessCode(code: string): void {
  current = code.trim() || null
  if (current) safeStorage.setItem(STORAGE_KEY, current)
  else safeStorage.removeItem(STORAGE_KEY)
}

export function clearAccessCode(): void {
  current = null
  safeStorage.removeItem(STORAGE_KEY)
}

/** `init` with the access header added, or `init` untouched when there is no code. */
export function withAccessHeader(init: RequestInit): RequestInit {
  const code = getAccessCode()
  if (!code) return init
  return {
    ...init,
    headers: { ...(init.headers as Record<string, string> | undefined), [ACCESS_HEADER]: code },
  }
}

/** `url` with `?code=` appended, for URLs the browser fetches without our headers. */
export function withAccessCode(url: string): string {
  const code = getAccessCode()
  if (!code) return url
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}${QUERY_PARAM}=${encodeURIComponent(code)}`
}

/** Subscribe to "the API wants a code". Returns the unsubscribe function. */
export function onAccessRequired(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** Forget the stored code and ask the gate to show itself. */
export function reportAccessRequired(): void {
  clearAccessCode()
  for (const listener of [...listeners]) listener()
}
