/**
 * The access code reaches every way the app talks to the API: headers on
 * `fetch`, `?code=` on URLs an `<img>` loads, and a 401 sends the gate back up.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ACCESS_HEADER,
  clearAccessCode,
  getAccessCode,
  onAccessPrompt,
  onAccessRequired,
  requireAccess,
  setAccessCode,
  setAccessRequired,
  settleAccess,
  withAccessCode,
  withAccessHeader,
} from '@/api/access'
import { artifactUrl, health } from '@/api/client'

beforeEach(() => clearAccessCode())
afterEach(() => {
  clearAccessCode()
  vi.unstubAllGlobals()
})

describe('access code storage', () => {
  it('trims and remembers the code', () => {
    setAccessCode('  hunter2  ')
    expect(getAccessCode()).toBe('hunter2')
  })

  it('treats a blank code as none', () => {
    setAccessCode('   ')
    expect(getAccessCode()).toBeNull()
  })
})

describe('withAccessHeader', () => {
  it('leaves the request untouched without a code', () => {
    const init = { headers: { accept: 'application/json' } }
    expect(withAccessHeader(init)).toBe(init)
  })

  it('adds the header alongside existing ones', () => {
    setAccessCode('k')
    expect(withAccessHeader({ headers: { accept: 'x' } }).headers).toEqual({
      accept: 'x',
      [ACCESS_HEADER]: 'k',
    })
  })
})

describe('withAccessCode', () => {
  it('appends the code as a query parameter, escaped', () => {
    setAccessCode('a b&c')
    expect(withAccessCode('/v1/artifacts/t/a.png')).toBe('/v1/artifacts/t/a.png?code=a%20b%26c')
    expect(withAccessCode('/x?y=1')).toBe('/x?y=1&code=a%20b%26c')
  })

  it('reaches artifact URLs', () => {
    setAccessCode('k')
    expect(artifactUrl({ url: '/v1/artifacts/t/a.png' } as never)).toMatch(/\/v1\/artifacts\/t\/a\.png\?code=k$/)
  })
})

describe('client', () => {
  it('sends the code on API requests', async () => {
    setAccessCode('k')
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await health()
    const init = fetchMock.mock.calls[0]![1] as RequestInit
    expect((init.headers as Record<string, string>)[ACCESS_HEADER]).toBe('k')
  })

  it('forgets the code and raises the gate on a 401', async () => {
    setAccessCode('stale')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 401 })))
    const listener = vi.fn()
    const unsubscribe = onAccessRequired(listener)
    await expect(health()).rejects.toMatchObject({ status: 401 })
    unsubscribe()
    expect(listener).toHaveBeenCalledOnce()
    expect(getAccessCode()).toBeNull()
  })
})

describe('requireAccess', () => {
  it('lets an action run when the API needs no code', async () => {
    setAccessRequired(false)
    await expect(requireAccess()).resolves.toBe(true)
  })

  it('lets an action run when a code is already held', async () => {
    setAccessRequired(true)
    setAccessCode('k')
    await expect(requireAccess()).resolves.toBe(true)
  })

  it('asks for a code and waits for the dialog to settle', async () => {
    setAccessRequired(true)
    const prompted = vi.fn()
    const off = onAccessPrompt(prompted)
    const waiting = requireAccess()
    expect(prompted).toHaveBeenCalledOnce()
    settleAccess(true)
    await expect(waiting).resolves.toBe(true)
    const dismissed = requireAccess()
    settleAccess(false)
    await expect(dismissed).resolves.toBe(false)
    off()
    setAccessRequired(false)
  })
})
