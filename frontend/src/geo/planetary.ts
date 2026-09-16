/**
 * Planetary Computer's SAS token, for reading an asset in the browser.
 *
 * The console does not read COGs itself — the backend does, with its own
 * token — but thumbnails that need signing and any future in-page preview go
 * through here. One token per collection, refreshed a minute before expiry.
 */
import { STAC_ROOT } from '@/geo/collections'
import { fetchWithDeadline } from '@/geo/deadline'

const SAS_ROOT = STAC_ROOT.replace(/\/stac\/v1$/, '/sas/v1/token')

interface Token {
  token: string
  expiresAt: number
}

const tokens = new Map<string, Token>()

export async function sasToken(collection: string, signal?: AbortSignal): Promise<string> {
  const cached = tokens.get(collection)
  if (cached && cached.expiresAt - 60_000 > Date.now()) return cached.token
  const response = await fetchWithDeadline(`${SAS_ROOT}/${collection}`, { headers: { accept: 'application/json' } }, signal)
  if (!response.ok) throw new Error(`Token request failed (${response.status}).`)
  const payload = (await response.json()) as { token: string; 'msft:expiry'?: string }
  const expiry = payload['msft:expiry'] ? Date.parse(payload['msft:expiry']) : NaN
  const token = { token: payload.token, expiresAt: Number.isFinite(expiry) ? expiry : Date.now() + 3_600_000 }
  tokens.set(collection, token)
  return token.token
}

export async function signedHref(href: string, collection: string, signal?: AbortSignal): Promise<string> {
  const token = await sasToken(collection, signal)
  return `${href}${href.includes('?') ? '&' : '?'}${token}`
}
