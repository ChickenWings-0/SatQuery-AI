/**
 * The single local identity.
 *
 * There is no auth. The profile card is honest about that: the name and role
 * are editable, persisted on this device, and "Sign out & clear this device"
 * does exactly what it says — wipes the local library and preferences — which
 * is the correct meaning of signing out of a device-local product.
 */
import { create } from 'zustand'

import { safeStorage } from '@/shell/storage'

const KEY = 'satquery.account'

export interface Account {
  name: string
  role: string
}

const DEFAULT: Account = { name: 'Aksh', role: 'Student · Researcher' }

function read(): Account {
  const raw = safeStorage.getItem(KEY)
  if (!raw) return DEFAULT
  try {
    const parsed = JSON.parse(raw) as Partial<Account>
    return {
      name: typeof parsed.name === 'string' && parsed.name.trim() ? parsed.name : DEFAULT.name,
      role: typeof parsed.role === 'string' && parsed.role.trim() ? parsed.role : DEFAULT.role,
    }
  } catch {
    return DEFAULT
  }
}

interface AccountState extends Account {
  setName: (name: string) => void
  setRole: (role: string) => void
  reset: () => void
}

export const useAccountStore = create<AccountState>((set, get) => ({
  ...read(),
  setName: (name) => {
    const next = name.trim() || DEFAULT.name
    set({ name: next })
    safeStorage.setItem(KEY, JSON.stringify({ name: next, role: get().role }))
  },
  setRole: (role) => {
    const next = role.trim() || DEFAULT.role
    set({ role: next })
    safeStorage.setItem(KEY, JSON.stringify({ name: get().name, role: next }))
  },
  reset: () => {
    safeStorage.removeItem(KEY)
    set(DEFAULT)
  },
}))

/** The avatar initial: first grapheme of the first word, upper-cased. */
export function initialOf(name: string): string {
  return (name.trim().split(/\s+/)[0] ?? '?').slice(0, 1).toUpperCase()
}
