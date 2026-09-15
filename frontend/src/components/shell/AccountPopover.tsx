/**
 * The account popover, anchored to the user card at the foot of the rail.
 *
 * There is no auth. The card is honest about it: the name and role are
 * editable in place, the storage quota is the browser's real estimate, and
 * "Sign out & clear this device" wipes the local library and preferences —
 * which is what signing out of a device-local product means. The full
 * preferences — instructions, seed, data controls, appearance — are one row
 * away in the settings dialog (`⌘,`); the popover keeps the quick things.
 */
import { useEffect, useRef, useState } from 'react'

import { ThemeSegmented } from '@/components/shell/ThemeToggle'
import { StorageQuota } from '@/components/shell/StorageQuota'
import { Graticule } from '@/components/ui/Graticule'
import { FolderIcon, KeyboardIcon, SettingsIcon, SignOutIcon } from '@/components/ui/icons'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { initialOf, useAccountStore } from '@/state/account'
import { useLibraryStore } from '@/state/library'
import { toast } from '@/state/notifications'
import { useSettingsStore } from '@/state/settings'
import { useShortcutStore } from '@/state/shortcuts'
import { useUiStore } from '@/state/ui'

function Kbd({ keys }: { keys: readonly string[] }) {
  return (
    <span className="ml-auto flex items-center gap-0.5">
      {keys.map((key) => (
        <kbd key={key} className="kbd">
          {key}
        </kbd>
      ))}
    </span>
  )
}

export function AccountPopover() {
  const name = useAccountStore((state) => state.name)
  const role = useAccountStore((state) => state.role)
  const setName = useAccountStore((state) => state.setName)
  const setRole = useAccountStore((state) => state.setRole)
  const resetAccount = useAccountStore((state) => state.reset)
  const openGuide = useShortcutStore((state) => state.openGuide)
  const openSettings = useSettingsStore((state) => state.openSettings)
  const setSection = useUiStore((state) => state.setSection)
  const clearLibrary = useLibraryStore((state) => state.clearAll)

  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const nameRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) nameRef.current?.focus()
  }, [editing])

  async function signOut() {
    await clearLibrary()
    resetAccount()
    try {
      localStorage.clear()
    } catch {
      // Nothing to clear, then.
    }
    setOpen(false)
    toast('This device has been cleared.', 'ok')
  }

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next)
        if (!next) {
          setEditing(false)
          setConfirming(false)
        }
      }}
    >
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={`Account: ${name}, ${role}`}
          className="flex w-full items-center gap-2 rounded-lg text-left transition-colors duration-[120ms] hover:bg-sidebar-hi wide:px-1 wide:py-1"
        >
          <span
            aria-hidden
            className="grid size-7 shrink-0 place-items-center rounded-full bg-surface-sand text-[11px] font-semibold text-on-evidence"
          >
            {initialOf(name)}
          </span>
          <span className="hidden min-w-0 flex-1 wide:block">
            <span className="block truncate text-[13px] leading-tight font-medium">{name}</span>
            <span className="block truncate text-[10.5px] leading-tight text-sidebar-text-lo">{role}</span>
          </span>
          <span className="hidden size-7 shrink-0 place-items-center rounded-lg text-sidebar-text-lo wide:grid">
            <SettingsIcon size={16} />
          </span>
        </button>
      </PopoverTrigger>

      <PopoverContent side="top" align="start" className="w-[280px] p-0" aria-label="Account">
        <div className="relative overflow-hidden rounded-t-[var(--radius-hud)] px-3.5 pt-3.5 pb-3">
          <Graticule fade={false} module={24} className="opacity-60" />
          <div className="relative flex items-start gap-3">
            <span
              aria-hidden
              className="grid size-8 shrink-0 place-items-center rounded-full bg-surface-sand text-[12.5px] font-semibold text-on-evidence"
            >
              {initialOf(name)}
            </span>
            <div className="min-w-0 flex-1">
              {editing ? (
                <form
                  className="space-y-1.5"
                  onSubmit={(event) => {
                    event.preventDefault()
                    setEditing(false)
                  }}
                >
                  <input
                    ref={nameRef}
                    aria-label="Display name"
                    defaultValue={name}
                    onBlur={(event) => setName(event.target.value)}
                    className="w-full rounded-md border border-line bg-bg-main px-2 py-1 text-[13px] text-text-hi"
                  />
                  <input
                    aria-label="Role"
                    defaultValue={role}
                    onBlur={(event) => setRole(event.target.value)}
                    className="w-full rounded-md border border-line bg-bg-main px-2 py-1 text-[11.5px] text-text-lo"
                  />
                  <button type="submit" className="btn-ghost-sm">
                    Done
                  </button>
                </form>
              ) : (
                <>
                  <p className="t-panel truncate">{name}</p>
                  <p className="t-meta">
                    {role}
                    <button
                      type="button"
                      onClick={() => setEditing(true)}
                      className="ml-2 text-accent-warm-text underline-offset-2 hover:underline"
                    >
                      edit
                    </button>
                  </p>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="border-t border-line-soft px-3.5 py-3">
          <p className="t-eyebrow mb-2">Storage on this device</p>
          <StorageQuota />
        </div>

        <nav aria-label="Account" className="border-t border-line-soft p-1.5">
          <button
            type="button"
            className="menu-row"
            onClick={() => {
              setOpen(false)
              openSettings()
            }}
          >
            <SettingsIcon size={16} className="text-text-lo" />
            Settings
            <Kbd keys={['⌘', ',']} />
          </button>
          <button
            type="button"
            className="menu-row"
            onClick={() => {
              setOpen(false)
              openGuide()
            }}
          >
            <KeyboardIcon size={16} className="text-text-lo" />
            Keyboard shortcuts
            <Kbd keys={['⌘', '/']} />
          </button>
          <button
            type="button"
            className="menu-row"
            onClick={() => {
              setOpen(false)
              setSection('projects')
            }}
          >
            <FolderIcon size={16} className="text-text-lo" />
            Workspaces &amp; projects
            <Kbd keys={['G', 'P']} />
          </button>
          <div className="menu-row">
            <span className="flex-1">Theme</span>
            <ThemeSegmented />
          </div>
        </nav>

        <div className="border-t border-line-soft p-1.5">
          {confirming ? (
            <div className="rounded-lg bg-fail/8 px-2.5 py-2">
              <p className="text-[12.5px] text-text-hi">
                Clears saved runs, projects and preferences on this device. This cannot be undone.
              </p>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => void signOut()}
                  className="rounded-md bg-fail px-2.5 py-1 text-[12px] font-medium text-white"
                >
                  Clear this device
                </button>
                <button type="button" onClick={() => setConfirming(false)} className="btn-ghost-sm">
                  Keep
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              className="menu-row hover:text-fail"
              onClick={() => setConfirming(true)}
            >
              <SignOutIcon size={16} className="text-text-lo" />
              Sign out &amp; clear this device
            </button>
          )}
        </div>
      </PopoverContent>
    </Popover>
  )
}
