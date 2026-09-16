/**
 * Settings. Three panes — Personalization, Data controls, Appearance — in
 * one dialog, reached from the account popover and by `⌘,`.
 *
 * Every control here is wired to something real, and says what it does in
 * the product's own terms:
 *
 *   Personalization   name and role (the local identity), custom
 *                     instructions (sent ahead of every question, inside
 *                     `query`), and the run seed (`options.seed`).
 *   Data controls     the browser's own storage estimate, an export of
 *                     everything this device holds, and the three clears —
 *                     session history, the saved library, the whole device —
 *                     each confirmed inline, never with a browser dialog.
 *   Appearance        theme, the saved-library layout, the WCAG 2.1.4
 *                     single-key switch, and where the system's reduced-
 *                     motion preference currently stands (read, not set:
 *                     the product follows the OS and does not pretend to
 *                     own that switch).
 *
 * Lazy-loaded by `App` and mounted only while open, so the console entry
 * pays nothing for it. Radix owns focus, `Esc`, scroll-locking and
 * `aria-modal`; the pane switcher is a real tablist with arrow-key roving.
 */
import { useId, useRef, useState } from 'react'

import { ThemeSegmented } from '@/components/shell/ThemeToggle'
import { StorageQuota } from '@/components/shell/StorageQuota'
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Graticule } from '@/components/ui/Graticule'
import {
  CloseIcon,
  DownloadIcon,
  GridIcon,
  KeyboardIcon,
  RowsIcon,
  SettingsIcon,
  TrashIcon,
  UsersIcon,
} from '@/components/ui/icons'
import { Segmented } from '@/components/ui/Segmented'
import { countOf } from '@/format'
import { useReducedMotion } from '@/shell/useReducedMotion'
import { initialOf, useAccountStore } from '@/state/account'
import { useLibraryStore, type LibraryView } from '@/state/library'
import { toast } from '@/state/notifications'
import { INSTRUCTIONS_MAX, queryRoom, useSettingsStore, type SettingsTab } from '@/state/settings'
import { useShortcutStore } from '@/state/shortcuts'
import { useThemeStore } from '@/state/theme'
import { useUiStore } from '@/state/ui'

const TABS: { id: SettingsTab; label: string; Icon: typeof SettingsIcon }[] = [
  { id: 'personalization', label: 'Personalization', Icon: UsersIcon },
  { id: 'data', label: 'Data controls', Icon: DownloadIcon },
  { id: 'appearance', label: 'Appearance', Icon: SettingsIcon },
]

function Field({
  label,
  hint,
  htmlFor,
  children,
}: {
  label: string
  hint?: string
  htmlFor?: string
  children: React.ReactNode
}) {
  return (
    <div className="grid gap-1.5">
      <label htmlFor={htmlFor} className="text-[13px] font-medium text-text-hi">
        {label}
      </label>
      {children}
      {hint ? <p className="t-meta max-w-[60ch]">{hint}</p> : null}
    </div>
  )
}

/** A row with a control on the right: label and explanation on the left. */
function Row({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-6 py-3.5">
      <div className="min-w-0">
        <p className="text-[13px] font-medium text-text-hi">{title}</p>
        <p className="t-meta mt-0.5 max-w-[52ch]">{hint}</p>
      </div>
      <div className="shrink-0 pt-0.5">{children}</div>
    </div>
  )
}

/**
 * A destructive action that confirms in place. The first click turns the
 * button into a sentence and a pair of choices; nothing is lost until the
 * second click, and `Keep` puts the button back.
 */
function Danger({
  label,
  consequence,
  confirm,
  onConfirm,
  disabled,
}: {
  label: string
  consequence: string
  confirm: string
  onConfirm: () => void | Promise<void>
  disabled?: boolean
}) {
  const [armed, setArmed] = useState(false)
  if (!armed) {
    return (
      <button
        type="button"
        onClick={() => setArmed(true)}
        disabled={disabled}
        className="btn-ghost-sm inline-flex items-center gap-1.5 !border-line !text-text-lo hover:!border-fail hover:!bg-transparent hover:!text-fail"
      >
        <TrashIcon size={13} />
        {label}
      </button>
    )
  }
  return (
    <div role="group" aria-label={label} className="rounded-lg bg-fail/8 px-3 py-2.5">
      <p className="text-[12.5px] text-text-hi">{consequence}</p>
      <div className="mt-2 flex gap-2">
        <button
          type="button"
          onClick={() => {
            setArmed(false)
            void onConfirm()
          }}
          className="rounded-md bg-fail px-2.5 py-1 text-[12px] font-medium text-white"
        >
          {confirm}
        </button>
        <button type="button" onClick={() => setArmed(false)} className="btn-ghost-sm">
          Keep
        </button>
      </div>
    </div>
  )
}

/** Everything this device holds, as one JSON document the user can keep. */
function exportDevice(): void {
  const account = useAccountStore.getState()
  const settings = useSettingsStore.getState()
  const library = useLibraryStore.getState()
  const ui = useUiStore.getState()
  const document_ = {
    product: 'SatQuery AI',
    schema: 1,
    exportedAt: new Date().toISOString(),
    account: { name: account.name, role: account.role },
    preferences: {
      theme: useThemeStore.getState().pref,
      singleKeyShortcuts: useShortcutStore.getState().enabled,
      customInstructions: settings.customInstructions,
      seed: settings.seed,
      libraryView: library.view,
    },
    session: { runs: ui.recentRuns },
    // Thumbnails are blobs and stay behind; the server keeps the full trace
    // for every `traceId` here.
    library: {
      runs: Object.values(library.runs).map(({ thumb: _thumb, ...run }) => run),
      projects: Object.values(library.projects),
    },
  }
  const blob = new Blob([JSON.stringify(document_, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `satquery-${new Date().toISOString().slice(0, 10)}.json`
  anchor.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000)
}

function Personalization() {
  const name = useAccountStore((state) => state.name)
  const role = useAccountStore((state) => state.role)
  const setName = useAccountStore((state) => state.setName)
  const setRole = useAccountStore((state) => state.setRole)
  const instructions = useSettingsStore((state) => state.customInstructions)
  const setInstructions = useSettingsStore((state) => state.setCustomInstructions)
  const seed = useSettingsStore((state) => state.seed)
  const setSeed = useSettingsStore((state) => state.setSeed)
  const ids = { name: useId(), role: useId(), instructions: useId(), seed: useId() }
  const room = queryRoom(instructions)

  return (
    <div className="space-y-7">
      <section aria-labelledby="st-identity">
        <h3 id="st-identity" className="t-eyebrow mb-3">
          This device
        </h3>
        <div className="flex items-start gap-4">
          <span
            aria-hidden
            className="grid size-11 shrink-0 place-items-center rounded-full bg-surface-sand text-[15px] font-semibold text-on-evidence"
          >
            {initialOf(name)}
          </span>
          <div className="grid min-w-0 flex-1 gap-3 sm:grid-cols-2">
            <Field label="Name" htmlFor={ids.name}>
              <input
                id={ids.name}
                defaultValue={name}
                onBlur={(event) => setName(event.target.value)}
                autoComplete="name"
                className="field"
              />
            </Field>
            <Field label="Role" htmlFor={ids.role}>
              <input
                id={ids.role}
                defaultValue={role}
                onBlur={(event) => setRole(event.target.value)}
                className="field"
              />
            </Field>
          </div>
        </div>
        <p className="t-meta mt-2.5 max-w-[60ch]">
          There is no account. The name and role live in this browser and appear on exports and
          reports from this device.
        </p>
      </section>

      <section aria-labelledby="st-instructions">
        <h3 id="st-instructions" className="t-eyebrow mb-3">
          Custom instructions
        </h3>
        <Field
          label="Sent ahead of every question"
          htmlFor={ids.instructions}
          hint={`Goes into the query the planner reads, before your question. The server takes 1000 characters for the two together, so the question box will offer ${room} once these are counted.`}
        >
          <textarea
            id={ids.instructions}
            value={instructions}
            onChange={(event) => setInstructions(event.target.value)}
            maxLength={INSTRUCTIONS_MAX}
            rows={4}
            placeholder="e.g. Report areas in hectares. Prefer SAR evidence under cloud. Keep answers to two sentences."
            className="field min-h-24 resize-y leading-relaxed"
          />
        </Field>
        <p className="tabular t-coord mt-1.5 text-right text-text-lo" aria-live="polite">
          {instructions.length} / {INSTRUCTIONS_MAX}
        </p>
      </section>

      <section aria-labelledby="st-seed">
        <h3 id="st-seed" className="t-eyebrow mb-3">
          Reruns
        </h3>
        <Field
          label="Seed"
          htmlFor={ids.seed}
          hint="0 lets the server choose. Any other value is sent as options.seed, and a rerun with the same imagery and question produces the same trace, byte for byte."
        >
          <input
            id={ids.seed}
            type="number"
            inputMode="numeric"
            min={0}
            step={1}
            value={seed}
            onChange={(event) => setSeed(Number.parseInt(event.target.value || '0', 10))}
            className="field tabular w-40 font-mono"
          />
        </Field>
      </section>
    </div>
  )
}

function DataControls() {
  const recentRuns = useUiStore((state) => state.recentRuns)
  const runCount = useLibraryStore((state) => Object.keys(state.runs).length)
  const projectCount = useLibraryStore((state) => Object.keys(state.projects).length)
  const clearLibrary = useLibraryStore((state) => state.clearAll)
  const resetAccount = useAccountStore((state) => state.reset)
  const resetSettings = useSettingsStore((state) => state.reset)
  const closeSettings = useSettingsStore((state) => state.closeSettings)
  const onlineFeatures = useSettingsStore((state) => state.onlineFeatures)
  const setOnlineFeatures = useSettingsStore((state) => state.setOnlineFeatures)

  async function clearDevice() {
    await clearLibrary()
    useUiStore.setState({ recentRuns: [], selectedTraceId: null })
    resetAccount()
    resetSettings()
    try {
      localStorage.clear()
    } catch {
      // Nothing to clear, then.
    }
    closeSettings()
    toast('This device has been cleared.', 'ok')
  }

  return (
    <div className="space-y-7">
      <section aria-labelledby="st-storage">
        <h3 id="st-storage" className="t-eyebrow mb-3">
          Storage on this device
        </h3>
        <StorageQuota />
        <p className="t-meta mt-2.5 max-w-[60ch]">
          Saved runs, projects and preferences live in this browser. Nothing is sent anywhere but the
          API on this machine.
        </p>
      </section>

      <section aria-labelledby="st-online" className="divide-y divide-line-soft">
        <h3 id="st-online" className="t-eyebrow pb-1">
          Online features
        </h3>
        <Row
          title="Imagery search on the Maps page"
          hint="Place search (OpenStreetMap Nominatim) and Sentinel scene search (Microsoft Planetary Computer). Off by default for offline judging; nothing is requested until this is on."
        >
          <button
            type="button"
            role="switch"
            aria-checked={onlineFeatures}
            aria-label="Imagery search on the Maps page"
            onClick={() => setOnlineFeatures(!onlineFeatures)}
            className="switch"
          />
        </Row>
      </section>

      <section aria-labelledby="st-export" className="divide-y divide-line-soft">
        <h3 id="st-export" className="t-eyebrow pb-1">
          Your data
        </h3>
        <Row
          title="Export"
          hint="One JSON file: saved runs and projects, this session's questions, and your preferences. Thumbnails stay behind; the server keeps the full trace for every run id."
        >
          <button type="button" onClick={exportDevice} className="btn-ghost-sm inline-flex items-center gap-1.5">
            <DownloadIcon size={13} />
            Export JSON
          </button>
        </Row>
        <Row
          title="Session history"
          hint={
            recentRuns.length === 0
              ? 'No questions asked this session.'
              : `${countOf(recentRuns.length, { one: 'question', other: 'questions' })} this session. The traces stay on the server.`
          }
        >
          <Danger
            label="Clear"
            consequence="Forgets the questions asked this session. Saved runs are untouched."
            confirm="Clear history"
            disabled={recentRuns.length === 0}
            onConfirm={() => {
              useUiStore.setState({ recentRuns: [], selectedTraceId: null })
              toast('Session history cleared.', 'ok')
            }}
          />
        </Row>
        <Row
          title="Saved runs & projects"
          hint={`${countOf(runCount, { one: 'run', other: 'runs' })} and ${countOf(projectCount, { one: 'project', other: 'projects' })} on this device.`}
        >
          <Danger
            label="Clear"
            consequence="Deletes every saved run and project on this device. This cannot be undone."
            confirm="Delete library"
            disabled={runCount === 0 && projectCount === 0}
            onConfirm={async () => {
              await clearLibrary()
              toast('Saved runs and projects cleared.', 'ok')
            }}
          />
        </Row>
        <Row
          title="Everything"
          hint="Library, history, name, instructions, theme — back to a fresh install. What signing out means on a device-local product."
        >
          <Danger
            label="Reset this device"
            consequence="Clears saved runs, projects, session history and every preference on this device. This cannot be undone."
            confirm="Reset device"
            onConfirm={clearDevice}
          />
        </Row>
      </section>
    </div>
  )
}

function Appearance() {
  const view = useLibraryStore((state) => state.view)
  const setView = useLibraryStore((state) => state.setView)
  const enabled = useShortcutStore((state) => state.enabled)
  const setEnabled = useShortcutStore((state) => state.setEnabled)
  const openGuide = useShortcutStore((state) => state.openGuide)
  const closeSettings = useSettingsStore((state) => state.closeSettings)
  const reduced = useReducedMotion()

  return (
    <div className="space-y-7">
      <section aria-labelledby="st-look" className="divide-y divide-line-soft">
        <h3 id="st-look" className="t-eyebrow pb-1">
          Look
        </h3>
        <Row title="Theme" hint="Dark keeps imagery the brightest thing on screen. System follows the OS and changes with it.">
          <ThemeSegmented />
        </Row>
        <Row title="Saved library" hint="How runs are laid out on the Saved page.">
          <Segmented<LibraryView>
            size="sm"
            label="Saved library layout"
            value={view}
            onChange={setView}
            options={[
              { value: 'cards', label: <GridIcon size={13} />, title: 'Cards' },
              { value: 'rows', label: <RowsIcon size={13} />, title: 'Rows' },
            ]}
          />
        </Row>
      </section>

      <section aria-labelledby="st-input" className="divide-y divide-line-soft">
        <h3 id="st-input" className="t-eyebrow pb-1">
          Keyboard & motion
        </h3>
        <Row
          title="Single-key shortcuts"
          hint="Turn off if you use speech input or a tool that types for you. Chords (⌘…) always work."
        >
          <button
            type="button"
            role="switch"
            aria-checked={enabled}
            aria-label="Single-key shortcuts"
            onClick={() => setEnabled(!enabled)}
            className="switch"
          />
        </Row>
        <Row title="Every shortcut" hint="The full list, grouped by where it works.">
          <button
            type="button"
            onClick={() => {
              closeSettings()
              openGuide()
            }}
            className="btn-ghost-sm inline-flex items-center gap-1.5"
          >
            <KeyboardIcon size={13} />
            Open guide
            <kbd className="kbd ml-1">⌘/</kbd>
          </button>
        </Row>
        <Row
          title="Reduced motion"
          hint="Follows your operating system. When it is on, the reveals and the running pulse hold still and keep their meaning."
        >
          <span className={`chip ${reduced ? 'bg-ok/15 text-ok-text' : 'bg-line text-text-lo'}`}>
            {reduced ? 'on' : 'off'}
          </span>
        </Row>
      </section>
    </div>
  )
}

const PANES: Record<SettingsTab, () => React.JSX.Element> = {
  personalization: Personalization,
  data: DataControls,
  appearance: Appearance,
}

export default function SettingsDialog() {
  const open = useSettingsStore((state) => state.open)
  const close = useSettingsStore((state) => state.closeSettings)
  const tab = useSettingsStore((state) => state.tab)
  const setTab = useSettingsStore((state) => state.setTab)
  const tabsRef = useRef<HTMLDivElement>(null)
  const Pane = PANES[tab]
  const baseId = useId()

  function onTabKey(event: React.KeyboardEvent) {
    const keys = ['ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft', 'Home', 'End']
    if (!keys.includes(event.key)) return
    event.preventDefault()
    const index = TABS.findIndex((t) => t.id === tab)
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? TABS.length - 1
          : (index + (event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length
    setTab(TABS[next]!.id)
    tabsRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus()
  }

  return (
    <Dialog open={open} onOpenChange={(next) => !next && close()}>
      <DialogContent
        className="!inset-auto !top-[8vh] !left-1/2 w-[min(860px,calc(100vw-1.5rem))] !-translate-x-1/2 !border-0 !bg-transparent !shadow-none"
        aria-describedby={`${baseId}-desc`}
        onOpenAutoFocus={(event) => {
          // Land on the current pane's tab, not on the close button: the
          // first thing a keyboard user wants is to move between panes.
          event.preventDefault()
          tabsRef.current?.querySelector<HTMLButtonElement>('[aria-selected="true"]')?.focus()
        }}
      >
        <div className="glass glass-glow flex max-h-[84vh] flex-col overflow-hidden">
          <header className="relative flex items-center gap-3 overflow-hidden border-b border-line-soft px-4 py-3">
            <Graticule fade={false} module={24} className="opacity-50" />
            <SettingsIcon size={16} className="relative text-accent-warm-text" />
            <DialogTitle className="t-panel relative">Settings</DialogTitle>
            <DialogDescription id={`${baseId}-desc`} className="sr-only">
              Personalization, data controls and appearance for SatQuery AI on this device.
            </DialogDescription>
            <DialogClose asChild>
              <button
                type="button"
                aria-label="Close settings"
                className="relative ml-auto flex items-center gap-1.5 text-text-lo hover:text-text-hi"
              >
                <kbd className="kbd">Esc</kbd>
                <CloseIcon size={14} />
              </button>
            </DialogClose>
          </header>

          <div className="flex min-h-0 flex-1 flex-col sm:flex-row">
            <div
              ref={tabsRef}
              role="tablist"
              aria-label="Settings sections"
              aria-orientation="vertical"
              onKeyDown={onTabKey}
              className="flex shrink-0 gap-1 overflow-x-auto border-b border-line-soft p-2 sm:w-52 sm:flex-col sm:overflow-visible sm:border-r sm:border-b-0 sm:p-2.5"
            >
              {TABS.map(({ id, label, Icon }) => {
                const active = id === tab
                return (
                  <button
                    key={id}
                    type="button"
                    role="tab"
                    id={`${baseId}-tab-${id}`}
                    aria-selected={active}
                    aria-controls={`${baseId}-pane-${id}`}
                    tabIndex={active ? 0 : -1}
                    onClick={() => setTab(id)}
                    className={`flex min-h-9 shrink-0 items-center gap-2.5 rounded-lg px-2.5 text-left text-[13px] whitespace-nowrap transition-colors duration-[120ms] ${
                      active
                        ? 'nav-active font-medium text-text-hi'
                        : 'text-text-lo hover:bg-accent-glow hover:text-text-hi'
                    }`}
                  >
                    {/* Icons from `sm` only: at phone width the three labels
                        need the whole strip to sit side by side unclipped. */}
                    <Icon size={16} className={`hidden sm:block ${active ? 'text-accent-warm-text' : ''}`} />
                    {label}
                  </button>
                )
              })}
            </div>

            <div
              role="tabpanel"
              id={`${baseId}-pane-${tab}`}
              aria-labelledby={`${baseId}-tab-${tab}`}
              tabIndex={0}
              className="min-h-0 min-w-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6"
            >
              <Pane />
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
