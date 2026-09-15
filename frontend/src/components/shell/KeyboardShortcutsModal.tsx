/**
 * The shortcuts guide. Command-palette in shape — a filter box on top, grouped
 * rows, a kbd column — but a guide, not a palette: it does not execute. Its
 * footer carries the WCAG 2.1.4 master switch, and it is reachable by `⌘/`
 * so it can be opened with the single keys off.
 */
import { useMemo, useRef, useState } from 'react'

import { CloseIcon, SearchIcon } from '@/components/ui/icons'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog'
import { BINDINGS, SCOPE_LABELS, type ShortcutScope } from '@/shell/shortcuts'
import { useShortcutStore } from '@/state/shortcuts'

const ORDER: ShortcutScope[] = ['global', 'thread', 'stage', 'navigation']

export function KeyboardShortcutsModal() {
  const open = useShortcutStore((state) => state.guideOpen)
  const closeGuide = useShortcutStore((state) => state.closeGuide)
  const enabled = useShortcutStore((state) => state.enabled)
  const setEnabled = useShortcutStore((state) => state.setEnabled)
  const [filter, setFilter] = useState('')
  const input = useRef<HTMLInputElement>(null)

  const groups = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    return ORDER.map((scope) => ({
      scope,
      rows: BINDINGS.filter(
        (b) =>
          b.scope === scope &&
          (!needle ||
            b.label.toLowerCase().includes(needle) ||
            b.keys.join(' ').toLowerCase().includes(needle)),
      ),
    })).filter((group) => group.rows.length > 0)
  }, [filter])

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (next) return
        closeGuide()
        setFilter('')
      }}
    >
      <DialogContent
        className="!inset-auto !top-[12vh] !left-1/2 w-[min(560px,calc(100vw-1.5rem))] !-translate-x-1/2 !bg-transparent !border-0 !shadow-none"
        onOpenAutoFocus={(event) => {
          event.preventDefault()
          input.current?.focus()
        }}
      >
        <div className="glass glass-glow flex max-h-[76vh] flex-col overflow-hidden">
          <DialogTitle className="sr-only">Keyboard shortcuts</DialogTitle>
          <DialogDescription className="sr-only">
            Every keyboard shortcut in SatQuery AI, grouped by where it works.
          </DialogDescription>

          <div className="flex items-center gap-2.5 border-b border-line-soft px-3.5 py-2.5">
            <SearchIcon size={16} className="shrink-0 text-text-lo" />
            <input
              ref={input}
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Filter shortcuts…"
              aria-label="Filter shortcuts"
              className="min-w-0 flex-1 bg-transparent text-[14px] text-text-hi placeholder:text-text-lo focus:outline-none"
            />
            <DialogClose asChild>
              <button type="button" aria-label="Close" className="flex items-center gap-1.5 text-text-lo hover:text-text-hi">
                <kbd className="kbd">Esc</kbd>
                <CloseIcon size={14} />
              </button>
            </DialogClose>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
            {groups.length === 0 ? (
              <p className="t-meta px-2 py-6 text-center">No shortcut matches “{filter}”.</p>
            ) : (
              groups.map((group) => (
                <section key={group.scope} aria-labelledby={`sc-${group.scope}`} className="mb-2">
                  <h3 id={`sc-${group.scope}`} className="t-eyebrow px-2 pt-2 pb-1.5">
                    {SCOPE_LABELS[group.scope]}
                  </h3>
                  <ul>
                    {group.rows.map((row) => {
                      const dim = !row.chord && !enabled
                      return (
                        <li
                          key={row.id}
                          className={`flex items-center gap-3 rounded-lg px-2 py-1.5 text-[13px] ${dim ? 'opacity-50' : ''}`}
                        >
                          <span className="min-w-0 flex-1 truncate text-text-hi">{row.label}</span>
                          {dim ? <span className="chip bg-line text-text-lo">off</span> : null}
                          <span className="flex items-center gap-1">
                            {row.keys.map((key, i) => (
                              <span key={`${row.id}-${i}`} className="flex items-center gap-1">
                                {i > 0 && !row.chord ? (
                                  <span className="t-coord text-text-lo">then</span>
                                ) : null}
                                <kbd className="kbd min-w-6 !text-[11px]">{key}</kbd>
                              </span>
                            ))}
                          </span>
                        </li>
                      )
                    })}
                  </ul>
                </section>
              ))
            )}
          </div>

          <footer className="flex items-center gap-3 border-t border-line-soft px-3.5 py-3">
            <div className="min-w-0 flex-1">
              <p className="text-[13px] font-medium text-text-hi">Single-key shortcuts</p>
              <p className="t-meta">
                Turn off if you use speech input or a tool that types for you. Chords (⌘…) always
                work.
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={enabled}
              aria-label="Single-key shortcuts"
              onClick={() => setEnabled(!enabled)}
              className="switch"
            />
          </footer>
        </div>
      </DialogContent>
    </Dialog>
  )
}
