/**
 * The right panel's tab bar: Chat · Results · Citations · History.
 *
 * Presentational state, so it lives in the panel's own `useState` rather than
 * in a store — nothing outside the panel needs to know which tab is showing,
 * and putting it in Zustand would only invite something to depend on it.
 *
 * A real tablist (WCAG 4.1.2): the active tab is `aria-selected`, arrow keys
 * move between tabs, and each panel is labelled by its tab. The selected tab
 * carries a terracotta underline and the high text colour; the rest sit in
 * the low colour and lift on hover.
 */
import type { KeyboardEvent } from 'react'

import { LABELS, TABS, panelId, tabId, type Tab } from '@/components/thread/tabs'


export function SidebarTabs({
  value,
  onChange,
  badges,
}: {
  value: Tab
  onChange: (tab: Tab) => void
  /** Optional counts shown after a label — citations found, runs remembered. */
  badges?: Partial<Record<Tab, number>>
}) {
  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const index = TABS.indexOf(value)
    let next: Tab | undefined
    if (event.key === 'ArrowRight') next = TABS[(index + 1) % TABS.length]
    else if (event.key === 'ArrowLeft') next = TABS[(index - 1 + TABS.length) % TABS.length]
    else if (event.key === 'Home') next = TABS[0]
    else if (event.key === 'End') next = TABS[TABS.length - 1]
    if (!next) return
    event.preventDefault()
    onChange(next)
    document.getElementById(tabId(next))?.focus()
  }

  return (
    <div
      role="tablist"
      aria-label="Thread"
      onKeyDown={onKeyDown}
      className="flex min-w-0 items-end gap-1 overflow-x-auto"
    >
      {TABS.map((tab) => {
        const selected = tab === value
        const badge = badges?.[tab]
        return (
          <button
            key={tab}
            id={tabId(tab)}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-controls={panelId(tab)}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(tab)}
            className={`-mb-px flex min-h-10 shrink-0 items-center gap-1.5 border-b-2 px-2.5 text-[13px] transition-colors ${
              selected
                ? 'border-accent-warm font-medium text-text-hi'
                : 'border-transparent text-text-lo hover:text-text-hi'
            }`}
          >
            {LABELS[tab]}
            {badge !== undefined && badge > 0 && (
              <span className="tabular rounded-full bg-line px-1.5 font-mono text-[10px] text-text-lo">
                {badge}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}
