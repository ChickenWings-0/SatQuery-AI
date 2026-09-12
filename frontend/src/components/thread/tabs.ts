/**
 * The thread panel's tab vocabulary, in its own module so `SidebarTabs.tsx`
 * exports only components (React Fast Refresh needs that to keep state on
 * edit) while `ThreadPanel` can still import the ids it labels panels with.
 */
export const TABS = ['chat', 'results', 'citations', 'history'] as const
export type Tab = (typeof TABS)[number]

export const LABELS: Record<Tab, string> = {
  chat: 'Chat',
  results: 'Results',
  citations: 'Citations',
  history: 'History',
}

export function tabId(tab: Tab): string {
  return `thread-tab-${tab}`
}

export function panelId(tab: Tab): string {
  return `thread-panel-${tab}`
}
