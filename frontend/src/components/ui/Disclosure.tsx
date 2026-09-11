/**
 * The progressive-disclosure primitive.
 *
 * The whole product is built on hiding the machine until it is asked for, so
 * the pattern gets one component rather than an ad-hoc `useState` per panel.
 * Native `<details>`: keyboard and screen-reader behaviour for free.
 */
import type { ReactNode } from 'react'

export function Disclosure({
  summary,
  badge,
  children,
  defaultOpen = false,
}: {
  summary: string
  badge?: ReactNode
  children: ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details open={defaultOpen} className="group card">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2.5 text-sm font-medium select-none">
        <span className="text-accent-warm-text transition-transform group-open:rotate-90">
          ▸
        </span>
        <span>{summary}</span>
        {badge}
      </summary>
      <div className="border-t border-line-soft px-4 py-3">{children}</div>
    </details>
  )
}
