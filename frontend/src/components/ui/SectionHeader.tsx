/**
 * A section page's title row: the page's one `<h1>`, a status line, and an
 * action slot. The sidebar wordmark used to be the `h1` on every screen; the
 * page title is the heading now, which is what a heading is for.
 */
import type { ReactNode } from 'react'

export function SectionHeader({
  title,
  meta,
  crumb,
  action,
  children,
}: {
  title: string
  /** A count or status, in `.t-meta`, right-aligned. */
  meta?: ReactNode
  /** A parent link, rendered before the title (`Projects /`). */
  crumb?: ReactNode
  action?: ReactNode
  /** A second row: filters, tabs. */
  children?: ReactNode
}) {
  return (
    <header className="mb-5 border-b border-line pb-4">
      <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          {crumb ? <div className="t-meta mb-1 flex items-center gap-1.5">{crumb}</div> : null}
          <h1 className="t-page truncate">{title}</h1>
        </div>
        <div className="flex items-center gap-3">
          {meta ? <p className="t-meta tabular">{meta}</p> : null}
          {action}
        </div>
      </div>
      {children ? <div className="mt-4">{children}</div> : null}
    </header>
  )
}
