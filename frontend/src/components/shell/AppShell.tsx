/**
 * The shell, in three stages. Written mobile-first: the base classes are the
 * phone layout and each breakpoint adds a column.
 *
 *   base          one column. The sidebar becomes a top bar, the stage takes
 *                 the height that is left, and the thread is a bottom sheet
 *                 sized to its own content.
 *   wide          two columns: sidebar beside stage, thread still a sheet.
 *   desk          the designed three columns, 200 / 1fr / 380, with a top
 *                 bar spanning the centre and right columns.
 *
 * `wide` and `desk` are defined in `theme.css` and are **not** `md:` and `xl:`.
 * They carry a height condition as well as a width one, because a landscape
 * phone is 844px wide and 390px tall — wide enough to satisfy any width query,
 * far too short to hold a vertical nav column with a device footer under it.
 *
 * The top bar is its own grid row rather than a sticky header inside `<main>`:
 * a sticky header scrolls with the column it sits in and would vanish behind
 * the viewer on a short window, and it cannot span the thread column at all.
 * The bar holds the scene crumb and the system status; on a phone both fold
 * into one 44px row above the stage.
 *
 * Two things are deliberately *not* done here.
 *
 * The sidebar does not become a hamburger. Eight sections would fit a drawer,
 * but a drawer hides a nav that fits and costs a tap to reach it. It becomes a
 * horizontally scrollable bar instead, and everything stays visible.
 *
 * The thread does not get a collapse control. It sits in an `auto` grid row
 * capped by `max-height`, so it already takes only the height its content needs
 * — an empty thread is a composer, not an empty 45vh panel — and a control for
 * a problem the layout does not have is a control to maintain for nothing.
 *
 * `min-w-0` and `min-h-0` on the centre cell are load-bearing rather than
 * decorative: a grid child defaults to `min-width: auto`, so a wide image or a
 * long monospace token silently pushes the column past its track and drags the
 * whole layout sideways.
 *
 * Note what is *not* here: there is no DAG canvas. The `@xyflow/react` graph is
 * never mounted on this screen — it lives behind the `View Processing Pipeline`
 * control in F5, lazily loaded. That is the whole progressive-disclosure bet.
 */
import type { ReactNode } from 'react'

import { HealthStrip } from '@/components/shell/HealthStrip'
import { SceneCrumb } from '@/components/shell/SceneCrumb'
import { Sidebar } from '@/components/shell/Sidebar'

export function AppShell({
  stage,
  thread,
  bleed = false,
}: {
  stage: ReactNode
  thread: ReactNode
  /** Full-bleed centre column (Maps): no padding, the page owns its edges. */
  bleed?: boolean
}) {
  return (
    <div
      className="
        grid h-full bg-bg-main
        grid-cols-1 grid-rows-[auto_auto_minmax(0,1fr)_auto]
        wide:grid-cols-[200px_minmax(0,1fr)] wide:grid-rows-[auto_minmax(0,1fr)_auto]
        desk:grid-cols-[200px_minmax(0,1fr)_380px] desk:grid-rows-[auto_minmax(0,1fr)]
      "
    >
      {/* Row 1 on a phone, a full-height column from `wide`. */}
      <div className="min-h-0 wide:row-span-3 desk:row-span-2">
        <Sidebar />
      </div>

      <header
        className="
          flex min-w-0 items-center gap-3 border-b border-line px-4 py-2
          wide:px-7
          desk:col-span-2
        "
      >
        <SceneCrumb />
        <div className="ml-auto">
          <HealthStrip />
        </div>
      </header>

      <main
        id="main"
        className={`min-h-0 min-w-0 ${bleed ? 'relative overflow-hidden' : 'overflow-y-auto px-4 py-5 wide:px-7 wide:py-6'}`}
      >
        {stage}
      </main>

      {/*
       * The bottom sheet. `max-height` caps it; the `auto` grid row means it is
       * otherwise only as tall as it needs to be.
       *
       * The phone cap is `min(55vh, 24rem)` rather than a bare percentage
       * because a landscape phone is ~390px tall, where 55vh would leave the
       * imagery about 175px. The rem term is inert in portrait and rescues
       * landscape; the vh term keeps the sheet from dominating a tall phone.
       *
       * `env(safe-area-inset-bottom)` keeps the composer clear of the iOS home
       * indicator — this is the one surface pinned to the bottom edge, so it is
       * the one surface that can be swallowed by it.
       */}
      <section
        className="
          max-h-[min(55vh,24rem)] min-h-0 min-w-0 overflow-y-auto border-t border-line
          bg-surface-card pb-[env(safe-area-inset-bottom)]
          wide:max-h-[45vh]
          desk:max-h-none desk:border-t-0 desk:border-l desk:pb-0
        "
      >
        {thread}
      </section>
    </div>
  )
}
