/**
 * Device honesty, as a badge in the top bar.
 *
 * One line at a glance — `● System Ready` / `● Degraded` / `● Offline` — and
 * the pre-demo checks behind it on hover and focus. API_CONTRACT §4.9 calls
 * `igpu_masked: false` "a red flag before any demo", so when it is false the
 * badge itself says Degraded rather than hiding the flag in the popover: the
 * 8B VLM is about to share VRAM with the desktop compositor, and that is
 * status, not detail.
 *
 * The popover is CSS-only (`group-hover` / `group-focus-within`), so it costs
 * no state and works for the keyboard: focusing the badge opens it, tabbing
 * away closes it.
 */
import { useQuery } from '@tanstack/react-query'

import { health } from '@/api/client'
import { BellIcon, DotIcon } from '@/components/ui/icons'
import { decimal, integer } from '@/format'

type Tone = 'ok' | 'warn' | 'fail' | 'pending'

const TONE: Record<Tone, { label: string; dot: string }> = {
  ok: { label: 'System Ready', dot: 'text-ok sq-glow' },
  warn: { label: 'Degraded', dot: 'text-warn' },
  fail: { label: 'Offline', dot: 'text-fail' },
  pending: { label: 'Checking…', dot: 'text-skip' },
}

export function HealthStrip() {
  const { data, isPending, isError } = useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => health(signal),
    refetchInterval: 15_000,
    retry: false,
  })

  const tone: Tone = isPending
    ? 'pending'
    : isError || !data
      ? 'fail'
      : data.status === 'ok' && data.device.igpu_masked
        ? 'ok'
        : 'warn'
  const { label, dot } = TONE[tone]

  // VRAM is nullable: on a CPU-only box there is none to report, and §1's
  // nulls rule makes that `null` rather than 0. Render the bar only when both
  // halves are known — a 0% bar would assert something the server did not say.
  const usedMb = data?.device.vram_used_mb
  const totalMb = data?.device.vram_total_mb
  const hasVram =
    typeof usedMb === 'number' &&
    Number.isFinite(usedMb) &&
    typeof totalMb === 'number' &&
    Number.isFinite(totalMb) &&
    totalMb > 0
  // Clamped: a driver that reports used > total would otherwise paint the bar
  // past its own track.
  const vramPct = hasVram ? Math.min(100, Math.max(0, Math.round((usedMb / totalMb) * 100))) : 0

  return (
    <div className="flex items-center gap-1.5">
      <div className="group relative">
        <button
          type="button"
          // A live region: the only notice a screen-reader user gets that the
          // API came back, or went away, between polls.
          role="status"
          aria-label={
            data
              ? `${label}. Device ${data.device.name}${data.device.igpu_masked ? '' : ', integrated GPU not masked'}`
              : label
          }
          className="flex h-9 items-center gap-2 rounded-full border border-line bg-surface-card px-3 text-[12px] font-medium text-text-hi transition-colors hover:border-accent-warm/40"
        >
          <DotIcon className={dot} />
          <span>{label}</span>
        </button>

        {/* The pre-demo checks. Hidden until hovered or focused; on a phone
            the badge alone carries the signal and the sentence above carries
            the rest. */}
        <div
          role="tooltip"
          className="pointer-events-none absolute top-full right-0 z-30 mt-2 hidden w-64 rounded-xl border border-line bg-surface-elevated p-3.5 text-[12px] shadow-2xl group-focus-within:block group-hover:block"
        >
          {isPending ? (
            <p className="text-text-lo">Checking the device…</p>
          ) : isError || !data ? (
            <>
              <p className="font-medium text-fail">API unreachable</p>
              <p className="mt-1 text-text-lo">Retrying every 15 s.</p>
            </>
          ) : (
            <div className="tabular space-y-2.5">
              <p className="truncate font-medium" title={data.device.name}>
                {data.device.name}
              </p>

              {hasVram ? (
                <div>
                  <div
                    role="progressbar"
                    aria-valuenow={vramPct}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label="VRAM in use"
                    className="h-1 overflow-hidden rounded-full bg-line"
                  >
                    <div
                      className="h-full rounded-full bg-accent-warm-strong"
                      style={{ width: `${vramPct}%` }}
                    />
                  </div>
                  <p className="mt-1 text-text-lo">
                    {decimal(usedMb / 1024, 1)}/{decimal(totalMb / 1024, 1)} GB VRAM
                  </p>
                </div>
              ) : (
                <p className="text-text-lo">no VRAM reported ({data.device.backend})</p>
              )}

              <p className="text-text-lo">
                {integer(data.tools_available)}/{integer(data.tools_total)} tools · schema{' '}
                {data.schema_version}
              </p>

              <p className={data.device.igpu_masked ? 'text-text-lo' : 'font-semibold text-warn-text'}>
                igpu_masked {data.device.igpu_masked ? '✓' : '✗ — mask it before demoing'}
              </p>
            </div>
          )}
        </div>
      </div>

      <button
        type="button"
        aria-label="Notifications"
        className="grid size-9 place-items-center rounded-full border border-line bg-surface-card text-text-lo transition-colors hover:border-accent-warm/40 hover:text-text-hi"
      >
        <BellIcon size={16} />
      </button>
    </div>
  )
}
