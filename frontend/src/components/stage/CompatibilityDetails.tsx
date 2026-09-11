/**
 * The ten-check compatibility battery (API_CONTRACT §2.6, §3.2).
 *
 * Collapsed by default: the summary shows `overall` and the count of non-PASS
 * rows, and the full table is one click away. §3.2 states `detail` is one
 * sentence safe to display verbatim, so it is — no rephrasing, no truncation.
 * `SKIP` rows are included rather than filtered: a check that did not apply is
 * information, and hiding it would make the battery look shorter than it is.
 */
import { Disclosure } from '@/components/ui/Disclosure'
import { StatusDot } from '@/components/ui/StatusDot'
import type { CompatibilityReport } from '@/api/types'

/*
 * The `-text` variants, not the base hues: this is a 14px headline on the ground,
 * where `text-warn` computed to 1.99:1 — the one word that says the pre-flight
 * found something was the least readable word on the screen. `text-fail` needs
 * no variant; #b3261e already clears 4.5:1 here.
 */
const OVERALL_TEXT: Record<string, string> = {
  PASS: 'text-ok-text',
  PASS_WITH_WARNINGS: 'text-warn-text',
  FAIL: 'text-fail',
}

function formatValue(value: number | string | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return typeof value === 'number' ? String(Math.round(value * 1000) / 1000) : value
}

export function CompatibilityDetails({ report }: { report: CompatibilityReport }) {
  const checks = report.checks ?? []
  const flagged = checks.filter((check) => check.status !== 'PASS' && check.status !== 'SKIP')
  const actions = report.actions_taken ?? []

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
        <span className={`font-semibold ${OVERALL_TEXT[report.overall] ?? ''}`}>
          {report.overall.replace(/_/g, ' ')}
        </span>
        <span className="text-text-lo">
          {report.pair_type} · pair type from {report.pair_type_source}
        </span>
        <span className="tabular text-text-lo">
          {checks.length} checks
          {flagged.length > 0 && `, ${flagged.length} flagged`}
        </span>
      </div>

      {actions.length > 0 && (
        <ul className="space-y-1 text-xs text-text-lo">
          {actions.map((action) => (
            <li key={action} className="flex gap-2">
              <span className="text-accent-warm-text">→</span>
              <span className="font-mono">{action}</span>
            </li>
          ))}
        </ul>
      )}

      <Disclosure
        summary="Compatibility details"
        badge={
          flagged.length > 0 ? (
            <span className="tabular ml-1 rounded-full bg-warn/20 px-2 py-0.5 text-[11px] font-medium">
              {flagged.length}
            </span>
          ) : undefined
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-[11px] tracking-[0.08em] text-text-lo uppercase">
                <th className="py-1.5 pr-3 font-medium">Check</th>
                <th className="py-1.5 pr-3 font-medium">Value</th>
                <th className="py-1.5 pr-3 font-medium">Threshold</th>
                <th className="py-1.5 font-medium">Detail</th>
              </tr>
            </thead>
            <tbody>
              {checks.map((check) => (
                <tr key={check.name} className="border-t border-line-soft align-top">
                  <td className="py-2 pr-3">
                    <span className="flex items-center gap-2">
                      <StatusDot status={check.status} />
                      <span className="font-mono text-[13px]">{check.name}</span>
                    </span>
                  </td>
                  <td className="tabular py-2 pr-3 font-mono text-[13px]">
                    {formatValue(check.value)}
                  </td>
                  <td className="py-2 pr-3 font-mono text-[13px] text-text-lo">
                    {check.threshold ?? '—'}
                  </td>
                  <td className="py-2 text-[13px] text-text-lo">{check.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Disclosure>
    </section>
  )
}
