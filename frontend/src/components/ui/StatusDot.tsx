/**
 * The one status vocabulary. `ToolStatus` and `CheckStatus` share a colour
 * scale by design (roadmap F0 rule 1), so both render through this component
 * and a PASS looks exactly like an OK everywhere in the product.
 */
import type { CheckStatus, NodeState } from '@/api/types'

type Status = CheckStatus | NodeState

const CLASS: Record<string, string> = {
  PASS: 'bg-ok',
  OK: 'bg-ok',
  WARN: 'bg-warn',
  DEGRADED: 'bg-warn',
  FAIL: 'bg-fail',
  FAILED: 'bg-fail',
  SKIP: 'bg-skip',
  SKIPPED: 'bg-skip',
  PENDING: 'bg-skip',
  RUNNING: 'bg-accent-warm',
}

export function StatusDot({ status, className = '' }: { status: Status; className?: string }) {
  return (
    <span
      // Drives the RUNNING pulse from `theme.css`, so "this step is working"
      // looks the same in the pipeline pulse, the compatibility table and the
      // DAG inspector without three components each animating their own way.
      data-status={status}
      // `role="img"` is load-bearing, not decoration: `aria-label` on a bare
      // role-less `<span>` is ignored by most screen readers, which left the
      // compatibility table and the pipeline conveying status by colour alone
      // (WCAG 1.4.1) in a product whose whole pitch is auditability.
      role="img"
      aria-label={`Status: ${status}`}
      title={status}
      className={`inline-block size-2.5 shrink-0 rounded-full ${CLASS[status] ?? 'bg-skip'} ${className}`}
    />
  )
}
