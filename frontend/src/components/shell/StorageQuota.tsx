/**
 * How much of this browser's storage the app is using — the one quota the
 * product can honestly report. `navigator.storage.estimate()` is real data;
 * where it is unsupported the row says so rather than drawing an empty bar.
 */
import { useEffect, useState } from 'react'

import { decimal } from '@/format'

interface Estimate {
  usage: number
  quota: number
}

function formatBytes(value: number): string {
  if (value >= 1024 ** 3) return `${decimal(value / 1024 ** 3, 1)} GB`
  if (value >= 1024 ** 2) return `${decimal(value / 1024 ** 2, 0)} MB`
  return `${decimal(value / 1024, 0)} KB`
}

export function StorageQuota() {
  const [estimate, setEstimate] = useState<Estimate | null | 'unsupported'>(() =>
    typeof navigator === 'undefined' || !navigator.storage?.estimate ? 'unsupported' : null,
  )

  useEffect(() => {
    let live = true
    if (estimate === 'unsupported') return
    void navigator.storage
      .estimate()
      .then((result) => {
        if (!live) return
        const usage = result.usage ?? 0
        const quota = result.quota ?? 0
        setEstimate(quota > 0 ? { usage, quota } : 'unsupported')
      })
      .catch(() => live && setEstimate('unsupported'))
    return () => {
      live = false
    }
    // Only the initial estimate is fetched; the value is a snapshot, not a feed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (estimate === 'unsupported') {
    return <p className="t-meta">Storage estimate unavailable in this browser.</p>
  }
  if (estimate === null) {
    return <div aria-hidden className="skeleton h-1 w-full" />
  }
  const pct = Math.min(100, Math.round((estimate.usage / estimate.quota) * 100))
  const tone = pct >= 95 ? 'bg-fail' : pct >= 80 ? 'bg-warn' : 'bg-accent-warm-strong'
  return (
    <div>
      <div
        role="progressbar"
        aria-label="Storage used on this device"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-1 overflow-hidden rounded-full bg-line"
      >
        <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.max(1, pct)}%` }} />
      </div>
      <p className="t-coord mt-1.5 text-text-lo">
        {formatBytes(estimate.usage)} of {formatBytes(estimate.quota)}
      </p>
    </div>
  )
}
