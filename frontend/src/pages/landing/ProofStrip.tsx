/**
 * Real numbers or an honest line. Same query keys as the console, so the
 * cache is shared and nothing is fetched twice.
 */
import { useQuery } from '@tanstack/react-query'

import { health, registry } from '@/api/client'
import { DotIcon } from '@/components/ui/icons'
import { integer } from '@/format'

export function ProofStrip() {
  const h = useQuery({ queryKey: ['health'], queryFn: ({ signal }) => health(signal), retry: false, refetchInterval: 15_000 })
  const r = useQuery({ queryKey: ['registry'], queryFn: ({ signal }) => registry(signal), retry: false })

  const offline = h.isError
  const ready = h.data?.status === 'ok' && h.data.device.igpu_masked

  return (
    <div data-hero="proof" role="status" className="t-coord flex flex-wrap items-center gap-x-3 gap-y-1.5 text-text-lo">
      {h.isPending ? (
        <span className="flex items-center gap-2">
          <DotIcon className="text-skip" /> Checking the device…
        </span>
      ) : offline ? (
        <span className="flex items-center gap-2">
          <DotIcon className="text-fail" /> API offline — the console still opens with recorded fixtures
        </span>
      ) : (
        <>
          <span className="flex items-center gap-2 text-text-hi">
            <DotIcon className={ready ? 'text-ok sq-glow' : 'text-warn'} />
            {ready ? 'System ready' : 'Degraded'}
          </span>
          <span aria-hidden>·</span>
          <span>
            {integer(h.data!.tools_available)}/{integer(h.data!.tools_total)} tools
          </span>
          <span aria-hidden>·</span>
          <span>schema {h.data!.schema_version}</span>
          {r.data ? (
            <>
              <span aria-hidden>·</span>
              <span>registry {r.data.registry_version}</span>
            </>
          ) : null}
          <span aria-hidden>·</span>
          <span className="truncate">{h.data!.device.name}</span>
        </>
      )}
      <span aria-hidden>·</span>
      <span>Qwen3-VL-8B + QLoRA</span>
      <span aria-hidden>·</span>
      <span>runs offline</span>
    </div>
  )
}
