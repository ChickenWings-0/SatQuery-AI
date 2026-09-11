/**
 * The capabilities panel, from `GET /v1/registry`.
 *
 * API_CONTRACT §4.8 is explicit: a tool whose weights are absent must be shown
 * **disabled, not hidden** — the registry being real, and honest about what is
 * not loaded on this machine, is part of the story. So unavailable tools render
 * greyed with their `unavailable_reason` spelled out rather than filtered away.
 */
import { useQuery } from '@tanstack/react-query'

import { registry } from '@/api/client'
import type { ToolSpec } from '@/api/types'

const CATEGORY_ORDER = ['geo', 'analysis', 'cv', 'fusion', 'vlm'] as const

function ToolCard({ tool }: { tool: ToolSpec }) {
  return (
    <li
      className={`rounded-xl border px-4 py-3 ${
        tool.available ? 'border-line bg-surface-card' : 'border-line bg-surface-card/40'
      }`}
      aria-disabled={!tool.available}
    >
      <div className="flex items-start justify-between gap-3">
        {/* An unavailable tool is still information — which tool, and what it
            would have done. `opacity-55` took the name to 3.64:1 and the
            description to 2.35:1, so the rows that most need reading were the
            hardest to read. The chip beside them already says "unavailable";
            dimming the text as well was saying it twice, illegibly. */}
        <div className="min-w-0">
          <p className="font-mono text-[13px] font-medium">{tool.name}</p>
          <p className="mt-1 text-[13px] leading-snug text-text-lo">{tool.description}</p>
        </div>
        <span
          className={`chip shrink-0 font-medium ${
            // A chip puts the hue on a wash of itself, which is the ground a
            // naive contrast check misses: `text-ok` on `bg-ok/12` was 4.17:1
            // and `text-skip` on `bg-skip/15` was 2.72:1, both at 11px.
            tool.available ? 'bg-ok/12 text-ok-text' : 'bg-skip/15 text-skip-text'
          }`}
        >
          {tool.available ? 'available' : 'unavailable'}
        </span>
      </div>

      <div className="tabular mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] text-text-lo">
        <span>v{tool.version}</span>
        <span>{tool.device}</span>
        <span>~{tool.est_ms} ms</span>
        {tool.fallback && <span>fallback: {tool.fallback}</span>}
        <span>{(tool.produces ?? []).join(', ')}</span>
      </div>

      {!tool.available && tool.unavailable_reason && (
        <p className="mt-2 border-l-2 border-warn pl-2 text-xs text-text-lo">
          {tool.unavailable_reason}
        </p>
      )}
    </li>
  )
}

export function ToolsPanel() {
  const { data, isPending, isError, error } = useQuery({
    queryKey: ['registry'],
    queryFn: ({ signal }) => registry(signal),
    retry: false,
  })

  if (isPending) return <p className="text-text-lo">Loading the registry…</p>
  if (isError || !data) {
    return <p className="text-fail">Could not load the registry: {String(error)}</p>
  }

  const byCategory = new Map<string, ToolSpec[]>()
  for (const tool of data.tools) {
    const bucket = byCategory.get(tool.category) ?? []
    bucket.push(tool)
    byCategory.set(tool.category, bucket)
  }
  const categories = [...byCategory.keys()].sort(
    (a, b) =>
      (CATEGORY_ORDER.indexOf(a as never) + 1 || 99) -
      (CATEGORY_ORDER.indexOf(b as never) + 1 || 99),
  )

  const availableCount = data.tools.filter((tool) => tool.available).length

  return (
    <div className="mx-auto max-w-4xl">
      <header>
        <h2 className="t-page">Tools</h2>
        <p className="tabular mt-1.5 text-[13px] text-text-lo">
          Registry {data.registry_version} · {availableCount} of {data.tools.length} available on
          this machine · adapter{' '}
          <span className="font-mono">{data.adapter_version ?? 'none'}</span>
        </p>
      </header>

      {categories.map((category) => (
        <section key={category} className="mt-6">
          <h3 className="t-eyebrow">{category}</h3>
          <ul className="mt-2.5 space-y-2">
            {(byCategory.get(category) ?? []).map((tool) => (
              <ToolCard key={tool.name} tool={tool} />
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
