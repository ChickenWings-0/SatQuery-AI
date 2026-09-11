/**
 * F5b — the live pipeline pulse.
 *
 * The DAG is not on the main screen, which trades away the thing API_CONTRACT
 * §8.4 calls the most persuasive element for a judge: a tool graph filling in.
 * This row is the repayment — one pill per plan step, appearing the moment the
 * `plan` event lands and colouring as the run proceeds. It keeps the machine
 * visibly working through the wait, and clicking it opens the full graph.
 *
 * Pills carry the tool name, not just the step number. "2" tells a judge
 * nothing; "image_diff_change" tells them a fallback ran.
 */
import { StatusDot } from '@/components/ui/StatusDot'
import type { StepNode } from '@/state/job'

/** A sub-millisecond step is real; printing "0 ms" makes it look broken. */
function timing(node: StepNode): string {
  if (node.durationMs === null) return node.estMs ? `~${node.estMs} ms` : ''
  return node.durationMs < 1 ? '<1 ms' : `${node.durationMs} ms`
}

export function PipelinePulse({
  nodes,
  onOpen,
}: {
  nodes: StepNode[]
  onOpen: (step?: number) => void
}) {
  if (nodes.length === 0) return null

  return (
    <ul className="space-y-1">
      {nodes.map((node) => (
        <li key={node.step}>
          <button
            type="button"
            onClick={() => onOpen(node.step)}
            title={`${node.tool} — ${node.state}`}
            // The row itself no longer pulses. `animate-pulse` here faded the
            // tool name in and out — the one string worth reading while the run
            // happens — so the motion moved to the status dot and the row keeps
            // the static rust border and tint that already said "running".
            className={`flex w-full items-center gap-2.5 rounded-md border px-2.5 py-1.5 text-left transition-colors ${
              node.state === 'RUNNING'
                ? 'border-accent-warm bg-accent-warm/10'
                : 'border-transparent hover:border-line hover:bg-bg-main'
            }`}
          >
            <StatusDot status={node.state} />
            <span className="tabular w-3 shrink-0 font-mono text-[10px] text-text-lo">
              {node.step}
            </span>
            <span className="truncate font-mono text-[11px]">{node.tool}</span>
            <span className="tabular ml-auto shrink-0 font-mono text-[10px] text-text-lo">
              {timing(node)}
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}
