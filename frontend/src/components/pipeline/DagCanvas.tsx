/**
 * The execution DAG.
 *
 * This module is the *only* place `@xyflow/react` is imported, and it is reached
 * exclusively through a `React.lazy()` boundary in `PipelineDialog`. That is the
 * hard requirement of the progressive-disclosure design: the graph must never be
 * mounted on the main screen, and its ~100 KB must never enter the entry chunk.
 *
 * What the graph is actually arguing, for a judge who opens it:
 *
 *   - The edges come from `PlanStep.depends_on` — a real dependency graph, not a
 *     drawn picture. Hovering one shows `PlanStep.reason`, the policy key that
 *     selected the step, which is the evidence the planner is declarative.
 *   - A `DEGRADED` node carries hazard stripes and names the tool it stood in
 *     for. §8.6: never hide a degradation.
 */
import '@xyflow/react/dist/style.css'

import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import dagre from 'dagre'
import { useCallback, useEffect, useMemo } from 'react'

import { useReducedMotion } from '@/shell/useReducedMotion'
import type { Execution } from '@/api/types'
import type { StepNode } from '@/state/job'

/**
 * Read a design token for the one consumer that cannot use a class.
 *
 * React Flow styles edges through inline `style` objects, so the stroke colour
 * has to be a string in JS. Reading it off the cascade keeps it a *token* —
 * these were hard-coded hexes, and one of them (`#f59e0b`) silently went stale
 * the moment `--color-warn` was darkened to clear WCAG on a white card.
 */
function token(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

const NODE_W = 224
const NODE_H = 96

/** A sub-millisecond step is real; printing "0 ms" makes it look broken. */
function timing(node: StepNode): string {
  if (node.durationMs === null) return node.estMs ? `~${node.estMs} ms` : ''
  return node.durationMs < 1 ? '<1 ms' : `${node.durationMs} ms`
}

const STATE_STYLE: Record<string, { border: string; dot: string }> = {
  PENDING: { border: 'border-line', dot: 'bg-skip' },
  RUNNING: { border: 'border-accent-warm', dot: 'bg-accent-warm' },
  OK: { border: 'border-ok/70', dot: 'bg-ok' },
  DEGRADED: { border: 'border-warn', dot: 'bg-warn' },
  FAILED: { border: 'border-fail', dot: 'bg-fail' },
  SKIPPED: { border: 'border-skip/60', dot: 'bg-skip' },
}

interface StepNodeData extends Record<string, unknown> {
  node: StepNode
  execution: Execution | undefined
  focused: boolean
}

function StepCard({ data }: NodeProps<Node<StepNodeData>>) {
  const { node, execution, focused } = data
  const style = STATE_STYLE[node.state] ?? STATE_STYLE['PENDING']!
  const degraded = node.state === 'DEGRADED'

  // `est_ms` drives the bar until the real number lands, so a running step has
  // determinate-looking progress rather than an indeterminate spinner.
  const progress = node.state === 'RUNNING' ? 65 : node.state === 'PENDING' ? 0 : 100

  return (
    <div
      className={`relative w-[224px] overflow-hidden rounded-lg border-2 bg-surface-card ${style.border} ${
        focused ? 'ring-4 ring-accent-cool' : ''
      }`}
    >
      <Handle type="target" position={Position.Left} className="!bg-surface-sand" />
      {degraded && (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-12"
          style={{
            // `var(--color-warn)`, not a literal: this was `#f59e0b`, which
            // stopped matching the token the moment it was darkened to clear
            // 3:1 as a mark. A copied hex is a token that silently stops
            // being one.
            backgroundImage:
              'repeating-linear-gradient(45deg,var(--color-warn) 0 6px,transparent 6px 12px)',
          }}
        />
      )}

      <div className="relative px-3 pt-2.5">
        <div className="flex items-center gap-1.5">
          <span
            data-status={node.state}
            className={`size-2 shrink-0 rounded-full ${style.dot}`}
          />
          <span className="tabular font-mono text-[10px] text-text-lo">{node.step}</span>
          <span className="truncate font-mono text-[12px] font-medium" title={node.tool}>
            {execution?.tool ?? node.tool}
          </span>
        </div>

        {degraded && execution?.fallback_of && (
          <p className="mt-1 truncate text-[10px] text-text-lo">
            fallback for <span className="font-mono">{execution.fallback_of}</span>
          </p>
        )}

        <div className="tabular mt-1.5 flex items-center gap-2 font-mono text-[10px] text-text-lo">
          <span>{execution?.device_used ?? '—'}</span>
          <span>{timing(node)}</span>
          {execution?.cache_hit && <span title="Cache hit">⚡</span>}
          {node.confidence !== null && (
            <span className="ml-auto">{node.confidence.toFixed(2)}</span>
          )}
        </div>
      </div>

      <div className="relative mt-2 h-1 bg-line">
        <div
          className={`h-full transition-all duration-500 ${style.dot}`}
          style={{ width: `${progress}%` }}
        />
      </div>
      <Handle type="source" position={Position.Right} className="!bg-surface-sand" />
    </div>
  )
}

const nodeTypes = { step: StepCard }

function layout(nodes: StepNode[]): Map<number, { x: number; y: number }> {
  const graph = new dagre.graphlib.Graph()
  graph.setGraph({ rankdir: 'LR', nodesep: 28, ranksep: 72 })
  graph.setDefaultEdgeLabel(() => ({}))

  for (const node of nodes) {
    graph.setNode(String(node.step), { width: NODE_W, height: NODE_H })
  }
  for (const node of nodes) {
    for (const dependency of node.dependsOn) {
      if (nodes.some((other) => other.step === dependency)) {
        graph.setEdge(String(dependency), String(node.step))
      }
    }
  }
  dagre.layout(graph)

  const positions = new Map<number, { x: number; y: number }>()
  for (const node of nodes) {
    const laid = graph.node(String(node.step))
    positions.set(node.step, {
      x: laid.x - NODE_W / 2,
      y: laid.y - NODE_H / 2,
    })
  }
  return positions
}

export default function DagCanvas({
  nodes,
  executions,
  focusedStep,
  onSelectStep,
}: {
  nodes: StepNode[]
  executions: Execution[]
  focusedStep: number | null
  onSelectStep: (step: number) => void
}) {
  const positions = useMemo(() => layout(nodes), [nodes])
  const reduced = useReducedMotion()
  const edgeStroke = token('--color-surface-sand', '#dfa878')
  const warmStroke = token('--color-accent-warm', '#ba704f')

  const flowNodes: Node<StepNodeData>[] = useMemo(
    () =>
      nodes.map((node) => ({
        id: String(node.step),
        type: 'step',
        position: positions.get(node.step) ?? { x: 0, y: 0 },
        data: {
          node,
          execution: executions.find((execution) => execution.step === node.step),
          focused: focusedStep === node.step,
        },
        draggable: false,
      })),
    [nodes, positions, executions, focusedStep],
  )

  const flowEdges: Edge[] = useMemo(
    () =>
      nodes.flatMap((node) =>
        node.dependsOn
          .filter((dependency) => nodes.some((other) => other.step === dependency))
          .map((dependency) => {
            const running = node.state === 'RUNNING'
            return {
              id: `${dependency}->${node.step}`,
              source: String(dependency),
              target: String(node.step),
              // The dash flow runs only while the downstream step is actually
              // executing, so motion means work rather than decoration.
              //
              // Under `prefers-reduced-motion` the dash would simply stop, which
              // would make a running edge identical to a finished one — the
              // signal deleted rather than reduced. So the moving dash is traded
              // for a heavier rust stroke: same meaning, no movement.
              animated: running && !reduced,
              style:
                running && reduced
                  ? { stroke: warmStroke, strokeWidth: 2.5 }
                  : { stroke: edgeStroke, strokeWidth: 1.5 },
              // No edge label: every edge in a plan carries the *same* policy key,
              // so painting it five times only overlapped the nodes. The key is
              // stated once in the dialog header, and each node's own reason is
              // in the inspector.
            }
          }),
      ),
    [nodes, reduced, edgeStroke, warmStroke],
  )

  const onNodeClick = useCallback(
    (_: unknown, node: Node) => onSelectStep(Number(node.id)),
    [onSelectStep],
  )

  useEffect(() => {
    // Nothing to do — the effect exists so a future "pan to focusedStep" has an
    // obvious home. fitView already frames the whole graph on open.
  }, [focusedStep])

  return (
    <ReactFlow
      nodes={flowNodes}
      edges={flowEdges}
      nodeTypes={nodeTypes}
      onNodeClick={onNodeClick}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      proOptions={{ hideAttribution: false }}
      nodesConnectable={false}
      className="bg-bg-main"
    >
      <Background color="#2a1f1a" gap={22} size={1} />
      <Controls showInteractive={false} />
    </ReactFlow>
  )
}
