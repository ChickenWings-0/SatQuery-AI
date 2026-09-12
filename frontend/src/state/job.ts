/**
 * The live job store: a reducer over the SSE event union, and the Zustand
 * store that hosts it.
 *
 * Everything the run produces flows through {@link reduce}. It is a pure
 * function of `(state, event)` with no I/O, which is what makes the streaming
 * path testable without a server: drive it with a recorded event sequence and
 * the terminal state must equal the `done` payload.
 *
 * The one piece of real modelling here is `nodes`. The contract's `ToolStatus`
 * describes only finished steps, so the client widens it to {@link NodeState}:
 * every step becomes `PENDING` the instant the `plan` event lands — which is
 * why the DAG can be drawn complete before anything runs — then `RUNNING` on
 * `step_started`, then whichever of the four contract outcomes `step_completed`
 * reports. `PENDING` and `RUNNING` are client-only and never sent back.
 */
import { create } from 'zustand'

import type { JobEvent } from '@/api/events'
import type {
  AnalyzeResponse,
  ApiError,
  ArtifactRef,
  NodeState,
  PlanStep,
  Stage,
  ToolStatus,
} from '@/api/types'

/** One DAG node: its plan-time shape plus everything execution has revealed. */
export interface StepNode {
  step: number
  tool: string
  dependsOn: number[]
  inputRefs: string[]
  /** The policy key that selected this step, e.g. `policy_table:CHANGE_VQA|…`. */
  reason: string
  state: NodeState
  /** From `step_started`; drives the determinate progress bar before the fact. */
  estMs: number | null
  /** From `step_completed`; the real number, which replaces the estimate. */
  durationMs: number | null
  confidence: number | null
  outputRefs: string[]
}

/**
 * `reconnecting` is client-only, like `PENDING`/`RUNNING` on nodes: the run is
 * live on the server as far as we know, and we have lost the stream to it. The
 * first event that arrives on the new stream moves the phase back to
 * `streaming` through the reducer; nothing else needs to know.
 */
export type JobPhase = 'idle' | 'streaming' | 'reconnecting' | 'succeeded' | 'failed'

export interface JobState {
  phase: JobPhase
  jobId: string | null
  stage: Stage | null
  pct: number
  /** Plan order is topological, so this array is already in execution order. */
  nodes: StepNode[]
  artifacts: ArtifactRef[]
  /** Accumulated `answer_delta` text. Superseded by `result.answer.text`. */
  streamedAnswer: string
  result: AnalyzeResponse | null
  error: ApiError | null
}

export const initialJobState: JobState = {
  phase: 'idle',
  jobId: null,
  stage: null,
  pct: 0,
  nodes: [],
  artifacts: [],
  streamedAnswer: '',
  result: null,
  error: null,
}

function nodeFromPlanStep(step: PlanStep): StepNode {
  return {
    step: step.step,
    tool: step.tool,
    dependsOn: step.depends_on ?? [],
    inputRefs: step.input_refs,
    reason: step.reason,
    state: 'PENDING',
    estMs: null,
    durationMs: null,
    confidence: null,
    outputRefs: [],
  }
}

function patchNode(
  nodes: StepNode[],
  step: number,
  patch: Partial<StepNode>,
): StepNode[] {
  return nodes.map((node) => (node.step === step ? { ...node, ...patch } : node))
}

/**
 * Fold one event into the job state.
 *
 * Pure and total: every member of the union is handled, and an event that
 * arrives out of the guaranteed order degrades to a no-op rather than throwing,
 * because a half-drawn DAG is a better failure than a blank screen.
 */
export function reduce(state: JobState, event: JobEvent): JobState {
  switch (event.type) {
    case 'queued':
      return {
        ...initialJobState,
        phase: 'streaming',
        jobId: event.data.job_id,
        stage: 'queued',
      }

    case 'stage':
      return {
        ...state,
        phase: 'streaming',
        stage: event.data.stage,
        // Monotonic: a late stage event must never drag the bar backwards.
        pct: Math.max(state.pct, event.data.pct),
      }

    case 'plan':
      // Emitted once, before execution: this is the whole graph, PENDING.
      return { ...state, phase: 'streaming', nodes: event.data.steps.map(nodeFromPlanStep) }

    case 'step_started':
      return {
        ...state,
        nodes: patchNode(state.nodes, event.data.step, {
          state: 'RUNNING',
          estMs: event.data.est_ms,
        }),
      }

    case 'step_completed': {
      const status: ToolStatus = event.data.status
      return {
        ...state,
        nodes: patchNode(state.nodes, event.data.step, {
          state: status,
          durationMs: event.data.duration_ms,
          confidence: event.data.confidence,
          outputRefs: event.data.output_refs,
        }),
      }
    }

    case 'artifact':
      // Artifacts are content-addressed and immutable (§4.6), so a replayed
      // stream must not double them up.
      return state.artifacts.some((existing) => existing.id === event.data.id)
        ? state
        : { ...state, artifacts: [...state.artifacts, event.data] }

    case 'answer_delta':
      return { ...state, streamedAnswer: state.streamedAnswer + event.data.text }

    case 'done':
      return {
        ...state,
        phase: 'succeeded',
        stage: 'done',
        pct: 100,
        result: event.data,
        // The terminal payload is authoritative: citation offsets are computed
        // once against this text, never against a partially streamed string.
        streamedAnswer: event.data.answer.text,
        // A run can finish having produced artifacts the stream missed if the
        // client subscribed late and the buffer was trimmed; trust the result.
        // `artifacts` is optional on the wire — include_rendered_views=false
        // omits it — so an absent array must not wipe what did stream in.
        artifacts:
          event.data.artifacts && event.data.artifacts.length > 0
            ? event.data.artifacts
            : state.artifacts,
      }

    case 'error':
      return { ...state, phase: 'failed', error: event.data }
  }
}

/** Replay a whole recorded sequence. The reducer's own test harness. */
export function reduceAll(events: JobEvent[], from: JobState = initialJobState): JobState {
  return events.reduce(reduce, from)
}

// --------------------------------------------------------------- derivations

/** Steps whose state is one of the contract's four finished outcomes. */
export function completedCount({ nodes }: Pick<JobState, 'nodes'>): number {
  return nodes.filter((node) => node.state !== 'PENDING' && node.state !== 'RUNNING').length
}

/** Steps that ran but not cleanly — the badge count on the pipeline button. */
export function degradedCount({ nodes }: Pick<JobState, 'nodes'>): number {
  return nodes.filter(
    (node) => node.state === 'DEGRADED' || node.state === 'FAILED' || node.state === 'SKIPPED',
  ).length
}

/**
 * Resolve a citation's `step:{n}/scalars.{path}` source to its DAG node.
 *
 * Multi-scalar claims join paths with `|`, so the step is parsed from the head.
 */
export function nodeForCitationSource(state: JobState, source: string): StepNode | undefined {
  const match = /^step:(\d+)\//.exec(source)
  if (!match?.[1]) return undefined
  const step = Number(match[1])
  return state.nodes.find((node) => node.step === step)
}

// ------------------------------------------------------------------- store

export interface JobStore extends JobState {
  apply: (event: JobEvent) => void
  /**
   * End the run as failed without inventing a server error.
   *
   * A dropped connection, a stalled stream or a server restart never produces
   * an `error` event — the stream just stops. Left alone the phase stays
   * `streaming` for ever, which disables the composer and leaves a spinner
   * running on a job that will never report again. This moves the phase and
   * says nothing more: `error` stays `null`, because §6 envelopes come from the
   * server and the client does not get to forge one. The transport failure is
   * rendered by whoever caught it, in its own words.
   *
   * A run that already finished is left alone: a late abort while the terminal
   * `done` is being applied must not turn a good run red.
   */
  markFailed: () => void
  /**
   * The stream dropped but the job may still be running: hold the DAG as it
   * is and say so, while `@/thread/resume` tries to reattach. A run that has
   * already finished is left alone, for the same reason as `markFailed`.
   */
  markReconnecting: () => void
  reset: () => void
}

const LIVE: ReadonlySet<JobPhase> = new Set(['streaming', 'reconnecting'])

/** True while a run is in progress as far as this client knows. */
export function isLive(phase: JobPhase): boolean {
  return LIVE.has(phase)
}

export const useJobStore = create<JobStore>((set) => ({
  ...initialJobState,
  apply: (event) => set((state) => reduce(state, event)),
  markFailed: () => set((state) => (LIVE.has(state.phase) ? { ...state, phase: 'failed' } : state)),
  markReconnecting: () =>
    set((state) => (state.phase === 'streaming' ? { ...state, phase: 'reconnecting' } : state)),
  reset: () => set({ ...initialJobState }),
}))
