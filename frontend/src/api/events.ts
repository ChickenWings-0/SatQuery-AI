/**
 * The SSE event union of API_CONTRACT §5, and the parser that produces it.
 *
 * The contract guarantees an ordering:
 *
 *     queued -> stage(ingesting…planning) -> plan
 *            -> interleaved step_started / step_completed / artifact
 *            -> answer_delta* -> done | error
 *
 * `done` and `error` are terminal and mutually exclusive. That guarantee is
 * what lets the reducer in `@/state/job` be a plain state machine rather than
 * defensive parsing — but it is a guarantee about *our* server, so
 * {@link parseJobEvent} still validates the envelope and refuses anything it
 * does not recognise. §2 is explicit that an unknown enum value is a contract
 * violation to be surfaced, never silently coerced.
 *
 * Note the absence of a streaming answer in practice: `answer_delta` is listed
 * as `answer_delta*` — zero or more — and the backend does not currently emit
 * any, because the aggregator produces the finished answer in one piece. The
 * union carries it so the reducer is complete, and the UI reads the authoritative
 * text from the terminal `done` payload regardless.
 */
import type {
  AnalyzeResponse,
  ApiError,
  ArtifactRef,
  PlanStep,
  Stage,
  ToolStatus,
} from '@/api/types'

export interface QueuedData {
  job_id: string
}

export interface StageData {
  stage: Stage
  pct: number
}

export interface PlanData {
  steps: PlanStep[]
}

export interface StepStartedData {
  step: number
  tool: string
  est_ms: number
}

export interface StepCompletedData {
  step: number
  status: ToolStatus
  duration_ms: number
  confidence: number
  output_refs: string[]
}

export interface AnswerDeltaData {
  text: string
}

export type JobEvent =
  | { type: 'queued'; data: QueuedData }
  | { type: 'stage'; data: StageData }
  | { type: 'plan'; data: PlanData }
  | { type: 'step_started'; data: StepStartedData }
  | { type: 'step_completed'; data: StepCompletedData }
  | { type: 'artifact'; data: ArtifactRef }
  | { type: 'answer_delta'; data: AnswerDeltaData }
  | { type: 'done'; data: AnalyzeResponse }
  | { type: 'error'; data: ApiError }

export type JobEventType = JobEvent['type']

export const JOB_EVENT_TYPES = [
  'queued',
  'stage',
  'plan',
  'step_started',
  'step_completed',
  'artifact',
  'answer_delta',
  'done',
  'error',
] as const

export const TERMINAL_EVENTS: ReadonlySet<JobEventType> = new Set(['done', 'error'])

/** Raised when the stream carries something the frozen contract does not define. */
export class ContractViolation extends Error {
  readonly event: string
  readonly payload: unknown

  constructor(message: string, event: string, payload: unknown) {
    super(message)
    this.name = 'ContractViolation'
    this.event = event
    this.payload = payload
  }
}

function isJobEventType(value: string): value is JobEventType {
  return (JOB_EVENT_TYPES as readonly string[]).includes(value)
}

/**
 * Turn one raw SSE frame into a typed {@link JobEvent}.
 *
 * @throws ContractViolation if the event name is unknown or the data is not
 *   JSON. Both are contract violations rather than transient noise, so they are
 *   surfaced to the user instead of being dropped on the floor.
 */
export function parseJobEvent(event: string, raw: string): JobEvent {
  if (!isJobEventType(event)) {
    throw new ContractViolation(`Unknown SSE event "${event}".`, event, raw)
  }
  let data: unknown
  try {
    data = JSON.parse(raw)
  } catch {
    throw new ContractViolation(`Malformed JSON on "${event}" event.`, event, raw)
  }
  // The event name determines the payload type; the server is the schema
  // authority and the contract is frozen, so this is the one narrowing the
  // client takes on trust rather than re-validating field by field.
  return { type: event, data } as JobEvent
}

export function isTerminal(event: JobEvent): boolean {
  return TERMINAL_EVENTS.has(event.type)
}
