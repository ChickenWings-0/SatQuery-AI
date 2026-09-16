/**
 * Friendly aliases over the generated contract types.
 *
 * `schema.d.ts` is generated from the committed `openapi.json` by
 * `npm run gen:api` and is never edited by hand: schema 1.0 is frozen, so the
 * client's types should be derived from it rather than restated. This module is
 * the only place allowed to reach into `components['schemas']`, so a rename in
 * a future 2.0 breaks in one file instead of forty.
 */
import type { components } from '@/api/schema'

type S = components['schemas']

export type AnalyzeResponse = S['AnalyzeResponse']
export type Answer = S['Answer']
export type ApiError = S['ApiError']
export type ArtifactRef = S['ArtifactRef']
export type AuditTrace = S['AuditTrace']
export type CheckResult = S['CheckResult']
export type CheckStatus = S['CheckStatus']
export type Citation = S['Citation']
export type CompatibilityReport = S['CompatibilityReport']
export type Confidence = S['Confidence']
export type Execution = S['Execution']
export type HealthResponse = S['HealthResponse']
export type ImageryFetchRequest = S['ImageryFetchRequest']
export type ImageryFetchResponse = S['ImageryFetchResponse']
export type ImageryFile = S['ImageryFile']
export type InputManifest = S['InputManifest']
export type JobAccepted = S['JobAccepted']
export type JobStatusResponse = S['JobStatusResponse']
export type Plan = S['Plan']
export type PlanStep = S['PlanStep']
export type RegistryResponse = S['RegistryResponse']
export type ResolvedTask = S['ResolvedTask']
export type TaskType = S['TaskType']
export type ToolSpec = S['ToolSpec']
export type ToolStatus = S['ToolStatus']
export type ValidateResponse = S['ValidateResponse']
export type WarningItem = S['WarningItem']

/**
 * The stages a job passes through, in order (API_CONTRACT §5 `stage` event).
 */
export const STAGES = [
  'queued',
  'ingesting',
  'validating',
  'rendering',
  'planning',
  'executing',
  'aggregating',
  'done',
] as const
export type Stage = (typeof STAGES)[number]

/**
 * The client-side widening of {@link ToolStatus}.
 *
 * The contract's enum has only the four *outcomes* — a step that has finished.
 * It has no name for a step that has not started or is running, because the
 * server never needs one: it reports a step once, when it is over. The UI needs
 * both, so `PENDING` (set for every step the moment the `plan` event lands) and
 * `RUNNING` (set on `step_started`) live here and only here. They are never sent
 * by the server and must never be written back to it.
 */
export type NodeState = 'PENDING' | 'RUNNING' | ToolStatus

export const TOOL_STATUSES = ['OK', 'DEGRADED', 'FAILED', 'SKIPPED'] as const

/** True when `value` is one of the four statuses the contract can send. */
export function isToolStatus(value: unknown): value is ToolStatus {
  return typeof value === 'string' && (TOOL_STATUSES as readonly string[]).includes(value)
}

/** A step is finished once its state is one of the contract's four outcomes. */
export function isTerminalState(state: NodeState): state is ToolStatus {
  return isToolStatus(state)
}
