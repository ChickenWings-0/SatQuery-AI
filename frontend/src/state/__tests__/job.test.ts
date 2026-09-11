/**
 * The reducer's contract: drive it with a recorded event sequence and the
 * terminal state must equal the `done` payload.
 *
 * These are the assertions that let phases F4-F5 be built against a mock: if
 * the reducer is right, every panel downstream is reading correct state.
 */
import { describe, expect, it } from 'vitest'

import { ContractViolation, parseJobEvent, type JobEvent } from '@/api/events'
import {
  completedCount,
  degradedCount,
  initialJobState,
  nodeForCitationSource,
  reduce,
  reduceAll,
} from '@/state/job'
import type { AnalyzeResponse, ArtifactRef, PlanStep } from '@/api/types'

const JOB_ID = 'b3f1000000000000000000000000000a'

const STEPS: PlanStep[] = [
  {
    step: 1,
    tool: 'spectral_renderer',
    depends_on: [],
    input_refs: ['img_0', 'img_1'],
    reason: 'policy_table:CHANGE_VQA|BI_TEMPORAL|optical',
  },
  {
    step: 2,
    tool: 'siamese_change_detector',
    depends_on: [1],
    input_refs: ['img_0', 'img_1'],
    reason: 'policy_table:CHANGE_VQA|BI_TEMPORAL|optical',
  },
  {
    step: 3,
    tool: 'change_statistics',
    depends_on: [2],
    input_refs: ['@2'],
    reason: 'policy_table:CHANGE_VQA|BI_TEMPORAL|optical',
  },
]

function artifact(id: string, step: number): ArtifactRef {
  return {
    id,
    type: 'RENDERED_VIEW',
    mime: 'image/png',
    label: `Image 1 (optical true colour) ${id}`,
    url: `/v1/artifacts/${JOB_ID}/${id}.png`,
    geotiff_url: null,
    geojson_url: null,
    geo: null,
    width: 256,
    height: 256,
    stats: null,
    inline: null,
    produced_by_step: step,
  }
}

const DONE_PAYLOAD = {
  trace_id: JOB_ID,
  answer: {
    text: 'Approximately 7.4% of the scene transitioned to built-up.',
    citations: [
      {
        claim: '7.4% of the scene',
        source: 'step:3/scalars.changed_area_pct',
        value: 7.4,
      },
    ],
    uncited_numeric_spans: ['3 km2'],
    generator: 'vlm_change_vqa@1.0.0',
    template_fallback: false,
  },
  artifacts: [artifact('art_0', 1), artifact('art_1', 2)],
  confidence: {
    overall: 0.86,
    method: 'weighted_tool_agreement_v1',
    components: {},
    caps_applied: [],
  },
  compatibility: {
    pair_type: 'BI_TEMPORAL',
    pair_type_source: 'heuristic',
    checks: [],
    overall: 'PASS',
    actions_taken: [],
    common_grid: null,
  },
  resolved_task: {
    primary: 'CHANGE_VQA',
    secondary: [],
    slots: {},
    confidence: 0.94,
    classifier: 'rules_v1',
  },
  trace: null,
} as unknown as AnalyzeResponse

/** The §5 order, as the server actually emits it. */
const SEQUENCE: JobEvent[] = [
  { type: 'queued', data: { job_id: JOB_ID } },
  { type: 'stage', data: { stage: 'ingesting', pct: 5 } },
  { type: 'stage', data: { stage: 'planning', pct: 30 } },
  { type: 'plan', data: { steps: STEPS } },
  { type: 'stage', data: { stage: 'executing', pct: 45 } },
  { type: 'step_started', data: { step: 1, tool: 'spectral_renderer', est_ms: 420 } },
  { type: 'artifact', data: artifact('art_0', 1) },
  {
    type: 'step_completed',
    data: { step: 1, status: 'OK', duration_ms: 431, confidence: 1.0, output_refs: ['art_0'] },
  },
  { type: 'step_started', data: { step: 2, tool: 'siamese_change_detector', est_ms: 1800 } },
  { type: 'artifact', data: artifact('art_1', 2) },
  {
    type: 'step_completed',
    data: { step: 2, status: 'DEGRADED', duration_ms: 1830, confidence: 0.62, output_refs: ['art_1'] },
  },
  { type: 'step_started', data: { step: 3, tool: 'change_statistics', est_ms: 90 } },
  {
    type: 'step_completed',
    data: { step: 3, status: 'OK', duration_ms: 88, confidence: 0.91, output_refs: [] },
  },
  { type: 'stage', data: { stage: 'aggregating', pct: 85 } },
  { type: 'done', data: DONE_PAYLOAD },
]

describe('reduce', () => {
  it('ends in a state that equals the done payload', () => {
    const state = reduceAll(SEQUENCE)

    expect(state.phase).toBe('succeeded')
    expect(state.jobId).toBe(JOB_ID)
    expect(state.stage).toBe('done')
    expect(state.pct).toBe(100)
    expect(state.result).toEqual(DONE_PAYLOAD)
    expect(state.error).toBeNull()
  })

  it('draws every node PENDING the moment the plan lands, before anything runs', () => {
    const uptoPlan = SEQUENCE.slice(0, SEQUENCE.findIndex((e) => e.type === 'plan') + 1)
    const state = reduceAll(uptoPlan)

    expect(state.nodes).toHaveLength(3)
    expect(state.nodes.map((n) => n.state)).toEqual(['PENDING', 'PENDING', 'PENDING'])
    expect(state.nodes[1]?.dependsOn).toEqual([1])
    expect(state.nodes[0]?.reason).toContain('policy_table:CHANGE_VQA|BI_TEMPORAL|optical')
    expect(completedCount(state)).toBe(0)
  })

  it('widens a running step to RUNNING and back to a contract status', () => {
    const started = SEQUENCE.findIndex(
      (e) => e.type === 'step_started' && e.data.step === 2,
    )
    const running = reduceAll(SEQUENCE.slice(0, started + 1))
    expect(running.nodes[1]?.state).toBe('RUNNING')
    expect(running.nodes[1]?.estMs).toBe(1800)
    expect(running.nodes[1]?.durationMs).toBeNull()

    const finished = reduceAll(SEQUENCE)
    expect(finished.nodes[1]?.state).toBe('DEGRADED')
    expect(finished.nodes[1]?.durationMs).toBe(1830)
    expect(finished.nodes[1]?.confidence).toBe(0.62)
  })

  it('counts a degraded step so the pipeline button can badge it', () => {
    const state = reduceAll(SEQUENCE)
    expect(completedCount(state)).toBe(3)
    expect(degradedCount(state)).toBe(1)
  })

  it('never lets progress run backwards', () => {
    const state = reduceAll([
      ...SEQUENCE.slice(0, 5),
      { type: 'stage', data: { stage: 'ingesting', pct: 5 } },
    ])
    expect(state.pct).toBe(45)
  })

  it('streams artifacts in order and does not duplicate a replayed one', () => {
    const state = reduceAll([...SEQUENCE, { type: 'artifact', data: artifact('art_0', 1) }])
    expect(state.artifacts.map((a) => a.id)).toEqual(['art_0', 'art_1'])
  })

  it('takes the answer text from the terminal payload, not the deltas', () => {
    const state = reduceAll([
      ...SEQUENCE.slice(0, -1),
      { type: 'answer_delta', data: { text: 'Approximately 7.4%' } },
      { type: 'answer_delta', data: { text: ' of the sce' } },
      SEQUENCE[SEQUENCE.length - 1]!,
    ])
    // The partially streamed string would have produced wrong citation offsets.
    expect(state.streamedAnswer).toBe(DONE_PAYLOAD.answer.text)
  })

  it('keeps streamed artifacts when include_rendered_views omitted them', () => {
    const withoutArtifacts = { ...DONE_PAYLOAD, artifacts: [] }
    const state = reduceAll([
      ...SEQUENCE.slice(0, -1),
      { type: 'done', data: withoutArtifacts as AnalyzeResponse },
    ])
    expect(state.artifacts.map((a) => a.id)).toEqual(['art_0', 'art_1'])
  })

  it('lands in failed on a terminal error, keeping the partial DAG', () => {
    const state = reduceAll([
      ...SEQUENCE.slice(0, 8),
      {
        type: 'error',
        data: {
          code: 'INSUFFICIENT_OVERLAP',
          http_status: 422,
          message: 'The two images overlap by only 12%.',
          hint: 'Upload images covering the same footprint.',
          ref: 'img_1',
          trace_id: JOB_ID,
        },
      },
    ])
    expect(state.phase).toBe('failed')
    expect(state.error?.code).toBe('INSUFFICIENT_OVERLAP')
    expect(state.nodes).toHaveLength(3)
    expect(state.result).toBeNull()
  })

  it('is pure — the input state is never mutated', () => {
    const before = structuredClone(initialJobState)
    reduce(initialJobState, { type: 'queued', data: { job_id: JOB_ID } })
    expect(initialJobState).toEqual(before)
  })

  it('ignores a step event for a step the plan never declared', () => {
    const state = reduce(reduceAll(SEQUENCE.slice(0, 4)), {
      type: 'step_started',
      data: { step: 99, tool: 'ghost', est_ms: 1 },
    })
    expect(state.nodes.map((n) => n.state)).toEqual(['PENDING', 'PENDING', 'PENDING'])
  })
})

describe('nodeForCitationSource', () => {
  it('resolves step:{n}/scalars.{path} back to its producing node', () => {
    const state = reduceAll(SEQUENCE)
    expect(nodeForCitationSource(state, 'step:3/scalars.changed_area_pct')?.tool).toBe(
      'change_statistics',
    )
  })

  it('handles the multi-scalar form joined with a pipe', () => {
    const state = reduceAll(SEQUENCE)
    const node = nodeForCitationSource(state, 'step:2/scalars.ndbi_mean_pre|ndbi_mean_post')
    expect(node?.tool).toBe('siamese_change_detector')
  })

  it('returns undefined rather than guessing at an unparseable source', () => {
    const state = reduceAll(SEQUENCE)
    expect(nodeForCitationSource(state, 'not-a-source')).toBeUndefined()
  })
})

describe('parseJobEvent', () => {
  it('parses a well-formed frame into the union', () => {
    const event = parseJobEvent('stage', '{"stage":"executing","pct":45}')
    expect(event.type).toBe('stage')
  })

  it('rejects an unknown event name as a contract violation', () => {
    // §2: a client receiving an unknown value must surface it, not coerce it.
    expect(() => parseJobEvent('teleport', '{}')).toThrow(ContractViolation)
  })

  it('rejects malformed JSON', () => {
    expect(() => parseJobEvent('stage', '{oops')).toThrow(ContractViolation)
  })
})
