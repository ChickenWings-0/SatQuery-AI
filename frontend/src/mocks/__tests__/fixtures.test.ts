/**
 * The recorded fixtures must stay a valid stream.
 *
 * These are real responses captured from a live backend, so this suite is the
 * thing that notices when a recording goes stale: re-record after a backend
 * change and any drift from the §5 ordering guarantee, or from what the reducer
 * expects, fails here rather than in the demo.
 */
import { describe, expect, it } from 'vitest'

import { parseJobEvent } from '@/api/events'
import type { JobEvent } from '@/api/events'
import {
  MOCK_TRACE_ID,
  eventsFixture,
  healthFixture,
  registryFixture,
  validateFixture,
} from '@/mocks/fixtures'
import { completedCount, degradedCount, reduceAll } from '@/state/job'

/** Parse the recording exactly as the SSE client would parse the wire. */
const events: JobEvent[] = eventsFixture.map((recorded) =>
  parseJobEvent(recorded.event, JSON.stringify(recorded.data)),
)

describe('recorded bi-temporal run', () => {
  it('parses every frame as a known contract event', () => {
    expect(events.length).toBeGreaterThan(10)
  })

  it('obeys the §5 ordering guarantee', () => {
    const names = events.map((event) => event.type)
    expect(names[0]).toBe('queued')
    expect(names.at(-1)).toBe('done')
    expect(names.filter((n) => n === 'done' || n === 'error')).toHaveLength(1)

    const planAt = names.indexOf('plan')
    expect(planAt).toBeGreaterThan(0)
    expect(new Set(names.slice(1, planAt))).toEqual(new Set(['stage']))
    for (const streaming of ['step_started', 'step_completed', 'artifact'] as const) {
      expect(names.indexOf(streaming)).toBeGreaterThan(planAt)
    }
  })

  it('drives the reducer to a complete terminal state', () => {
    const state = reduceAll(events)

    expect(state.phase).toBe('succeeded')
    expect(state.jobId).toBe(MOCK_TRACE_ID)
    expect(state.pct).toBe(100)
    expect(state.result?.trace_id).toBe(MOCK_TRACE_ID)
    expect(state.nodes.length).toBeGreaterThan(0)
    expect(completedCount(state)).toBe(state.nodes.length)
    expect(state.artifacts.length).toBeGreaterThan(0)
    expect(state.streamedAnswer).toBe(state.result?.answer.text)
  })

  it('leaves no node stuck PENDING or RUNNING', () => {
    // A node the stream never resolved reads as a hang in the pipeline pulse.
    const state = reduceAll(events)
    expect(state.nodes.filter((n) => n.state === 'PENDING' || n.state === 'RUNNING')).toEqual([])
  })

  it('serves artifact urls that resolve to committed mock renders', () => {
    const state = reduceAll(events)
    for (const artifact of state.artifacts) {
      if (!artifact.url) continue
      expect(artifact.url).toContain(`/v1/artifacts/${MOCK_TRACE_ID}/`)
    }
  })

  it('exercises the degraded path the demo needs to survive', () => {
    // Not an assertion that this run *is* degraded — only that whatever the
    // recording contains, the counts agree with the node states.
    const state = reduceAll(events)
    const flagged = state.nodes.filter(
      (n) => n.state === 'DEGRADED' || n.state === 'FAILED' || n.state === 'SKIPPED',
    )
    expect(degradedCount(state)).toBe(flagged.length)
  })
})

describe('static fixtures', () => {
  it('health reports a device and a tool count', () => {
    expect(healthFixture.schema_version).toBe('1.0')
    expect(healthFixture.tools_total).toBeGreaterThan(0)
    expect(healthFixture.tools_available).toBeLessThanOrEqual(healthFixture.tools_total)
  })

  it('registry exposes unavailable tools with a reason, per §4.8', () => {
    expect(registryFixture.tools.length).toBeGreaterThan(0)
    for (const tool of registryFixture.tools) {
      if (!tool.available) expect(tool.unavailable_reason).toBeTruthy()
    }
  })

  it('validate returns manifests, a full check battery and supported tasks', () => {
    expect(validateFixture.inputs.map((m) => m.id)).toEqual(['img_0', 'img_1'])
    expect(validateFixture.compatibility.pair_type).toBe('BI_TEMPORAL')
    expect(validateFixture.compatibility.checks?.length).toBeGreaterThan(0)
    expect(validateFixture.supported_tasks).toContain('CHANGE_VQA')
  })
})

describe('ungrounded scenario', () => {
  it('renders both ungrounded numbers as uncited spans', async () => {
    const { eventsForScenario } = await import('@/mocks/scenarios')
    const { annotate } = await import('@/thread/annotate')

    const swapped = eventsForScenario(eventsFixture, 'ungrounded')
    const done = swapped.find((event) => event.event === 'done')!.data as {
      answer: { text: string; citations: never[]; uncited_numeric_spans: string[] }
    }

    expect(done.answer.uncited_numeric_spans).toEqual(['12 hectares', '8.5%'])

    const segments = annotate(
      done.answer.text,
      done.answer.citations,
      done.answer.uncited_numeric_spans,
    )
    expect(segments.filter((s) => s.kind === 'uncited').map((s) => s.text)).toEqual([
      '12 hectares',
      '8.5%',
    ])
    // Citations and uncited spans coexist without overlapping.
    expect(segments.filter((s) => s.kind === 'citation')).toHaveLength(4)
    expect(segments.map((s) => s.text).join('')).toBe(done.answer.text)
  })

  it('leaves the canonical recording untouched', () => {
    const done = eventsFixture.find((event) => event.event === 'done')!.data as {
      answer: { uncited_numeric_spans: string[]; template_fallback: boolean }
    }
    expect(done.answer.uncited_numeric_spans).toEqual([])
    expect(done.answer.template_fallback).toBe(true)
  })
})
