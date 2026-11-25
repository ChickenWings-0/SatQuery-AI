/** A finished run replays into the same events a live run streams. */
import { describe, expect, it } from 'vitest'

import type { AnalyzeResponse } from '@/api/types'
import { useJobStore } from '@/state/job'
import { replayEvents } from '@/thread/restore'

const result = {
  trace_id: 't1',
  answer: { text: 'Open water covers 21.81% of the scene.', citations: [], uncited_numeric_spans: [], generator: 'hf', template_fallback: false },
  artifacts: [],
  confidence: { overall: 0.87 },
  compatibility: { pair_type: 'SINGLE' },
  resolved_task: { primary: 'SEGMENTATION' },
  trace: {
    plan: { steps: [{ step: 1, tool: 'spectral_renderer' }, { step: 2, tool: 'class_outliner', depends_on: [1] }] },
    executions: [
      { step: 1, tool: 'spectral_renderer', status: 'OK', duration_ms: 60, confidence: 0.99, output_refs: ['art_0'] },
      { step: 2, tool: 'class_outliner', status: 'OK', duration_ms: 80, confidence: 0.85, output_refs: ['art_1'] },
    ],
    artifacts: [
      { id: 'art_0', type: 'RENDERED_VIEW', label: 'Image 1 (optical true colour)', url: '/v1/artifacts/t1/art_0.jpg' },
      { id: 'art_1', type: 'OVERLAY_PNG', label: 'Water bodies outlined over true colour (img_0)', url: '/v1/artifacts/t1/art_1.png' },
    ],
  },
} as unknown as AnalyzeResponse

describe('replayEvents', () => {
  it('rebuilds queued, plan, both steps, the artifacts and done', () => {
    const types = replayEvents(result).map((event) => event.type)
    expect(types).toEqual(['queued', 'plan', 'step_started', 'step_completed', 'step_started', 'step_completed', 'artifact', 'artifact', 'done'])
  })

  it('restores a finished workspace through the one reducer', () => {
    const job = useJobStore.getState()
    job.reset()
    for (const event of replayEvents(result)) job.apply(event)
    const state = useJobStore.getState()
    expect(state.phase).toBe('succeeded')
    expect(state.artifacts.map((a) => a.id)).toEqual(['art_0', 'art_1'])
    expect(state.nodes.map((n) => n.state)).toEqual(['OK', 'OK'])
    expect(state.result?.answer.text).toContain('21.81%')
  })
})
